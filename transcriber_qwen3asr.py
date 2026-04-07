"""Qwen3-ASR 1.7B transcriber module.

Provides Qwen3ASRTranscriber class for real-time and batch transcription
using Qwen/Qwen3-ASR-1.7B with bfloat16 (~3.4 GB VRAM).

Context boost: Qwen3-ASR accepts domain-specific terms/phrases as context
for improved recognition. This integrates with HotwordLoader to pass
hotwords as context strings.
"""

import gc
import logging
import re
import time

import numpy as np

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# Qwen3-ASR uses full language names, not ISO codes
_LANG_MAP = {
    "ja": "Japanese", "en": "English", "zh": "Chinese",
    "ko": "Korean", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    "ar": "Arabic", "it": "Italian", "th": "Thai",
    "vi": "Vietnamese", "id": "Indonesian", "tr": "Turkish",
    "hi": "Hindi", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "el": "Greek", "ro": "Romanian",
    "hu": "Hungarian", "ms": "Malay", "fa": "Persian",
}

# Engine-agnostic filters (shared with Whisper pipeline)
# Hallucination patterns: Whisper-specific ones included initially for safety,
# will be tuned after observing Qwen3-ASR's actual failure modes.
_HALLUCINATION_PATTERNS = [
    re.compile(r"ご視聴ありがとうございま(した|す)"),
    re.compile(r"チャンネル登録.*?お願いします"),
    re.compile(r"チャンネル登録"),
    re.compile(r"高評価"),
    re.compile(r"ご清聴ありがとうございま(した|す)"),
    re.compile(r"最後まで(ご覧|見て)いただき.*"),
    re.compile(r"ご覧いただきありがとうございま(した|す)"),
    re.compile(r"字幕.*?(視聴者|翻訳|提供)"),
    re.compile(r"Thanks? for watching\.?", re.IGNORECASE),
    re.compile(r"See you next time\.?", re.IGNORECASE),
    re.compile(r"Subscribe", re.IGNORECASE),
    re.compile(r"Bye[\.\s]*bye\.?", re.IGNORECASE),
    re.compile(r"♪+"),
    re.compile(r"^\s*\.+\s*$"),
    re.compile(r"(.{2,}?)\1{3,}"),  # 4+ repeats of same phrase
    re.compile(r"^[。、.,!?・…―\s]+$"),  # Punctuation-only
    re.compile(r"\ufffd"),  # Unicode replacement character
]

_HALLUCINATION_FULL_MATCH = [
    re.compile(r"^お(つかれさま|疲れ様)でした[。.!]*$"),
    re.compile(r"^(ではまた|また次回)[。.!]*$"),
    re.compile(r"^おやすみなさい[。.!]*$"),
]

# Trailing comma fragments
_TRAILING_FRAGMENT_RE = re.compile(
    r"([,，、]\s*[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fffa-zA-Z]{1,6}){2,}\s*$"
)

# Sentence-boundary regex for segment splitting
_SEGMENT_SPLIT_RE = re.compile(
    r'(?<=[。.!?？！])|'
    r'(?<=けど\s)|'
    r'(?<=から\s)|'
    r'(?<=のに\s)|'
    r'(?<=ので\s)|'
    r'(?<=けれど\s)'
)

# Punctuation normalization
_PERIOD_RE = re.compile(r'(?<=[^\d\s])\.(?=[^\d\s]|$)')
_DASH_RE = re.compile(r'\s*-\s+')
_TRAILING_COMMA_RE = re.compile(r'[,，、]+\s*$')
_CONSECUTIVE_PUNCT_RE = re.compile(r'([。、!?！？])\1+')

# Repetitive kana detection
_PARTICLE_PATTERNS = re.compile(
    r'[ぁ-ん\u30a0-\u30ff](は|が|を|へ|から|まで|より|ばかり|さえ|すら|だけ|ほど|の|で|と|に|も|や)[ぁ-ん\u30a0-\u30ff]'
)
_GRAMMATICAL_ENDINGS = re.compile(
    r'(です|ます|ました|ません|でした|ている|ていた|ていない|ないで|なかった'
    r'|ことが|ものが|ように|ようだ|だろう|でしょう|ですか|ますか|ますよ|ですね'
    r'|ですよ|ましょう|ください|ながら|けれど|ったら|ったり|のです|んです'
    r'|よかった|じゃない|なさい|だよ|だね|だな|てる|てた|ちゃう|ちゃった'
    r'|って|かな|かも|でしょ|っけ|じゃん|んだ|のに|けど)'
)

# Qwen3-ASR specific: word-level repetition (e.g., "カボ。カボ。カボ。")
# Whisper repeats at kana level; Qwen3-ASR repeats at word level.
# Matches: word + separator, repeated 3+ times total (with optional trailing separator)
_WORD_REPEAT_RE = re.compile(
    r'([\u3040-\u9fff]{2,6})[。、.,\s]+'
    r'(?:\1[。、.,\s]*){2,}'
)
# Interjection/aizuchi repeats without separators: はいはいはいはい → はい
_AIZUCHI_REPEAT_RE = re.compile(
    r'(はい|うん|ああ|えー|おー|そう|ねー|なー|へー){3,}'
)

# Sentence splitting for long Qwen3-ASR output
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[。！？!?])\s*|'    # After sentence-ending punctuation
    r'(?<=\n)\s*'              # After newline
)

SEGMENT_MAX_DURATION_DEFAULT = 15.0


class Qwen3ASRTranscriber:
    """Qwen3-ASR 1.7B transcriber with model lifecycle management."""

    def __init__(self):
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load_model(self):
        """Load Qwen3-ASR model with bfloat16 (~3.4 GB VRAM).

        Must be called from a thread (run_in_executor) as model loading is blocking.
        """
        if self.loaded:
            return

        import torch
        from qwen_asr import Qwen3ASRModel

        log.info("Loading Qwen3-ASR model: %s", MODEL_ID)
        t0 = time.time()

        kwargs = dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # max_new_tokens set per-call based on audio duration
            # (default 1024 covers most real-time chunks)
            max_new_tokens=1024,
        )

        # Try flash_attention_2, fall back to sdpa, then eager
        for attn_impl in ("flash_attention_2", "sdpa", None):
            try:
                if attn_impl:
                    kwargs["attn_implementation"] = attn_impl
                self._model = Qwen3ASRModel.from_pretrained(MODEL_ID, **kwargs)
                break
            except (ValueError, ImportError) as e:
                if attn_impl:
                    log.info("attn_implementation=%s not available: %s", attn_impl, e)
                    kwargs.pop("attn_implementation", None)
                    continue
                raise

        # Enable TF32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        elapsed = time.time() - t0
        log.info("Qwen3-ASR model loaded in %.1fs", elapsed)

    def unload_model(self):
        """Unload model and free VRAM."""
        import torch

        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("Qwen3-ASR model unloaded")

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        session_offset: float = 0.0,
        hotwords: str = "",
        language: str | None = "ja",
        term_corrector=None,
        segment_max_duration: float = SEGMENT_MAX_DURATION_DEFAULT,
        audio_preprocessor=None,
    ) -> list[dict]:
        """Transcribe PCM 16-bit LE audio bytes.

        Args:
            audio_bytes: PCM 16-bit LE audio data.
            sample_rate: Audio sample rate (default 16000).
            session_offset: Timestamp offset in seconds.
            hotwords: Comma-separated hotwords string (converted to context).
            language: Language code ("ja", "en", or None for auto-detect).
            term_corrector: Optional TermCorrector instance for Layer 2.
            segment_max_duration: Max segment duration before splitting.
            audio_preprocessor: Optional AudioPreprocessor for loudness normalization.

        Returns:
            List of segment dicts with keys:
            text, speaker, speaker_id, start, end, is_user, person_id
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < 1600:  # < 0.1s
            return []

        # Audio preprocessing: loudness normalization only.
        # Research (arXiv:2512.17562) shows spectral denoising HURTS
        # modern transformer ASR models — they have built-in noise robustness.
        # Only apply volume normalization, skip noise reduction.
        if audio_preprocessor is not None:
            audio_np = audio_preprocessor.normalize_only(audio_np, sample_rate)

        return self.transcribe_array(
            audio_np, sample_rate, session_offset, hotwords,
            language, term_corrector, segment_max_duration,
        )

    def transcribe_array(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        session_offset: float = 0.0,
        hotwords: str = "",
        language: str | None = "ja",
        term_corrector=None,
        segment_max_duration: float = SEGMENT_MAX_DURATION_DEFAULT,
        phonetic_corrector=None,
    ) -> list[dict]:
        """Transcribe pre-loaded audio array.

        Args:
            audio_np: Float32 audio array.
            sample_rate: Audio sample rate.
            session_offset: Timestamp offset in seconds.
            hotwords: Comma-separated hotwords string.
            language: Language code or None for auto-detect.
            term_corrector: Optional TermCorrector for Layer 2.
            segment_max_duration: Max segment duration before splitting.

        Returns:
            List of segment dicts.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        chunk_duration = len(audio_np) / sample_rate

        # Build context from hotwords for Qwen3-ASR's context boost.
        # IMPORTANT: Passing too many terms causes the model to output
        # the context words as transcription instead of actual speech.
        # Limit to max 15 short, high-value terms (custom words only).
        context = []
        if hotwords:
            terms = [w.strip() for w in hotwords.split(",") if w.strip()]
            # Filter: only short terms (2-8 chars) to reduce context noise
            terms = [t for t in terms if 2 <= len(t) <= 8]
            terms = terms[:15]  # Hard cap to prevent context leak
            if terms:
                context = [" ".join(terms)]

        # Build transcribe kwargs — greedy decoding is official recommendation
        # (no temperature/beam_size — the model defaults to greedy)
        transcribe_kwargs = {}
        if language:
            # Map ISO code to full name (Qwen3-ASR expects "Japanese" not "ja")
            lang_name = _LANG_MAP.get(language.lower(), language)
            transcribe_kwargs["language"] = lang_name
        if context:
            transcribe_kwargs["context"] = context

        try:
            results = self._model.transcribe(
                audio=(audio_np, sample_rate),
                **transcribe_kwargs,
            )
        except Exception as e:
            log.error("Qwen3-ASR transcription failed: %s", e, exc_info=True)
            return []

        # Collect context terms for leak detection
        context_terms = set()
        if context:
            context_terms = {t for t in context[0].split() if len(t) >= 2}

        # Parse results
        segments_out = []
        for result in results:
            text = result.text.strip() if hasattr(result, 'text') else str(result).strip()

            if not text:
                continue

            # Context leak detection: if >50% of output chars match context terms,
            # the model is echoing context instead of transcribing speech.
            if context_terms and _is_context_leak(text, context_terms):
                log.info("Context leak detected, dropping: '%s'", text[:80])
                continue

            detected_lang = None
            if hasattr(result, 'language'):
                detected_lang = result.language

            # Qwen3-ASR returns one big text block for the entire audio.
            # Split into sentences FIRST, then apply filters per-sentence.
            # This prevents _has_repetitive_kana from discarding the entire
            # text when only a small portion contains repetitions (e.g. "はいはいはいはい").
            sentences = _split_into_sentences(text)
            filtered_sentences = []
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                filtered = _filter_text(sent, term_corrector, phonetic_corrector)
                if filtered:
                    filtered_sentences.append(filtered)

            if not filtered_sentences:
                continue

            # Distribute timestamps proportionally across sentences
            chars_per_sentence = [len(s) for s in filtered_sentences]
            total_chars = sum(chars_per_sentence)
            if total_chars == 0:
                continue

            cur_start = session_offset
            for i, sent_text in enumerate(filtered_sentences):
                ratio = chars_per_sentence[i] / total_chars
                sent_dur = chunk_duration * ratio
                sent_end = cur_start + sent_dur
                if i == len(filtered_sentences) - 1:
                    sent_end = session_offset + chunk_duration

                seg_dict = {
                    "text": sent_text,
                    "speaker": "SPEAKER_00",
                    "speaker_id": 0,
                    "start": round(cur_start, 2),
                    "end": round(sent_end, 2),
                    "is_user": True,
                    "person_id": None,
                }
                if detected_lang:
                    seg_dict["language"] = detected_lang
                segments_out.append(seg_dict)
                cur_start = sent_end

        return segments_out


# ---- Context leak detection ----

def _split_into_sentences(text: str) -> list[str]:
    """Split long text into sentences for per-sentence filtering.

    Qwen3-ASR returns one big text block. Splitting before filtering
    prevents partial repetitions (e.g. "はいはいはい" in one part)
    from discarding the entire text.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    # If no sentence-ending punctuation, split by reasonable length
    if len(parts) <= 1 and len(text) > 100:
        # Fall back to splitting at commas/midpoints
        parts = re.split(r'(?<=[、,])\s*', text)
    return [p for p in parts if p.strip()]


def _is_context_leak(text: str, context_terms: set) -> bool:
    """Detect if output is mostly context terms echoed back.

    When Qwen3-ASR receives context but audio has no/little speech,
    it sometimes outputs the context terms as transcription.
    Returns True if >50% of the text (by char count) matches context terms.
    """
    if not context_terms or not text:
        return False
    matched_chars = 0
    for term in context_terms:
        count = text.count(term)
        matched_chars += count * len(term)
    text_chars = len(re.sub(r'\s+', '', text))
    if text_chars == 0:
        return False
    ratio = matched_chars / text_chars
    return ratio > 0.5


# ---- Text filtering (engine-agnostic Layer 1 + Layer 2) ----

def _filter_text(text: str, term_corrector=None, phonetic_corrector=None) -> str:
    """Apply engine-agnostic text filters.

    Filters applied:
    - Empty/whitespace check
    - Unicode replacement char
    - Single-char drop
    - Repetitive kana detection
    - Word-level repeat collapse (Qwen3-ASR specific)
    - Hallucination removal
    - Trailing fragment cleanup
    - Punctuation normalization
    - Text normalization (half/full-width, etc.)
    - Layer 2 term correction
    - Phonetic similarity correction
    """
    if not text:
        return ""

    # Unicode replacement character
    if '\ufffd' in text:
        log.debug("Segment with replacement char dropped: '%s'", text[:40])
        return ""

    # Single-char drop
    if len(text) <= 1:
        log.debug("Single-char segment dropped: '%s'", text)
        return ""

    # Repetitive kana: for short text (<50 chars), drop entirely.
    # For longer text, only flag — the word-repeat collapse below will clean it.
    if len(text) < 50 and _has_repetitive_kana(text):
        log.info("Repetitive kana filtered: '%s'", text[:80])
        return ""

    # Qwen3-ASR specific: collapse word-level repetitions
    # e.g., "カボ。カボ。カボ。カボ。" → "カボ。"
    text = _collapse_word_repeats(text)
    if not text:
        return ""

    # Hallucination removal
    cleaned = _remove_hallucinations(text)
    if not cleaned:
        log.info("Hallucination removed: '%s'", text[:80])
        return ""
    text = cleaned

    # Trailing fragment cleanup
    text = _clean_trailing_fragments(text)

    # Punctuation normalization
    text = _normalize_punctuation(text)
    if not text:
        return ""

    # Text normalization (half/full-width, duplicate chars, etc.)
    try:
        from text_normalizer import normalize as _normalize_text
        text = _normalize_text(text)
    except ImportError:
        pass

    # Layer 2: Term correction (dictionary-based)
    if term_corrector is not None:
        text = term_corrector.correct(text)

    # Layer 2.5: Phonetic similarity correction
    if phonetic_corrector is not None:
        text = phonetic_corrector.correct(text)

    return text


def _remove_hallucinations(text: str) -> str:
    """Remove known hallucination patterns."""
    stripped = text.strip()
    for pattern in _HALLUCINATION_FULL_MATCH:
        if pattern.match(stripped):
            return ""
    cleaned = text
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _collapse_word_repeats(text: str) -> str:
    """Collapse word-level repetitions specific to Qwen3-ASR.

    Qwen3-ASR sometimes repeats the same word with punctuation separators:
    "カボ。カボ。カボ。カボ。" → "カボ。"
    "バナナ。バナナ。バナナ。" → "バナナ。"

    Also handles comma-separated repeats:
    "高松。高松。高松ね。" → "高松ね。"
    """
    result = _WORD_REPEAT_RE.sub(r'\1。', text)
    # Collapse aizuchi repeats: はいはいはいはい → はい
    result = _AIZUCHI_REPEAT_RE.sub(r'\1', result)
    # If collapse reduced text to mostly nothing, return cleaned version
    stripped = result.strip()
    if not stripped:
        return text  # preserve original if collapse eliminated everything
    return stripped


def _clean_trailing_fragments(text: str) -> str:
    """Remove trailing comma-separated fragment noise."""
    cleaned = _TRAILING_FRAGMENT_RE.sub("", text).strip()
    if not cleaned:
        return text
    return cleaned


def _normalize_punctuation(text: str) -> str:
    """Normalize punctuation."""
    result = _PERIOD_RE.sub('。', text)
    result = _DASH_RE.sub(' ', result).strip()
    result = _TRAILING_COMMA_RE.sub('', result)
    result = _CONSECUTIVE_PUNCT_RE.sub(r'\1', result)
    return result.strip()


def _has_repetitive_kana(text: str) -> bool:
    """Detect excessive kana repetition (noise artifacts)."""
    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)
    kana_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', body)
    if len(kana_chars) < 6:
        return False
    unique_count = len(set(kana_chars))
    unique_ratio = unique_count / len(kana_chars)
    if unique_ratio < 0.30:
        return True
    from collections import Counter
    counts = Counter(kana_chars)
    most_common_ratio = counts.most_common(1)[0][1] / len(kana_chars)
    return most_common_ratio >= 0.6


def _is_gibberish_kana(text: str) -> bool:
    """Detect gibberish kana without logprob dependency.

    Simplified version for Qwen3-ASR (no logprob available).
    Only checks structural patterns, not confidence scores.
    """
    if _has_repetitive_kana(text):
        return True

    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)
    if len(body) <= 15:
        return False

    # Contains kanji → legitimate Japanese
    if re.search(r'[\u4e00-\u9fff]', body):
        return False

    katakana_count = len(re.findall(r'[\u30a0-\u30ff]', body))
    hiragana_count = len(re.findall(r'[\u3040-\u309f]', body))
    body_len = len(body)

    katakana_ratio = katakana_count / body_len if body_len else 0
    hiragana_ratio = hiragana_count / body_len if body_len else 0

    # Must be kana-dominant (80%+ — stricter than Whisper since no logprob gating)
    if katakana_ratio < 0.80 and hiragana_ratio < 0.80:
        return False

    # Contains grammatical patterns → real Japanese
    if _PARTICLE_PATTERNS.search(body):
        return False
    if _GRAMMATICAL_ENDINGS.search(body):
        return False

    return True


def _split_long_segment(seg_dict: dict, max_duration: float) -> list[dict]:
    """Split long segment at sentence boundaries.

    Timestamps proportionally distributed by character count.
    Fragments shorter than 5 chars merged into neighbours.
    """
    text = seg_dict["text"]
    start = seg_dict["start"]
    end = seg_dict["end"]
    duration = end - start

    if duration <= max_duration:
        return [seg_dict]

    parts = _SEGMENT_SPLIT_RE.split(text)
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        return [seg_dict]

    # Merge short fragments
    merged: list[str] = []
    for p in parts:
        if merged and len(p.strip()) < 5:
            merged[-1] += p
        else:
            merged.append(p)

    if len(merged) > 1 and len(merged[-1].strip()) < 5:
        merged[-2] += merged[-1]
        merged.pop()

    if len(merged) <= 1:
        return [seg_dict]

    total_chars = sum(len(p) for p in merged)
    if total_chars == 0:
        return [seg_dict]

    result = []
    current_start = start
    for i, part in enumerate(merged):
        ratio = len(part) / total_chars
        part_duration = duration * ratio
        part_end = current_start + part_duration
        if i == len(merged) - 1:
            part_end = end

        new_seg = dict(seg_dict)
        new_seg["text"] = part.strip()
        new_seg["start"] = round(current_start, 2)
        new_seg["end"] = round(part_end, 2)
        if new_seg["text"]:
            result.append(new_seg)

        current_start = part_end

    return result if result else [seg_dict]
