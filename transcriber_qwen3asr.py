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

# Compatibility shim: qwen-asr 0.0.6 uses @check_model_inputs() (decorator factory)
# but transformers >=5.x changed it to @check_model_inputs (direct decorator).
# Patch before importing qwen_asr to support both calling conventions.
try:
    import transformers.utils.generic as _tug
    if hasattr(_tug, "check_model_inputs"):
        _orig_cmi = _tug.check_model_inputs

        def _compat_cmi(func=None):
            if func is None:
                return _orig_cmi
            return _orig_cmi(func)

        _tug.check_model_inputs = _compat_cmi
except (ImportError, AttributeError):
    pass

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

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
            max_new_tokens=512,
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

        # Audio preprocessing (loudness normalization)
        if audio_preprocessor is not None:
            audio_np = audio_preprocessor.process_array(audio_np, sample_rate)

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

        # Build context from hotwords (comma-separated → space-separated)
        context = []
        if hotwords:
            terms = [w.strip() for w in hotwords.split(",") if w.strip()]
            if terms:
                context = [" ".join(terms)]

        # Build transcribe kwargs
        transcribe_kwargs = {}
        if language:
            transcribe_kwargs["language"] = language
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

        # Parse results
        segments_out = []
        for result in results:
            text = result.text.strip() if hasattr(result, 'text') else str(result).strip()

            if not text:
                continue

            detected_lang = None
            if hasattr(result, 'language'):
                detected_lang = result.language

            # Apply engine-agnostic filters
            text = _filter_text(text, term_corrector)
            if not text:
                continue

            # Qwen3-ASR returns a single text per audio input (no segment timestamps).
            # Create one segment spanning the chunk duration.
            seg_dict = {
                "text": text,
                "speaker": "SPEAKER_00",
                "speaker_id": 0,
                "start": round(session_offset, 2),
                "end": round(session_offset + chunk_duration, 2),
                "is_user": True,
                "person_id": None,
            }
            if detected_lang:
                seg_dict["language"] = detected_lang

            # Split long segments at sentence boundaries
            if segment_max_duration > 0 and chunk_duration > segment_max_duration:
                split_segs = _split_long_segment(seg_dict, segment_max_duration)
                segments_out.extend(split_segs)
            else:
                segments_out.append(seg_dict)

        return segments_out


# ---- Text filtering (engine-agnostic Layer 1 + Layer 2) ----

def _filter_text(text: str, term_corrector=None) -> str:
    """Apply engine-agnostic text filters.

    Filters applied:
    - Empty/whitespace check
    - Unicode replacement char
    - Single-char drop
    - Repetitive kana detection
    - Character rate (skipped here, needs duration)
    - Hallucination removal
    - Trailing fragment cleanup
    - Punctuation normalization
    - Layer 2 term correction
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

    # Repetitive kana (logprob-independent check)
    if _has_repetitive_kana(text):
        log.info("Repetitive kana filtered: '%s'", text[:80])
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

    # Layer 2: Term correction
    if term_corrector is not None:
        text = term_corrector.correct(text)

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
