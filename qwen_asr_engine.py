"""
QwenASREngine — Qwen3-ASR-1.7B backend for the ASREngine protocol.

Uses the qwen-asr package (transformers backend, no vLLM required).
Model is loaded/unloaded on demand just like the Whisper engine.
"""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import re
import tempfile
import wave
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger("omi-whisper.engine.qwen")

# Re-use bridge-level config constants
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
_lang = os.environ.get("LANGUAGE", "ja")
LANGUAGE = None if _lang.lower() == "auto" else _lang
SEGMENT_MAX_DURATION = float(os.environ.get("SEGMENT_MAX_DURATION", "15.0"))

# Qwen3-ASR language name mapping (ISO 639-1 → Qwen language name)
_LANG_MAP = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
}

_LOAD_TIMEOUT = 600  # seconds


class QwenASREngine:
    """ASREngine implementation backed by Qwen3-ASR-1.7B."""

    MODEL_NAME = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")

    def __init__(self):
        self._model = None
        self._term_corrector = None  # set externally after init

    # --- ASREngine protocol ---

    @property
    def name(self) -> str:
        return "qwen3-asr"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def set_term_corrector(self, tc) -> None:
        """Allow bridge to inject the shared TermCorrector instance."""
        self._term_corrector = tc

    def transcribe(
        self,
        audio_bytes: bytes,
        session_offset: float,
        hotwords: str,
    ) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Qwen3-ASR model not loaded")

        # PCM 16-bit LE → float32 numpy
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < 1600:  # < 0.1s
            return []

        chunk_duration = len(audio_np) / SAMPLE_RATE

        # Qwen3-ASR accepts (numpy_array, sample_rate) tuples
        qwen_lang = _LANG_MAP.get(LANGUAGE, None) if LANGUAGE else None

        results = self._model.transcribe(
            audio=[(audio_np, SAMPLE_RATE)],
            language=[qwen_lang] if qwen_lang else None,
            return_time_stamps=True,
        )

        if not results:
            return []

        result = results[0]
        text = result.text.strip() if result.text else ""
        if not text:
            return []

        # Build segments from timestamps
        segments_out = self._build_segments(
            result, chunk_duration, session_offset
        )

        # Post-processing: hallucination removal + term correction
        segments_out = self._post_process(segments_out)

        return segments_out

    async def load(self) -> None:
        if self._model is not None:
            log.info("Qwen3-ASR: already loaded")
            return

        log.info(f"Loading Qwen3-ASR model: {self.MODEL_NAME}")
        loop = asyncio.get_running_loop()
        self._model = await asyncio.wait_for(
            loop.run_in_executor(None, self._load_model_sync),
            timeout=_LOAD_TIMEOUT,
        )
        log.info("Qwen3-ASR model loaded successfully")

    async def unload(self) -> None:
        if self._model is None:
            return

        log.info("Unloading Qwen3-ASR model...")
        del self._model
        self._model = None
        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
            log.info("VRAM released (torch.cuda.empty_cache)")
        except ImportError:
            pass

        log.info("Qwen3-ASR model unloaded")

    def status(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self.loaded,
            "model_name": self.MODEL_NAME,
        }

    # --- Internal ---

    def _load_model_sync(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        model = Qwen3ASRModel.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=512,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=torch.float16,
                device_map="cuda:0",
            ),
        )
        return model

    def _build_segments(
        self,
        result,
        chunk_duration: float,
        session_offset: float,
    ) -> list[dict]:
        """Convert Qwen3-ASR result into segment dicts.

        If timestamps are available, split by sentence boundaries.
        Otherwise, create a single segment for the whole chunk.
        """
        full_text = result.text.strip()
        timestamps = getattr(result, "time_stamps", None)

        # If no timestamps, return a single segment
        if not timestamps or not timestamps[0]:
            return [self._make_segment(full_text, 0.0, chunk_duration, session_offset)]

        # Build word-level list from timestamps
        words = []
        for ts in timestamps[0] if isinstance(timestamps[0], list) else timestamps:
            w_text = getattr(ts, "text", str(ts))
            w_start = getattr(ts, "start_time", 0.0)
            w_end = getattr(ts, "end_time", w_start + 0.1)
            words.append({"text": w_text, "start": w_start, "end": w_end})

        if not words:
            return [self._make_segment(full_text, 0.0, chunk_duration, session_offset)]

        # Group words into sentence-level segments (split at 。！？)
        segments = []
        current_words: list[dict] = []

        for w in words:
            current_words.append(w)
            # Sentence boundary check
            if re.search(r'[。！？!?]$', w["text"]):
                seg_text = "".join(cw["text"] for cw in current_words).strip()
                seg_start = current_words[0]["start"]
                seg_end = current_words[-1]["end"]
                if seg_text:
                    segments.append(self._make_segment(
                        seg_text,
                        min(seg_start, chunk_duration),
                        min(seg_end, chunk_duration),
                        session_offset,
                    ))
                current_words = []

        # Remaining words (no sentence-ending punctuation)
        if current_words:
            seg_text = "".join(cw["text"] for cw in current_words).strip()
            seg_start = current_words[0]["start"]
            seg_end = current_words[-1]["end"]
            if seg_text:
                segments.append(self._make_segment(
                    seg_text,
                    min(seg_start, chunk_duration),
                    min(seg_end, chunk_duration),
                    session_offset,
                ))

        # Split over-long segments
        if SEGMENT_MAX_DURATION > 0:
            split_segments = []
            for seg in segments:
                dur = seg["end"] - seg["start"]
                if dur > SEGMENT_MAX_DURATION:
                    split_segments.extend(self._split_long_segment(seg))
                else:
                    split_segments.append(seg)
            segments = split_segments

        return segments if segments else [
            self._make_segment(full_text, 0.0, chunk_duration, session_offset)
        ]

    @staticmethod
    def _make_segment(
        text: str,
        start: float,
        end: float,
        session_offset: float,
    ) -> dict:
        return {
            "text": text,
            "speaker": "SPEAKER_00",
            "speaker_id": 0,
            "start": round(start + session_offset, 2),
            "end": round(end + session_offset, 2),
            "is_user": True,
            "person_id": None,
        }

    @staticmethod
    def _split_long_segment(seg: dict) -> list[dict]:
        """Split a segment longer than SEGMENT_MAX_DURATION at sentence boundaries."""
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        # Split at sentence boundaries
        parts = re.split(r'(?<=[。！？!?])', text)
        parts = [p for p in parts if p.strip()]

        if len(parts) <= 1:
            return [seg]

        total_chars = sum(len(p) for p in parts)
        if total_chars == 0:
            return [seg]

        result = []
        current_start = start
        for p in parts:
            char_ratio = len(p) / total_chars
            part_duration = duration * char_ratio
            part_end = current_start + part_duration

            new_seg = dict(seg)
            new_seg["text"] = p.strip()
            new_seg["start"] = round(current_start, 2)
            new_seg["end"] = round(part_end, 2)
            if new_seg["text"]:
                result.append(new_seg)
            current_start = part_end

        return result if result else [seg]

    def _post_process(self, segments: list[dict]) -> list[dict]:
        """Apply hallucination removal + term correction.

        Qwen3-ASR adds its own punctuation, so we skip _normalize_punctuation.
        logprob/no_speech_prob filters are Whisper-specific, so we skip them too.
        """
        # Import bridge-level filters lazily to avoid circular imports
        from omi_whisper_bridge import (
            _remove_hallucinations,
            _clean_trailing_fragments,
        )

        result = []
        for seg in segments:
            text = seg["text"]

            # Hallucination removal
            cleaned = _remove_hallucinations(text)
            if not cleaned:
                log.info(f"Qwen hallucination removed: '{text[:80]}'")
                continue
            text = cleaned

            # Trailing fragment cleaner
            text = _clean_trailing_fragments(text)

            # Term correction
            if self._term_corrector:
                text = self._term_corrector.correct(text)

            if not text.strip():
                continue

            seg = dict(seg)
            seg["text"] = text
            result.append(seg)

        return result
