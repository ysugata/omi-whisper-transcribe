"""
ASR Engine interface — abstract Protocol for swappable speech-to-text backends.

All engines must produce segments in the same dict format so that downstream
processing (LLM correction, diarization, speaker tagging) is engine-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ASREngine(Protocol):
    """Common interface implemented by every ASR backend."""

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'faster-whisper' or 'qwen3-asr'."""
        ...

    @property
    def loaded(self) -> bool:
        """True when the model is on GPU and ready to transcribe."""
        ...

    def transcribe(
        self,
        audio_bytes: bytes,
        session_offset: float,
        hotwords: str,
    ) -> list[dict]:
        """Transcribe a PCM 16-bit LE audio chunk.

        Returns a list of segment dicts with at least:
            text, start, end, speaker, speaker_id, is_user, person_id
        Timestamps are relative to the *chunk* (not the session).
        """
        ...

    async def load(self) -> None:
        """Load model onto GPU (idempotent)."""
        ...

    async def unload(self) -> None:
        """Release VRAM (idempotent)."""
        ...

    def status(self) -> dict:
        """Return a JSON-serialisable status dict."""
        ...
