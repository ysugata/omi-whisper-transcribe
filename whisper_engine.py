"""
WhisperEngine — wraps the existing ModelManager + transcribe_audio()
to satisfy the ASREngine protocol.

This is a thin adapter; all real work stays in model_manager.py and
omi_whisper_bridge.transcribe_audio().
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from model_manager import ModelManager

log = logging.getLogger("omi-whisper.engine.whisper")


class WhisperEngine:
    """ASREngine implementation backed by faster-whisper."""

    def __init__(
        self,
        model_manager: ModelManager,
        transcribe_fn: Callable[..., list[dict]],
    ):
        self._mm = model_manager
        self._transcribe_fn = transcribe_fn

    # --- ASREngine protocol ---

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def loaded(self) -> bool:
        return self._mm.loaded

    @property
    def executor(self) -> ThreadPoolExecutor | None:
        return self._mm._executor

    def transcribe(
        self,
        audio_bytes: bytes,
        session_offset: float,
        hotwords: str,
    ) -> list[dict]:
        if self._mm._model is None:
            raise RuntimeError("Whisper model not loaded")
        return self._transcribe_fn(
            audio_bytes, self._mm._model, session_offset, hotwords,
        )

    async def load(self) -> None:
        await self._mm.load_model()

    async def unload(self) -> None:
        await self._mm.force_unload()

    def status(self) -> dict:
        s = self._mm.status()
        s["engine"] = self.name
        return s
