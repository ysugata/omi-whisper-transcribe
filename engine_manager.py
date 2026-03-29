"""
EngineManager — manages ASR engine lifecycle and switching.

Only one engine is loaded on GPU at a time. Switching unloads the current
engine and loads the new one. Both engines share the same gpu_lock from
ModelManager so that batch operations (diarization, speaker tagging) remain
mutually exclusive with ASR.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_manager import ModelManager

log = logging.getLogger("omi-whisper.engine-manager")

VALID_ENGINES = ("faster-whisper", "qwen3-asr")


class EngineManager:
    """Manages ASR engine selection, loading, and switching."""

    def __init__(
        self,
        default_engine: str,
        model_manager: ModelManager,
        transcribe_fn,
        term_corrector=None,
    ):
        from whisper_engine import WhisperEngine
        from qwen_asr_engine import QwenASREngine

        self._model_manager = model_manager
        self._gpu_lock = model_manager.gpu_lock
        self._lock = asyncio.Lock()
        self._session_count = 0

        # Build engines
        self._whisper = WhisperEngine(model_manager, transcribe_fn)
        self._qwen = QwenASREngine()
        if term_corrector:
            self._qwen.set_term_corrector(term_corrector)

        self._engines = {
            "faster-whisper": self._whisper,
            "qwen3-asr": self._qwen,
        }

        if default_engine not in self._engines:
            log.warning(f"Unknown engine '{default_engine}', falling back to faster-whisper")
            default_engine = "faster-whisper"

        self._current_name = default_engine
        self._current = self._engines[default_engine]
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")

        log.info(f"EngineManager initialized: default={default_engine}")

    @property
    def current_engine(self):
        return self._current

    @property
    def current_engine_name(self) -> str:
        return self._current_name

    @property
    def gpu_lock(self) -> asyncio.Lock:
        return self._gpu_lock

    @property
    def session_count(self) -> int:
        return self._session_count

    async def acquire(self):
        """Acquire the current engine for a WS session.

        Waits for gpu_lock (blocks during diarization etc.), then
        auto-loads the engine if needed.

        Returns: (engine, executor)
        """
        async with self._gpu_lock:
            async with self._lock:
                self._session_count += 1
                log.info(f"acquire: engine={self._current_name}, sessions={self._session_count}")

                if not self._current.loaded:
                    try:
                        await self._current.load()
                    except BaseException:
                        self._session_count -= 1
                        raise

                return self._current, self._executor

    def release(self):
        """Release engine after WS session ends."""
        self._session_count = max(0, self._session_count - 1)
        log.info(f"release: sessions={self._session_count}")

    async def switch_engine(self, engine_name: str) -> dict:
        """Switch to a different ASR engine.

        Unloads the current engine and loads the new one.
        Active sessions will use the new engine from their next chunk.
        """
        if engine_name not in self._engines:
            return {"error": f"unknown engine: {engine_name}", "valid": list(VALID_ENGINES)}

        if engine_name == self._current_name:
            return {"status": "already_active", "engine": engine_name}

        async with self._gpu_lock:
            async with self._lock:
                log.info(f"Switching engine: {self._current_name} → {engine_name}")

                # Unload current
                try:
                    await self._current.unload()
                except Exception as e:
                    log.warning(f"Error unloading {self._current_name}: {e}")

                # Switch
                self._current_name = engine_name
                self._current = self._engines[engine_name]

                # Load new engine
                try:
                    await self._current.load()
                except Exception as e:
                    log.error(f"Failed to load {engine_name}: {e}", exc_info=True)
                    # Rollback to whisper as safe default
                    self._current_name = "faster-whisper"
                    self._current = self._whisper
                    try:
                        await self._current.load()
                    except Exception:
                        pass
                    return {"error": f"failed to load {engine_name}: {e}"}

                log.info(f"Engine switched to {engine_name}")
                return {"status": "switched", "engine": engine_name}

    async def unload_current(self) -> bool:
        """Unload the current engine (for batch GPU operations)."""
        async with self._lock:
            if self._session_count > 0:
                log.warning(f"unload_current: rejected, sessions={self._session_count}")
                return False
            await self._current.unload()
            return True

    async def force_unload_current(self) -> bool:
        """Force unload ignoring session count (caller must hold gpu_lock)."""
        async with self._lock:
            await self._current.unload()
            return True

    async def load_current(self):
        """Explicitly load the current engine."""
        async with self._lock:
            await self._current.load()

    def status(self) -> dict:
        """Return overall engine status."""
        return {
            "current_engine": self._current_name,
            "session_count": self._session_count,
            "engines": {name: eng.status() for name, eng in self._engines.items()},
        }
