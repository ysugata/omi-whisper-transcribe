"""Unit tests for the ASR engine abstraction layer."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================
# WhisperEngine tests
# ============================================================

class TestWhisperEngine:
    def test_name(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm.loaded = False
        engine = WhisperEngine(mm, lambda *a, **k: [])
        assert engine.name == "faster-whisper"

    def test_loaded_delegates(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm.loaded = True
        engine = WhisperEngine(mm, lambda *a, **k: [])
        assert engine.loaded is True
        mm.loaded = False
        assert engine.loaded is False

    def test_transcribe_calls_fn(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm._model = "fake-model"
        fake_fn = MagicMock(return_value=[{"text": "hello"}])
        engine = WhisperEngine(mm, fake_fn)

        result = engine.transcribe(b"\x00\x00", 1.0, "word1,word2")
        fake_fn.assert_called_once_with(b"\x00\x00", "fake-model", 1.0, "word1,word2")
        assert result == [{"text": "hello"}]

    def test_transcribe_raises_if_not_loaded(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm._model = None
        engine = WhisperEngine(mm, lambda *a: [])
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.transcribe(b"\x00\x00", 0.0, "")

    @pytest.mark.asyncio
    async def test_load_delegates(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm.load_model = AsyncMock()
        engine = WhisperEngine(mm, lambda *a: [])
        await engine.load()
        mm.load_model.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unload_delegates(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm.force_unload = AsyncMock(return_value=True)
        engine = WhisperEngine(mm, lambda *a: [])
        await engine.unload()
        mm.force_unload.assert_awaited_once()

    def test_status(self):
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm.status.return_value = {"loaded": True, "model_name": "large-v3"}
        engine = WhisperEngine(mm, lambda *a: [])
        s = engine.status()
        assert s["engine"] == "faster-whisper"
        assert s["loaded"] is True


# ============================================================
# EngineManager tests
# ============================================================

class TestEngineManager:
    @pytest.fixture
    def mock_mm(self):
        mm = MagicMock()
        mm.gpu_lock = asyncio.Lock()
        mm.loaded = False
        mm._model = None
        mm._executor = None
        mm.load_model = AsyncMock()
        mm.force_unload = AsyncMock(return_value=True)
        mm.unload_model = AsyncMock(return_value=True)
        mm.status.return_value = {"loaded": False, "model_name": "large-v3"}
        return mm

    def test_init_default_whisper(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)
        assert em.current_engine_name == "faster-whisper"

    def test_init_default_qwen(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("qwen3-asr", mock_mm, lambda *a: [], None)
        assert em.current_engine_name == "qwen3-asr"

    def test_init_invalid_falls_back(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("invalid-engine", mock_mm, lambda *a: [], None)
        assert em.current_engine_name == "faster-whisper"

    @pytest.mark.asyncio
    async def test_acquire_loads_engine(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

        engine, executor = await em.acquire()
        assert engine.name == "faster-whisper"
        assert em.session_count == 1
        mock_mm.load_model.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_release(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

        await em.acquire()
        assert em.session_count == 1
        em.release()
        assert em.session_count == 0

    @pytest.mark.asyncio
    async def test_switch_engine(self, mock_mm):
        from engine_manager import EngineManager

        with patch("qwen_asr_engine.QwenASREngine.load", new_callable=AsyncMock) as qwen_load:
            em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

            result = await em.switch_engine("qwen3-asr")
            assert result["status"] == "switched"
            assert em.current_engine_name == "qwen3-asr"
            qwen_load.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_switch_same_engine(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

        result = await em.switch_engine("faster-whisper")
        assert result["status"] == "already_active"

    @pytest.mark.asyncio
    async def test_switch_invalid_engine(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

        result = await em.switch_engine("nonexistent")
        assert "error" in result

    def test_status(self, mock_mm):
        from engine_manager import EngineManager
        em = EngineManager("faster-whisper", mock_mm, lambda *a: [], None)

        s = em.status()
        assert s["current_engine"] == "faster-whisper"
        assert "faster-whisper" in s["engines"]
        assert "qwen3-asr" in s["engines"]


# ============================================================
# QwenASREngine tests (mocked model)
# ============================================================

class TestQwenASREngine:
    def test_name(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        assert engine.name == "qwen3-asr"

    def test_not_loaded_initially(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        assert engine.loaded is False

    def test_transcribe_raises_when_not_loaded(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.transcribe(b"\x00\x00", 0.0, "")

    def test_status(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        s = engine.status()
        assert s["engine"] == "qwen3-asr"
        assert s["loaded"] is False

    @pytest.mark.asyncio
    async def test_unload_when_not_loaded(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        # Should not raise
        await engine.unload()

    def test_build_segments_no_timestamps(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()

        result = MagicMock()
        result.text = "こんにちは世界"
        result.time_stamps = None

        segments = engine._build_segments(result, 5.0, 0.0)
        assert len(segments) == 1
        assert segments[0]["text"] == "こんにちは世界"
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 5.0

    def test_build_segments_with_timestamps(self):
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()

        # Mock timestamps
        ts1 = MagicMock(text="こんにちは。", start_time=0.5, end_time=2.0)
        ts2 = MagicMock(text="世界", start_time=2.5, end_time=3.5)

        result = MagicMock()
        result.text = "こんにちは。世界"
        result.time_stamps = [[ts1, ts2]]

        segments = engine._build_segments(result, 5.0, 0.0)
        # Should split at 。
        assert len(segments) == 2
        assert segments[0]["text"] == "こんにちは。"
        assert segments[1]["text"] == "世界"


# ============================================================
# ASREngine Protocol conformance
# ============================================================

class TestProtocol:
    def test_whisper_engine_conforms(self):
        from asr_engine import ASREngine
        from whisper_engine import WhisperEngine
        mm = MagicMock()
        mm._model = "fake"
        engine = WhisperEngine(mm, lambda *a: [])
        assert isinstance(engine, ASREngine)

    def test_qwen_engine_conforms(self):
        from asr_engine import ASREngine
        from qwen_asr_engine import QwenASREngine
        engine = QwenASREngine()
        assert isinstance(engine, ASREngine)
