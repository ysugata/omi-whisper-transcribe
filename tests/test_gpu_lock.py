"""Tests for CUDA OOM fix: gpu_lock holding pattern and VRAM cleanup.

Verifies that:
1. diarizer._free_vram() calls torch.cuda.synchronize() before empty_cache()
2. speaker_filter._free_vram() and _unload_model() call torch.cuda.synchronize()
3. _batch_diarize_task() acquires gpu_lock per-session (not for entire batch)
4. _batch_tag_task() acquires gpu_lock per-session (load→process→unload per session)
5. ModelManager.force_unload() ignores session_count and always unloads
6. Batch tasks use force_unload() once before the loop, then per-session locks
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Helper: mock heavy dependencies for omi_whisper_bridge import
# =====================================================================

def _mock_bridge_deps():
    """Pre-mock modules that omi_whisper_bridge imports at top level.

    These packages (faster_whisper, whisperx, torch, etc.) are only
    available inside the Docker container, so we stub them out for
    host-side unit testing.
    """
    stubs = {}
    for mod_name in [
        "faster_whisper", "faster_whisper.vad",
        "torch", "torch.cuda",
        "whisperx", "whisperx.diarize",
        "pyannote", "pyannote.audio",
        "torchaudio", "torchaudio.functional",
        "noisereduce",
        "openai",
    ]:
        if mod_name not in sys.modules:
            stubs[mod_name] = MagicMock()

    # faster_whisper.vad needs specific names
    if "faster_whisper.vad" in stubs:
        stubs["faster_whisper.vad"].get_speech_timestamps = MagicMock()
        stubs["faster_whisper.vad"].VadOptions = MagicMock()

    sys.modules.update(stubs)
    return stubs


# =====================================================================
# 1. diarizer._free_vram() — torch.cuda.synchronize() call
# =====================================================================

class TestDiarizerFreeVram:
    def test_free_vram_calls_synchronize_before_empty_cache(self):
        """_free_vram() must call synchronize() before empty_cache()."""
        from diarizer import Diarizer

        d = Diarizer(hf_token="dummy", device="cpu")

        call_order = []
        mock_torch = MagicMock()
        mock_torch.cuda.synchronize = lambda: call_order.append("synchronize")
        mock_torch.cuda.empty_cache = lambda: call_order.append("empty_cache")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            d._free_vram()

        assert "synchronize" in call_order, "synchronize() was not called"
        assert "empty_cache" in call_order, "empty_cache() was not called"
        assert call_order.index("synchronize") < call_order.index("empty_cache"), \
            "synchronize() must be called before empty_cache()"

    def test_run_pipeline_cleans_up_before_return(self):
        """_run_pipeline() should call _free_vram() after diarization model release
        and after speaker assignment to release intermediate VRAM."""
        from diarizer import Diarizer

        d = Diarizer(hf_token="dummy", device="cpu")

        # Count _free_vram calls
        free_vram_calls = []
        def counting_free_vram():
            free_vram_calls.append(1)
        d._free_vram = counting_free_vram

        segments = [{"text": "hello", "start": 0.0, "end": 1.0}]

        # Mock torchaudio.load → return 2s waveform
        mock_torchaudio = MagicMock()
        mock_waveform = MagicMock()
        mock_waveform.shape = [1, 32000]  # 2s at 16kHz
        mock_torchaudio.load.return_value = (mock_waveform, 16000)

        # Mock pyannote Pipeline
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]

        mock_pipeline_cls = MagicMock()
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_annotation
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

        mock_pyannote_audio = MagicMock(Pipeline=mock_pipeline_cls)

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torchaudio": mock_torchaudio,
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
        }):
            result = d._run_pipeline("dummy.wav", segments, None, None)

        # Should have 2 _free_vram calls:
        # 1. After diarization model release
        # 2. After speaker assignment (cleanup before return)
        assert len(free_vram_calls) >= 2, \
            f"Expected at least 2 _free_vram() calls, got {len(free_vram_calls)}"


# =====================================================================
# 2. speaker_filter._free_vram() and _unload_model()
# =====================================================================

class TestSpeakerFilterVram:
    def test_free_vram_calls_synchronize(self):
        """_free_vram() must call torch.cuda.synchronize()."""
        from speaker_filter import SpeakerFilter

        call_order = []
        mock_torch = MagicMock()
        mock_torch.cuda.synchronize = lambda: call_order.append("synchronize")
        mock_torch.cuda.empty_cache = lambda: call_order.append("empty_cache")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            SpeakerFilter._free_vram()

        assert "synchronize" in call_order
        assert call_order.index("synchronize") < call_order.index("empty_cache")

    def test_unload_model_calls_synchronize(self):
        """_unload_model() must call torch.cuda.synchronize()."""
        from speaker_filter import SpeakerFilter

        call_order = []
        mock_torch = MagicMock()
        mock_torch.cuda.synchronize = lambda: call_order.append("synchronize")
        mock_torch.cuda.empty_cache = lambda: call_order.append("empty_cache")

        mock_inference = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            SpeakerFilter._unload_model(mock_inference)

        assert "synchronize" in call_order
        assert call_order.index("synchronize") < call_order.index("empty_cache")


# =====================================================================
# 3. _batch_diarize_task() — per-session gpu_lock
# =====================================================================

class TestBatchDiarizeLock:
    """Verify gpu_lock is acquired per-session to avoid system freeze."""

    @pytest.fixture
    def recordings_dir(self, tmp_path):
        """Create a temp recordings dir with 2 sessions."""
        for i, time_name in enumerate(["10-00-00", "11-00-00"]):
            session_dir = tmp_path / "2026-03-29" / time_name
            session_dir.mkdir(parents=True)
            (session_dir / "audio.ogg").write_bytes(b"\x00" * 100)
            transcript = {
                "session": {"audio_file": "audio.ogg"},
                "segments": [{"text": f"test{i}", "start": 0.0, "end": 1.0}],
            }
            with open(session_dir / "transcript.json", "w") as f:
                json.dump(transcript, f)
        return tmp_path

    @pytest.mark.asyncio
    async def test_gpu_lock_per_session(self, recordings_dir):
        """gpu_lock must be acquired per-session (1 for force_unload + 1 per session)."""
        _mock_bridge_deps()

        lock_events = []

        class TrackingLock:
            def __init__(self):
                self._lock = asyncio.Lock()
            async def __aenter__(self):
                await self._lock.acquire()
                lock_events.append("acquire")
                return self
            async def __aexit__(self, *args):
                lock_events.append("release")
                self._lock.release()

        mock_mm = MagicMock()
        mock_mm.gpu_lock = TrackingLock()
        mock_mm.force_unload = AsyncMock(return_value=True)
        # Prevent per-session force_unload inside loop (loaded is False after initial unload)
        mock_mm.loaded = False

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = (
            [{"text": "test", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            True,
        )

        import omi_whisper_bridge as bridge
        original_mm = bridge.model_manager
        original_diarizer = bridge.diarizer
        original_recordings = bridge.RECORDINGS_DIR

        try:
            bridge.model_manager = mock_mm
            bridge.diarizer = mock_diarizer
            bridge.RECORDINGS_DIR = str(recordings_dir)

            await bridge._batch_diarize_task(force=True)
        finally:
            bridge.model_manager = original_mm
            bridge.diarizer = original_diarizer
            bridge.RECORDINGS_DIR = original_recordings

        # 1 acquire for force_unload + 2 acquires for 2 sessions = 3
        assert lock_events.count("acquire") == 3, \
            f"gpu_lock should be acquired 3 times (1 force_unload + 2 sessions), got {lock_events.count('acquire')}"
        assert lock_events.count("release") == 3, \
            f"gpu_lock should be released 3 times, got {lock_events.count('release')}"
        assert mock_mm.force_unload.call_count == 1, \
            f"force_unload should be called once, got {mock_mm.force_unload.call_count}"
        assert mock_diarizer.diarize.call_count == 2, \
            f"Both sessions should be processed, got {mock_diarizer.diarize.call_count}"

    @pytest.mark.asyncio
    async def test_lock_released_between_sessions(self, recordings_dir):
        """gpu_lock must be released between sessions so WS handlers can run."""
        _mock_bridge_deps()

        lock_held = []

        class TrackingLock:
            def __init__(self):
                self._lock = asyncio.Lock()
                self.is_locked = False
            async def __aenter__(self):
                await self._lock.acquire()
                self.is_locked = True
                return self
            async def __aexit__(self, *args):
                self.is_locked = False
                self._lock.release()

        tracking_lock = TrackingLock()

        mock_mm = MagicMock()
        mock_mm.gpu_lock = tracking_lock
        mock_mm.force_unload = AsyncMock(return_value=True)

        session_count = [0]
        def diarize_check_lock(*args, **kwargs):
            lock_held.append(tracking_lock.is_locked)
            session_count[0] += 1
            return ([{"text": "t", "start": 0, "end": 1, "speaker": "SPEAKER_00"}], True)

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.side_effect = diarize_check_lock

        import omi_whisper_bridge as bridge
        original_mm = bridge.model_manager
        original_diarizer = bridge.diarizer
        original_recordings = bridge.RECORDINGS_DIR

        try:
            bridge.model_manager = mock_mm
            bridge.diarizer = mock_diarizer
            bridge.RECORDINGS_DIR = str(recordings_dir)

            await bridge._batch_diarize_task(force=True)
        finally:
            bridge.model_manager = original_mm
            bridge.diarizer = original_diarizer
            bridge.RECORDINGS_DIR = original_recordings

        assert all(lock_held), "diarize() must run while gpu_lock is held"
        assert not tracking_lock.is_locked, "gpu_lock must be released after batch"

    @pytest.mark.asyncio
    async def test_cancel_breaks_loop_releases_lock(self, recordings_dir):
        """Cancel should break the loop and release the lock."""
        _mock_bridge_deps()

        mock_mm = MagicMock()
        mock_mm.gpu_lock = asyncio.Lock()
        mock_mm.force_unload = AsyncMock(return_value=True)

        import omi_whisper_bridge as bridge

        def diarize_with_cancel(*args, **kwargs):
            bridge._batch_state["cancel_requested"] = True
            return ([{"text": "t", "start": 0, "end": 1, "speaker": "SPEAKER_00"}], True)

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.side_effect = diarize_with_cancel

        original_mm = bridge.model_manager
        original_diarizer = bridge.diarizer
        original_recordings = bridge.RECORDINGS_DIR

        try:
            bridge.model_manager = mock_mm
            bridge.diarizer = mock_diarizer
            bridge.RECORDINGS_DIR = str(recordings_dir)

            await bridge._batch_diarize_task(force=True)
        finally:
            bridge.model_manager = original_mm
            bridge.diarizer = original_diarizer
            bridge.RECORDINGS_DIR = original_recordings

        assert not mock_mm.gpu_lock.locked(), "gpu_lock must be released after cancel"
        assert mock_diarizer.diarize.call_count == 1


# =====================================================================
# 4. _batch_tag_task() — per-session gpu_lock with load→process→unload
# =====================================================================

class TestBatchTagLock:
    """Verify gpu_lock is acquired per-session with model load/unload each time."""

    @pytest.fixture
    def recordings_dir(self, tmp_path):
        for i, time_name in enumerate(["10-00-00", "11-00-00"]):
            session_dir = tmp_path / "2026-03-29" / time_name
            session_dir.mkdir(parents=True)
            (session_dir / "audio.ogg").write_bytes(b"\x00" * 100)
            transcript = {
                "session": {"audio_file": "audio.ogg"},
                "segments": [{"text": f"test{i}", "start": 0.0, "end": 1.0}],
            }
            with open(session_dir / "transcript.json", "w") as f:
                json.dump(transcript, f)
        return tmp_path

    @pytest.mark.asyncio
    async def test_gpu_lock_per_session_tag(self, recordings_dir):
        """gpu_lock must be acquired per-session (1 for force_unload + 1 per session)."""
        _mock_bridge_deps()

        lock_events = []
        processing_while_locked = []

        class TrackingLock:
            def __init__(self):
                self._lock = asyncio.Lock()
                self.is_locked = False
            async def __aenter__(self):
                await self._lock.acquire()
                self.is_locked = True
                lock_events.append("acquire")
                return self
            async def __aexit__(self, *args):
                self.is_locked = False
                lock_events.append("release")
                self._lock.release()

        tracking_lock = TrackingLock()

        mock_mm = MagicMock()
        mock_mm.gpu_lock = tracking_lock
        mock_mm.force_unload = AsyncMock(return_value=True)
        # Prevent per-session force_unload inside loop (loaded is False after initial unload)
        mock_mm.loaded = False

        mock_inference = MagicMock()

        mock_sf = MagicMock()
        mock_sf._load_model.return_value = mock_inference
        mock_sf.tag_segments_with_model.side_effect = lambda inf, audio, segs: (
            processing_while_locked.append(tracking_lock.is_locked),
            segs,
        )[1]
        mock_sf._unload_model = MagicMock()

        import omi_whisper_bridge as bridge
        original_mm = bridge.model_manager
        original_sf = bridge.speaker_filter
        original_recordings = bridge.RECORDINGS_DIR

        try:
            bridge.model_manager = mock_mm
            bridge.speaker_filter = mock_sf
            bridge.RECORDINGS_DIR = str(recordings_dir)

            await bridge._batch_tag_task(force=True)
        finally:
            bridge.model_manager = original_mm
            bridge.speaker_filter = original_sf
            bridge.RECORDINGS_DIR = original_recordings

        # 1 acquire for force_unload + 2 acquires for 2 sessions = 3
        assert lock_events.count("acquire") == 3, \
            f"gpu_lock should be acquired 3 times, got {lock_events.count('acquire')}"
        assert lock_events.count("release") == 3
        assert all(processing_while_locked), \
            f"All tag operations must happen while gpu_lock is held, got {processing_while_locked}"
        assert mock_mm.force_unload.call_count == 1
        # Model loaded/unloaded per session
        assert mock_sf._load_model.call_count == 2
        assert mock_sf._unload_model.call_count == 2

    @pytest.mark.asyncio
    async def test_model_unloaded_even_on_error(self, recordings_dir):
        """Embedding model must be unloaded even if processing fails."""
        _mock_bridge_deps()

        mock_mm = MagicMock()
        mock_mm.gpu_lock = asyncio.Lock()
        mock_mm.force_unload = AsyncMock(return_value=True)

        mock_inference = MagicMock()
        mock_sf = MagicMock()
        mock_sf._load_model.return_value = mock_inference
        mock_sf.tag_segments_with_model.side_effect = RuntimeError("GPU error")
        mock_sf._unload_model = MagicMock()

        import omi_whisper_bridge as bridge
        original_mm = bridge.model_manager
        original_sf = bridge.speaker_filter
        original_recordings = bridge.RECORDINGS_DIR

        try:
            bridge.model_manager = mock_mm
            bridge.speaker_filter = mock_sf
            bridge.RECORDINGS_DIR = str(recordings_dir)

            await bridge._batch_tag_task(force=True)
        finally:
            bridge.model_manager = original_mm
            bridge.speaker_filter = original_sf
            bridge.RECORDINGS_DIR = original_recordings

        # Model should be unloaded for each session, even on error
        assert mock_sf._unload_model.call_count == 2, \
            f"Embedding model must be unloaded per session even after errors, got {mock_sf._unload_model.call_count}"
        assert not mock_mm.gpu_lock.locked(), \
            "gpu_lock must be released after batch completes"


# =====================================================================
# 5. ModelManager.force_unload() — ignores session_count
# =====================================================================

class TestModelManagerForceUnload:
    """Verify force_unload() always unloads regardless of session_count."""

    @pytest.mark.asyncio
    async def test_force_unload_with_active_sessions(self):
        """force_unload() must unload even when session_count > 0."""
        from model_manager import ModelManager

        mm = ModelManager("tiny", "int8")
        # Simulate a loaded model with active sessions
        mm._model = MagicMock()
        mm._executor = MagicMock()
        mm._session_count = 3

        # Patch torch to avoid ImportError
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = await mm.force_unload()

        assert result is True
        assert mm._model is None
        assert mm._executor is None
        # session_count is NOT reset — callers handle reconnection
        assert mm._session_count == 3

    @pytest.mark.asyncio
    async def test_force_unload_already_unloaded(self):
        """force_unload() returns True when model is already unloaded."""
        from model_manager import ModelManager

        mm = ModelManager("tiny", "int8")
        mm._model = None

        result = await mm.force_unload()
        assert result is True

    @pytest.mark.asyncio
    async def test_unload_model_rejected_with_active_sessions(self):
        """Regular unload_model() must be rejected when sessions are active."""
        from model_manager import ModelManager

        mm = ModelManager("tiny", "int8")
        mm._model = MagicMock()
        mm._session_count = 1

        result = await mm.unload_model()
        assert result is False
        assert mm._model is not None  # Not unloaded

    @pytest.mark.asyncio
    async def test_force_unload_calls_unload_model_internal(self):
        """force_unload() must call _unload_model() (the internal method)."""
        from model_manager import ModelManager

        mm = ModelManager("tiny", "int8")
        mm._model = MagicMock()
        mm._session_count = 2

        with patch.object(mm, "_unload_model") as mock_internal:
            result = await mm.force_unload()

        assert result is True
        mock_internal.assert_called_once()
