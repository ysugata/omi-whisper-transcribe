"""Tests for empty session skip in SessionRecorder.finalize()."""

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

# Mock heavy dependencies before importing omi_whisper_bridge
_MOCK_MODULES = [
    "faster_whisper", "faster_whisper.vad",
    "torch", "torchaudio",
    "whisperx",
    "noisereduce",
    "pyannote", "pyannote.audio",
]

_saved = {}
for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        _saved[mod_name] = None
        sys.modules[mod_name] = types.ModuleType(mod_name)
    else:
        _saved[mod_name] = sys.modules[mod_name]

# Provide stubs for referenced names
sys.modules["faster_whisper.vad"].get_speech_timestamps = mock.MagicMock()
sys.modules["faster_whisper.vad"].VadOptions = mock.MagicMock()


def _cleanup_mocks():
    for mod_name, orig in _saved.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig


@pytest.fixture(autouse=True)
def cleanup_after_all():
    yield
    _cleanup_mocks()


def _make_recorder(tmp_dir, total_bytes=0, segments=None):
    """Create a SessionRecorder with patched paths."""
    import omi_whisper_bridge as bridge

    with mock.patch.object(bridge, "SAVE_AUDIO", True), \
         mock.patch.object(bridge, "RECORDINGS_DIR", str(tmp_dir)):
        recorder = bridge.SessionRecorder()
        if total_bytes > 0:
            silence = b'\x00' * total_bytes
            recorder.write_audio(silence)
        if segments is not None:
            recorder.all_segments = segments
        return recorder


def test_finalize_removes_empty_audio(tmp_path):
    """Session with 0 bytes audio should be removed."""
    recorder = _make_recorder(tmp_path, total_bytes=0, segments=[])
    session_dir = recorder.session_dir
    assert session_dir.exists()
    recorder.finalize()
    assert not session_dir.exists()


def test_finalize_removes_zero_segments(tmp_path):
    """Session with audio but 0 segments (all filtered) should be removed."""
    recorder = _make_recorder(
        tmp_path,
        total_bytes=32000,
        segments=[],
    )
    session_dir = recorder.session_dir
    assert session_dir.exists()
    recorder.finalize()
    assert not session_dir.exists()


def test_finalize_keeps_session_with_segments(tmp_path):
    """Session with audio and segments should be kept."""
    recorder = _make_recorder(
        tmp_path,
        total_bytes=32000,
        segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
    )
    session_dir = recorder.session_dir
    assert session_dir.exists()
    recorder.finalize()
    assert session_dir.exists()
    assert (session_dir / "transcript.json").exists()

    data = json.loads((session_dir / "transcript.json").read_text())
    assert data["segments"][0]["text"] == "Hello"
    assert data["session"]["duration_seconds"] == 1.0
