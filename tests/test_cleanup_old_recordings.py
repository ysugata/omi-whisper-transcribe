"""Tests for cleanup_old_recordings mtime guard.

SD card uploads carry the device clock as the directory date; when that
clock is months behind, freshly ingested sessions land in "old" dirs.
Cleanup must not delete dirs that were written recently.
"""
import os
import sys
import time
import types
from datetime import datetime, timedelta
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

sys.modules["faster_whisper.vad"].get_speech_timestamps = mock.MagicMock()
sys.modules["faster_whisper.vad"].VadOptions = mock.MagicMock()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import omi_whisper_bridge as bridge  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup_after_all():
    yield
    for mod_name, orig in _saved.items():
        if orig is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = orig


def _make_day_dir(base: Path, date_name: str, mtime_days_ago: float) -> Path:
    day_dir = base / date_name
    day_dir.mkdir(parents=True)
    (day_dir / "10-00-00").mkdir()
    ts = time.time() - mtime_days_ago * 86400
    os.utime(day_dir, (ts, ts))
    return day_dir


def _run_cleanup(tmp_path, max_days=30):
    with mock.patch.object(bridge, "SAVE_AUDIO", True), \
         mock.patch.object(bridge, "RECORDINGS_DIR", str(tmp_path)), \
         mock.patch.object(bridge, "MAX_RECORDING_DAYS", max_days):
        bridge.cleanup_old_recordings()


def test_old_dir_with_old_mtime_is_removed(tmp_path):
    old_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    day_dir = _make_day_dir(tmp_path, old_date, mtime_days_ago=90)
    _run_cleanup(tmp_path)
    assert not day_dir.exists()


def test_old_dir_with_recent_mtime_is_kept(tmp_path):
    """Freshly ingested SD sessions in an old-dated dir must survive."""
    old_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    day_dir = _make_day_dir(tmp_path, old_date, mtime_days_ago=0)
    _run_cleanup(tmp_path)
    assert day_dir.exists()


def test_recent_dir_is_kept(tmp_path):
    recent_date = datetime.now().strftime("%Y-%m-%d")
    day_dir = _make_day_dir(tmp_path, recent_date, mtime_days_ago=0)
    _run_cleanup(tmp_path)
    assert day_dir.exists()


def test_disabled_when_max_days_zero(tmp_path):
    old_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    day_dir = _make_day_dir(tmp_path, old_date, mtime_days_ago=90)
    _run_cleanup(tmp_path, max_days=0)
    assert day_dir.exists()


def test_non_date_entries_ignored(tmp_path):
    (tmp_path / ".sdcard_registry.json").write_text("{}", encoding="utf-8")
    misc = tmp_path / "not-a-date"
    misc.mkdir()
    ts = time.time() - 90 * 86400
    os.utime(misc, (ts, ts))
    _run_cleanup(tmp_path)
    assert (tmp_path / ".sdcard_registry.json").exists()
    assert misc.exists()
