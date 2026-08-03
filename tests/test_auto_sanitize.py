"""Tests for nightly auto-sanitize session selection and the outcome ledger."""
import json
import os
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

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules["faster_whisper.vad"].get_speech_timestamps = mock.MagicMock()
sys.modules["faster_whisper.vad"].VadOptions = mock.MagicMock()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import omi_whisper_bridge as bridge  # noqa: E402


SEGMENTS = [{"text": f"seg{i}", "start": i, "end": i + 1} for i in range(5)]


def _make_session(recordings: Path, sid: str, tags: dict | None,
                  n_segments: int = 5):
    date, time_str = sid.split("/")
    d = recordings / date / time_str
    d.mkdir(parents=True, exist_ok=True)
    session: dict = {"start": f"{date}T00:00:00", "duration_seconds": 60.0}
    if tags is not None:
        session["tags"] = tags
        session["session_tagged"] = True
    data = {"session": session, "segments": SEGMENTS[:n_segments]}
    (d / "transcript.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


HIGH_VALUE_TAGS = {
    "conversation_type": "商談",
    "outcomes": ["意思決定あり"],
    "domains": ["経営"],
    "importance": "高",
}

LOW_VALUE_TAGS = {
    "conversation_type": "独り言・メモ",
    "outcomes": [],
    "domains": ["経営"],
    "importance": "低",
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    recordings = tmp_path / "recordings"
    config = tmp_path / "config"
    recordings.mkdir()
    config.mkdir()
    monkeypatch.setattr(bridge, "RECORDINGS_DIR", str(recordings))
    monkeypatch.setattr(bridge, "CONFIG_DIR", str(config))
    monkeypatch.setattr(
        bridge, "_SANITIZE_LEDGER_PATH", str(config / "sanitize_ledger.json")
    )
    monkeypatch.setattr(bridge, "NIGHTLY_SANITIZE_MAX_SESSIONS", 50)
    monkeypatch.setattr(bridge, "NIGHTLY_SANITIZE_HIGH_VALUE_ONLY", True)
    monkeypatch.setattr(bridge, "NIGHTLY_SANITIZE_MAX_ATTEMPTS", 2)
    return recordings, config


def test_selects_high_value_tagged_sessions(env):
    recordings, config = env
    _make_session(recordings, "2026-07-10/10-00-00", HIGH_VALUE_TAGS)
    _make_session(recordings, "2026-07-11/10-00-00", LOW_VALUE_TAGS)
    _make_session(recordings, "2026-07-12/10-00-00", None)  # untagged

    assert bridge._find_unsanitized_sessions() == ["2026-07-10/10-00-00"]


def test_low_value_included_when_filter_off(env, monkeypatch):
    recordings, config = env
    monkeypatch.setattr(bridge, "NIGHTLY_SANITIZE_HIGH_VALUE_ONLY", False)
    _make_session(recordings, "2026-07-10/10-00-00", HIGH_VALUE_TAGS)
    _make_session(recordings, "2026-07-11/10-00-00", LOW_VALUE_TAGS)

    assert bridge._find_unsanitized_sessions() == [
        "2026-07-10/10-00-00", "2026-07-11/10-00-00",
    ]


def test_skips_already_sanitized(env):
    recordings, config = env
    _make_session(recordings, "2026-07-10/10-00-00", HIGH_VALUE_TAGS)
    rec = {"session_id": "2026-07-10/10-00-00", "type": "summary",
           "input": "i", "output": "o", "risk_level": "low"}
    (config / "safe_training_20260401_000000.jsonl").write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert bridge._find_unsanitized_sessions() == []


def test_skips_personal_only_and_external_audio(env):
    recordings, config = env
    _make_session(recordings, "2026-07-10/10-00-00",
                  {**HIGH_VALUE_TAGS, "domains": ["個人"]})
    _make_session(recordings, "2026-07-11/10-00-00",
                  {**HIGH_VALUE_TAGS, "external_audio_ratio": 0.9})
    _make_session(recordings, "2026-07-12/10-00-00", HIGH_VALUE_TAGS)

    assert bridge._find_unsanitized_sessions() == ["2026-07-12/10-00-00"]


def test_skips_too_few_segments(env):
    recordings, config = env
    _make_session(recordings, "2026-07-10/10-00-00", HIGH_VALUE_TAGS, n_segments=2)
    assert bridge._find_unsanitized_sessions() == []


def test_ledger_controls_retries(env):
    recordings, config = env
    for i, sid in enumerate([
        "2026-07-10/10-00-00",  # done → skip
        "2026-07-11/10-00-00",  # dropped → skip
        "2026-07-12/10-00-00",  # failed x1 → retry
        "2026-07-13/10-00-00",  # failed x2 → skip (max attempts)
    ]):
        _make_session(recordings, sid, HIGH_VALUE_TAGS)

    bridge._record_sanitize_outcome("2026-07-10/10-00-00", "done")
    bridge._record_sanitize_outcome("2026-07-11/10-00-00", "dropped")
    bridge._record_sanitize_outcome("2026-07-12/10-00-00", "failed")
    bridge._record_sanitize_outcome("2026-07-13/10-00-00", "failed")
    bridge._record_sanitize_outcome("2026-07-13/10-00-00", "failed")

    assert bridge._find_unsanitized_sessions() == ["2026-07-12/10-00-00"]


def test_cap_and_oldest_first(env, monkeypatch):
    recordings, config = env
    monkeypatch.setattr(bridge, "NIGHTLY_SANITIZE_MAX_SESSIONS", 2)
    for sid in ["2026-07-12/10-00-00", "2026-07-10/10-00-00", "2026-07-11/10-00-00"]:
        _make_session(recordings, sid, HIGH_VALUE_TAGS)

    assert bridge._find_unsanitized_sessions() == [
        "2026-07-10/10-00-00", "2026-07-11/10-00-00",
    ]


def test_ledger_attempts_increment(env):
    recordings, config = env
    bridge._record_sanitize_outcome("a/b", "failed")
    bridge._record_sanitize_outcome("a/b", "done")
    ledger = bridge._load_sanitize_ledger()
    assert ledger["a/b"]["attempts"] == 2
    assert ledger["a/b"]["outcome"] == "done"
