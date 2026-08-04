"""Tests for the quality suite: KC judge parsing, eval metrics, media gate."""
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

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
sys.path.insert(0, str(ROOT / "tools"))

from kc_judge import KCJudge  # noqa: E402
from eval_quality import accuracy, kc_score_summary, set_prf  # noqa: E402
import omi_whisper_bridge as bridge  # noqa: E402


# ---------------------------------------------------------------------------
# KCJudge.parse_scores
# ---------------------------------------------------------------------------

def test_parse_scores_index_mapping():
    raw = '{"scores": [{"index": 2, "score": 4, "reason": "具体的"}, {"index": 1, "score": 2, "reason": "曖昧"}]}'
    out = KCJudge.parse_scores(raw, 3)
    assert out[0] == {"score": 2, "reason": "曖昧"}
    assert out[1] == {"score": 4, "reason": "具体的"}
    assert out[2] is None  # missing entry stays None (not shifted)


def test_parse_scores_rejects_garbage():
    assert KCJudge.parse_scores("not json", 2) == [None, None]
    assert KCJudge.parse_scores('{"scores": "x"}', 1) == [None]
    out = KCJudge.parse_scores(
        '{"scores": [{"index": 9, "score": 5}, {"index": 1, "score": 7}]}', 2
    )
    assert out == [None, None]  # out-of-range index and score are dropped


# ---------------------------------------------------------------------------
# eval metrics
# ---------------------------------------------------------------------------

def test_set_prf():
    m = set_prf([["a", "b"], ["c"]], [["a"], ["c", "d"]])
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == round(2 / 3, 3)
    assert m["recall"] == round(2 / 3, 3)


def test_accuracy_ignores_empty_golden():
    m = accuracy(["x", "", "y"], ["x", "z", "z"])
    assert m["n"] == 2
    assert m["accuracy"] == 0.5


def test_kc_score_summary(monkeypatch):
    s = kc_score_summary([5, 3, 2, None, 4])
    assert s["n"] == 4
    assert s["mean"] == 3.5
    assert s["judge_failures"] == 1
    assert s["distribution"]["3"] == 1


# ---------------------------------------------------------------------------
# media gate decision + KC quality attach
# ---------------------------------------------------------------------------

def test_is_media_scene():
    assert bridge._is_media_scene({"scene_label_en": "media"}) is True
    assert bridge._is_media_scene({"scene_label_en": "conversation"}) is False
    assert bridge._is_media_scene(None) is False


def test_attach_kc_quality(monkeypatch):
    fake = mock.MagicMock()
    fake.enabled = True
    fake.judge.return_value = [
        {"score": 4, "reason": "良い"}, None, {"score": 2, "reason": "曖昧"},
    ]
    monkeypatch.setattr(bridge, "kc_judge", fake)
    rec = {
        "knowledge_cards": [
            {"question": "q1", "answer": "a1"},
            {"question": "q2", "answer": "a2"},
        ],
        "knowledge_card": {"question": "q3", "answer": "a3"},
    }
    bridge._attach_kc_quality(rec)
    assert rec["knowledge_cards"][0]["quality"] == 4
    assert "quality" not in rec["knowledge_cards"][1]  # judge failure → no score
    assert rec["knowledge_card"]["quality"] == 2


def test_attach_kc_quality_chat_records(monkeypatch):
    fake = mock.MagicMock()
    fake.enabled = True
    fake.judge.return_value = [{"score": 5, "reason": ""}]
    monkeypatch.setattr(bridge, "kc_judge", fake)
    records = [
        {"type": "summary", "input": "i", "output": "o"},
        {"type": "knowledge_card", "input": "q", "output": "a"},
    ]
    bridge._attach_kc_quality_chat_records(records)
    assert records[1]["quality"] == 5
    assert "quality" not in records[0]
