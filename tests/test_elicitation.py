"""Tests for the daily elicitation store, question parsing, and gap detection."""
import json
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

from elicitation import ElicitationEngine, ElicitationStore  # noqa: E402
import omi_whisper_bridge as bridge  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return ElicitationStore(str(tmp_path / "elicitation_questions.json"))


def test_add_respects_cap_and_dedup(store):
    qs = [{"question": f"質問その{i}はなぜそう判断したか？"} for i in range(5)]
    added = store.add_questions(qs, max_pending=3)
    assert added == 3
    assert len(store.pending()) == 3
    # 同じ質問は追加されない
    added2 = store.add_questions(qs[:1], max_pending=10)
    assert added2 == 0


def test_answer_and_process_lifecycle(store):
    store.add_questions([{"question": "X社を選んだ決め手と捨てた選択肢は？"}], 9)
    qid = store.pending()[0]["id"]

    assert store.answer(qid, "") is None  # 空回答は拒否
    q = store.answer(qid, "価格ではなく納期。Y社は品質が読めなかった。")
    assert q["status"] == "answered"
    assert len(store.pending()) == 0

    unproc = store.unprocessed_answers()
    assert len(unproc) == 1
    store.mark_processed(qid, "done")
    assert store.unprocessed_answers() == []


def test_skip(store):
    store.add_questions([{"question": "この判断が変わる条件は何ですか？"}], 9)
    qid = store.pending()[0]["id"]
    store.skip(qid)
    assert store.pending() == []


def test_parse_questions_filters_short_and_caps():
    raw = json.dumps({"questions": [
        {"question": "短い", "anchor": "a"},
        {"question": "なぜY社ではなくX社を選び、何を捨てましたか？", "anchor": "商談"},
        {"question": "その判断をやめるべきシグナルは何ですか？"},
        {"question": "初めての人が同じ場面で間違えそうな点は？"},
    ]})
    out = ElicitationEngine.parse_questions(raw, max_questions=2)
    assert len(out) == 2
    assert out[0]["question"].startswith("なぜY社")
    assert ElicitationEngine.parse_questions("junk", 3) == []


def test_is_rag_gap():
    gap, reason = bridge._is_rag_gap("この質問に対応する記録はありません。", [], [], 0.45)
    assert gap and reason == "abstained"
    gap, reason = bridge._is_rag_gap("答えです", [], ["no_sources"], 0.45)
    assert gap and reason == "no_sources"
    gap, reason = bridge._is_rag_gap("答えです", [{"similarity": 0.2}], [], 0.45)
    assert gap and reason == "low_similarity"
    gap, _ = bridge._is_rag_gap("答えです", [{"similarity": 0.8}], [], 0.45)
    assert not gap


def test_rag_gap_log_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "_RAG_GAP_LOG_PATH", str(tmp_path / "gaps.json"))
    bridge._append_rag_gap("資金繰りが苦しい時の優先順位は？", 0.2, "low_similarity")
    bridge._append_rag_gap("資金繰りが苦しい時の優先順位は？", 0.2, "low_similarity")  # dedup
    gaps = bridge._load_rag_gaps()
    assert len(gaps) == 1
    bridge._mark_rag_gaps_consumed(["資金繰りが苦しい時の優先順位は？"])
    assert bridge._load_rag_gaps()[0]["consumed"] is True


def test_collect_decision_contexts(tmp_path, monkeypatch):
    recordings = tmp_path / "recordings"
    monkeypatch.setattr(bridge, "RECORDINGS_DIR", str(recordings))
    monkeypatch.setattr(
        bridge, "elicitation_store",
        ElicitationStore(str(tmp_path / "eq.json")),
    )

    def make(sid, tags):
        d = recordings / sid
        d.mkdir(parents=True)
        (d / "transcript.json").write_text(json.dumps({
            "session": {"tags": tags},
            "segments": [{"text": f"発言{i}", "start": i, "end": i + 1} for i in range(4)],
        }, ensure_ascii=False), encoding="utf-8")

    make("2026-08-01/10-00-00", {"conversation_type": "商談",
                                 "outcomes": ["意思決定あり"], "importance": "中",
                                 "summary": "投資判断"})
    make("2026-08-02/10-00-00", {"conversation_type": "雑談・移動中メモ",
                                 "outcomes": [], "importance": "低"})

    ctx = bridge._collect_decision_contexts(max_sessions=3)
    assert [c["session_id"] for c in ctx] == ["2026-08-01/10-00-00"]
    assert "投資判断" in ctx[0]["summary"]

    # 使用済みセッションは再利用しない
    bridge.elicitation_store.add_questions(
        [{"question": "なぜその投資判断をしたのですか、決め手は？",
          "source_ref": "2026-08-01/10-00-00"}], 9,
    )
    assert bridge._collect_decision_contexts(max_sessions=3) == []
