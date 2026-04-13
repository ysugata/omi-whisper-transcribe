"""Unit tests for exporter.py (V2 KC-sourced).

Since the V2 rewrite, chat-style formats (openai, conversation, knowledge,
decision_log, sop) source their data from sanitized JSONL records rather
than raw transcript speaker mapping. These tests verify the new behavior.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omi-recordings-viewer"))

from exporter import (
    SanitizedIndex,
    export_session,
    merge_turns,
    to_comprehensive,
    to_conversation,
    to_decision_log,
    to_knowledge,
    to_openai,
    to_sop,
    _parse_bullet_list,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_transcript(segments, tags=None):
    session = {"start": "2026-03-28T00:26:29", "duration_seconds": 120.0, "language": "ja"}
    if tags:
        session["tags"] = tags
    return {"session": session, "segments": segments}


SEGMENTS_CONV = [
    {"text": "こんにちは", "start": 0, "end": 2, "person_name": "自分", "is_user": True},
    {"text": "元気ですか", "start": 2, "end": 4, "person_name": "自分", "is_user": True},
    {"text": "はい、元気です", "start": 5, "end": 7, "person_name": "田中", "is_user": False},
    {"text": "よかった", "start": 8, "end": 9, "person_name": "自分", "is_user": True},
]

TAGS_V2 = {
    "conversation_type": "ミーティング",
    "outcomes": ["意思決定あり", "TODO発生"],
    "domains": ["AI", "経営"],
    "importance": "中",
    "summary": "AI戦略について議論した",
    "decision_memo": "来期のAI投資を増やす",
}


def _sanitized_with_kc(question="AI投資をどう判断しましたか?",
                      answer="市場成長率と競合動向を比較し、来期の投資を30%増やすと決めた。",
                      with_summary=True):
    """Build a sanitized record fixture (ResponsesSanitizer-like shape)."""
    data = {
        "session_id": "2026-03-28/00-26-29",
        "risk": "low",
        "summary_input": None,
        "summary_output": "AI戦略を議論し、来期の投資を30%増やすと決定した。" if with_summary else None,
        "keypoints": [
            "市場成長率と競合動向を分析",
            "来期のAI投資を30%増やす",
            "四半期ごとに進捗をレビュー",
        ],
        "safe_conversation": [
            {"speaker": "user", "text": "AI投資の優先順位はどう決める?"},
            {"speaker": "assistant", "text": "ROIと市場成長率で評価すべきです。"},
            {"speaker": "user", "text": "では来期は30%増やそう。"},
        ],
        "knowledge_cards": [
            {
                "question": question,
                "answer": answer,
                "evidence": ["来期のAI投資を30%増やすと決定した。"],
                "confidence": "high",
            },
        ],
    }
    return data


# ---------------------------------------------------------------------------
# _parse_bullet_list
# ---------------------------------------------------------------------------


class TestParseBulletList:
    def test_dash(self):
        assert _parse_bullet_list("- A\n- B\n- C") == ["A", "B", "C"]

    def test_japanese_bullet(self):
        assert _parse_bullet_list("・A\n・B") == ["A", "B"]

    def test_empty(self):
        assert _parse_bullet_list("") == []

    def test_mixed(self):
        assert _parse_bullet_list("- A\n\n* B\n・C") == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# SanitizedIndex
# ---------------------------------------------------------------------------


class TestSanitizedIndex:
    def test_empty_dir(self, tmp_path):
        idx = SanitizedIndex(tmp_path)
        assert len(idx) == 0

    def test_responses_format(self, tmp_path):
        rec = {
            "session_id": "2026-03-28/00-26-29",
            "risk": "low",
            "summary": {"input": "会話を要約してください。", "output": "AIの議論。"},
            "keypoints": ["A", "B"],
            "safe_conversation": [
                {"speaker": "user", "text": "hi"},
                {"speaker": "assistant", "text": "hello"},
            ],
            "issues_detected": [],
            "knowledge_card": {
                "question": "AIの議論のポイントは何でしたか?",
                "answer": "AIの投資判断について話し合った。来期は増やす方針。",
                "evidence": ["来期は増やす。"],
                "confidence": "high",
            },
        }
        (tmp_path / "safe_training_20260328_000000.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        idx = SanitizedIndex(tmp_path)
        bucket = idx.get("2026-03-28/00-26-29")
        assert bucket is not None
        assert bucket["risk"] == "low"
        assert bucket["summary_output"] == "AIの議論。"
        assert bucket["keypoints"] == ["A", "B"]
        assert len(bucket["safe_conversation"]) == 2
        assert len(bucket["knowledge_cards"]) == 1
        assert bucket["knowledge_cards"][0]["confidence"] == "high"

    def test_multi_card_format(self, tmp_path):
        """SanitizedIndex reads the new knowledge_cards array shape."""
        rec = {
            "session_id": "2026-04-13/00-00-00",
            "risk": "low",
            "summary": {"input": "", "output": "AI 戦略議論"},
            "keypoints": ["K1"],
            "safe_conversation": [{"speaker": "user", "text": "x"}],
            "issues_detected": [],
            "knowledge_cards": [
                {"question": "Q1 とても長い具体的な質問?", "answer": "A1 十分な長さの回答です。",
                 "evidence": ["e1"], "confidence": "high"},
                {"question": "Q2 別の角度からの具体的な質問?", "answer": "A2 別の具体的な回答です。",
                 "evidence": ["e2"], "confidence": "medium"},
            ],
            "knowledge_card": None,
        }
        (tmp_path / "safe_training_20260413_000000.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        idx = SanitizedIndex(tmp_path)
        bucket = idx.get("2026-04-13/00-00-00")
        assert bucket is not None
        assert len(bucket["knowledge_cards"]) == 2
        confs = [c["confidence"] for c in bucket["knowledge_cards"]]
        assert "high" in confs
        assert "medium" in confs

    def test_multi_card_dedupes_within_session(self, tmp_path):
        """Duplicate questions within one record must be deduped."""
        rec = {
            "session_id": "sid",
            "risk": "low",
            "summary": {"input": "", "output": ""},
            "keypoints": [],
            "safe_conversation": [],
            "issues_detected": [],
            "knowledge_cards": [
                {"question": "同じ質問", "answer": "A1", "evidence": [], "confidence": "high"},
                {"question": "同じ質問", "answer": "A2", "evidence": [], "confidence": "medium"},
            ],
            "knowledge_card": None,
        }
        (tmp_path / "safe_training_x.jsonl").write_text(
            json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        idx = SanitizedIndex(tmp_path)
        bucket = idx.get("sid")
        assert len(bucket["knowledge_cards"]) == 1

    def test_training_sanitizer_format(self, tmp_path):
        records = [
            {"type": "summary", "session_id": "sid", "risk_level": "low",
             "input": "要約してください", "output": "要約テキスト。"},
            {"type": "keypoints", "session_id": "sid", "risk_level": "low",
             "input": "要点を抽出", "output": "- A\n- B\n- C"},
            {"type": "knowledge_card", "session_id": "sid", "risk_level": "low",
             "input": "この質問は?", "output": "この回答です。"},
        ]
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        (tmp_path / "safe_training_20260328_010000.jsonl").write_text(
            lines + "\n", encoding="utf-8"
        )
        idx = SanitizedIndex(tmp_path)
        bucket = idx.get("sid")
        assert bucket is not None
        assert bucket["summary_output"] == "要約テキスト。"
        assert bucket["keypoints"] == ["A", "B", "C"]
        assert len(bucket["knowledge_cards"]) == 1


# ---------------------------------------------------------------------------
# merge_turns (unchanged helper)
# ---------------------------------------------------------------------------


class TestMergeTurns:
    def test_merges_consecutive_same_speaker(self):
        turns = merge_turns(SEGMENTS_CONV)
        assert len(turns) == 3
        assert turns[0]["text"] == "こんにちは 元気ですか"


# ---------------------------------------------------------------------------
# comprehensive (still sourced from raw transcript)
# ---------------------------------------------------------------------------


class TestComprehensive:
    def test_basic(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        result = to_comprehensive("2026-03-28/00-26-29", transcript)
        assert result is not None
        assert result["session_id"] == "2026-03-28/00-26-29"
        assert len(result["turns"]) == 3

    def test_v2_tags_included(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        result = to_comprehensive("id", transcript)
        assert result["tags"]["conversation_type"] == "ミーティング"
        assert result["tags"]["domains"] == ["AI", "経営"]
        assert result["tags"]["decision_memo"] == "来期のAI投資を増やす"

    def test_empty_segments(self):
        assert to_comprehensive("id", _make_transcript([], TAGS_V2)) is None

    def test_sanitized_metadata_attached(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        result = to_comprehensive("id", transcript, sanitized=sanitized)
        assert result["sanitized"]["risk"] == "low"
        assert result["sanitized"]["knowledge_card_count"] == 1


# ---------------------------------------------------------------------------
# openai (KC-sourced)
# ---------------------------------------------------------------------------


class TestOpenAI:
    def test_requires_sanitized(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        assert to_openai("id", transcript, sanitized=None) == []

    def test_kc_task(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        out = to_openai("id", transcript, sanitized=sanitized)
        assert len(out) >= 1  # at least one KC task
        qa = out[0]
        assert "messages" in qa
        assert qa["messages"][0]["role"] == "system"
        assert "事業経営者" in qa["messages"][0]["content"]
        assert "AI" in qa["messages"][0]["content"]  # domain injected
        assert qa["messages"][1]["role"] == "user"
        assert "AI投資" in qa["messages"][1]["content"]
        assert qa["messages"][2]["role"] == "assistant"

    def test_summarization_task(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        # Extend conversation so summary task threshold is met
        sanitized["safe_conversation"] = [
            {"speaker": "user", "text": "AI投資の話をしよう。今期の予算配分について相談がある。"},
            {"speaker": "assistant", "text": "市場成長率と競合動向を分析して判断しましょう。"},
            {"speaker": "user", "text": "来期のAI投資を30%増やそう。ROIは四半期ごとにチェックする。"},
        ]
        out = to_openai("id", transcript, sanitized=sanitized)
        # Should include both KC task and summary task
        summaries = [r for r in out if r["messages"][0]["content"].startswith("次の会話を")]
        assert len(summaries) == 1
        assert "要約" in summaries[0]["messages"][0]["content"]

    def test_kc_too_short_skipped(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc(question="なぜ?", answer="はい。")
        out = to_openai("id", transcript, sanitized=sanitized)
        # KC filtered out (q<10 chars), possibly summary task still generated
        qa_tasks = [r for r in out if r["messages"][1]["content"] == "なぜ?"]
        assert len(qa_tasks) == 0


# ---------------------------------------------------------------------------
# conversation (ShareGPT, KC-sourced)
# ---------------------------------------------------------------------------


class TestConversation:
    def test_kc_task_sharegpt_shape(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        out = to_conversation("id", transcript, sanitized=sanitized)
        assert len(out) >= 1
        first = out[0]
        assert "conversations" in first
        assert "system" in first
        # Must NOT have top-level id/metadata (per LLaMA-Factory spec)
        assert "id" not in first
        assert "metadata" not in first
        convs = first["conversations"]
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"

    def test_empty_without_sanitized(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        assert to_conversation("id", transcript, sanitized=None) == []


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------


class TestKnowledge:
    def test_from_sanitized(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        result = to_knowledge("id", transcript, sanitized=sanitized)
        assert result is not None
        assert result["category"] == "AI, 経営"
        assert result["body"]
        assert len(result["qa_pairs"]) == 1
        assert result["qa_pairs"][0]["confidence"] == "high"

    def test_falls_back_to_tags_summary(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        result = to_knowledge("id", transcript, sanitized=None)
        assert result is not None
        assert result["body"].startswith("AI戦略")

    def test_no_data_returns_none(self):
        transcript = _make_transcript(SEGMENTS_CONV, {})
        result = to_knowledge("id", transcript, sanitized=None)
        assert result is None


# ---------------------------------------------------------------------------
# decision_log
# ---------------------------------------------------------------------------


class TestDecisionLog:
    def test_decision_memo_triggers(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        result = to_decision_log("id", transcript, sanitized=None)
        assert result is not None
        assert "来期のAI投資" in result["title"]
        assert "来期のAI投資" in result["decision"]

    def test_kc_high_confidence_triggers_without_memo(self):
        tags_no_memo = {**TAGS_V2, "decision_memo": "", "outcomes": []}
        transcript = _make_transcript(SEGMENTS_CONV, tags_no_memo)
        sanitized = _sanitized_with_kc()
        result = to_decision_log("id", transcript, sanitized=sanitized)
        assert result is not None
        assert len(result["knowledge_cards"]) == 1

    def test_no_signal_returns_none(self):
        tags = {"conversation_type": "雑談", "importance": "低"}
        transcript = _make_transcript(SEGMENTS_CONV, tags)
        assert to_decision_log("id", transcript, sanitized=None) is None


# ---------------------------------------------------------------------------
# sop
# ---------------------------------------------------------------------------


class TestSop:
    def test_todo_outcome_with_keypoints(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        result = to_sop("id", transcript, sanitized=sanitized)
        assert result is not None
        assert len(result["steps"]) == 3  # keypoints used as steps
        assert result["domain"] == "AI, 経営"

    def test_no_outcome_returns_none(self):
        tags = {**TAGS_V2, "outcomes": ["情報共有のみ"]}
        transcript = _make_transcript(SEGMENTS_CONV, tags)
        assert to_sop("id", transcript, sanitized=None) is None

    def test_legacy_action_items(self):
        tags = {"action_items": ["ステップ1", "ステップ2"], "business_domain": "technology"}
        transcript = _make_transcript(SEGMENTS_CONV, tags)
        result = to_sop("id", transcript, sanitized=None)
        assert result is not None
        assert result["steps"] == ["ステップ1", "ステップ2"]


# ---------------------------------------------------------------------------
# export_session
# ---------------------------------------------------------------------------


class TestExportSession:
    def test_returns_list(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        lines = export_session("openai", "id", transcript, sanitized=sanitized)
        assert isinstance(lines, list)
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert "messages" in parsed

    def test_empty_list_when_no_sanitized_for_chat_format(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        lines = export_session("openai", "id", transcript, sanitized=None)
        assert lines == []

    def test_comprehensive_works_without_sanitized(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        lines = export_session("comprehensive", "id", transcript)
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["session_id"] == "id"

    def test_invalid_format(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        with pytest.raises(ValueError, match="Unknown format"):
            export_session("invalid", "id", transcript)

    def test_all_formats_accept_sanitized_kwarg(self):
        transcript = _make_transcript(SEGMENTS_CONV, TAGS_V2)
        sanitized = _sanitized_with_kc()
        for fmt in ("comprehensive", "openai", "conversation",
                    "knowledge", "decision_log", "sop"):
            lines = export_session(fmt, "id", transcript, sanitized=sanitized)
            assert isinstance(lines, list)
            # Each line must parse as JSON
            for line in lines:
                json.loads(line)
