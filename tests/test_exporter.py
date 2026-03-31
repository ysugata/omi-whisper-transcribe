"""Unit tests for exporter.py."""

import json
import sys
from pathlib import Path

import pytest

# Make the viewer module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omi-recordings-viewer"))

from exporter import (
    merge_turns, to_comprehensive, to_openai, to_conversation,
    to_knowledge, to_decision_log, to_sop, export_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transcript(segments, tags=None):
    """Build a minimal transcript dict."""
    session = {"start": "2026-03-28T00:26:29", "duration_seconds": 120.0, "language": "ja"}
    if tags:
        session["tags"] = tags
    return {"session": session, "segments": segments}


SEGMENTS_CONVERSATION = [
    {"text": "こんにちは", "start": 0, "end": 2, "person_name": "自分", "is_user": True, "speaker": "SPEAKER_00"},
    {"text": "元気ですか", "start": 2, "end": 4, "person_name": "自分", "is_user": True, "speaker": "SPEAKER_00"},
    {"text": "はい、元気です", "start": 5, "end": 7, "person_name": "田中", "is_user": False, "speaker": "SPEAKER_01"},
    {"text": "よかった", "start": 8, "end": 9, "person_name": "自分", "is_user": True, "speaker": "SPEAKER_00"},
]

SEGMENTS_MONOLOGUE = [
    {"text": "AIの未来について", "start": 0, "end": 3, "person_name": "自分", "is_user": True},
    {"text": "考えてみた", "start": 3, "end": 5, "person_name": "自分", "is_user": True},
]

TAGS = {
    "topics": ["programming", "AI"],
    "activity": "brainstorming",
    "mood": "mixed",
    "quality": "clean",
    "entities": ["Docker"],
    "summary": "AIについて議論した",
}

TAGS_EXTENDED = {
    **TAGS,
    "source_type": "meeting",
    "business_domain": "technology",
    "importance": "high",
    "confidentiality": "internal",
    "decision_required": True,
    "decision_summary": "新しいフレームワークを採用する",
    "reasoning_summary": "開発効率が上がるため",
    "action_items": ["技術検証を行う", "チームに共有する"],
    "outcome": "pending",
    "next_action": "来週までにPoCを完成させる",
}


# ---------------------------------------------------------------------------
# merge_turns tests
# ---------------------------------------------------------------------------

class TestMergeTurns:
    def test_empty(self):
        assert merge_turns([]) == []

    def test_merges_consecutive_same_speaker(self):
        turns = merge_turns(SEGMENTS_CONVERSATION)
        assert len(turns) == 3
        assert turns[0]["text"] == "こんにちは 元気ですか"
        assert turns[0]["speaker"] == "自分"
        assert turns[0]["is_user"] is True
        assert turns[0]["start"] == 0
        assert turns[0]["end"] == 4
        assert turns[1]["text"] == "はい、元気です"
        assert turns[1]["speaker"] == "田中"
        assert turns[2]["text"] == "よかった"

    def test_skips_empty_text(self):
        segs = [
            {"text": "hello", "start": 0, "end": 1, "person_name": "A"},
            {"text": "  ", "start": 1, "end": 2, "person_name": "A"},
            {"text": "world", "start": 2, "end": 3, "person_name": "A"},
        ]
        turns = merge_turns(segs)
        assert len(turns) == 1
        assert turns[0]["text"] == "hello world"

    def test_no_merge_different_speakers(self):
        segs = [
            {"text": "A says", "start": 0, "end": 1, "person_name": "A"},
            {"text": "B says", "start": 1, "end": 2, "person_name": "B"},
        ]
        turns = merge_turns(segs)
        assert len(turns) == 2


# ---------------------------------------------------------------------------
# comprehensive format tests
# ---------------------------------------------------------------------------

class TestComprehensive:
    def test_basic(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_comprehensive("2026-03-28/00-26-29", transcript)
        assert result is not None
        assert result["session_id"] == "2026-03-28/00-26-29"
        assert len(result["turns"]) == 3
        assert result["tags"]["activity"] == "brainstorming"

    def test_user_only(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_comprehensive("id", transcript, speaker_mode="user_only")
        assert result is not None
        assert all(t["is_user"] for t in result["turns"])

    def test_empty_segments(self):
        transcript = _make_transcript([], TAGS)
        assert to_comprehensive("id", transcript) is None

    def test_extended_tags_included(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS_EXTENDED)
        result = to_comprehensive("id", transcript)
        assert result is not None
        assert result["tags"]["source_type"] == "meeting"
        assert result["tags"]["business_domain"] == "technology"
        assert result["tags"]["importance"] == "high"
        assert result["tags"]["decision_required"] is True
        assert result["tags"]["decision_summary"] == "新しいフレームワークを採用する"
        assert result["tags"]["action_items"] == ["技術検証を行う", "チームに共有する"]
        assert result["tags"]["outcome"] == "pending"
        assert result["tags"]["next_action"] == "来週までにPoCを完成させる"


# ---------------------------------------------------------------------------
# openai format tests
# ---------------------------------------------------------------------------

class TestOpenAI:
    def test_conversation(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_openai("id", transcript)
        assert result is not None
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert "事業経営者" in messages[0]["content"]
        # user=田中, assistant=自分
        roles = [m["role"] for m in messages[1:]]
        assert "assistant" in roles
        assert "user" in roles

    def test_monologue_skipped_by_default(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_openai("id", transcript)
        assert result is None

    def test_monologue_included_when_requested(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_openai("id", transcript, include_monologue=True)
        assert result is not None
        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert "独り言" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "programming" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert "AIの未来について" in messages[2]["content"]

    def test_no_user_segments_returns_none(self):
        segs = [
            {"text": "他者の発話", "start": 0, "end": 2, "person_name": "他者", "is_user": False},
        ]
        transcript = _make_transcript(segs, TAGS)
        # user_only mode → no assistant turns → None
        result = to_openai("id", transcript, speaker_mode="user_only")
        assert result is None

    def test_user_only_mode(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_openai("id", transcript, speaker_mode="user_only")
        assert result is not None
        non_system = [m for m in result["messages"] if m["role"] != "system"]
        assert all(m["role"] == "assistant" for m in non_system)


# ---------------------------------------------------------------------------
# conversation (ShareGPT) format tests
# ---------------------------------------------------------------------------

class TestConversation:
    def test_basic(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_conversation("2026-03-28/00-26-29", transcript)
        assert result is not None
        assert result["id"] == "2026-03-28/00-26-29"
        assert "事業経営者" in result["system"]
        convs = result["conversations"]
        assert any(c["from"] == "gpt" for c in convs)
        assert any(c["from"] == "human" for c in convs)

    def test_monologue_skipped_by_default(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_conversation("id", transcript)
        assert result is None

    def test_monologue_included_when_requested(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_conversation("id", transcript, include_monologue=True)
        assert result is not None
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_metadata_included(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_conversation("id", transcript)
        assert "activity" in result["metadata"]
        assert "quality" in result["metadata"]


# ---------------------------------------------------------------------------
# knowledge format tests
# ---------------------------------------------------------------------------

class TestKnowledge:
    def test_basic(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS_EXTENDED)
        result = to_knowledge("id", transcript)
        assert result is not None
        assert result["id"] == "id"
        assert result["category"] == "technology"
        assert result["importance"] == "high"
        assert "topics" in result
        assert result["body"]  # non-empty body

    def test_no_summary_returns_none(self):
        tags_no_summary = {**TAGS_EXTENDED, "summary": ""}
        transcript = _make_transcript(SEGMENTS_CONVERSATION, tags_no_summary)
        result = to_knowledge("id", transcript)
        assert result is None

    def test_no_tags_returns_none(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION)
        result = to_knowledge("id", transcript)
        assert result is None


# ---------------------------------------------------------------------------
# decision_log format tests
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_with_decision(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS_EXTENDED)
        result = to_decision_log("id", transcript)
        assert result is not None
        assert result["decision_id"] == "id"
        assert result["title"] == "新しいフレームワークを採用する"
        assert result["decision"] == "新しいフレームワークを採用する"
        assert result["reason"] == "開発効率が上がるため"
        assert len(result["action_items"]) == 2
        assert result["outcome"] == "pending"
        assert result["next_action"] == "来週までにPoCを完成させる"

    def test_no_decision_returns_none(self):
        tags_no_decision = {**TAGS_EXTENDED, "decision_required": False}
        transcript = _make_transcript(SEGMENTS_CONVERSATION, tags_no_decision)
        result = to_decision_log("id", transcript)
        assert result is None

    def test_basic_tags_returns_none(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_decision_log("id", transcript)
        assert result is None


# ---------------------------------------------------------------------------
# sop format tests
# ---------------------------------------------------------------------------

class TestSop:
    def test_with_action_items(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS_EXTENDED)
        result = to_sop("id", transcript)
        assert result is not None
        assert result["id"] == "id"
        assert len(result["steps"]) == 2
        assert result["domain"] == "technology"

    def test_no_action_items_returns_none(self):
        tags_no_actions = {**TAGS_EXTENDED, "action_items": []}
        transcript = _make_transcript(SEGMENTS_CONVERSATION, tags_no_actions)
        result = to_sop("id", transcript)
        assert result is None

    def test_casual_chat_returns_none(self):
        tags_casual = {**TAGS_EXTENDED, "activity": "casual_chat"}
        transcript = _make_transcript(SEGMENTS_CONVERSATION, tags_casual)
        result = to_sop("id", transcript)
        assert result is None

    def test_casual_chat_source_type_returns_none(self):
        tags_casual = {**TAGS_EXTENDED, "source_type": "casual_chat"}
        transcript = _make_transcript(SEGMENTS_CONVERSATION, tags_casual)
        result = to_sop("id", transcript)
        assert result is None


# ---------------------------------------------------------------------------
# monologue skip tests
# ---------------------------------------------------------------------------

class TestMonologueSkip:
    def test_openai_skips_monologue_by_default(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        assert to_openai("id", transcript) is None

    def test_openai_includes_monologue_when_requested(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_openai("id", transcript, include_monologue=True)
        assert result is not None

    def test_conversation_skips_monologue_by_default(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        assert to_conversation("id", transcript) is None

    def test_conversation_includes_monologue_when_requested(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_conversation("id", transcript, include_monologue=True)
        assert result is not None

    def test_export_session_respects_include_monologue(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        # Default (include_monologue=False) → openai skips monologue
        assert export_session("openai", "id", transcript) is None
        # Explicit include → produces output
        line = export_session("openai", "id", transcript, include_monologue=True)
        assert line is not None

    def test_comprehensive_includes_monologue_by_default(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        # comprehensive has include_monologue=True by default
        line = export_session("comprehensive", "id", transcript, include_monologue=True)
        assert line is not None


# ---------------------------------------------------------------------------
# export_session tests
# ---------------------------------------------------------------------------

class TestExportSession:
    def test_returns_jsonl_string(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        line = export_session("openai", "id", transcript)
        assert line is not None
        parsed = json.loads(line)
        assert "messages" in parsed

    def test_invalid_format(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        with pytest.raises(ValueError, match="Unknown format"):
            export_session("invalid", "id", transcript)

    def test_all_formats_produce_valid_json(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS_EXTENDED)
        for fmt in ("comprehensive", "openai", "conversation",
                     "knowledge", "decision_log", "sop"):
            line = export_session(fmt, "id", transcript, include_monologue=True)
            assert line is not None, f"Format {fmt} returned None"
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
