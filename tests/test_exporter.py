"""Unit tests for exporter.py."""

import json
import sys
from pathlib import Path

import pytest

# Make the viewer module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "omi-recordings-viewer"))

from exporter import merge_turns, to_comprehensive, to_openai, to_conversation, export_session


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

    def test_monologue(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_openai("id", transcript)
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

    def test_monologue(self):
        transcript = _make_transcript(SEGMENTS_MONOLOGUE, TAGS)
        result = to_conversation("id", transcript)
        assert result is not None
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"

    def test_metadata_included(self):
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        result = to_conversation("id", transcript)
        assert "activity" in result["metadata"]
        assert "quality" in result["metadata"]


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
        transcript = _make_transcript(SEGMENTS_CONVERSATION, TAGS)
        for fmt in ("comprehensive", "openai", "conversation"):
            line = export_session(fmt, "id", transcript)
            assert line is not None, f"Format {fmt} returned None"
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
