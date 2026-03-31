"""Unit tests for training_sanitizer.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training_sanitizer import TrainingSanitizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_transcript(segments, tags=None):
    """Build a minimal transcript dict."""
    session = {"start": "2026-03-31T10:00:00", "duration_seconds": 60.0}
    if tags:
        session["tags"] = tags
    return {"session": session, "segments": segments}


SEGMENTS = [
    {"text": "こんにちは", "start": 0, "end": 2, "speaker": "SPEAKER_00"},
    {"text": "今日は天気がいいですね", "start": 2, "end": 5, "speaker": "SPEAKER_01"},
    {"text": "そうですね", "start": 5, "end": 7, "speaker": "SPEAKER_00"},
]

TAGS = {
    "summary": "天気について雑談した",
    "topics": ["weather"],
    "activity": "casual_chat",
}


def _make_sanitizer():
    """Create a sanitizer with mocked OpenAI client."""
    with patch("training_sanitizer.TrainingSanitizer.__init__", return_value=None):
        s = TrainingSanitizer.__new__(TrainingSanitizer)
        s._enabled = True
        s._model = "gemini-2.0-flash"
        s._temperature = 0.3
        s._max_tokens = 4096
        s._use_max_completion_tokens = False
        s._client = MagicMock()
        return s


def _mock_response(content):
    """Create a mock API response."""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildInput:
    def test_basic(self):
        s = _make_sanitizer()
        transcript = _make_transcript(SEGMENTS, TAGS)
        text = s._build_input(transcript)
        assert "[セッション情報]" in text
        assert "天気について雑談した" in text
        assert "[発話]" in text
        assert "こんにちは" in text

    def test_no_tags(self):
        s = _make_sanitizer()
        transcript = _make_transcript(SEGMENTS)
        text = s._build_input(transcript)
        assert "[セッション情報]" not in text
        assert "こんにちは" in text

    def test_empty_segments(self):
        s = _make_sanitizer()
        transcript = _make_transcript([])
        text = s._build_input(transcript)
        assert text == ""

    def test_truncation(self):
        s = _make_sanitizer()
        long_segments = [
            {"text": "あ" * 200, "start": i, "end": i + 1}
            for i in range(100)
        ]
        transcript = _make_transcript(long_segments)
        text = s._build_input(transcript)
        # Should be truncated by char limit
        assert len(text) < 5000


class TestParseResponse:
    def test_valid_json(self):
        s = _make_sanitizer()
        result = s._parse_response('{"risk_level": "low", "records": []}')
        assert result == {"risk_level": "low", "records": []}

    def test_markdown_fences(self):
        s = _make_sanitizer()
        result = s._parse_response('```json\n{"risk_level": "low", "records": []}\n```')
        assert result is not None
        assert result["risk_level"] == "low"

    def test_invalid_json(self):
        s = _make_sanitizer()
        result = s._parse_response("not json at all")
        assert result is None

    def test_not_dict(self):
        s = _make_sanitizer()
        result = s._parse_response("[1, 2, 3]")
        assert result is None


class TestValidateResponse:
    def test_low_risk_with_records(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "low",
            "records": [
                {"type": "summary", "input": "要約してください", "output": "天気の雑談"},
                {"type": "keypoints", "input": "要点を抽出", "output": "- 天気がいい"},
            ],
        }
        result = s._validate_response(data)
        assert result is not None
        assert result["risk_level"] == "low"
        assert len(result["records"]) == 2

    def test_high_risk_forces_empty(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "high",
            "records": [
                {"type": "summary", "input": "x", "output": "y"},
            ],
        }
        result = s._validate_response(data)
        assert result is not None
        assert result["risk_level"] == "high"
        assert result["records"] == []

    def test_medium_risk(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "medium",
            "records": [
                {"type": "summary", "input": "要約", "output": "Aさんと会話した"},
            ],
        }
        result = s._validate_response(data)
        assert result is not None
        assert result["risk_level"] == "medium"
        assert len(result["records"]) == 1

    def test_invalid_risk_level(self):
        s = _make_sanitizer()
        data = {"risk_level": "unknown", "records": []}
        result = s._validate_response(data)
        assert result is None

    def test_safe_conversation_type(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "low",
            "records": [
                {
                    "type": "safe_conversation",
                    "messages": [
                        {"role": "user", "content": "今日の天気は？"},
                        {"role": "assistant", "content": "晴れです"},
                    ],
                },
            ],
        }
        result = s._validate_response(data)
        assert len(result["records"]) == 1
        assert result["records"][0]["type"] == "safe_conversation"
        assert len(result["records"][0]["messages"]) == 2

    def test_safe_conversation_invalid(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "low",
            "records": [
                {
                    "type": "safe_conversation",
                    "messages": [{"role": "user"}],  # missing content
                },
            ],
        }
        result = s._validate_response(data)
        assert result["records"] == []

    def test_missing_input_output(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "low",
            "records": [
                {"type": "summary", "input": "x"},  # missing output
                {"type": "keypoints", "output": "y"},  # missing input
                {"type": "summary", "input": "", "output": "y"},  # empty input
            ],
        }
        result = s._validate_response(data)
        assert result["records"] == []

    def test_invalid_record_type(self):
        s = _make_sanitizer()
        data = {
            "risk_level": "low",
            "records": [
                {"type": "invalid_type", "input": "x", "output": "y"},
            ],
        }
        result = s._validate_response(data)
        assert result["records"] == []


class TestSanitizeSession:
    def test_low_risk(self):
        s = _make_sanitizer()
        response_data = {
            "risk_level": "low",
            "records": [
                {"type": "summary", "input": "要約してください", "output": "天気の雑談"},
            ],
        }
        s._client.chat.completions.create.return_value = _mock_response(
            json.dumps(response_data)
        )
        transcript = _make_transcript(SEGMENTS, TAGS)
        result = s.sanitize_session(transcript)
        assert result is not None
        assert result["risk_level"] == "low"
        assert len(result["records"]) == 1

    def test_high_risk_drops(self):
        s = _make_sanitizer()
        response_data = {"risk_level": "high", "records": []}
        s._client.chat.completions.create.return_value = _mock_response(
            json.dumps(response_data)
        )
        transcript = _make_transcript(SEGMENTS, TAGS)
        result = s.sanitize_session(transcript)
        assert result is not None
        assert result["risk_level"] == "high"
        assert result["records"] == []

    def test_api_error(self):
        s = _make_sanitizer()
        s._client.chat.completions.create.side_effect = Exception("API error")
        transcript = _make_transcript(SEGMENTS, TAGS)
        result = s.sanitize_session(transcript)
        assert result is None

    def test_invalid_json_response(self):
        s = _make_sanitizer()
        s._client.chat.completions.create.return_value = _mock_response(
            "not valid json"
        )
        transcript = _make_transcript(SEGMENTS, TAGS)
        result = s.sanitize_session(transcript)
        assert result is None

    def test_empty_transcript(self):
        s = _make_sanitizer()
        transcript = _make_transcript([])
        result = s.sanitize_session(transcript)
        assert result is None

    def test_short_text(self):
        s = _make_sanitizer()
        transcript = _make_transcript([{"text": "あ", "start": 0, "end": 1}])
        result = s.sanitize_session(transcript)
        assert result is None

    def test_disabled(self):
        s = _make_sanitizer()
        s._enabled = False
        transcript = _make_transcript(SEGMENTS)
        result = s.sanitize_session(transcript)
        assert result is None

    def test_response_format_parameter(self):
        s = _make_sanitizer()
        response_data = {"risk_level": "low", "records": []}
        s._client.chat.completions.create.return_value = _mock_response(
            json.dumps(response_data)
        )
        transcript = _make_transcript(SEGMENTS, TAGS)
        s.sanitize_session(transcript)
        # Verify response_format=json_object was passed
        call_kwargs = s._client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}
