"""Tests for SessionTagger."""

import json
from unittest import mock

import pytest

from session_tagger import SessionTagger


@pytest.fixture
def mock_openai():
    """Patch OpenAI client to return controlled responses."""
    with mock.patch("session_tagger.OpenAI", create=True) as MockOpenAI:
        # Ignore the import inside __init__
        with mock.patch.dict("sys.modules", {"openai": mock.MagicMock()}):
            yield MockOpenAI


def _make_tagger(mock_openai, response_content):
    """Create a SessionTagger with mocked API response."""
    mock_client = mock.MagicMock()
    mock_openai.return_value = mock_client

    mock_response = mock.MagicMock()
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = response_content
    mock_client.chat.completions.create.return_value = mock_response

    # We need to mock the import inside __init__
    with mock.patch("builtins.__import__", side_effect=lambda name, *a, **kw: mock.MagicMock() if name == "openai" else __builtins__.__import__(name, *a, **kw)):
        pass

    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._model = "gemini-2.5-flash-lite"
    tagger._temperature = 0
    tagger._max_tokens = 512
    tagger._client = mock_client
    return tagger


VALID_RESPONSE = json.dumps({
    "topics": ["technology", "work"],
    "activity": "casual_chat",
    "intent": "planning",
    "mood": "positive",
    "quality": "clean",
    "entities": ["Docker", "GPU"],
    "summary": "DockerとGPU共有の設計について雑談",
})

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "DockerでGPU共有の設計について話しましょう"},
    {"start": 5.0, "end": 10.0, "text": "RTX3060で12GBのVRAMを共有する方法を検討中です"},
]


def test_tag_session_valid(mock_openai):
    """Valid API response should return normalized tags."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["topics"] == ["technology", "work"]
    assert result["activity"] == "casual_chat"
    assert result["intent"] == "planning"
    assert result["mood"] == "positive"
    assert result["quality"] == "clean"
    assert result["entities"] == ["Docker", "GPU"]
    assert "Docker" in result["summary"]
    assert "tagged_at" in result


def test_tag_session_markdown_fenced(mock_openai):
    """Response wrapped in markdown code fences should be parsed."""
    fenced = "```json\n" + VALID_RESPONSE + "\n```"
    tagger = _make_tagger(mock_openai, fenced)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["activity"] == "casual_chat"


def test_tag_session_invalid_enum(mock_openai):
    """Invalid enum values should fall back to defaults."""
    bad_response = json.dumps({
        "topics": ["tech"],
        "activity": "invalid_activity",
        "intent": "invalid_intent",
        "mood": "invalid_mood",
        "quality": "invalid_quality",
        "entities": [],
        "summary": "テスト",
    })
    tagger = _make_tagger(mock_openai, bad_response)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["activity"] == "other"
    assert result["intent"] == "other"
    assert result["mood"] == "neutral"
    assert result["quality"] == "clean"


def test_tag_session_empty_segments(mock_openai):
    """Empty segments should return None."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.tag_session([])
    assert result is None


def test_tag_session_short_text(mock_openai):
    """Very short text should return None."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.tag_session([{"start": 0, "end": 1, "text": "はい"}])
    assert result is None


def test_tag_session_api_error(mock_openai):
    """API error should return None (not raise)."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    tagger._client.chat.completions.create.side_effect = Exception("API timeout")
    result = tagger.tag_session(SEGMENTS)
    assert result is None


def test_tag_session_invalid_json(mock_openai):
    """Non-JSON response should return None."""
    tagger = _make_tagger(mock_openai, "This is not JSON at all")
    result = tagger.tag_session(SEGMENTS)
    assert result is None


def test_tag_session_disabled():
    """Disabled tagger should return None."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = False
    tagger._client = None
    result = tagger.tag_session(SEGMENTS)
    assert result is None


def test_tag_session_with_speaker_names(mock_openai):
    """Segments with person_name should include speaker labels in input."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    segments_with_names = [
        {"start": 0, "end": 5, "text": "Hello", "person_name": "Taro"},
        {"start": 5, "end": 10, "text": "Hi there", "speaker": "SPEAKER_01"},
    ]
    result = tagger.tag_session(segments_with_names)
    assert result is not None

    # Verify the input sent to the API includes speaker labels
    call_args = tagger._client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "[Taro]" in user_msg
    assert "[SPEAKER_01]" in user_msg


def test_validate_tags_truncates():
    """Validation should truncate long values."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tags = {
        "topics": ["a"] * 10,  # should be capped to 5
        "entities": ["e"] * 30,  # should be capped to 20
        "summary": "x" * 300,  # should be capped to 200
        "activity": "casual_chat",
        "intent": "planning",
        "mood": "positive",
        "quality": "clean",
    }
    result = tagger._validate_tags(tags)
    assert len(result["topics"]) == 5
    assert len(result["entities"]) == 20
    assert len(result["summary"]) == 200
