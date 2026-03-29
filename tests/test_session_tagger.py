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
    tagger._user_tags_path = ""
    tagger._user_tags = []
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
    tagger._user_tags_path = ""
    tagger._user_tags = []
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
    tagger._user_tags_path = ""
    tagger._user_tags = []
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


# ============================================================
# User Tags tests
# ============================================================

USER_TAGS = [
    {"id": "ut_abc123", "name": "仕事", "color": "#4CAF50", "description": "仕事関連"},
    {"id": "ut_def456", "name": "重要", "color": "#FF5722", "description": "重要な内容"},
]

RESPONSE_WITH_USER_TAGS = json.dumps({
    "topics": ["work"],
    "activity": "meeting",
    "intent": "planning",
    "mood": "neutral",
    "quality": "clean",
    "entities": [],
    "summary": "仕事の打ち合わせ",
    "user_tags": ["仕事", "重要"],
})


def _make_tagger_with_user_tags(mock_openai, response_content, user_tags):
    """Create a SessionTagger with user tags and mocked API response."""
    tagger = _make_tagger(mock_openai, response_content)
    tagger._user_tags = user_tags
    return tagger


def test_user_tags_resolved_to_ids(mock_openai):
    """user_tags names should be resolved to IDs."""
    tagger = _make_tagger_with_user_tags(mock_openai, RESPONSE_WITH_USER_TAGS, USER_TAGS)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert "user_tags" in result
    assert "ut_abc123" in result["user_tags"]
    assert "ut_def456" in result["user_tags"]


def test_user_tags_unknown_name_ignored(mock_openai):
    """Unknown tag names from LLM should be silently ignored."""
    response = json.dumps({
        "topics": ["work"],
        "activity": "meeting",
        "intent": "planning",
        "mood": "neutral",
        "quality": "clean",
        "entities": [],
        "summary": "テスト",
        "user_tags": ["仕事", "未定義タグ"],
    })
    tagger = _make_tagger_with_user_tags(mock_openai, response, USER_TAGS)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["user_tags"] == ["ut_abc123"]


def test_user_tags_empty_when_no_definitions(mock_openai):
    """user_tags should be empty when no user tag definitions exist."""
    tagger = _make_tagger(mock_openai, RESPONSE_WITH_USER_TAGS)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["user_tags"] == []


def test_build_system_prompt_with_user_tags():
    """System prompt should include user tag definitions."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = USER_TAGS
    prompt = tagger._build_system_prompt()
    assert "user_tags" in prompt
    assert "仕事" in prompt
    assert "重要" in prompt
    assert "仕事関連" in prompt


def test_build_system_prompt_without_user_tags():
    """System prompt should not include user_tags section when none defined."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    prompt = tagger._build_system_prompt()
    assert "user_tags" not in prompt


def test_reload_user_tags(tmp_path):
    """reload_user_tags() should re-read the file."""
    tags_file = tmp_path / "user_tags.json"
    tags_file.write_text("[]", encoding="utf-8")

    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = str(tags_file)
    tagger._user_tags = []

    # Write tags and reload
    tags_file.write_text(json.dumps(USER_TAGS), encoding="utf-8")
    tagger.reload_user_tags()
    assert len(tagger._user_tags) == 2
    assert tagger._user_tags[0]["name"] == "仕事"
