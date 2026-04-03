"""Tests for SessionTagger (V2 3-axis taxonomy)."""

import json
import os
from unittest import mock

import pytest

from session_tagger import SessionTagger, _load_taxonomy, _DEFAULT_TAXONOMY


@pytest.fixture
def mock_openai():
    """Patch OpenAI client to return controlled responses."""
    with mock.patch("session_tagger.OpenAI", create=True) as MockOpenAI:
        with mock.patch.dict("sys.modules", {"openai": mock.MagicMock()}):
            yield MockOpenAI


def _make_tagger(mock_openai, response_content, taxonomy=None):
    """Create a SessionTagger with mocked API response."""
    mock_client = mock.MagicMock()
    mock_openai.return_value = mock_client

    mock_response = mock.MagicMock()
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = response_content
    mock_client.chat.completions.create.return_value = mock_response

    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._model = "gemini-2.5-flash-lite"
    tagger._temperature = 0
    tagger._max_tokens = 768
    tagger._client = mock_client
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    tagger._taxonomy = taxonomy or dict(_DEFAULT_TAXONOMY)
    return tagger


VALID_RESPONSE = json.dumps({
    "conversation_type": "商談",
    "outcomes": ["意思決定あり", "TODO発生"],
    "domains": ["水中超音波"],
    "entities": ["Docker", "GPU"],
    "summary": "DockerとGPU共有の設計について商談",
    "decision_memo": "GPU共有方式をDocker使用に決定",
    "quality": "clean",
    "importance": "高",
})

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "DockerでGPU共有の設計について話しましょう"},
    {"start": 5.0, "end": 10.0, "text": "RTX3060で12GBのVRAMを共有する方法を検討中です"},
]


# ============================================================
# Core tagging tests
# ============================================================

def test_tag_session_valid(mock_openai):
    """Valid API response should return normalized V2 tags."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["conversation_type"] == "商談"
    assert result["outcomes"] == ["意思決定あり", "TODO発生"]
    assert result["domains"] == ["水中超音波"]
    assert result["entities"] == ["Docker", "GPU"]
    assert "Docker" in result["summary"]
    assert result["decision_memo"] == "GPU共有方式をDocker使用に決定"
    assert result["quality"] == "clean"
    assert result["importance"] == "高"
    assert result["tag_version"] == 2
    assert "tagged_at" in result


def test_tag_session_markdown_fenced(mock_openai):
    """Response wrapped in markdown code fences should be parsed."""
    fenced = "```json\n" + VALID_RESPONSE + "\n```"
    tagger = _make_tagger(mock_openai, fenced)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["conversation_type"] == "商談"


def test_tag_session_invalid_values(mock_openai):
    """Invalid taxonomy values should fall back to defaults."""
    bad_response = json.dumps({
        "conversation_type": "存在しない種類",
        "outcomes": ["存在しない成果物"],
        "domains": ["存在しない領域"],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "invalid_quality",
        "importance": "存在しない重要度",
    })
    tagger = _make_tagger(mock_openai, bad_response)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["conversation_type"] == "その他"
    assert result["outcomes"] == ["なし"]
    assert result["domains"] == []
    assert result["quality"] == "clean"
    assert result["importance"] == "低"


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
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    result = tagger.tag_session(SEGMENTS)
    assert result is None


def test_tag_session_with_speaker_names(mock_openai):
    """Segments with person_name should include speaker labels in input."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    segments_with_names = [
        {"start": 0, "end": 5, "text": "Hello", "person_name": "Taro", "is_user": True},
        {"start": 5, "end": 10, "text": "Hi there", "person_name": "unknown"},
    ]
    result = tagger.tag_session(segments_with_names)
    assert result is not None

    call_args = tagger._client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "[Taro(自分)]" in user_msg
    assert "[unknown]" in user_msg
    assert "[話者情報]" in user_msg
    # external_audio_ratio should be set
    assert result.get("external_audio_ratio") == 0.5


def test_speaker_stats_all_unknown(mock_openai):
    """All unknown speakers should yield external_audio_ratio=1.0."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    segments = [
        {"start": 0, "end": 5, "text": "Hello", "person_name": "unknown"},
        {"start": 5, "end": 10, "text": "World", "person_name": "unknown"},
    ]
    result = tagger.tag_session(segments)
    assert result["external_audio_ratio"] == 1.0


def test_speaker_stats_no_voice_tags(mock_openai):
    """No voice tags should yield external_audio_ratio=None."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    segments = [
        {"start": 0, "end": 5, "text": "Hello"},
        {"start": 5, "end": 10, "text": "World"},
    ]
    result = tagger.tag_session(segments)
    assert result["external_audio_ratio"] is None


def test_input_with_speaker_preamble():
    """Speaker preamble should warn about external audio when ratio >= 80%."""
    from session_tagger import SessionTagger
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = _DEFAULT_TAXONOMY
    segments = [
        {"text": "A", "person_name": "unknown"},
        {"text": "B", "person_name": "unknown"},
        {"text": "C", "person_name": "unknown"},
        {"text": "D", "person_name": "unknown"},
        {"text": "E", "person_name": "Taro", "is_user": True},
    ]
    text = tagger._build_input(segments)
    assert "unknown=4" in text
    assert "外部音声" in text

    # Below 80% threshold: no warning
    segments2 = [
        {"text": "A", "person_name": "unknown"},
        {"text": "B", "person_name": "Taro", "is_user": True},
        {"text": "C", "person_name": "Taro", "is_user": True},
    ]
    text2 = tagger._build_input(segments2)
    assert "外部音声" not in text2


# ============================================================
# 5-tag limit tests
# ============================================================

def test_five_tag_limit_trims_domains(mock_openai):
    """When total tags > 5, domains should be trimmed first."""
    response = json.dumps({
        "conversation_type": "商談",
        "outcomes": ["意思決定あり", "TODO発生", "方針転換"],
        "domains": ["水中超音波", "物販", "経営"],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "中",
    })
    tagger = _make_tagger(mock_openai, response)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    # 1 (conv_type) + 3 (outcomes) + domains trimmed to 1 = 5
    total = 1 + len(result["outcomes"]) + len(result["domains"])
    assert total <= 5
    assert len(result["outcomes"]) == 3  # outcomes preserved


def test_five_tag_limit_max_outcomes(mock_openai):
    """outcomes should be capped at 4."""
    response = json.dumps({
        "conversation_type": "商談",
        "outcomes": ["意思決定あり", "TODO発生", "方針転換", "情報共有のみ", "アイデア創出"],
        "domains": [],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "中",
    })
    tagger = _make_tagger(mock_openai, response)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert len(result["outcomes"]) <= 4


def test_five_tag_limit_zero_domains(mock_openai):
    """0 domains should be OK."""
    response = json.dumps({
        "conversation_type": "雑談・移動中メモ",
        "outcomes": ["なし"],
        "domains": [],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "低",
    })
    tagger = _make_tagger(mock_openai, response)
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["conversation_type"] == "雑談・移動中メモ"
    assert result["outcomes"] == ["なし"]
    assert result["domains"] == []
    total = 1 + len(result["outcomes"]) + len(result["domains"])
    assert total == 2


# ============================================================
# Taxonomy loading tests
# ============================================================

def test_load_taxonomy_from_file(tmp_path):
    """Taxonomy should load from a valid JSON file."""
    taxonomy = {
        "version": 1,
        "conversation_type": ["テスト種類"],
        "outcomes": ["テスト成果"],
        "domains": ["テスト領域"],
        "importance": ["高", "低"],
    }
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(taxonomy, ensure_ascii=False), encoding="utf-8")

    result = _load_taxonomy(str(path))
    assert result["conversation_type"] == ["テスト種類"]
    assert result["outcomes"] == ["テスト成果"]


def test_load_taxonomy_fallback_on_missing_file():
    """Missing file should return default taxonomy."""
    result = _load_taxonomy("/nonexistent/path.json")
    assert result["conversation_type"] == _DEFAULT_TAXONOMY["conversation_type"]


def test_load_taxonomy_fallback_on_invalid_json(tmp_path):
    """Invalid JSON should return default taxonomy."""
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    result = _load_taxonomy(str(path))
    assert result["conversation_type"] == _DEFAULT_TAXONOMY["conversation_type"]


def test_load_taxonomy_fallback_on_empty_path():
    """Empty path should return default taxonomy."""
    result = _load_taxonomy("")
    assert result["conversation_type"] == _DEFAULT_TAXONOMY["conversation_type"]


def test_reload_taxonomy(tmp_path):
    """reload_taxonomy() should re-read the file."""
    path = tmp_path / "taxonomy.json"
    taxonomy_v1 = dict(_DEFAULT_TAXONOMY)
    path.write_text(json.dumps(taxonomy_v1, ensure_ascii=False), encoding="utf-8")

    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._taxonomy_path = str(path)
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    tagger._user_tags_path = ""
    tagger._user_tags = []

    # Write updated taxonomy
    taxonomy_v2 = dict(_DEFAULT_TAXONOMY)
    taxonomy_v2["conversation_type"] = ["新種類1", "新種類2"]
    path.write_text(json.dumps(taxonomy_v2, ensure_ascii=False), encoding="utf-8")

    tagger.reload_taxonomy()
    assert tagger._taxonomy["conversation_type"] == ["新種類1", "新種類2"]


def test_custom_taxonomy_validation(mock_openai, tmp_path):
    """Custom taxonomy values should be used for validation."""
    custom_taxonomy = {
        "version": 1,
        "conversation_type": ["カスタム種類", "その他"],
        "outcomes": ["カスタム成果", "なし"],
        "domains": ["カスタム領域"],
        "importance": ["高", "中", "低"],
    }
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(custom_taxonomy, ensure_ascii=False), encoding="utf-8")

    response = json.dumps({
        "conversation_type": "カスタム種類",
        "outcomes": ["カスタム成果"],
        "domains": ["カスタム領域"],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "高",
    })
    tagger = _make_tagger(mock_openai, response, _load_taxonomy(str(path)))
    result = tagger.tag_session(SEGMENTS)
    assert result is not None
    assert result["conversation_type"] == "カスタム種類"
    assert result["outcomes"] == ["カスタム成果"]
    assert result["domains"] == ["カスタム領域"]


# ============================================================
# Validation edge cases
# ============================================================

def test_validate_truncates_values(mock_openai):
    """Validation should truncate long values."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    tags = {
        "conversation_type": "商談",
        "outcomes": ["意思決定あり"],
        "domains": [],
        "entities": ["e"] * 30,  # should be capped to 20
        "summary": "x" * 300,  # should be capped to 200
        "decision_memo": "y" * 100,  # should be capped to 80
        "quality": "clean",
        "importance": "高",
    }
    result = tagger._validate_tags(tags)
    assert len(result["entities"]) == 20
    assert len(result["summary"]) == 200
    assert len(result["decision_memo"]) == 80


def test_validate_decision_memo_null_when_empty(mock_openai):
    """decision_memo should be None when empty/null."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    tags = {
        "conversation_type": "商談",
        "outcomes": ["情報共有のみ"],
        "domains": [],
        "entities": [],
        "summary": "テスト",
        "decision_memo": "",
        "quality": "clean",
        "importance": "中",
    }
    result = tagger._validate_tags(tags)
    assert result["decision_memo"] is None


def test_domains_capped_at_3(mock_openai):
    """domains should be capped at 3."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    tags = {
        "conversation_type": "商談",
        "outcomes": ["なし"],
        "domains": ["水中超音波", "物販", "SNS集客", "観光業", "投資"],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "中",
    }
    result = tagger._validate_tags(tags)
    assert len(result["domains"]) <= 3


# ============================================================
# User Tags tests
# ============================================================

USER_TAGS = [
    {"id": "ut_abc123", "name": "仕事", "color": "#4CAF50", "description": "仕事関連"},
    {"id": "ut_def456", "name": "重要", "color": "#FF5722", "description": "重要な内容"},
]

RESPONSE_WITH_USER_TAGS = json.dumps({
    "conversation_type": "社内ミーティング",
    "outcomes": ["情報共有のみ"],
    "domains": [],
    "entities": [],
    "summary": "仕事の打ち合わせ",
    "decision_memo": None,
    "quality": "clean",
    "importance": "中",
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
        "conversation_type": "社内ミーティング",
        "outcomes": ["情報共有のみ"],
        "domains": [],
        "entities": [],
        "summary": "テスト",
        "decision_memo": None,
        "quality": "clean",
        "importance": "中",
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


def test_build_system_prompt_includes_taxonomy():
    """System prompt should include taxonomy values."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    prompt = tagger._build_system_prompt()
    assert "商談" in prompt
    assert "意思決定あり" in prompt
    assert "水中超音波" in prompt
    assert "conversation_type" in prompt


def test_build_system_prompt_with_user_tags():
    """System prompt should include user tag definitions."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = USER_TAGS
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    prompt = tagger._build_system_prompt()
    assert "user_tags" in prompt
    assert "仕事" in prompt
    assert "仕事関連" in prompt


def test_build_system_prompt_without_user_tags():
    """System prompt should not include user_tags section when none defined."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
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
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)

    tags_file.write_text(json.dumps(USER_TAGS), encoding="utf-8")
    tagger.reload_user_tags()
    assert len(tagger._user_tags) == 2
    assert tagger._user_tags[0]["name"] == "仕事"


# ============================================================
# build_batch_input / validate_and_save_tags
# ============================================================

def test_build_batch_input(mock_openai):
    """build_batch_input should return system prompt and user text."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    system, user = tagger.build_batch_input(SEGMENTS)
    assert system != ""
    assert user != ""
    assert "Docker" in user


def test_build_batch_input_short_text(mock_openai):
    """Too short text should return empty strings."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    system, user = tagger.build_batch_input([{"text": "あ"}])
    assert system == ""
    assert user == ""


def test_validate_and_save_tags(mock_openai):
    """validate_and_save_tags should parse and validate raw JSON."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.validate_and_save_tags(VALID_RESPONSE)
    assert result is not None
    assert result["conversation_type"] == "商談"
    assert result["tag_version"] == 2
    assert "tagged_at" in result


def test_validate_and_save_tags_invalid_json(mock_openai):
    """Invalid JSON should return None."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    result = tagger.validate_and_save_tags("not json")
    assert result is None
