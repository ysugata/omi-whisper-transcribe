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
    tagger._prefilter_enabled = True
    tagger._prefilter_min_matches = 1
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


def test_external_audio_domains_forced_empty(mock_openai):
    """Domains should be forced empty when external_audio_ratio >= 0.8."""
    response = json.dumps({
        "conversation_type": "その他",
        "outcomes": ["情報共有のみ"],
        "domains": ["経営", "物販"],
        "entities": [],
        "summary": "TV audio about business",
        "decision_memo": None,
        "quality": "noisy",
        "importance": "低",
    })
    tagger = _make_tagger(mock_openai, response)
    # 9 unknown + 1 user = 90% unknown
    segments = [{"start": i, "end": i+1, "text": f"Segment {i}", "person_name": "unknown"} for i in range(9)]
    segments.append({"start": 9, "end": 10, "text": "My comment", "person_name": "Taro", "is_user": True})
    result = tagger.tag_session(segments)
    assert result is not None
    assert result["domains"] == []
    assert result["external_audio_ratio"] == 0.9


def test_external_audio_domains_kept_below_threshold(mock_openai):
    """Domains should be preserved when unknown ratio < 0.8."""
    response = json.dumps({
        "conversation_type": "商談",
        "outcomes": ["意思決定あり"],
        "domains": ["水中超音波"],
        "entities": [],
        "summary": "Business meeting",
        "decision_memo": None,
        "quality": "clean",
        "importance": "高",
    })
    tagger = _make_tagger(mock_openai, response)
    # 3 unknown + 7 user = 30% unknown
    segments = [{"start": i, "end": i+1, "text": f"Segment {i}", "person_name": "unknown"} for i in range(3)]
    segments += [{"start": i, "end": i+1, "text": f"My seg {i}", "person_name": "Taro", "is_user": True} for i in range(3, 10)]
    result = tagger.tag_session(segments)
    assert result is not None
    assert result["domains"] == ["水中超音波"]
    assert result["external_audio_ratio"] == 0.3


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


def test_build_system_prompt_with_domain_descriptions():
    """System prompt should include domain descriptions when available."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    taxonomy = dict(_DEFAULT_TAXONOMY)
    taxonomy["domain_descriptions"] = {
        "水中超音波": "水中超音波技術の研究開発・製品化",
        "物販": "物販ビジネスの仕入れ・販売・在庫管理",
        "経営": "特定事業に属さない全社横断の意思決定・財務・組織",
    }
    tagger._taxonomy = taxonomy
    prompt = tagger._build_system_prompt()
    # Descriptions should appear alongside domain names
    assert '"水中超音波": 水中超音波技術の研究開発・製品化' in prompt
    assert '"物販": 物販ビジネスの仕入れ・販売・在庫管理' in prompt
    assert '"経営": 特定事業に属さない全社横断の意思決定・財務・組織' in prompt
    # Domains without descriptions should appear name-only
    assert '"SNS集客"' in prompt


def test_build_system_prompt_with_keyword_anchors():
    """System prompt should include keyword anchors under domain descriptions."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    taxonomy = dict(_DEFAULT_TAXONOMY)
    taxonomy["domain_descriptions"] = {
        "水中超音波": "水中超音波技術の研究開発・製品化",
        "AI": "AI技術そのものが主題の研究・開発",
    }
    taxonomy["keyword_anchors"] = {
        "水中超音波": ["超音波", "トランスデューサ", "音響"],
        "AI": ["Claude", "ChatGPT", "GPT", "LLM"],
    }
    tagger._taxonomy = taxonomy
    prompt = tagger._build_system_prompt()
    # Keyword anchors should appear as indented lines after domain descriptions
    assert "関連キーワード: 超音波, トランスデューサ, 音響" in prompt
    assert "関連キーワード: Claude, ChatGPT, GPT, LLM" in prompt
    # Domains without anchors should NOT have keyword line
    assert "関連キーワード:" in prompt
    # Count occurrences - only 2 domains have anchors
    assert prompt.count("関連キーワード:") == 2


def test_build_system_prompt_without_keyword_anchors():
    """System prompt should work normally when keyword_anchors is absent."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    taxonomy = dict(_DEFAULT_TAXONOMY)
    taxonomy["domain_descriptions"] = {
        "水中超音波": "水中超音波技術の研究開発・製品化",
    }
    # No keyword_anchors key at all
    tagger._taxonomy = taxonomy
    prompt = tagger._build_system_prompt()
    assert '"水中超音波": 水中超音波技術の研究開発・製品化' in prompt
    assert "関連キーワード:" not in prompt


def test_build_system_prompt_without_domain_descriptions():
    """System prompt should use comma-separated list when no descriptions."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    prompt = tagger._build_system_prompt()
    # Without descriptions, should be comma-separated format
    assert '"水中超音波", "物販"' in prompt


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


def test_validate_and_save_tags_external_audio_domains_cleared(mock_openai):
    """validate_and_save_tags should clear domains for external audio sessions."""
    response_with_domains = json.dumps({
        "conversation_type": "その他",
        "outcomes": ["情報共有のみ"],
        "domains": ["経営"],
        "entities": [],
        "summary": "TV audio",
        "decision_memo": None,
        "quality": "noisy",
        "importance": "低",
    })
    tagger = _make_tagger(mock_openai, VALID_RESPONSE)
    segments = [{"start": i, "end": i+1, "text": f"Seg {i}", "person_name": "unknown"} for i in range(10)]
    result = tagger.validate_and_save_tags(response_with_domains, segments=segments)
    assert result is not None
    assert result["domains"] == []
    assert result["external_audio_ratio"] == 1.0


# ============================================================
# Domain prefilter tests
# ============================================================

TAXONOMY_WITH_ANCHORS = {
    **_DEFAULT_TAXONOMY,
    "domain_descriptions": {
        "水中超音波": "水中超音波技術の研究開発・製品化",
        "物販": "物販ビジネスの仕入れ・販売・在庫管理",
        "AI": "AI技術そのものが主題の研究・開発",
    },
    "keyword_anchors": {
        "水中超音波": ["超音波", "トランスデューサ", "音響"],
        "物販": ["Amazon", "仕入れ", "在庫", "FBA"],
        "AI": ["Claude", "ChatGPT", "LLM"],
        "個人": ["家族", "趣味", "健康"],
    },
}


def test_scan_domain_keywords_basic():
    """Basic keyword matching should find candidates."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 1

    result = tagger._scan_domain_keywords("超音波の実験を行いました。トランスデューサを使います。")
    assert len(result["candidates"]) >= 1
    cand_domains = [c["domain"] for c in result["candidates"]]
    assert "水中超音波" in cand_domains
    # Check hits count
    for c in result["candidates"]:
        if c["domain"] == "水中超音波":
            assert c["hits"] == 2
            assert c["total"] == 3


def test_scan_domain_keywords_no_matches():
    """Text with no matching keywords should return empty candidates."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 1

    result = tagger._scan_domain_keywords("今日はいい天気ですね。散歩に行きましょう。")
    assert len(result["candidates"]) == 0
    assert len(result["non_candidates"]) > 0


def test_scan_domain_keywords_case_insensitive():
    """Keyword matching should be case-insensitive."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 1

    result = tagger._scan_domain_keywords("amazonで在庫を確認しました。fbaの設定もしました。")
    cand_domains = [c["domain"] for c in result["candidates"]]
    assert "物販" in cand_domains
    for c in result["candidates"]:
        if c["domain"] == "物販":
            assert c["hits"] == 3  # amazon, 在庫, fba


def test_scan_domain_keywords_no_anchors():
    """Taxonomy without keyword_anchors should return empty candidates."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)  # no keyword_anchors
    tagger._prefilter_min_matches = 1

    result = tagger._scan_domain_keywords("超音波の実験")
    assert result["candidates"] == []
    assert len(result["non_candidates"]) == len(_DEFAULT_TAXONOMY["domains"])


def test_scan_domain_keywords_min_matches_filter():
    """min_matches=2 should filter out domains with only 1 hit."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 2

    # "超音波" = 1 hit for 水中超音波, should be excluded
    result = tagger._scan_domain_keywords("超音波について調べています。Claudeも使っています。")
    cand_domains = [c["domain"] for c in result["candidates"]]
    assert "水中超音波" not in cand_domains  # only 1 hit < min_matches=2
    assert "AI" not in cand_domains  # only 1 hit < min_matches=2


def test_format_prefilter_hint_with_candidates():
    """Hint should list candidates with hit counts."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 1

    scan_result = {
        "candidates": [
            {"domain": "AI", "hits": 3, "total": 3, "score": 1.0},
            {"domain": "物販", "hits": 2, "total": 4, "score": 0.5},
        ],
        "non_candidates": ["個人", "経営"],
    }
    hint = tagger._format_prefilter_hint(scan_result)
    assert "[キーワード走査結果]" in hint
    assert "AI(3hit)" in hint
    assert "物販(2hit)" in hint
    assert "個人" in hint
    assert "経営" in hint


def test_format_prefilter_hint_no_candidates():
    """No candidates should produce fallback message."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._taxonomy = TAXONOMY_WITH_ANCHORS
    tagger._prefilter_min_matches = 1

    scan_result = {"candidates": [], "non_candidates": ["水中超音波", "物販"]}
    hint = tagger._format_prefilter_hint(scan_result)
    assert "候補なし" in hint
    assert "全ドメインから判定してください" in hint


def test_prefilter_injected_in_tag_session(mock_openai):
    """tag_session should inject prefilter hint into user message."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE, TAXONOMY_WITH_ANCHORS)
    tagger._prefilter_enabled = True
    tagger._prefilter_min_matches = 1

    segments = [
        {"start": 0, "end": 5, "text": "超音波のトランスデューサを使って音響テストを実施します"},
        {"start": 5, "end": 10, "text": "周波数の調整が必要です"},
    ]
    result = tagger.tag_session(segments)
    assert result is not None

    # Check that the API was called with the prefilter hint in user message
    call_args = tagger._client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "[キーワード走査結果]" in user_msg
    assert "水中超音波" in user_msg


def test_prefilter_disabled(mock_openai):
    """Disabled prefilter should not inject hint."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE, TAXONOMY_WITH_ANCHORS)
    tagger._prefilter_enabled = False

    segments = [
        {"start": 0, "end": 5, "text": "超音波のトランスデューサを使って音響テストを実施します"},
        {"start": 5, "end": 10, "text": "周波数の調整が必要です"},
    ]
    result = tagger.tag_session(segments)
    assert result is not None

    call_args = tagger._client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "[キーワード走査結果]" not in user_msg


def test_prefilter_in_build_batch_input(mock_openai):
    """build_batch_input should also inject prefilter hint."""
    tagger = _make_tagger(mock_openai, VALID_RESPONSE, TAXONOMY_WITH_ANCHORS)
    tagger._prefilter_enabled = True
    tagger._prefilter_min_matches = 1

    segments = [
        {"start": 0, "end": 5, "text": "Amazonで仕入れた商品のFBA在庫を確認する"},
        {"start": 5, "end": 10, "text": "物販の売上レポートを作成します"},
    ]
    system, user = tagger.build_batch_input(segments)
    assert system != ""
    assert "[キーワード走査結果]" in user
    assert "物販" in user


def test_prefilter_system_prompt_guidance():
    """System prompt should include the prefilter guidance rule."""
    tagger = SessionTagger.__new__(SessionTagger)
    tagger._enabled = True
    tagger._user_tags_path = ""
    tagger._user_tags = []
    tagger._taxonomy_path = ""
    tagger._taxonomy = dict(_DEFAULT_TAXONOMY)
    prompt = tagger._build_system_prompt()
    assert "キーワード走査結果が提示された場合" in prompt
    assert "候補ドメインを優先的に検討" in prompt


def test_prefilter_ext_audio_skip(mock_openai):
    """External audio sessions (ext>=0.8) should still have domains cleared."""
    response = json.dumps({
        "conversation_type": "その他",
        "outcomes": ["情報共有のみ"],
        "domains": ["物販"],
        "entities": [],
        "summary": "TV audio with business keywords",
        "decision_memo": None,
        "quality": "noisy",
        "importance": "低",
    })
    tagger = _make_tagger(mock_openai, response, TAXONOMY_WITH_ANCHORS)
    tagger._prefilter_enabled = True
    tagger._prefilter_min_matches = 1

    # 9 unknown + 1 user = 90% unknown → ext_audio >= 0.8
    segments = [
        {"start": i, "end": i+1, "text": f"Amazonで仕入れ {i}", "person_name": "unknown"}
        for i in range(9)
    ]
    segments.append({"start": 9, "end": 10, "text": "My comment", "person_name": "Taro", "is_user": True})

    result = tagger.tag_session(segments)
    assert result is not None
    # Prefilter may suggest 物販, but ext_audio should override and clear domains
    assert result["domains"] == []
    assert result["external_audio_ratio"] == 0.9
