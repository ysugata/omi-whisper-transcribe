"""Unit tests for responses_sanitizer.py + sanitize_schema.py.

テスト方針:
- OpenAI API はモックする (実 API 呼び出しなし)
- スキーマ検証・品質ゲートはユニットテスト
- fixture 2件 (low risk / high risk) を使用
- 2-step pipeline: Step 1 (sanitize) + Step 2 (knowledge card extraction)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from responses_sanitizer import ResponsesSanitizer, _RetryableError, _FatalError
from sanitize_schema import (
    KC_EXTRACTION_SCHEMA,
    KC_MIN_KEYPOINTS,
    SANITIZE_SCHEMA,
    run_quality_gates,
    should_keep_knowledge_card,
    validate_knowledge_card,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Fixture data: Step 1 LLM response (no knowledge_card)
# ---------------------------------------------------------------------------

FIXTURE_LOW_RISK = {
    "risk": "low",
    "summary": {
        "input": "以下の会話内容を要約してください。",
        "output": "ミーティングで来月のプロジェクト計画について話し合い、新機能の優先順位とテストスケジュールを確認した。バックエンドの進捗報告も行われた。",
    },
    "keypoints": [
        "来月のプロジェクト計画を協議",
        "新機能の優先順位を決定",
        "テストスケジュールを確認",
        "バックエンドの進捗報告があった",
    ],
    "safe_conversation": [
        {"speaker": "user", "text": "今日のミーティングでは何を話しましたか？"},
        {"speaker": "assistant", "text": "来月のプロジェクト計画について話し合いました。新機能の優先順位を決めて、テストスケジュールも確認しました。"},
        {"speaker": "user", "text": "バックエンドの進捗はどうでしたか？"},
        {"speaker": "assistant", "text": "Aさんから進捗報告がありました。順調に進んでいるとのことです。"},
    ],
    "issues_detected": ["asr_noise"],
}

FIXTURE_HIGH_RISK = {
    "risk": "high",
    "summary": {
        "input": "以下の会話内容を要約してください。",
        "output": "採用候補者に関する詳細な個人情報を含む会話が行われました。複数の個人識別情報が含まれているため、高リスクと判定しました。",
    },
    "keypoints": [
        "採用計画について議論があった",
    ],
    "safe_conversation": [
        {"speaker": "user", "text": "来年度の採用計画はどうなっていますか？"},
        {"speaker": "assistant", "text": "増員を予定しています。詳細は非開示です。"},
        {"speaker": "user", "text": "了解です。"},
    ],
    "issues_detected": ["pii_redacted", "asr_noise"],
}

# Step 2: KC extraction response fixtures
FIXTURE_KC_EXTRACTABLE = {
    "extractable": True,
    "question": "プロジェクト計画ミーティングではどのような決定がされましたか？",
    "answer": "新機能の優先順位を決定し、テストスケジュールを確認した。バックエンドは順調に進んでいる。",
    "evidence": [
        "来月のプロジェクト計画について話し合いました。新機能の優先順位を決めて、テストスケジュールも確認しました。",
        "Aさんから進捗報告がありました。順調に進んでいるとのことです。",
    ],
    "confidence": "high",
}

FIXTURE_KC_NOT_EXTRACTABLE = {
    "extractable": False,
    "question": "",
    "answer": "",
    "evidence": [],
    "confidence": "none",
}

FIXTURE_KC_WEAK = {
    "extractable": True,
    "question": "何を話した？",
    "answer": "色々。",
    "evidence": [],
    "confidence": "low",
}


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_response(data: dict, status: str = "completed"):
    """Responses API のレスポンスをモックする."""
    mock = MagicMock()
    mock.status = status
    mock.output_text = json.dumps(data, ensure_ascii=False)
    mock.incomplete_details = None
    mock.usage = MagicMock()
    mock.usage.input_tokens = 500
    mock.usage.output_tokens = 300
    mock.usage.total_tokens = 800
    mock.usage.output_tokens_details = None
    return mock


def _make_sanitizer(model: str = "gpt-5-nano") -> ResponsesSanitizer:
    """モック OpenAI クライアント付き ResponsesSanitizer を作成."""
    with patch("responses_sanitizer.ResponsesSanitizer.__init__", return_value=None):
        s = ResponsesSanitizer.__new__(ResponsesSanitizer)
        s._enabled = True
        s._model = model
        s._reasoning_effort = "low"
        s._max_output_tokens = 1800
        s._timeout_sec = 60.0
        s._max_retries = 3
        s._client = MagicMock()
        return s


# ---------------------------------------------------------------------------
# Test: sanitize_schema.py — Step 1 Schema
# ---------------------------------------------------------------------------

class TestSanitizeSchema:
    """SANITIZE_SCHEMA の構造テスト."""

    def test_schema_has_strict(self):
        assert SANITIZE_SCHEMA["strict"] is True

    def test_schema_type(self):
        assert SANITIZE_SCHEMA["type"] == "json_schema"

    def test_schema_name(self):
        assert SANITIZE_SCHEMA["name"] == "sanitize_result"

    def test_schema_required_fields(self):
        required = SANITIZE_SCHEMA["schema"]["required"]
        for field in ["risk", "summary", "keypoints", "safe_conversation",
                      "issues_detected"]:
            assert field in required

    def test_schema_no_knowledge_card(self):
        """Step 1 schema does not include knowledge_card."""
        required = SANITIZE_SCHEMA["schema"]["required"]
        assert "knowledge_card" not in required
        props = SANITIZE_SCHEMA["schema"]["properties"]
        assert "knowledge_card" not in props

    def test_schema_additional_properties_false(self):
        schema = SANITIZE_SCHEMA["schema"]
        assert schema["additionalProperties"] is False


class TestKCExtractionSchema:
    """KC_EXTRACTION_SCHEMA の構造テスト."""

    def test_schema_has_strict(self):
        assert KC_EXTRACTION_SCHEMA["strict"] is True

    def test_schema_required_fields(self):
        required = KC_EXTRACTION_SCHEMA["schema"]["required"]
        for field in ["extractable", "question", "answer", "evidence", "confidence"]:
            assert field in required

    def test_confidence_enum(self):
        conf = KC_EXTRACTION_SCHEMA["schema"]["properties"]["confidence"]
        assert conf["enum"] == ["none", "low", "medium", "high"]

    def test_evidence_is_array(self):
        ev = KC_EXTRACTION_SCHEMA["schema"]["properties"]["evidence"]
        assert ev["type"] == "array"


class TestValidateSchema:
    """validate_schema() のテスト."""

    def test_valid_low_risk(self):
        assert validate_schema(FIXTURE_LOW_RISK) is True

    def test_valid_high_risk(self):
        assert validate_schema(FIXTURE_HIGH_RISK) is True

    def test_invalid_risk(self):
        data = {**FIXTURE_LOW_RISK, "risk": "unknown"}
        assert validate_schema(data) is False

    def test_missing_summary(self):
        data = {k: v for k, v in FIXTURE_LOW_RISK.items() if k != "summary"}
        assert validate_schema(data) is False

    def test_empty_summary_input(self):
        data = {**FIXTURE_LOW_RISK, "summary": {"input": "", "output": "test"}}
        assert validate_schema(data) is False

    def test_not_dict(self):
        assert validate_schema("string") is False
        assert validate_schema([]) is False


class TestKnowledgeCardValidation:
    """validate_knowledge_card() と should_keep_knowledge_card() のテスト."""

    def test_extractable_high_confidence(self):
        warnings = validate_knowledge_card(FIXTURE_KC_EXTRACTABLE)
        assert len(warnings) == 0
        assert should_keep_knowledge_card(FIXTURE_KC_EXTRACTABLE) is True

    def test_not_extractable(self):
        warnings = validate_knowledge_card(FIXTURE_KC_NOT_EXTRACTABLE)
        assert len(warnings) == 0  # extractable=false → no validation
        assert should_keep_knowledge_card(FIXTURE_KC_NOT_EXTRACTABLE) is False

    def test_weak_confidence_rejected(self):
        warnings = validate_knowledge_card(FIXTURE_KC_WEAK)
        assert any("kc_no_evidence" in w for w in warnings)
        assert should_keep_knowledge_card(FIXTURE_KC_WEAK) is False

    def test_low_confidence_rejected(self):
        kc = {**FIXTURE_KC_EXTRACTABLE, "confidence": "low"}
        assert should_keep_knowledge_card(kc) is False

    def test_short_question_rejected(self):
        kc = {**FIXTURE_KC_EXTRACTABLE, "question": "何？"}
        assert should_keep_knowledge_card(kc) is False

    def test_short_answer_rejected(self):
        kc = {**FIXTURE_KC_EXTRACTABLE, "answer": "わからない"}
        assert should_keep_knowledge_card(kc) is False

    def test_medium_confidence_accepted(self):
        kc = {**FIXTURE_KC_EXTRACTABLE, "confidence": "medium"}
        assert should_keep_knowledge_card(kc) is True


class TestQualityGates:
    """run_quality_gates() のテスト."""

    def test_clean_data_no_critical_warnings(self):
        warnings = run_quality_gates(FIXTURE_LOW_RISK)
        assert "keypoints_empty" not in warnings
        assert "conversation_too_short" not in warnings

    def test_empty_keypoints(self):
        data = {**FIXTURE_LOW_RISK, "keypoints": []}
        warnings = run_quality_gates(data)
        assert "keypoints_empty" in warnings

    def test_short_conversation(self):
        data = {**FIXTURE_LOW_RISK, "safe_conversation": [
            {"speaker": "user", "text": "test"},
        ]}
        warnings = run_quality_gates(data)
        assert "conversation_too_short" in warnings

    def test_no_knowledge_card_null_warning(self):
        """knowledge_card_null warning is removed (Step 2 handles KC)."""
        data = {**FIXTURE_LOW_RISK}
        # No knowledge_card key at all
        warnings = run_quality_gates(data)
        assert "knowledge_card_null" not in warnings

    def test_kc_quality_gate_integrated(self):
        """KC quality warnings integrated into run_quality_gates."""
        data = {
            **FIXTURE_LOW_RISK,
            "knowledge_card": FIXTURE_KC_WEAK,
        }
        warnings = run_quality_gates(data)
        assert any("kc_" in w for w in warnings)

    def test_kc_good_no_extra_warnings(self):
        """Good KC adds no warnings."""
        data = {
            **FIXTURE_LOW_RISK,
            "knowledge_card": FIXTURE_KC_EXTRACTABLE,
        }
        warnings = run_quality_gates(data)
        kc_warnings = [w for w in warnings if w.startswith("kc_")]
        assert len(kc_warnings) == 0

    def test_pii_in_low_risk(self):
        data = {
            **FIXTURE_LOW_RISK,
            "risk": "low",
            "summary": {
                "input": "test",
                "output": "090-1234-5678に連絡、35歳の方がtaro@example.comへ",
            },
        }
        warnings = run_quality_gates(data)
        pii_warnings = [w for w in warnings if w.startswith("risk_low_but_pii_found")]
        assert len(pii_warnings) > 0

    def test_gibberish_in_conversation(self):
        data = {
            **FIXTURE_LOW_RISK,
            "safe_conversation": [
                {"speaker": "user", "text": "ラフィクションスケーカーカーデンタル"},
                {"speaker": "assistant", "text": "普通の返事"},
                {"speaker": "user", "text": "普通の質問"},
            ],
        }
        warnings = run_quality_gates(data)
        gibberish_warnings = [w for w in warnings if "gibberish" in w]
        assert len(gibberish_warnings) > 0


# ---------------------------------------------------------------------------
# Test: responses_sanitizer.py
# ---------------------------------------------------------------------------

class TestResponsesSanitizer:
    """ResponsesSanitizer のユニットテスト."""

    def test_sanitize_low_risk_with_kc(self):
        """Step 1 + Step 2: KC extraction succeeds."""
        s = _make_sanitizer()
        # Step 1: sanitize, Step 2: KC extraction
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_EXTRACTABLE),
        ]

        result = s.sanitize("test-session/01", "今日のミーティングでは計画を話し合いました。" * 3)
        assert result is not None
        assert result.session_id == "test-session/01"
        assert result.data["risk"] == "low"
        assert len(result.data["keypoints"]) >= KC_MIN_KEYPOINTS
        # KC should be present
        assert result.data["knowledge_card"] is not None
        assert result.data["knowledge_card"]["confidence"] == "high"
        assert len(result.data["knowledge_card"]["evidence"]) >= 1

    def test_sanitize_low_risk_kc_not_extractable(self):
        """Step 2 returns extractable=false → knowledge_card=null."""
        s = _make_sanitizer()
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_NOT_EXTRACTABLE),
        ]

        result = s.sanitize("test-session/01", "今日のミーティングでは計画を話し合いました。" * 3)
        assert result is not None
        assert result.data["knowledge_card"] is None

    def test_sanitize_few_keypoints_skips_kc(self):
        """keypoints < KC_MIN_KEYPOINTS → KC extraction skipped."""
        s = _make_sanitizer()
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_HIGH_RISK),
            # Step 2 should NOT be called
        ]

        result = s.sanitize("test-session/02", "佐藤部長が採用計画について説明しました。" * 3)
        assert result is not None
        assert result.data["risk"] == "high"
        assert result.data["knowledge_card"] is None
        # Only 1 API call (Step 1 only)
        assert s._client.responses.create.call_count == 1

    def test_sanitize_kc_quality_gate_rejects(self):
        """Step 2 returns weak KC → quality gate rejects → null."""
        s = _make_sanitizer()
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_WEAK),
        ]

        result = s.sanitize("test-session/01", "今日のミーティングでは計画を話し合いました。" * 3)
        assert result is not None
        assert result.data["knowledge_card"] is None

    def test_sanitize_kc_api_error_graceful(self):
        """Step 2 API error → knowledge_card=null (doesn't fail whole sanitize)."""
        s = _make_sanitizer()
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_LOW_RISK),
            Exception("API error"),
        ]

        result = s.sanitize("test-session/01", "今日のミーティングでは計画を話し合いました。" * 3)
        assert result is not None
        assert result.data["knowledge_card"] is None
        # Step 1 still succeeded
        assert result.data["risk"] == "low"

    def test_sanitize_disabled(self):
        s = _make_sanitizer()
        s._enabled = False
        result = s.sanitize("test", "some text" * 10)
        assert result is None

    def test_sanitize_short_text(self):
        s = _make_sanitizer()
        result = s.sanitize("test", "短い")
        assert result is None

    def test_sanitize_api_error_retries(self):
        """リトライ可能なエラー → リトライ後に成功."""
        s = _make_sanitizer()
        s._max_retries = 2

        # 1回目: タイムアウト, 2回目: Step 1 成功 + Step 2 成功
        s._client.responses.create.side_effect = [
            Exception("timeout error"),
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_EXTRACTABLE),
        ]

        with patch("responses_sanitizer.time.sleep"):  # sleep をスキップ
            result = s.sanitize("test", "テストテキスト" * 10)

        assert result is not None
        assert result.data["risk"] == "low"

    def test_sanitize_schema_validation_retries(self):
        """スキーマ不一致 → リトライして成功."""
        s = _make_sanitizer()
        s._max_retries = 2

        invalid_data = {"risk": "invalid"}  # スキーマ不一致
        s._client.responses.create.side_effect = [
            _make_mock_response(invalid_data),
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_EXTRACTABLE),
        ]

        result = s.sanitize("test", "テストテキスト" * 10)
        assert result is not None

    def test_sanitize_all_retries_fail(self):
        """全リトライ失敗 → None."""
        s = _make_sanitizer()
        s._max_retries = 2

        s._client.responses.create.side_effect = Exception("timeout error")

        with patch("responses_sanitizer.time.sleep"):
            result = s.sanitize("test", "テストテキスト" * 10)

        assert result is None

    def test_sanitize_incomplete_response(self):
        """incomplete レスポンス → リトライ."""
        s = _make_sanitizer()
        s._max_retries = 2

        incomplete = _make_mock_response(FIXTURE_LOW_RISK, status="incomplete")
        incomplete.incomplete_details = MagicMock()
        incomplete.incomplete_details.reason = "max_output_tokens"

        s._client.responses.create.return_value = incomplete

        with patch("responses_sanitizer.time.sleep"):
            result = s.sanitize("test", "テストテキスト" * 10)

        # max_output_tokens は fatal → None
        assert result is None

    def test_usage_tracking(self):
        """usage 情報がトラッキングされること."""
        s = _make_sanitizer()
        s._client.responses.create.side_effect = [
            _make_mock_response(FIXTURE_LOW_RISK),
            _make_mock_response(FIXTURE_KC_EXTRACTABLE),
        ]

        result = s.sanitize("test", "テストテキスト" * 10)
        assert result is not None
        assert "input_tokens" in result.usage
        assert result.usage["input_tokens"] == 500

    def test_warnings_tracked(self):
        """品質ゲート warnings が結果に含まれること."""
        s = _make_sanitizer()
        # Return data with empty keypoints to trigger warning
        data_few_kp = {**FIXTURE_LOW_RISK, "keypoints": []}
        s._client.responses.create.side_effect = [
            _make_mock_response(data_few_kp),
        ]

        result = s.sanitize("test", "テストテキスト" * 10)
        assert result is not None
        assert "keypoints_empty" in result.warnings

    def test_provider_name(self):
        s = _make_sanitizer()
        assert "gpt-5-nano" in s.provider_name

    def test_enabled_property(self):
        s = _make_sanitizer()
        assert s.enabled is True
        s._enabled = False
        assert s.enabled is False
