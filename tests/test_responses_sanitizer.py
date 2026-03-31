"""Unit tests for responses_sanitizer.py + sanitize_schema.py.

テスト方針:
- OpenAI API はモックする (実 API 呼び出しなし)
- スキーマ検証・品質ゲートはユニットテスト
- fixture 2件 (low risk / high risk) を使用
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from responses_sanitizer import ResponsesSanitizer, _RetryableError, _FatalError
from sanitize_schema import SANITIZE_SCHEMA, validate_schema, run_quality_gates


# ---------------------------------------------------------------------------
# Fixture data: LLM の応答をシミュレーション
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
    "knowledge_card": {
        "question": "プロジェクト計画ミーティングではどのような内容を話し合いますか？",
        "answer": "新機能の優先順位決定、テストスケジュールの確認、各担当からの進捗報告などを行います。",
    },
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
    "knowledge_card": None,
    "issues_detected": ["pii_redacted", "asr_noise"],
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
# Test: sanitize_schema.py
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
                      "knowledge_card", "issues_detected"]:
            assert field in required

    def test_schema_additional_properties_false(self):
        schema = SANITIZE_SCHEMA["schema"]
        assert schema["additionalProperties"] is False


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

    def test_null_knowledge_card(self):
        data = {**FIXTURE_LOW_RISK, "knowledge_card": None}
        assert validate_schema(data) is True

    def test_invalid_knowledge_card(self):
        data = {**FIXTURE_LOW_RISK, "knowledge_card": {"only_question": "test"}}
        assert validate_schema(data) is False


class TestQualityGates:
    """run_quality_gates() のテスト."""

    def test_clean_data_no_warnings(self):
        warnings = run_quality_gates(FIXTURE_LOW_RISK)
        # knowledge_card が有効なので knowledge_card_null は出ない
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

    def test_null_knowledge_card_warning(self):
        data = {**FIXTURE_LOW_RISK, "knowledge_card": None}
        warnings = run_quality_gates(data)
        assert "knowledge_card_null" in warnings

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

    def test_sanitize_low_risk(self):
        s = _make_sanitizer()
        s._client.responses.create.return_value = _make_mock_response(FIXTURE_LOW_RISK)

        result = s.sanitize("test-session/01", "今日のミーティングでは計画を話し合いました。" * 3)
        assert result is not None
        assert result.session_id == "test-session/01"
        assert result.data["risk"] == "low"
        assert len(result.data["keypoints"]) >= 1

    def test_sanitize_high_risk(self):
        s = _make_sanitizer()
        s._client.responses.create.return_value = _make_mock_response(FIXTURE_HIGH_RISK)

        result = s.sanitize("test-session/02", "佐藤部長が採用計画について説明しました。" * 3)
        assert result is not None
        assert result.data["risk"] == "high"
        assert result.data["knowledge_card"] is None
        assert "pii_redacted" in result.data["issues_detected"]

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

        # 1回目: タイムアウト, 2回目: 成功
        s._client.responses.create.side_effect = [
            Exception("timeout error"),
            _make_mock_response(FIXTURE_LOW_RISK),
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
        s._client.responses.create.return_value = _make_mock_response(FIXTURE_LOW_RISK)

        result = s.sanitize("test", "テストテキスト" * 10)
        assert result is not None
        assert "input_tokens" in result.usage
        assert result.usage["input_tokens"] == 500

    def test_warnings_tracked(self):
        """品質ゲート warnings が結果に含まれること."""
        s = _make_sanitizer()
        data_no_kcard = {**FIXTURE_LOW_RISK, "knowledge_card": None}
        s._client.responses.create.return_value = _make_mock_response(data_no_kcard)

        result = s.sanitize("test", "テストテキスト" * 10)
        assert result is not None
        assert "knowledge_card_null" in result.warnings

    def test_provider_name(self):
        s = _make_sanitizer()
        assert "gpt-5-nano" in s.provider_name

    def test_enabled_property(self):
        s = _make_sanitizer()
        assert s.enabled is True
        s._enabled = False
        assert s.enabled is False
