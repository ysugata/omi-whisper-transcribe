"""Tests for LLMCorrector — optimized batch mode with JSON schema."""

import json
from unittest import mock

import pytest

from llm_corrector import LLMCorrector, BATCH_SYSTEM_PROMPT, CORRECTIONS_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corrector(response_content: str):
    """Create an LLMCorrector with mocked API client."""
    corrector = LLMCorrector.__new__(LLMCorrector)
    corrector._enabled = True
    corrector._model = "gemini-2.5-flash-lite"
    corrector._temperature = 0
    corrector._max_tokens = 2048
    corrector._warned = False

    mock_client = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = response_content
    mock_client.chat.completions.create.return_value = mock_response
    corrector._client = mock_client

    return corrector, mock_client


# ---------------------------------------------------------------------------
# _is_punctuation_only_change tests
# ---------------------------------------------------------------------------

class TestIsPunctuationOnlyChange:
    def test_identical(self):
        assert LLMCorrector._is_punctuation_only_change("こんにちは", "こんにちは") is True

    def test_half_to_full_question(self):
        assert LLMCorrector._is_punctuation_only_change("本当に?", "本当に？") is True

    def test_half_to_full_exclamation(self):
        assert LLMCorrector._is_punctuation_only_change("すごい!", "すごい！") is True

    def test_added_period(self):
        assert LLMCorrector._is_punctuation_only_change("おはよう", "おはよう。") is True

    def test_real_word_change(self):
        assert LLMCorrector._is_punctuation_only_change("ドッカー", "Docker") is False

    def test_word_plus_punct(self):
        assert LLMCorrector._is_punctuation_only_change("クエス", "クエスト") is False


# ---------------------------------------------------------------------------
# _filter_corrections tests
# ---------------------------------------------------------------------------

class TestFilterCorrections:
    def test_remove_no_change(self):
        corrections = [
            {"line": 1, "original": "テスト", "corrected": "テスト", "reason": "none"},
        ]
        result = LLMCorrector._filter_corrections(corrections, {1: "テスト"})
        assert len(result) == 0

    def test_remove_punct_only(self):
        corrections = [
            {"line": 1, "original": "本当に?", "corrected": "本当に？", "reason": "punct"},
        ]
        result = LLMCorrector._filter_corrections(corrections, {1: "本当に?"})
        assert len(result) == 0

    def test_keep_real_correction(self):
        corrections = [
            {"line": 1, "original": "ドッカー", "corrected": "Docker", "reason": "tech term"},
        ]
        result = LLMCorrector._filter_corrections(corrections, {1: "ドッカー"})
        assert len(result) == 1
        assert result[0]["corrected"] == "Docker"

    def test_uses_actual_original(self):
        """Should use original_texts map, not LLM-echoed original."""
        corrections = [
            {"line": 1, "original": "wrong echo", "corrected": "テスト文", "reason": "fix"},
        ]
        # The actual original is different from what LLM echoed
        result = LLMCorrector._filter_corrections(corrections, {1: "テスト文"})
        # Should be filtered as no-change (actual original == corrected)
        assert len(result) == 0

    def test_remove_empty_corrected(self):
        corrections = [
            {"line": 1, "original": "テスト", "corrected": "", "reason": "empty"},
        ]
        result = LLMCorrector._filter_corrections(corrections, {1: "テスト"})
        assert len(result) == 0

    def test_mixed_corrections(self):
        corrections = [
            {"line": 1, "original": "テスト", "corrected": "テスト", "reason": "none"},
            {"line": 2, "original": "本当?", "corrected": "本当？", "reason": "punct"},
            {"line": 3, "original": "クエス", "corrected": "クエスト", "reason": "truncation"},
        ]
        original_texts = {1: "テスト", 2: "本当?", 3: "クエス"}
        result = LLMCorrector._filter_corrections(corrections, original_texts)
        assert len(result) == 1
        assert result[0]["line"] == 3


# ---------------------------------------------------------------------------
# correct_segments tests — JSON schema mode
# ---------------------------------------------------------------------------

class TestCorrectSegmentsJson:
    def test_json_corrections_applied(self):
        """JSON corrections should be applied to segments."""
        json_response = json.dumps({
            "corrections": [
                {"line": 2, "original": "ドッカーのコンテナ", "corrected": "Dockerのコンテナ", "reason": "tech term"},
            ]
        })
        corrector, mock_client = _make_corrector(json_response)
        segments = [
            {"text": "こんにちは"},
            {"text": "ドッカーのコンテナ"},
            {"text": "終わり"},
        ]
        result = corrector.correct_segments(segments, known_terms=["Docker"])
        assert result[1]["text"] == "Dockerのコンテナ"
        # Others unchanged
        assert result[0]["text"] == "こんにちは"
        assert result[2]["text"] == "終わり"

    def test_json_punct_filtered(self):
        """Punctuation-only corrections should be filtered out."""
        json_response = json.dumps({
            "corrections": [
                {"line": 1, "original": "本当に?", "corrected": "本当に？", "reason": "punct"},
            ]
        })
        corrector, _ = _make_corrector(json_response)
        segments = [{"text": "本当に?"}]
        result = corrector.correct_segments(segments)
        assert result[0]["text"] == "本当に?"

    def test_json_no_change_filtered(self):
        """No-change corrections should be filtered out."""
        json_response = json.dumps({
            "corrections": [
                {"line": 1, "original": "テストです", "corrected": "テストです", "reason": "no fix needed"},
            ]
        })
        corrector, _ = _make_corrector(json_response)
        segments = [{"text": "テストです"}]
        result = corrector.correct_segments(segments)
        assert result[0]["text"] == "テストです"

    def test_empty_corrections(self):
        """Empty corrections array should leave all segments unchanged."""
        json_response = json.dumps({"corrections": []})
        corrector, _ = _make_corrector(json_response)
        segments = [{"text": "テスト文です"}]
        result = corrector.correct_segments(segments)
        assert result[0]["text"] == "テスト文です"

    def test_uses_batch_system_prompt(self):
        """Should use BATCH_SYSTEM_PROMPT, not legacy prompt."""
        json_response = json.dumps({"corrections": []})
        corrector, mock_client = _make_corrector(json_response)
        segments = [{"text": "テスト文です"}]
        corrector.correct_segments(segments)
        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args[1]["messages"][0]["content"]
        assert system_msg == BATCH_SYSTEM_PROMPT

    def test_uses_json_schema_format(self):
        """Should pass response_format=CORRECTIONS_SCHEMA."""
        json_response = json.dumps({"corrections": []})
        corrector, mock_client = _make_corrector(json_response)
        segments = [{"text": "テスト文です"}]
        corrector.correct_segments(segments)
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["response_format"] == CORRECTIONS_SCHEMA

    def test_max_tokens_at_least_4096(self):
        """max_tokens should be at least 4096 for JSON format."""
        json_response = json.dumps({"corrections": []})
        corrector, mock_client = _make_corrector(json_response)
        segments = [{"text": "テスト文です"}]
        corrector.correct_segments(segments)
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] >= 4096

    def test_short_segments_skipped(self):
        """Segments shorter than min_text_length should be skipped."""
        json_response = json.dumps({"corrections": []})
        corrector, mock_client = _make_corrector(json_response)
        segments = [{"text": "ab"}, {"text": "長い文章です"}]
        result = corrector.correct_segments(segments, min_text_length=5)
        assert result[0]["text"] == "ab"

    def test_disabled_returns_original(self):
        """Disabled corrector should return segments unchanged."""
        corrector = LLMCorrector.__new__(LLMCorrector)
        corrector._enabled = False
        segments = [{"text": "テスト"}]
        assert corrector.correct_segments(segments) is segments


# ---------------------------------------------------------------------------
# Fallback to numbered format
# ---------------------------------------------------------------------------

class TestCorrectSegmentsFallback:
    def test_json_parse_failure_falls_back(self):
        """When JSON parse fails, should fall back to numbered format parser."""
        # Response in numbered format instead of JSON
        numbered_response = "[1] Dockerのコンテナ\n[2] 終わりです"
        corrector, _ = _make_corrector(numbered_response)
        segments = [
            {"text": "ドッカーのコンテナ"},
            {"text": "終わりです"},
        ]
        result = corrector.correct_segments(segments)
        # Numbered fallback should parse and apply
        assert result[0]["text"] == "Dockerのコンテナ"

    def test_api_error_preserves_text(self):
        """API errors should preserve original text."""
        corrector = LLMCorrector.__new__(LLMCorrector)
        corrector._enabled = True
        corrector._model = "gemini-2.5-flash-lite"
        corrector._temperature = 0
        corrector._max_tokens = 2048

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        corrector._client = mock_client

        segments = [{"text": "テスト文です"}]
        result = corrector.correct_segments(segments)
        assert result[0]["text"] == "テスト文です"


# ---------------------------------------------------------------------------
# validate_response (existing, just ensure it works with new flow)
# ---------------------------------------------------------------------------

class TestValidateResponse:
    def test_too_long_rejected(self):
        corrector = LLMCorrector.__new__(LLMCorrector)
        result = corrector._validate_response("短い", "とても長い文章" * 100)
        assert result == "短い"

    def test_too_short_rejected(self):
        corrector = LLMCorrector.__new__(LLMCorrector)
        result = corrector._validate_response("これはテストの文章です", "あ")
        assert result == "これはテストの文章です"

    def test_valid_correction_accepted(self):
        corrector = LLMCorrector.__new__(LLMCorrector)
        result = corrector._validate_response("ドッカーのコンテナ", "Dockerのコンテナ")
        assert result == "Dockerのコンテナ"
