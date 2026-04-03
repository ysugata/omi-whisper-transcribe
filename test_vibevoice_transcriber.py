"""Tests for vibevoice_transcriber module."""

import json
import pytest

from vibevoice_transcriber import _parse_output, _normalize_segments


class TestParseOutput:
    def test_valid_json_array(self):
        text = json.dumps([
            {"Start": 0, "End": 5.4, "Speaker": 0, "Content": "Hello world"},
            {"Start": 5.4, "End": 10.2, "Speaker": 1, "Content": "こんにちは"},
        ])
        result = _parse_output(text)
        assert len(result) == 2
        assert result[0]["Content"] == "Hello world"
        assert result[1]["Speaker"] == 1

    def test_json_with_surrounding_text(self):
        text = 'some prefix [{"Start": 0, "End": 3, "Speaker": 0, "Content": "test"}] suffix'
        result = _parse_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "test"

    def test_plain_text_fallback(self):
        text = "This is just plain text"
        result = _parse_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == text
        assert result[0]["Start"] == 0

    def test_empty_string(self):
        result = _parse_output("")
        assert result == []

    def test_whitespace_only(self):
        result = _parse_output("   \n  ")
        assert result == []

    def test_malformed_json(self):
        text = '[{"Start": 0, "End": 5, "Speaker": 0, "Content": "broken'
        result = _parse_output(text)
        # Falls through to plain text fallback
        assert len(result) == 1


class TestNormalizeSegments:
    def test_basic_normalization(self):
        raw = [
            {"Start": 0, "End": 5.4, "Speaker": 0, "Content": "Hello"},
            {"Start": 5.4, "End": 10.2, "Speaker": 1, "Content": "World"},
        ]
        result = _normalize_segments(raw)
        assert len(result) == 2
        assert result[0] == {"start": 0.0, "end": 5.4, "speaker": 0, "text": "Hello"}
        assert result[1] == {"start": 5.4, "end": 10.2, "speaker": 1, "text": "World"}

    def test_missing_fields(self):
        raw = [{"Content": "only text"}]
        result = _normalize_segments(raw)
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.0
        assert result[0]["speaker"] == 0
        assert result[0]["text"] == "only text"

    def test_empty_list(self):
        result = _normalize_segments([])
        assert result == []

    def test_string_number_coercion(self):
        raw = [{"Start": "1.5", "End": "3.0", "Speaker": 0, "Content": "test"}]
        result = _normalize_segments(raw)
        assert result[0]["start"] == 1.5
        assert result[0]["end"] == 3.0
