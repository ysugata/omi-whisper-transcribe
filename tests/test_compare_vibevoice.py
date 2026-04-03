"""Tests for tools/compare_vibevoice.py"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from compare_vibevoice import (
    _parse_vibevoice_output,
    compare_text,
    compare_speakers,
    compare_timestamps,
    find_diverse_sessions,
    load_session,
    build_hotwords_prompt,
)


# ---------------------------------------------------------------------------
# _parse_vibevoice_output
# ---------------------------------------------------------------------------

class TestParseVibevoiceOutput:
    def test_valid_json_array(self):
        text = '[{"Start": 0, "End": 5.0, "Speaker": 0, "Content": "hello"}]'
        result = _parse_vibevoice_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "hello"
        assert result[0]["Start"] == 0
        assert result[0]["End"] == 5.0

    def test_json_with_whitespace(self):
        text = '  [{"Start": 0, "End": 2, "Speaker": 0, "Content": "test"}]  '
        result = _parse_vibevoice_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "test"

    def test_json_embedded_in_text(self):
        text = 'Output: [{"Start": 0, "End": 3, "Speaker": 0, "Content": "abc"}] done'
        result = _parse_vibevoice_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "abc"

    def test_multiple_segments(self):
        segs = [
            {"Start": 0, "End": 5, "Speaker": 0, "Content": "Hello"},
            {"Start": 5, "End": 10, "Speaker": 1, "Content": "World"},
        ]
        text = json.dumps(segs)
        result = _parse_vibevoice_output(text)
        assert len(result) == 2
        assert result[0]["Speaker"] == 0
        assert result[1]["Speaker"] == 1

    def test_empty_string(self):
        result = _parse_vibevoice_output("")
        assert result == []

    def test_plain_text_fallback(self):
        text = "This is plain transcription text"
        result = _parse_vibevoice_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == text
        assert result[0]["Start"] == 0


# ---------------------------------------------------------------------------
# compare_text
# ---------------------------------------------------------------------------

class TestCompareText:
    def test_identical_text(self):
        whisper = [{"text": "hello world", "start": 0, "end": 5}]
        vibe = [{"Content": "hello world", "Start": 0, "End": 5}]
        result = compare_text(whisper, vibe)
        assert result["text_similarity"] == 1.0
        assert result["notable_differences"] == []

    def test_similar_text(self):
        whisper = [{"text": "今日はいい天気ですね", "start": 0, "end": 5}]
        vibe = [{"Content": "今日は良い天気ですね", "Start": 0, "End": 5}]
        result = compare_text(whisper, vibe)
        assert 0.5 < result["text_similarity"] < 1.0

    def test_empty_segments(self):
        result = compare_text([], [])
        # difflib returns 1.0 for two empty strings (identical)
        assert result["text_similarity"] == 1.0
        assert result["whisper_chars"] == 0
        assert result["vibevoice_chars"] == 0

    def test_char_counts(self):
        whisper = [{"text": "abc", "start": 0, "end": 1}, {"text": "def", "start": 1, "end": 2}]
        vibe = [{"Content": "abcdefg", "Start": 0, "End": 2}]
        result = compare_text(whisper, vibe)
        assert result["whisper_chars"] == 6
        assert result["vibevoice_chars"] == 7


# ---------------------------------------------------------------------------
# compare_speakers
# ---------------------------------------------------------------------------

class TestCompareSpeakers:
    def test_single_speaker(self):
        whisper = [{"speaker": "SPEAKER_00", "start": 0, "end": 10}]
        vibe = [{"Speaker": 0, "Start": 0, "End": 10}]
        result = compare_speakers(whisper, vibe)
        assert result["speaker_agreement"] == 1.0
        assert result["whisper_speakers"] == 1
        assert result["vibevoice_speakers"] == 1

    def test_two_speakers_matching(self):
        whisper = [
            {"speaker": "SPEAKER_00", "start": 0, "end": 5},
            {"speaker": "SPEAKER_01", "start": 5, "end": 10},
        ]
        vibe = [
            {"Speaker": 0, "Start": 0, "End": 5},
            {"Speaker": 1, "Start": 5, "End": 10},
        ]
        result = compare_speakers(whisper, vibe)
        assert result["speaker_agreement"] == 1.0

    def test_empty_segments(self):
        result = compare_speakers([], [])
        assert result["speaker_agreement"] == 0


# ---------------------------------------------------------------------------
# compare_timestamps
# ---------------------------------------------------------------------------

class TestCompareTimestamps:
    def test_identical_timestamps(self):
        whisper = [{"start": 0, "end": 5}, {"start": 5, "end": 10}]
        vibe = [{"Start": 0, "End": 5}, {"Start": 5, "End": 10}]
        result = compare_timestamps(whisper, vibe)
        assert result["avg_boundary_diff_s"] == 0.0
        assert result["coverage_overlap"] == 1.0

    def test_different_seg_counts(self):
        whisper = [{"start": 0, "end": 10}]
        vibe = [{"Start": 0, "End": 5}, {"Start": 5, "End": 10}]
        result = compare_timestamps(whisper, vibe)
        assert result["whisper_seg_count"] == 1
        assert result["vibevoice_seg_count"] == 2

    def test_empty_segments(self):
        result = compare_timestamps([], [])
        assert result["avg_boundary_diff_s"] == 0
        assert result["coverage_overlap"] == 0


# ---------------------------------------------------------------------------
# find_diverse_sessions
# ---------------------------------------------------------------------------

class TestFindDiverseSessions:
    def test_specific_sessions_nonexistent(self, tmp_path):
        with patch("compare_vibevoice.RECORDINGS_DIR", tmp_path):
            result = find_diverse_sessions(specific=["2026-01-01/00-00-00"])
            assert result == []

    def test_specific_sessions_found(self, tmp_path):
        sess_dir = tmp_path / "2026-01-01" / "00-00-00"
        sess_dir.mkdir(parents=True)
        (sess_dir / "audio.ogg").write_text("fake")
        (sess_dir / "transcript.json").write_text('{"session": {}, "segments": []}')

        with patch("compare_vibevoice.RECORDINGS_DIR", tmp_path):
            result = find_diverse_sessions(specific=["2026-01-01/00-00-00"])
            assert result == ["2026-01-01/00-00-00"]


# ---------------------------------------------------------------------------
# build_hotwords_prompt
# ---------------------------------------------------------------------------

class TestBuildHotwordsPrompt:
    def test_returns_string(self):
        # Should not crash even if hotwords config doesn't exist
        result = build_hotwords_prompt()
        assert isinstance(result, str)
