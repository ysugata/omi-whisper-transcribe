"""Tests for diarizer.py — pyannote direct pipeline.

Covers:
1. _validate_segments() — timestamp clamping and filtering
2. _assign_speakers() — time-overlap speaker assignment
3. _split_at_speaker_boundaries() — text splitting at speaker changes
4. _map_segment_to_speakers() — single/multi speaker handling
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diarizer import Diarizer


@pytest.fixture
def diarizer():
    return Diarizer(hf_token="dummy", device="cpu")


# =====================================================================
# _validate_segments
# =====================================================================

class TestValidateSegments:
    def test_clamps_timestamps(self, diarizer):
        segments = [{"text": "hello", "start": -1.0, "end": 100.0}]
        result = diarizer._validate_segments(segments, 10.0)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 10.0

    def test_skips_empty_text(self, diarizer):
        segments = [
            {"text": "", "start": 0.0, "end": 1.0},
            {"text": "  ", "start": 1.0, "end": 2.0},
            {"text": "valid", "start": 2.0, "end": 3.0},
        ]
        result = diarizer._validate_segments(segments, 5.0)
        assert len(result) == 1
        assert result[0]["text"] == "valid"

    def test_fixes_inverted_timestamps(self, diarizer):
        segments = [{"text": "hello", "start": 5.0, "end": 3.0}]
        result = diarizer._validate_segments(segments, 10.0)
        assert len(result) == 1
        assert result[0]["end"] == 5.5  # start + 0.5

    def test_skips_beyond_audio(self, diarizer):
        segments = [{"text": "hello", "start": 11.0, "end": 12.0}]
        result = diarizer._validate_segments(segments, 10.0)
        assert len(result) == 0

    def test_preserves_extra_fields(self, diarizer):
        segments = [{"text": "hello", "start": 0.0, "end": 1.0, "custom": "data"}]
        result = diarizer._validate_segments(segments, 5.0)
        assert result[0]["custom"] == "data"


# =====================================================================
# _assign_speakers
# =====================================================================

def _make_annotation(tracks):
    """Create a mock pyannote Annotation from (start, end, speaker) tuples."""
    mock = MagicMock()
    track_items = []
    for start, end, speaker in tracks:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        track_items.append((seg, None, speaker))
    mock.itertracks.return_value = track_items
    return mock


class TestAssignSpeakers:
    def test_single_speaker(self, diarizer):
        segments = [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "world", "start": 1.0, "end": 2.0},
        ]
        annotation = _make_annotation([
            (0.0, 2.0, "SPEAKER_00"),
        ])
        result = diarizer._assign_speakers(segments, annotation)
        assert len(result) == 2
        assert all(s["speaker"] == "SPEAKER_00" for s in result)

    def test_two_speakers_no_split(self, diarizer):
        """Two segments, each fully within one speaker region."""
        segments = [
            {"text": "hello", "start": 0.0, "end": 1.0},
            {"text": "hi", "start": 1.5, "end": 2.5},
        ]
        annotation = _make_annotation([
            (0.0, 1.2, "SPEAKER_00"),
            (1.3, 3.0, "SPEAKER_01"),
        ])
        result = diarizer._assign_speakers(segments, annotation)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_no_diarization_segments(self, diarizer):
        segments = [{"text": "hello", "start": 0.0, "end": 1.0}]
        annotation = _make_annotation([])
        result = diarizer._assign_speakers(segments, annotation)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_segment_spanning_two_speakers_gets_split(self, diarizer):
        """A 10s segment spanning two speakers should be split."""
        segments = [
            {"text": "Hello world how are you doing today", "start": 0.0, "end": 10.0},
        ]
        annotation = _make_annotation([
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ])
        result = diarizer._assign_speakers(segments, annotation)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_no_overlap_gets_default(self, diarizer):
        """Segment outside all diarization regions gets default speaker."""
        segments = [{"text": "lost", "start": 50.0, "end": 51.0}]
        annotation = _make_annotation([
            (0.0, 10.0, "SPEAKER_00"),
        ])
        result = diarizer._assign_speakers(segments, annotation)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["speaker_id"] == 0


# =====================================================================
# _split_at_speaker_boundaries
# =====================================================================

class TestSplitAtSpeakerBoundaries:
    def test_proportional_text_split(self, diarizer):
        seg = {"text": "abcdefghij", "start": 0.0, "end": 10.0}
        # 50-50 split
        overlaps = [
            (0.0, 5.0, "SPEAKER_00", 5.0),
            (5.0, 10.0, "SPEAKER_01", 5.0),
        ]
        result = diarizer._split_at_speaker_boundaries(seg, overlaps)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
        # Text should be split roughly in half
        total_text = result[0]["text"] + result[1]["text"]
        assert len(total_text) == 10

    def test_preserves_extra_fields(self, diarizer):
        seg = {"text": "hello world", "start": 0.0, "end": 4.0, "custom": "data"}
        overlaps = [
            (0.0, 2.0, "SPEAKER_00", 2.0),
            (2.0, 4.0, "SPEAKER_01", 2.0),
        ]
        result = diarizer._split_at_speaker_boundaries(seg, overlaps)
        assert all(s["custom"] == "data" for s in result)

    def test_skips_empty_chunks(self, diarizer):
        seg = {"text": "ab", "start": 0.0, "end": 10.0}
        # 90-10 split: second chunk would be empty
        overlaps = [
            (0.0, 9.0, "SPEAKER_00", 9.0),
            (9.0, 10.0, "SPEAKER_01", 1.0),
        ]
        result = diarizer._split_at_speaker_boundaries(seg, overlaps)
        # Very short chunk may be skipped
        assert len(result) >= 1
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merges_consecutive_same_speaker(self, diarizer):
        seg = {"text": "hello world", "start": 0.0, "end": 6.0}
        overlaps = [
            (0.0, 2.0, "SPEAKER_00", 2.0),
            (2.0, 4.0, "SPEAKER_00", 2.0),  # same speaker
            (4.0, 6.0, "SPEAKER_01", 2.0),
        ]
        result = diarizer._split_at_speaker_boundaries(seg, overlaps)
        # The two SPEAKER_00 entries should be merged
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"


# =====================================================================
# _parse_speaker_id
# =====================================================================

class TestParseSpeakerId:
    def test_normal(self):
        assert Diarizer._parse_speaker_id("SPEAKER_01") == 1

    def test_double_digit(self):
        assert Diarizer._parse_speaker_id("SPEAKER_12") == 12

    def test_invalid(self):
        assert Diarizer._parse_speaker_id("unknown") == 0

    def test_empty(self):
        assert Diarizer._parse_speaker_id("") == 0
