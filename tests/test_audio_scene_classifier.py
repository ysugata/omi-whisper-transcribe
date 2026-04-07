"""Tests for AudioSceneClassifier."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from audio_scene_classifier import (
    AudioSceneClassifier,
    SCENE_CATEGORIES,
    SPATIAL_CATEGORIES,
    MEDIA_INDICES,
    NOISE_SOURCE_CATEGORIES,
    EVENT_CATEGORIES,
    MUSIC_INSTRUMENT_INDICES,
    MUSIC_BGM_INDICES,
    MUSIC_SINGING_INDEX,
    MUSIC_ALL_INDICES,
    SPEAKER_COUNT_CATEGORIES,
    QUALITY_CATEGORIES,
    DEFAULT_DIMENSION_THRESHOLDS,
    _noisy_or,
    _max_score_for,
)


def _make_scores(overrides: dict) -> np.ndarray:
    """Create a 527-element scores array with specific indices set."""
    scores = np.zeros(527, dtype=np.float32)
    for idx, val in overrides.items():
        scores[idx] = val
    return scores


# =========================================================================
# Mapping validation
# =========================================================================

class TestLabelMapping:
    """Verify category mapping constants."""

    def test_scene_categories_have_required_keys(self):
        for key, cat in SCENE_CATEGORIES.items():
            assert "indices" in cat, f"{key} missing indices"
            assert "label" in cat, f"{key} missing label"
            assert isinstance(cat["indices"], list)
            assert len(cat["indices"]) > 0

    def test_spatial_categories_have_required_keys(self):
        for key, cat in SPATIAL_CATEGORIES.items():
            assert "index" in cat, f"{key} missing index"
            assert "label" in cat, f"{key} missing label"
            assert isinstance(cat["index"], int)

    def test_scene_indices_within_audioset_range(self):
        for key, cat in SCENE_CATEGORIES.items():
            for idx in cat["indices"]:
                assert 0 <= idx < 527, f"{key} index {idx} out of AudioSet range"

    def test_spatial_indices_within_audioset_range(self):
        for key, cat in SPATIAL_CATEGORIES.items():
            assert 0 <= cat["index"] < 527, f"{key} index {cat['index']} out of range"

    def test_media_indices_within_audioset_range(self):
        for idx in MEDIA_INDICES:
            assert 0 <= idx < 527, f"MEDIA_INDICES {idx} out of range"


class TestExtendedMappingValidation:
    """Verify all extended dimension mapping constants."""

    def test_noise_source_indices_valid(self):
        for key, cat in NOISE_SOURCE_CATEGORIES.items():
            assert "indices" in cat and "label" in cat, f"{key} missing fields"
            for idx in cat["indices"]:
                assert 0 <= idx < 527, f"NOISE {key} index {idx} out of range"

    def test_event_indices_valid(self):
        for key, cat in EVENT_CATEGORIES.items():
            assert "indices" in cat and "label" in cat, f"{key} missing fields"
            for idx in cat["indices"]:
                assert 0 <= idx < 527, f"EVENT {key} index {idx} out of range"

    def test_speaker_count_indices_valid(self):
        for key, cat in SPEAKER_COUNT_CATEGORIES.items():
            assert "indices" in cat and "label" in cat, f"{key} missing fields"
            for idx in cat["indices"]:
                assert 0 <= idx < 527, f"SPEAKER {key} index {idx} out of range"

    def test_quality_indices_valid(self):
        for key, cat in QUALITY_CATEGORIES.items():
            assert "indices" in cat and "label" in cat, f"{key} missing fields"
            for idx in cat["indices"]:
                assert 0 <= idx < 527, f"QUALITY {key} index {idx} out of range"

    def test_music_instrument_indices_valid(self):
        for idx in MUSIC_INSTRUMENT_INDICES:
            assert 0 <= idx < 527, f"INSTRUMENT index {idx} out of range"

    def test_music_bgm_indices_valid(self):
        for idx in MUSIC_BGM_INDICES:
            assert 0 <= idx < 527, f"BGM index {idx} out of range"

    def test_music_singing_index_valid(self):
        assert 0 <= MUSIC_SINGING_INDEX < 527

    def test_music_all_indices_valid(self):
        for idx in MUSIC_ALL_INDICES:
            assert 0 <= idx < 527, f"MUSIC_ALL index {idx} out of range"


# =========================================================================
# Helper functions
# =========================================================================

class TestNoisyOr:
    """Test _noisy_or function directly."""

    def test_all_zero(self):
        scores = _make_scores({})
        result = _noisy_or(scores, MEDIA_INDICES)
        assert result == pytest.approx(0.0)

    def test_single_high(self):
        scores = _make_scores({524: 0.9})
        result = _noisy_or(scores, MEDIA_INDICES)
        assert result == pytest.approx(0.9)

    def test_multiple_high(self):
        """Multiple high scores should combine to near 1.0."""
        scores = _make_scores({137: 0.87, 524: 0.74, 525: 0.5})
        result = _noisy_or(scores, MEDIA_INDICES)
        # 1 - (0.13 * 0.26 * 0.5) = 1 - 0.0169 = 0.9831
        assert result > 0.95


class TestMaxScoreFor:
    """Test _max_score_for helper."""

    def test_basic(self):
        scores = _make_scores({10: 0.5, 20: 0.8, 30: 0.3})
        assert _max_score_for(scores, [10, 20, 30]) == pytest.approx(0.8)

    def test_empty_indices(self):
        scores = _make_scores({10: 0.5})
        assert _max_score_for(scores, []) == 0.0

    def test_out_of_range_indices(self):
        scores = _make_scores({10: 0.5})
        assert _max_score_for(scores, [10, 999]) == pytest.approx(0.5)


# =========================================================================
# Scene detection (backward compatible)
# =========================================================================

class TestInterpretScores:
    """Test _interpret_scores with mock numpy arrays."""

    def test_telephone_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({389: 0.85, 506: 0.72})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["scene_label"] == "電話"
        assert result["scene_label_en"] == "telephone"
        assert result["scene_confidence"] == 0.85
        assert result["spatial"] == "室内（小）"
        assert result["spatial_en"] == "indoor_small"
        assert result["model"] == "PANNs-CNN14"

    def test_media_detected(self):
        """Media with speech present — Noisy OR should detect media."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        # Speech high + multiple media indicators high → Noisy OR >> 0.6
        scores = _make_scores({0: 0.5, 524: 0.9, 507: 0.6})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "TV・ラジオ"
        assert result["scene_label_en"] == "media"

    def test_conversation_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.7, 4: 0.5})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "会話"
        assert result["scene_label_en"] == "conversation"

    def test_noise_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({513: 0.6})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "騒音環境"
        assert result["scene_label_en"] == "noise"

    def test_outdoor_nature_spatial(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 510: 0.8})
        result = classifier._interpret_scores(scores)

        assert result["spatial"] == "屋外（自然）"
        assert result["spatial_en"] == "outdoor_nature"
        assert result["spatial_confidence"] == 0.8

    def test_below_threshold_returns_none(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.5)
        scores = _make_scores({389: 0.2, 506: 0.1})
        result = classifier._interpret_scores(scores)

        assert result is None

    def test_scene_below_threshold_spatial_above(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.5)
        scores = _make_scores({389: 0.2, 509: 0.7})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["scene_label"] is None
        assert result["spatial"] == "屋外（都市）"

    def test_top_classes_present(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.8, 389: 0.5, 506: 0.4})
        result = classifier._interpret_scores(scores)

        assert "top_classes" in result
        assert len(result["top_classes"]) == 10
        # Highest score should be first
        assert result["top_classes"][0]["score"] == 0.8

    def test_classified_at_present(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5})
        result = classifier._interpret_scores(scores)

        assert "classified_at" in result
        assert isinstance(result["classified_at"], str)

    def test_highest_scene_wins(self):
        """Telephone has priority over media even when media score is higher."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        # telephone=0.4 (above threshold) → telephone wins by priority
        scores = _make_scores({389: 0.4, 0: 0.5, 524: 0.9})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "telephone"

    def test_silence_when_speech_low(self):
        """Silence detected when speech is below threshold."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({500: 0.6, 0: 0.1})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "静寂"
        assert result["scene_label_en"] == "silence"


class TestSpatialLabelKey:
    """Test spatial_label output key (backward compat alias)."""

    def test_spatial_label_present(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 506: 0.7})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["spatial_label"] == "室内（小）"
        # Both keys should have the same value
        assert result["spatial"] == result["spatial_label"]

    def test_spatial_label_none_when_below_threshold(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.5)
        scores = _make_scores({0: 0.6})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["spatial_label"] is None
        assert result["spatial"] is None


# =========================================================================
# Composite media detection (backward compatible)
# =========================================================================

class TestCompositeMediaDetection:
    """Test Noisy OR media detection with realistic score patterns."""

    def test_anime_high(self):
        """Strong anime: Speech=0.88, Music=0.87, TV=0.74 → media."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.88, 137: 0.87, 524: 0.74})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"
        assert result["scene_confidence"] > 0.9

    def test_anime_moderate(self):
        """Moderate anime: Speech=0.79, Music=0.76, TV=0.69 → media."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.79, 137: 0.76, 524: 0.69})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"

    def test_real_conversation(self):
        """Real conversation: Speech=0.84, Music=0.08, TV=0.03 → conversation."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.84, 137: 0.08, 524: 0.03})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "conversation"

    def test_music_dominant(self):
        """BGM dominant: Speech=0.81, Music=0.86, TV=0.69 → media."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.81, 137: 0.86, 524: 0.69, 270: 0.5})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"

    def test_tv_and_music(self):
        """TV + Music + Radio all high → media (high Noisy OR)."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.7, 137: 0.6, 524: 0.7, 525: 0.5})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"
        assert result["scene_confidence"] > 0.9

    def test_high_speech_low_companions(self):
        """Speech=0.82, Music=0.27, TV=0.08 → conversation."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({0: 0.82, 137: 0.27, 524: 0.08})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "conversation"

    def test_telephone_priority(self):
        """Telephone takes priority even when media Noisy OR is high."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({389: 0.5, 0: 0.8, 137: 0.9, 524: 0.9})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "telephone"

    def test_soundtrack(self):
        """Soundtrack + Singing high, Speech low → media without speech."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({270: 0.7, 27: 0.6, 0: 0.1})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"


# =========================================================================
# Noise source detection
# =========================================================================

class TestNoiseSources:
    """Test noise source detection dimension."""

    def test_vehicle_traffic_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 327: 0.45})  # Traffic noise
        result = classifier._interpret_scores(scores)

        noise = result["noise_sources"]
        assert len(noise) >= 1
        types = [n["type"] for n in noise]
        assert "vehicle_traffic" in types

    def test_multiple_noise_sources(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 327: 0.5, 413: 0.35, 384: 0.3})
        result = classifier._interpret_scores(scores)

        noise = result["noise_sources"]
        types = [n["type"] for n in noise]
        assert "vehicle_traffic" in types
        assert "air_conditioning" in types
        assert "typing_keyboard" in types

    def test_sorted_by_confidence_desc(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 327: 0.3, 413: 0.6, 283: 0.45})
        result = classifier._interpret_scores(scores)

        noise = result["noise_sources"]
        confs = [n["confidence"] for n in noise]
        assert confs == sorted(confs, reverse=True)

    def test_below_threshold_excluded(self):
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.3,
            dimension_thresholds={"noise_source": 0.4},
        )
        scores = _make_scores({0: 0.5, 327: 0.35})  # Below 0.4
        result = classifier._interpret_scores(scores)

        assert result["noise_sources"] == []

    def test_empty_when_no_noise(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5})  # Only speech
        result = classifier._interpret_scores(scores)

        assert result["noise_sources"] == []

    def test_noise_source_structure(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 289: 0.4})  # Rain
        result = classifier._interpret_scores(scores)

        noise = result["noise_sources"]
        assert len(noise) >= 1
        item = [n for n in noise if n["type"] == "rain"][0]
        assert "type" in item
        assert "label" in item
        assert "confidence" in item
        assert item["label"] == "雨"


# =========================================================================
# Event detection
# =========================================================================

class TestEvents:
    """Test event detection dimension."""

    def test_laughter_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 16: 0.7})  # Laughter
        result = classifier._interpret_scores(scores)

        events = result["events"]
        types = [e["type"] for e in events]
        assert "laughter" in types

    def test_multiple_events(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 16: 0.8, 67: 0.6})  # Laughter + Applause
        result = classifier._interpret_scores(scores)

        events = result["events"]
        types = [e["type"] for e in events]
        assert "laughter" in types
        assert "applause" in types

    def test_crying_via_baby_cry(self):
        """Baby cry (index 23) should trigger crying category."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 23: 0.5})
        result = classifier._interpret_scores(scores)

        types = [e["type"] for e in result["events"]]
        assert "crying" in types

    def test_cough_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 47: 0.4})
        result = classifier._interpret_scores(scores)

        types = [e["type"] for e in result["events"]]
        assert "cough" in types

    def test_below_threshold_excluded(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 16: 0.05})  # Laughter well below 0.10 default
        result = classifier._interpret_scores(scores)

        assert result["events"] == []

    def test_sorted_by_confidence(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 16: 0.4, 67: 0.9, 49: 0.35})
        result = classifier._interpret_scores(scores)

        confs = [e["confidence"] for e in result["events"]]
        assert confs == sorted(confs, reverse=True)


# =========================================================================
# Music detection
# =========================================================================

class TestMusicDetection:
    """Test music detection and type classification."""

    def test_no_music(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.8, 4: 0.5})  # Just speech
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is False
        assert result["music"]["type"] is None

    def test_recorded_music(self):
        """BGM indices high → recorded type."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 137: 0.7, 267: 0.5, 524: 0.6})
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is True
        assert result["music"]["type"] == "recorded"
        assert result["music"]["type_label"] == "録音/BGM"

    def test_live_music(self):
        """Instrument indices high, BGM low → live type."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 137: 0.7, 153: 0.6, 191: 0.5})  # Piano, Violin
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is True
        assert result["music"]["type"] == "live"
        assert result["music"]["type_label"] == "生演奏"

    def test_singing(self):
        """Singing index high, instruments/BGM low → singing type."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 27: 0.6, 137: 0.5})
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is True
        assert result["music"]["type"] == "singing"
        assert result["music"]["type_label"] == "歌唱"

    def test_music_confidence(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 137: 0.8, 524: 0.7})
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is True
        assert result["music"]["confidence"] > 0.5

    def test_music_below_detection_threshold(self):
        """Low music scores → not detected even if some type signal exists."""
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.3,
            dimension_thresholds={"music_detection": 0.8},
        )
        scores = _make_scores({0: 0.5, 137: 0.3, 267: 0.25})
        result = classifier._interpret_scores(scores)

        assert result["music"]["detected"] is False


# =========================================================================
# Speaker count estimation
# =========================================================================

class TestSpeakerCount:
    """Test speaker count estimation."""

    def test_monologue(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 5: 0.6, 4: 0.2})  # Narration high, Conversation low
        result = classifier._interpret_scores(scores)

        assert result["speaker_count"]["estimate"] == "monologue"
        assert result["speaker_count"]["label"] == "独り言"

    def test_conversation(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 4: 0.7, 5: 0.3})  # Conversation high
        result = classifier._interpret_scores(scores)

        assert result["speaker_count"]["estimate"] == "conversation"

    def test_crowd(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 69: 0.8, 70: 0.6, 4: 0.3})  # Crowd high
        result = classifier._interpret_scores(scores)

        assert result["speaker_count"]["estimate"] == "crowd"
        assert result["speaker_count"]["label"] == "群衆"

    def test_below_threshold(self):
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.3,
            dimension_thresholds={"speaker_count": 0.5},
        )
        scores = _make_scores({0: 0.5, 4: 0.3, 5: 0.2})  # All below 0.5
        result = classifier._interpret_scores(scores)

        assert result["speaker_count"]["estimate"] is None

    def test_mutually_exclusive(self):
        """Highest score wins among the three categories."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 5: 0.4, 4: 0.6, 69: 0.55})
        result = classifier._interpret_scores(scores)

        assert result["speaker_count"]["estimate"] == "conversation"
        assert result["speaker_count"]["confidence"] == pytest.approx(0.6, abs=0.01)


# =========================================================================
# Quality issues
# =========================================================================

class TestQualityIssues:
    """Test quality/anomaly detection."""

    def test_echo_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 512: 0.35})  # Echo
        result = classifier._interpret_scores(scores)

        types = [q["type"] for q in result["quality_issues"]]
        assert "echo" in types

    def test_multiple_issues(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 517: 0.4, 515: 0.3, 511: 0.5})
        result = classifier._interpret_scores(scores)

        types = [q["type"] for q in result["quality_issues"]]
        assert "distortion" in types
        assert "static" in types
        assert "reverberation" in types

    def test_sorted_by_confidence(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 517: 0.3, 511: 0.6, 512: 0.4})
        result = classifier._interpret_scores(scores)

        confs = [q["confidence"] for q in result["quality_issues"]]
        assert confs == sorted(confs, reverse=True)

    def test_below_threshold_excluded(self):
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.3,
            dimension_thresholds={"quality_issue": 0.5},
        )
        scores = _make_scores({0: 0.5, 512: 0.3})  # Echo below 0.5
        result = classifier._interpret_scores(scores)

        assert result["quality_issues"] == []

    def test_wind_noise_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5, 285: 0.4})
        result = classifier._interpret_scores(scores)

        types = [q["type"] for q in result["quality_issues"]]
        assert "wind_noise" in types


# =========================================================================
# Clipping detection
# =========================================================================

class TestClipping:
    """Test waveform-based clipping detection."""

    def test_clipping_detected_high_ratio(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        # Simulate audio_meta with high clipping
        scores = _make_scores({0: 0.5})
        result = classifier._interpret_scores(scores)
        # Manually attach clipping (as classify() does)
        result["clipping_detected"] = 0.02 > classifier._dim_thresholds["clipping_ratio"]
        assert result["clipping_detected"] is True

    def test_clipping_not_detected_low_ratio(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.5})
        result = classifier._interpret_scores(scores)
        result["clipping_detected"] = 0.001 > classifier._dim_thresholds["clipping_ratio"]
        assert result["clipping_detected"] is False


# =========================================================================
# Backward compatibility
# =========================================================================

class TestBackwardCompatibility:
    """Ensure existing output fields are unchanged."""

    def test_all_existing_keys_present(self):
        """All original output keys must still be present."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.7, 4: 0.5, 506: 0.6})
        result = classifier._interpret_scores(scores)

        expected_keys = {
            "model", "scene_label", "scene_label_en", "scene_confidence",
            "spatial", "spatial_label", "spatial_en", "spatial_confidence",
            "top_classes", "classified_at",
        }
        assert expected_keys.issubset(result.keys())

    def test_new_keys_present(self):
        """New dimension keys must be present."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({0: 0.7, 4: 0.5, 506: 0.6})
        result = classifier._interpret_scores(scores)

        new_keys = {"noise_sources", "events", "music", "speaker_count", "quality_issues"}
        assert new_keys.issubset(result.keys())

    def test_existing_scene_values_unchanged(self):
        """Verify that scene detection logic produces same results."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)

        # Conversation
        scores = _make_scores({0: 0.84, 137: 0.08, 524: 0.03})
        result = classifier._interpret_scores(scores)
        assert result["scene_label_en"] == "conversation"
        assert result["scene_label"] == "会話"

        # Media
        scores = _make_scores({0: 0.88, 137: 0.87, 524: 0.74})
        result = classifier._interpret_scores(scores)
        assert result["scene_label_en"] == "media"

    def test_result_returns_none_when_all_below_threshold(self):
        """Still returns None when nothing is above threshold."""
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.5,
            dimension_thresholds={
                "noise_source": 0.5, "event": 0.5,
                "music_detection": 0.9, "speaker_count": 0.5,
                "quality_issue": 0.5,
            },
        )
        scores = _make_scores({389: 0.2, 506: 0.1})
        result = classifier._interpret_scores(scores)
        assert result is None

    def test_dimension_thresholds_customizable(self):
        """Custom dimension thresholds override defaults."""
        custom = {"noise_source": 0.99, "event": 0.99}
        classifier = AudioSceneClassifier(
            enabled=True, threshold=0.3,
            dimension_thresholds=custom,
        )
        assert classifier._dim_thresholds["noise_source"] == 0.99
        assert classifier._dim_thresholds["event"] == 0.99
        # Non-overridden keep defaults
        assert classifier._dim_thresholds["music_detection"] == DEFAULT_DIMENSION_THRESHOLDS["music_detection"]


# =========================================================================
# Disabled state & classify integration
# =========================================================================

class TestDisabledState:
    """Test behavior when classifier is disabled."""

    def test_classify_returns_none_when_disabled(self):
        classifier = AudioSceneClassifier(enabled=False)
        result = classifier.classify("/some/path/audio.wav")
        assert result is None

    def test_enabled_property(self):
        c1 = AudioSceneClassifier(enabled=True)
        assert c1.enabled is True

        c2 = AudioSceneClassifier(enabled=False)
        assert c2.enabled is False


class TestClassifyIntegration:
    """Test classify method with mocked model."""

    def test_classify_missing_file(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        result = classifier.classify("/nonexistent/audio.wav")
        assert result is None

    @patch("audio_scene_classifier.os.path.exists", return_value=True)
    @patch("audio_scene_classifier.AudioSceneClassifier._load_model")
    @patch("audio_scene_classifier.AudioSceneClassifier._load_audio")
    def test_classify_with_mocked_model(self, mock_load_audio, mock_load_model, mock_exists):
        scores = np.zeros(527, dtype=np.float32)
        scores[0] = 0.8  # Speech
        scores[506] = 0.6  # Indoor small

        mock_model = MagicMock()
        mock_model.inference.return_value = (np.array([scores]), None)
        mock_load_model.return_value = mock_model
        mock_load_audio.return_value = (np.zeros(32000, dtype=np.float32), {"clipping_ratio": 0.0})

        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        result = classifier.classify("/fake/audio.wav")

        assert result is not None
        assert result["scene_label"] == "会話"
        assert result["spatial"] == "室内（小）"
        # New fields present
        assert "noise_sources" in result
        assert "events" in result
        assert "music" in result
        assert "speaker_count" in result
        assert "quality_issues" in result
        assert "clipping_detected" in result
        assert result["clipping_detected"] is False

    @patch("audio_scene_classifier.os.path.exists", return_value=True)
    @patch("audio_scene_classifier.AudioSceneClassifier._load_model")
    @patch("audio_scene_classifier.AudioSceneClassifier._load_audio")
    def test_clipping_detected_via_classify(self, mock_load_audio, mock_load_model, mock_exists):
        scores = np.zeros(527, dtype=np.float32)
        scores[0] = 0.8

        mock_model = MagicMock()
        mock_model.inference.return_value = (np.array([scores]), None)
        mock_load_model.return_value = mock_model
        # High clipping ratio
        mock_load_audio.return_value = (np.zeros(32000, dtype=np.float32), {"clipping_ratio": 0.02})

        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        result = classifier.classify("/fake/audio.wav")

        assert result is not None
        assert result["clipping_detected"] is True

    @patch("audio_scene_classifier.AudioSceneClassifier._load_model")
    def test_classify_model_load_failure(self, mock_load_model):
        mock_load_model.return_value = None
        classifier = AudioSceneClassifier(enabled=True)
        result = classifier.classify("/fake/audio.wav")
        assert result is None


class TestUnload:
    """Test model unload."""

    def test_unload_clears_model(self):
        classifier = AudioSceneClassifier(enabled=True)
        classifier._model = MagicMock()
        classifier.unload()
        assert classifier._model is None

    def test_unload_when_no_model(self):
        classifier = AudioSceneClassifier(enabled=True)
        classifier.unload()  # Should not raise
        assert classifier._model is None


# =========================================================================
# Realistic composite scenarios
# =========================================================================

class TestRealisticScenarios:
    """Test realistic multi-dimensional detection scenarios."""

    def test_meeting_room(self):
        """Indoor meeting: conversation + typing + AC."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({
            0: 0.85, 4: 0.7,  # Speech, Conversation
            506: 0.65,  # Indoor small
            413: 0.4,  # Air conditioning
            384: 0.35,  # Typing
        })
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "conversation"
        assert result["spatial_en"] == "indoor_small"
        assert result["speaker_count"]["estimate"] == "conversation"
        noise_types = [n["type"] for n in result["noise_sources"]]
        assert "air_conditioning" in noise_types
        assert "typing_keyboard" in noise_types

    def test_outdoor_event(self):
        """Outdoor event: crowd + applause + music."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({
            0: 0.6, 69: 0.7, 70: 0.5,  # Speech, Crowd, Hubbub
            509: 0.55,  # Outdoor urban
            67: 0.8,  # Applause
            137: 0.6, 153: 0.5,  # Music, Piano
        })
        result = classifier._interpret_scores(scores)

        assert result["spatial_en"] == "outdoor_urban"
        assert result["speaker_count"]["estimate"] == "crowd"
        event_types = [e["type"] for e in result["events"]]
        assert "applause" in event_types
        assert result["music"]["detected"] is True

    def test_poor_quality_recording(self):
        """Recording with quality issues: distortion + hum + echo."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = _make_scores({
            0: 0.5,
            517: 0.45,  # Distortion
            516: 0.35,  # Mains hum
            512: 0.3,   # Echo
        })
        result = classifier._interpret_scores(scores)

        quality_types = [q["type"] for q in result["quality_issues"]]
        assert "distortion" in quality_types
        assert "mains_hum" in quality_types
        assert "echo" in quality_types

    def test_anime_session_full(self):
        """Anime: media scene + music detected + laughter event."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3, media_threshold=0.6)
        scores = _make_scores({
            0: 0.88, 137: 0.87, 524: 0.74,  # Speech, Music, TV
            267: 0.5,  # Background music
            16: 0.4,   # Laughter
            506: 0.5,  # Indoor small
        })
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"
        assert result["music"]["detected"] is True
        assert result["music"]["type"] == "recorded"
        event_types = [e["type"] for e in result["events"]]
        assert "laughter" in event_types
