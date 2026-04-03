"""Tests for AudioSceneClassifier."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from audio_scene_classifier import (
    AudioSceneClassifier,
    SCENE_CATEGORIES,
    SPATIAL_CATEGORIES,
)


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


class TestInterpretScores:
    """Test _interpret_scores with mock numpy arrays."""

    def _make_scores(self, overrides: dict) -> np.ndarray:
        """Create a 527-element scores array with specific indices set."""
        scores = np.zeros(527, dtype=np.float32)
        for idx, val in overrides.items():
            scores[idx] = val
        return scores

    def test_telephone_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({389: 0.85, 506: 0.72})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["scene_label"] == "電話"
        assert result["scene_label_en"] == "telephone"
        assert result["scene_confidence"] == 0.85
        assert result["spatial"] == "室内（小）"
        assert result["spatial_en"] == "indoor_small"
        assert result["model"] == "PANNs-CNN14"

    def test_media_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({524: 0.9, 507: 0.6})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "TV・ラジオ"
        assert result["scene_label_en"] == "media"

    def test_conversation_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({0: 0.7, 4: 0.5})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "会話"
        assert result["scene_label_en"] == "conversation"

    def test_noise_detected(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({513: 0.6})
        result = classifier._interpret_scores(scores)

        assert result["scene_label"] == "騒音環境"
        assert result["scene_label_en"] == "noise"

    def test_outdoor_nature_spatial(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({0: 0.5, 510: 0.8})
        result = classifier._interpret_scores(scores)

        assert result["spatial"] == "屋外（自然）"
        assert result["spatial_en"] == "outdoor_nature"
        assert result["spatial_confidence"] == 0.8

    def test_below_threshold_returns_none(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.5)
        scores = self._make_scores({389: 0.2, 506: 0.1})
        result = classifier._interpret_scores(scores)

        assert result is None

    def test_scene_below_threshold_spatial_above(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.5)
        scores = self._make_scores({389: 0.2, 509: 0.7})
        result = classifier._interpret_scores(scores)

        assert result is not None
        assert result["scene_label"] is None
        assert result["spatial"] == "屋外（都市）"

    def test_top_classes_present(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({0: 0.8, 389: 0.5, 506: 0.4})
        result = classifier._interpret_scores(scores)

        assert "top_classes" in result
        assert len(result["top_classes"]) == 5
        # Highest score should be first
        assert result["top_classes"][0]["score"] == 0.8

    def test_classified_at_present(self):
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        scores = self._make_scores({0: 0.5})
        result = classifier._interpret_scores(scores)

        assert "classified_at" in result
        assert isinstance(result["classified_at"], str)

    def test_highest_scene_wins(self):
        """When multiple scenes have high scores, the highest wins."""
        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        # telephone=0.4, media=0.9 -> media wins
        scores = self._make_scores({389: 0.4, 524: 0.9})
        result = classifier._interpret_scores(scores)

        assert result["scene_label_en"] == "media"


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
        mock_load_audio.return_value = np.zeros(32000, dtype=np.float32)

        classifier = AudioSceneClassifier(enabled=True, threshold=0.3)
        result = classifier.classify("/fake/audio.wav")

        assert result is not None
        assert result["scene_label"] == "会話"
        assert result["spatial"] == "室内（小）"

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
