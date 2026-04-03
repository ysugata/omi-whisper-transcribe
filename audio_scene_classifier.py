"""Audio scene classifier using PANNs CNN14.

Classifies recording sessions into scene categories (telephone, media, conversation,
silence, noise) and spatial categories (indoor, outdoor, public) using AudioSet
pretrained CNN14 model (~80MB, MIT license).

Runs on CPU only — no GPU lock required.
"""

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AudioSet class index → category mapping
# ---------------------------------------------------------------------------

# Scene categories: highest-scoring category wins
SCENE_CATEGORIES = {
    "telephone": {
        "indices": [389, 390, 391, 392, 393],
        "label": "電話",
    },
    "media": {
        "indices": [137, 524, 525],
        "label": "TV・ラジオ",
    },
    "conversation": {
        "indices": [0, 4],
        "label": "会話",
    },
    "silence": {
        "indices": [500],
        "label": "静寂",
    },
    "noise": {
        "indices": [513, 520],
        "label": "騒音環境",
    },
}

# Spatial categories: highest-scoring spatial wins
SPATIAL_CATEGORIES = {
    "indoor_small": {"index": 506, "label": "室内（小）"},
    "indoor_large": {"index": 507, "label": "室内（大）"},
    "public_space": {"index": 508, "label": "公共空間"},
    "outdoor_urban": {"index": 509, "label": "屋外（都市）"},
    "outdoor_nature": {"index": 510, "label": "屋外（自然）"},
}

# AudioSet class labels (only the ones we use — loaded on demand for top_classes)
_AUDIOSET_LABELS: Optional[list] = None


def _get_audioset_labels() -> list:
    """Lazy-load AudioSet class labels from panns_inference metadata."""
    global _AUDIOSET_LABELS
    if _AUDIOSET_LABELS is not None:
        return _AUDIOSET_LABELS
    try:
        from panns_inference import labels
        _AUDIOSET_LABELS = labels.labels if hasattr(labels, "labels") else []
    except Exception:
        _AUDIOSET_LABELS = []
    return _AUDIOSET_LABELS


class AudioSceneClassifier:
    """Classify audio scenes using PANNs CNN14."""

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self._device = device
        self._threshold = threshold
        self._enabled = enabled
        self._model = None
        if enabled:
            log.info(
                "AudioSceneClassifier: initialized (device=%s, threshold=%.2f)",
                device, threshold,
            )
        else:
            log.info("AudioSceneClassifier: disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load_model(self):
        """Load PANNs CNN14 model (lazy, first call only)."""
        if self._model is not None:
            return self._model
        try:
            from panns_inference import AudioTagging
            self._model = AudioTagging(
                checkpoint_path=None,  # auto-download CNN14
                device=self._device,
            )
            log.info("AudioSceneClassifier: CNN14 model loaded (device=%s)", self._device)
            return self._model
        except Exception as e:
            log.error("AudioSceneClassifier: failed to load model: %s", e)
            return None

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            self._model = None
            log.info("AudioSceneClassifier: model unloaded")

    def classify(self, audio_path: str) -> Optional[dict]:
        """Classify an audio file and return scene info dict.

        Returns None if disabled, model fails to load, or audio can't be read.
        """
        if not self._enabled:
            return None

        if not os.path.exists(audio_path):
            log.warning("AudioSceneClassifier: audio not found: %s", audio_path)
            return None

        model = self._load_model()
        if model is None:
            return None

        try:
            waveform = self._load_audio(audio_path)
            if waveform is None:
                return None

            # PANNs expects (batch, samples) shape
            if waveform.ndim == 1:
                waveform = waveform[None, :]

            clipwise_output, _ = model.inference(waveform)
            scores = clipwise_output[0]  # (527,) array

            return self._interpret_scores(scores)

        except Exception as e:
            log.error("AudioSceneClassifier: classify error for %s: %s", audio_path, e)
            return None

    @staticmethod
    def _load_audio(audio_path: str, target_sr: int = 32000) -> Optional[np.ndarray]:
        """Load audio file using torchaudio, resample to target_sr, return numpy."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy()
        except Exception as e:
            log.warning("AudioSceneClassifier: audio load error: %s", e)
            return None

    def _interpret_scores(self, scores: np.ndarray) -> Optional[dict]:
        """Interpret PANNs output scores into scene classification.

        Args:
            scores: numpy array of shape (527,) with class probabilities.

        Returns:
            Scene classification dict or None if all scores below threshold.
        """
        # --- Scene category ---
        best_scene_key = None
        best_scene_score = 0.0

        for key, cat in SCENE_CATEGORIES.items():
            cat_scores = [float(scores[i]) for i in cat["indices"] if i < len(scores)]
            if cat_scores:
                max_score = max(cat_scores)
                if max_score > best_scene_score:
                    best_scene_score = max_score
                    best_scene_key = key

        # --- Spatial category ---
        best_spatial_key = None
        best_spatial_score = 0.0

        for key, cat in SPATIAL_CATEGORIES.items():
            idx = cat["index"]
            if idx < len(scores):
                score = float(scores[idx])
                if score > best_spatial_score:
                    best_spatial_score = score
                    best_spatial_key = key

        # If best scene score is below threshold, return unknown
        if best_scene_score < self._threshold:
            best_scene_key = None

        if best_spatial_score < self._threshold:
            best_spatial_key = None

        if best_scene_key is None and best_spatial_key is None:
            return None

        # --- Top classes (for debugging / detail view) ---
        top_indices = np.argsort(scores)[::-1][:5]
        audioset_labels = _get_audioset_labels()
        top_classes = []
        for idx in top_indices:
            idx = int(idx)
            label = audioset_labels[idx] if idx < len(audioset_labels) else f"class_{idx}"
            top_classes.append({
                "label": label,
                "index": idx,
                "score": round(float(scores[idx]), 3),
            })

        result = {
            "model": "PANNs-CNN14",
            "scene_label": SCENE_CATEGORIES[best_scene_key]["label"] if best_scene_key else None,
            "scene_label_en": best_scene_key,
            "scene_confidence": round(best_scene_score, 3) if best_scene_key else None,
            "spatial": SPATIAL_CATEGORIES[best_spatial_key]["label"] if best_spatial_key else None,
            "spatial_en": best_spatial_key,
            "spatial_confidence": round(best_spatial_score, 3) if best_spatial_key else None,
            "top_classes": top_classes,
            "classified_at": datetime.now().isoformat(),
        }

        log.info(
            "AudioSceneClassifier: scene=%s (%.2f), spatial=%s (%.2f)",
            result["scene_label"] or "unknown",
            best_scene_score,
            result["spatial"] or "unknown",
            best_spatial_score,
        )

        return result
