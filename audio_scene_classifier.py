"""Audio scene classifier using PANNs CNN14.

Classifies recording sessions into scene categories (telephone, media, conversation,
silence, noise) and spatial categories (indoor, outdoor, public) using AudioSet
pretrained CNN14 model (~80MB, MIT license).

Uses Noisy OR probability combination for media detection — since PANNs uses
sigmoid outputs (independent probabilities per class), Noisy OR is the
theoretically correct way to combine them (Kong et al. 2020).

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

# Scene categories (used for priority-based detection, not max-score-wins)
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

# Media companion indices — classes that fire high for TV/anime but low for
# real conversation. Used with Noisy OR to detect media presence.
# 137=Music, 524=Television, 525=Radio, 267=Background music,
# 268=Theme music, 270=Soundtrack, 27=Singing
MEDIA_INDICES = [137, 524, 525, 267, 268, 270, 27]

# Default Noisy OR threshold for media detection
DEFAULT_MEDIA_NOISY_OR_THRESHOLD = 0.6

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
        if isinstance(labels, list):
            _AUDIOSET_LABELS = labels
        elif hasattr(labels, "labels"):
            _AUDIOSET_LABELS = labels.labels
        else:
            _AUDIOSET_LABELS = []
    except Exception:
        _AUDIOSET_LABELS = []
    return _AUDIOSET_LABELS


def _noisy_or(scores: np.ndarray, indices: list) -> float:
    """Noisy OR combination of independent sigmoid probabilities.

    P(any) = 1 - prod(1 - P(i))

    This is the theoretically correct way to combine independent sigmoid
    outputs from PANNs (BCE loss → each class is an independent probability).
    """
    prob_none = 1.0
    for i in indices:
        if i < len(scores):
            prob_none *= (1.0 - float(scores[i]))
    return 1.0 - prob_none


class AudioSceneClassifier:
    """Classify audio scenes using PANNs CNN14."""

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.3,
        media_threshold: float = DEFAULT_MEDIA_NOISY_OR_THRESHOLD,
        enabled: bool = True,
    ) -> None:
        self._device = device
        self._threshold = threshold
        self._media_threshold = media_threshold
        self._enabled = enabled
        self._model = None
        if enabled:
            log.info(
                "AudioSceneClassifier: initialized (device=%s, threshold=%.2f, media_threshold=%.2f)",
                device, threshold, media_threshold,
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
            # Ensure panns_data directory exists before import (panns_inference
            # downloads class_labels_indices.csv to ~/panns_data/ on import)
            from pathlib import Path
            panns_dir = Path.home() / "panns_data"
            panns_dir.mkdir(parents=True, exist_ok=True)
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
    def _load_audio(audio_path: str, target_sr: int = 32000, max_seconds: int = 60) -> Optional[np.ndarray]:
        """Load audio file using torchaudio, resample to target_sr, return numpy.

        Truncates to max_seconds to avoid excessive CPU time on long recordings.
        Scene classification only needs a representative sample, not the full audio.
        """
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Truncate to max_seconds before resampling (cheaper)
            max_samples = sr * max_seconds
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
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

        Priority-based detection with Noisy OR for media:
        1. Telephone: max(telephone indices) >= threshold
        2. Silence: silence >= threshold AND speech < threshold
        3. Media vs Conversation (when speech >= threshold):
           - media_prob = noisy_or(MEDIA_INDICES)
           - media_prob >= media_threshold → TV・ラジオ
           - otherwise → 会話
        4. Media (speech < threshold): noisy_or >= threshold
        5. Noise: max(noise indices) >= threshold
        """
        threshold = self._threshold

        # --- Helper: max score for a category ---
        def _max_score(indices):
            return max(
                (float(scores[i]) for i in indices if i < len(scores)),
                default=0.0,
            )

        # --- Priority-based scene detection ---
        best_scene_key = None
        best_scene_score = 0.0

        # 1. Telephone (highest priority)
        tel_score = _max_score(SCENE_CATEGORIES["telephone"]["indices"])
        if tel_score >= threshold:
            best_scene_key = "telephone"
            best_scene_score = tel_score
        else:
            speech_score = _max_score(SCENE_CATEGORIES["conversation"]["indices"])
            silence_score = _max_score(SCENE_CATEGORIES["silence"]["indices"])

            # 2. Silence (speech must be low)
            if silence_score >= threshold and speech_score < threshold:
                best_scene_key = "silence"
                best_scene_score = silence_score
            elif speech_score >= threshold:
                # 3. Media vs Conversation — Noisy OR decides
                media_prob = _noisy_or(scores, MEDIA_INDICES)
                if media_prob >= self._media_threshold:
                    best_scene_key = "media"
                    best_scene_score = media_prob
                else:
                    best_scene_key = "conversation"
                    best_scene_score = speech_score
            else:
                # 4. Media without speech
                media_prob = _noisy_or(scores, MEDIA_INDICES)
                if media_prob >= threshold:
                    best_scene_key = "media"
                    best_scene_score = media_prob
                else:
                    # 5. Noise
                    noise_score = _max_score(SCENE_CATEGORIES["noise"]["indices"])
                    if noise_score >= threshold:
                        best_scene_key = "noise"
                        best_scene_score = noise_score

        # --- Spatial category (unchanged: highest-scoring wins) ---
        best_spatial_key = None
        best_spatial_score = 0.0

        for key, cat in SPATIAL_CATEGORIES.items():
            idx = cat["index"]
            if idx < len(scores):
                score = float(scores[idx])
                if score > best_spatial_score:
                    best_spatial_score = score
                    best_spatial_key = key

        if best_spatial_score < threshold:
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

        spatial_label = SPATIAL_CATEGORIES[best_spatial_key]["label"] if best_spatial_key else None

        result = {
            "model": "PANNs-CNN14",
            "scene_label": SCENE_CATEGORIES[best_scene_key]["label"] if best_scene_key else None,
            "scene_label_en": best_scene_key,
            "scene_confidence": round(best_scene_score, 3) if best_scene_key else None,
            "spatial": spatial_label,
            "spatial_label": spatial_label,  # alias for indexer.py / session_tagger.py
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
