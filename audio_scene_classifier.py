"""Audio scene classifier using PANNs CNN14.

Classifies recording sessions into scene categories (telephone, media, conversation,
silence, noise) and spatial categories (indoor, outdoor, public) using AudioSet
pretrained CNN14 model (~80MB, MIT license).

Extracts 6 analysis dimensions from a single 527-class score vector:
1. Scene (telephone, media, conversation, silence, noise) — priority-based
2. Spatial (indoor small/large, public, outdoor urban/nature) — max-score
3. Noise sources (traffic, AC, fan, typing, wind, rain, etc.) — multi-label
4. Events (laughter, crying, cough, sneeze, applause, shout, sigh) — multi-label
5. Music (detected + live/recorded/singing type) — Noisy OR + max comparison
6. Speaker count (monologue, conversation, crowd) — max-score
7. Quality issues (distortion, static, hum, echo, reverb, wind noise) — multi-label
Plus waveform-based clipping detection.

Uses Noisy OR probability combination for media/music detection — since PANNs uses
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

# ---------------------------------------------------------------------------
# Extended dimension mappings (all from the same 527-class score vector)
# ---------------------------------------------------------------------------

# Noise source categories — multi-label, max-score per category
NOISE_SOURCE_CATEGORIES = {
    "vehicle_traffic": {
        "indices": [300, 306, 307, 316, 321, 326, 327],
        "label": "交通",
    },
    "air_conditioning": {
        "indices": [413],
        "label": "空調",
    },
    "mechanical_fan": {
        "indices": [412],
        "label": "ファン",
    },
    "typing_keyboard": {
        "indices": [384, 386],
        "label": "タイピング",
    },
    "wind": {
        "indices": [283, 285],
        "label": "風",
    },
    "rain": {
        "indices": [289, 290, 291],
        "label": "雨",
    },
    "construction": {
        "indices": [419, 420, 424, 425],
        "label": "工事",
    },
    "rustle": {
        "indices": [487],
        "label": "衣擦れ",
    },
    "siren_emergency": {
        "indices": [322, 323, 324, 325, 396],
        "label": "サイレン",
    },
}

# Event categories — multi-label, max-score per category
EVENT_CATEGORIES = {
    "laughter": {
        "indices": [16, 17, 18, 19, 20, 21],
        "label": "笑い",
    },
    "crying": {
        "indices": [22, 23, 24, 25],
        "label": "泣き声",
    },
    "cough": {
        "indices": [47, 48],
        "label": "咳",
    },
    "sneeze": {
        "indices": [49],
        "label": "くしゃみ",
    },
    "applause": {
        "indices": [66, 67],
        "label": "拍手・歓声",
    },
    "shout": {
        "indices": [8, 11, 14],
        "label": "叫び",
    },
    "sigh": {
        "indices": [26],
        "label": "ため息",
    },
}

# Music: instrument indices for live music detection (138-215)
MUSIC_INSTRUMENT_INDICES = list(range(138, 216))
# Background/recorded music indices
MUSIC_BGM_INDICES = [267, 268, 270]
# Singing index
MUSIC_SINGING_INDEX = 27
# All music-related indices for Noisy OR detection
MUSIC_ALL_INDICES = MEDIA_INDICES + MUSIC_INSTRUMENT_INDICES + [MUSIC_SINGING_INDEX]

# Speaker count categories — mutually exclusive, max-score wins
SPEAKER_COUNT_CATEGORIES = {
    "monologue": {
        "indices": [5],
        "label": "独り言",
    },
    "conversation": {
        "indices": [4],
        "label": "会話",
    },
    "crowd": {
        "indices": [68, 69, 70],
        "label": "群衆",
    },
}

# Quality/anomaly categories — multi-label, max-score per category
QUALITY_CATEGORIES = {
    "distortion": {
        "indices": [517],
        "label": "歪み",
    },
    "static": {
        "indices": [515],
        "label": "ノイズ",
    },
    "mains_hum": {
        "indices": [516],
        "label": "ハム",
    },
    "echo": {
        "indices": [512],
        "label": "エコー",
    },
    "reverberation": {
        "indices": [511],
        "label": "残響",
    },
    "wind_noise": {
        "indices": [285],
        "label": "風切り音",
    },
}

# Default thresholds per dimension
DEFAULT_DIMENSION_THRESHOLDS = {
    "noise_source": 0.25,
    "event": 0.30,
    "music_detection": 0.40,
    "music_type": 0.20,
    "speaker_count": 0.25,
    "quality_issue": 0.20,
    "clipping_ratio": 0.005,
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


def _max_score_for(scores: np.ndarray, indices: list) -> float:
    """Return max score across given AudioSet indices."""
    return max(
        (float(scores[i]) for i in indices if i < len(scores)),
        default=0.0,
    )


class AudioSceneClassifier:
    """Classify audio scenes using PANNs CNN14.

    Extracts 6+ analysis dimensions from a single inference call.
    """

    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.3,
        media_threshold: float = DEFAULT_MEDIA_NOISY_OR_THRESHOLD,
        dimension_thresholds: dict = None,
        enabled: bool = True,
    ) -> None:
        self._device = device
        self._threshold = threshold
        self._media_threshold = media_threshold
        self._dim_thresholds = {**DEFAULT_DIMENSION_THRESHOLDS}
        if dimension_thresholds:
            self._dim_thresholds.update(dimension_thresholds)
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
            audio_result = self._load_audio(audio_path)
            if audio_result is None:
                return None

            waveform, audio_meta = audio_result

            # PANNs expects (batch, samples) shape
            if waveform.ndim == 1:
                waveform = waveform[None, :]

            clipwise_output, _ = model.inference(waveform)
            scores = clipwise_output[0]  # (527,) array

            result = self._interpret_scores(scores)
            if result is not None:
                result["clipping_detected"] = (
                    audio_meta["clipping_ratio"] > self._dim_thresholds["clipping_ratio"]
                )
            return result

        except Exception as e:
            log.error("AudioSceneClassifier: classify error for %s: %s", audio_path, e)
            return None

    @staticmethod
    def _load_audio(
        audio_path: str, target_sr: int = 32000, max_seconds: int = 60,
    ) -> Optional[tuple]:
        """Load audio file using torchaudio, resample to target_sr, return numpy.

        Truncates to max_seconds to avoid excessive CPU time on long recordings.
        Normalizes loudness to -16 dBFS so that low-volume chest-mic recordings
        produce meaningful PANNs scores (trained on YouTube audio at higher levels).

        Returns (audio_array, metadata_dict) or None on error.
        metadata_dict contains clipping_ratio measured before normalization.
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
            audio = waveform.squeeze(0).numpy()

            # Measure clipping BEFORE loudness normalization
            clipping_ratio = float(np.mean(np.abs(audio) > 0.99))

            # Loudness normalization to -16 dBFS
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-10:
                target_rms = 10 ** (-16.0 / 20)
                audio = (audio * (target_rms / rms)).astype(np.float32)
                audio = np.clip(audio, -1.0, 1.0)

            meta = {"clipping_ratio": clipping_ratio}
            return audio, meta
        except Exception as e:
            log.warning("AudioSceneClassifier: audio load error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Dimension extractors (all operate on the same 527 scores vector)
    # ------------------------------------------------------------------

    def _detect_noise_sources(self, scores: np.ndarray) -> list:
        """Detect noise sources — multi-label, max-score per category."""
        threshold = self._dim_thresholds["noise_source"]
        detected = []
        for key, cat in NOISE_SOURCE_CATEGORIES.items():
            score = _max_score_for(scores, cat["indices"])
            if score >= threshold:
                detected.append({
                    "type": key,
                    "label": cat["label"],
                    "confidence": round(score, 3),
                })
        detected.sort(key=lambda x: x["confidence"], reverse=True)
        return detected

    def _detect_events(self, scores: np.ndarray) -> list:
        """Detect audio events — multi-label, max-score per category."""
        threshold = self._dim_thresholds["event"]
        detected = []
        for key, cat in EVENT_CATEGORIES.items():
            score = _max_score_for(scores, cat["indices"])
            if score >= threshold:
                detected.append({
                    "type": key,
                    "label": cat["label"],
                    "confidence": round(score, 3),
                })
        detected.sort(key=lambda x: x["confidence"], reverse=True)
        return detected

    def _detect_music(self, scores: np.ndarray) -> dict:
        """Detect music presence and type (live/recorded/singing).

        Uses Noisy OR for overall detection, then max-score comparison
        between instrument indices and BGM indices to determine type.
        """
        music_prob = _noisy_or(scores, MUSIC_ALL_INDICES)
        if music_prob < self._dim_thresholds["music_detection"]:
            return {
                "detected": False,
                "confidence": round(music_prob, 3),
                "type": None,
                "type_label": None,
            }

        type_threshold = self._dim_thresholds["music_type"]
        instrument_max = _max_score_for(scores, MUSIC_INSTRUMENT_INDICES)
        bgm_max = _max_score_for(scores, MUSIC_BGM_INDICES)
        singing_score = float(scores[MUSIC_SINGING_INDEX]) if MUSIC_SINGING_INDEX < len(scores) else 0.0

        mtype, mlabel = None, None
        if instrument_max > bgm_max and instrument_max >= type_threshold:
            mtype, mlabel = "live", "生演奏"
        elif bgm_max >= type_threshold:
            mtype, mlabel = "recorded", "録音/BGM"
        elif singing_score >= type_threshold:
            mtype, mlabel = "singing", "歌唱"

        return {
            "detected": True,
            "confidence": round(music_prob, 3),
            "type": mtype,
            "type_label": mlabel,
        }

    def _estimate_speaker_count(self, scores: np.ndarray) -> dict:
        """Estimate speaker count — mutually exclusive, max-score wins."""
        threshold = self._dim_thresholds["speaker_count"]
        best_key = None
        best_score = 0.0

        for key, cat in SPEAKER_COUNT_CATEGORIES.items():
            score = _max_score_for(scores, cat["indices"])
            if score > best_score:
                best_score = score
                best_key = key

        if best_score < threshold:
            return {"estimate": None, "label": None, "confidence": None}

        return {
            "estimate": best_key,
            "label": SPEAKER_COUNT_CATEGORIES[best_key]["label"],
            "confidence": round(best_score, 3),
        }

    def _detect_quality_issues(self, scores: np.ndarray) -> list:
        """Detect quality/anomaly issues — multi-label, max-score per category."""
        threshold = self._dim_thresholds["quality_issue"]
        detected = []
        for key, cat in QUALITY_CATEGORIES.items():
            score = _max_score_for(scores, cat["indices"])
            if score >= threshold:
                detected.append({
                    "type": key,
                    "label": cat["label"],
                    "confidence": round(score, 3),
                })
        detected.sort(key=lambda x: x["confidence"], reverse=True)
        return detected

    # ------------------------------------------------------------------
    # Score interpretation
    # ------------------------------------------------------------------

    def _interpret_scores(self, scores: np.ndarray) -> Optional[dict]:
        """Interpret PANNs output scores into multi-dimensional classification.

        Scene detection (priority-based with Noisy OR for media):
        1. Telephone: max(telephone indices) >= threshold
        2. Silence: silence >= threshold AND speech < threshold
        3. Media vs Conversation (when speech >= threshold):
           - media_prob = noisy_or(MEDIA_INDICES)
           - media_prob >= media_threshold → TV・ラジオ
           - otherwise → 会話
        4. Media (speech < threshold): noisy_or >= threshold
        5. Noise: max(noise indices) >= threshold

        Plus 5 additional dimensions extracted from the same score vector.
        """
        threshold = self._threshold

        # --- Priority-based scene detection ---
        best_scene_key = None
        best_scene_score = 0.0

        # 1. Telephone (highest priority)
        tel_score = _max_score_for(scores, SCENE_CATEGORIES["telephone"]["indices"])
        if tel_score >= threshold:
            best_scene_key = "telephone"
            best_scene_score = tel_score
        else:
            speech_score = _max_score_for(scores, SCENE_CATEGORIES["conversation"]["indices"])
            silence_score = _max_score_for(scores, SCENE_CATEGORIES["silence"]["indices"])

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
                    noise_score = _max_score_for(scores, SCENE_CATEGORIES["noise"]["indices"])
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

        # --- Extended dimensions ---
        noise_sources = self._detect_noise_sources(scores)
        events = self._detect_events(scores)
        music = self._detect_music(scores)
        speaker_count = self._estimate_speaker_count(scores)
        quality_issues = self._detect_quality_issues(scores)

        # Check if we have any detection at all
        has_any = (
            best_scene_key is not None
            or best_spatial_key is not None
            or noise_sources
            or events
            or music["detected"]
            or speaker_count["estimate"] is not None
            or quality_issues
        )
        if not has_any:
            return None

        # --- Top classes (for debugging / detail view) ---
        top_indices = np.argsort(scores)[::-1][:10]
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
            # Scene (backward compatible)
            "scene_label": SCENE_CATEGORIES[best_scene_key]["label"] if best_scene_key else None,
            "scene_label_en": best_scene_key,
            "scene_confidence": round(best_scene_score, 3) if best_scene_key else None,
            # Spatial (backward compatible)
            "spatial": spatial_label,
            "spatial_label": spatial_label,  # alias for indexer.py / session_tagger.py
            "spatial_en": best_spatial_key,
            "spatial_confidence": round(best_spatial_score, 3) if best_spatial_key else None,
            # Extended dimensions
            "noise_sources": noise_sources,
            "events": events,
            "music": music,
            "speaker_count": speaker_count,
            "quality_issues": quality_issues,
            # Debug / metadata
            "top_classes": top_classes,
            "classified_at": datetime.now().isoformat(),
        }

        log.info(
            "AudioSceneClassifier: scene=%s (%.2f), spatial=%s (%.2f), "
            "noise=%d, events=%d, music=%s, speakers=%s, quality=%d",
            result["scene_label"] or "unknown",
            best_scene_score,
            result["spatial"] or "unknown",
            best_spatial_score,
            len(noise_sources),
            len(events),
            music.get("type") or ("yes" if music["detected"] else "no"),
            speaker_count.get("estimate") or "unknown",
            len(quality_issues),
        )

        return result
