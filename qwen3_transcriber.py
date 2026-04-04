"""Qwen3-ASR transcriber module (evaluation only).

Provides Qwen3ASRTranscriber class for batch transcription using
Qwen/Qwen3-ASR-1.7B with optional Qwen3-ForcedAligner-0.6B for timestamps.

NOT integrated into Bridge — used only by benchmark/comparison tools.
bf16 (~4GB VRAM) fits easily on RTX 3060 alongside other models.
"""

import gc
import logging
import os
import subprocess
import tempfile
import time

import numpy as np

log = logging.getLogger(__name__)

ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


class Qwen3ASRTranscriber:
    """Qwen3-ASR transcriber with model lifecycle management."""

    def __init__(self):
        self.model = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load_model(self, use_aligner: bool = True):
        """Load Qwen3-ASR model with optional forced aligner.

        Args:
            use_aligner: If True, also load ForcedAligner for timestamps.
        """
        if self.loaded:
            return

        import torch

        log.info("Loading Qwen3-ASR model (bf16, aligner=%s)...", use_aligner)
        t0 = time.time()

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr package not installed. Run: pip install qwen-asr"
            )

        kwargs = dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=8,
            max_new_tokens=4096,
        )

        if use_aligner:
            kwargs["forced_aligner"] = ALIGNER_MODEL_ID
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=torch.bfloat16,
                device_map="cuda:0",
            )

        self.model = Qwen3ASRModel.from_pretrained(ASR_MODEL_ID, **kwargs)

        elapsed = time.time() - t0
        log.info("Qwen3-ASR loaded in %.1fs (aligner=%s)", elapsed, use_aligner)

    def unload_model(self):
        """Unload model and free VRAM."""
        import torch

        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("Qwen3-ASR model unloaded")

    def transcribe(self, audio_path: str, language: str = "Japanese"):
        """Run Qwen3-ASR on an audio file.

        Args:
            audio_path: Path to audio file (WAV, OGG, etc.).
            language: Language hint (None for auto-detect).

        Returns:
            List of segments [{"start", "end", "speaker": 0, "text"}].
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.transcribe(
            audio=audio_path,
            language=language,
            return_time_stamps=True,
        )

        if not results:
            return []

        return _convert_result(results[0])

    def transcribe_audio(self, audio_data: np.ndarray, sr: int,
                         language: str = "Japanese"):
        """Run inference on pre-loaded audio data.

        Args:
            audio_data: Numpy array of audio samples (float32).
            sr: Sample rate.
            language: Language hint.

        Returns:
            List of segments [{"start", "end", "speaker": 0, "text"}].
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.transcribe(
            audio=(audio_data, sr),
            language=language,
            return_time_stamps=True,
        )

        if not results:
            return []

        return _convert_result(results[0])

    @staticmethod
    def convert_to_wav(audio_path: str, target_sr: int = 16000) -> str:
        """Convert audio to WAV for Qwen3-ASR.

        Returns path to temporary WAV file. Caller must delete it.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name],
                capture_output=True, check=True, timeout=60,
            )
            return tmp.name
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            os.unlink(tmp.name)
            raise RuntimeError(f"ffmpeg conversion failed: {e}")


def _convert_result(result) -> list:
    """Convert Qwen3-ASR result to standard segment format.

    Qwen3-ASR returns a single result with .text and .time_stamps.
    time_stamps is a list of token-level timestamps.
    We group these into sentence-level segments.
    """
    segments = []

    if result.time_stamps and len(result.time_stamps) > 0:
        # Group timestamps into segments by punctuation boundaries
        current_text = ""
        current_start = None

        for ts in result.time_stamps:
            if current_start is None:
                current_start = ts.start_time

            current_text += ts.text
            current_end = ts.end_time

            # Split on sentence-ending punctuation
            if ts.text and ts.text[-1] in "。？！.?!\n":
                text = current_text.strip()
                if text:
                    segments.append({
                        "start": round(current_start, 2),
                        "end": round(current_end, 2),
                        "speaker": 0,
                        "text": text,
                    })
                current_text = ""
                current_start = None

        # Flush remaining text
        if current_text.strip():
            segments.append({
                "start": round(current_start, 2) if current_start is not None else 0.0,
                "end": round(current_end, 2),
                "speaker": 0,
                "text": current_text.strip(),
            })
    else:
        # No timestamps: single segment with full text
        text = result.text.strip() if result.text else ""
        if text:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "speaker": 0,
                "text": text,
            })

    return segments
