"""VibeVoice-ASR transcriber module.

Provides VibeVoiceTranscriber class for batch transcription using
microsoft/VibeVoice-ASR-HF with 4-bit quantization (~6.6 GB VRAM).

Extracted from tools/compare_vibevoice.py for use in Bridge batch processing.
"""

import gc
import json
import logging
import os
import subprocess
import tempfile
import time

log = logging.getLogger(__name__)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"


class VibeVoiceTranscriber:
    """VibeVoice-ASR transcriber with model lifecycle management."""

    def __init__(self):
        self.model = None
        self.processor = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load_model(self):
        """Load VibeVoice-ASR model with 4-bit quantization.

        Requires ~6.6 GB VRAM. Must be called from a thread (run_in_executor)
        as model loading is blocking.
        """
        if self.loaded:
            return

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig

        log.info("Loading VibeVoice-ASR model (4-bit quantized)...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        elapsed = time.time() - t0
        log.info("VibeVoice-ASR model loaded in %.1fs", elapsed)

    def unload_model(self):
        """Unload model and free VRAM."""
        import torch

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("VibeVoice-ASR model unloaded")

    def transcribe(self, audio_path: str, prompt: str = "", chunk_size: int = 1440000):
        """Run VibeVoice-ASR inference on a WAV file.

        Args:
            audio_path: Path to WAV file (24kHz mono recommended).
            prompt: Optional hotwords prompt string.
            chunk_size: Samples per chunk for OOM recovery.

        Returns:
            List of normalized segments [{"start", "end", "speaker", "text"}],
            or None on unrecoverable OOM.
        """
        import torch

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        audio_data, sr = _load_audio(audio_path)
        duration = len(audio_data) / sr

        if duration < 2.0:
            return []

        try:
            segments = self._infer(audio_data, sr, prompt)
            return _normalize_segments(segments)
        except torch.cuda.OutOfMemoryError:
            if chunk_size > 480000:
                new_chunk = 480000
                log.warning("OOM: retrying with chunk_size=%d", new_chunk)
                torch.cuda.empty_cache()
                segments = self._infer_chunked(audio_data, sr, prompt, new_chunk)
                return _normalize_segments(segments)
            else:
                log.error("OOM: skipping (audio too large for available VRAM)")
                torch.cuda.empty_cache()
                return None

    def _infer(self, audio_data, sr: int, prompt: str):
        """Single-pass inference. Returns raw VibeVoice segments."""
        import torch

        inputs = self.processor.apply_transcription_request(
            audio=audio_data,
            prompt=prompt if prompt else None,
            sampling_rate=sr,
        )
        inputs = {k: _move_tensor(v, self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=4096)

        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return _parse_output(text)

    def _infer_chunked(self, audio_data, sr: int, prompt: str, chunk_size: int):
        """Chunked inference for OOM recovery."""
        import torch

        total_samples = len(audio_data)
        all_segments = []
        offset_s = 0.0

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = audio_data[start:end]
            chunk_duration = len(chunk) / sr

            if chunk_duration < 1.0:
                break

            try:
                inputs = self.processor.apply_transcription_request(
                    audio=chunk,
                    prompt=prompt if prompt else None,
                    sampling_rate=sr,
                )
                inputs = {k: _move_tensor(v, self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=4096)

                text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                chunk_segs = _parse_output(text)

                for seg in chunk_segs:
                    seg["Start"] += offset_s
                    seg["End"] += offset_s
                    all_segments.append(seg)

            except torch.cuda.OutOfMemoryError:
                log.error("OOM on chunk at %.1fs, skipping", offset_s)
                torch.cuda.empty_cache()

            offset_s += chunk_duration

        return all_segments

    @staticmethod
    def convert_to_wav(audio_path: str, target_sr: int = 24000) -> str:
        """Convert audio to WAV (24kHz mono) for VibeVoice-ASR.

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


# ---- Module-level helpers ----

def _move_tensor(v, device):
    """Move tensor to model device with correct dtype."""
    import torch

    if not hasattr(v, "to"):
        return v
    if v.is_floating_point():
        return v.to(device, dtype=torch.bfloat16)
    return v.to(device)


def _load_audio(wav_path: str, target_sr: int = 24000):
    """Load audio as numpy array at target sample rate using soundfile.

    Returns (audio_array, sample_rate).
    """
    import numpy as np
    import soundfile as sf

    audio_data, sr = sf.read(wav_path, dtype="float32")

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torch as _torch
            import torchaudio
            tensor = _torch.from_numpy(audio_data).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, target_sr)
            audio_data = tensor.squeeze(0).numpy()
        except ImportError:
            # Linear interpolation fallback
            ratio = target_sr / sr
            new_len = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_len)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
        sr = target_sr

    return audio_data, sr


def _parse_output(text: str) -> list:
    """Parse VibeVoice-ASR output text into raw segment list.

    VibeVoice outputs JSON array:
    [{"Start": 0, "End": 15.43, "Speaker": 0, "Content": "text"}]
    """
    text = text.strip()

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx >= 0 and end_idx > start_idx:
        try:
            result = json.loads(text[start_idx:end_idx + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: treat entire text as single segment
    if text:
        return [{"Start": 0, "End": 0, "Speaker": 0, "Content": text}]
    return []


def _normalize_segments(raw_segments: list) -> list:
    """Normalize VibeVoice output format to standard segment format.

    Converts {"Start","End","Speaker","Content"} to {"start","end","speaker","text"}.
    This is needed for compatibility with speaker_filter.tag_segments_with_model()
    which expects "start"/"end" fields.
    """
    normalized = []
    for seg in raw_segments:
        normalized.append({
            "start": float(seg.get("Start", 0)),
            "end": float(seg.get("End", 0)),
            "speaker": seg.get("Speaker", 0),
            "text": seg.get("Content", ""),
        })
    return normalized
