#!/usr/bin/env python3
"""Compare Whisper (large-v3) vs VibeVoice-ASR transcription quality.

Runs VibeVoice-ASR (microsoft/VibeVoice-ASR-HF, 4-bit quantized) on existing
recordings and compares the results against Whisper transcripts.

VibeVoice-ASR includes built-in speaker diarization, so pyannote is not needed.
The model requires ~6.6 GB VRAM with 4-bit quantization.

Usage:
    # Stop omi-whisper first to free GPU
    docker compose stop omi-whisper

    # Install deps (if needed)
    pip install "transformers>=5.3.0" bitsandbytes soundfile

    # Compare a single session
    python tools/compare_vibevoice.py --sessions 1 --ids 2026-03-27/23-47-21

    # Compare 8 diverse sessions
    python tools/compare_vibevoice.py --sessions 8

    # Limit max audio duration
    python tools/compare_vibevoice.py --sessions 8 --max-duration 300
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k, _v = _k.strip(), _v.strip()
                if _k and _v and _k not in os.environ:
                    os.environ[_k] = _v

RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CONFIG_DIR = PROJECT_ROOT / "config"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def check_dependencies():
    """Check required packages are installed."""
    missing = []
    for pkg in ["transformers", "bitsandbytes", "soundfile", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"{RED}ERROR: Missing packages: {', '.join(missing)}{RESET}")
        print(f"  pip install \"transformers>=5.3.0\" bitsandbytes soundfile")
        sys.exit(1)

    # Check transformers version
    import transformers
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if ver < (5, 3):
        print(f"{RED}ERROR: transformers >= 5.3.0 required (have {transformers.__version__}){RESET}")
        sys.exit(1)

    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"{RED}ERROR: ffmpeg not found{RESET}")
        sys.exit(1)


def check_gpu_available():
    """Check GPU availability and VRAM. Returns (available, free_mb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        print(f"  GPU VRAM: {free_mb} MB free / {total_mb} MB total")
        if free_mb < 7000:
            print(f"  {YELLOW}WARNING: VibeVoice-ASR needs ~6.6 GB VRAM. "
                  f"Only {free_mb} MB free.{RESET}")
            print(f"  {YELLOW}Consider: docker compose stop omi-whisper{RESET}")
        return True, free_mb
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  {YELLOW}WARNING: nvidia-smi not available. GPU status unknown.{RESET}")
        return False, 0


# ---------------------------------------------------------------------------
# Session finder with diversity
# ---------------------------------------------------------------------------

def find_diverse_sessions(limit=8, specific=None, max_duration=600):
    """Find diverse sessions for comparison.

    Selects from 4 duration buckets and 2 speaker buckets for variety.
    Requires audio.ogg and >= 3 segments.
    """
    if specific:
        result = []
        for s in specific:
            sess_dir = RECORDINGS_DIR / s
            transcript = sess_dir / "transcript.json"
            audio = sess_dir / "audio.ogg"
            if not transcript.exists():
                print(f"  {YELLOW}SKIP {s}: no transcript.json{RESET}")
                continue
            if not audio.exists():
                print(f"  {YELLOW}SKIP {s}: no audio.ogg{RESET}")
                continue
            result.append(s)
        return result

    # Collect all valid sessions with metadata
    candidates = []
    for date_dir in sorted(RECORDINGS_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            transcript = sess_dir / "transcript.json"
            audio = sess_dir / "audio.ogg"
            if not transcript.exists() or not audio.exists():
                continue
            try:
                with open(transcript) as f:
                    data = json.load(f)
                segs = data.get("segments", []) if isinstance(data, dict) else data
                session_meta = data.get("session", {}) if isinstance(data, dict) else {}

                if len(segs) < 3:
                    continue

                # Compute duration from segments
                duration = 0
                if segs:
                    duration = max(s.get("end", 0) for s in segs)
                if session_meta.get("duration_seconds"):
                    duration = session_meta["duration_seconds"]
                if duration > max_duration:
                    continue
                if duration < 2:
                    continue

                # Speaker info
                num_speakers = session_meta.get("num_speakers", 1)
                has_diarize = session_meta.get("speaker_tagged", False)
                unique_speakers = len(set(s.get("speaker", "") for s in segs if s.get("speaker")))
                if unique_speakers > num_speakers:
                    num_speakers = unique_speakers

                # Quality heuristic
                avg_logprob_vals = [s.get("avg_logprob", 0) for s in segs if s.get("avg_logprob") is not None]
                avg_logprob = sum(avg_logprob_vals) / len(avg_logprob_vals) if avg_logprob_vals else -0.5

                sess_id = f"{date_dir.name}/{sess_dir.name}"
                candidates.append({
                    "id": sess_id,
                    "duration": duration,
                    "seg_count": len(segs),
                    "num_speakers": num_speakers,
                    "has_diarize": has_diarize,
                    "avg_logprob": avg_logprob,
                })
            except Exception:
                continue

    if not candidates:
        return []

    # Bucket by duration
    buckets = {
        "short": [],      # <= 60s
        "medium": [],     # 60-300s
        "long": [],       # 300-600s
        "very_long": [],  # > 600s
    }
    for c in candidates:
        d = c["duration"]
        if d <= 60:
            buckets["short"].append(c)
        elif d <= 300:
            buckets["medium"].append(c)
        elif d <= 600:
            buckets["long"].append(c)
        else:
            buckets["very_long"].append(c)

    # Sub-bucket by speaker count
    selected = []
    per_bucket = max(1, limit // 4)
    remainder = limit

    for bname in ["medium", "long", "short", "very_long"]:
        bucket = buckets[bname]
        if not bucket:
            continue

        # Split into single vs multi speaker
        single = [c for c in bucket if c["num_speakers"] <= 1]
        multi = [c for c in bucket if c["num_speakers"] > 1]

        # Sort by segment count (prefer more content) and quality diversity
        single.sort(key=lambda x: (-x["seg_count"], x["avg_logprob"]))
        multi.sort(key=lambda x: (-x["seg_count"], x["avg_logprob"]))

        take = min(per_bucket, remainder)
        added = 0

        # Alternate single/multi
        si, mi = 0, 0
        prefer_multi = True
        while added < take and (si < len(single) or mi < len(multi)):
            if prefer_multi and mi < len(multi):
                selected.append(multi[mi]["id"])
                mi += 1
                added += 1
            elif si < len(single):
                selected.append(single[si]["id"])
                si += 1
                added += 1
            elif mi < len(multi):
                selected.append(multi[mi]["id"])
                mi += 1
                added += 1
            prefer_multi = not prefer_multi

        remainder -= added

    # Fill remaining from all candidates
    if remainder > 0:
        for c in candidates:
            if c["id"] not in selected:
                selected.append(c["id"])
                remainder -= 1
                if remainder <= 0:
                    break

    return selected[:limit]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(session_id):
    """Load transcript and return (session_meta, segments)."""
    path = RECORDINGS_DIR / session_id / "transcript.json"
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        segs = data.get("segments", [])
        meta = data.get("session", {})
    else:
        segs = data
        meta = {}

    return meta, segs


def get_audio_path(session_id):
    """Return path to audio file (ogg preferred, wav fallback)."""
    sess_dir = RECORDINGS_DIR / session_id
    ogg = sess_dir / "audio.ogg"
    if ogg.exists():
        return ogg
    wav = sess_dir / "audio.wav"
    if wav.exists():
        return wav
    return None


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_to_wav(audio_path, target_sr=24000):
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


# ---------------------------------------------------------------------------
# Hotwords prompt
# ---------------------------------------------------------------------------

def build_hotwords_prompt(session_id=None):
    """Build hotwords prompt string from config."""
    try:
        from hotword_loader import HotwordLoader
        hl = HotwordLoader(str(CONFIG_DIR))
        custom = hl.get_custom_words()
        hotwords_list = hl.get_hotwords_list()

        # Combine custom + category hotwords, limit to reasonable size
        prompt_words = list(custom)
        for w in hotwords_list:
            if w not in prompt_words:
                prompt_words.append(w)
            if len(prompt_words) >= 100:
                break

        if prompt_words:
            return "Hotwords: " + ", ".join(prompt_words)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# VibeVoice model management
# ---------------------------------------------------------------------------

def load_vibevoice_model():
    """Load VibeVoice-ASR model with 4-bit quantization.

    Returns (model, processor).
    """
    from transformers import AutoProcessor, BitsAndBytesConfig

    # Import model class - VibeVoice-ASR uses auto-registration
    from transformers import AutoModelForSpeechSeq2Seq

    print(f"  Loading VibeVoice-ASR model (4-bit quantized)...")
    t0 = time.time()

    model_id = "microsoft/VibeVoice-ASR-HF"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    import torch
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s")
    return model, processor


def unload_vibevoice_model(model, processor):
    """Unload model to free VRAM."""
    import torch
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# VibeVoice inference
# ---------------------------------------------------------------------------

def _load_audio_array(wav_path, target_sr=24000):
    """Load audio as numpy array at target sample rate using soundfile.

    Returns (audio_array, sample_rate).
    Uses soundfile instead of torchcodec to avoid libnvrtc dependency.
    """
    import soundfile as sf
    import numpy as np

    audio_data, sr = sf.read(wav_path, dtype="float32")

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import torchaudio
            import torch as _torch
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

    # Normalize audio level to -16 dBFS
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms > 0:
        target_rms = 10 ** (-16.0 / 20)  # -16 dBFS
        audio_data = (audio_data * (target_rms / rms)).astype(np.float32)

    return audio_data, sr


def run_vibevoice(model, processor, wav_path, prompt="", chunk_size=1440000):
    """Run VibeVoice-ASR inference.

    Returns list of segments: [{"Start": float, "End": float, "Speaker": int, "Content": str}]
    On CUDA OOM, retries with smaller chunk_size.
    """
    import torch

    audio_data, sr = _load_audio_array(wav_path)
    duration = len(audio_data) / sr

    if duration < 2.0:
        return []

    try:
        # Pass numpy array instead of file path to avoid torchcodec dependency
        inputs = processor.apply_transcription_request(
            audio=audio_data,
            prompt=prompt if prompt else None,
            sampling_rate=sr,
        )
        # Move inputs to model device with correct dtype
        def _move_tensor(v):
            if not hasattr(v, "to"):
                return v
            if v.is_floating_point():
                return v.to(model.device, dtype=torch.bfloat16)
            return v.to(model.device)

        inputs = {k: _move_tensor(v) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=8192,
                repetition_penalty=1.2,
            )

        new_tokens = output_ids[:, input_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        # Parse JSON output from VibeVoice
        segments = _parse_vibevoice_output(text)
        return segments

    except torch.cuda.OutOfMemoryError:
        if chunk_size > 480000:
            new_chunk = 480000
            print(f"    {YELLOW}OOM: retrying with chunk_size={new_chunk}{RESET}")
            torch.cuda.empty_cache()
            return _run_vibevoice_chunked(model, processor, wav_path, prompt, new_chunk)
        else:
            print(f"    {RED}OOM: skipping (audio too large for available VRAM){RESET}")
            torch.cuda.empty_cache()
            return None


def _run_vibevoice_chunked(model, processor, wav_path, prompt, chunk_size):
    """Run VibeVoice on audio in chunks to avoid OOM."""
    import torch

    audio_data, sr = _load_audio_array(wav_path)
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
            inputs = processor.apply_transcription_request(
                audio=chunk,
                prompt=prompt if prompt else None,
                sampling_rate=sr,
            )
            def _move(v):
                if not hasattr(v, "to"):
                    return v
                if v.is_floating_point():
                    return v.to(model.device, dtype=torch.bfloat16)
                return v.to(model.device)

            inputs = {k: _move(v)
                      for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    repetition_penalty=1.2,
                )

            new_tokens = output_ids[:, input_len:]
            text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            chunk_segs = _parse_vibevoice_output(text)

            # Adjust timestamps by offset
            for seg in chunk_segs:
                seg["Start"] += offset_s
                seg["End"] += offset_s
                all_segments.append(seg)

        except torch.cuda.OutOfMemoryError:
            print(f"    {RED}OOM on chunk at {offset_s:.1f}s, skipping{RESET}")
            torch.cuda.empty_cache()

        offset_s += chunk_duration

    return all_segments


def _parse_vibevoice_output(text):
    """Parse VibeVoice-ASR output text into segment list.

    VibeVoice outputs JSON array:
    [{"Start": 0, "End": 15.43, "Speaker": 0, "Content": "text"}]

    When input tokens are not stripped, the decoded text may contain the
    full prompt (system/user/assistant markers and hotwords).  We handle
    this by looking for an ``assistant\\n`` marker and only searching for
    JSON after it.
    """
    text = text.strip()

    # 1. Direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. If "assistant\n" marker present, only search after it
    marker = text.rfind("assistant\n")
    search_text = text[marker + 10:].strip() if marker >= 0 else text

    # 3. Extract JSON array
    start_idx = search_text.find("[")
    end_idx = search_text.rfind("]")
    if start_idx >= 0 and end_idx > start_idx:
        try:
            result = json.loads(search_text[start_idx:end_idx + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # 4. Fallback: return empty list (do not create corrupted segments)
    if text:
        print(f"    {YELLOW}WARNING: VibeVoice output parse failed ({len(text)} chars){RESET}")
    return []


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def compare_text(whisper_segs, vibe_segs):
    """Compare full text output between Whisper and VibeVoice.

    Returns dict with similarity ratio, char counts, and notable differences.
    """
    # Build full text
    whisper_text = "".join(s.get("text", "") for s in whisper_segs)
    vibe_text = "".join(s.get("Content", "") for s in vibe_segs)

    # Overall similarity
    similarity = difflib.SequenceMatcher(None, whisper_text, vibe_text).ratio()

    # Find notable differences using segment-level alignment
    differences = []
    for ws in whisper_segs:
        w_start = ws.get("start", 0)
        w_end = ws.get("end", 0)
        w_text = ws.get("text", "").strip()
        if not w_text:
            continue

        # Find overlapping VibeVoice segments
        v_texts = []
        for vs in vibe_segs:
            v_start = vs.get("Start", 0)
            v_end = vs.get("End", 0)
            # Check time overlap
            overlap = min(w_end, v_end) - max(w_start, v_start)
            if overlap > 0:
                v_texts.append(vs.get("Content", "").strip())

        v_combined = " ".join(v_texts).strip()
        if not v_combined:
            continue

        seg_sim = difflib.SequenceMatcher(None, w_text, v_combined).ratio()
        if seg_sim < 0.8 and (len(w_text) > 5 or len(v_combined) > 5):
            differences.append({
                "time": f"{w_start:.2f}-{w_end:.2f}s",
                "whisper": w_text[:100],
                "vibevoice": v_combined[:100],
                "similarity": round(seg_sim, 3),
            })

    # Sort by most different first
    differences.sort(key=lambda x: x["similarity"])

    return {
        "text_similarity": round(similarity, 4),
        "whisper_chars": len(whisper_text),
        "vibevoice_chars": len(vibe_text),
        "notable_differences": differences[:10],  # Top 10
    }


def compare_speakers(whisper_segs, vibe_segs):
    """Compare speaker assignments between Whisper and VibeVoice.

    Uses time-overlap based matching to compute speaker agreement.
    """
    if not whisper_segs or not vibe_segs:
        return {"speaker_agreement": 0, "whisper_speakers": 0, "vibevoice_speakers": 0}

    # Count unique speakers
    w_speakers = set()
    for s in whisper_segs:
        sp = s.get("speaker", s.get("person_name", "unknown"))
        w_speakers.add(sp)

    v_speakers = set()
    for s in vibe_segs:
        v_speakers.add(s.get("Speaker", 0))

    # Build speaker mapping via maximum overlap
    # For each Whisper speaker, find the most common overlapping VibeVoice speaker
    speaker_map = {}  # whisper_speaker -> vibevoice_speaker
    for ws in whisper_segs:
        w_sp = ws.get("speaker", ws.get("person_name", "unknown"))
        w_start = ws.get("start", 0)
        w_end = ws.get("end", 0)
        if w_sp in speaker_map:
            continue

        overlap_counts = {}
        for vs in vibe_segs:
            v_start = vs.get("Start", 0)
            v_end = vs.get("End", 0)
            overlap = max(0, min(w_end, v_end) - max(w_start, v_start))
            if overlap > 0:
                v_sp = vs.get("Speaker", 0)
                overlap_counts[v_sp] = overlap_counts.get(v_sp, 0) + overlap

        if overlap_counts:
            speaker_map[w_sp] = max(overlap_counts, key=overlap_counts.get)

    # Compute agreement using the mapping
    total_duration = 0
    agreed_duration = 0

    for ws in whisper_segs:
        w_sp = ws.get("speaker", ws.get("person_name", "unknown"))
        w_start = ws.get("start", 0)
        w_end = ws.get("end", 0)
        w_dur = w_end - w_start
        if w_dur <= 0:
            continue

        mapped_v_sp = speaker_map.get(w_sp)

        for vs in vibe_segs:
            v_start = vs.get("Start", 0)
            v_end = vs.get("End", 0)
            overlap = max(0, min(w_end, v_end) - max(w_start, v_start))
            if overlap > 0:
                total_duration += overlap
                v_sp = vs.get("Speaker", 0)
                if mapped_v_sp is not None and v_sp == mapped_v_sp:
                    agreed_duration += overlap

    agreement = agreed_duration / total_duration if total_duration > 0 else 0

    return {
        "speaker_agreement": round(agreement, 4),
        "whisper_speakers": len(w_speakers),
        "vibevoice_speakers": len(v_speakers),
        "speaker_map": {str(k): v for k, v in speaker_map.items()},
    }


def compare_timestamps(whisper_segs, vibe_segs):
    """Compare segment boundaries and coverage overlap."""
    if not whisper_segs or not vibe_segs:
        return {"avg_boundary_diff_s": 0, "coverage_overlap": 0}

    # Segment boundary comparison
    w_boundaries = []
    for s in whisper_segs:
        w_boundaries.append(s.get("start", 0))
        w_boundaries.append(s.get("end", 0))
    w_boundaries = sorted(set(w_boundaries))

    v_boundaries = []
    for s in vibe_segs:
        v_boundaries.append(s.get("Start", 0))
        v_boundaries.append(s.get("End", 0))
    v_boundaries = sorted(set(v_boundaries))

    # Average nearest-boundary distance
    if w_boundaries and v_boundaries:
        diffs = []
        for wb in w_boundaries:
            nearest = min(abs(wb - vb) for vb in v_boundaries)
            diffs.append(nearest)
        avg_diff = sum(diffs) / len(diffs)
    else:
        avg_diff = 0

    # Coverage overlap (how much of whisper time is covered by vibevoice)
    w_max = max(s.get("end", 0) for s in whisper_segs)
    v_max = max(s.get("End", 0) for s in vibe_segs)
    total_max = max(w_max, v_max)

    if total_max > 0:
        # Simple overlap: fraction of time covered by both
        resolution = 0.5  # 0.5s resolution
        steps = int(total_max / resolution) + 1
        both_covered = 0
        either_covered = 0

        for i in range(steps):
            t = i * resolution
            w_covered = any(
                s.get("start", 0) <= t <= s.get("end", 0) for s in whisper_segs
            )
            v_covered = any(
                s.get("Start", 0) <= t <= s.get("End", 0) for s in vibe_segs
            )
            if w_covered or v_covered:
                either_covered += 1
            if w_covered and v_covered:
                both_covered += 1

        coverage = both_covered / either_covered if either_covered > 0 else 0
    else:
        coverage = 0

    return {
        "avg_boundary_diff_s": round(avg_diff, 3),
        "coverage_overlap": round(coverage, 4),
        "whisper_seg_count": len(whisper_segs),
        "vibevoice_seg_count": len(vibe_segs),
    }


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

def print_session_result(idx, total, sess_id, result):
    """Print colored session comparison result."""
    print(f"\n{'=' * 70}")
    print(f"[{idx}/{total}] {BOLD}{sess_id}{RESET}")
    print(f"  Duration: {result['duration_s']:.1f}s")
    print()

    w = result["whisper"]
    v = result["vibevoice"]
    c = result["comparison"]

    # Stats table
    print(f"  {'':30} {'Whisper':>12} {'VibeVoice':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Segments':30} {w['seg_count']:>12} {v['seg_count']:>12}")
    print(f"  {'Characters':30} {w['char_count']:>12} {v['char_count']:>12}")
    print(f"  {'Speakers':30} {w['speaker_count']:>12} {v['speaker_count']:>12}")
    if v.get("elapsed_s"):
        print(f"  {'VibeVoice inference time':30} {'':>12} {v['elapsed_s']:>11.1f}s")
    print()

    # Similarity scores
    sim = c["text_similarity"]
    color = GREEN if sim >= 0.85 else YELLOW if sim >= 0.7 else RED
    print(f"  Text similarity:     {color}{sim:.1%}{RESET}")

    if "speaker_agreement" in c:
        sa = c["speaker_agreement"]
        color = GREEN if sa >= 0.85 else YELLOW if sa >= 0.7 else RED
        print(f"  Speaker agreement:   {color}{sa:.1%}{RESET}")

    if "avg_boundary_diff_s" in c:
        bd = c["avg_boundary_diff_s"]
        color = GREEN if bd < 0.5 else YELLOW if bd < 1.0 else RED
        print(f"  Avg boundary diff:   {color}{bd:.2f}s{RESET}")

    if "coverage_overlap" in c:
        co = c["coverage_overlap"]
        color = GREEN if co >= 0.9 else YELLOW if co >= 0.7 else RED
        print(f"  Coverage overlap:    {color}{co:.1%}{RESET}")

    # Notable differences
    diffs = c.get("notable_differences", [])
    if diffs:
        print(f"\n  {BOLD}Notable differences ({len(diffs)}):{RESET}")
        for i, d in enumerate(diffs[:5]):
            print(f"    {DIM}[{d['time']}]{RESET} sim={d['similarity']:.0%}")
            print(f"      W: {d['whisper'][:70]}")
            print(f"      V: {d['vibevoice'][:70]}")


def print_summary(results, total_elapsed):
    """Print final summary table."""
    print(f"\n{'=' * 70}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'=' * 70}\n")

    if not results:
        print("No sessions processed.")
        return

    sims = [r["comparison"]["text_similarity"] for r in results]
    spk_agreements = [r["comparison"].get("speaker_agreement", 0) for r in results]
    boundary_diffs = [r["comparison"].get("avg_boundary_diff_s", 0) for r in results]
    elapsed_times = [r["vibevoice"]["elapsed_s"] for r in results if r["vibevoice"].get("elapsed_s")]

    n = len(results)

    print(f"  Sessions compared:       {n}")
    print(f"  Total elapsed:           {total_elapsed:.1f}s")
    print()

    avg_sim = sum(sims) / len(sims)
    color = GREEN if avg_sim >= 0.85 else YELLOW if avg_sim >= 0.7 else RED
    print(f"  Avg text similarity:     {color}{avg_sim:.1%}{RESET}  "
          f"(min={min(sims):.1%}, max={max(sims):.1%})")

    if spk_agreements:
        avg_spk = sum(spk_agreements) / len(spk_agreements)
        color = GREEN if avg_spk >= 0.85 else YELLOW if avg_spk >= 0.7 else RED
        print(f"  Avg speaker agreement:   {color}{avg_spk:.1%}{RESET}  "
              f"(min={min(spk_agreements):.1%}, max={max(spk_agreements):.1%})")

    if boundary_diffs:
        avg_bd = sum(boundary_diffs) / len(boundary_diffs)
        print(f"  Avg boundary diff:       {avg_bd:.2f}s")

    if elapsed_times:
        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        print(f"  Avg inference time:      {avg_elapsed:.1f}s per session")

    # Per-session summary table
    print(f"\n  {'Session':<30} {'Sim':>6} {'Spkr':>6} {'BndDf':>6} {'Time':>6}")
    print(f"  {'-' * 54}")
    for r in results:
        sid = r["session_id"]
        if len(sid) > 28:
            sid = "..." + sid[-25:]
        sim = r["comparison"]["text_similarity"]
        spk = r["comparison"].get("speaker_agreement", 0)
        bd = r["comparison"].get("avg_boundary_diff_s", 0)
        et = r["vibevoice"].get("elapsed_s", 0)
        print(f"  {sid:<30} {sim:>5.1%} {spk:>5.1%} {bd:>5.2f}s {et:>5.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Whisper (large-v3) vs VibeVoice-ASR transcription"
    )
    parser.add_argument("--sessions", type=int, default=8,
                        help="Number of sessions to compare (default: 8)")
    parser.add_argument("--ids", nargs="*",
                        help="Specific session IDs (e.g., 2026-03-27/23-47-21)")
    parser.add_argument("--max-duration", type=int, default=600,
                        help="Max audio duration in seconds (default: 600)")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"{BOLD}WHISPER vs VIBEVOICE-ASR COMPARISON{RESET}")
    print(f"{'=' * 70}")
    print(f"  Model: microsoft/VibeVoice-ASR-HF (4-bit quantized)")
    print(f"  Max duration: {args.max_duration}s")
    print()

    # Check dependencies
    check_dependencies()
    gpu_ok, free_mb = check_gpu_available()
    print()

    # Find sessions
    print("Finding sessions...")
    session_ids = find_diverse_sessions(
        limit=args.sessions,
        specific=args.ids,
        max_duration=args.max_duration,
    )
    if not session_ids:
        print(f"{RED}ERROR: No valid sessions found{RESET}")
        sys.exit(1)
    print(f"  Selected {len(session_ids)} sessions:")
    for sid in session_ids:
        meta, segs = load_session(sid)
        duration = max((s.get("end", 0) for s in segs), default=0)
        print(f"    {sid}  ({len(segs)} segs, {duration:.0f}s)")
    print()

    # Load VibeVoice model
    model, processor = load_vibevoice_model()
    print()

    # Process sessions
    all_results = []
    total_t0 = time.time()

    for idx, sess_id in enumerate(session_ids, 1):
        meta, whisper_segs = load_session(sess_id)
        audio_path = get_audio_path(sess_id)

        if not audio_path:
            print(f"\n[{idx}/{len(session_ids)}] {sess_id}: {RED}no audio file, skipping{RESET}")
            continue

        # Duration from segments
        duration = max((s.get("end", 0) for s in whisper_segs), default=0)
        if meta.get("duration_seconds"):
            duration = meta["duration_seconds"]

        # Convert to WAV
        wav_path = None
        try:
            print(f"\n[{idx}/{len(session_ids)}] {sess_id}  ({duration:.0f}s)")
            print(f"  Converting audio to WAV (24kHz)...", end=" ", flush=True)
            wav_path = convert_to_wav(audio_path)
            print("done")

            # Build hotwords prompt
            prompt = build_hotwords_prompt(sess_id)

            # Run VibeVoice
            print(f"  Running VibeVoice-ASR...", end=" ", flush=True)
            t0 = time.time()
            vibe_segs = run_vibevoice(model, processor, wav_path, prompt)
            elapsed = time.time() - t0

            if vibe_segs is None:
                print(f"{RED}failed (OOM){RESET}")
                continue

            print(f"done ({elapsed:.1f}s, {len(vibe_segs)} segments)")

            # Whisper stats
            w_chars = sum(len(s.get("text", "")) for s in whisper_segs)
            w_speakers = len(set(s.get("speaker", "") for s in whisper_segs if s.get("speaker")))

            # VibeVoice stats
            v_chars = sum(len(s.get("Content", "")) for s in vibe_segs)
            v_speakers = len(set(s.get("Speaker", 0) for s in vibe_segs))

            # Compare
            text_comp = compare_text(whisper_segs, vibe_segs)
            speaker_comp = compare_speakers(whisper_segs, vibe_segs)
            timestamp_comp = compare_timestamps(whisper_segs, vibe_segs)

            result = {
                "session_id": sess_id,
                "duration_s": duration,
                "whisper": {
                    "seg_count": len(whisper_segs),
                    "char_count": w_chars,
                    "speaker_count": w_speakers,
                },
                "vibevoice": {
                    "seg_count": len(vibe_segs),
                    "char_count": v_chars,
                    "speaker_count": v_speakers,
                    "elapsed_s": round(elapsed, 2),
                },
                "comparison": {
                    "text_similarity": text_comp["text_similarity"],
                    "notable_differences": text_comp["notable_differences"],
                    "speaker_agreement": speaker_comp["speaker_agreement"],
                    "speaker_map": speaker_comp.get("speaker_map", {}),
                    "avg_boundary_diff_s": timestamp_comp["avg_boundary_diff_s"],
                    "coverage_overlap": timestamp_comp["coverage_overlap"],
                },
                "whisper_segments": whisper_segs,
                "vibevoice_segments": vibe_segs,
            }
            all_results.append(result)

            # Print per-session results
            print_session_result(idx, len(session_ids), sess_id, result)

        except Exception as e:
            print(f"\n  {RED}ERROR: {e}{RESET}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    total_elapsed = time.time() - total_t0

    # Summary
    print_summary(all_results, total_elapsed)

    # Unload model
    print(f"\n  Unloading model...")
    unload_vibevoice_model(model, processor)

    # Save results
    if all_results:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = CONFIG_DIR / f"compare_vibevoice_{ts}.json"
        save_data = {
            "model": "microsoft/VibeVoice-ASR-HF (4-bit)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sessions": all_results,
            "summary": {
                "session_count": len(all_results),
                "avg_similarity": round(
                    sum(r["comparison"]["text_similarity"] for r in all_results) / len(all_results), 4
                ),
                "avg_speaker_agreement": round(
                    sum(r["comparison"]["speaker_agreement"] for r in all_results) / len(all_results), 4
                ),
                "total_elapsed_s": round(total_elapsed, 2),
            },
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
