#!/usr/bin/env python3
"""Compare Whisper model transcription quality on existing recordings.

Transcribes saved audio with a candidate model and compares against the
original large-v3 transcript. Reports per-session and aggregate metrics.

Requires: faster-whisper, ffmpeg

Usage:
    # Stop omi-whisper first to free GPU
    docker compose stop omi-whisper

    # Compare kotoba-whisper vs large-v3 on 8 diverse sessions
    python tools/compare_whisper_models.py \
        --model kotoba-tech/kotoba-whisper-v2.0-faster \
        --sessions 8

    # Compare on specific sessions (noisy + clean mix)
    python tools/compare_whisper_models.py \
        --model kotoba-tech/kotoba-whisper-v2.0-faster \
        --ids 2026-04-07/00-01-14 2026-04-03/10-22-58

    # With custom compute type
    python tools/compare_whisper_models.py \
        --model kotoba-tech/kotoba-whisper-v2.0-faster \
        --compute-type float16 --sessions 8
"""

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
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

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Session finder
# ---------------------------------------------------------------------------

def find_sessions(limit=8, specific=None, max_duration=600, quality_filter=None):
    """Find sessions for comparison.

    Args:
        limit: Max sessions to return
        specific: List of specific session IDs
        max_duration: Max audio duration in seconds
        quality_filter: "noisy", "clean", or None for mixed
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
                meta = data.get("session", {}) if isinstance(data, dict) else {}
                if len(segs) < 3:
                    continue
                duration = meta.get("duration_seconds", 0)
                if not duration and segs:
                    duration = max(s.get("end", 0) for s in segs)
                if duration > max_duration or duration < 5:
                    continue

                quality = meta.get("tags", {}).get("quality", "unknown")
                if quality_filter and quality != quality_filter:
                    continue

                avg_lp = [s.get("avg_logprob", -0.5) for s in segs if s.get("avg_logprob") is not None]
                avg_logprob = sum(avg_lp) / len(avg_lp) if avg_lp else -0.5

                sess_id = f"{date_dir.name}/{sess_dir.name}"
                candidates.append({
                    "id": sess_id,
                    "duration": duration,
                    "seg_count": len(segs),
                    "quality": quality,
                    "avg_logprob": avg_logprob,
                })
            except Exception:
                continue

    if not candidates:
        return []

    # Pick a diverse mix: half noisy, half clean/other
    noisy = [c for c in candidates if c["quality"] == "noisy"]
    clean = [c for c in candidates if c["quality"] in ("clean", "partial")]
    other = [c for c in candidates if c["quality"] not in ("noisy", "clean", "partial")]

    noisy.sort(key=lambda x: -x["seg_count"])
    clean.sort(key=lambda x: -x["seg_count"])
    other.sort(key=lambda x: -x["seg_count"])

    selected = []
    half = limit // 2

    for lst, take in [(noisy, half), (clean, half), (other, limit)]:
        for c in lst:
            if len(selected) >= limit:
                break
            if c["id"] not in [s for s in selected]:
                selected.append(c["id"])
                take -= 1
                if take <= 0:
                    break

    return selected[:limit]


def load_session(session_id):
    """Load transcript and return (session_meta, segments)."""
    path = RECORDINGS_DIR / session_id / "transcript.json"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("session", {}), data.get("segments", [])
    return {}, data


def get_audio_path(session_id):
    """Return path to audio file."""
    sess_dir = RECORDINGS_DIR / session_id
    for name in ("audio.ogg", "audio.wav"):
        p = sess_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_as_float32(audio_path, target_sr=16000, normalize_dbfs=-16.0):
    """Convert audio to float32 numpy array at target sample rate.

    Applies loudness normalization to match the Bridge's preprocessing
    pipeline (AudioPreprocessor with NORMALIZE_TARGET_DBFS=-16.0).
    """
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ar", str(target_sr), "-ac", "1",
             "-acodec", "pcm_s16le", "-f", "wav", tmp.name],
            capture_output=True, check=True, timeout=60,
        )
        import wave
        with wave.open(tmp.name, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Apply loudness normalization (same as Bridge's AudioPreprocessor)
        rms = np.sqrt(np.mean(samples**2))
        if rms > 1e-10 and normalize_dbfs is not None:
            current_dbfs = 20 * np.log10(rms + 1e-10)
            gain_db = normalize_dbfs - current_dbfs
            gain_linear = 10 ** (gain_db / 20)
            samples = np.clip(samples * gain_linear, -1.0, 1.0).astype(np.float32)

        return samples
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Whisper model operations
# ---------------------------------------------------------------------------

def load_whisper_model(model_name, compute_type="int8_float16"):
    """Load a faster-whisper model."""
    from faster_whisper import WhisperModel

    print(f"  Loading model: {model_name} ({compute_type})...")
    t0 = time.time()
    model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
    elapsed = time.time() - t0
    print(f"  Model loaded in {elapsed:.1f}s")
    return model


def transcribe_with_model(model, audio_np, hotwords="", beam_size=5):
    """Run transcription and return segments list."""
    transcribe_kwargs = dict(
        beam_size=beam_size,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            threshold=0.5,
        ),
        word_timestamps=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=5,
        patience=2,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-0.5,
        no_speech_threshold=0.6,
    )
    if hotwords:
        # NOTE: distil-whisper models (e.g. kotoba-whisper) may not support hotwords.
        # If transcription returns 0 segments with hotwords, retry without them.
        transcribe_kwargs["hotwords"] = hotwords

    t0 = time.time()
    segments, info = model.transcribe(audio_np, **transcribe_kwargs)
    segments = list(segments)  # consume generator

    # Fallback: if hotwords caused 0 results, retry without them
    if not segments and hotwords:
        del transcribe_kwargs["hotwords"]
        segments, info = model.transcribe(audio_np, **transcribe_kwargs)
        segments = list(segments)

    results = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        results.append({
            "text": text,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "avg_logprob": round(seg.avg_logprob, 3),
            "no_speech_prob": round(seg.no_speech_prob, 3),
        })

    elapsed = time.time() - t0
    return results, elapsed


def unload_model(model):
    """Free GPU memory."""
    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref_segments, hyp_segments):
    """Compare reference (original) vs hypothesis (candidate) segments.

    Returns dict with similarity metrics.
    """
    ref_text = " ".join(s.get("text", "") for s in ref_segments)
    hyp_text = " ".join(s.get("text", "") for s in hyp_segments)

    # Character-level similarity (SequenceMatcher)
    sm = difflib.SequenceMatcher(None, ref_text, hyp_text)
    char_similarity = sm.ratio()

    # Char counts
    ref_chars = len(ref_text.replace(" ", ""))
    hyp_chars = len(hyp_text.replace(" ", ""))

    # Segment counts
    ref_seg_count = len(ref_segments)
    hyp_seg_count = len(hyp_segments)

    # Average logprob
    ref_lp = [s.get("avg_logprob", -0.5) for s in ref_segments if s.get("avg_logprob") is not None]
    hyp_lp = [s.get("avg_logprob", -0.5) for s in hyp_segments if s.get("avg_logprob") is not None]
    ref_avg_logprob = sum(ref_lp) / len(ref_lp) if ref_lp else -0.5
    hyp_avg_logprob = sum(hyp_lp) / len(hyp_lp) if hyp_lp else -0.5

    # Coverage: time overlap ratio
    ref_coverage = set()
    for s in ref_segments:
        start_ms = int(s.get("start", 0) * 10)
        end_ms = int(s.get("end", 0) * 10)
        ref_coverage.update(range(start_ms, end_ms))
    hyp_coverage = set()
    for s in hyp_segments:
        start_ms = int(s.get("start", 0) * 10)
        end_ms = int(s.get("end", 0) * 10)
        hyp_coverage.update(range(start_ms, end_ms))
    overlap = len(ref_coverage & hyp_coverage)
    coverage_ratio = overlap / len(ref_coverage) if ref_coverage else 0

    # Notable differences (difflib unified diff style)
    diffs = []
    ref_lines = [s.get("text", "") for s in ref_segments]
    hyp_lines = [s.get("text", "") for s in hyp_segments]
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref_lines, hyp_lines).get_opcodes():
        if tag == "equal":
            continue
        ref_chunk = " ".join(ref_lines[i1:i2])
        hyp_chunk = " ".join(hyp_lines[j1:j2])
        if ref_chunk or hyp_chunk:
            diffs.append({
                "type": tag,
                "ref": ref_chunk[:200],
                "hyp": hyp_chunk[:200],
            })

    return {
        "char_similarity": round(char_similarity, 4),
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "ref_segments": ref_seg_count,
        "hyp_segments": hyp_seg_count,
        "ref_avg_logprob": round(ref_avg_logprob, 3),
        "hyp_avg_logprob": round(hyp_avg_logprob, 3),
        "coverage_ratio": round(coverage_ratio, 4),
        "notable_differences": diffs[:10],
    }


# ---------------------------------------------------------------------------
# Hotwords
# ---------------------------------------------------------------------------

def build_hotwords():
    """Build hotwords string from config."""
    try:
        from hotword_loader import HotwordLoader
        hl = HotwordLoader(str(CONFIG_DIR))
        words = hl.get_hotwords_list()
        custom = hl.get_custom_words()
        all_words = list(custom) + [w for w in words if w not in custom]
        return ", ".join(all_words[:100])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Whisper models")
    parser.add_argument("--model", required=True, help="Candidate model name (e.g. kotoba-tech/kotoba-whisper-v2.0-faster)")
    parser.add_argument("--baseline", default="large-v3", help="Baseline model (default: use existing transcripts from large-v3)")
    parser.add_argument("--compute-type", default="int8_float16", help="Compute type (default: int8_float16)")
    parser.add_argument("--sessions", type=int, default=8, help="Number of sessions to compare")
    parser.add_argument("--ids", nargs="+", help="Specific session IDs")
    parser.add_argument("--max-duration", type=int, default=120, help="Max audio duration in seconds")
    parser.add_argument("--quality", choices=["noisy", "clean"], help="Filter by quality tag")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for candidate model")
    parser.add_argument("--retranscribe-baseline", action="store_true",
                        help="Re-transcribe with baseline model instead of using saved transcripts")
    parser.add_argument("--output", help="Output JSON path (default: config/compare_models_<timestamp>.json)")
    args = parser.parse_args()

    print(f"\n{BOLD}=== Whisper Model Comparison ==={RESET}")
    print(f"  Candidate: {CYAN}{args.model}{RESET}")
    print(f"  Baseline:  {args.baseline} {'(re-transcribe)' if args.retranscribe_baseline else '(saved transcripts)'}")
    print(f"  Compute:   {args.compute_type}")
    print()

    # Find sessions
    print(f"{BOLD}Finding sessions...{RESET}")
    session_ids = find_sessions(
        limit=args.sessions,
        specific=args.ids,
        max_duration=args.max_duration,
        quality_filter=args.quality,
    )
    if not session_ids:
        print(f"{RED}No valid sessions found!{RESET}")
        sys.exit(1)

    print(f"  Selected {len(session_ids)} sessions:")
    for sid in session_ids:
        meta, segs = load_session(sid)
        quality = meta.get("tags", {}).get("quality", "?")
        dur = meta.get("duration_seconds", 0)
        print(f"    {sid}  ({dur:.0f}s, {len(segs)} segs, quality={quality})")
    print()

    # Build hotwords
    hotwords = build_hotwords()
    if hotwords:
        hw_count = len(hotwords.split(", "))
        print(f"  Hotwords: {hw_count} words loaded")

    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().split("\n")[0]
        free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        print(f"  GPU: {free_mb} MB free / {total_mb} MB total")
    except Exception:
        pass
    print()

    # Load candidate model
    print(f"{BOLD}Loading candidate model...{RESET}")
    candidate_model = load_whisper_model(args.model, args.compute_type)

    # Optionally load baseline model
    baseline_model = None
    if args.retranscribe_baseline:
        print(f"\n{BOLD}Loading baseline model...{RESET}")
        baseline_model = load_whisper_model(args.baseline, args.compute_type)

    # Run comparisons
    print(f"\n{BOLD}Running comparisons...{RESET}\n")
    results = []

    for i, sid in enumerate(session_ids, 1):
        print(f"  [{i}/{len(session_ids)}] {sid}")
        meta, ref_segments = load_session(sid)
        audio_path = get_audio_path(sid)
        if not audio_path:
            print(f"    {YELLOW}SKIP: no audio file{RESET}")
            continue

        try:
            audio_np = load_audio_as_float32(audio_path, target_sr=16000)
            audio_duration = len(audio_np) / 16000
            print(f"    Audio: {audio_duration:.1f}s")
        except Exception as e:
            print(f"    {RED}Audio load failed: {e}{RESET}")
            continue

        # Baseline
        if baseline_model:
            print(f"    Transcribing with {args.baseline}...")
            ref_segments_raw, ref_elapsed = transcribe_with_model(
                baseline_model, audio_np, hotwords=hotwords, beam_size=args.beam_size
            )
            ref_text_for_compare = ref_segments_raw
        else:
            # Use saved transcript (strip LLM corrections by using raw text if available)
            ref_text_for_compare = ref_segments
            ref_elapsed = 0

        # Candidate
        print(f"    Transcribing with {args.model}...")
        hyp_segments, hyp_elapsed = transcribe_with_model(
            candidate_model, audio_np, hotwords=hotwords, beam_size=args.beam_size
        )

        # Compare
        metrics = compute_metrics(ref_text_for_compare, hyp_segments)
        quality = meta.get("tags", {}).get("quality", "unknown")

        # Color-code similarity
        sim = metrics["char_similarity"]
        if sim >= 0.8:
            sim_color = GREEN
        elif sim >= 0.5:
            sim_color = YELLOW
        else:
            sim_color = RED

        logprob_delta = metrics["hyp_avg_logprob"] - metrics["ref_avg_logprob"]
        lp_color = GREEN if logprob_delta > 0 else (YELLOW if logprob_delta > -0.05 else RED)

        print(f"    Similarity: {sim_color}{sim:.1%}{RESET}  "
              f"LogProb: ref={metrics['ref_avg_logprob']:.3f} → hyp={lp_color}{metrics['hyp_avg_logprob']:.3f}{RESET} "
              f"({'+' if logprob_delta >= 0 else ''}{logprob_delta:.3f})  "
              f"Speed: {hyp_elapsed:.1f}s ({audio_duration/hyp_elapsed:.1f}x RT)")

        # Show top differences
        for d in metrics["notable_differences"][:3]:
            if d["ref"] and d["hyp"]:
                print(f"      {DIM}ref: {d['ref'][:80]}{RESET}")
                print(f"      {CYAN}hyp: {d['hyp'][:80]}{RESET}")

        session_result = {
            "session_id": sid,
            "quality": quality,
            "audio_duration_s": round(audio_duration, 1),
            "baseline_elapsed_s": round(ref_elapsed, 2),
            "candidate_elapsed_s": round(hyp_elapsed, 2),
            "rtf": round(hyp_elapsed / audio_duration, 3) if audio_duration > 0 else 0,
            "metrics": metrics,
        }
        results.append(session_result)
        print()

    # Cleanup models
    unload_model(candidate_model)
    if baseline_model:
        unload_model(baseline_model)

    # Aggregate metrics
    if results:
        avg_sim = sum(r["metrics"]["char_similarity"] for r in results) / len(results)
        avg_ref_lp = sum(r["metrics"]["ref_avg_logprob"] for r in results) / len(results)
        avg_hyp_lp = sum(r["metrics"]["hyp_avg_logprob"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        total_audio = sum(r["audio_duration_s"] for r in results)
        total_candidate_time = sum(r["candidate_elapsed_s"] for r in results)

        # Per-quality breakdown
        quality_groups = {}
        for r in results:
            q = r["quality"]
            quality_groups.setdefault(q, []).append(r)

        print(f"{BOLD}{'='*60}")
        print(f"SUMMARY: {args.model} vs {args.baseline}")
        print(f"{'='*60}{RESET}")
        print(f"  Sessions:         {len(results)}")
        print(f"  Total audio:      {total_audio:.0f}s")
        print(f"  Avg similarity:   {avg_sim:.1%}")
        print(f"  Avg logprob:      ref={avg_ref_lp:.3f} → hyp={avg_hyp_lp:.3f} "
              f"({'+' if avg_hyp_lp - avg_ref_lp >= 0 else ''}{avg_hyp_lp - avg_ref_lp:.3f})")
        print(f"  Avg RTF:          {avg_rtf:.3f} ({1/avg_rtf:.1f}x realtime)")
        print(f"  Total time:       {total_candidate_time:.1f}s")

        if len(quality_groups) > 1:
            print(f"\n  {BOLD}Per-quality breakdown:{RESET}")
            for q, group in sorted(quality_groups.items()):
                q_sim = sum(r["metrics"]["char_similarity"] for r in group) / len(group)
                q_hyp_lp = sum(r["metrics"]["hyp_avg_logprob"] for r in group) / len(group)
                q_ref_lp = sum(r["metrics"]["ref_avg_logprob"] for r in group) / len(group)
                print(f"    {q:12s}: sim={q_sim:.1%}  "
                      f"logprob={q_ref_lp:.3f}→{q_hyp_lp:.3f}  "
                      f"({len(group)} sessions)")

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(CONFIG_DIR / f"compare_models_{ts}.json")

    report = {
        "candidate_model": args.model,
        "baseline_model": args.baseline,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "retranscribed_baseline": args.retranscribe_baseline,
        "timestamp": datetime.now().isoformat(),
        "session_count": len(results),
        "summary": {
            "avg_char_similarity": round(avg_sim, 4) if results else 0,
            "avg_ref_logprob": round(avg_ref_lp, 3) if results else 0,
            "avg_hyp_logprob": round(avg_hyp_lp, 3) if results else 0,
            "avg_rtf": round(avg_rtf, 3) if results else 0,
        },
        "sessions": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved: {output_path}")


if __name__ == "__main__":
    main()
