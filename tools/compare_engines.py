#!/usr/bin/env python3
"""Compare faster-whisper and Qwen3-ASR transcription on real Omi recordings.

Runs each engine sequentially (RTX 3060 12GB cannot fit both simultaneously),
then prints a side-by-side comparison report.

Usage (inside Docker container):
  python tools/compare_engines.py                         # 5 random sessions
  python tools/compare_engines.py --count 10              # 10 random sessions
  python tools/compare_engines.py --sessions 2026-03-28/02-14-51,2026-03-29/15-30-00
  python tools/compare_engines.py --output results.json   # save JSON report
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retranscribe_all import find_all_sessions, load_audio_as_pcm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(
    recordings_dir: str,
    count: int,
    session_names: list[str] | None = None,
) -> list[dict]:
    """Select sessions for comparison.

    If session_names is given, use those specifically.
    Otherwise pick `count` random sessions that have audio files,
    preferring diversity in date and audio length.
    """
    all_sessions = find_all_sessions(recordings_dir)
    if not all_sessions:
        log.error("No sessions found in %s", recordings_dir)
        return []

    if session_names:
        selected = []
        for name in session_names:
            name = name.strip()
            match = [s for s in all_sessions if s["name"] == name]
            if match:
                selected.append(match[0])
            else:
                log.warning("Session not found: %s", name)
        return selected

    # Random selection with diversity
    if len(all_sessions) <= count:
        return all_sessions

    random.shuffle(all_sessions)
    return all_sessions[:count]


def get_audio_duration(pcm_bytes: bytes) -> float:
    """Return duration in seconds for PCM 16-bit LE data."""
    return len(pcm_bytes) / (SAMPLE_RATE * 2)  # 2 bytes per sample


# ---------------------------------------------------------------------------
# Phase 1: faster-whisper
# ---------------------------------------------------------------------------

def run_faster_whisper(samples: list[dict]) -> list[dict]:
    """Transcribe all samples with faster-whisper, then unload."""
    from faster_whisper import WhisperModel
    import omi_whisper_bridge

    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    compute_type = os.environ.get("COMPUTE_TYPE", "int8_float16")
    device = os.environ.get("DEVICE", "cuda")

    log.info("=" * 60)
    log.info("[Phase 1] Loading faster-whisper: %s (%s)", model_size, compute_type)
    t_load = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    load_time = time.time() - t_load
    log.info("  Model loaded in %.1fs", load_time)

    results = []
    for i, sample in enumerate(samples, 1):
        log.info("  [%d/%d] %s", i, len(samples), sample["name"])
        pcm = load_audio_as_pcm(sample["audio"])
        if pcm is None:
            results.append({"error": "audio load failed"})
            continue

        duration = get_audio_duration(pcm)
        t0 = time.time()
        try:
            segments = omi_whisper_bridge.transcribe_audio(
                pcm, model, session_offset=0.0, hotwords="",
            )
        except Exception as e:
            log.error("    Transcription failed: %s", e)
            results.append({"error": str(e), "duration": duration})
            continue
        elapsed = time.time() - t0

        results.append({
            "segments": segments,
            "duration": duration,
            "processing_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else 0,
        })
        log.info(
            "    %d segments, %.1fs processing (RTF: %.3fx)",
            len(segments), elapsed, results[-1]["rtf"],
        )

    # Unload
    log.info("  Unloading faster-whisper...")
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    log.info("  VRAM released")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Qwen3-ASR
# ---------------------------------------------------------------------------

def run_qwen_asr(samples: list[dict]) -> list[dict]:
    """Transcribe all samples with Qwen3-ASR, then unload."""
    import asyncio
    from qwen_asr_engine import QwenASREngine
    from term_corrector import TermCorrector

    engine = QwenASREngine()
    config_dir = os.environ.get("CONFIG_DIR", "./config")
    tc = TermCorrector(os.path.join(config_dir, "term_corrections.json"))
    engine.set_term_corrector(tc)

    log.info("=" * 60)
    log.info("[Phase 2] Loading Qwen3-ASR: %s", engine.MODEL_NAME)
    t_load = time.time()
    asyncio.get_event_loop().run_until_complete(engine.load())
    load_time = time.time() - t_load
    log.info("  Model loaded in %.1fs", load_time)

    results = []
    for i, sample in enumerate(samples, 1):
        log.info("  [%d/%d] %s", i, len(samples), sample["name"])
        pcm = load_audio_as_pcm(sample["audio"])
        if pcm is None:
            results.append({"error": "audio load failed"})
            continue

        duration = get_audio_duration(pcm)
        t0 = time.time()
        try:
            segments = engine.transcribe(
                pcm, session_offset=0.0, hotwords="",
            )
        except Exception as e:
            log.error("    Transcription failed: %s", e)
            results.append({"error": str(e), "duration": duration})
            continue
        elapsed = time.time() - t0

        results.append({
            "segments": segments,
            "duration": duration,
            "processing_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else 0,
        })
        log.info(
            "    %d segments, %.1fs processing (RTF: %.3fx)",
            len(segments), elapsed, results[-1]["rtf"],
        )

    # Unload
    log.info("  Unloading Qwen3-ASR...")
    asyncio.get_event_loop().run_until_complete(engine.unload())
    log.info("  VRAM released")

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    samples: list[dict],
    whisper_results: list[dict],
    qwen_results: list[dict],
) -> dict:
    """Print comparison report and return structured data."""
    print()
    print("=" * 70)
    print("  Engine Comparison: faster-whisper vs qwen3-asr")
    print(f"  Samples: {len(samples)} sessions")
    print("=" * 70)

    report_sessions = []
    total_whisper_time = 0.0
    total_qwen_time = 0.0
    total_audio_duration = 0.0
    total_whisper_segs = 0
    total_qwen_segs = 0

    for i, sample in enumerate(samples):
        wr = whisper_results[i]
        qr = qwen_results[i]
        duration = wr.get("duration", qr.get("duration", 0))

        print()
        print(f"--- Session: {sample['name']} ({duration:.1f}s audio) ---")
        print()

        session_data = {
            "name": sample["name"],
            "audio_duration": duration,
        }

        # faster-whisper results
        if "error" in wr:
            print(f"  [faster-whisper] ERROR: {wr['error']}")
            session_data["whisper"] = {"error": wr["error"]}
        else:
            segs = wr["segments"]
            print(
                f"  [faster-whisper] {len(segs)} segments, "
                f"{wr['processing_time']:.1f}s processing "
                f"(RTF: {wr['rtf']:.3f}x)"
            )
            for seg in segs[:10]:  # Show first 10
                print(f"    [{seg['start']:.2f}-{seg['end']:.2f}] {seg['text'][:70]}")
            if len(segs) > 10:
                print(f"    ... ({len(segs) - 10} more segments)")
            total_whisper_time += wr["processing_time"]
            total_whisper_segs += len(segs)
            session_data["whisper"] = {
                "segments": segs,
                "processing_time": wr["processing_time"],
                "rtf": wr["rtf"],
            }

        print()

        # Qwen3-ASR results
        if "error" in qr:
            print(f"  [qwen3-asr] ERROR: {qr['error']}")
            session_data["qwen"] = {"error": qr["error"]}
        else:
            segs = qr["segments"]
            print(
                f"  [qwen3-asr] {len(segs)} segments, "
                f"{qr['processing_time']:.1f}s processing "
                f"(RTF: {qr['rtf']:.3f}x)"
            )
            for seg in segs[:10]:
                print(f"    [{seg['start']:.2f}-{seg['end']:.2f}] {seg['text'][:70]}")
            if len(segs) > 10:
                print(f"    ... ({len(segs) - 10} more segments)")
            total_qwen_time += qr["processing_time"]
            total_qwen_segs += len(segs)
            session_data["qwen"] = {
                "segments": segs,
                "processing_time": qr["processing_time"],
                "rtf": qr["rtf"],
            }

        total_audio_duration += duration

        # Existing transcript for reference
        transcript_path = sample.get("transcript")
        if transcript_path and os.path.exists(transcript_path):
            try:
                with open(transcript_path) as f:
                    existing = json.load(f)
                existing_segs = existing.get("segments", [])
                if existing_segs:
                    print()
                    print(
                        f"  [existing transcript] {len(existing_segs)} segments (reference)"
                    )
                    for seg in existing_segs[:5]:
                        text = seg.get("text", "")
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        print(f"    [{start:.2f}-{end:.2f}] {text[:70]}")
                    if len(existing_segs) > 5:
                        print(f"    ... ({len(existing_segs) - 5} more)")
            except (json.JSONDecodeError, OSError):
                pass

        report_sessions.append(session_data)

    # Summary table
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()

    w_rtf = (
        round(total_whisper_time / total_audio_duration, 4)
        if total_audio_duration > 0 else 0
    )
    q_rtf = (
        round(total_qwen_time / total_audio_duration, 4)
        if total_audio_duration > 0 else 0
    )

    header = f"  {'Metric':<25} {'faster-whisper':>15} {'qwen3-asr':>15}"
    sep = f"  {'-'*25} {'-'*15} {'-'*15}"
    print(header)
    print(sep)
    print(f"  {'Total audio (s)':<25} {total_audio_duration:>15.1f} {total_audio_duration:>15.1f}")
    print(f"  {'Total processing (s)':<25} {total_whisper_time:>15.1f} {total_qwen_time:>15.1f}")
    print(f"  {'Overall RTF':<25} {w_rtf:>15.4f}x {q_rtf:>15.4f}x")
    print(f"  {'Total segments':<25} {total_whisper_segs:>15} {total_qwen_segs:>15}")
    if total_audio_duration > 0:
        w_seg_rate = round(total_whisper_segs / total_audio_duration, 2)
        q_seg_rate = round(total_qwen_segs / total_audio_duration, 2)
        print(f"  {'Segments/s audio':<25} {w_seg_rate:>15.2f} {q_seg_rate:>15.2f}")
    print()

    return {
        "samples": len(samples),
        "total_audio_duration": total_audio_duration,
        "summary": {
            "faster_whisper": {
                "total_processing_time": total_whisper_time,
                "overall_rtf": w_rtf,
                "total_segments": total_whisper_segs,
            },
            "qwen3_asr": {
                "total_processing_time": total_qwen_time,
                "overall_rtf": q_rtf,
                "total_segments": total_qwen_segs,
            },
        },
        "sessions": report_sessions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper and Qwen3-ASR on real Omi recordings"
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of random sessions to compare (default: 5)",
    )
    parser.add_argument(
        "--sessions", type=str, default=None,
        help="Comma-separated session names (e.g. 2026-03-28/02-14-51)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--recordings-dir", type=str, default=RECORDINGS_DIR,
        help=f"Recordings directory (default: {RECORDINGS_DIR})",
    )
    args = parser.parse_args()

    session_names = None
    if args.sessions:
        session_names = [s.strip() for s in args.sessions.split(",")]

    samples = select_samples(
        args.recordings_dir,
        count=args.count,
        session_names=session_names,
    )

    if not samples:
        log.error("No valid samples found. Exiting.")
        sys.exit(1)

    log.info("Selected %d sessions for comparison", len(samples))

    # Pre-load all audio to verify availability before starting engines
    valid_samples = []
    for s in samples:
        pcm = load_audio_as_pcm(s["audio"])
        if pcm is not None:
            dur = get_audio_duration(pcm)
            log.info("  %s (%.1fs audio)", s["name"], dur)
            valid_samples.append(s)
        else:
            log.warning("  %s — SKIPPED (audio load failed)", s["name"])

    if not valid_samples:
        log.error("No valid audio found. Exiting.")
        sys.exit(1)

    samples = valid_samples

    # Phase 1: faster-whisper
    whisper_results = run_faster_whisper(samples)

    # Phase 2: Qwen3-ASR
    qwen_results = run_qwen_asr(samples)

    # Report
    report = print_report(samples, whisper_results, qwen_results)

    # Save JSON if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
