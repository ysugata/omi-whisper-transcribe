#!/usr/bin/env python3
"""Compare Whisper vs VibeVoice-ASR vs Qwen3-ASR transcription quality.

Runs all three models on the same sessions and compares output text,
segment counts, and RTF.

Usage (inside container):
    python3 /app/tools/benchmark_qwen3_vs_vibevoice.py --ids ID1 ID2 ...
    python3 /app/tools/benchmark_qwen3_vs_vibevoice.py --ids ID1 --skip-whisper

Output: JSON + terminal comparison table.
"""

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", str(PROJECT_ROOT / "recordings"))


def _build_vv_prompt():
    """Build VibeVoice-ASR prompt with hotwords."""
    try:
        from hotword_loader import HotwordLoader
        loader = HotwordLoader(os.environ.get("CONFIG_DIR", str(PROJECT_ROOT / "config")))
        custom = loader.get_custom_words()
        hw_list = loader.get_hotwords_list()
        words = list(custom)
        for w in hw_list:
            if w not in words:
                words.append(w)
            if len(words) >= 100:
                break
        hotwords_str = ", ".join(words) if words else ""
    except Exception:
        hotwords_str = ""

    vv_prompt = "日本語の会話を書き起こしてください。"
    if hotwords_str:
        vv_prompt += f"\nHotwords: {hotwords_str}"
    return vv_prompt


def _text_similarity(text_a: str, text_b: str) -> float:
    """Character-level similarity between two texts."""
    if text_a and text_b:
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()
    elif not text_a and not text_b:
        return 1.0
    return 0.0


def _run_vibevoice(sessions, prompt):
    """Run VibeVoice-ASR on all sessions."""
    from vibevoice_transcriber import VibeVoiceTranscriber, load_audio

    transcriber = VibeVoiceTranscriber()
    print("Loading VibeVoice-ASR...", flush=True)
    t0 = time.time()
    transcriber.load_model()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", flush=True)

    results = {}
    for sess in sessions:
        sid = sess["session_id"]
        wav_path = transcriber.convert_to_wav(str(sess["audio_path"]))
        try:
            audio_data, sr = load_audio(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        duration = len(audio_data) / sr
        print(f"  VibeVoice: {sid} ({duration:.1f}s)...", end=" ", flush=True)
        t0 = time.time()
        segments = transcriber.transcribe_audio(audio_data, sr, prompt)
        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0

        text = " ".join(s.get("text", "") for s in (segments or []))
        results[sid] = {
            "text": text,
            "seg_count": len(segments or []),
            "elapsed_s": round(elapsed, 2),
            "rtf": round(rtf, 3),
        }
        print(f"{elapsed:.2f}s (RTF={rtf:.3f})", flush=True)

    transcriber.unload_model()
    return results


def _run_qwen3(sessions):
    """Run Qwen3-ASR on all sessions."""
    from qwen3_transcriber import Qwen3ASRTranscriber

    transcriber = Qwen3ASRTranscriber()
    print("Loading Qwen3-ASR...", flush=True)
    t0 = time.time()
    transcriber.load_model(use_aligner=True)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", flush=True)

    results = {}
    for sess in sessions:
        sid = sess["session_id"]
        audio_path = sess["audio_path"]
        duration = sess["duration_s"]

        print(f"  Qwen3: {sid} ({duration:.1f}s)...", end=" ", flush=True)
        t0 = time.time()
        segments = transcriber.transcribe(str(audio_path))
        elapsed = time.time() - t0
        rtf = elapsed / duration if duration > 0 else 0

        text = " ".join(s.get("text", "") for s in segments)
        results[sid] = {
            "text": text,
            "seg_count": len(segments),
            "elapsed_s": round(elapsed, 2),
            "rtf": round(rtf, 3),
        }
        print(f"{elapsed:.2f}s (RTF={rtf:.3f})", flush=True)

    transcriber.unload_model()
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Whisper vs VibeVoice vs Qwen3-ASR")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="Session IDs (e.g. 2026-03-27/20-44-00)")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper baseline (use existing transcript)")
    parser.add_argument("--skip-vibevoice", action="store_true",
                        help="Skip VibeVoice (use existing vibevoice_segments)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Collect sessions
    sessions = []
    for session_id in args.ids:
        session_dir = Path(RECORDINGS_DIR) / session_id
        transcript_path = session_dir / "transcript.json"

        if not transcript_path.exists():
            print(f"SKIP {session_id}: no transcript.json", flush=True)
            continue

        audio_path = session_dir / "audio.ogg"
        if not audio_path.exists():
            audio_path = session_dir / "audio.wav"
        if not audio_path.exists():
            print(f"SKIP {session_id}: no audio file", flush=True)
            continue

        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)

        # Calculate duration from Whisper segments
        whisper_segs = data.get("segments", [])
        whisper_text = " ".join(s.get("text", "") for s in whisper_segs)
        vv_segs = data.get("vibevoice_segments", [])
        vv_text = " ".join(s.get("text", "") for s in vv_segs)

        # Estimate duration from last segment end time
        duration_s = 0
        for segs in [whisper_segs, vv_segs]:
            for s in segs:
                end = float(s.get("end", 0))
                if end > duration_s:
                    duration_s = end
        if duration_s == 0:
            duration_s = 60  # fallback

        sessions.append({
            "session_id": session_id,
            "audio_path": audio_path,
            "duration_s": duration_s,
            "whisper_text": whisper_text,
            "whisper_seg_count": len(whisper_segs),
            "existing_vv_text": vv_text,
            "existing_vv_seg_count": len(vv_segs),
        })

    if not sessions:
        print("No valid sessions found.")
        sys.exit(1)

    print(f"\nPrepared {len(sessions)} sessions\n", flush=True)

    # Run models
    vv_results = {}
    qwen3_results = {}
    vv_prompt = _build_vv_prompt()

    if not args.skip_vibevoice:
        vv_results = _run_vibevoice(sessions, vv_prompt)

    qwen3_results = _run_qwen3(sessions)

    # Build comparison
    comparison = []
    for sess in sessions:
        sid = sess["session_id"]
        whisper_text = sess["whisper_text"]

        # VibeVoice text
        if sid in vv_results:
            vv_text = vv_results[sid]["text"]
            vv_rtf = vv_results[sid]["rtf"]
            vv_segs = vv_results[sid]["seg_count"]
        else:
            vv_text = sess["existing_vv_text"]
            vv_rtf = None
            vv_segs = sess["existing_vv_seg_count"]

        # Qwen3 text
        qwen3_text = qwen3_results.get(sid, {}).get("text", "")
        qwen3_rtf = qwen3_results.get(sid, {}).get("rtf")
        qwen3_segs = qwen3_results.get(sid, {}).get("seg_count", 0)

        entry = {
            "session_id": sid,
            "duration_s": round(sess["duration_s"], 1),
            "whisper": {
                "char_count": len(whisper_text),
                "seg_count": sess["whisper_seg_count"],
            },
            "vibevoice": {
                "char_count": len(vv_text),
                "seg_count": vv_segs,
                "rtf": vv_rtf,
                "sim_to_whisper": round(_text_similarity(vv_text, whisper_text), 4),
            },
            "qwen3": {
                "char_count": len(qwen3_text),
                "seg_count": qwen3_segs,
                "rtf": qwen3_rtf,
                "sim_to_whisper": round(_text_similarity(qwen3_text, whisper_text), 4),
                "sim_to_vibevoice": round(_text_similarity(qwen3_text, vv_text), 4),
            },
        }
        comparison.append(entry)

    # Print table
    print(f"\n{'='*80}", flush=True)
    print("3-WAY COMPARISON", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Session':<25} {'Dur':>5} │ {'W_chars':>7} {'VV_chars':>8} {'Q3_chars':>8} │ "
          f"{'VV_RTF':>7} {'Q3_RTF':>7} │ {'VV~W':>5} {'Q3~W':>5} {'Q3~VV':>5}", flush=True)
    print("-" * 100, flush=True)
    for c in comparison:
        sid = c["session_id"]
        vv_rtf_str = f"{c['vibevoice']['rtf']:.3f}" if c['vibevoice']['rtf'] is not None else "N/A"
        q3_rtf_str = f"{c['qwen3']['rtf']:.3f}" if c['qwen3']['rtf'] is not None else "N/A"
        print(f"{sid:<25} {c['duration_s']:>5.0f} │ "
              f"{c['whisper']['char_count']:>7} {c['vibevoice']['char_count']:>8} "
              f"{c['qwen3']['char_count']:>8} │ "
              f"{vv_rtf_str:>7} {q3_rtf_str:>7} │ "
              f"{c['vibevoice']['sim_to_whisper']:.3f} "
              f"{c['qwen3']['sim_to_whisper']:.3f} "
              f"{c['qwen3']['sim_to_vibevoice']:.3f}", flush=True)

    # Summary averages
    if comparison:
        avg_vv_sim = sum(c["vibevoice"]["sim_to_whisper"] for c in comparison) / len(comparison)
        avg_q3_sim_w = sum(c["qwen3"]["sim_to_whisper"] for c in comparison) / len(comparison)
        avg_q3_sim_vv = sum(c["qwen3"]["sim_to_vibevoice"] for c in comparison) / len(comparison)

        vv_rtfs = [c["vibevoice"]["rtf"] for c in comparison if c["vibevoice"]["rtf"] is not None]
        q3_rtfs = [c["qwen3"]["rtf"] for c in comparison if c["qwen3"]["rtf"] is not None]

        print("-" * 100, flush=True)
        avg_vv_rtf = f"{sum(vv_rtfs)/len(vv_rtfs):.3f}" if vv_rtfs else "N/A"
        avg_q3_rtf = f"{sum(q3_rtfs)/len(q3_rtfs):.3f}" if q3_rtfs else "N/A"
        print(f"{'AVERAGE':<25} {'':>5} │ {'':>7} {'':>8} {'':>8} │ "
              f"{avg_vv_rtf:>7} {avg_q3_rtf:>7} │ "
              f"{avg_vv_sim:.3f} {avg_q3_sim_w:.3f} {avg_q3_sim_vv:.3f}", flush=True)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_ids": [s["session_id"] for s in sessions],
        "comparison": comparison,
    }

    out_path = args.output or str(
        PROJECT_ROOT / "config" / f"compare_qwen3_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
