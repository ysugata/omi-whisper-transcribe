#!/usr/bin/env python3
"""Benchmark VibeVoice-ASR across multiple configurations.

Groups tests to minimize model reloads:
  Group A (4-bit): baseline → prompt_lookup_5 → prompt_lookup_10
  Group B (8-bit): 8bit_baseline → 8bit+prompt_lookup_5

Usage (inside container):
    python3 /app/tools/benchmark_vibevoice_configs.py --ids ID1 ID2 ...
    python3 /app/tools/benchmark_vibevoice_configs.py --ids ID1 --configs 4bit 8bit
    python3 /app/tools/benchmark_vibevoice_configs.py --ids ID1 --configs 4bit_pl5 8bit_pl5

Output: JSON with per-config per-session timing + text comparison.
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

# Available configurations: (label, quantization, prompt_lookup_tokens)
ALL_CONFIGS = [
    ("4bit", "4bit", 0),
    ("4bit_pl5", "4bit", 5),
    ("4bit_pl10", "4bit", 10),
    ("8bit", "8bit", 0),
    ("8bit_pl5", "8bit", 5),
]


def _build_prompt():
    """Build Japanese prompt with hotwords (same as bridge)."""
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


def _prepare_sessions(session_ids):
    """Load and convert audio for all sessions. Returns list of dicts."""
    from vibevoice_transcriber import VibeVoiceTranscriber, load_audio

    sessions = []
    converter = VibeVoiceTranscriber()

    for session_id in session_ids:
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

        wav_path = converter.convert_to_wav(str(audio_path))
        try:
            audio_data, sr = load_audio(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        sessions.append({
            "session_id": session_id,
            "audio_data": audio_data,
            "sr": sr,
            "duration_s": len(audio_data) / sr,
            "old_vv_segments": data.get("vibevoice_segments", []),
        })

    return sessions


def _run_config(config_label, quant, pl_tokens, sessions, prompt):
    """Run transcription for one config across all sessions."""
    import vibevoice_transcriber
    from vibevoice_transcriber import VibeVoiceTranscriber

    # Set module-level config
    vibevoice_transcriber.QUANTIZATION_MODE = quant
    vibevoice_transcriber.PROMPT_LOOKUP_TOKENS = pl_tokens

    transcriber = VibeVoiceTranscriber()

    print(f"\n{'='*60}", flush=True)
    print(f"Config: {config_label} (quant={quant}, prompt_lookup={pl_tokens})", flush=True)
    print(f"{'='*60}", flush=True)

    t_load = time.time()
    transcriber.load_model()
    load_time = time.time() - t_load
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    results = []
    for sess in sessions:
        sid = sess["session_id"]
        duration_s = sess["duration_s"]

        print(f"  {sid} ({duration_s:.1f}s)...", end=" ", flush=True)
        t0 = time.time()
        segments = transcriber.transcribe_audio(
            sess["audio_data"], sess["sr"], prompt)
        elapsed = time.time() - t0
        rtf = elapsed / duration_s if duration_s > 0 else 0

        new_text = " ".join(s.get("text", "") for s in (segments or []))
        old_text = " ".join(s.get("text", "") for s in sess["old_vv_segments"])

        if new_text and old_text:
            similarity = difflib.SequenceMatcher(None, new_text, old_text).ratio()
        elif not new_text and not old_text:
            similarity = 1.0
        else:
            similarity = 0.0

        result = {
            "session_id": sid,
            "duration_s": round(duration_s, 1),
            "elapsed_s": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "seg_count": len(segments) if segments else 0,
            "char_count": len(new_text),
            "text_similarity_to_previous": round(similarity, 4),
        }
        results.append(result)
        print(f"{elapsed:.2f}s (RTF={rtf:.3f}), {len(segments or [])} segs, "
              f"sim={similarity:.3f}", flush=True)

    transcriber.unload_model()

    # Summary
    if results:
        total_dur = sum(r["duration_s"] for r in results)
        total_elapsed = sum(r["elapsed_s"] for r in results)
        avg_sim = sum(r["text_similarity_to_previous"] for r in results) / len(results)
        summary = {
            "session_count": len(results),
            "total_duration_s": round(total_dur, 1),
            "total_elapsed_s": round(total_elapsed, 2),
            "overall_rtf": round(total_elapsed / total_dur, 3) if total_dur > 0 else 0,
            "avg_similarity": round(avg_sim, 4),
            "model_load_s": round(load_time, 1),
        }
    else:
        summary = {"session_count": 0, "model_load_s": round(load_time, 1)}

    return {
        "config": config_label,
        "quantization": quant,
        "prompt_lookup_tokens": pl_tokens,
        "results": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark VibeVoice-ASR configurations")
    parser.add_argument("--ids", nargs="+", required=True,
                        help="Session IDs (e.g. 2026-03-27/20-44-00)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Config labels to test. Available: "
                             f"{', '.join(c[0] for c in ALL_CONFIGS)}")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Filter configs
    if args.configs:
        configs = [c for c in ALL_CONFIGS if c[0] in args.configs]
        if not configs:
            print(f"No matching configs. Available: {', '.join(c[0] for c in ALL_CONFIGS)}")
            sys.exit(1)
    else:
        configs = ALL_CONFIGS

    # Group by quantization to minimize reloads
    from collections import OrderedDict
    groups = OrderedDict()
    for label, quant, pl in configs:
        groups.setdefault(quant, []).append((label, quant, pl))

    prompt = _build_prompt()
    print(f"Prompt: {prompt[:80]}...", flush=True)

    print(f"\nPreparing {len(args.ids)} sessions...", flush=True)
    sessions = _prepare_sessions(args.ids)
    if not sessions:
        print("No valid sessions found.")
        sys.exit(1)
    print(f"Prepared {len(sessions)} sessions", flush=True)

    all_configs_results = []
    for quant_group, config_list in groups.items():
        print(f"\n{'#'*60}", flush=True)
        print(f"Group: {quant_group}", flush=True)
        print(f"{'#'*60}", flush=True)

        for label, quant, pl in config_list:
            config_result = _run_config(label, quant, pl, sessions, prompt)
            all_configs_results.append(config_result)

    # Print comparison table
    print(f"\n{'='*60}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Config':<15} {'RTF':>7} {'Elapsed':>8} {'Sim':>7} {'Segs':>6}", flush=True)
    print("-" * 50, flush=True)
    for cr in all_configs_results:
        s = cr["summary"]
        print(f"{cr['config']:<15} {s.get('overall_rtf',0):>7.3f} "
              f"{s.get('total_elapsed_s',0):>7.1f}s "
              f"{s.get('avg_similarity',0):>6.3f} "
              f"{sum(r['seg_count'] for r in cr['results']):>6}", flush=True)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_ids": [s["session_id"] for s in sessions],
        "configs": all_configs_results,
    }

    out_path = args.output or str(
        PROJECT_ROOT / "config" / f"benchmark_configs_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
