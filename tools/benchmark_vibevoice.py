#!/usr/bin/env python3
"""Benchmark VibeVoice-ASR inference speed and accuracy.

Runs VibeVoice-ASR on specified sessions, measures per-session timing,
and compares output text against existing vibevoice_segments for accuracy.

Usage (inside container):
    python3 /app/tools/benchmark_vibevoice.py --ids ID1 ID2 ...

Output: JSON with per-session timing + text comparison to previous results.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", str(PROJECT_ROOT / "recordings"))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True, help="Session IDs (e.g. 2026-03-27/20-44-00)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    from vibevoice_transcriber import VibeVoiceTranscriber, load_audio

    transcriber = VibeVoiceTranscriber()

    # Load model
    print("Loading VibeVoice-ASR model...", flush=True)
    t_load = time.time()
    transcriber.load_model()
    load_time = time.time() - t_load
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    # Build prompt (same as bridge)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
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

    results = []

    for session_id in args.ids:
        session_dir = Path(RECORDINGS_DIR) / session_id
        transcript_path = session_dir / "transcript.json"

        if not transcript_path.exists():
            print(f"SKIP {session_id}: no transcript.json", flush=True)
            continue

        # Find audio
        audio_path = session_dir / "audio.ogg"
        if not audio_path.exists():
            audio_path = session_dir / "audio.wav"
        if not audio_path.exists():
            print(f"SKIP {session_id}: no audio file", flush=True)
            continue

        # Load existing data for comparison
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
        old_segments = data.get("vibevoice_segments", [])

        # Convert + load audio
        wav_path = transcriber.convert_to_wav(str(audio_path))
        try:
            audio_data, sr = load_audio(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        duration_s = len(audio_data) / sr

        # Run inference with timing
        print(f"Transcribing {session_id} ({duration_s:.1f}s)...", end=" ", flush=True)
        t0 = time.time()
        segments = transcriber.transcribe_audio(audio_data, sr, vv_prompt)
        elapsed = time.time() - t0
        rtf = elapsed / duration_s if duration_s > 0 else 0

        # Compare text with previous results
        new_text = " ".join(s.get("text", "") for s in (segments or []))
        old_text = " ".join(s.get("text", "") for s in old_segments)

        # Simple character-level similarity
        if new_text and old_text:
            import difflib
            similarity = difflib.SequenceMatcher(None, new_text, old_text).ratio()
        elif not new_text and not old_text:
            similarity = 1.0
        else:
            similarity = 0.0

        result = {
            "session_id": session_id,
            "duration_s": round(duration_s, 1),
            "elapsed_s": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "seg_count": len(segments) if segments else 0,
            "char_count": len(new_text),
            "old_seg_count": len(old_segments),
            "old_char_count": len(old_text),
            "text_similarity_to_previous": round(similarity, 4),
        }
        results.append(result)

        print(f"{elapsed:.2f}s (RTF={rtf:.3f}), {len(segments or [])} segs, sim={similarity:.3f}", flush=True)

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
            "avg_text_similarity_to_previous": round(avg_sim, 4),
        }
        print(f"\n{'='*60}")
        print(f"Sessions: {len(results)}")
        print(f"Total duration: {total_dur:.1f}s")
        print(f"Total elapsed: {total_elapsed:.2f}s")
        print(f"Overall RTF: {total_elapsed/total_dur:.3f}")
        print(f"Avg similarity to previous: {avg_sim:.3f}")
    else:
        summary = {"session_count": 0}

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": summary,
    }

    out_path = args.output or str(PROJECT_ROOT / "config" / f"benchmark_vibevoice_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Unload
    transcriber.unload_model()


if __name__ == "__main__":
    main()
