#!/usr/bin/env python3
"""Test PANNs CNN14 audio scene classifier with real recorded audio.

Runs classification on many sessions and prints a readable summary.
Fixes AudioSet label display by importing labels correctly.
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/app")

from audio_scene_classifier import AudioSceneClassifier, _get_audioset_labels


def find_audio_file(session_dir: str):
    for ext in ("wav", "ogg"):
        path = os.path.join(session_dir, f"audio.{ext}")
        if os.path.exists(path):
            return path
    return None


def get_transcript_preview(session_dir: str, max_chars: int = 80) -> str:
    t_path = os.path.join(session_dir, "transcript.json")
    if not os.path.exists(t_path):
        return "(no transcript)"
    try:
        with open(t_path) as f:
            data = json.load(f)
        segments = data.get("segments", [])
        if not segments:
            return "(empty)"
        texts = [s.get("text", "") for s in segments[:5]]
        combined = " ".join(texts).strip()
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        return combined or "(empty text)"
    except Exception:
        return "(read error)"


def get_speaker_info(session_dir: str) -> str:
    """Get speaker info from transcript."""
    t_path = os.path.join(session_dir, "transcript.json")
    if not os.path.exists(t_path):
        return ""
    try:
        with open(t_path) as f:
            data = json.load(f)
        segments = data.get("segments", [])
        speakers = set()
        for s in segments:
            sp = s.get("speaker", "")
            if sp:
                speakers.add(sp)
        if speakers:
            return ",".join(sorted(speakers))
        return ""
    except Exception:
        return ""


def main():
    # Support running from host (tools/) or container (/app/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(script_dir, "recordings")):
        recordings_dir = os.path.join(script_dir, "recordings")
    else:
        recordings_dir = os.path.join(os.path.dirname(script_dir), "recordings")

    print(f"Recordings dir: {recordings_dir}", flush=True)

    # Collect all sessions with audio files, sorted by date
    import glob
    all_session_dirs = sorted(glob.glob(os.path.join(recordings_dir, "2026-*/*")))

    # Filter to sessions with non-empty audio
    valid_sessions = []
    for sd in all_session_dirs:
        af = find_audio_file(sd)
        if af and os.path.getsize(af) > 0:
            valid_sessions.append(sd)

    print(f"Total sessions with audio: {len(valid_sessions)}", flush=True)

    # Sample ~40 sessions: pick every Nth to get even spread
    max_sessions = 40
    if len(valid_sessions) > max_sessions:
        step = len(valid_sessions) // max_sessions
        sampled = [valid_sessions[i] for i in range(0, len(valid_sessions), step)][:max_sessions]
    else:
        sampled = valid_sessions

    # Ensure some specific interesting sessions are included
    must_include = [
        "2026-03-25/09-27-45",  # Long real conversation
        "2026-03-28/05-46-37",  # Anime/game
        "2026-04-02/19-46-41",  # Anime dialogue
        "2026-03-27/23-47-21",  # Tech discussion
        "2026-03-27/22-42-13",  # API pricing
        "2026-03-31/22-36-12",  # Story dialogue
        "2026-03-25/09-18-52",  # Possibly silence
        "2026-03-29/21-29-58",  # Gibberish labels
    ]
    sampled_rels = {os.path.relpath(s, recordings_dir) for s in sampled}
    for mi in must_include:
        if mi not in sampled_rels:
            full = os.path.join(recordings_dir, mi)
            if os.path.isdir(full):
                sampled.append(full)

    sampled.sort()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nTesting {len(sampled)} sessions (device={device})", flush=True)
    print("=" * 130, flush=True)

    classifier = AudioSceneClassifier(device=device, threshold=0.3, enabled=True)

    header = f"{'Session':<28} {'Size':>6} {'Scene':<10} {'Conf':>5} {'Spatial':<16} {'Conf':>5} {'Time':>5}  Top-5 AudioSet Classes"
    print(header, flush=True)
    print("-" * 130, flush=True)

    results = []
    for session_dir in sampled:
        session_rel = os.path.relpath(session_dir, recordings_dir)
        audio_path = find_audio_file(session_dir)
        if not audio_path:
            continue
        audio_size = os.path.getsize(audio_path)
        if audio_size == 0:
            continue

        t0 = time.time()
        result = classifier.classify(audio_path)
        elapsed = time.time() - t0

        if result is None:
            scene = "None"
            scene_conf = ""
            spatial = "None"
            spatial_conf = ""
            top = ""
        else:
            scene = result.get("scene_label") or "None"
            scene_conf = f"{result['scene_confidence']:.2f}" if result.get("scene_confidence") else ""
            spatial = result.get("spatial") or "None"
            spatial_conf = f"{result['spatial_confidence']:.2f}" if result.get("spatial_confidence") else ""
            top_classes = result.get("top_classes", [])
            top = ", ".join(f"{c['label']}({c['score']:.2f})" for c in top_classes[:5])

        line = (
            f"{session_rel:<28} {audio_size // 1024:>5}K "
            f"{scene:<10} {scene_conf:>5} "
            f"{spatial:<16} {spatial_conf:>5} "
            f"{elapsed:>4.1f}s  {top}"
        )
        print(line, flush=True)

        results.append({
            "session": session_rel,
            "size_kb": audio_size // 1024,
            "result": result,
            "elapsed": elapsed,
        })

    # Transcript previews
    print(flush=True)
    print("=" * 130, flush=True)
    print("Transcript Previews (with scene label):", flush=True)
    print("-" * 130, flush=True)
    for r in results:
        session_dir = os.path.join(recordings_dir, r["session"])
        preview = get_transcript_preview(session_dir)
        speakers = get_speaker_info(session_dir)
        res = r["result"]
        scene = (res.get("scene_label") or "?") if res else "?"
        sp_info = f" [{speakers}]" if speakers else ""
        print(f"  [{scene:<6}]{sp_info} {r['session']}: {preview}", flush=True)

    # Summary
    print(flush=True)
    print("=" * 130, flush=True)
    print("Summary:", flush=True)
    print(f"  Total tested: {len(results)}", flush=True)
    print(f"  Avg time per session: {sum(r['elapsed'] for r in results) / max(len(results), 1):.2f}s", flush=True)
    print(flush=True)

    scene_counts = {}
    spatial_counts = {}
    for r in results:
        res = r["result"]
        s = (res.get("scene_label_en") or "none") if res else "none"
        scene_counts[s] = scene_counts.get(s, 0) + 1
        sp = (res.get("spatial_en") or "none") if res else "none"
        spatial_counts[sp] = spatial_counts.get(sp, 0) + 1

    print("  Scene distribution:", flush=True)
    for label, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count} ({count * 100 // len(results)}%)", flush=True)

    print("  Spatial distribution:", flush=True)
    for label, count in sorted(spatial_counts.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count} ({count * 100 // len(results)}%)", flush=True)

    # Check if TV/media detection is working
    print(flush=True)
    print("  Sessions with high Television/Music scores (>0.3):", flush=True)
    for r in results:
        res = r["result"]
        if res and res.get("top_classes"):
            for c in res["top_classes"]:
                if c["label"] in ("Television", "Music", "Radio") and c["score"] > 0.3:
                    scene = res.get("scene_label") or "?"
                    print(f"    {r['session']}: {c['label']}={c['score']:.2f} (labeled as: {scene})", flush=True)

    classifier.unload()
    print(flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
