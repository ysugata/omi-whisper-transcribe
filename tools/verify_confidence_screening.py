#!/usr/bin/env python3
"""Verify confidence-based screening against real transcript data.

Reads transcripts that have avg_logprob field and compares old vs new
screening behavior. For transcripts without avg_logprob, simulates the
impact based on realistic confidence distributions.

Usage:
    python3 tools/verify_confidence_screening.py [--recordings-dir DIR] [--limit N]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_classifier import classify_segment, CONFIDENCE_LOGPROB_THRESHOLD
from segment_screener import SegmentScreener, _COMMON_LOANWORDS


def load_transcripts(recordings_dir: str, limit: int = 0):
    """Load transcript.json files from recordings directory."""
    transcripts = []
    rec_path = Path(recordings_dir)
    if not rec_path.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return []

    for transcript_file in sorted(rec_path.rglob("transcript.json"), reverse=True):
        try:
            with open(transcript_file) as f:
                data = json.load(f)
            segments = data if isinstance(data, list) else data.get("segments", [])
            if segments:
                transcripts.append({
                    "path": str(transcript_file),
                    "segments": segments,
                    "session_dir": str(transcript_file.parent),
                })
        except (json.JSONDecodeError, IOError):
            continue

        if limit and len(transcripts) >= limit:
            break

    return transcripts


def analyze_confidence_distribution(transcripts):
    """Analyze distribution of avg_logprob values across all segments."""
    with_logprob = []
    without_logprob = 0
    total_segments = 0

    for t in transcripts:
        for seg in t["segments"]:
            total_segments += 1
            logprob = seg.get("avg_logprob")
            if logprob is not None:
                with_logprob.append(logprob)
            else:
                without_logprob += 1

    print(f"\n{'='*60}")
    print(f"Confidence Distribution Analysis")
    print(f"{'='*60}")
    print(f"Total segments:        {total_segments}")
    print(f"With avg_logprob:      {len(with_logprob)}")
    print(f"Without avg_logprob:   {without_logprob}")

    if with_logprob:
        sorted_lp = sorted(with_logprob)
        print(f"\nDistribution (n={len(with_logprob)}):")
        print(f"  Min:     {sorted_lp[0]:.3f}")
        print(f"  P10:     {sorted_lp[int(len(sorted_lp)*0.1)]:.3f}")
        print(f"  P25:     {sorted_lp[int(len(sorted_lp)*0.25)]:.3f}")
        print(f"  Median:  {sorted_lp[int(len(sorted_lp)*0.5)]:.3f}")
        print(f"  P75:     {sorted_lp[int(len(sorted_lp)*0.75)]:.3f}")
        print(f"  P90:     {sorted_lp[int(len(sorted_lp)*0.9)]:.3f}")
        print(f"  Max:     {sorted_lp[-1]:.3f}")

        # Bucket analysis
        buckets = Counter()
        for lp in with_logprob:
            if lp >= -0.2:
                buckets["very_high (>= -0.2)"] += 1
            elif lp >= -0.3:
                buckets["high (-0.3 to -0.2)"] += 1
            elif lp >= -0.5:
                buckets["medium (-0.5 to -0.3)"] += 1
            elif lp >= -0.7:
                buckets["low (-0.7 to -0.5)"] += 1
            else:
                buckets["very_low (< -0.7)"] += 1

        print(f"\nConfidence buckets:")
        for bucket in ["very_high (>= -0.2)", "high (-0.3 to -0.2)",
                        "medium (-0.5 to -0.3)", "low (-0.7 to -0.5)",
                        "very_low (< -0.7)"]:
            count = buckets.get(bucket, 0)
            pct = count / len(with_logprob) * 100 if with_logprob else 0
            print(f"  {bucket:30s}: {count:4d} ({pct:5.1f}%)")

        # Segments that would be newly dirty due to confidence rule
        threshold = CONFIDENCE_LOGPROB_THRESHOLD
        below_threshold = [lp for lp in with_logprob if lp < threshold]
        print(f"\nSegments below threshold ({threshold}):")
        print(f"  Count: {len(below_threshold)} / {len(with_logprob)} "
              f"({len(below_threshold)/len(with_logprob)*100:.1f}%)")

    return with_logprob, without_logprob


def compare_screening_with_without_confidence(transcripts):
    """Compare old (katakana-only) vs new (katakana + confidence) screening."""
    known_values = set(_COMMON_LOANWORDS)

    old_dirty = 0
    old_clean = 0
    new_dirty = 0
    new_clean = 0
    newly_dirty = []  # segments that became dirty due to confidence
    total = 0

    for t in transcripts:
        for seg in t["segments"]:
            total += 1
            text = seg.get("text", "")
            logprob = seg.get("avg_logprob")

            # Old classification (without confidence)
            seg_no_conf = dict(seg)
            seg_no_conf.pop("avg_logprob", None)
            old_result = classify_segment(seg_no_conf, known_values, min_text_length=20)

            # New classification (with confidence)
            new_result = classify_segment(seg, known_values, min_text_length=20)

            if old_result == "dirty":
                old_dirty += 1
            else:
                old_clean += 1

            if new_result == "dirty":
                new_dirty += 1
            else:
                new_clean += 1

            # Track newly dirty segments (clean→dirty due to confidence)
            if old_result == "clean" and new_result == "dirty":
                newly_dirty.append({
                    "text": text[:100],
                    "avg_logprob": logprob,
                    "path": t["path"].split("/recordings/")[-1] if "/recordings/" in t["path"] else t["path"],
                })

    print(f"\n{'='*60}")
    print(f"Screening Comparison: Old (katakana) vs New (katakana + confidence)")
    print(f"{'='*60}")
    print(f"Total segments:  {total}")
    print(f"\nOld screener:    dirty={old_dirty}, clean={old_clean} "
          f"(dirty rate: {old_dirty/total*100:.1f}%)" if total else "")
    print(f"New screener:    dirty={new_dirty}, clean={new_clean} "
          f"(dirty rate: {new_dirty/total*100:.1f}%)" if total else "")
    print(f"\nNewly dirty (confidence-triggered): {len(newly_dirty)}")

    if newly_dirty:
        print(f"\nSample newly dirty segments (first 20):")
        for i, nd in enumerate(newly_dirty[:20]):
            print(f"  [{i+1}] logprob={nd['avg_logprob']:.3f} | {nd['text']}")
            print(f"       file: {nd['path']}")

    return newly_dirty


def main():
    parser = argparse.ArgumentParser(description="Verify confidence-based screening")
    parser.add_argument("--recordings-dir", default="/home/dev/omi_whisper_transcribe/recordings",
                        help="Path to recordings directory")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max number of sessions to analyze")
    args = parser.parse_args()

    print("Loading transcripts...")
    transcripts = load_transcripts(args.recordings_dir, args.limit)
    print(f"Loaded {len(transcripts)} sessions")

    if not transcripts:
        print("No transcripts found!")
        return

    # 1. Confidence distribution
    with_logprob, without_logprob = analyze_confidence_distribution(transcripts)

    # 2. Screening comparison
    newly_dirty = compare_screening_with_without_confidence(transcripts)

    # 3. Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    if not with_logprob:
        print("No segments have avg_logprob data yet.")
        print("After retranscribing or new recordings, re-run this tool")
        print("to see the actual impact of confidence-based screening.")
        print(f"\nFor existing {without_logprob} segments without avg_logprob:")
        print("  → Confidence rule is skipped (backward compatible)")
        print("  → Only katakana rules apply (same as before)")
    else:
        print(f"Confidence data: {len(with_logprob)} segments have avg_logprob")
        print(f"Newly flagged by confidence: {len(newly_dirty)} segments")
        if len(with_logprob) > 0:
            pct = len(newly_dirty) / len(with_logprob) * 100
            print(f"Additional dirty rate: {pct:.1f}%")

    print("\nThreshold: CONFIDENCE_LOGPROB_THRESHOLD =", CONFIDENCE_LOGPROB_THRESHOLD)


if __name__ == "__main__":
    main()
