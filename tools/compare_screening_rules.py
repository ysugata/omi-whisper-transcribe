#!/usr/bin/env python3
"""Compare old vs new screening rules on real transcript data.

Shows exactly which segments changed classification and why.

Old rules (before adjustments):
  1. llm_corrected → clean
  2. short text → clean
  3. Pure Japanese (ratio check) → clean   ← was checked BEFORE katakana
  4. Unknown katakana → dirty
  5. Default → clean
  No common loanwords, no confidence check.

New rules (after adjustments):
  1. llm_corrected → clean
  2. short text → clean
  3. Unknown katakana → dirty              ← now checked FIRST
  4. Low confidence (avg_logprob < -0.5) → dirty  ← NEW
  5. Pure Japanese (ratio check) → clean
  6. Default → clean
  ~300 common loanwords in known_values.

Usage:
    python3 tools/compare_screening_rules.py [--recordings-dir DIR] [--limit N]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_screener import _COMMON_LOANWORDS

# --- Regex patterns (same as segment_classifier.py) ---
_KATAKANA_RE = re.compile(r'[\u30A0-\u30FF]')
_KANJI_HIRA_RE = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F]')
_KATAKANA_WORD_RE = re.compile(r'[\u30A0-\u30FF]{3,}')

MIN_TEXT_LENGTH = 20


def classify_old(segment: dict, known_values: set) -> tuple:
    """Old classification rules (before all adjustments).

    Returns (result, reason) where result is 'dirty' or 'clean'.
    """
    if segment.get("llm_corrected"):
        return "clean", "R1: already corrected"

    text = segment.get("text", "")
    if len(text) < MIN_TEXT_LENGTH:
        return "clean", "R2: short text"

    # OLD ORDER: ratio check BEFORE katakana check
    # And NO common loanwords in known_values
    katakana_chars = len(_KATAKANA_RE.findall(text))
    kanji_hira_chars = len(_KANJI_HIRA_RE.findall(text))
    total_chars = len(text)

    if total_chars > 0:
        katakana_ratio = katakana_chars / total_chars
        kanji_hira_ratio = kanji_hira_chars / total_chars
        if katakana_ratio < 0.2 and kanji_hira_ratio > 0.7:
            return "clean", "R3-old: pure Japanese (ratio)"

    katakana_words = _KATAKANA_WORD_RE.findall(text)
    # Old: no common loanwords, only term_corrector keys
    unknown_katakana = [kw for kw in katakana_words if kw not in known_values]
    if unknown_katakana:
        return "dirty", f"R4-old: unknown katakana [{', '.join(unknown_katakana[:3])}]"

    return "clean", "R5-old: default"


def classify_new(segment: dict, known_values_new: set) -> tuple:
    """New classification rules (after all adjustments).

    Returns (result, reason) where result is 'dirty' or 'clean'.
    """
    if segment.get("llm_corrected"):
        return "clean", "R1: already corrected"

    text = segment.get("text", "")
    if len(text) < MIN_TEXT_LENGTH:
        return "clean", "R2: short text"

    # NEW ORDER: katakana check FIRST, with common loanwords
    katakana_words = _KATAKANA_WORD_RE.findall(text)
    unknown_katakana = [kw for kw in katakana_words if kw not in known_values_new]
    if unknown_katakana:
        return "dirty", f"R3-new: unknown katakana [{', '.join(unknown_katakana[:3])}]"

    # NEW: confidence check
    avg_logprob = segment.get("avg_logprob")
    if avg_logprob is not None and avg_logprob < -0.5:
        return "dirty", f"R4-new: low confidence (logprob={avg_logprob:.3f})"

    katakana_chars = len(_KATAKANA_RE.findall(text))
    kanji_hira_chars = len(_KANJI_HIRA_RE.findall(text))
    total_chars = len(text)

    if total_chars > 0:
        katakana_ratio = katakana_chars / total_chars
        kanji_hira_ratio = kanji_hira_chars / total_chars
        if katakana_ratio < 0.2 and kanji_hira_ratio > 0.7:
            return "clean", "R5-new: pure Japanese (ratio)"

    return "clean", "R6-new: default"


def load_transcripts(recordings_dir: str, limit: int = 0):
    """Load transcript.json files."""
    transcripts = []
    for transcript_file in sorted(Path(recordings_dir).rglob("transcript.json"), reverse=True):
        try:
            with open(transcript_file) as f:
                data = json.load(f)
            segments = data if isinstance(data, list) else data.get("segments", [])
            if segments:
                session_id = "/".join(transcript_file.parts[-3:-1])
                transcripts.append({"session_id": session_id, "segments": segments})
        except (json.JSONDecodeError, IOError):
            continue
        if limit and len(transcripts) >= limit:
            break
    return transcripts


def load_known_values_old():
    """Load known values as they were before (no common loanwords)."""
    known = set()
    # Only term_corrector keys
    tc_path = Path("config/term_corrections.json")
    if tc_path.exists():
        with open(tc_path) as f:
            corrections = json.load(f)
        for wrong in corrections:
            known.add(wrong)
    return known


def load_known_values_new():
    """Load known values as they are now (with common loanwords + term_corrector)."""
    known = set(_COMMON_LOANWORDS)
    tc_path = Path("config/term_corrections.json")
    if tc_path.exists():
        with open(tc_path) as f:
            corrections = json.load(f)
        for wrong in corrections:
            known.add(wrong)
    return known


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new screening rules")
    parser.add_argument("--recordings-dir", default="./recordings")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    print("Loading transcripts...")
    transcripts = load_transcripts(args.recordings_dir, args.limit)
    total_segs = sum(len(t["segments"]) for t in transcripts)
    print(f"Loaded {len(transcripts)} sessions, {total_segs} segments\n")

    known_old = load_known_values_old()
    known_new = load_known_values_new()

    print(f"Known values: old={len(known_old)}, new={len(known_new)} "
          f"(+{len(known_new) - len(known_old)} common loanwords)\n")

    # Compare
    changes = []  # segments that changed classification
    stats = {
        "old_dirty": 0, "old_clean": 0,
        "new_dirty": 0, "new_clean": 0,
        "clean_to_dirty": 0, "dirty_to_clean": 0,
        "same": 0,
    }

    for t in transcripts:
        for seg in t["segments"]:
            old_result, old_reason = classify_old(seg, known_old)
            new_result, new_reason = classify_new(seg, known_new)

            if old_result == "dirty":
                stats["old_dirty"] += 1
            else:
                stats["old_clean"] += 1
            if new_result == "dirty":
                stats["new_dirty"] += 1
            else:
                stats["new_clean"] += 1

            if old_result != new_result:
                direction = "clean→dirty" if new_result == "dirty" else "dirty→clean"
                if new_result == "dirty":
                    stats["clean_to_dirty"] += 1
                else:
                    stats["dirty_to_clean"] += 1
                changes.append({
                    "session": t["session_id"],
                    "text": seg.get("text", "")[:100],
                    "direction": direction,
                    "old_reason": old_reason,
                    "new_reason": new_reason,
                    "avg_logprob": seg.get("avg_logprob"),
                })
            else:
                stats["same"] += 1

    # --- Output ---
    print("=" * 70)
    print("SCREENING RULE COMPARISON: OLD vs NEW")
    print("=" * 70)

    print(f"\n{'':20s}  {'OLD':>10s}  {'NEW':>10s}  {'DIFF':>10s}")
    print(f"{'─'*20}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"{'dirty':20s}  {stats['old_dirty']:10d}  {stats['new_dirty']:10d}  "
          f"{stats['new_dirty'] - stats['old_dirty']:+10d}")
    print(f"{'clean':20s}  {stats['old_clean']:10d}  {stats['new_clean']:10d}  "
          f"{stats['new_clean'] - stats['old_clean']:+10d}")
    pct_old = stats['old_dirty'] / total_segs * 100 if total_segs else 0
    pct_new = stats['new_dirty'] / total_segs * 100 if total_segs else 0
    print(f"{'dirty rate':20s}  {pct_old:9.1f}%  {pct_new:9.1f}%  "
          f"{pct_new - pct_old:+9.1f}%")

    print(f"\n{'Unchanged':20s}  {stats['same']:10d}")
    print(f"{'clean → dirty':20s}  {stats['clean_to_dirty']:10d}")
    print(f"{'dirty → clean':20s}  {stats['dirty_to_clean']:10d}")

    if changes:
        # Group by direction
        c2d = [c for c in changes if c["direction"] == "clean→dirty"]
        d2c = [c for c in changes if c["direction"] == "dirty→clean"]

        if c2d:
            print(f"\n{'='*70}")
            print(f"NEWLY DIRTY (clean→dirty): {len(c2d)} segments")
            print(f"These segments were previously skipped but now get sent to LLM")
            print(f"{'='*70}")
            for i, c in enumerate(c2d):
                print(f"\n  [{i+1}] {c['session']}")
                print(f"      Text: {c['text']}")
                print(f"      OLD:  {c['old_reason']}")
                print(f"      NEW:  {c['new_reason']}")
                if c["avg_logprob"] is not None:
                    print(f"      Logprob: {c['avg_logprob']:.3f}")

        if d2c:
            print(f"\n{'='*70}")
            print(f"NEWLY CLEAN (dirty→clean): {len(d2c)} segments")
            print(f"These segments were previously sent to LLM but now get skipped")
            print(f"{'='*70}")
            for i, c in enumerate(d2c):
                print(f"\n  [{i+1}] {c['session']}")
                print(f"      Text: {c['text']}")
                print(f"      OLD:  {c['old_reason']}")
                print(f"      NEW:  {c['new_reason']}")

    else:
        print(f"\nNo changes detected.")

    # Summary
    print(f"\n{'='*70}")
    print(f"CHANGES SUMMARY")
    print(f"{'='*70}")
    print(f"\nRule changes applied:")
    print(f"  1. Rule order swap: katakana check (R3) now runs BEFORE ratio check (R5)")
    print(f"     → Catches unknown katakana in mostly-Japanese text")
    print(f"  2. ~{len(_COMMON_LOANWORDS)} common loanwords added to known_values")
    print(f"     → Prevents common words (レストラン, テスト) from triggering dirty")
    print(f"  3. Confidence rule (R4): avg_logprob < -0.5 → dirty")
    print(f"     → Catches non-katakana ASR errors when Whisper is uncertain")

    if stats["clean_to_dirty"] > 0 or stats["dirty_to_clean"] > 0:
        print(f"\nNet effect:")
        net = stats["new_dirty"] - stats["old_dirty"]
        if net > 0:
            print(f"  +{net} more segments sent to LLM (better coverage)")
        elif net < 0:
            print(f"  {net} fewer segments sent to LLM (less noise)")
        else:
            print(f"  Same number of dirty segments (classification shifted, not increased)")
        if stats["dirty_to_clean"] > 0:
            print(f"  {stats['dirty_to_clean']} segments saved from unnecessary LLM calls (common loanwords)")
        if stats["clean_to_dirty"] > 0:
            print(f"  {stats['clean_to_dirty']} segments now correctly caught (rule order fix)")


if __name__ == "__main__":
    main()
