#!/usr/bin/env python3
"""Compare actual LLM correction quality: with vs without screening.

Runs the same sessions through two modes:
  Mode A (no screening): ALL segments → LLM Pipeline Stage B
  Mode B (with screening): Screener (dict+learned+classify) → dirty only → LLM Stage B

Then compares the final corrected text to show:
  - Corrections that both modes make (consistent quality)
  - Corrections only Mode A makes (potential quality loss from screening)
  - Dictionary corrections Mode B applies pre-LLM
  - Net quality assessment

Usage:
    python tools/compare_screening_quality.py [--sessions N] [--ids SESSION_ID...]
    python tools/compare_screening_quality.py --sessions 3
    python tools/compare_screening_quality.py --ids 2026-03-27/23-47-21
"""

import argparse
import json
import os
import sys
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

from llm_pipeline import LLMPipeline
from term_corrector import TermCorrector
from hotword_loader import HotwordLoader
from segment_screener import SegmentScreener
from suggestion_learner import SuggestionLearner

RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CONFIG_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Setup helpers (same as compare_llm_pipeline.py)
# ---------------------------------------------------------------------------

def get_known_terms() -> list[str]:
    terms = set()
    tc = TermCorrector(str(CONFIG_DIR / "term_corrections.json"))
    for v in tc.list_corrections().values():
        terms.add(v)
    hl = HotwordLoader(str(CONFIG_DIR / "hotwords.json"))
    for w in hl.get_custom_words():
        terms.add(w)
    return sorted(terms)


def build_session_glossary(segments, known_terms, max_glossary=50):
    import re
    combined_text = " ".join(seg.get("text", "") for seg in segments)
    if not combined_text.strip():
        return []
    katakana_words = set(re.findall(r'[\u30A0-\u30FF]{3,}', combined_text))
    glossary = []
    seen = set()
    if known_terms:
        for term in known_terms:
            if term in combined_text and term not in seen:
                glossary.append(term)
                seen.add(term)
    for kw in sorted(katakana_words):
        if kw not in seen:
            glossary.append(kw)
            seen.add(kw)
    if known_terms:
        for term in known_terms:
            if term not in seen:
                glossary.append(term)
                seen.add(term)
            if len(glossary) >= max_glossary:
                break
    return glossary[:max_glossary]


# ---------------------------------------------------------------------------
# Session finder
# ---------------------------------------------------------------------------

def find_sessions(limit=5, specific=None):
    if specific:
        return [s for s in specific if (RECORDINGS_DIR / s / "transcript.json").exists()]
    candidates = []
    for date_dir in sorted(RECORDINGS_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for sess_dir in sorted(date_dir.iterdir()):
            if not sess_dir.is_dir():
                continue
            transcript = sess_dir / "transcript.json"
            if not transcript.exists():
                continue
            try:
                with open(transcript) as f:
                    data = json.load(f)
                segs = data if isinstance(data, list) else data.get("segments", [])
                meaningful = [s for s in segs if len(s.get("text", "").strip()) > 10]
                if len(meaningful) >= 8:
                    candidates.append((len(meaningful), f"{date_dir.name}/{sess_dir.name}"))
            except Exception:
                continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:limit]]


def load_segments(session_id):
    path = RECORDINGS_DIR / session_id / "transcript.json"
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("segments", [])


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def compute_diff(original_segs, corrected_segs):
    changes = []
    for i, (orig, corr) in enumerate(zip(original_segs, corrected_segs)):
        orig_text = orig.get("text", "")
        corr_text = corr.get("text", "") if isinstance(corr, dict) else corr
        if orig_text != corr_text:
            changes.append({
                "seg": i,
                "original": orig_text,
                "corrected": corr_text,
            })
    return changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM correction quality: with vs without screening"
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--max-segs", type=int, default=30,
                        help="Max segments per session (default: 30)")
    parser.add_argument("--ids", nargs="*", help="Specific session IDs")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
    min_text_length = 20

    print("=" * 70)
    print("SCREENING QUALITY COMPARISON")
    print("Mode A: ALL segments → LLM (no screening)")
    print("Mode B: Screener → dirty only → LLM (with screening)")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Min text length: {min_text_length}")
    print(f"  Max segments/session: {args.max_segs}")
    print()

    # Initialize pipeline (Stage B only, which is the default with screening)
    pipeline = LLMPipeline(
        api_key=gemini_key, base_url=base_url, model=model, timeout=30,
        stages="B",
    )

    # Initialize screener
    tc = TermCorrector(str(CONFIG_DIR / "term_corrections.json"))
    hl = HotwordLoader(str(CONFIG_DIR / "hotwords.json"))
    sl = SuggestionLearner(str(CONFIG_DIR / "pending_suggestions.json"))
    screener = SegmentScreener(
        term_corrector=tc,
        suggestion_learner=sl,
        hotword_loader=hl,
        min_text_length=min_text_length,
    )

    known_terms = get_known_terms()
    print(f"  Known terms: {len(known_terms)}")

    session_ids = find_sessions(limit=args.sessions, specific=args.ids)
    if not session_ids:
        print("ERROR: No sessions found")
        sys.exit(1)
    print(f"  Sessions: {len(session_ids)}")
    print()

    all_results = []
    totals = {
        "mode_a": {"changes": 0, "api_calls": 0, "time": 0},
        "mode_b": {"changes": 0, "api_calls": 0, "time": 0,
                    "dict_corrections": 0, "dirty": 0, "clean": 0},
        "both": 0,       # corrections found by both
        "a_only": 0,     # corrections found only without screening (missed)
        "b_only": 0,     # corrections found only with screening
        "segments": 0,
        "sessions": 0,
    }

    for idx, sess_id in enumerate(session_ids):
        print(f"{'=' * 70}")
        print(f"[{idx + 1}/{len(session_ids)}] Session: {sess_id}")
        all_segs = load_segments(sess_id)

        # Limit segments
        segs = all_segs[:args.max_segs]
        eligible = [s for s in segs if len(s.get("text", "")) >= min_text_length]
        print(f"  Total: {len(all_segs)}, batch: {len(segs)}, eligible: {len(eligible)}")

        if len(eligible) < 3:
            print("  SKIP (too few eligible segments)")
            print()
            continue

        glossary = build_session_glossary(segs, known_terms)

        # ---------------------------------------------------------------
        # Mode A: No screening, ALL segments → LLM
        # ---------------------------------------------------------------
        print(f"\n  [Mode A] All {len(eligible)} eligible → LLM...", end=" ", flush=True)
        t0 = time.time()
        result_a = pipeline.process_segments(
            segs,
            known_terms=known_terms,
            session_glossary=glossary,
            min_text_length=min_text_length,
            segment_indices=None,  # ALL segments
        )
        time_a = time.time() - t0
        changes_a = compute_diff(segs, result_a.corrected_segments)
        print(f"{time_a:.1f}s, {len(changes_a)} changes, {result_a.api_calls} API calls")
        totals["mode_a"]["changes"] += len(changes_a)
        totals["mode_a"]["api_calls"] += result_a.api_calls
        totals["mode_a"]["time"] += time_a

        # ---------------------------------------------------------------
        # Mode B: With screening, dirty only → LLM
        # ---------------------------------------------------------------
        print("  [Mode B] Screening...", end=" ", flush=True)
        t0 = time.time()
        screening_result = screener.screen(segs)
        dirty = screening_result.dirty_indices
        clean = screening_result.clean_indices
        screened_segs = screening_result.segments  # dict corrections already applied
        dict_corr = (screening_result.metrics.dict_corrections
                     + screening_result.metrics.learned_corrections)

        print(f"dirty={len(dirty)}, clean={len(clean)}, dict={dict_corr}")

        if dirty:
            print(f"  [Mode B] {len(dirty)} dirty → LLM...", end=" ", flush=True)
            result_b = pipeline.process_segments(
                screened_segs,
                known_terms=known_terms,
                session_glossary=glossary,
                min_text_length=min_text_length,
                segment_indices=dirty,
            )
            time_b = time.time() - t0
            changes_b = compute_diff(segs, result_b.corrected_segments)
            print(f"{time_b:.1f}s, {len(changes_b)} changes, {result_b.api_calls} API calls")
            api_calls_b = result_b.api_calls
        else:
            time_b = time.time() - t0
            # No LLM needed — only dict corrections
            changes_b = compute_diff(segs, screened_segs)
            print(f"  [Mode B] All clean, LLM skipped. {time_b:.1f}s, {len(changes_b)} dict-only changes")
            api_calls_b = 0

        totals["mode_b"]["changes"] += len(changes_b)
        totals["mode_b"]["api_calls"] += api_calls_b
        totals["mode_b"]["time"] += time_b
        totals["mode_b"]["dict_corrections"] += dict_corr
        totals["mode_b"]["dirty"] += len(dirty)
        totals["mode_b"]["clean"] += len(clean)

        # ---------------------------------------------------------------
        # Compare corrections
        # ---------------------------------------------------------------
        map_a = {c["seg"]: c for c in changes_a}
        map_b = {c["seg"]: c for c in changes_b}
        all_changed = set(map_a.keys()) | set(map_b.keys())

        both_segs = []
        a_only_segs = []
        b_only_segs = []

        for seg_idx in sorted(all_changed):
            in_a = seg_idx in map_a
            in_b = seg_idx in map_b
            if in_a and in_b:
                both_segs.append(seg_idx)
            elif in_a and not in_b:
                a_only_segs.append(seg_idx)
            else:
                b_only_segs.append(seg_idx)

        totals["both"] += len(both_segs)
        totals["a_only"] += len(a_only_segs)
        totals["b_only"] += len(b_only_segs)
        totals["segments"] += len(segs)
        totals["sessions"] += 1

        # --- Print results ---
        if both_segs:
            print(f"\n  BOTH ({len(both_segs)} segments corrected by both modes):")
            for si in both_segs:
                orig = segs[si]["text"][:60]
                fix_a = map_a[si]["corrected"][:60]
                fix_b = map_b[si]["corrected"][:60]
                same = "✓ same" if map_a[si]["corrected"] == map_b[si]["corrected"] else "≠ DIFFERENT"
                print(f"    [{si:>3}] {same}")
                print(f"          orig: {orig}")
                if map_a[si]["corrected"] == map_b[si]["corrected"]:
                    print(f"          fix:  {fix_a}")
                else:
                    print(f"          A:    {fix_a}")
                    print(f"          B:    {fix_b}")

        if a_only_segs:
            print(f"\n  A-ONLY ({len(a_only_segs)} corrections MISSED by screening):")
            for si in a_only_segs:
                orig = segs[si]["text"][:70]
                fix = map_a[si]["corrected"][:70]
                # Check if this seg was classified as clean
                is_clean = si not in dirty
                tag = " [was clean]" if is_clean else " [was dirty]"
                print(f"    [{si:>3}]{tag}")
                print(f"          orig: {orig}")
                print(f"          fix:  {fix}")

        if b_only_segs:
            print(f"\n  B-ONLY ({len(b_only_segs)} corrections only with screening):")
            for si in b_only_segs:
                orig = segs[si]["text"][:70]
                fix = map_b[si]["corrected"][:70]
                is_dict = si not in dirty
                tag = " [dict only]" if is_dict else " [LLM]"
                print(f"    [{si:>3}]{tag}")
                print(f"          orig: {orig}")
                print(f"          fix:  {fix}")

        if not all_changed:
            print("\n  (No corrections from either mode)")

        print()

        all_results.append({
            "session_id": sess_id,
            "total_segs": len(all_segs),
            "batch_segs": len(segs),
            "eligible": len(eligible),
            "mode_a": {
                "changes": len(changes_a),
                "api_calls": result_a.api_calls,
                "time": time_a,
                "details": changes_a,
            },
            "mode_b": {
                "changes": len(changes_b),
                "api_calls": api_calls_b,
                "time": time_b,
                "dirty": len(dirty),
                "clean": len(clean),
                "dict_corrections": dict_corr,
                "details": changes_b,
            },
            "comparison": {
                "both": len(both_segs),
                "a_only": len(a_only_segs),
                "b_only": len(b_only_segs),
                "a_only_details": [
                    {"seg": si, "original": segs[si]["text"], "missed_fix": map_a[si]["corrected"]}
                    for si in a_only_segs
                ],
                "b_only_details": [
                    {"seg": si, "original": segs[si]["text"], "screening_fix": map_b[si]["corrected"]}
                    for si in b_only_segs
                ],
            },
        })

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    n = totals["sessions"]
    if n == 0:
        print("No sessions processed.")
        return

    print(f"Sessions: {n}, Segments: {totals['segments']}")
    print()

    header = f"{'Metric':<35} {'Mode A (no screen)':>20} {'Mode B (screened)':>20}"
    print(header)
    print("-" * len(header))
    print(f"{'Total corrections':<35} {totals['mode_a']['changes']:>20} {totals['mode_b']['changes']:>20}")
    print(f"{'API calls':<35} {totals['mode_a']['api_calls']:>20} {totals['mode_b']['api_calls']:>20}")
    print(f"{'Total time (s)':<35} {totals['mode_a']['time']:>19.1f}s {totals['mode_b']['time']:>19.1f}s")
    print(f"{'Dict corrections (pre-LLM)':<35} {'—':>20} {totals['mode_b']['dict_corrections']:>20}")
    print(f"{'Segments sent to LLM':<35} {'all':>20} {totals['mode_b']['dirty']:>20}")
    print(f"{'Segments skipped (clean)':<35} {'0':>20} {totals['mode_b']['clean']:>20}")
    print()

    print(f"{'='*70}")
    print(f"QUALITY ANALYSIS")
    print(f"{'='*70}")
    print()
    print(f"  Corrections found by both:       {totals['both']:>5}")
    print(f"  A-only (MISSED by screening):    {totals['a_only']:>5}")
    print(f"  B-only (screening-exclusive):    {totals['b_only']:>5}")
    print()

    total_corrections = totals['both'] + totals['a_only'] + totals['b_only']
    if total_corrections > 0:
        coverage = (totals['both'] + totals['b_only']) / (totals['both'] + totals['a_only']) * 100 if (totals['both'] + totals['a_only']) > 0 else 100
        print(f"  Screening coverage: {coverage:.1f}%")
        print(f"  (= corrections captured by Mode B / corrections in Mode A)")
        if totals['a_only'] > 0:
            print(f"\n  WARNING: {totals['a_only']} correction(s) missed by screening.")
            print(f"  Review the A-ONLY details above to assess significance.")
        else:
            print(f"\n  PERFECT: No corrections missed by screening!")
            print(f"  Screening safely reduces API calls with no quality loss.")

    if totals['mode_a']['api_calls'] > 0:
        saving = (1 - totals['mode_b']['api_calls'] / totals['mode_a']['api_calls']) * 100
        print(f"\n  API call reduction: {saving:.0f}%")

    # Detailed missed corrections
    missed_all = []
    for r in all_results:
        for m in r["comparison"]["a_only_details"]:
            missed_all.append({
                "session": r["session_id"],
                **m,
            })

    if missed_all:
        print(f"\n{'='*70}")
        print(f"DETAILED: ALL MISSED CORRECTIONS")
        print(f"{'='*70}")
        for i, m in enumerate(missed_all):
            print(f"\n  [{i+1}] {m['session']} seg {m['seg']}")
            print(f"      orig: {m['original'][:80]}")
            print(f"      fix:  {m['missed_fix'][:80]}")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = CONFIG_DIR / f"compare_screening_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRaw results saved: {out_path}")


if __name__ == "__main__":
    main()
