#!/usr/bin/env python3
"""LLM correction quality comparison: Legacy single-call vs 3-stage pipeline.

Runs the same segments through both methods and compares results.
Also measures token usage and estimates cost.

Usage:
    python tools/compare_llm_pipeline.py [--sessions N] [--ids SESSION_ID...]
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

from llm_corrector import LLMCorrector
from llm_pipeline import LLMPipeline
from term_corrector import TermCorrector
from hotword_loader import HotwordLoader

RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def get_known_terms() -> list[str]:
    """Build known terms list (same as bridge)."""
    terms = set()
    tc = TermCorrector(str(CONFIG_DIR / "term_corrections.json"))
    for v in tc.list_corrections().values():
        terms.add(v)
    hl = HotwordLoader(str(CONFIG_DIR / "hotwords.json"))
    for w in hl.get_custom_words():
        terms.add(w)
    return sorted(terms)


def build_session_glossary(segments, known_terms, max_glossary=50):
    """Build session glossary (same as bridge)."""
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
                segs = data.get("segments", [])
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
    return data.get("segments", [])


# ---------------------------------------------------------------------------
# Diff analysis
# ---------------------------------------------------------------------------

def compute_diff(original_segs, corrected_segs):
    """Compute text-level diffs between original and corrected segments."""
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


def estimate_tokens(text):
    """Rough token estimation for Japanese text (chars / 1.5)."""
    return max(1, int(len(text) / 1.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM correction comparison")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--max-segs", type=int, default=20,
                        help="Max segments per batch (default: 20)")
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
    print("LLM Correction Comparison: Legacy (single-call) vs Pipeline (3-stage)")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Min text length: {min_text_length}")
    print(f"  Max segments/batch: {args.max_segs}")
    print()

    # Initialize both methods
    legacy = LLMCorrector(
        api_key=gemini_key, base_url=base_url, model=model, timeout=30,
    )
    pipeline = LLMPipeline(
        api_key=gemini_key, base_url=base_url, model=model, timeout=30,
        stages="ABC",
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
        "legacy": {"changes": 0, "time": 0, "api_calls": 0},
        "pipeline": {"changes": 0, "time": 0, "api_calls": 0, "retries": 0},
        "segments_processed": 0,
        "sessions_processed": 0,
    }

    for idx, sess_id in enumerate(session_ids):
        print(f"{'=' * 70}")
        print(f"[{idx + 1}/{len(session_ids)}] Session: {sess_id}")
        all_segs = load_segments(sess_id)

        # Take first N segments (simulate a real batch)
        segs = all_segs[:args.max_segs]
        eligible = [s for s in segs if len(s.get("text", "")) >= min_text_length]
        print(f"  Total segs: {len(all_segs)}, batch: {len(segs)}, eligible: {len(eligible)}")

        if len(eligible) < 3:
            print("  SKIP (too few eligible segments)")
            print()
            continue

        # Show first 3 segment texts
        for i, s in enumerate(eligible[:3]):
            print(f"  [{i+1}] {s['text'][:70]}...")
        print()

        glossary = build_session_glossary(segs, known_terms)

        # --- Legacy (single-call) ---
        print("  [Legacy] Running...", end=" ", flush=True)
        t0 = time.time()
        legacy_result = legacy.correct_segments(
            segs, known_terms=known_terms, min_text_length=min_text_length,
        )
        legacy_time = time.time() - t0
        legacy_changes = compute_diff(segs, legacy_result)
        print(f"{legacy_time:.1f}s, {len(legacy_changes)} changes")
        totals["legacy"]["changes"] += len(legacy_changes)
        totals["legacy"]["time"] += legacy_time
        totals["legacy"]["api_calls"] += 1

        # --- Pipeline (3-stage) ---
        print("  [Pipeline ABC] Running...", end=" ", flush=True)
        t0 = time.time()
        pipeline_result = pipeline.process_segments(
            segs, known_terms=known_terms,
            session_glossary=glossary,
            min_text_length=min_text_length,
        )
        pipeline_time = time.time() - t0
        pipeline_changes = compute_diff(segs, pipeline_result.corrected_segments)
        retry_str = " (retried!)" if pipeline_result.retried else ""
        print(f"{pipeline_time:.1f}s, {len(pipeline_changes)} changes, "
              f"api_calls={pipeline_result.api_calls}{retry_str}")
        totals["pipeline"]["changes"] += len(pipeline_changes)
        totals["pipeline"]["time"] += pipeline_time
        totals["pipeline"]["api_calls"] += pipeline_result.api_calls
        if pipeline_result.retried:
            totals["pipeline"]["retries"] += 1

        # --- 3-A candidates ---
        if pipeline_result.candidates and pipeline_result.candidates.candidates:
            cands = pipeline_result.candidates.candidates
            print(f"\n  3-A candidates ({len(cands)}):")
            for c in cands[:8]:
                sf = f' → "{c.suggested_form}"' if c.suggested_form else ""
                print(f"    [{c.line}] 「{c.text_span}」{sf} ({c.category}, {c.confidence})")
            if len(cands) > 8:
                print(f"    ... +{len(cands) - 8} more")

        # --- 3-C audit ---
        if pipeline_result.audit and pipeline_result.audit.items:
            items = pipeline_result.audit.items
            print(f"\n  3-C audit ({len(items)} issues, pass={pipeline_result.audit.passed}):")
            for it in items[:5]:
                print(f"    [{it.line}] {it.severity}: 「{it.text_span}」"
                      f" ({it.issue_type}) → {it.suggestion}")

        # --- Side-by-side diff ---
        # Merge all changed lines
        all_changed_lines = set()
        legacy_map = {c["seg"]: c for c in legacy_changes}
        pipeline_map = {c["seg"]: c for c in pipeline_changes}
        all_changed_lines = set(legacy_map.keys()) | set(pipeline_map.keys())

        if all_changed_lines:
            print(f"\n  {'Seg':>4} | {'Original':^35} | {'Legacy':^35} | {'Pipeline':^35}")
            print(f"  {'─' * 4}─┼─{'─' * 35}─┼─{'─' * 35}─┼─{'─' * 35}")
            for seg_idx in sorted(all_changed_lines):
                orig_text = segs[seg_idx]["text"][:35]
                leg_text = legacy_map[seg_idx]["corrected"][:35] if seg_idx in legacy_map else "—"
                pip_text = pipeline_map[seg_idx]["corrected"][:35] if seg_idx in pipeline_map else "—"
                print(f"  {seg_idx:>4} | {orig_text:<35} | {leg_text:<35} | {pip_text:<35}")
        else:
            print("\n  (No changes from either method)")

        print()

        totals["segments_processed"] += len(eligible)
        totals["sessions_processed"] += 1

        all_results.append({
            "session_id": sess_id,
            "total_segs": len(all_segs),
            "batch_segs": len(segs),
            "eligible": len(eligible),
            "legacy": {
                "time": legacy_time,
                "changes": len(legacy_changes),
                "details": legacy_changes,
            },
            "pipeline": {
                "time": pipeline_time,
                "changes": len(pipeline_changes),
                "api_calls": pipeline_result.api_calls,
                "retried": pipeline_result.retried,
                "stages": pipeline_result.stages_run,
                "candidates": len(pipeline_result.candidates.candidates) if pipeline_result.candidates else 0,
                "audit_items": len(pipeline_result.audit.items) if pipeline_result.audit else 0,
                "audit_pass": pipeline_result.audit.passed if pipeline_result.audit else None,
                "details": pipeline_changes,
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

    n = totals["sessions_processed"]
    segs = totals["segments_processed"]
    print(f"Sessions: {n}, Segments processed: {segs}")
    print()

    header = f"{'Metric':<30} {'Legacy (1-call)':>20} {'Pipeline (ABC)':>20}"
    print(header)
    print("─" * len(header))
    print(f"{'Total changes':<30} {totals['legacy']['changes']:>20} {totals['pipeline']['changes']:>20}")
    print(f"{'Total API calls':<30} {totals['legacy']['api_calls']:>20} {totals['pipeline']['api_calls']:>20}")
    print(f"{'Total time (s)':<30} {totals['legacy']['time']:>19.1f}s {totals['pipeline']['time']:>19.1f}s")
    if n > 0:
        print(f"{'Avg time/batch':<30} {totals['legacy']['time']/n:>19.1f}s {totals['pipeline']['time']/n:>19.1f}s")
    print(f"{'Retries (3-C → 3-B)':<30} {'—':>20} {totals['pipeline']['retries']:>20}")
    print()

    # Cost
    print("─" * 70)
    print("COST ESTIMATE (Gemini 2.5 Flash-Lite)")
    print("─" * 70)
    print()
    print("  Standard pricing: input $0.10/1M tokens, output $0.40/1M tokens")
    print()

    # Rough estimate: ~2500 input + ~500 output per stage call
    legacy_calls = totals["legacy"]["api_calls"]
    pipeline_calls = totals["pipeline"]["api_calls"]

    # Estimate tokens per call
    in_per_call = 2500
    out_per_call = 500
    cost_per_call = (in_per_call * 0.10 + out_per_call * 0.40) / 1_000_000

    legacy_cost = legacy_calls * cost_per_call
    pipeline_cost = pipeline_calls * cost_per_call

    print(f"  {'':30} {'Legacy':>20} {'Pipeline':>20}")
    print(f"  {'API calls':30} {legacy_calls:>20} {pipeline_calls:>20}")
    print(f"  {'Est. cost (this run)':30} {'${:.6f}'.format(legacy_cost):>20} {'${:.6f}'.format(pipeline_cost):>20}")

    if n > 0:
        # Monthly estimate: 8h/day, 60s flush interval = 480 batches/day
        batches_per_day = 480
        legacy_monthly = batches_per_day * 30 * cost_per_call
        pipeline_monthly = batches_per_day * 30 * cost_per_call * (pipeline_calls / max(legacy_calls, 1))
        print()
        print(f"  Monthly estimate (8h/day, 480 batches/day):")
        print(f"  {'Legacy (1 call/batch)':30} {'${:.2f}'.format(legacy_monthly):>20}")
        print(f"  {'Pipeline (ABC, ~3 calls/batch)':30} {'${:.2f}'.format(pipeline_monthly):>20}")
        diff = pipeline_monthly - legacy_monthly
        print(f"  {'Difference':30} {'+${:.2f}'.format(diff):>20}")

    # Pipeline-only changes analysis
    print()
    print("─" * 70)
    print("PIPELINE-ONLY CORRECTIONS (found by pipeline but not legacy)")
    print("─" * 70)
    print()

    pipeline_only_count = 0
    for r in all_results:
        leg_set = {c["seg"] for c in r["legacy"]["details"]}
        for c in r["pipeline"]["details"]:
            if c["seg"] not in leg_set:
                pipeline_only_count += 1
                print(f"  [{r['session_id']}] seg {c['seg']}:")
                print(f"    orig: {c['original'][:80]}")
                print(f"    fix:  {c['corrected'][:80]}")
                print()
    if pipeline_only_count == 0:
        print("  (none)")

    print()
    print("─" * 70)
    print("LEGACY-ONLY CORRECTIONS (found by legacy but not pipeline)")
    print("─" * 70)
    print()

    legacy_only_count = 0
    for r in all_results:
        pip_set = {c["seg"] for c in r["pipeline"]["details"]}
        for c in r["legacy"]["details"]:
            if c["seg"] not in pip_set:
                legacy_only_count += 1
                print(f"  [{r['session_id']}] seg {c['seg']}:")
                print(f"    orig: {c['original'][:80]}")
                print(f"    fix:  {c['corrected'][:80]}")
                print()
    if legacy_only_count == 0:
        print("  (none)")

    print(f"\nPipeline-only: {pipeline_only_count}, Legacy-only: {legacy_only_count}")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"compare_pipeline_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nRaw results saved: {out_path}")


if __name__ == "__main__":
    main()
