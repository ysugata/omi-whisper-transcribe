#!/usr/bin/env python3
"""Verify screening accuracy against real transcripts.

Checks whether the screener's dirty/clean classification would miss
segments that actually need LLM correction by:
1. Running the screener on real transcripts
2. Analyzing what gets classified as clean vs dirty
3. Flagging potential false negatives (clean segments that look like they need fixing)
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from term_corrector import TermCorrector
from suggestion_learner import SuggestionLearner
from segment_screener import SegmentScreener
from segment_classifier import classify_segment

CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")

# Known patterns that LLM should catch but screener might miss
_SUSPICIOUS_PATTERNS = [
    # Broken words (ASR truncation)
    re.compile(r'[\u30A0-\u30FF\u3040-\u309F]{1,3}$'),  # ends with 1-3 kana
    # English-looking words mixed with Japanese (potential ASR artifacts)
    re.compile(r'[A-Z][a-z]+[,、]'),  # Capitalized English word followed by comma
    # Repeated comma-separated short fragments
    re.compile(r'(?:[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{1,4}[,、]\s*){3,}'),
]


def load_transcripts(limit=50):
    """Load recent transcript.json files."""
    transcripts = []
    for date_dir in sorted(Path(RECORDINGS_DIR).iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir(), reverse=True):
            transcript_path = session_dir / "transcript.json"
            if transcript_path.exists():
                try:
                    with open(transcript_path) as f:
                        data = json.load(f)
                    segments = data.get("segments", [])
                    if segments:
                        session_id = f"{date_dir.name}/{session_dir.name}"
                        transcripts.append((session_id, segments))
                except (json.JSONDecodeError, OSError):
                    continue
            if len(transcripts) >= limit:
                break
        if len(transcripts) >= limit:
            break
    return transcripts


def check_false_negatives(segments, clean_indices):
    """Check if any 'clean' segments actually look like they need correction."""
    false_negatives = []
    for idx in clean_indices:
        text = segments[idx].get("text", "")
        if len(text) < 20:
            continue  # Short segments are legitimately skipped

        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(text):
                false_negatives.append({
                    "index": idx,
                    "text": text,
                    "pattern": pattern.pattern,
                })
                break
    return false_negatives


def main():
    # Initialize screener
    tc = TermCorrector(os.path.join(CONFIG_DIR, "term_corrections.json"))
    sl = SuggestionLearner(os.path.join(CONFIG_DIR, "pending_suggestions.json"))

    # Mock hotword_loader with just get_custom_words
    class MockHotwordLoader:
        def get_custom_words(self):
            hotwords_custom = os.path.join(CONFIG_DIR, "hotwords_custom.json")
            if os.path.exists(hotwords_custom):
                try:
                    with open(hotwords_custom) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return data.get("words", [])
                except (json.JSONDecodeError, OSError):
                    pass
            return []

    hl = MockHotwordLoader()

    screener = SegmentScreener(
        term_corrector=tc,
        suggestion_learner=sl,
        hotword_loader=hl,
        min_text_length=20,
    )

    print(f"=== Screening Accuracy Verification ===")
    print(f"Term corrections: {len(tc.list_corrections())} entries")
    print(f"Learned patterns: {sl.pattern_count}")
    print(f"Known values: {len(screener._build_known_values())}")
    print()

    transcripts = load_transcripts(limit=50)
    print(f"Loaded {len(transcripts)} sessions")
    print()

    total_segments = 0
    total_clean = 0
    total_dirty = 0
    total_dict_corrections = 0
    total_learned_corrections = 0
    total_false_negatives = 0
    sessions_all_clean = 0
    sessions_all_dirty = 0

    false_negative_examples = []

    for session_id, segments in transcripts:
        result = screener.screen(segments)

        total_segments += len(segments)
        total_clean += len(result.clean_indices)
        total_dirty += len(result.dirty_indices)
        total_dict_corrections += result.metrics.dict_corrections
        total_learned_corrections += result.metrics.learned_corrections

        if not result.dirty_indices:
            sessions_all_clean += 1
        if not result.clean_indices:
            sessions_all_dirty += 1

        # Check for false negatives
        fn = check_false_negatives(result.segments, result.clean_indices)
        if fn:
            total_false_negatives += len(fn)
            for item in fn[:2]:  # Max 2 per session
                false_negative_examples.append({
                    "session": session_id,
                    **item,
                })

    print(f"=== Results ===")
    print(f"Total segments: {total_segments}")
    print(f"  Clean (skip LLM): {total_clean} ({total_clean/max(total_segments,1):.1%})")
    print(f"  Dirty (send LLM): {total_dirty} ({total_dirty/max(total_segments,1):.1%})")
    print(f"Dict corrections:   {total_dict_corrections}")
    print(f"Learned corrections:{total_learned_corrections}")
    print()
    print(f"Sessions all-clean: {sessions_all_clean}/{len(transcripts)}")
    print(f"Sessions all-dirty: {sessions_all_dirty}/{len(transcripts)}")
    print()

    api_call_estimate = len(transcripts) - sessions_all_clean
    print(f"=== API Call Estimate ===")
    print(f"Current (no screening): {len(transcripts)} sessions × 1 call = {len(transcripts)} calls")
    print(f"With screening: {api_call_estimate} calls (all-clean sessions skip)")
    reduction = (1 - api_call_estimate / max(len(transcripts), 1)) * 100
    print(f"Reduction: {reduction:.0f}%")
    print()

    # Show dirty segment examples
    print(f"=== Dirty Segment Examples (first 10) ===")
    dirty_shown = 0
    for session_id, segments in transcripts:
        result = screener.screen(segments)
        for idx in result.dirty_indices:
            text = result.segments[idx].get("text", "")
            # Find unknown katakana
            from segment_classifier import _KATAKANA_WORD_RE
            katakana_words = _KATAKANA_WORD_RE.findall(text)
            known = screener._build_known_values()
            unknown = [w for w in katakana_words if w not in known]
            print(f"  [{session_id}] seg{idx}: {text[:80]}...")
            if unknown:
                print(f"    → Unknown katakana: {', '.join(unknown)}")
            dirty_shown += 1
            if dirty_shown >= 10:
                break
        if dirty_shown >= 10:
            break
    print()

    # Show clean segment examples (with katakana)
    print(f"=== Clean Segment Examples with Katakana (first 10) ===")
    clean_shown = 0
    for session_id, segments in transcripts:
        result = screener.screen(segments)
        for idx in result.clean_indices:
            text = result.segments[idx].get("text", "")
            if len(text) < 20:
                continue
            from segment_classifier import _KATAKANA_WORD_RE
            katakana_words = _KATAKANA_WORD_RE.findall(text)
            if katakana_words:
                known = screener._build_known_values()
                all_known = all(w in known for w in katakana_words)
                status = "ALL KNOWN" if all_known else "MIXED"
                print(f"  [{session_id}] seg{idx}: {text[:80]}")
                print(f"    → Katakana: {', '.join(katakana_words)} ({status})")
                clean_shown += 1
                if clean_shown >= 10:
                    break
        if clean_shown >= 10:
            break
    print()

    # Show false negatives
    if false_negative_examples:
        print(f"=== Potential False Negatives ({total_false_negatives} total) ===")
        for item in false_negative_examples[:15]:
            print(f"  [{item['session']}] seg{item['index']}: {item['text'][:80]}")
            print(f"    → Pattern: {item['pattern']}")
    else:
        print(f"=== No potential false negatives detected ===")

    print()
    print(f"=== Summary ===")
    if total_false_negatives > 0:
        fn_rate = total_false_negatives / max(total_clean, 1) * 100
        print(f"⚠ False negative rate: {total_false_negatives}/{total_clean} ({fn_rate:.1f}%)")
        print(f"  Review the examples above to determine if these actually need LLM correction")
    else:
        print(f"✓ No false negatives detected in pattern check")
    print(f"✓ API call reduction: {reduction:.0f}%")


if __name__ == "__main__":
    main()
