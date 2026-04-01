#!/usr/bin/env python3
"""Deep verification of screening accuracy.

Checks for false negatives that the segment_classifier might miss:
1. Non-katakana ASR errors (kanji misreadings, hiragana errors)
2. Short katakana (2 chars) that _KATAKANA_WORD_RE misses
3. Segments where LLM corrections were actually needed but screener says clean

Uses pending_suggestions.json patterns to check if clean segments
contain known correction patterns.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from term_corrector import TermCorrector
from suggestion_learner import SuggestionLearner
from segment_screener import SegmentScreener, _COMMON_LOANWORDS

CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")

# Katakana word patterns
_KATAKANA_WORD_3PLUS = re.compile(r'[\u30A0-\u30FF]{3,}')
_KATAKANA_WORD_2 = re.compile(r'(?<![ー\u30A0-\u30FF])[\u30A0-\u30FF]{2}(?![ー\u30A0-\u30FF])')
_ALL_KATAKANA = re.compile(r'[\u30A0-\u30FF]+')

# Patterns that suggest non-katakana ASR errors
# (kanji sequences that sound like English words)
_SUSPICIOUS_KANJI_FOR_ENGLISH = [
    # Pattern: unusual kanji compound that might be ASR mishearing English
    # These are actual examples from pending_suggestions
    ("炎症部屋", "Ansible"),
]


def load_transcripts(limit=100):
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
                        session_meta = data.get("session", {})
                        transcripts.append((session_id, segments, session_meta))
                except (json.JSONDecodeError, OSError):
                    continue
            if len(transcripts) >= limit:
                break
        if len(transcripts) >= limit:
            break
    return transcripts


def load_suggestions():
    path = os.path.join(CONFIG_DIR, "pending_suggestions.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def init_screener():
    tc = TermCorrector(os.path.join(CONFIG_DIR, "term_corrections.json"))
    sl = SuggestionLearner(os.path.join(CONFIG_DIR, "pending_suggestions.json"))

    class MockHotwordLoader:
        def get_custom_words(self):
            path = os.path.join(CONFIG_DIR, "hotwords_custom.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return data.get("words", [])
                except (json.JSONDecodeError, OSError):
                    pass
            return []

    hl = MockHotwordLoader()
    return SegmentScreener(term_corrector=tc, suggestion_learner=sl,
                           hotword_loader=hl, min_text_length=20)


def analyze_suggestion_coverage(suggestions, screener):
    """Check if each suggestion's 'wrong' pattern would be detected by screener."""
    known = screener._build_known_values()
    results = []
    for s in suggestions:
        wrong = s.get("wrong", "")
        correct = s.get("correct", "")
        context = s.get("context", "")

        # Would the 'wrong' text trigger the katakana check?
        katakana_3plus = _KATAKANA_WORD_3PLUS.findall(wrong)
        has_unknown_katakana = any(kw not in known for kw in katakana_3plus)

        # Check context for OTHER unknown katakana
        ctx_katakana = _KATAKANA_WORD_3PLUS.findall(context)
        ctx_unknown = [kw for kw in ctx_katakana if kw not in known]

        # Classify
        if has_unknown_katakana:
            detection = "DETECTED_SELF"
        elif ctx_unknown:
            detection = "DETECTED_CTX"
        else:
            detection = "MISSED"

        results.append({
            "wrong": wrong,
            "correct": correct,
            "detection": detection,
            "katakana_3plus_in_wrong": katakana_3plus,
            "unknown_in_context": ctx_unknown[:5],
        })
    return results


def check_clean_segment_quality(segments, clean_indices, suggestions, known):
    """Deep check: do any clean segments contain known error patterns?"""
    issues = []

    # Build pattern list from suggestions
    suggestion_wrongs = set()
    for s in suggestions:
        wrong = s.get("wrong", "").strip()
        if wrong and len(wrong) >= 2:
            suggestion_wrongs.add(wrong)

    for idx in clean_indices:
        text = segments[idx].get("text", "")
        if len(text) < 20:
            continue

        segment_issues = []

        # Check 1: Known wrong patterns from suggestions
        for wrong in suggestion_wrongs:
            if wrong in text:
                segment_issues.append(f"suggestion_match: '{wrong}'")

        # Check 2: Short katakana (2 chars) that might be ASR artifacts
        # (カモ, ウチ, ソウ etc. that should be hiragana)
        short_katakana = _KATAKANA_WORD_2.findall(text)
        # Filter: only flag if surrounded by hiragana/kanji (suggests ASR error)
        for sk in short_katakana:
            pos = text.find(sk)
            if pos > 0 and pos + len(sk) < len(text):
                before = text[pos-1]
                after = text[pos+len(sk)]
                # If surrounded by non-katakana, might be error
                if not re.match(r'[\u30A0-\u30FF]', before) and not re.match(r'[\u30A0-\u30FF]', after):
                    # But skip common 2-char katakana used in Japanese
                    if sk not in {"パン", "ドア", "バス", "ゲル", "メモ", "ネコ",
                                  "バカ", "ダメ", "カギ", "ヤバ", "カレ", "カナ",
                                  "ヒト", "モノ", "コト", "トキ", "トコ", "アレ",
                                  "ソレ", "コレ", "ナニ", "ドコ", "ダレ", "イヤ",
                                  "マジ", "ヤツ", "ワケ", "ホラ", "ネタ", "ムリ",
                                  "ウソ", "ゼロ", "ミス", "メス", "クセ", "ハズ",
                                  "アタ", "キセ", "ゼニ"}:
                        segment_issues.append(f"short_katakana: '{sk}'")

        # Check 3: Unusual kanji that might be ASR artifacts
        for kanji_pattern, english_form in _SUSPICIOUS_KANJI_FOR_ENGLISH:
            if kanji_pattern in text:
                segment_issues.append(f"suspicious_kanji: '{kanji_pattern}' → {english_form}")

        # Check 4: Comma-separated katakana fragments (ASR artifact pattern)
        comma_katakana = re.findall(r'[\u30A0-\u30FF]{2,4}[,、]\s*', text)
        if len(comma_katakana) >= 3:
            segment_issues.append(f"comma_katakana_fragments: {len(comma_katakana)}x")

        # Check 5: Mixed script anomaly (English words in mostly-Japanese text)
        # English words mixed with Japanese might indicate ASR issues
        english_words = re.findall(r'[A-Z][a-z]+', text)
        if english_words and len(text) > 30:
            jp_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
            if jp_chars / len(text) > 0.5:
                # Mostly Japanese but has English words
                segment_issues.append(f"mixed_english: {english_words}")

        if segment_issues:
            issues.append({
                "index": idx,
                "text": text[:100],
                "issues": segment_issues,
            })

    return issues


def main():
    print("=" * 70)
    print("DEEP SCREENING ACCURACY VERIFICATION")
    print("=" * 70)

    screener = init_screener()
    known = screener._build_known_values()
    suggestions = load_suggestions()
    transcripts = load_transcripts(limit=100)

    print(f"\nKnown values: {len(known)}")
    print(f"  _COMMON_LOANWORDS: {len(_COMMON_LOANWORDS)}")
    print(f"  From term_corrector + hotwords: {len(known) - len(_COMMON_LOANWORDS)}")
    print(f"Suggestions: {len(suggestions)}")
    print(f"Sessions loaded: {len(transcripts)}")

    # ---------------------------------------------------------------
    # Part 1: Suggestion coverage analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PART 1: SUGGESTION PATTERN COVERAGE")
    print(f"{'='*70}")
    print("Do screener's rules catch the patterns that LLM has historically corrected?\n")

    coverage = analyze_suggestion_coverage(suggestions, screener)
    detected_self = [c for c in coverage if c["detection"] == "DETECTED_SELF"]
    detected_ctx = [c for c in coverage if c["detection"] == "DETECTED_CTX"]
    missed = [c for c in coverage if c["detection"] == "MISSED"]

    print(f"  DETECTED (wrong has unknown katakana 3+): {len(detected_self)}/{len(coverage)}")
    for c in detected_self:
        print(f"    '{c['wrong']}' → '{c['correct']}'  kata={c['katakana_3plus_in_wrong']}")

    print(f"\n  DETECTED (other unknown katakana in context): {len(detected_ctx)}/{len(coverage)}")
    for c in detected_ctx:
        print(f"    '{c['wrong']}' → '{c['correct']}'  ctx_unknown={c['unknown_in_context']}")

    print(f"\n  MISSED (no unknown katakana signal): {len(missed)}/{len(coverage)}")
    for c in missed:
        print(f"    '{c['wrong']}' → '{c['correct']}'  kata={c['katakana_3plus_in_wrong']}")

    miss_rate = len(missed) / max(len(coverage), 1) * 100
    print(f"\n  Suggestion miss rate: {miss_rate:.1f}%")

    # ---------------------------------------------------------------
    # Part 2: Deep clean segment analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PART 2: DEEP CLEAN SEGMENT ANALYSIS")
    print(f"{'='*70}")
    print("Checking clean segments for hidden errors...\n")

    total_segments = 0
    total_clean = 0
    total_dirty = 0
    all_issues = []
    sessions_with_issues = 0

    for session_id, segments, meta in transcripts:
        result = screener.screen(segments)
        total_segments += len(segments)
        total_clean += len(result.clean_indices)
        total_dirty += len(result.dirty_indices)

        issues = check_clean_segment_quality(
            result.segments, result.clean_indices, suggestions, known,
        )
        if issues:
            sessions_with_issues += 1
            for issue in issues:
                all_issues.append({"session": session_id, **issue})

    print(f"  Total segments: {total_segments}")
    print(f"  Clean: {total_clean} ({total_clean/max(total_segments,1):.1%})")
    print(f"  Dirty: {total_dirty} ({total_dirty/max(total_segments,1):.1%})")
    print(f"  Sessions with issues in clean segments: {sessions_with_issues}/{len(transcripts)}")
    print(f"  Total issues found: {len(all_issues)}")

    # Categorize issues
    by_type = {}
    for issue in all_issues:
        for iss in issue["issues"]:
            category = iss.split(":")[0]
            by_type.setdefault(category, []).append(issue)

    print(f"\n  Issues by type:")
    for cat, items in sorted(by_type.items()):
        print(f"    {cat}: {len(items)}")

    # Show examples
    print(f"\n  --- Issue Examples (all) ---")
    for issue in all_issues[:30]:
        print(f"\n  [{issue['session']}] seg{issue['index']}:")
        print(f"    text: {issue['text']}")
        for iss in issue["issues"]:
            print(f"    >> {iss}")

    # ---------------------------------------------------------------
    # Part 3: LLM-corrected session analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PART 3: LLM-CORRECTED SESSION ANALYSIS")
    print(f"{'='*70}")
    print("For already-corrected sessions, what would screening have done?\n")

    for session_id, segments, meta in transcripts:
        if not meta.get("llm_corrected"):
            continue

        print(f"  Session: {session_id}")
        print(f"    llm_stages: {meta.get('llm_stages')}")
        print(f"    llm_api_calls: {meta.get('llm_api_calls')}")
        print(f"    Segments: {len(segments)}")

        result = screener.screen(segments)
        print(f"    Screener: clean={len(result.clean_indices)}, dirty={len(result.dirty_indices)}")

        for idx in result.dirty_indices:
            text = result.segments[idx].get("text", "")[:80]
            kata = _KATAKANA_WORD_3PLUS.findall(result.segments[idx].get("text", ""))
            unknown = [k for k in kata if k not in known]
            print(f"    dirty[{idx}]: {text}")
            if unknown:
                print(f"      unknown: {unknown}")

        for idx in result.clean_indices:
            text = result.segments[idx].get("text", "")
            if len(text) >= 20:
                print(f"    clean[{idx}]: {text[:80]}")

    # ---------------------------------------------------------------
    # Part 4: Non-katakana error risk assessment
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PART 4: NON-KATAKANA ERROR RISK ASSESSMENT")
    print(f"{'='*70}")

    # Count segments by katakana content
    no_katakana_count = 0
    all_known_katakana = 0
    has_unknown_katakana = 0

    for session_id, segments, meta in transcripts:
        result = screener.screen(segments)
        for idx in result.clean_indices:
            text = result.segments[idx].get("text", "")
            if len(text) < 20:
                continue
            kata = _KATAKANA_WORD_3PLUS.findall(text)
            unknown = [k for k in kata if k not in known]
            if not kata:
                no_katakana_count += 1
            elif not unknown:
                all_known_katakana += 1
            else:
                has_unknown_katakana += 1  # Should not happen (would be dirty)

    total_eligible_clean = no_katakana_count + all_known_katakana + has_unknown_katakana
    print(f"\n  Clean segments (len >= 20):")
    print(f"    No katakana at all: {no_katakana_count} ({no_katakana_count/max(total_eligible_clean,1):.1%})")
    print(f"    All known katakana: {all_known_katakana} ({all_known_katakana/max(total_eligible_clean,1):.1%})")
    print(f"    Has unknown katakana (bug?): {has_unknown_katakana}")

    print(f"\n  Risk analysis:")
    print(f"    '{no_katakana_count}' segments have NO katakana → screener has ZERO ability")
    print(f"    to detect non-katakana ASR errors in these segments.")
    print(f"    Examples of errors the screener fundamentally cannot catch:")
    print(f"    - Kanji misreading: 炎症部屋 → Ansible")
    print(f"    - Homophone confusion: 変える → 帰る")
    print(f"    - Missing/wrong particles: が/を/は confusion")
    print(f"    - Context-dependent: 切手 vs 切って")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Screening performance:")
    print(f"    Total segments: {total_segments}")
    print(f"    Clean (LLM skip): {total_clean} ({total_clean/max(total_segments,1):.1%})")
    print(f"    Dirty (LLM send): {total_dirty} ({total_dirty/max(total_segments,1):.1%})")

    print(f"\n  Historical correction coverage:")
    print(f"    Suggestions detected by screener: {len(detected_self)+len(detected_ctx)}/{len(coverage)} ({(len(detected_self)+len(detected_ctx))/max(len(coverage),1):.0%})")
    print(f"    Suggestions missed: {len(missed)}/{len(coverage)} ({miss_rate:.0%})")

    print(f"\n  Clean segment issue analysis:")
    print(f"    Segments with detectable issues: {len(all_issues)}/{total_clean}")
    issue_rate = len(all_issues) / max(total_clean, 1) * 100
    print(f"    Issue rate: {issue_rate:.1f}%")

    print(f"\n  Fundamental limitation:")
    print(f"    {no_katakana_count} clean segments have no katakana at all.")
    print(f"    These segments CANNOT be checked by the current screener.")
    print(f"    Non-katakana ASR errors in these segments will be missed.")

    if len(all_issues) > 0 or len(missed) > 0:
        print(f"\n  VERDICT: Potential accuracy loss exists.")
        print(f"  See detailed issue list above for specifics.")
    else:
        print(f"\n  VERDICT: No detectable accuracy loss in current data.")


if __name__ == "__main__":
    main()
