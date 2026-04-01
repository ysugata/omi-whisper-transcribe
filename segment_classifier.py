"""Classify ASR segments as dirty (needs LLM) or clean (skip LLM).

Rule-based classification to minimize unnecessary LLM API calls.
"""

import re
import logging

log = logging.getLogger(__name__)

# Katakana detection pattern
_KATAKANA_RE = re.compile(r'[\u30A0-\u30FF]')
# Kanji + Hiragana pattern
_KANJI_HIRA_RE = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F]')
# Katakana word (3+ chars)
_KATAKANA_WORD_RE = re.compile(r'[\u30A0-\u30FF]{3,}')
# Trailing truncation pattern (sentence cut off mid-word)
_TRAILING_TRUNC_RE = re.compile(r'[\u30A0-\u30FF\u3040-\u309F\u4E00-\u9FFF]{1,3}$')


def classify_segment(
    segment: dict,
    known_values: set,
    min_text_length: int = 20,
) -> str:
    """Classify a single segment as 'dirty' or 'clean'.

    Args:
        segment: Segment dict with 'text' and optional 'llm_corrected' flag.
        known_values: Set of known correct values (from term_corrector + hotwords).
        min_text_length: Minimum text length to consider for LLM.

    Returns:
        'dirty' if the segment likely needs LLM correction, 'clean' otherwise.
    """
    # Rule 1: Already corrected by LLM → skip
    if segment.get("llm_corrected"):
        return "clean"

    text = segment.get("text", "")

    # Rule 2: Too short → skip
    if len(text) < min_text_length:
        return "clean"

    # Rule 3: Pure Japanese (low katakana, high kanji+hira) → clean
    katakana_chars = len(_KATAKANA_RE.findall(text))
    kanji_hira_chars = len(_KANJI_HIRA_RE.findall(text))
    total_chars = len(text)

    if total_chars > 0:
        katakana_ratio = katakana_chars / total_chars
        kanji_hira_ratio = kanji_hira_chars / total_chars
        if katakana_ratio < 0.2 and kanji_hira_ratio > 0.7:
            return "clean"

    # Rule 4: Unknown katakana words (not in dictionary/hotwords) → dirty
    katakana_words = _KATAKANA_WORD_RE.findall(text)
    for kw in katakana_words:
        if kw not in known_values:
            return "dirty"

    # Rule 5: Trailing truncation pattern → dirty
    # Check if text ends abruptly (common ASR artifact)
    if text and not text[-1] in "。、.!?！？」』）)…":
        # Text doesn't end with punctuation - check for truncation
        last_few = text[-6:] if len(text) >= 6 else text
        if _TRAILING_TRUNC_RE.search(last_few):
            # Only flag if the truncated part looks like a broken word
            # (not a complete word ending naturally)
            pass  # This is too aggressive; skip for now

    # Rule 6: Default → clean
    return "clean"


def classify_segments(
    segments: list,
    known_values: set,
    min_text_length: int = 20,
) -> tuple:
    """Classify all segments and return indices of dirty/clean ones.

    Returns:
        (dirty_indices: list[int], clean_indices: list[int])
    """
    dirty = []
    clean = []
    for i, seg in enumerate(segments):
        result = classify_segment(seg, known_values, min_text_length)
        if result == "dirty":
            dirty.append(i)
        else:
            clean.append(i)
    return dirty, clean
