"""Phonetic-similarity based ASR error correction.

Corrects ASR misrecognitions by matching output text against a
known-correct glossary using phonetic (reading) similarity.
Works with both Whisper and Qwen3-ASR output.

Dependencies: pykakasi (kanji→kana/romaji conversion)
Optional: MeCab + unidic-lite (morphological analysis for better tokenization)

No GPU required — runs entirely on CPU.
"""

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

# Lazy-loaded singletons
_kakasi = None
_mecab_tagger = None


def _get_kakasi():
    """Lazy-load pykakasi converter."""
    global _kakasi
    if _kakasi is None:
        try:
            import pykakasi
            _kakasi = pykakasi.kakasi()
            log.info("pykakasi loaded")
        except ImportError:
            log.warning("pykakasi not installed, phonetic correction disabled")
    return _kakasi


def _get_mecab():
    """Lazy-load MeCab tagger (optional, for better tokenization)."""
    global _mecab_tagger
    if _mecab_tagger is None:
        try:
            import MeCab
            _mecab_tagger = MeCab.Tagger("-Owakati")
            log.info("MeCab loaded")
        except ImportError:
            log.debug("MeCab not available, using regex tokenization")
    return _mecab_tagger


def _to_reading(text: str) -> str:
    """Convert Japanese text to katakana reading using pykakasi.

    Returns empty string if pykakasi is not available.
    """
    kakasi = _get_kakasi()
    if kakasi is None:
        return ""
    try:
        items = kakasi.convert(text)
        return "".join(item["kana"] for item in items)
    except Exception:
        return ""


def _to_romaji(text: str) -> str:
    """Convert Japanese text to romaji for distance calculation."""
    kakasi = _get_kakasi()
    if kakasi is None:
        return ""
    try:
        items = kakasi.convert(text)
        return "".join(item["hepburn"] for item in items)
    except Exception:
        return ""


@lru_cache(maxsize=4096)
def _cached_romaji(text: str) -> str:
    return _to_romaji(text)


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _phonetic_similarity(word: str, candidate: str) -> float:
    """Compute phonetic similarity between two Japanese words.

    Returns 0.0 (totally different) to 1.0 (identical pronunciation).
    Uses romaji-level edit distance for language-agnostic comparison.
    """
    r1 = _cached_romaji(word)
    r2 = _cached_romaji(candidate)
    if not r1 or not r2:
        return 0.0
    max_len = max(len(r1), len(r2))
    if max_len == 0:
        return 1.0
    dist = _edit_distance(r1, r2)
    return 1.0 - (dist / max_len)


# Tokenization patterns
_KATAKANA_WORD_RE = re.compile(r'[\u30A0-\u30FF]{2,}')
_HIRAGANA_WORD_RE = re.compile(r'[\u3040-\u309F]{3,}')
_MIXED_WORD_RE = re.compile(r'[\u3040-\u30FF]{2,}')


class PhoneticCorrector:
    """ASR error corrector using phonetic similarity matching.

    Builds a glossary from hotwords and term corrections, then
    matches katakana/hiragana words in ASR output against the glossary
    using pronunciation (romaji) similarity.

    Typical correction: クーバネティス → Kubernetes (読み: くーばねてぃす ≈ くーばねてぃす)
    """

    def __init__(
        self,
        glossary: list[str] | None = None,
        threshold: float = 0.75,
        min_word_length: int = 3,
    ):
        """
        Args:
            glossary: List of correct terms to match against.
            threshold: Minimum phonetic similarity (0.0-1.0) for correction.
            min_word_length: Minimum characters in a word to attempt correction.
        """
        self._glossary: list[str] = glossary or []
        self._glossary_romaji: dict[str, str] = {}
        self._threshold = threshold
        self._min_word_length = min_word_length
        self._corrections_made = 0

        if self._glossary:
            self._build_index()

    def _build_index(self):
        """Pre-compute romaji for all glossary terms."""
        for term in self._glossary:
            romaji = _to_romaji(term)
            if romaji:
                self._glossary_romaji[term] = romaji
        log.info(
            "PhoneticCorrector: %d glossary terms indexed (%d with romaji)",
            len(self._glossary), len(self._glossary_romaji),
        )

    def set_glossary(self, glossary: list[str]):
        """Update glossary and rebuild index."""
        self._glossary = glossary
        self._glossary_romaji = {}
        self._build_index()

    @property
    def corrections_made(self) -> int:
        return self._corrections_made

    def correct(self, text: str) -> str:
        """Correct ASR output using phonetic similarity matching.

        Finds katakana/hiragana words in the text and checks if any
        glossary term has a similar pronunciation. If so, replaces
        the misrecognized word with the correct term.

        Returns corrected text.
        """
        if not self._glossary_romaji or not text:
            return text

        kakasi = _get_kakasi()
        if kakasi is None:
            return text

        # Extract candidate words for correction (katakana words are most likely ASR errors)
        result = text
        candidates = _KATAKANA_WORD_RE.findall(text)

        for word in candidates:
            if len(word) < self._min_word_length:
                continue

            best_match = self._find_best_match(word)
            if best_match and best_match != word:
                result = result.replace(word, best_match, 1)
                self._corrections_made += 1
                log.debug("Phonetic correction: '%s' → '%s'", word, best_match)

        return result

    def _find_best_match(self, word: str) -> str | None:
        """Find the best phonetic match from the glossary.

        Returns the glossary term if similarity >= threshold, else None.
        """
        word_romaji = _cached_romaji(word)
        if not word_romaji:
            return None

        best_sim = 0.0
        best_term = None

        for term, term_romaji in self._glossary_romaji.items():
            # Quick length filter: skip if romaji lengths differ too much
            len_ratio = len(word_romaji) / len(term_romaji) if term_romaji else 0
            if len_ratio < 0.5 or len_ratio > 2.0:
                continue

            dist = _edit_distance(word_romaji, term_romaji)
            max_len = max(len(word_romaji), len(term_romaji))
            sim = 1.0 - (dist / max_len)

            if sim > best_sim:
                best_sim = sim
                best_term = term

        if best_sim >= self._threshold:
            return best_term
        return None
