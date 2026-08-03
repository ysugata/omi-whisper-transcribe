"""Auto-learn high-confidence correction patterns from pending_suggestions.json.

Extracts patterns where katakana → English/technical term corrections
appear 2+ times with 3+ characters, and applies them deterministically
without needing LLM calls.
"""

import json
import logging
import os
import re
from pathlib import Path

from term_corrector import compile_boundary_pattern

log = logging.getLogger(__name__)

# Minimum occurrences before a suggestion is considered learned
_MIN_OCCURRENCES = 2
# Minimum character length for wrong/correct
_MIN_LENGTH = 3


class SuggestionLearner:
    """Learns from pending_suggestions.json and applies high-confidence corrections."""

    def __init__(self, suggestions_path: str) -> None:
        self._path = Path(suggestions_path)
        self._mtime: float = 0.0
        self._learned: dict[str, str] = {}  # wrong → correct
        self._load()

    def _load(self) -> None:
        """Load and analyze pending_suggestions.json for learnable patterns."""
        if not self._path.exists():
            self._learned = {}
            return
        try:
            mtime = self._path.stat().st_mtime
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._mtime = mtime
            if not isinstance(data, list):
                # A dict-shaped file would raise AttributeError inside
                # _extract_patterns and abort bridge startup entirely.
                log.warning("SuggestionLearner: %s is not a list, ignoring", self._path)
                self._learned = {}
                return
            self._learned = self._extract_patterns(data)
            if self._learned:
                log.info(
                    "SuggestionLearner: loaded %d learned pattern(s) from %s",
                    len(self._learned), self._path,
                )
        except (json.JSONDecodeError, OSError) as e:
            log.warning("SuggestionLearner: failed to load %s: %s", self._path, e)
            self._learned = {}

    def _reload_if_changed(self) -> None:
        """Reload if file has been modified."""
        try:
            if self._path.exists():
                mtime = self._path.stat().st_mtime
                if mtime > self._mtime:
                    self._load()
        except OSError:
            pass

    @staticmethod
    def _extract_patterns(suggestions: list) -> dict[str, str]:
        """Extract high-confidence patterns from suggestion list.

        Criteria:
        - wrong is katakana (3+ chars)
        - correct is English/technical term or standard Japanese (3+ chars)
        - Pattern appears 2+ times
        """
        counts: dict[tuple[str, str], int] = {}
        for s in suggestions:
            wrong = s.get("wrong", "").strip()
            correct = s.get("correct", "").strip()
            if not wrong or not correct:
                continue
            if len(wrong) < _MIN_LENGTH or len(correct) < _MIN_LENGTH:
                continue
            # Skip if wrong contains comma (multi-value entries are unreliable)
            if "," in wrong:
                continue
            # wrong should be primarily katakana or a clear ASR artifact
            katakana_ratio = len(re.findall(r'[\u30A0-\u30FF]', wrong)) / len(wrong)
            if katakana_ratio < 0.5:
                continue
            counts[(wrong, correct)] = counts.get((wrong, correct), 0) + 1

        learned = {}
        for (wrong, correct), count in counts.items():
            if count >= _MIN_OCCURRENCES:
                learned[wrong] = correct
        return learned

    def apply_learned(self, text: str) -> tuple[str, int]:
        """Apply learned corrections to text.

        Word-boundary aware: katakana patterns do not match inside larger
        katakana compounds (same rule as TermCorrector).

        Returns:
            (corrected_text, number_of_corrections_applied)
        """
        self._reload_if_changed()
        corrections = 0
        for wrong in sorted(self._learned, key=len, reverse=True):
            correct = self._learned[wrong]
            if not wrong or wrong == correct:
                continue
            pattern = compile_boundary_pattern(wrong)
            text, n = pattern.subn(lambda _m, _c=correct: _c, text)
            if n:
                corrections += 1
        return text, corrections

    def get_learned_patterns(self) -> dict[str, str]:
        """Return current learned patterns."""
        self._reload_if_changed()
        return dict(self._learned)

    @property
    def pattern_count(self) -> int:
        """Number of learned patterns."""
        self._reload_if_changed()
        return len(self._learned)
