"""Segment screener: orchestrates deterministic pre-processing before LLM.

Applies dictionary corrections, auto-learned corrections, and classifies
segments as dirty (needs LLM) or clean (skip LLM).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from correction_metrics import CorrectionMetrics
from segment_classifier import classify_segments

log = logging.getLogger(__name__)

# Katakana word pattern for building known_values
_KATAKANA_WORD_RE = re.compile(r'[\u30A0-\u30FF]{3,}')


@dataclass
class ScreeningResult:
    """Result of segment screening."""
    segments: list                      # All segments (with dict/learned corrections applied)
    dirty_indices: list = field(default_factory=list)
    clean_indices: list = field(default_factory=list)
    metrics: CorrectionMetrics = field(default_factory=CorrectionMetrics)


class SegmentScreener:
    """Orchestrates pre-LLM deterministic processing."""

    def __init__(
        self,
        term_corrector=None,
        suggestion_learner=None,
        hotword_loader=None,
        min_text_length: int = 20,
    ) -> None:
        self._term_corrector = term_corrector
        self._suggestion_learner = suggestion_learner
        self._hotword_loader = hotword_loader
        self._min_text_length = min_text_length

    def screen(self, segments: list) -> ScreeningResult:
        """Screen segments: apply deterministic corrections then classify.

        Steps:
        1. Apply term_corrector (dictionary corrections)
        2. Apply suggestion_learner (auto-learned corrections)
        3. Classify each segment as dirty or clean
        4. Return ScreeningResult with corrected segments and indices

        Args:
            segments: List of segment dicts with 'text' field.

        Returns:
            ScreeningResult with corrected segments and dirty/clean indices.
        """
        metrics = CorrectionMetrics(total_segments=len(segments))

        # Work on copies to avoid mutating originals
        result_segments = [dict(seg) for seg in segments]

        # Step 1: Dictionary corrections
        if self._term_corrector:
            for seg in result_segments:
                text = seg.get("text", "")
                if len(text) >= self._min_text_length:
                    corrected, count = self._term_corrector.correct_count(text)
                    if count > 0:
                        seg["text"] = corrected
                        metrics.dict_corrections += count

        # Step 2: Auto-learned corrections
        if self._suggestion_learner:
            for seg in result_segments:
                text = seg.get("text", "")
                if len(text) >= self._min_text_length:
                    corrected, count = self._suggestion_learner.apply_learned(text)
                    if count > 0:
                        seg["text"] = corrected
                        metrics.learned_corrections += count

        # Step 3: Build known_values set for classification
        known_values = self._build_known_values()

        # Step 4: Classify
        dirty_indices, clean_indices = classify_segments(
            result_segments, known_values, self._min_text_length,
        )

        metrics.dirty_segments = len(dirty_indices)
        metrics.screened_out = len(clean_indices)

        log.info("Screener: %s", metrics.format_log())

        return ScreeningResult(
            segments=result_segments,
            dirty_indices=dirty_indices,
            clean_indices=clean_indices,
            metrics=metrics,
        )

    def _build_known_values(self) -> set:
        """Build set of known katakana values from dictionary + hotwords.

        Includes:
        - correction target values from term_corrector (wrong side, since those are the katakana we recognize)
        - katakana words from hotword_loader custom words
        """
        known = set()

        # Add correction keys (wrong side = katakana we know about)
        if self._term_corrector:
            for wrong in self._term_corrector.list_corrections():
                known.add(wrong)
            # Also add the correct values that are katakana
            for val in self._term_corrector.get_known_values():
                katakana_words = _KATAKANA_WORD_RE.findall(val)
                known.update(katakana_words)

        # Add hotword custom words
        if self._hotword_loader:
            for word in self._hotword_loader.get_custom_words():
                known.add(word)

        return known
