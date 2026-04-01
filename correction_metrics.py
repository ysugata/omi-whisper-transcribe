"""Correction metrics for LLM screening pipeline."""

from dataclasses import dataclass, field


@dataclass
class CorrectionMetrics:
    """Track correction statistics for logging and monitoring."""
    total_segments: int = 0
    screened_out: int = 0       # Segments classified as clean (skipped)
    dirty_segments: int = 0     # Segments sent to LLM
    dict_corrections: int = 0   # Corrections from term_corrector
    learned_corrections: int = 0  # Corrections from suggestion_learner
    llm_corrections: int = 0    # Corrections from LLM pipeline
    api_calls: int = 0

    def format_log(self) -> str:
        """Format metrics for log output."""
        skip_rate = (
            f"{self.screened_out}/{self.total_segments} "
            f"({self.screened_out/self.total_segments:.0%})"
            if self.total_segments > 0 else "0/0"
        )
        return (
            f"total={self.total_segments}, "
            f"screened_out={skip_rate}, "
            f"dirty={self.dirty_segments}, "
            f"dict={self.dict_corrections}, "
            f"learned={self.learned_corrections}, "
            f"llm={self.llm_corrections}, "
            f"api_calls={self.api_calls}"
        )

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "total_segments": self.total_segments,
            "screened_out": self.screened_out,
            "dirty_segments": self.dirty_segments,
            "dict_corrections": self.dict_corrections,
            "learned_corrections": self.learned_corrections,
            "llm_corrections": self.llm_corrections,
            "api_calls": self.api_calls,
        }
