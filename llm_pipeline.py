"""Layer 3: 3-stage LLM pipeline for ASR transcript correction.

Stage 3-A: Term candidate extraction (no text changes)
Stage 3-B: Notation correction (diff-only corrections)
Stage 3-C: Quality audit (detect remaining issues, optionally retry 3-B)

Uses OpenAI-compatible API (Gemini via OpenAI shim by default).
All stages gracefully degrade: API errors return original text unchanged.
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llm_pipeline_prompt import (
    AUDIT_SYSTEM_PROMPT,
    CANDIDATE_SYSTEM_PROMPT,
    CORRECTION_SYSTEM_PROMPT,
)
from llm_pipeline_schema import AUDIT_SCHEMA, CANDIDATE_SCHEMA, CORRECTION_SCHEMA

log = logging.getLogger(__name__)

# Punctuation chars for diff filtering (reused from llm_corrector)
_PUNCT_CHARS = set('、。,.!?！？・…―「」『』（）()-–—　 ')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TermCandidate:
    line: int
    text_span: str
    category: str
    suggested_form: str
    confidence: str


@dataclass
class CandidateResult:
    candidates: list
    inferred_domain: str = ""
    raw_response: str = ""


@dataclass
class AuditItem:
    line: int
    text_span: str
    issue_type: str
    severity: str
    suggestion: str


@dataclass
class AuditResult:
    items: list
    passed: bool = True
    raw_response: str = ""


@dataclass
class PipelineResult:
    corrected_segments: list
    candidates: Optional[CandidateResult] = None
    audit: Optional[AuditResult] = None
    stages_run: list = field(default_factory=list)
    api_calls: int = 0
    retried: bool = False
    segments_total: int = 0
    segments_sent: int = 0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class LLMPipeline:
    """3-stage LLM pipeline for ASR transcript correction."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-3.1-flash-lite",
        temperature: float = 0,
        timeout: int = 10,
        stages: str = "ABC",
        max_tokens_a: int = 2048,
        max_tokens_b: int = 4096,
        max_tokens_c: int = 2048,
    ) -> None:
        self._enabled = bool(api_key)
        self._model = model
        self._temperature = temperature
        self._stages = stages.upper()
        self._max_tokens_a = max_tokens_a
        self._max_tokens_b = max_tokens_b
        self._max_tokens_c = max_tokens_c
        self._client = None

        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                log.info("LLMPipeline initialized: model=%s, stages=%s", model, stages)
            except Exception as e:
                log.error("LLMPipeline init failed: %s", e)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Internal: API call
    # ------------------------------------------------------------------

    def _call_api(
        self,
        system_prompt: str,
        user_message: str,
        response_format: dict,
        max_tokens: int,
        cache_key: Optional[str] = None,
    ) -> Optional[str]:
        """Call the LLM API and return raw response content, or None on error."""
        if not self._client:
            return None
        try:
            kwargs = dict(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format=response_format,
                temperature=self._temperature,
                max_tokens=max_tokens,
            )
            # Minimize thinking tokens for cost control.
            # Gemini 3.x: reasoning_effort="low" (cannot be fully disabled).
            # Gemini 2.5 Flash: reasoning_effort="none" disables thinking.
            if "gemini-3" in self._model:
                kwargs["reasoning_effort"] = "low"
            elif self._model.startswith("gemini-2.5-flash") and "lite" not in self._model:
                kwargs["reasoning_effort"] = "none"
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            log.warning("LLM API error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Internal: Build numbered input
    # ------------------------------------------------------------------

    @staticmethod
    def _build_numbered_input(
        segments: list,
        min_text_length: int = 0,
        segment_indices: Optional[list] = None,
    ) -> tuple:
        """Build numbered text input and mapping from line number to text.

        Args:
            segments: List of segment dicts.
            min_text_length: Minimum text length to include.
            segment_indices: If provided, only include these segment indices.

        Returns:
            (numbered_text, {1-indexed line: original text}, {1-indexed line: segment index})
        """
        lines = []
        original_by_line = {}
        line_to_seg_idx = {}
        line_num = 0
        for i, seg in enumerate(segments):
            if segment_indices is not None and i not in segment_indices:
                continue
            text = seg.get("text", "")
            if len(text) >= min_text_length:
                line_num += 1
                lines.append(f"[{line_num}] {text}")
                original_by_line[line_num] = text
                line_to_seg_idx[line_num] = i
        return "\n".join(lines), original_by_line, line_to_seg_idx

    # ------------------------------------------------------------------
    # Internal: Validation & filtering (ported from llm_corrector)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_response(original: str, corrected: str) -> str:
        """Validate LLM correction; return original if anomalous."""
        if not corrected or not corrected.strip():
            return original
        corrected = corrected.strip()
        if len(corrected) > len(original) * 2 or len(corrected) < len(original) * 0.5:
            log.warning(
                "LLM output length anomaly: orig=%d corrected=%d",
                len(original), len(corrected),
            )
            return original
        ratio = difflib.SequenceMatcher(None, original, corrected).ratio()
        if ratio < 0.5:
            log.warning(
                "LLM output diverged (similarity=%.2f): '%s'",
                ratio, corrected[:60],
            )
            return original
        return corrected

    @staticmethod
    def _is_punctuation_only_change(original: str, corrected: str) -> bool:
        """Check if the only difference is punctuation."""
        if original == corrected:
            return True
        strip = lambda s: ''.join(c for c in s if c not in _PUNCT_CHARS)
        return strip(original) == strip(corrected)

    @staticmethod
    def _filter_corrections(
        corrections: list,
        original_texts: dict,
        candidates: Optional["CandidateResult"] = None,
        session_glossary: Optional[list] = None,
    ) -> list:
        """Post-filter corrections: remove no-ops, punct-only, empty snippets,
        and unauthorized changes (corrected text not backed by glossary or 3-A)."""
        # Build set of authorized target forms from glossary + 3-A candidates
        authorized_forms: set[str] = set()
        if session_glossary:
            authorized_forms.update(session_glossary)
        candidate_map: dict[tuple, str] = {}  # (line, text_span) -> suggested_form
        if candidates:
            for c in candidates.candidates:
                if c.suggested_form:
                    authorized_forms.add(c.suggested_form)
                    candidate_map[(c.line, c.text_span)] = c.suggested_form

        filtered = []
        for c in corrections:
            line = c.get("line", 0)
            original_snippet = c.get("original_snippet", "").strip()
            corrected = c.get("corrected", "").strip()
            source = c.get("source", "")

            # Empty original_snippet is invalid
            if not original_snippet:
                log.debug("Filter: empty original_snippet on line %d", line)
                continue

            # No-change
            if original_snippet == corrected:
                log.debug("Filter: no-change on line %d", line)
                continue

            # Empty corrected
            if not corrected:
                log.debug("Filter: empty corrected on line %d", line)
                continue

            # Punctuation-only change
            if LLMPipeline._is_punctuation_only_change(original_snippet, corrected):
                log.debug(
                    "Filter: punct-only on line %d: '%s' → '%s'",
                    line, original_snippet[:40], corrected[:40],
                )
                continue

            # Validate snippet exists in original text
            original_text = original_texts.get(line, "")
            if original_text and original_snippet not in original_text:
                log.debug(
                    "Filter: snippet not found in original on line %d: '%s'",
                    line, original_snippet[:40],
                )
                continue

            # Guard: reject unauthorized changes not backed by glossary or 3-A.
            # Allow: source=glossary/candidate_3a, dictionary corrections (common
            # words/idioms), and small edits (truncation fix, typo).
            # Reject: large substitutions where corrected text is completely
            # different and not in any authorized source.
            if source == "context" and authorized_forms:
                # Check if the corrected text (or a key part) is authorized
                is_authorized = any(
                    form in corrected or corrected in form
                    for form in authorized_forms
                )
                # Small edit (<=3 char diff) is likely truncation fix, allow
                len_diff = abs(len(corrected) - len(original_snippet))
                is_small_edit = len_diff <= 3 and len(original_snippet) >= 3

                if not is_authorized and not is_small_edit:
                    log.info(
                        "Filter: unauthorized context change on line %d: "
                        "'%s' → '%s' (not in glossary/candidates)",
                        line, original_snippet[:40], corrected[:40],
                    )
                    continue

            filtered.append(c)
        return filtered

    @staticmethod
    def _apply_corrections(
        segments: list,
        corrections: list,
        line_to_seg_idx: dict,
        original_by_line: dict,
    ) -> list:
        """Apply filtered corrections to segments (copy), return new list."""
        result = [dict(seg) for seg in segments]
        for c in corrections:
            line = c.get("line", 0)
            seg_idx = line_to_seg_idx.get(line)
            if seg_idx is None:
                continue
            original_snippet = c.get("original_snippet", "").strip()
            corrected = c.get("corrected", "").strip()
            if not original_snippet or not corrected:
                continue

            current_text = result[seg_idx].get("text", "")
            if original_snippet in current_text:
                new_text = current_text.replace(original_snippet, corrected, 1)
                validated = LLMPipeline._validate_response(current_text, new_text)
                result[seg_idx]["text"] = validated
                if validated != current_text:
                    log.debug(
                        "Applied [%d]: '%s' → '%s'",
                        line, original_snippet[:40], corrected[:40],
                    )
        return result

    # ------------------------------------------------------------------
    # Stage 3-A: Term candidate extraction
    # ------------------------------------------------------------------

    def extract_candidates(
        self,
        segments: list,
        known_terms: Optional[list] = None,
        session_glossary: Optional[list] = None,
        min_text_length: int = 0,
        segment_indices: Optional[list] = None,
    ) -> Optional[CandidateResult]:
        """Stage 3-A: Extract term candidates without modifying text."""
        if "A" not in self._stages:
            return None

        numbered_text, _, _ = self._build_numbered_input(
            segments, min_text_length, segment_indices=segment_indices,
        )
        if not numbered_text:
            return CandidateResult(candidates=[])

        # Build user message
        # Use only session_glossary (text-matched terms) to avoid domain bias.
        # known_terms are NOT added here — they are already reflected in
        # session_glossary via _build_session_glossary() which filters by
        # substring match against actual segment text.
        parts = []
        if session_glossary:
            parts.append(f"参考用語リスト: {', '.join(session_glossary[:50])}")
        parts.append(numbered_text)
        user_msg = "\n\n".join(parts)

        raw = self._call_api(
            CANDIDATE_SYSTEM_PROMPT, user_msg, CANDIDATE_SCHEMA, self._max_tokens_a,
        )
        if raw is None:
            log.warning("3-A: API call failed, continuing without candidates")
            return None

        try:
            data = json.loads(raw)
            candidates = []
            for c in data.get("candidates", []):
                candidates.append(TermCandidate(
                    line=c.get("line", 0),
                    text_span=c.get("text_span", ""),
                    category=c.get("category", ""),
                    suggested_form=c.get("suggested_form", ""),
                    confidence=c.get("confidence", "low"),
                ))
            inferred_domain = data.get("inferred_domain", "")
            log.info("3-A: Extracted %d candidate(s), domain=%s", len(candidates), inferred_domain or "unknown")
            return CandidateResult(candidates=candidates, inferred_domain=inferred_domain, raw_response=raw)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("3-A: JSON parse failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Stage 3-B: Notation correction
    # ------------------------------------------------------------------

    def correct_notation(
        self,
        segments: list,
        candidates: Optional[CandidateResult] = None,
        known_terms: Optional[list] = None,
        session_glossary: Optional[list] = None,
        min_text_length: int = 0,
        audit_hints: Optional[list] = None,
        segment_indices: Optional[list] = None,
    ) -> list:
        """Stage 3-B: Apply notation corrections, return new segment list."""
        if "B" not in self._stages:
            return [dict(seg) for seg in segments]

        numbered_text, original_by_line, line_to_seg_idx = self._build_numbered_input(
            segments, min_text_length, segment_indices=segment_indices,
        )
        if not numbered_text:
            return [dict(seg) for seg in segments]

        # Build user message
        parts = []

        # Inferred domain from 3-A
        if candidates and candidates.inferred_domain:
            parts.append(f"inferred_domain: {candidates.inferred_domain}")

        # Glossary — only text-matched terms to avoid domain bias
        if session_glossary:
            parts.append(f"参考用語リスト: {', '.join(session_glossary[:50])}")

        # 3-A candidates
        if candidates and candidates.candidates:
            candidate_lines = []
            for c in candidates.candidates:
                suggested = f'"{c.suggested_form}"' if c.suggested_form else "不明"
                candidate_lines.append(
                    f"- [行{c.line}] 「{c.text_span}」→ {suggested} "
                    f"({c.category}, {c.confidence})"
                )
            parts.append("## 用語候補 (3-A 抽出結果)\n" + "\n".join(candidate_lines))

        # Audit hints (retry)
        if audit_hints:
            hint_lines = []
            for item in audit_hints:
                hint_lines.append(
                    f"- [行{item.line}] {item.issue_type}: 「{item.text_span}」"
                    f" → {item.suggestion}"
                )
            parts.append(
                "## 品質監査の指摘 (要修正)\n"
                "以下の問題が前回の修正で解消されていません。重点的に修正してください。\n"
                + "\n".join(hint_lines)
            )

        parts.append(numbered_text)
        user_msg = "\n\n".join(parts)

        raw = self._call_api(
            CORRECTION_SYSTEM_PROMPT, user_msg, CORRECTION_SCHEMA, self._max_tokens_b,
        )
        if raw is None:
            log.warning("3-B: API call failed, returning original segments")
            return [dict(seg) for seg in segments]

        log.info("3-B request:\n%s", user_msg)
        log.info("3-B response:\n%s", raw)

        try:
            data = json.loads(raw)
            corrections = data.get("corrections", [])
            corrections = self._filter_corrections(
                corrections, original_by_line, candidates, session_glossary,
            )
            log.info("3-B: %d correction(s) after filter", len(corrections))
            return self._apply_corrections(
                segments, corrections, line_to_seg_idx, original_by_line,
            )
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("3-B: JSON parse failed: %s", e)
            return [dict(seg) for seg in segments]

    # ------------------------------------------------------------------
    # Stage 3-C: Quality audit
    # ------------------------------------------------------------------

    def audit_quality(
        self,
        segments: list,
        candidates: Optional[CandidateResult] = None,
        min_text_length: int = 0,
        segment_indices: Optional[list] = None,
    ) -> Optional[AuditResult]:
        """Stage 3-C: Audit corrected text for remaining quality issues."""
        if "C" not in self._stages:
            return None

        numbered_text, _, _ = self._build_numbered_input(
            segments, min_text_length, segment_indices=segment_indices,
        )
        if not numbered_text:
            return AuditResult(items=[], passed=True)

        # Build user message
        parts = []
        if candidates and candidates.candidates:
            candidate_lines = []
            for c in candidates.candidates:
                candidate_lines.append(
                    f"- [行{c.line}] 「{c.text_span}」({c.category})"
                )
            parts.append("## 前段の用語候補\n" + "\n".join(candidate_lines))
        parts.append(numbered_text)
        user_msg = "\n\n".join(parts)

        raw = self._call_api(
            AUDIT_SYSTEM_PROMPT, user_msg, AUDIT_SCHEMA, self._max_tokens_c,
        )
        if raw is None:
            log.warning("3-C: API call failed, skipping audit")
            return None

        try:
            data = json.loads(raw)
            items = []
            for item in data.get("items", []):
                items.append(AuditItem(
                    line=item.get("line", 0),
                    text_span=item.get("text_span", ""),
                    issue_type=item.get("issue_type", ""),
                    severity=item.get("severity", "low"),
                    suggestion=item.get("suggestion", ""),
                ))
            passed = data.get("pass", True)
            log.info(
                "3-C: %d issue(s), pass=%s",
                len(items), passed,
            )
            return AuditResult(items=items, passed=passed, raw_response=raw)
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("3-C: JSON parse failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Conditional 3-C: should_run_stage_c
    # ------------------------------------------------------------------

    @staticmethod
    def should_run_stage_c(
        candidates: Optional[CandidateResult],
        corrected_segments: list,
        original_segments: list,
        min_text_length: int = 0,
    ) -> tuple:
        """Decide whether 3-C audit is warranted based on 3-A/3-B results.

        Returns (should_run: bool, reason: str).

        Triggers:
        - 3+ broken_katakana or misrecognized candidates from 3-A
        - 2+ low-confidence candidates from 3-A
        - 3-B correction density > 30% of eligible segments
        """
        reasons = []

        # Check 3-A candidates
        if candidates and candidates.candidates:
            broken_or_misrec = [
                c for c in candidates.candidates
                if c.category in ("broken_katakana", "misrecognized")
            ]
            if len(broken_or_misrec) >= 3:
                reasons.append(
                    f"broken/misrecognized={len(broken_or_misrec)}"
                )

            low_conf = [
                c for c in candidates.candidates
                if c.confidence == "low"
            ]
            if len(low_conf) >= 2:
                reasons.append(f"low_confidence={len(low_conf)}")

        # Check 3-B correction density
        eligible = 0
        changed = 0
        for orig, corr in zip(original_segments, corrected_segments):
            orig_text = orig.get("text", "")
            if len(orig_text) >= min_text_length:
                eligible += 1
                if orig_text != corr.get("text", ""):
                    changed += 1
        if eligible > 0 and changed / eligible > 0.3:
            reasons.append(
                f"correction_density={changed}/{eligible}={changed/eligible:.0%}"
            )

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    # ------------------------------------------------------------------
    # Orchestrator: process_segments
    # ------------------------------------------------------------------

    def process_segments(
        self,
        segments: list,
        known_terms: Optional[list] = None,
        session_glossary: Optional[list] = None,
        min_text_length: int = 0,
        auto_c: bool = False,
        segment_indices: Optional[list] = None,
    ) -> PipelineResult:
        """Run the configured pipeline stages on segments.

        Args:
            auto_c: If True, 3-C runs only when should_run_stage_c() triggers.
                    If False, 3-C runs whenever "C" is in stages (always).
            segment_indices: If provided, only process these segment indices
                           (dirty segments from screener). Others pass through unchanged.

        Returns a PipelineResult with corrected segments and metadata.
        """
        if not self._enabled:
            return PipelineResult(
                corrected_segments=[dict(seg) for seg in segments],
                segments_total=len(segments),
                segments_sent=0,
            )

        result = PipelineResult(
            corrected_segments=[dict(seg) for seg in segments],
            segments_total=len(segments),
        )
        api_calls = 0

        # Count segments that will be sent
        if segment_indices is not None:
            result.segments_sent = len(segment_indices)
        else:
            result.segments_sent = len(segments)

        # Stage 3-A: Candidate extraction
        candidates = None
        if "A" in self._stages:
            candidates = self.extract_candidates(
                segments, known_terms, session_glossary, min_text_length,
                segment_indices=segment_indices,
            )
            api_calls += 1
            result.candidates = candidates
            result.stages_run.append("A")

        # Stage 3-B: Notation correction
        if "B" in self._stages:
            corrected = self.correct_notation(
                segments, candidates, known_terms, session_glossary,
                min_text_length, segment_indices=segment_indices,
            )
            api_calls += 1
            result.corrected_segments = corrected
            result.stages_run.append("B")
        else:
            corrected = [dict(seg) for seg in segments]
            result.corrected_segments = corrected

        # Stage 3-C: Quality audit + optional retry
        run_c = "C" in self._stages
        if run_c and auto_c:
            should_run, reason = self.should_run_stage_c(
                candidates, corrected, segments, min_text_length,
            )
            if should_run:
                log.info("3-C auto-triggered: %s", reason)
            else:
                log.info("3-C skipped (auto_c): no trigger conditions met")
                run_c = False

        if run_c:
            audit = self.audit_quality(
                corrected, candidates, min_text_length,
                segment_indices=segment_indices,
            )
            api_calls += 1
            result.audit = audit
            result.stages_run.append("C")

            # Retry logic: re-run 3-B if high severity items exist
            if audit and "B" in self._stages:
                high_items = [i for i in audit.items if i.severity == "high"]
                if high_items:
                    # Log each high item before retry
                    for item in high_items:
                        log.info(
                            "3-C audit [HIGH] line=%d issue=%s span='%s' suggestion='%s'",
                            item.line, item.issue_type,
                            item.text_span, item.suggestion,
                        )
                    log.info(
                        "3-C audit triggered retry: %d high-severity item(s)",
                        len(high_items),
                    )

                    # Re-run 3-B with audit hints
                    corrected_retry = self.correct_notation(
                        corrected, candidates, known_terms, session_glossary,
                        min_text_length, audit_hints=high_items,
                        segment_indices=segment_indices,
                    )
                    api_calls += 1

                    # Re-run 3-C
                    audit_retry = self.audit_quality(
                        corrected_retry, candidates, min_text_length,
                        segment_indices=segment_indices,
                    )
                    api_calls += 1

                    # Log retry results
                    new_high = []
                    if audit_retry:
                        new_high = [i for i in audit_retry.items if i.severity == "high"]

                    before_keys = {
                        (i.line, i.issue_type, i.text_span) for i in high_items
                    }
                    after_keys = {
                        (i.line, i.issue_type, i.text_span) for i in new_high
                    }
                    resolved_keys = before_keys - after_keys
                    log.info(
                        "3-C retry result: %d/%d resolved, %d remaining",
                        len(resolved_keys), len(before_keys), len(after_keys),
                    )
                    for line_num, issue, span in sorted(resolved_keys):
                        log.info(
                            "3-C resolved: line=%d issue=%s span='%s'",
                            line_num, issue, span,
                        )
                    for item in new_high:
                        log.info(
                            "3-C still unresolved: line=%d issue=%s span='%s'",
                            item.line, item.issue_type, item.text_span,
                        )

                    result.corrected_segments = corrected_retry
                    result.audit = audit_retry
                    result.retried = True

        result.api_calls = api_calls
        return result

    # ------------------------------------------------------------------
    # Batch API hook (future use)
    # ------------------------------------------------------------------

    def prepare_batch_requests(
        self,
        segments: list,
        known_terms: Optional[list] = None,
        session_glossary: Optional[list] = None,
    ) -> list:
        """Prepare batch API request dicts for future Gemini Batch API."""
        # Placeholder for future batch API integration
        return []
