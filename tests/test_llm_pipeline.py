"""Tests for LLMPipeline — 3-stage ASR correction pipeline."""

import json
from unittest import mock

import pytest

from llm_pipeline import (
    AuditItem,
    AuditResult,
    CandidateResult,
    LLMPipeline,
    PipelineResult,
    TermCandidate,
)
from llm_pipeline_schema import (
    AUDIT_SCHEMA,
    CANDIDATE_SCHEMA,
    CORRECTION_SCHEMA,
    wrap_openai_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(responses: list, stages="ABC"):
    """Create an LLMPipeline with mocked API client that returns sequential responses."""
    pipeline = LLMPipeline.__new__(LLMPipeline)
    pipeline._enabled = True
    pipeline._model = "gemini-2.5-flash-lite"
    pipeline._temperature = 0
    pipeline._stages = stages
    pipeline._max_tokens_a = 2048
    pipeline._max_tokens_b = 4096
    pipeline._max_tokens_c = 2048
    pipeline._client = mock.MagicMock()

    side_effects = []
    for resp_content in responses:
        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = resp_content
        side_effects.append(mock_response)

    pipeline._client.chat.completions.create.side_effect = side_effects
    return pipeline


def _make_segments(texts: list) -> list:
    """Create segment dicts from text strings."""
    return [{"text": t} for t in texts]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_wrap_openai_schema(self):
        raw = {"type": "object", "properties": {}, "required": []}
        wrapped = wrap_openai_schema("test", raw)
        assert wrapped["type"] == "json_schema"
        assert wrapped["json_schema"]["name"] == "test"
        assert wrapped["json_schema"]["strict"] is True
        assert wrapped["json_schema"]["schema"] is raw

    def test_candidate_schema_structure(self):
        assert CANDIDATE_SCHEMA["json_schema"]["name"] == "term_candidates"
        schema = CANDIDATE_SCHEMA["json_schema"]["schema"]
        assert "candidates" in schema["properties"]

    def test_correction_schema_structure(self):
        assert CORRECTION_SCHEMA["json_schema"]["name"] == "notation_corrections"
        schema = CORRECTION_SCHEMA["json_schema"]["schema"]
        assert "corrections" in schema["properties"]

    def test_audit_schema_structure(self):
        assert AUDIT_SCHEMA["json_schema"]["name"] == "quality_audit"
        schema = AUDIT_SCHEMA["json_schema"]["schema"]
        assert "items" in schema["properties"]
        assert "pass" in schema["properties"]


# ---------------------------------------------------------------------------
# _build_numbered_input tests
# ---------------------------------------------------------------------------

class TestBuildNumberedInput:
    def test_basic(self):
        segs = _make_segments(["hello", "world"])
        text, orig, line_map = LLMPipeline._build_numbered_input(segs)
        assert text == "[1] hello\n[2] world"
        assert orig == {1: "hello", 2: "world"}
        assert line_map == {1: 0, 2: 1}

    def test_min_text_length(self):
        segs = _make_segments(["ab", "long text here"])
        text, orig, line_map = LLMPipeline._build_numbered_input(segs, min_text_length=5)
        assert text == "[1] long text here"
        assert orig == {1: "long text here"}
        assert line_map == {1: 1}

    def test_empty(self):
        text, orig, line_map = LLMPipeline._build_numbered_input([])
        assert text == ""
        assert orig == {}
        assert line_map == {}

    def test_segment_indices_filter(self):
        """Only include segments at specified indices."""
        segs = _make_segments(["aaa", "bbb", "ccc", "ddd"])
        text, orig, line_map = LLMPipeline._build_numbered_input(
            segs, segment_indices=[1, 3],
        )
        assert text == "[1] bbb\n[2] ddd"
        assert orig == {1: "bbb", 2: "ddd"}
        assert line_map == {1: 1, 2: 3}

    def test_segment_indices_empty(self):
        """Empty segment_indices → no output."""
        segs = _make_segments(["aaa", "bbb"])
        text, orig, line_map = LLMPipeline._build_numbered_input(
            segs, segment_indices=[],
        )
        assert text == ""

    def test_segment_indices_with_min_length(self):
        """segment_indices + min_text_length combo."""
        segs = _make_segments(["ab", "long text here", "cd", "another long one"])
        text, orig, line_map = LLMPipeline._build_numbered_input(
            segs, min_text_length=5, segment_indices=[0, 1, 3],
        )
        # index 0 is "ab" (too short), index 1 is long, index 3 is long
        assert text == "[1] long text here\n[2] another long one"
        assert line_map == {1: 1, 2: 3}


# ---------------------------------------------------------------------------
# _validate_response tests
# ---------------------------------------------------------------------------

class TestValidateResponse:
    def test_too_long_rejected(self):
        result = LLMPipeline._validate_response("短い", "とても長い文章" * 100)
        assert result == "短い"

    def test_too_short_rejected(self):
        result = LLMPipeline._validate_response("これはテストの文章です", "あ")
        assert result == "これはテストの文章です"

    def test_valid_correction_accepted(self):
        result = LLMPipeline._validate_response("ドッカーのコンテナ", "Dockerのコンテナ")
        assert result == "Dockerのコンテナ"

    def test_empty_rejected(self):
        result = LLMPipeline._validate_response("テスト", "")
        assert result == "テスト"

    def test_low_similarity_rejected(self):
        result = LLMPipeline._validate_response(
            "これは日本語のテストです", "completely different text here"
        )
        assert result == "これは日本語のテストです"


# ---------------------------------------------------------------------------
# _filter_corrections tests
# ---------------------------------------------------------------------------

class TestFilterCorrections:
    def test_remove_empty_snippet(self):
        corrections = [
            {"line": 1, "original_snippet": "", "corrected": "fixed",
             "reason": "test", "source": "context"},
        ]
        result = LLMPipeline._filter_corrections(corrections, {1: "original"})
        assert len(result) == 0

    def test_remove_no_change(self):
        corrections = [
            {"line": 1, "original_snippet": "テスト", "corrected": "テスト",
             "reason": "none", "source": "context"},
        ]
        result = LLMPipeline._filter_corrections(corrections, {1: "テスト"})
        assert len(result) == 0

    def test_remove_punct_only(self):
        corrections = [
            {"line": 1, "original_snippet": "本当に?", "corrected": "本当に？",
             "reason": "punct", "source": "context"},
        ]
        result = LLMPipeline._filter_corrections(corrections, {1: "本当に?"})
        assert len(result) == 0

    def test_keep_real_correction(self):
        corrections = [
            {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
             "reason": "tech term", "source": "glossary"},
        ]
        result = LLMPipeline._filter_corrections(
            corrections, {1: "ドッカーのコンテナ"},
        )
        assert len(result) == 1

    def test_snippet_not_in_original(self):
        corrections = [
            {"line": 1, "original_snippet": "存在しない", "corrected": "修正",
             "reason": "test", "source": "context"},
        ]
        result = LLMPipeline._filter_corrections(
            corrections, {1: "これはテストです"},
        )
        assert len(result) == 0

    def test_empty_corrected(self):
        corrections = [
            {"line": 1, "original_snippet": "テスト", "corrected": "",
             "reason": "test", "source": "context"},
        ]
        result = LLMPipeline._filter_corrections(corrections, {1: "テスト"})
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _apply_corrections tests
# ---------------------------------------------------------------------------

class TestApplyCorrections:
    def test_basic_apply(self):
        segs = _make_segments(["ドッカーのコンテナ", "終わり"])
        corrections = [
            {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
             "reason": "tech", "source": "glossary"},
        ]
        result = LLMPipeline._apply_corrections(
            segs, corrections,
            line_to_seg_idx={1: 0, 2: 1},
            original_by_line={1: "ドッカーのコンテナ", 2: "終わり"},
        )
        assert result[0]["text"] == "Dockerのコンテナ"
        assert result[1]["text"] == "終わり"

    def test_no_corrections(self):
        segs = _make_segments(["テスト"])
        result = LLMPipeline._apply_corrections(
            segs, [], {1: 0}, {1: "テスト"},
        )
        assert result[0]["text"] == "テスト"

    def test_invalid_line(self):
        segs = _make_segments(["テスト"])
        corrections = [
            {"line": 99, "original_snippet": "テスト", "corrected": "修正",
             "reason": "test", "source": "context"},
        ]
        result = LLMPipeline._apply_corrections(
            segs, corrections, {1: 0}, {1: "テスト"},
        )
        # Line 99 not in mapping, so no change
        assert result[0]["text"] == "テスト"


# ---------------------------------------------------------------------------
# Stage 3-A: extract_candidates tests
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_basic_extraction(self):
        response = json.dumps({
            "candidates": [
                {"line": 1, "text_span": "ドッカー", "category": "technical_term",
                 "suggested_form": "Docker", "confidence": "high"},
            ]
        })
        pipeline = _make_pipeline([response], stages="A")
        segs = _make_segments(["ドッカーのコンテナ"])
        result = pipeline.extract_candidates(segs, known_terms=["Docker"])
        assert result is not None
        assert len(result.candidates) == 1
        assert result.candidates[0].text_span == "ドッカー"
        assert result.candidates[0].suggested_form == "Docker"

    def test_empty_candidates(self):
        response = json.dumps({"candidates": []})
        pipeline = _make_pipeline([response], stages="A")
        segs = _make_segments(["正常なテキスト"])
        result = pipeline.extract_candidates(segs)
        assert result is not None
        assert len(result.candidates) == 0

    def test_api_error_returns_none(self):
        pipeline = _make_pipeline([], stages="A")
        pipeline._client.chat.completions.create.side_effect = Exception("API error")
        segs = _make_segments(["テスト"])
        result = pipeline.extract_candidates(segs)
        assert result is None

    def test_stage_disabled(self):
        pipeline = _make_pipeline([], stages="B")
        segs = _make_segments(["テスト"])
        result = pipeline.extract_candidates(segs)
        assert result is None


# ---------------------------------------------------------------------------
# Stage 3-B: correct_notation tests
# ---------------------------------------------------------------------------

class TestCorrectNotation:
    def test_basic_correction(self):
        response = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
                 "reason": "tech term", "source": "glossary"},
            ]
        })
        pipeline = _make_pipeline([response], stages="B")
        segs = _make_segments(["ドッカーのコンテナ"])
        result = pipeline.correct_notation(segs, known_terms=["Docker"])
        assert result[0]["text"] == "Dockerのコンテナ"

    def test_no_corrections(self):
        response = json.dumps({"corrections": []})
        pipeline = _make_pipeline([response], stages="B")
        segs = _make_segments(["正常なテキスト"])
        result = pipeline.correct_notation(segs)
        assert result[0]["text"] == "正常なテキスト"

    def test_with_candidates(self):
        candidates = CandidateResult(candidates=[
            TermCandidate(
                line=1, text_span="ガーバナ", category="broken_katakana",
                suggested_form="Grafana", confidence="high",
            ),
        ])
        response = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ガーバナ", "corrected": "Grafana",
                 "reason": "broken katakana", "source": "candidate_3a"},
            ]
        })
        pipeline = _make_pipeline([response], stages="B")
        segs = _make_segments(["ガーバナのダッシュボード"])
        result = pipeline.correct_notation(segs, candidates=candidates)
        assert result[0]["text"] == "Grafanaのダッシュボード"

    def test_with_audit_hints(self):
        """Retry with audit hints should include hints in the message."""
        response = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ガーバナ", "corrected": "Grafana",
                 "reason": "broken katakana", "source": "context"},
            ]
        })
        pipeline = _make_pipeline([response], stages="B")
        segs = _make_segments(["ガーバナのダッシュボードを確認してください"])
        hints = [AuditItem(
            line=1, text_span="ガーバナ", issue_type="broken_katakana",
            severity="high", suggestion="Grafana",
        )]
        result = pipeline.correct_notation(segs, audit_hints=hints)
        assert result[0]["text"] == "Grafanaのダッシュボードを確認してください"
        # Check that audit hints were included in the API call message
        call_args = pipeline._client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "品質監査の指摘" in user_msg
        assert "ガーバナ" in user_msg

    def test_api_error_returns_original(self):
        pipeline = _make_pipeline([], stages="B")
        pipeline._client.chat.completions.create.side_effect = Exception("API error")
        segs = _make_segments(["テスト文"])
        result = pipeline.correct_notation(segs)
        assert result[0]["text"] == "テスト文"

    def test_stage_disabled(self):
        pipeline = _make_pipeline([], stages="AC")
        segs = _make_segments(["テスト"])
        result = pipeline.correct_notation(segs)
        assert result[0]["text"] == "テスト"

    def test_min_text_length(self):
        response = json.dumps({"corrections": []})
        pipeline = _make_pipeline([response], stages="B")
        segs = _make_segments(["ab", "これは長いテキストです"])
        result = pipeline.correct_notation(segs, min_text_length=5)
        assert result[0]["text"] == "ab"  # short one untouched


# ---------------------------------------------------------------------------
# Stage 3-C: audit_quality tests
# ---------------------------------------------------------------------------

class TestAuditQuality:
    def test_pass_true(self):
        response = json.dumps({"items": [], "pass": True})
        pipeline = _make_pipeline([response], stages="C")
        segs = _make_segments(["正常なテキスト"])
        result = pipeline.audit_quality(segs)
        assert result is not None
        assert result.passed is True
        assert len(result.items) == 0

    def test_issues_found(self):
        response = json.dumps({
            "items": [
                {"line": 1, "text_span": "ガーバナ",
                 "issue_type": "broken_katakana", "severity": "high",
                 "suggestion": "Grafana"},
            ],
            "pass": False,
        })
        pipeline = _make_pipeline([response], stages="C")
        segs = _make_segments(["ガーバナのダッシュボード"])
        result = pipeline.audit_quality(segs)
        assert result is not None
        assert result.passed is False
        assert len(result.items) == 1
        assert result.items[0].severity == "high"
        assert result.items[0].text_span == "ガーバナ"

    def test_api_error_returns_none(self):
        pipeline = _make_pipeline([], stages="C")
        pipeline._client.chat.completions.create.side_effect = Exception("API error")
        segs = _make_segments(["テスト"])
        result = pipeline.audit_quality(segs)
        assert result is None

    def test_stage_disabled(self):
        pipeline = _make_pipeline([], stages="AB")
        segs = _make_segments(["テスト"])
        result = pipeline.audit_quality(segs)
        assert result is None


# ---------------------------------------------------------------------------
# process_segments integration tests
# ---------------------------------------------------------------------------

class TestProcessSegments:
    def test_full_pipeline_no_retry(self):
        """ABC pipeline with no high severity items → no retry."""
        resp_a = json.dumps({
            "candidates": [
                {"line": 1, "text_span": "ドッカー", "category": "technical_term",
                 "suggested_form": "Docker", "confidence": "high"},
            ]
        })
        resp_b = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
                 "reason": "tech term", "source": "glossary"},
            ]
        })
        resp_c = json.dumps({"items": [], "pass": True})

        pipeline = _make_pipeline([resp_a, resp_b, resp_c], stages="ABC")
        segs = _make_segments(["ドッカーのコンテナ"])
        result = pipeline.process_segments(segs, known_terms=["Docker"])

        assert result.corrected_segments[0]["text"] == "Dockerのコンテナ"
        assert result.stages_run == ["A", "B", "C"]
        assert result.api_calls == 3
        assert result.retried is False

    def test_full_pipeline_with_retry(self):
        """ABC pipeline with high severity → retry 3-B + 3-C."""
        resp_a = json.dumps({"candidates": []})
        resp_b1 = json.dumps({"corrections": []})  # first 3-B: no corrections
        resp_c1 = json.dumps({
            "items": [
                {"line": 1, "text_span": "ガーバナ",
                 "issue_type": "broken_katakana", "severity": "high",
                 "suggestion": "Grafana"},
            ],
            "pass": False,
        })
        # Retry 3-B with hints
        resp_b2 = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ガーバナ", "corrected": "Grafana",
                 "reason": "broken katakana", "source": "candidate_3a"},
            ]
        })
        # Retry 3-C
        resp_c2 = json.dumps({"items": [], "pass": True})

        pipeline = _make_pipeline(
            [resp_a, resp_b1, resp_c1, resp_b2, resp_c2], stages="ABC",
        )
        segs = _make_segments(["ガーバナのダッシュボード"])
        result = pipeline.process_segments(segs)

        assert result.corrected_segments[0]["text"] == "Grafanaのダッシュボード"
        assert result.retried is True
        assert result.api_calls == 5  # A + B + C + B(retry) + C(retry)

    def test_b_only_stage(self):
        """B-only mode — equivalent to legacy corrector."""
        resp_b = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
                 "reason": "tech term", "source": "context"},
            ]
        })
        pipeline = _make_pipeline([resp_b], stages="B")
        segs = _make_segments(["ドッカーのコンテナ"])
        result = pipeline.process_segments(segs)
        assert result.corrected_segments[0]["text"] == "Dockerのコンテナ"
        assert result.stages_run == ["B"]
        assert result.api_calls == 1
        assert result.retried is False

    def test_ab_stages(self):
        """AB mode — no audit."""
        resp_a = json.dumps({"candidates": []})
        resp_b = json.dumps({"corrections": []})
        pipeline = _make_pipeline([resp_a, resp_b], stages="AB")
        segs = _make_segments(["テスト"])
        result = pipeline.process_segments(segs)
        assert result.stages_run == ["A", "B"]
        assert result.api_calls == 2
        assert result.audit is None

    def test_disabled_returns_original(self):
        """Disabled pipeline returns segments unchanged."""
        pipeline = LLMPipeline.__new__(LLMPipeline)
        pipeline._enabled = False
        segs = _make_segments(["テスト"])
        result = pipeline.process_segments(segs)
        assert result.corrected_segments[0]["text"] == "テスト"
        assert result.api_calls == 0

    def test_segment_indices_only_dirty(self):
        """Only dirty segments (by index) should be sent to LLM."""
        resp_b = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
                 "reason": "tech term", "source": "context"},
            ]
        })
        pipeline = _make_pipeline([resp_b], stages="B")
        segs = _make_segments([
            "今日は天気がいいです",          # index 0: clean
            "ドッカーのコンテナを起動する",   # index 1: dirty
            "明日も晴れるでしょう",           # index 2: clean
        ])
        result = pipeline.process_segments(
            segs, segment_indices=[1],
        )
        # Only index 1 should be corrected
        assert result.corrected_segments[0]["text"] == "今日は天気がいいです"
        assert result.corrected_segments[1]["text"] == "Dockerのコンテナを起動する"
        assert result.corrected_segments[2]["text"] == "明日も晴れるでしょう"
        assert result.segments_sent == 1
        assert result.segments_total == 3

    def test_3a_failure_graceful_degradation(self):
        """3-A failure → 3-B runs without candidates."""
        resp_b = json.dumps({
            "corrections": [
                {"line": 1, "original_snippet": "ドッカー", "corrected": "Docker",
                 "reason": "tech term", "source": "context"},
            ]
        })
        resp_c = json.dumps({"items": [], "pass": True})

        pipeline = _make_pipeline([None, resp_b, resp_c], stages="ABC")
        # Make 3-A fail
        call_count = [0]
        original_create = pipeline._client.chat.completions.create

        def side_effect_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("3-A failed")
            # Return from the pre-configured side_effects for subsequent calls
            mock_resp = mock.MagicMock()
            mock_resp.choices = [mock.MagicMock()]
            if call_count[0] == 2:
                mock_resp.choices[0].message.content = resp_b
            else:
                mock_resp.choices[0].message.content = resp_c
            return mock_resp

        pipeline._client.chat.completions.create.side_effect = side_effect_fn
        segs = _make_segments(["ドッカーのコンテナ"])
        result = pipeline.process_segments(segs, known_terms=["Docker"])
        # 3-B should still work
        assert result.corrected_segments[0]["text"] == "Dockerのコンテナ"
        assert result.candidates is None  # 3-A failed


# ---------------------------------------------------------------------------
# Session glossary in user message
# ---------------------------------------------------------------------------

class TestSessionGlossary:
    def test_glossary_in_message(self):
        resp_b = json.dumps({"corrections": []})
        pipeline = _make_pipeline([resp_b], stages="B")
        segs = _make_segments(["テストテキスト"])
        pipeline.correct_notation(
            segs,
            known_terms=["Docker"],
            session_glossary=["Kubernetes", "Grafana"],
        )
        call_args = pipeline._client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        assert "Kubernetes" in user_msg
        assert "Grafana" in user_msg
        assert "Docker" in user_msg

    def test_dedup_terms(self):
        """known_terms already in glossary should not duplicate."""
        resp_b = json.dumps({"corrections": []})
        pipeline = _make_pipeline([resp_b], stages="B")
        segs = _make_segments(["テスト"])
        pipeline.correct_notation(
            segs,
            known_terms=["Docker", "Grafana"],
            session_glossary=["Docker", "Kubernetes"],
        )
        call_args = pipeline._client.chat.completions.create.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        # Docker should appear once
        assert user_msg.count("Docker") == 1


# ---------------------------------------------------------------------------
# should_run_stage_c tests
# ---------------------------------------------------------------------------

class TestShouldRunStageC:
    def test_no_trigger_empty(self):
        """No candidates, no changes → no trigger."""
        segs = _make_segments(["テストテキスト二十文字以上必要です"])
        should, reason = LLMPipeline.should_run_stage_c(
            None, segs, segs, min_text_length=20,
        )
        assert should is False
        assert reason == ""

    def test_trigger_broken_katakana(self):
        """3+ broken_katakana candidates → trigger."""
        candidates = CandidateResult(candidates=[
            TermCandidate(1, "ガーバナ", "broken_katakana", "Grafana", "high"),
            TermCandidate(1, "ドッカー", "broken_katakana", "Docker", "high"),
            TermCandidate(1, "クバネテス", "broken_katakana", "Kubernetes", "medium"),
        ])
        segs = _make_segments(["テスト"])
        should, reason = LLMPipeline.should_run_stage_c(candidates, segs, segs)
        assert should is True
        assert "broken/misrecognized=3" in reason

    def test_trigger_low_confidence(self):
        """2+ low confidence candidates → trigger."""
        candidates = CandidateResult(candidates=[
            TermCandidate(1, "フォーデックス", "proper_noun", "", "low"),
            TermCandidate(2, "ナノクロー", "misrecognized", "", "low"),
        ])
        segs = _make_segments(["テスト"])
        should, reason = LLMPipeline.should_run_stage_c(candidates, segs, segs)
        assert should is True
        assert "low_confidence=2" in reason

    def test_trigger_high_correction_density(self):
        """Correction density > 30% → trigger."""
        orig = _make_segments([
            "セグメント一のテキスト二十文字以上必要です",
            "セグメント二のテキスト二十文字以上必要です",
            "セグメント三のテキスト二十文字以上必要です",
        ])
        corrected = [
            {"text": "セグメント1のテキスト二十文字以上必要です"},  # changed
            {"text": "セグメント二のテキスト二十文字以上必要です"},  # same
            {"text": "セグメント三のテキスト二十文字以上必要です"},  # same
        ]
        should, reason = LLMPipeline.should_run_stage_c(
            None, corrected, orig, min_text_length=10,
        )
        # 1/3 = 33% > 30%
        assert should is True
        assert "correction_density" in reason

    def test_no_trigger_low_density(self):
        """1/10 correction density → no trigger."""
        orig = _make_segments([f"セグメント{i}のテキスト二十文字以上必要です" for i in range(10)])
        corrected = [dict(s) for s in orig]
        corrected[0]["text"] = "変更済みセグメントのテキスト二十文字以上必要です"
        should, _ = LLMPipeline.should_run_stage_c(
            None, corrected, orig, min_text_length=10,
        )
        assert should is False


class TestAutoC:
    def test_auto_c_skips_when_no_trigger(self):
        """auto_c=True with AB stages: 3-C skipped when no trigger."""
        resp_a = json.dumps({"candidates": []})
        resp_b = json.dumps({"corrections": []})
        # No 3-C response needed since it should be skipped
        pipeline = _make_pipeline([resp_a, resp_b], stages="ABC")
        segs = _make_segments(["テストテキスト"])
        result = pipeline.process_segments(segs, auto_c=True)
        assert "C" not in result.stages_run
        assert result.api_calls == 2

    def test_auto_c_runs_when_triggered(self):
        """auto_c=True: 3-C runs when broken_katakana count triggers."""
        candidates_resp = json.dumps({
            "candidates": [
                {"line": 1, "text_span": "ガーバナ", "category": "broken_katakana",
                 "suggested_form": "Grafana", "confidence": "high"},
                {"line": 1, "text_span": "ドッカー", "category": "broken_katakana",
                 "suggested_form": "Docker", "confidence": "high"},
                {"line": 1, "text_span": "クバネテス", "category": "broken_katakana",
                 "suggested_form": "Kubernetes", "confidence": "medium"},
            ]
        })
        resp_b = json.dumps({"corrections": []})
        resp_c = json.dumps({"items": [], "pass": True})
        pipeline = _make_pipeline([candidates_resp, resp_b, resp_c], stages="ABC")
        segs = _make_segments(["ガーバナとドッカーとクバネテスのテスト"])
        result = pipeline.process_segments(segs, auto_c=True)
        assert "C" in result.stages_run
        assert result.api_calls == 3
