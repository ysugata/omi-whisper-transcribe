"""Tests for SegmentScreener — orchestrates pre-LLM processing."""

import json
import os
import tempfile
from unittest import mock

import pytest

from segment_screener import SegmentScreener, ScreeningResult
from term_corrector import TermCorrector
from suggestion_learner import SuggestionLearner


def _make_segments(texts):
    return [{"text": t} for t in texts]


def _make_segments_with_logprob(items):
    """Create segments from (text, avg_logprob) tuples."""
    result = []
    for text, logprob in items:
        seg = {"text": text}
        if logprob is not None:
            seg["avg_logprob"] = logprob
        result.append(seg)
    return result


class TestScreenerBasic:
    def test_empty_segments(self):
        """Empty input → empty result."""
        screener = SegmentScreener()
        result = screener.screen([])
        assert result.segments == []
        assert result.dirty_indices == []
        assert result.clean_indices == []
        assert result.metrics.total_segments == 0

    def test_no_correctors(self):
        """Without correctors, just classify."""
        segments = _make_segments([
            "今日は天気がとても良いので散歩に行きましょう",
            "ガーバナのダッシュボードを確認してください",
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)

        assert result.metrics.total_segments == 2
        assert result.metrics.dict_corrections == 0
        # First is pure Japanese → clean, second has unknown katakana → dirty
        assert 0 in result.clean_indices
        assert 1 in result.dirty_indices


class TestScreenerWithTermCorrector:
    def test_dict_correction_applied(self):
        """Term corrector should correct before classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = os.path.join(tmpdir, "corrections.json")
            with open(dict_path, "w") as f:
                json.dump({"ガーバナ": "Grafana"}, f)

            tc = TermCorrector(dict_path)
            screener = SegmentScreener(term_corrector=tc, min_text_length=5)

            segments = _make_segments([
                "ガーバナのダッシュボードを確認してください",
            ])
            result = screener.screen(segments)

            # ガーバナ → Grafana (dict correction applied)
            assert "Grafana" in result.segments[0]["text"]
            assert result.metrics.dict_corrections == 1


class TestScreenerWithSuggestionLearner:
    def test_learned_correction_applied(self):
        """Suggestion learner should correct before classification."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([
                {"wrong": "プロリクエスト", "correct": "Pull Request"},
                {"wrong": "プロリクエスト", "correct": "Pull Request"},
            ], f, ensure_ascii=False)
            path = f.name

        try:
            learner = SuggestionLearner(path)
            screener = SegmentScreener(suggestion_learner=learner, min_text_length=5)

            segments = _make_segments([
                "プロリクエストを出してくださいね、テスト環境でお願いします",
            ])
            result = screener.screen(segments)

            assert "Pull Request" in result.segments[0]["text"]
            assert result.metrics.learned_corrections == 1
        finally:
            os.unlink(path)


class TestScreenerClassification:
    def test_mixed_segments(self):
        """Mix of clean and dirty segments."""
        segments = _make_segments([
            "今日は天気がとても良いので散歩に行きましょう",  # clean: pure Japanese
            "ドッカーのコンテナを起動してください",          # dirty: unknown katakana
            "短い",                                          # clean: too short
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)

        assert 0 in result.clean_indices
        assert 1 in result.dirty_indices
        assert 2 in result.clean_indices
        assert result.metrics.dirty_segments == 1
        assert result.metrics.screened_out == 2

    def test_all_clean(self):
        """All segments are clean → no dirty indices."""
        segments = _make_segments([
            "今日は天気がいいです、とても楽しい一日になりそう",
            "明日も晴れるといいですね、洗濯物を干しましょう",
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)

        assert len(result.dirty_indices) == 0
        assert len(result.clean_indices) == 2

    def test_does_not_mutate_originals(self):
        """Screening should not mutate original segments."""
        segments = _make_segments(["ガーバナのダッシュボード"])
        original_text = segments[0]["text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = os.path.join(tmpdir, "corrections.json")
            with open(dict_path, "w") as f:
                json.dump({"ガーバナ": "Grafana"}, f)
            tc = TermCorrector(dict_path)
            screener = SegmentScreener(term_corrector=tc, min_text_length=5)
            screener.screen(segments)

        assert segments[0]["text"] == original_text


class TestScreenerKnownValues:
    def test_known_values_include_corrections(self):
        """Known values should include term_corrector keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_path = os.path.join(tmpdir, "corrections.json")
            with open(dict_path, "w") as f:
                json.dump({"ドッカー": "Docker"}, f)

            tc = TermCorrector(dict_path)
            screener = SegmentScreener(term_corrector=tc, min_text_length=5)
            known = screener._build_known_values()

            # The wrong side (ドッカー) should be known
            assert "ドッカー" in known

    def test_known_values_include_hotwords(self):
        """Known values should include hotword custom words."""
        hotword_loader = mock.MagicMock()
        hotword_loader.get_custom_words.return_value = ["スターバックス", "コンビニエンス"]

        screener = SegmentScreener(hotword_loader=hotword_loader, min_text_length=5)
        known = screener._build_known_values()

        assert "スターバックス" in known
        assert "コンビニエンス" in known


class TestCommonLoanwords:
    def test_common_loanword_clean(self):
        """Segments with only common loanwords → clean (not dirty)."""
        segments = _make_segments([
            "レストランでランチを食べてからカフェでコーヒーを飲みました",
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        # All katakana words (レストラン, ランチ, カフェ, コーヒー) are common loanwords
        assert 0 in result.clean_indices

    def test_unknown_katakana_still_dirty(self):
        """Unknown katakana not in common loanwords → dirty."""
        segments = _make_segments([
            "ガーバナのダッシュボードでメトリクスを確認する",
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        # ガーバナ, ダッシュボード, メトリクス not in common loanwords
        assert 0 in result.dirty_indices

    def test_known_values_include_common_loanwords(self):
        """_build_known_values should include common loanwords."""
        screener = SegmentScreener(min_text_length=5)
        known = screener._build_known_values()
        assert "レストラン" in known
        assert "コンピュータ" in known
        assert "テスト" in known


class TestConfidenceScreening:
    """Tests for confidence-based (avg_logprob) screening."""

    def test_low_confidence_dirty(self):
        """Low avg_logprob → dirty even without katakana."""
        segments = _make_segments_with_logprob([
            ("今日は天気がとても良いので散歩に行きましょう", -0.8),
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        assert 0 in result.dirty_indices

    def test_high_confidence_clean(self):
        """High avg_logprob → clean (pure Japanese)."""
        segments = _make_segments_with_logprob([
            ("今日は天気がとても良いので散歩に行きましょう", -0.2),
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        assert 0 in result.clean_indices

    def test_missing_logprob_backward_compatible(self):
        """Missing avg_logprob → old behavior (clean for pure Japanese)."""
        segments = _make_segments_with_logprob([
            ("今日は天気がとても良いので散歩に行きましょう", None),
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        assert 0 in result.clean_indices

    def test_mixed_confidence(self):
        """Mix of segments with and without avg_logprob."""
        segments = _make_segments_with_logprob([
            ("今日は天気がとても良いので散歩に行きましょう", -0.2),  # high → clean
            ("昨日の会議で話した内容を整理しておいてください", -0.8),  # low → dirty
            ("来週のスケジュールを確認して連絡をください", None),     # no data → clean
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        assert 0 in result.clean_indices
        assert 1 in result.dirty_indices
        assert 2 in result.clean_indices

    def test_low_confidence_counts_in_metrics(self):
        """Low-confidence dirty segments are counted in metrics."""
        segments = _make_segments_with_logprob([
            ("今日は天気がとても良いので散歩に行きましょう", -0.8),
            ("明日も晴れるといいですね、洗濯物を干しましょう", -0.2),
        ])
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen(segments)
        assert result.metrics.dirty_segments == 1
        assert result.metrics.screened_out == 1


class TestMetrics:
    def test_metrics_format_log(self):
        """Metrics should format nicely."""
        screener = SegmentScreener(min_text_length=5)
        segments = _make_segments([
            "今日は天気がとても良いので散歩に行きましょう",
            "ガーバナのダッシュボードを確認してください",
        ])
        result = screener.screen(segments)
        log_str = result.metrics.format_log()
        assert "total=2" in log_str
        assert "dirty=" in log_str

    def test_metrics_to_dict(self):
        """Metrics should serialize to dict."""
        screener = SegmentScreener(min_text_length=5)
        result = screener.screen([])
        d = result.metrics.to_dict()
        assert "total_segments" in d
        assert "api_calls" in d
