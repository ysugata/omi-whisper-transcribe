"""Tests for SuggestionLearner — auto-learn patterns from pending_suggestions."""

import json
import os
import tempfile

import pytest

from suggestion_learner import SuggestionLearner


def _write_suggestions(path, suggestions):
    """Write suggestions to a temp file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False)


class TestExtractPatterns:
    def test_basic_learning(self):
        """Pattern appearing 2+ times should be learned."""
        suggestions = [
            {"wrong": "ガーバナ", "correct": "Grafana", "status": "pending"},
            {"wrong": "ガーバナ", "correct": "Grafana", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {"ガーバナ": "Grafana"}

    def test_below_threshold(self):
        """Pattern appearing only once should NOT be learned."""
        suggestions = [
            {"wrong": "ガーバナ", "correct": "Grafana", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {}

    def test_short_wrong_skipped(self):
        """Short wrong values (<3 chars) should be skipped."""
        suggestions = [
            {"wrong": "ガ", "correct": "Grafana", "status": "pending"},
            {"wrong": "ガ", "correct": "Grafana", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {}

    def test_short_correct_skipped(self):
        """Short correct values (<3 chars) should be skipped."""
        suggestions = [
            {"wrong": "ガーバナ", "correct": "GF", "status": "pending"},
            {"wrong": "ガーバナ", "correct": "GF", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {}

    def test_non_katakana_wrong_skipped(self):
        """Wrong values with <50% katakana should be skipped."""
        suggestions = [
            {"wrong": "炎症部屋", "correct": "Ansible", "status": "pending"},
            {"wrong": "炎症部屋", "correct": "Ansible", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {}

    def test_comma_in_wrong_skipped(self):
        """Wrong values containing commas should be skipped."""
        suggestions = [
            {"wrong": "サイキョウ, サイキョー", "correct": "最強", "status": "pending"},
            {"wrong": "サイキョウ, サイキョー", "correct": "最強", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert patterns == {}

    def test_multiple_patterns(self):
        """Multiple different patterns should all be learned."""
        suggestions = [
            {"wrong": "ガーバナ", "correct": "Grafana", "status": "pending"},
            {"wrong": "ガーバナ", "correct": "Grafana", "status": "pending"},
            {"wrong": "プロリクエスト", "correct": "Pull Request", "status": "pending"},
            {"wrong": "プロリクエスト", "correct": "Pull Request", "status": "pending"},
        ]
        patterns = SuggestionLearner._extract_patterns(suggestions)
        assert len(patterns) == 2
        assert patterns["ガーバナ"] == "Grafana"
        assert patterns["プロリクエスト"] == "Pull Request"


class TestApplyLearned:
    def test_apply_single(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([
                {"wrong": "ガーバナ", "correct": "Grafana"},
                {"wrong": "ガーバナ", "correct": "Grafana"},
            ], f, ensure_ascii=False)
            path = f.name

        try:
            learner = SuggestionLearner(path)
            text, count = learner.apply_learned("ガーバナのダッシュボード")
            assert text == "Grafanaのダッシュボード"
            assert count == 1
        finally:
            os.unlink(path)

    def test_apply_no_match(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([
                {"wrong": "ガーバナ", "correct": "Grafana"},
                {"wrong": "ガーバナ", "correct": "Grafana"},
            ], f, ensure_ascii=False)
            path = f.name

        try:
            learner = SuggestionLearner(path)
            text, count = learner.apply_learned("テストテキスト")
            assert text == "テストテキスト"
            assert count == 0
        finally:
            os.unlink(path)

    def test_missing_file(self):
        learner = SuggestionLearner("/tmp/nonexistent_suggestions.json")
        assert learner.pattern_count == 0
        text, count = learner.apply_learned("テスト")
        assert text == "テスト"
        assert count == 0

    def test_auto_reload(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([], f)
            path = f.name

        try:
            learner = SuggestionLearner(path)
            assert learner.pattern_count == 0

            # Write new data
            _write_suggestions(path, [
                {"wrong": "ドッカー", "correct": "Docker"},
                {"wrong": "ドッカー", "correct": "Docker"},
            ])

            # Force mtime to be newer than what was recorded
            import time
            future_mtime = time.time() + 2
            os.utime(path, (future_mtime, future_mtime))

            text, count = learner.apply_learned("ドッカーのコンテナ")
            assert text == "Dockerのコンテナ"
            assert count == 1
        finally:
            os.unlink(path)

    def test_longer_pattern_first(self):
        """Longer patterns should be applied first."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([
                {"wrong": "ドッカー", "correct": "Docker"},
                {"wrong": "ドッカー", "correct": "Docker"},
                {"wrong": "ドッカーコンテナ", "correct": "Docker Container"},
                {"wrong": "ドッカーコンテナ", "correct": "Docker Container"},
            ], f, ensure_ascii=False)
            path = f.name

        try:
            learner = SuggestionLearner(path)
            text, count = learner.apply_learned("ドッカーコンテナを起動")
            assert text == "Docker Containerを起動"
            assert count == 1  # Only the longer pattern matched
        finally:
            os.unlink(path)
