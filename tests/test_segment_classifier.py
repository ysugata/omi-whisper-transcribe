"""Tests for segment_classifier — dirty/clean classification."""

import pytest

from segment_classifier import classify_segment, classify_segments, CONFIDENCE_LOGPROB_THRESHOLD


class TestClassifySegment:
    def test_already_corrected(self):
        """llm_corrected=True → clean."""
        seg = {"text": "ドッカーのコンテナを起動する", "llm_corrected": True}
        assert classify_segment(seg, set()) == "clean"

    def test_short_text(self):
        """Short text below threshold → clean."""
        seg = {"text": "テスト"}
        assert classify_segment(seg, set(), min_text_length=20) == "clean"

    def test_pure_japanese(self):
        """Pure Japanese text (low katakana, high kanji+hira) → clean."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう"}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_unknown_katakana_dirty(self):
        """Unknown katakana word not in dictionary → dirty."""
        seg = {"text": "ガーバナのダッシュボードを確認してください"}
        # "ガーバナ" is not in known_values
        known = {"ダッシュボード"}
        assert classify_segment(seg, known, min_text_length=5) == "dirty"

    def test_known_katakana_clean(self):
        """All katakana words are in dictionary → clean."""
        seg = {"text": "ダッシュボードのスターバックスメニュー確認してください"}
        known = {"ダッシュボード", "スターバックスメニュー"}
        assert classify_segment(seg, known, min_text_length=5) == "clean"

    def test_mixed_known_unknown(self):
        """Some unknown katakana → dirty."""
        seg = {"text": "ドッカーとクバネテスのテスト環境を構築してください"}
        known = {"テスト"}  # ドッカー and クバネテス not known
        assert classify_segment(seg, known, min_text_length=5) == "dirty"

    def test_no_katakana_words_clean(self):
        """No katakana words of 3+ chars → clean."""
        seg = {"text": "これは漢字と平仮名だけの文章です、問題ありません"}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_empty_text(self):
        """Empty text → clean (below min length)."""
        seg = {"text": ""}
        assert classify_segment(seg, set()) == "clean"

    def test_english_text_clean(self):
        """Primarily English text → clean (katakana ratio low)."""
        seg = {"text": "This is an English text with no katakana whatsoever right"}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_mostly_japanese_with_unknown_katakana_dirty(self):
        """Japanese text with unknown katakana → dirty (Rule 3 before Rule 4)."""
        # This text is mostly Japanese but contains unknown katakana "エドラソー"
        seg = {"text": "彼らはエドラソーの世界に行ったことがあります"}
        known = set()  # エドラソー is not known
        assert classify_segment(seg, known, min_text_length=5) == "dirty"

    def test_mostly_japanese_with_known_katakana_clean(self):
        """Japanese text with known katakana → clean."""
        seg = {"text": "彼らはレストランの予約をしたことがあります"}
        known = {"レストラン"}
        assert classify_segment(seg, known, min_text_length=5) == "clean"

    # --- Confidence-based tests (Rule 4: avg_logprob) ---

    def test_low_confidence_dirty(self):
        """Low avg_logprob segment → dirty regardless of katakana."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう", "avg_logprob": -0.8}
        # Pure Japanese text that would normally be clean, but low confidence
        assert classify_segment(seg, set(), min_text_length=5) == "dirty"

    def test_high_confidence_clean(self):
        """High avg_logprob segment → clean (confidence rule doesn't trigger)."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう", "avg_logprob": -0.2}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_medium_confidence_clean(self):
        """Medium avg_logprob (above threshold) → not flagged by confidence rule."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう", "avg_logprob": -0.4}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_no_logprob_backward_compatible(self):
        """Missing avg_logprob (old transcript) → confidence rule skipped."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう"}
        # Should be clean based on pure-Japanese rule, no confidence check
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_low_confidence_with_unknown_katakana(self):
        """Low confidence + unknown katakana → dirty (caught by katakana rule first)."""
        seg = {"text": "ガーバナのダッシュボードを確認してください", "avg_logprob": -0.8}
        known = {"ダッシュボード"}
        assert classify_segment(seg, known, min_text_length=5) == "dirty"

    def test_confidence_threshold_boundary(self):
        """At exactly the threshold → not dirty (threshold is exclusive)."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう",
               "avg_logprob": CONFIDENCE_LOGPROB_THRESHOLD}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"

    def test_confidence_just_below_threshold(self):
        """Just below threshold → dirty."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう",
               "avg_logprob": CONFIDENCE_LOGPROB_THRESHOLD - 0.01}
        assert classify_segment(seg, set(), min_text_length=5) == "dirty"

    def test_low_confidence_short_text_still_clean(self):
        """Low confidence but short text → clean (Rule 2 takes priority)."""
        seg = {"text": "短い", "avg_logprob": -0.9}
        assert classify_segment(seg, set(), min_text_length=20) == "clean"

    def test_low_confidence_already_corrected_clean(self):
        """Low confidence but already corrected → clean (Rule 1 takes priority)."""
        seg = {"text": "今日は天気がとても良いので散歩に行きましょう",
               "avg_logprob": -0.9, "llm_corrected": True}
        assert classify_segment(seg, set(), min_text_length=5) == "clean"


class TestClassifySegments:
    def test_mixed_segments(self):
        """Multiple segments with mixed results."""
        segments = [
            {"text": "今日は天気がいいですね、散歩に行きましょう"},
            {"text": "ガーバナのダッシュボードを確認してください"},
            {"text": "短い"},
            {"text": "ドッカーとクバネテスの環境を構築してください"},
        ]
        known = {"ダッシュボード"}
        dirty, clean = classify_segments(segments, known, min_text_length=5)

        # Seg 0: pure Japanese → clean
        assert 0 in clean
        # Seg 1: ガーバナ unknown → dirty
        assert 1 in dirty
        # Seg 2: too short → clean
        assert 2 in clean
        # Seg 3: ドッカー and クバネテス unknown → dirty
        assert 3 in dirty

    def test_all_clean(self):
        """All segments clean → no dirty indices."""
        segments = [
            {"text": "今日は天気がいいです、楽しい一日になりそう"},
            {"text": "明日も晴れるといいですね、洗濯物が乾きます"},
        ]
        dirty, clean = classify_segments(segments, set(), min_text_length=5)
        assert len(dirty) == 0
        assert len(clean) == 2

    def test_all_dirty(self):
        """All segments have unknown katakana → all dirty."""
        segments = [
            {"text": "ガーバナのダッシュボードを確認してください"},
            {"text": "プロリクエストを出してください、レビュー待ちです"},
        ]
        dirty, clean = classify_segments(segments, set(), min_text_length=5)
        assert len(dirty) == 2
        assert len(clean) == 0

    def test_empty_list(self):
        """Empty segment list."""
        dirty, clean = classify_segments([], set())
        assert dirty == []
        assert clean == []

    def test_confidence_mixed(self):
        """Segments with varying confidence levels."""
        segments = [
            {"text": "今日は天気がいいですね、散歩に行きましょう", "avg_logprob": -0.2},  # high conf → clean
            {"text": "今日は天気がいいですね、散歩に行きましょう", "avg_logprob": -0.8},  # low conf → dirty
            {"text": "今日は天気がいいですね、散歩に行きましょう"},  # no conf → clean (backward compat)
        ]
        dirty, clean = classify_segments(segments, set(), min_text_length=5)
        assert 0 in clean   # high confidence
        assert 1 in dirty   # low confidence
        assert 2 in clean   # no confidence data
