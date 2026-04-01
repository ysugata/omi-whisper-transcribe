"""Tests for segment_classifier — dirty/clean classification."""

import pytest

from segment_classifier import classify_segment, classify_segments


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
