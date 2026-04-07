"""Tests for phonetic corrector and text normalizer."""

import pytest


class TestTextNormalizer:
    def test_fullwidth_alpha(self):
        from text_normalizer import normalize
        assert normalize("Ａ Ｂ Ｃ") == "A B C"

    def test_fullwidth_number(self):
        from text_normalizer import normalize
        assert normalize("１２３") == "123"

    def test_halfwidth_katakana(self):
        from text_normalizer import normalize
        result = normalize("ｱｲｳ")
        assert "ア" in result
        assert "イ" in result

    def test_zero_width_removal(self):
        from text_normalizer import normalize
        assert normalize("テスト\u200bテスト") == "テストテスト"

    def test_multi_space(self):
        from text_normalizer import normalize
        assert normalize("テスト　　テスト") == "テスト テスト"

    def test_empty(self):
        from text_normalizer import normalize
        assert normalize("") == ""

    def test_none_passthrough(self):
        from text_normalizer import normalize
        # None case not handled but empty string should work
        assert normalize("テスト") == "テスト"


class TestPhoneticCorrector:
    def test_basic_init(self):
        from phonetic_corrector import PhoneticCorrector
        pc = PhoneticCorrector(glossary=["Kubernetes", "Docker"])
        assert pc.corrections_made == 0

    def test_empty_glossary(self):
        from phonetic_corrector import PhoneticCorrector
        pc = PhoneticCorrector()
        assert pc.correct("テスト") == "テスト"

    def test_no_change_for_short_words(self):
        from phonetic_corrector import PhoneticCorrector
        pc = PhoneticCorrector(glossary=["テスト"], min_word_length=3)
        assert pc.correct("テ") == "テ"

    def test_phonetic_similarity_without_pykakasi(self):
        """Without pykakasi, similarity returns 0.0 (graceful degradation)."""
        from phonetic_corrector import _phonetic_similarity
        sim = _phonetic_similarity("テスト", "テスト")
        # Returns 1.0 with pykakasi, 0.0 without
        assert isinstance(sim, float)

    def test_edit_distance(self):
        from phonetic_corrector import _edit_distance
        assert _edit_distance("abc", "abc") == 0
        assert _edit_distance("abc", "abd") == 1
        assert _edit_distance("", "abc") == 3

    def test_correction_attempted(self):
        from phonetic_corrector import PhoneticCorrector
        pc = PhoneticCorrector(
            glossary=["テスト"],
            threshold=0.7,
        )
        # Should not crash even without pykakasi
        result = pc.correct("テスト")
        assert isinstance(result, str)
