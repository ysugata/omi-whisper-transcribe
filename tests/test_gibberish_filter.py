"""Tests for gibberish kana filter and repetitive kana detection.

These functions are pure text processing, so we extract and test them
without importing the full bridge module (which needs CUDA/faster-whisper).
"""

import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# --- Inline copies of the filter functions for isolated testing ---
# These MUST stay in sync with omi_whisper_bridge.py

_PARTICLE_PATTERNS = re.compile(
    r'[ぁ-ん\u30a0-\u30ff](は|が|を|へ|から|まで|より|ばかり|さえ|すら|だけ|ほど|の|で|と|に|も|や)[ぁ-ん\u30a0-\u30ff]'
)
_GRAMMATICAL_ENDINGS = re.compile(
    r'(です|ます|ました|ません|でした|ている|ていた|ていない|ないで|なかった'
    r'|ことが|ものが|ように|ようだ|だろう|でしょう|ですか|ますか|ますよ|ですね'
    r'|ですよ|ましょう|ください|ながら|けれど|ったら|ったり|のです|んです'
    r'|よかった|じゃない|なさい|だよ|だね|だな|てる|てた|ちゃう|ちゃった'
    r'|って|かな|かも|でしょ|っけ|じゃん|んだ|のに|けど)'
)
_GIBBERISH_LOGPROB_THRESHOLD = -0.7


def _has_repetitive_kana(text: str) -> bool:
    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)
    kana_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', body)
    if len(kana_chars) < 6:
        return False
    unique_count = len(set(kana_chars))
    unique_ratio = unique_count / len(kana_chars)
    if unique_ratio < 0.30:
        return True
    counts = Counter(kana_chars)
    most_common_ratio = counts.most_common(1)[0][1] / len(kana_chars)
    return most_common_ratio >= 0.6


def _is_gibberish_kana(text: str, avg_logprob: float) -> bool:
    if _has_repetitive_kana(text):
        return True
    if avg_logprob >= -0.3:
        return False
    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)
    if len(body) <= 15:
        return False
    if re.search(r'[\u4e00-\u9fff]', body):
        return False
    katakana_count = len(re.findall(r'[\u30a0-\u30ff]', body))
    hiragana_count = len(re.findall(r'[\u3040-\u309f]', body))
    body_len = len(body)
    katakana_ratio = katakana_count / body_len if body_len else 0
    hiragana_ratio = hiragana_count / body_len if body_len else 0
    if katakana_ratio < 0.7 and hiragana_ratio < 0.7:
        return False
    if avg_logprob >= _GIBBERISH_LOGPROB_THRESHOLD:
        return False
    if _PARTICLE_PATTERNS.search(body):
        return False
    if _GRAMMATICAL_ENDINGS.search(body):
        return False
    return True


class TestHasRepetitiveKana:
    """_has_repetitive_kana: detect noise-generated kana repetition."""

    def test_pure_repetition(self):
        # "ンンンンンンン" — 1 unique kana out of 7
        assert _has_repetitive_kana("ンンンンンンン") is True

    def test_near_repetition(self):
        # "オォォォオォゥオォオゥ" — 3 unique kana, very low diversity
        assert _has_repetitive_kana("オォォォオォゥオォオゥ") is True

    def test_dominant_single_char(self):
        # "アアアアアイ" — 'ア' is 83% of 6 chars
        assert _has_repetitive_kana("アアアアアイ") is True

    def test_normal_katakana_word(self):
        # "プロメテウス" — 6 unique / 6 total = 1.0
        assert _has_repetitive_kana("プロメテウス") is False

    def test_normal_hiragana_sentence(self):
        # Normal conversational hiragana
        assert _has_repetitive_kana("それでは次のステップに進みましょう") is False

    def test_short_text_skipped(self):
        # Under 6 kana chars → always False
        assert _has_repetitive_kana("アアア") is False

    def test_laughter_like_pattern(self):
        # "ハハハハハハハハ" — 1 unique / 8 total
        assert _has_repetitive_kana("ハハハハハハハハ") is True

    def test_mixed_kana_real_speech(self):
        # "おはようございます" — diverse kana, real speech
        assert _has_repetitive_kana("おはようございます") is False

    def test_katakana_name(self):
        # "ドッカー" (Docker) — should not be filtered
        assert _has_repetitive_kana("ドッカー") is False

    def test_repeated_word_not_noise(self):
        # "テレビテレビテレビ" — 3 unique / 9 total = 0.33, legitimate repetition
        assert _has_repetitive_kana("テレビテレビテレビ") is False

    def test_infrastructure_katakana(self):
        assert _has_repetitive_kana("インフラストラクチャー") is False

    def test_filler_words(self):
        assert _has_repetitive_kana("えーっと、それでですね") is False


class TestIsGibberishKana:
    """_is_gibberish_kana: detect noise-generated gibberish text."""

    def test_repetitive_kana_detected_regardless_of_logprob(self):
        # Repetitive kana should be caught even with good logprob
        assert _is_gibberish_kana("ンンンンンンン", avg_logprob=-0.1) is True

    def test_high_confidence_real_text_passes(self):
        assert _is_gibberish_kana("カタカナテキストのテスト", avg_logprob=-0.1) is False

    def test_short_text_passes(self):
        # 15 chars or fewer always pass (unless repetitive)
        assert _is_gibberish_kana("カタカナテキスト", avg_logprob=-0.8) is False

    def test_kanji_text_passes(self):
        assert _is_gibberish_kana("漢字を含むカタカナテキストのサンプル", avg_logprob=-0.8) is False

    def test_grammatical_text_passes(self):
        # Contains grammatical endings → real Japanese
        assert _is_gibberish_kana("それはとてもすばらしいことですね", avg_logprob=-0.8) is False

    def test_pure_gibberish_kana(self):
        # Long low-confidence kana without grammar → gibberish
        text = "コシャナニコモロカシャナリコモロコシャ"
        assert _is_gibberish_kana(text, avg_logprob=-0.8) is True
