"""Unit tests for Qwen3-ASR transcriber module.

Tests text filtering, segment splitting, and transcriber initialization.
GPU/model tests are skipped if qwen_asr is not installed.
"""

import pytest
from transcriber_qwen3asr import (
    Qwen3ASRTranscriber,
    _filter_text,
    _remove_hallucinations,
    _clean_trailing_fragments,
    _collapse_word_repeats,
    _normalize_punctuation,
    _has_repetitive_kana,
    _is_gibberish_kana,
    _split_long_segment,
)


class TestFilterText:
    def test_empty(self):
        assert _filter_text("") == ""
        assert _filter_text("  ") == ""

    def test_single_char_dropped(self):
        assert _filter_text("あ") == ""

    def test_replacement_char_dropped(self):
        assert _filter_text("テスト\ufffdです") == ""

    def test_normal_text_passes(self):
        assert _filter_text("今日は天気がいいですね") == "今日は天気がいいですね"

    def test_term_corrector_applied(self):
        class MockCorrector:
            def correct(self, text):
                return text.replace("クーバネティス", "Kubernetes")

        result = _filter_text("クーバネティスを使う", MockCorrector())
        assert result == "Kubernetesを使う"


class TestHallucinationRemoval:
    def test_youtube_hallucination(self):
        assert _remove_hallucinations("ご視聴ありがとうございました") == ""

    def test_subscribe_hallucination(self):
        assert _remove_hallucinations("Subscribe") == ""

    def test_music_notes(self):
        assert _remove_hallucinations("♪♪♪") == ""

    def test_repeated_phrase(self):
        assert _remove_hallucinations("はいはいはいはい") == ""

    def test_normal_text_preserved(self):
        text = "今日のミーティングについて話しましょう"
        assert _remove_hallucinations(text) == text

    def test_full_match_otsukaresama(self):
        assert _remove_hallucinations("お疲れ様でした") == ""
        assert _remove_hallucinations("お疲れ様でした。") == ""

    def test_partial_hallucination(self):
        result = _remove_hallucinations("今日の話は面白かった ♪♪")
        assert "今日の話は面白かった" in result
        assert "♪" not in result


class TestTrailingFragments:
    def test_trailing_fragments_removed(self):
        # Pattern: short fragments (1-6 chars) separated by commas at end
        result = _clean_trailing_fragments("全部で、あ、い、う、え")
        assert result == "全部で"

    def test_normal_text_preserved(self):
        text = "これは普通のテキストです"
        assert _clean_trailing_fragments(text) == text


class TestPunctuationNormalization:
    def test_period_to_maru(self):
        assert _normalize_punctuation("テスト.テスト") == "テスト。テスト"

    def test_trailing_comma_removed(self):
        assert _normalize_punctuation("テスト、") == "テスト"

    def test_consecutive_punct_collapsed(self):
        assert _normalize_punctuation("テスト。。。") == "テスト。"

    def test_decimal_preserved(self):
        result = _normalize_punctuation("3.14です")
        assert "3.14" in result


class TestCollapseWordRepeats:
    def test_basic_repeat(self):
        result = _collapse_word_repeats("カボ。カボ。カボ。カボ。")
        assert result.count("カボ") == 1

    def test_comma_repeat(self):
        result = _collapse_word_repeats("バナナ、バナナ、バナナ、バナナ")
        assert result.count("バナナ") == 1

    def test_mixed_repeat(self):
        result = _collapse_word_repeats("前の文。高松。高松。高松。後の文")
        assert result.count("高松") == 1
        assert "前の文" in result
        assert "後の文" in result

    def test_no_repeat_preserved(self):
        text = "これは普通のテキストです"
        assert _collapse_word_repeats(text) == text

    def test_short_repeat_not_collapsed(self):
        # Single char repeats should not be collapsed (handled by kana filter)
        text = "あ。い。う。え。"
        assert _collapse_word_repeats(text) == text


class TestRepetitiveKana:
    def test_repetitive_detected(self):
        assert _has_repetitive_kana("オォォォオォゥオォオゥ") is True

    def test_single_kana_repeat(self):
        assert _has_repetitive_kana("ンンンンンンン") is True

    def test_normal_text_not_flagged(self):
        assert _has_repetitive_kana("テレビテレビ") is False

    def test_short_text_not_flagged(self):
        assert _has_repetitive_kana("あいう") is False


class TestGibberishKana:
    def test_repetitive_is_gibberish(self):
        assert _is_gibberish_kana("オォォォオォゥオォオゥオォォ") is True

    def test_normal_japanese_not_gibberish(self):
        assert _is_gibberish_kana("今日は天気がいいですね") is False

    def test_short_text_not_gibberish(self):
        assert _is_gibberish_kana("カタカナ") is False

    def test_with_kanji_not_gibberish(self):
        assert _is_gibberish_kana("漢字を含むカタカナのテキストです") is False

    def test_grammatical_ending_not_gibberish(self):
        # Has grammatical ending → not gibberish even if kana-heavy
        assert _is_gibberish_kana("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもです") is False


class TestSegmentSplitting:
    def test_short_segment_not_split(self):
        seg = {"text": "短い文", "start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}
        result = _split_long_segment(seg, 15.0)
        assert len(result) == 1
        assert result[0]["text"] == "短い文"

    def test_long_segment_split_at_sentence(self):
        seg = {
            "text": "最初の文です。次の文です。最後の文です。",
            "start": 0.0,
            "end": 30.0,
            "speaker": "SPEAKER_00",
        }
        result = _split_long_segment(seg, 15.0)
        assert len(result) >= 2
        # All text preserved
        combined = "".join(s["text"] for s in result)
        assert "最初の文です" in combined
        assert "最後の文です" in combined

    def test_unsplittable_segment_returned_as_is(self):
        seg = {"text": "分割できない長い文", "start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}
        result = _split_long_segment(seg, 15.0)
        assert len(result) == 1


class TestTranscriberInit:
    def test_init_not_loaded(self):
        t = Qwen3ASRTranscriber()
        assert t.loaded is False

    def test_transcribe_without_load_raises(self):
        t = Qwen3ASRTranscriber()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            t.transcribe_audio(b"\x00" * 3200)

    def test_transcribe_short_audio_returns_empty(self):
        """Audio shorter than 0.1s should return empty list."""
        t = Qwen3ASRTranscriber()
        t._model = "dummy"  # Bypass loaded check
        # 100 samples = 0.00625s at 16kHz (too short)
        import numpy as np
        short_audio = np.zeros(100, dtype=np.int16).tobytes()
        result = t.transcribe_audio(short_audio)
        assert result == []


class TestModelManagerEngineSupport:
    """Test that ModelManager accepts engine parameter."""

    def test_whisper_engine_default(self):
        from model_manager import ModelManager
        mm = ModelManager(model_name="large-v3", compute_type="int8_float16")
        assert mm.engine == "whisper"

    def test_qwen3asr_engine(self):
        from model_manager import ModelManager
        mm = ModelManager(
            model_name="Qwen/Qwen3-ASR-1.7B",
            compute_type="bfloat16",
            engine="qwen3asr",
        )
        assert mm.engine == "qwen3asr"

    def test_status_includes_engine(self):
        from model_manager import ModelManager
        mm = ModelManager(
            model_name="Qwen/Qwen3-ASR-1.7B",
            compute_type="bfloat16",
            engine="qwen3asr",
        )
        status = mm.status()
        assert status["engine"] == "qwen3asr"
        assert status["model_name"] == "Qwen/Qwen3-ASR-1.7B"
