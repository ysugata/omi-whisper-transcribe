"""Tests for vibevoice_transcriber module."""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest

from vibevoice_transcriber import (
    _parse_output, _normalize_segments, load_audio,
    _estimate_max_tokens, _build_gen_kwargs,
    _is_nontextual_tag, _split_vv_segment, VV_SEGMENT_MAX_DURATION,
)


class TestParseOutput:
    def test_valid_json_array(self):
        text = json.dumps([
            {"Start": 0, "End": 5.4, "Speaker": 0, "Content": "Hello world"},
            {"Start": 5.4, "End": 10.2, "Speaker": 1, "Content": "こんにちは"},
        ])
        result = _parse_output(text)
        assert len(result) == 2
        assert result[0]["Content"] == "Hello world"
        assert result[1]["Speaker"] == 1

    def test_json_with_surrounding_text(self):
        text = 'some prefix [{"Start": 0, "End": 3, "Speaker": 0, "Content": "test"}] suffix'
        result = _parse_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "test"

    def test_plain_text_returns_empty(self):
        """Plain text without JSON should return empty list (no corrupted segments)."""
        text = "This is just plain text"
        result = _parse_output(text)
        assert result == []

    def test_empty_string(self):
        result = _parse_output("")
        assert result == []

    def test_whitespace_only(self):
        result = _parse_output("   \n  ")
        assert result == []

    def test_malformed_json_returns_empty(self):
        """Malformed JSON should return empty list (no corrupted segments)."""
        text = '[{"Start": 0, "End": 5, "Speaker": 0, "Content": "broken'
        result = _parse_output(text)
        assert result == []

    def test_prompt_contaminated_output(self):
        """Input tokens mixed into output: prompt text before JSON array."""
        json_part = json.dumps([
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "Hello"}
        ])
        # Simulate prompt leaking into decoded text
        text = (
            "<|im_start|>system\nYou are a speech recognizer...<|im_end|>\n"
            "<|im_start|>user\nHotwords: テスト, 会議[audio_data]<|im_end|>\n"
            "<|im_start|>assistant\n" + json_part
        )
        result = _parse_output(text)
        assert len(result) == 1
        assert result[0]["Content"] == "Hello"

    def test_assistant_marker_extraction(self):
        """JSON extraction should work after assistant\\n marker."""
        json_part = json.dumps([
            {"Start": 0, "End": 10.0, "Speaker": 0, "Content": "テスト"},
            {"Start": 10.0, "End": 20.0, "Speaker": 1, "Content": "確認"},
        ])
        text = "Some preamble with [brackets] in it\nassistant\n" + json_part
        result = _parse_output(text)
        assert len(result) == 2
        assert result[0]["Content"] == "テスト"
        assert result[1]["Content"] == "確認"

    def test_prompt_with_brackets_no_valid_json(self):
        """Prompt containing brackets but no valid JSON array should return empty."""
        text = (
            "system\nYou are [a speech recognizer].\n"
            "user\nHotwords: [test]\nassistant\nNo output"
        )
        result = _parse_output(text)
        assert result == []


class TestNormalizeSegments:
    def test_basic_normalization(self):
        raw = [
            {"Start": 0, "End": 5.4, "Speaker": 0, "Content": "Hello"},
            {"Start": 5.4, "End": 10.2, "Speaker": 1, "Content": "World"},
        ]
        result = _normalize_segments(raw)
        assert len(result) == 2
        assert result[0] == {"start": 0.0, "end": 5.4, "speaker": 0, "text": "Hello"}
        assert result[1] == {"start": 5.4, "end": 10.2, "speaker": 1, "text": "World"}

    def test_missing_fields(self):
        raw = [{"Content": "only text"}]
        result = _normalize_segments(raw)
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.0
        assert result[0]["speaker"] == 0
        assert result[0]["text"] == "only text"

    def test_empty_list(self):
        result = _normalize_segments([])
        assert result == []

    def test_string_number_coercion(self):
        raw = [{"Start": "1.5", "End": "3.0", "Speaker": 0, "Content": "test"}]
        result = _normalize_segments(raw)
        assert result[0]["start"] == 1.5
        assert result[0]["end"] == 3.0

    def test_corrupted_segment_filtered(self):
        """Segments with End=0 and Content > 500 chars are filtered out."""
        long_text = "あ" * 600
        raw = [
            {"Start": 0, "End": 0, "Speaker": 0, "Content": long_text},
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "Normal segment"},
        ]
        result = _normalize_segments(raw)
        assert len(result) == 1
        assert result[0]["text"] == "Normal segment"

    def test_short_end_zero_segment_kept(self):
        """Short segments with End=0 should be kept (not corrupted)."""
        raw = [{"Start": 0, "End": 0, "Speaker": 0, "Content": "Short text"}]
        result = _normalize_segments(raw)
        assert len(result) == 1
        assert result[0]["text"] == "Short text"

    def test_long_content_with_valid_end_kept(self):
        """Long Content is OK if End > 0 (real long segment, not corrupted).

        Note: 120s segment has no sentence boundaries in 'あ'*600, so it
        cannot be split and is returned as-is.
        """
        long_text = "あ" * 600
        raw = [{"Start": 0, "End": 120.0, "Speaker": 0, "Content": long_text}]
        result = _normalize_segments(raw)
        assert len(result) == 1
        assert result[0]["text"] == long_text


class TestLoadAudioNormalization:
    def test_audio_normalized_to_minus16_dbfs(self, tmp_path):
        """Audio should be normalized to approximately -16 dBFS."""
        import soundfile as sf

        # Create a quiet test signal (sine wave at -40 dBFS)
        sr = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        quiet_amplitude = 10 ** (-40.0 / 20)  # -40 dBFS
        audio = (quiet_amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, sr)

        result_audio, result_sr = load_audio(wav_path)

        assert result_sr == 24000
        result_rms = np.sqrt(np.mean(result_audio ** 2))
        target_rms = 10 ** (-16.0 / 20)
        # Allow 1% tolerance
        assert abs(result_rms - target_rms) / target_rms < 0.01

    def test_silent_audio_not_normalized(self, tmp_path):
        """All-zero audio should not cause division by zero."""
        import soundfile as sf

        sr = 24000
        audio = np.zeros(sr, dtype=np.float32)

        wav_path = str(tmp_path / "silence.wav")
        sf.write(wav_path, audio, sr)

        result_audio, result_sr = load_audio(wav_path)

        assert result_sr == 24000
        assert np.all(result_audio == 0)


class TestTranscriberInferParsedDecode:
    """Test processor.decode(return_format='parsed') path in _infer."""

    def test_parsed_decode_returns_list_directly(self):
        """When processor.decode returns a list, _infer should use it directly."""
        torch = pytest.importorskip("torch")
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()

        # Mock model and processor
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_processor = MagicMock()

        transcriber.model = mock_model
        transcriber.processor = mock_processor

        # Simulate processor.apply_transcription_request
        mock_inputs = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_processor.apply_transcription_request.return_value = mock_inputs

        # Simulate model.generate output
        mock_output = torch.zeros(1, 20, dtype=torch.long)
        mock_model.generate.return_value = mock_output

        # processor.decode returns parsed list
        expected_parsed = [
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "テスト"},
        ]
        mock_processor.decode.return_value = expected_parsed

        audio_data = np.random.randn(24000).astype(np.float32)
        result = transcriber._infer(audio_data, 24000, "日本語の会話を書き起こしてください。")

        assert result == expected_parsed
        mock_processor.decode.assert_called_once()
        # batch_decode should NOT have been called
        mock_processor.batch_decode.assert_not_called()

    def test_parsed_decode_fallback_to_batch_decode(self):
        """When processor.decode raises, _infer should fall back to batch_decode."""
        torch = pytest.importorskip("torch")
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_processor = MagicMock()

        transcriber.model = mock_model
        transcriber.processor = mock_processor

        mock_inputs = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_processor.apply_transcription_request.return_value = mock_inputs
        mock_model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        # processor.decode raises (return_format not supported)
        mock_processor.decode.side_effect = TypeError("unexpected keyword")

        # batch_decode returns raw text
        json_text = json.dumps([
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "fallback"},
        ])
        mock_processor.batch_decode.return_value = [json_text]

        audio_data = np.random.randn(24000).astype(np.float32)
        result = transcriber._infer(audio_data, 24000, "")

        assert len(result) == 1
        assert result[0]["Content"] == "fallback"
        mock_processor.batch_decode.assert_called_once()


class TestLoadAudioPublicAPI:
    """Test that load_audio is a public module-level function."""

    def test_load_audio_importable(self):
        """load_audio should be importable from vibevoice_transcriber."""
        from vibevoice_transcriber import load_audio as la
        assert callable(la)

    def test_load_audio_reads_wav(self, tmp_path):
        """load_audio should read a WAV file and return (array, sr)."""
        import soundfile as sf

        sr = 24000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
        wav_path = str(tmp_path / "test.wav")
        sf.write(wav_path, audio, sr)

        result_audio, result_sr = load_audio(wav_path)
        assert result_sr == 24000
        assert len(result_audio) == sr
        assert result_audio.dtype == np.float32


class TestTranscribeAudio:
    """Test transcribe_audio() method (pre-loaded audio path)."""

    def test_short_audio_returns_empty(self):
        """Audio < 2.0s should return empty list without inference."""
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()
        transcriber.model = MagicMock()
        transcriber.processor = MagicMock()

        # 1 second of audio at 24kHz
        short_audio = np.zeros(24000, dtype=np.float32)
        result = transcriber.transcribe_audio(short_audio, 24000)
        assert result == []

    def test_not_loaded_raises(self):
        """Should raise RuntimeError when model is not loaded."""
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()
        audio = np.zeros(48000, dtype=np.float32)  # 2s
        with pytest.raises(RuntimeError, match="Model not loaded"):
            transcriber.transcribe_audio(audio, 24000)

    def test_calls_infer_directly(self):
        """transcribe_audio should call _infer without loading from file."""
        torch = pytest.importorskip("torch")
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()
        transcriber.model = MagicMock()
        transcriber.model.device = "cpu"
        transcriber.processor = MagicMock()

        # Mock _infer to return raw segments
        raw_segs = [{"Start": 0, "End": 5.0, "Speaker": 0, "Content": "テスト"}]
        transcriber._infer = MagicMock(return_value=raw_segs)

        audio = np.random.randn(72000).astype(np.float32)  # 3s at 24kHz
        result = transcriber.transcribe_audio(audio, 24000, "prompt")

        transcriber._infer.assert_called_once_with(audio, 24000, "prompt")
        assert len(result) == 1
        assert result[0]["text"] == "テスト"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0


class TestJapanesePromptConstruction:
    """Test Japanese prompt construction logic (mirrors bridge behavior)."""

    def test_prompt_with_hotwords(self):
        """Prompt should include Japanese prefix and hotwords."""
        hotwords_str = "テスト, 会議, プロジェクト"
        vv_prompt = "日本語の会話を書き起こしてください。"
        if hotwords_str:
            vv_prompt += f"\nHotwords: {hotwords_str}"

        assert vv_prompt.startswith("日本語の会話を書き起こしてください。")
        assert "Hotwords: テスト, 会議, プロジェクト" in vv_prompt

    def test_prompt_without_hotwords(self):
        """Prompt should be just the Japanese prefix when no hotwords."""
        hotwords_str = ""
        vv_prompt = "日本語の会話を書き起こしてください。"
        if hotwords_str:
            vv_prompt += f"\nHotwords: {hotwords_str}"

        assert vv_prompt == "日本語の会話を書き起こしてください。"
        assert "Hotwords" not in vv_prompt


class TestBuildGenKwargs:
    """Test _build_gen_kwargs helper."""

    def test_basic_kwargs(self):
        """Should include core generation parameters."""
        kwargs = _build_gen_kwargs(1000)
        assert kwargs["max_new_tokens"] == 1000
        assert kwargs["repetition_penalty"] == 1.05
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        assert kwargs["use_cache"] is True
        assert kwargs["acoustic_tokenizer_chunk_size"] == 1440000

    def test_custom_chunk_size(self):
        """Should accept custom acoustic_tokenizer_chunk_size."""
        kwargs = _build_gen_kwargs(1000, chunk_size=720000)
        assert kwargs["acoustic_tokenizer_chunk_size"] == 720000

    def test_generate_called_with_gen_kwargs(self):
        """model.generate() should receive gen_kwargs from _build_gen_kwargs."""
        torch = pytest.importorskip("torch")
        from vibevoice_transcriber import VibeVoiceTranscriber

        transcriber = VibeVoiceTranscriber()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_processor = MagicMock()

        transcriber.model = mock_model
        transcriber.processor = mock_processor

        mock_inputs = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_processor.apply_transcription_request.return_value = mock_inputs
        mock_model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        expected_parsed = [
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "テスト"},
        ]
        mock_processor.decode.return_value = expected_parsed

        audio_data = np.random.randn(72000).astype(np.float32)  # 3s
        transcriber._infer(audio_data, 24000, "")

        # Verify generate was called with the expected kwargs
        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs.get("max_new_tokens") is not None or \
               "max_new_tokens" in (call_kwargs[1] if len(call_kwargs) > 1 else {})
        # Check the merged kwargs contain our expected keys
        all_kwargs = {**call_kwargs.kwargs} if call_kwargs.kwargs else {}
        assert all_kwargs.get("repetition_penalty") == 1.05
        assert all_kwargs.get("use_cache") is True
        assert all_kwargs.get("acoustic_tokenizer_chunk_size") == 1440000


class TestEstimateMaxTokens:
    """Test _estimate_max_tokens helper."""

    def test_short_audio_uses_minimum(self):
        """Audio shorter than ~2.6s should use minimum (256 tokens)."""
        assert _estimate_max_tokens(2.0) == 256
        assert _estimate_max_tokens(2.5) == 256

    def test_normal_duration_scales_linearly(self):
        """Should be ~100 tokens per second of audio."""
        assert _estimate_max_tokens(10.0) == 1000
        assert _estimate_max_tokens(30.0) == 3000
        assert _estimate_max_tokens(60.0) == 6000

    def test_long_audio_capped_at_16384(self):
        """Audio over ~164s should cap at 16384 tokens."""
        assert _estimate_max_tokens(200.0) == 16384
        assert _estimate_max_tokens(600.0) == 16384

    def test_medium_duration_not_capped(self):
        """100s audio should scale linearly (10000 < 16384)."""
        assert _estimate_max_tokens(100.0) == 10000

    def test_zero_duration(self):
        """Edge case: zero duration should return minimum."""
        assert _estimate_max_tokens(0.0) == 256


class TestNontextualTagFilter:
    """Test non-textual event tag detection and filtering."""

    def test_official_tags_detected(self):
        """All official VibeVoice non-speech tags should be detected."""
        for tag in [
            "[Unintelligible Speech]", "[Silence]", "[Music]",
            "[Human Sounds]", "[Environmental Sounds]", "[Noise]",
            "[Hard Speech]",
        ]:
            assert _is_nontextual_tag(tag), f"{tag} should be nontextual"

    def test_padded_brackets_detected(self):
        """Tags with whitespace inside brackets should be detected."""
        assert _is_nontextual_tag("[ Silence ]")
        assert _is_nontextual_tag("  [Music]  ")

    def test_normal_text_not_flagged(self):
        """Normal transcription text should not be flagged."""
        assert not _is_nontextual_tag("こんにちは")
        assert not _is_nontextual_tag("Hello world")
        assert not _is_nontextual_tag("[Custom Tag]")
        assert not _is_nontextual_tag("")

    def test_mixed_text_not_flagged(self):
        """Text containing a tag but also other content is not a pure tag."""
        assert not _is_nontextual_tag("[Silence] and some text")
        assert not _is_nontextual_tag("before [Music]")

    def test_normalize_filters_nontextual(self):
        """_normalize_segments should drop non-textual tag segments."""
        raw = [
            {"Start": 0, "End": 5.0, "Speaker": 0, "Content": "[Unintelligible Speech]"},
            {"Start": 5.0, "End": 10.0, "Speaker": 0, "Content": "実際の発話"},
            {"Start": 10.0, "End": 15.0, "Speaker": 0, "Content": "[Silence]"},
            {"Start": 15.0, "End": 20.0, "Speaker": 1, "Content": "[Music]"},
            {"Start": 20.0, "End": 25.0, "Speaker": 1, "Content": "二人目の発話"},
        ]
        result = _normalize_segments(raw)
        assert len(result) == 2
        assert result[0]["text"] == "実際の発話"
        assert result[1]["text"] == "二人目の発話"


class TestSegmentSplitting:
    """Test long segment splitting at sentence boundaries."""

    def test_short_segment_not_split(self):
        """Segments within VV_SEGMENT_MAX_DURATION should not be split."""
        seg = {"start": 0.0, "end": 10.0, "speaker": 0, "text": "短いセグメント。"}
        result = _split_vv_segment(seg, VV_SEGMENT_MAX_DURATION)
        assert len(result) == 1
        assert result[0] is seg

    def test_split_at_sentence_boundary(self):
        """Long segment should be split at Japanese sentence boundaries."""
        seg = {
            "start": 0.0, "end": 30.0, "speaker": 0,
            "text": "最初の文です。次の文です。三番目の文です。",
        }
        result = _split_vv_segment(seg, VV_SEGMENT_MAX_DURATION)
        assert len(result) == 3
        assert result[0]["text"] == "最初の文です。"
        assert result[1]["text"] == "次の文です。"
        assert result[2]["text"] == "三番目の文です。"
        # Timestamps should be proportionally distributed
        assert result[0]["start"] == 0.0
        assert result[-1]["end"] == 30.0
        # All sub-segments should have the same speaker
        for r in result:
            assert r["speaker"] == 0

    def test_no_boundary_returns_original(self):
        """Segment with no sentence boundaries should not be split."""
        seg = {
            "start": 0.0, "end": 30.0, "speaker": 0,
            "text": "境界のないとても長いテキストがここにあります",
        }
        result = _split_vv_segment(seg, VV_SEGMENT_MAX_DURATION)
        assert len(result) == 1
        assert result[0] is seg

    def test_short_fragments_merged(self):
        """Fragments under 5 chars should merge into neighbours."""
        seg = {
            "start": 0.0, "end": 30.0, "speaker": 0,
            "text": "長い文です。え。もう一つの長い文です。",
        }
        result = _split_vv_segment(seg, VV_SEGMENT_MAX_DURATION)
        # "え。" is < 5 chars and should be merged
        texts = [r["text"] for r in result]
        assert not any(len(t) < 5 for t in texts)

    def test_normalize_splits_long_segments(self):
        """_normalize_segments should split segments exceeding VV_SEGMENT_MAX_DURATION."""
        raw = [
            {
                "Start": 0, "End": 60.0, "Speaker": 0,
                "Content": "最初の文です。二番目の文。三番目の文です。四番目。",
            },
            {
                "Start": 60.0, "End": 65.0, "Speaker": 1,
                "Content": "短い文。",
            },
        ]
        result = _normalize_segments(raw)
        # First segment should be split (60s >> 15s), second should remain
        assert len(result) > 2
        assert result[-1]["text"] == "短い文。"
        assert result[-1]["start"] == 60.0
        # All sub-segments from the first raw segment should have speaker 0
        for r in result[:-1]:
            assert r["speaker"] == 0
