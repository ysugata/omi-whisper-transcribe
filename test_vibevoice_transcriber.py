"""Tests for vibevoice_transcriber module."""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from vibevoice_transcriber import _parse_output, _normalize_segments, _load_audio


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
        """Long Content is OK if End > 0 (real long segment, not corrupted)."""
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

        result_audio, result_sr = _load_audio(wav_path)

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

        result_audio, result_sr = _load_audio(wav_path)

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
