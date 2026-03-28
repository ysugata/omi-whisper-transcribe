"""
文字起こし精度改善 + 話者分離修正のユニットテスト

GPU / コンテナ不要でロジックを検証する。
faster_whisper 等の重い依存はモジュールレベルでモックする。
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# ============================================================
# モジュールレベルの依存モック (faster_whisper 等)
# ============================================================

_MOCK_MODULES = {
    "faster_whisper": MagicMock(),
    "faster_whisper.vad": MagicMock(),
    "ctranslate2": MagicMock(),
    "websockets": MagicMock(),
    "websockets.exceptions": MagicMock(),
    "aiohttp": MagicMock(),
    "soundfile": MagicMock(),
    "torch": MagicMock(),
    "whisperx": MagicMock(),
    "whisperx.diarize": MagicMock(),
}

# omi_whisper_bridge のインポート前にモックを差し込む
for mod_name, mock_mod in _MOCK_MODULES.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock_mod

# omi_whisper_bridge がキャッシュされないようにする
for mod in list(sys.modules.keys()):
    if mod in ("omi_whisper_bridge", "model_manager", "prompt_builder",
               "term_corrector", "llm_corrector", "diarizer",
               "audio_preprocessor"):
        del sys.modules[mod]

# 依存モジュールの簡易モック
sys.modules["model_manager"] = MagicMock()

# 今度はインポートできる
import omi_whisper_bridge  # noqa: E402
from omi_whisper_bridge import (
    _remove_hallucinations,
    _is_prompt_echo,
    _is_gibberish_kana,
    _clean_trailing_fragments,
    transcribe_audio,
    _find_undiarized_sessions,
    _batch_diarize_task,
    _batch_state,
)


# ============================================================
# テスト 1: ハルシネーション除去
# ============================================================

class TestHallucinationRemoval(unittest.TestCase):

    def test_youtube_outro_removed(self):
        self.assertEqual(_remove_hallucinations("ご視聴ありがとうございました"), "")

    def test_youtube_outro_present_tense(self):
        self.assertEqual(_remove_hallucinations("ご視聴ありがとうございます"), "")

    def test_subscribe_removed(self):
        self.assertEqual(_remove_hallucinations("チャンネル登録お願いします"), "")

    def test_music_symbols_removed(self):
        self.assertEqual(_remove_hallucinations("♪♪♪"), "")

    def test_repetition_removed(self):
        # 同一フレーズ 4 回以上
        self.assertEqual(_remove_hallucinations("あいうあいうあいうあいう"), "")

    def test_normal_text_preserved(self):
        text = "今日のミーティングではAPIの設計について議論しました"
        self.assertEqual(_remove_hallucinations(text), text)

    def test_mixed_hallucination_and_real(self):
        result = _remove_hallucinations("テスト文です。ご視聴ありがとうございました")
        self.assertIn("テスト文です", result)
        self.assertNotIn("ご視聴", result)

    def test_otsukare_full_match_removed(self):
        """Full-match: segment is only "お疲れ様でした" → removed"""
        self.assertEqual(_remove_hallucinations("お疲れ様でした"), "")
        self.assertEqual(_remove_hallucinations("おつかれさまでした"), "")
        self.assertEqual(_remove_hallucinations("お疲れ様でした。"), "")

    def test_oyasumi_full_match_removed(self):
        """Full-match: segment is only "おやすみなさい" → removed"""
        self.assertEqual(_remove_hallucinations("おやすみなさい"), "")
        self.assertEqual(_remove_hallucinations("おやすみなさい。"), "")

    def test_otsukare_within_sentence_preserved(self):
        """お疲れ様でした inside a longer sentence → preserved"""
        text = "今日もお疲れ様でしたね"
        self.assertEqual(_remove_hallucinations(text), text)

    def test_dewa_mata_within_sentence_preserved(self):
        """ではまた inside a longer sentence → preserved"""
        text = "ではまた来週お会いしましょう"
        self.assertEqual(_remove_hallucinations(text), text)

    def test_casual_comma_speech_preserved(self):
        """Casual comma-separated speech like "やば、あれ、もう" → preserved"""
        text = "やば、あれ、もう、まじで"
        result = _remove_hallucinations(text)
        self.assertEqual(result, text)


# ============================================================
# テスト 1b: ギブリッシュ仮名フィルタ
# ============================================================

class TestGibberishKana(unittest.TestCase):

    def test_casual_kana_not_gibberish(self):
        """Casual speech like "くればよかったよ" (14 chars, logprob=-0.6) → NOT gibberish"""
        # 15文字以下なので gibberish 判定されない
        self.assertFalse(_is_gibberish_kana("くればよかったよ", -0.6))

    def test_casual_kana_with_grammar_not_gibberish(self):
        """Casual speech with grammatical endings → NOT gibberish"""
        # 16文字以上だが文法パターンを含む
        self.assertFalse(_is_gibberish_kana("あれってさもうちょっとよかったかも", -0.6))

    def test_casual_particle_not_gibberish(self):
        """Text with particles の/で/と/に/も/や → NOT gibberish"""
        self.assertFalse(_is_gibberish_kana("これもあれもそれもぜんぶいらないって", -0.6))

    def test_real_gibberish_still_caught(self):
        """20-char random kana with logprob=-0.8 → IS gibberish"""
        # ランダムなカタカナ20文字 (文法パターンなし)
        gibberish = "ヌゼボピグムフリワケヨタチスハマクニソ"
        self.assertTrue(_is_gibberish_kana(gibberish, -0.8))

    def test_high_confidence_not_gibberish(self):
        """High confidence (logprob >= -0.3) is never gibberish"""
        self.assertFalse(_is_gibberish_kana("アアアアアアアアアアアアアアアア", -0.2))

    def test_kanji_text_not_gibberish(self):
        """Text with kanji → NOT gibberish"""
        self.assertFalse(_is_gibberish_kana("今日はとてもいい天気ですね、散歩しました", -0.8))

    def test_short_text_not_gibberish(self):
        """15 chars or less → NOT gibberish (regardless of other factors)"""
        self.assertFalse(_is_gibberish_kana("アイウエオカキクケ", -0.8))  # 9 chars


# ============================================================
# テスト 1c: 末尾カンマ断片クリーナー
# ============================================================

class TestTrailingFragments(unittest.TestCase):

    def test_trailing_fragments_cleaned(self):
        """Trailing comma-separated fragments → cleaned"""
        text = "今日は天気がいい、あれ、それ"
        result = _clean_trailing_fragments(text)
        self.assertEqual(result, "今日は天気がいい")

    def test_normal_text_preserved(self):
        """Normal text without trailing fragments → preserved"""
        text = "今日のミーティングではAPIの設計を議論しました"
        self.assertEqual(_clean_trailing_fragments(text), text)

    def test_all_fragments_preserves_original(self):
        """If cleaning removes everything → preserve original"""
        text = "、あれ、それ"
        result = _clean_trailing_fragments(text)
        self.assertEqual(result, text)


# ============================================================
# テスト 2: プロンプトエコー検出
# ============================================================

class TestPromptEcho(unittest.TestCase):

    def test_echo_detected(self):
        prompt = "Docker、Kubernetes、API設計の会議"
        text = "Docker、Kubernetes、API設計"
        self.assertTrue(_is_prompt_echo(text, prompt))

    def test_normal_text_not_echo(self):
        prompt = "Docker、Kubernetes"
        text = "今日は天気がいいですね"
        self.assertFalse(_is_prompt_echo(text, prompt))

    def test_empty_prompt_no_echo(self):
        self.assertFalse(_is_prompt_echo("テスト", ""))


# ============================================================
# テスト 3: transcribe_audio フィルタリング
# ============================================================

@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float


@dataclass
class FakeInfo:
    language: str = "ja"
    language_probability: float = 0.99


def _run_transcribe(segments_data: list[dict], chunk_duration: float = 10.0):
    """transcribe_audio を指定した FakeSegment リストで呼び出す"""
    import numpy as np
    num_samples = int(chunk_duration * 16000)
    dummy_pcm = np.zeros(num_samples, dtype=np.int16).tobytes()

    fake_segs = [
        FakeSegment(
            text=s["text"],
            start=s.get("start", 0.0),
            end=s.get("end", chunk_duration),
            avg_logprob=s.get("avg_logprob", -0.3),
            no_speech_prob=s.get("no_speech_prob", 0.1),
        )
        for s in segments_data
    ]

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segs), FakeInfo())

    with patch.object(omi_whisper_bridge, "audio_preprocessor", None), \
         patch.object(omi_whisper_bridge, "llm_corrector", None):
        return transcribe_audio(dummy_pcm, fake_model, session_offset=0.0)


class TestTranscribeFilters(unittest.TestCase):

    # --- 信頼度フィルタ ---

    def test_high_no_speech_low_logprob_filtered(self):
        """no_speech_prob > 0.8 AND avg_logprob < -0.3 → 除去"""
        result = _run_transcribe([
            {"text": "何かのテスト", "no_speech_prob": 0.9, "avg_logprob": -0.5},
        ])
        self.assertEqual(len(result), 0)

    def test_high_no_speech_high_logprob_kept(self):
        """no_speech_prob > 0.8 でも avg_logprob >= -0.3 なら保持 (Whisper が自信あり)"""
        result = _run_transcribe([
            {"text": "GPUがあるパソコンで", "no_speech_prob": 0.85, "avg_logprob": -0.16},
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "GPUがあるパソコンで")

    def test_dual_filter_triggers(self):
        """avg_logprob < -0.6 AND no_speech_prob > 0.4 で除去"""
        result = _run_transcribe([
            {"text": "何かテスト", "avg_logprob": -0.8, "no_speech_prob": 0.55},
        ])
        self.assertEqual(len(result), 0)

    def test_low_logprob_only_passes(self):
        """avg_logprob が低くても no_speech_prob が低ければ残す"""
        result = _run_transcribe([
            {"text": "これはテストです", "avg_logprob": -0.8, "no_speech_prob": 0.2},
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "これはテストです")

    def test_normal_segment_passes(self):
        """通常セグメントは通過"""
        result = _run_transcribe([
            {"text": "今日は天気がいい", "avg_logprob": -0.3, "no_speech_prob": 0.1},
        ])
        self.assertEqual(len(result), 1)

    # --- 短文フィルタ ---

    def test_single_char_dropped(self):
        """1 文字は無条件除去"""
        result = _run_transcribe([
            {"text": "ん", "avg_logprob": -0.2, "no_speech_prob": 0.1},
        ])
        self.assertEqual(len(result), 0)

    def test_short_low_confidence_dropped(self):
        """3 文字以下かつ低信頼度は除去"""
        result = _run_transcribe([
            {"text": "あー", "avg_logprob": -0.6, "no_speech_prob": 0.1},
        ])
        self.assertEqual(len(result), 0)

    def test_short_high_confidence_kept(self):
        """3 文字以下でも高信頼度なら残す"""
        result = _run_transcribe([
            {"text": "はい", "avg_logprob": -0.2, "no_speech_prob": 0.1},
        ])
        self.assertEqual(len(result), 1)

    # --- タイムスタンプクランプ ---

    def test_timestamp_clamped_to_chunk(self):
        """end が chunk_duration にクランプされる"""
        result = _run_transcribe(
            [{"text": "テスト音声です", "start": 0.0, "end": 29.98}],
            chunk_duration=4.0,
        )
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0]["end"], 4.0)

    def test_negative_start_clamped(self):
        """負の start は 0.0 にクランプ"""
        result = _run_transcribe(
            [{"text": "テスト", "start": -1.0, "end": 2.0}],
            chunk_duration=5.0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 0.0)

    # --- ハルシネーション (transcribe 内) ---

    def test_hallucination_filtered_in_transcribe(self):
        """ハルシネーションが transcribe_audio 内で除去される"""
        result = _run_transcribe([
            {"text": "ご視聴ありがとうございました"},
        ])
        self.assertEqual(len(result), 0)

    # --- Whisper パラメータ ---

    def test_condition_on_previous_text_false(self):
        """condition_on_previous_text=False で呼ばれる"""
        import numpy as np
        dummy_pcm = np.zeros(16000, dtype=np.int16).tobytes()
        fake_model = MagicMock()
        fake_model.transcribe.return_value = (iter([]), FakeInfo())

        with patch.object(omi_whisper_bridge, "audio_preprocessor", None), \
             patch.object(omi_whisper_bridge, "llm_corrector", None):
            transcribe_audio(dummy_pcm, fake_model)

        kwargs = fake_model.transcribe.call_args[1]
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_vad_parameters(self):
        """VAD パラメータが更新された値で渡される"""
        import numpy as np
        dummy_pcm = np.zeros(16000, dtype=np.int16).tobytes()
        fake_model = MagicMock()
        fake_model.transcribe.return_value = (iter([]), FakeInfo())

        with patch.object(omi_whisper_bridge, "audio_preprocessor", None), \
             patch.object(omi_whisper_bridge, "llm_corrector", None):
            transcribe_audio(dummy_pcm, fake_model)

        kwargs = fake_model.transcribe.call_args[1]
        vad = kwargs["vad_parameters"]
        self.assertEqual(vad["min_speech_duration_ms"], 500)
        self.assertEqual(vad["threshold"], 0.5)


# ============================================================
# テスト 4: Diarizer 返り値
# ============================================================

class TestDiarizerReturnType(unittest.TestCase):

    def _make_diarizer(self, enabled=True):
        # diarizer モジュールを再インポート (モック済み環境)
        if "diarizer" in sys.modules:
            del sys.modules["diarizer"]
        from diarizer import Diarizer
        return Diarizer(
            hf_token="test-token" if enabled else "",
            device="cpu",
            language="ja",
        )

    def test_disabled_returns_false(self):
        d = self._make_diarizer(enabled=False)
        segs = [{"text": "test", "start": 0, "end": 1}]
        result, success = d.diarize("dummy.wav", segs)
        self.assertFalse(success)
        self.assertEqual(result, segs)

    def test_empty_segments_returns_false(self):
        d = self._make_diarizer(enabled=True)
        result, success = d.diarize("dummy.wav", [])
        self.assertFalse(success)

    def test_pipeline_exception_returns_false(self):
        d = self._make_diarizer(enabled=True)
        segs = [{"text": "test", "start": 0, "end": 1}]
        with patch.object(d, "_run_pipeline", side_effect=RuntimeError("GPU error")):
            result, success = d.diarize("dummy.wav", segs)
        self.assertFalse(success)
        self.assertEqual(result, segs)

    def test_success_returns_true(self):
        d = self._make_diarizer(enabled=True)
        segs = [{"text": "test", "start": 0, "end": 1}]
        updated = [{"text": "test", "start": 0, "end": 1, "speaker": "SPEAKER_01"}]
        with patch.object(d, "_run_pipeline", return_value=updated):
            result, success = d.diarize("dummy.wav", segs)
        self.assertTrue(success)
        self.assertEqual(result, updated)


# ============================================================
# テスト 5: タイムスタンプクランプ (diarizer._run_pipeline)
# ============================================================

class TestDiarizerTimestampClamping(unittest.TestCase):

    def test_segments_clamped_before_alignment(self):
        """音声長を超えるタイムスタンプが alignment 前にクランプされる"""
        import numpy as np

        if "diarizer" in sys.modules:
            del sys.modules["diarizer"]

        mock_whisperx = sys.modules["whisperx"]
        mock_whisperx.reset_mock()

        fake_audio = np.zeros(64000, dtype=np.float32)  # 4s at 16kHz
        mock_whisperx.load_audio.return_value = fake_audio
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [{"text": "テスト", "start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
        }
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [{"text": "テスト", "start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
        }

        mock_dp = MagicMock()
        sys.modules["whisperx.diarize"].DiarizationPipeline = mock_dp

        from diarizer import Diarizer
        d = Diarizer(hf_token="test", device="cpu", language="ja")

        segments = [{"text": "テスト", "start": 0.0, "end": 29.98}]  # end > 4s!
        d._run_pipeline("dummy.wav", segments, None, None)

        align_call = mock_whisperx.align.call_args
        aligned_segs = align_call[0][0]
        self.assertLessEqual(aligned_segs[0]["end"], 4.0)

    def test_empty_text_filtered(self):
        """空テキストセグメントが除去される"""
        import numpy as np

        if "diarizer" in sys.modules:
            del sys.modules["diarizer"]

        mock_whisperx = sys.modules["whisperx"]
        mock_whisperx.reset_mock()

        fake_audio = np.zeros(64000, dtype=np.float32)
        mock_whisperx.load_audio.return_value = fake_audio
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [{"text": "有効", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        }
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [{"text": "有効", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        }

        mock_dp = MagicMock()
        sys.modules["whisperx.diarize"].DiarizationPipeline = mock_dp

        from diarizer import Diarizer
        d = Diarizer(hf_token="test", device="cpu", language="ja")

        segments = [
            {"text": "", "start": 0.0, "end": 2.0},
            {"text": "有効", "start": 1.0, "end": 2.0},
        ]
        d._run_pipeline("dummy.wav", segments, None, None)

        align_call = mock_whisperx.align.call_args
        aligned_segs = align_call[0][0]
        self.assertEqual(len(aligned_segs), 1)
        self.assertEqual(aligned_segs[0]["text"], "有効")

    def test_short_audio_raises(self):
        """1 秒未満の音声は ValueError を送出"""
        import numpy as np

        if "diarizer" in sys.modules:
            del sys.modules["diarizer"]

        mock_whisperx = sys.modules["whisperx"]
        mock_whisperx.reset_mock()
        fake_audio = np.zeros(8000, dtype=np.float32)  # 0.5s
        mock_whisperx.load_audio.return_value = fake_audio

        from diarizer import Diarizer
        d = Diarizer(hf_token="test", device="cpu", language="ja")

        segments = [{"text": "test", "start": 0, "end": 0.5}]
        with self.assertRaises(ValueError):
            d._run_pipeline("dummy.wav", segments, None, None)


# ============================================================
# テスト 6: _merge_speakers / _find_speaker_by_time
# ============================================================

class TestMergeSpeakers(unittest.TestCase):

    def setUp(self):
        if "diarizer" in sys.modules:
            del sys.modules["diarizer"]
        from diarizer import Diarizer
        self.d = Diarizer(hf_token="test", device="cpu", language="ja")

    def test_1to1_mapping(self):
        original = [
            {"text": "A", "start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"text": "B", "start": 5, "end": 10, "speaker": "SPEAKER_00"},
        ]
        wx = [
            {"text": "A", "start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"text": "B", "start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]
        result = self.d._merge_speakers(original, wx)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertEqual(result[1]["speaker_id"], 1)

    def test_time_overlap_matching(self):
        original = [
            {"text": "A", "start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"text": "B", "start": 5, "end": 10, "speaker": "SPEAKER_00"},
        ]
        wx = [
            {"text": "A1", "start": 0, "end": 3, "speaker": "SPEAKER_00"},
            {"text": "A2", "start": 3, "end": 5, "speaker": "SPEAKER_00"},
            {"text": "B", "start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]
        result = self.d._merge_speakers(original, wx)
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")

    def test_find_speaker_by_time(self):
        wx_segments = [
            {"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 5, "end": 10, "speaker": "SPEAKER_01"},
        ]
        # 4-8 → SPEAKER_00 と 1s overlap, SPEAKER_01 と 3s overlap
        speaker = self.d._find_speaker_by_time(4, 8, wx_segments)
        self.assertEqual(speaker, "SPEAKER_01")

    def test_parse_speaker_id(self):
        self.assertEqual(self.d._parse_speaker_id("SPEAKER_00"), 0)
        self.assertEqual(self.d._parse_speaker_id("SPEAKER_03"), 3)


# ============================================================
# テスト 7: _find_undiarized_sessions (force)
# ============================================================

class TestFindUndiarizedSessions(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # diarized=True
        d1 = Path(self.tmpdir) / "2026-03-26" / "12-00-00"
        d1.mkdir(parents=True)
        (d1 / "audio.ogg").write_bytes(b"\x00")
        with open(d1 / "transcript.json", "w") as f:
            json.dump({
                "session": {"diarized": True, "audio_file": "audio.ogg"},
                "segments": [{"text": "test", "start": 0, "end": 1}],
            }, f)

        # diarized=False (未処理)
        d2 = Path(self.tmpdir) / "2026-03-26" / "13-00-00"
        d2.mkdir(parents=True)
        (d2 / "audio.ogg").write_bytes(b"\x00")
        with open(d2 / "transcript.json", "w") as f:
            json.dump({
                "session": {"audio_file": "audio.ogg"},
                "segments": [{"text": "test2", "start": 0, "end": 1}],
            }, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_default_skips_diarized(self):
        with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", self.tmpdir):
            sessions = _find_undiarized_sessions(force=False)
        self.assertEqual(len(sessions), 1)
        self.assertIn("13-00-00", sessions[0]["path"])

    def test_force_includes_diarized(self):
        with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", self.tmpdir):
            sessions = _find_undiarized_sessions(force=True)
        self.assertEqual(len(sessions), 2)


# ============================================================
# テスト 8: バッチ話者分離 (バックアップ + 成功/失敗)
# ============================================================

class TestBatchDiarize(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        d = Path(self.tmpdir) / "2026-03-26" / "14-00-00"
        d.mkdir(parents=True)
        (d / "audio.ogg").write_bytes(b"\x00")
        self.transcript_path = d / "transcript.json"
        with open(self.transcript_path, "w") as f:
            json.dump({
                "session": {"audio_file": "audio.ogg"},
                "segments": [{"text": "original", "start": 0, "end": 1}],
            }, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _make_mock_mm(self):
        mm = MagicMock()
        mm.gpu_lock = asyncio.Lock()

        async def _unload():
            return True
        mm.unload_model = _unload
        return mm

    def test_backup_created_on_success(self):
        """成功時に .bak が作成され、diarized=True になる"""
        updated_segs = [
            {"text": "original", "start": 0, "end": 1,
             "speaker": "SPEAKER_00", "speaker_id": 0},
        ]
        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = (updated_segs, True)

        with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", self.tmpdir), \
             patch.object(omi_whisper_bridge, "diarizer", mock_diarizer), \
             patch.object(omi_whisper_bridge, "model_manager", self._make_mock_mm()):
            asyncio.run(_batch_diarize_task(force=False))

        bak_path = str(self.transcript_path) + ".bak"
        self.assertTrue(os.path.exists(bak_path), ".bak should exist on success")

        with open(self.transcript_path) as f:
            data = json.load(f)
        self.assertTrue(data["session"]["diarized"])

    def test_no_backup_on_failure(self):
        """失敗時は .bak もトランスクリプト更新もされない"""
        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = (
            [{"text": "original", "start": 0, "end": 1}],
            False,  # 失敗
        )

        with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", self.tmpdir), \
             patch.object(omi_whisper_bridge, "diarizer", mock_diarizer), \
             patch.object(omi_whisper_bridge, "model_manager", self._make_mock_mm()):
            asyncio.run(_batch_diarize_task(force=False))

        bak_path = str(self.transcript_path) + ".bak"
        self.assertFalse(os.path.exists(bak_path), ".bak should NOT exist on failure")

        with open(self.transcript_path) as f:
            data = json.load(f)
        self.assertNotIn("diarized", data.get("session", {}))

        self.assertEqual(_batch_state["failed"], 1)

    def test_batch_no_sessions(self):
        """対象 0 件で正常終了"""
        with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", "/nonexistent"):
            asyncio.run(_batch_diarize_task(force=False))
        self.assertFalse(_batch_state["running"])
        self.assertEqual(_batch_state["total"], 0)


# ============================================================
# テスト 9: handle_batch_start (force + no_sessions レスポンス)
# ============================================================

class TestHandleBatchStart(unittest.TestCase):
    """管理 API エンドポイントの force/no_sessions 対応テスト"""

    def test_no_sessions_response(self):
        """全セッション diarized 済みで force=False → no_sessions"""
        tmpdir = tempfile.mkdtemp()
        try:
            d = Path(tmpdir) / "2026-03-26" / "10-00-00"
            d.mkdir(parents=True)
            (d / "audio.ogg").write_bytes(b"\x00")
            with open(d / "transcript.json", "w") as f:
                json.dump({
                    "session": {"diarized": True, "audio_file": "audio.ogg"},
                    "segments": [{"text": "done", "start": 0, "end": 1}],
                }, f)

            with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", tmpdir):
                sessions = _find_undiarized_sessions(force=False)
            self.assertEqual(len(sessions), 0)

            with patch.object(omi_whisper_bridge, "RECORDINGS_DIR", tmpdir):
                sessions_force = _find_undiarized_sessions(force=True)
            self.assertEqual(len(sessions_force), 1)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
