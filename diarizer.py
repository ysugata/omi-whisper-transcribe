"""WhisperX ベースの話者分離モジュール

セッション終了時に完全な音声ファイルに対して話者分離を実行する。
RTX 3060 12GB の VRAM 制約のため、各ステップを順次実行し、
不要になったモデルは即座に解放する。

VRAM フロー:
  1. Whisper モデルをアンロード (呼び出し元で実施)
  2. wav2vec2 (alignment) → 処理 → 解放
  3. pyannote (diarization) → 処理 → 解放
  4. 結果を元セグメントにマージ

必要パッケージ:
  pip install torch whisperx

pyannote/speaker-diarization-3.1 は gated model のため HF トークン必須:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
"""

import gc
import logging
from typing import Optional

log = logging.getLogger("omi-whisper.diarizer")


class Diarizer:
    """WhisperX を使った話者分離。"""

    def __init__(
        self,
        hf_token: str,
        device: str = "cuda",
        language: str = "ja",
    ) -> None:
        self._hf_token = hf_token
        self._device = device
        self._language = language
        self._enabled = bool(hf_token)

        if self._enabled:
            log.info(f"Diarizer initialized: device={device}, language={language}")
        else:
            log.warning("Diarizer disabled: HF_AUTH_TOKEN not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _free_vram(self):
        """GPU メモリを解放する。"""
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def diarize(
        self,
        audio_path: str,
        segments: list[dict],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[dict]:
        """音声ファイルに対して話者分離を実行し、更新されたセグメントを返す。

        エラー時は元のセグメントをそのまま返す (テキストを失わない)。
        """
        if not self._enabled or not segments:
            return segments

        try:
            return self._run_pipeline(audio_path, segments, min_speakers, max_speakers)
        except Exception as e:
            log.error(f"Diarization failed (original segments preserved): {e}", exc_info=True)
            self._free_vram()
            return segments

    def _run_pipeline(
        self,
        audio_path: str,
        segments: list[dict],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> list[dict]:
        """話者分離パイプラインの実体。同期メソッド (executor から呼ぶ)。"""
        import torch
        import whisperx

        log.info(f"Starting diarization: {audio_path}")

        # 音声ロード
        audio = whisperx.load_audio(audio_path)
        log.info(f"Audio loaded: {len(audio) / 16000:.1f}s")

        # --- Step 1: Alignment (wav2vec2) ---
        log.info("Step 1/3: Loading alignment model (wav2vec2)...")
        model_a, metadata = whisperx.load_align_model(
            language_code=self._language,
            device=self._device,
        )

        wx_segments = [
            {"text": s["text"], "start": s["start"], "end": s["end"]}
            for s in segments
        ]

        log.info(f"Aligning {len(wx_segments)} segments...")
        aligned = whisperx.align(
            wx_segments,
            model_a,
            metadata,
            audio,
            self._device,
            return_char_alignments=False,
        )

        del model_a, metadata
        self._free_vram()
        log.info("Alignment done, VRAM released")

        # --- Step 2: Diarization (pyannote) ---
        log.info("Step 2/3: Loading diarization model (pyannote)...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self._hf_token,
            device=self._device,
        )

        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        log.info("Running diarization...")
        diarize_result = diarize_model(audio, **diarize_kwargs)

        del diarize_model
        self._free_vram()
        log.info("Diarization done, VRAM released")

        # --- Step 3: Speaker assignment ---
        log.info("Step 3/3: Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_result, aligned)
        wx_result_segs = result.get("segments", [])

        updated = self._merge_speakers(segments, wx_result_segs)

        speakers = sorted(set(s["speaker"] for s in updated))
        log.info(f"Diarization complete: {len(speakers)} speaker(s): {speakers}")

        return updated

    def _merge_speakers(
        self,
        original: list[dict],
        wx_segments: list[dict],
    ) -> list[dict]:
        """WhisperX の結果を元セグメントにマージする。"""
        updated = []

        if len(original) == len(wx_segments):
            # 1:1 マッピング
            for orig, wx in zip(original, wx_segments):
                seg = dict(orig)
                speaker = wx.get("speaker", "SPEAKER_00")
                seg["speaker"] = speaker
                seg["speaker_id"] = self._parse_speaker_id(speaker)
                updated.append(seg)
        else:
            # 数が合わない場合はタイムスタンプで最も近いものを探す
            log.warning(
                f"Segment count mismatch: original={len(original)}, "
                f"whisperx={len(wx_segments)}"
            )
            for orig in original:
                seg = dict(orig)
                speaker = self._find_speaker_by_time(
                    orig["start"], orig["end"], wx_segments
                )
                seg["speaker"] = speaker
                seg["speaker_id"] = self._parse_speaker_id(speaker)
                updated.append(seg)

        return updated

    @staticmethod
    def _parse_speaker_id(speaker: str) -> int:
        """'SPEAKER_01' -> 1"""
        try:
            return int(speaker.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _find_speaker_by_time(
        start: float,
        end: float,
        wx_segments: list[dict],
    ) -> str:
        """タイムスタンプのオーバーラップが最大の speaker を返す。"""
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        for wx in wx_segments:
            wx_start = wx.get("start", 0)
            wx_end = wx.get("end", 0)
            overlap = max(0, min(end, wx_end) - max(start, wx_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = wx.get("speaker", "SPEAKER_00")

        return best_speaker
