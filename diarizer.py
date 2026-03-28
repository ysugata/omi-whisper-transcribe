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

pyannote/speaker-diarization-community-1 は gated model のため HF トークン必須:
  https://huggingface.co/pyannote/speaker-diarization-community-1
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
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def diarize(
        self,
        audio_path: str,
        segments: list[dict],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> tuple[list[dict], bool]:
        """音声ファイルに対して話者分離を実行し、(更新セグメント, 成功フラグ) を返す。

        エラー時は (元のセグメント, False) を返す (テキストを失わない)。
        """
        if not self._enabled:
            log.warning("Diarizer is disabled (no HF token)")
            return segments, False
        if not segments:
            log.warning("No segments to diarize")
            return segments, False

        try:
            result = self._run_pipeline(audio_path, segments, min_speakers, max_speakers)
            return result, True
        except Exception as e:
            log.error(f"Diarization FAILED (original segments preserved): {e}", exc_info=True)
            self._free_vram()
            return segments, False

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
        from whisperx.diarize import DiarizationPipeline

        log.info(f"Starting diarization: {audio_path}")

        # 音声ロード
        audio = whisperx.load_audio(audio_path)
        audio_duration = len(audio) / 16000.0
        log.info(f"Audio loaded: {audio_duration:.1f}s")

        if audio_duration < 1.0:
            log.warning(f"Audio too short for diarization: {audio_duration:.1f}s")
            raise ValueError(f"Audio too short: {audio_duration:.1f}s")

        # セグメントのタイムスタンプを音声長にクランプし、無効なものを除去
        wx_segments = []
        for s in segments:
            text = s.get("text", "").strip()
            if not text:
                continue
            start = max(0.0, min(s.get("start", 0.0), audio_duration))
            end = max(0.0, min(s.get("end", 0.0), audio_duration))
            if start >= end:
                end = min(start + 0.5, audio_duration)
            if start >= audio_duration:
                continue
            wx_segments.append({"text": text, "start": start, "end": end})

        if not wx_segments:
            log.warning("No valid segments after timestamp clamping")
            raise ValueError("No valid segments for alignment")

        log.info(f"Prepared {len(wx_segments)} segments for alignment "
                 f"(from {len(segments)} original, audio={audio_duration:.1f}s)")

        # --- Step 1: Alignment (wav2vec2) ---
        log.info("Step 1/3: Loading alignment model (wav2vec2)...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=self._language,
                device=self._device,
            )
        except Exception as e:
            log.error(f"Failed to load alignment model for '{self._language}': {e}")
            raise

        log.info(f"Aligning {len(wx_segments)} segments...")
        try:
            aligned = whisperx.align(
                wx_segments,
                model_a,
                metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
        except Exception as e:
            log.error(f"Alignment failed: {e}")
            del model_a, metadata
            self._free_vram()
            raise

        del model_a, metadata
        self._free_vram()
        log.info("Alignment done, VRAM released")

        # --- Step 2: Diarization (pyannote) ---
        log.info("Step 2/3: Loading diarization model (pyannote)...")
        try:
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-community-1",
                token=self._hf_token,
                device=self._device,
            )
        except Exception as e:
            log.error(f"Failed to load diarization model (check HF token and model access): {e}")
            raise

        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        log.info("Running diarization...")
        try:
            diarize_result = diarize_model(audio, **diarize_kwargs)
        except Exception as e:
            log.error(f"Diarization inference failed: {e}")
            del diarize_model
            self._free_vram()
            raise

        del diarize_model
        self._free_vram()
        log.info("Diarization done, VRAM released")

        # --- Step 3: Speaker assignment ---
        log.info("Step 3/3: Assigning speakers...")
        try:
            result = whisperx.assign_word_speakers(diarize_result, aligned)
        except Exception as e:
            log.error(f"Speaker assignment failed: {e}")
            raise
        wx_result_segs = result.get("segments", [])

        if not wx_result_segs:
            log.warning("whisperx returned 0 segments after speaker assignment")
            raise ValueError("Speaker assignment produced no segments")

        updated = self._merge_speakers(segments, wx_result_segs)

        # Free intermediate variables to ensure VRAM is fully released
        # before returning (prevents OOM when caller releases gpu_lock)
        del audio, aligned, diarize_result, result, wx_result_segs
        self._free_vram()

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
