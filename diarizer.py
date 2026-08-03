"""pyannote ベースの話者分離モジュール

セッション終了時に完全な音声ファイルに対して話者分離を実行する。
RTX 3060 12GB の VRAM 制約のため、各ステップを順次実行し、
不要になったモデルは即座に解放する。

VRAM フロー:
  1. Whisper モデルをアンロード (呼び出し元で実施)
  2. pyannote (diarization) → 処理 → 解放
  3. 結果を元セグメントにマージ (CPU のみ)

必要パッケージ:
  pip install torch torchaudio pyannote.audio

pyannote/speaker-diarization-community-1 は gated model のため HF トークン必須:
  https://huggingface.co/pyannote/speaker-diarization-community-1
  https://huggingface.co/pyannote/segmentation-3.0
"""

import gc
import logging
from typing import Optional

log = logging.getLogger("omi-whisper.diarizer")


class Diarizer:
    """pyannote Pipeline を直接使った話者分離。

    whisperx の wav2vec2 alignment を除去し、
    pyannote の diarization 結果とセグメントの時間オーバーラップで
    話者を割り当てる。複数話者にまたがるセグメントは話者境界で分割する。
    """

    def __init__(
        self,
        hf_token: str,
        device: str = "cuda",
        language: str = "ja",
    ) -> None:
        self._hf_token = hf_token
        self._device = device
        self._language = language  # kept for backward compat (not used internally)
        self._enabled = bool(hf_token)

        if self._enabled:
            log.info(f"Diarizer initialized: device={device}")
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

    def load_pipeline(self):
        """Load pyannote pipeline once for batch processing. Returns pipeline object."""
        import torch
        from pyannote.audio import Pipeline

        log.info("Loading diarization model (pyannote)...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self._hf_token,
        )
        pipeline.to(torch.device(self._device))
        log.info("Diarization model loaded")
        return pipeline

    def unload_pipeline(self, pipeline):
        """Unload pipeline and free VRAM.

        NOTE: `del pipeline` only drops THIS frame's reference — the caller
        must also drop its own reference (pipeline = None) and then call
        _free_vram(), otherwise empty_cache() runs while the ~2GB model is
        still alive and nothing is actually released.
        """
        del pipeline
        self._free_vram()
        log.info("Diarization model unloaded, VRAM released")

    def diarize_with_pipeline(
        self,
        pipeline,
        audio_path: str,
        segments: list[dict],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> tuple[list[dict], bool]:
        """Diarize using a pre-loaded pipeline (for batch mode)."""
        if not segments:
            return segments, False
        try:
            result = self._run_pipeline_with(pipeline, audio_path, segments, min_speakers, max_speakers)
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
        from pyannote.audio import Pipeline

        log.info(f"Starting diarization: {audio_path}")
        pipeline = self.load_pipeline()
        try:
            result = self._run_pipeline_with(pipeline, audio_path, segments, min_speakers, max_speakers)
        finally:
            self.unload_pipeline(pipeline)
            pipeline = None
            self._free_vram()
        return result

    def _run_pipeline_with(
        self,
        pipeline,
        audio_path: str,
        segments: list[dict],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> list[dict]:
        """Run diarization with a pre-loaded pipeline."""
        import torchaudio

        log.info(f"Starting diarization: {audio_path}")

        # --- Step 1: Load audio ---
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        audio_duration = waveform.shape[1] / sr
        log.info(f"Audio loaded: {audio_duration:.1f}s")

        if audio_duration < 1.0:
            log.warning(f"Audio too short for diarization: {audio_duration:.1f}s")
            raise ValueError(f"Audio too short: {audio_duration:.1f}s")

        # Validate and clamp segment timestamps
        valid_segments = self._validate_segments(segments, audio_duration)
        if not valid_segments:
            log.warning("No valid segments after timestamp clamping")
            raise ValueError("No valid segments for diarization")

        log.info(f"Prepared {len(valid_segments)} segments "
                 f"(from {len(segments)} original, audio={audio_duration:.1f}s)")

        # --- Step 2: Diarization ---
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

        log.info("Running diarization...")
        diarize_output = pipeline(
            {"waveform": waveform, "sample_rate": sr},
            **diarize_kwargs,
        )

        # Free audio waveform immediately
        del waveform
        self._free_vram()

        # pyannote 4.x returns DiarizeOutput dataclass
        if hasattr(diarize_output, 'exclusive_speaker_diarization'):
            annotation = diarize_output.exclusive_speaker_diarization
        elif hasattr(diarize_output, 'speaker_diarization'):
            annotation = diarize_output.speaker_diarization
        else:
            annotation = diarize_output

        # --- Step 3: Speaker assignment (CPU only) ---
        log.info("Step 2/2: Assigning speakers...")
        updated = self._assign_speakers(valid_segments, annotation)

        del annotation
        gc.collect()

        speakers = sorted(set(s.get("speaker", "SPEAKER_00") for s in updated))
        log.info(f"Diarization complete: {len(speakers)} speaker(s): {speakers}")

        return updated

    def _validate_segments(
        self,
        segments: list[dict],
        audio_duration: float,
    ) -> list[dict]:
        """セグメントのタイムスタンプを音声長にクランプし、無効なものを除去する。"""
        valid = []
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
            # Preserve all original fields
            seg = dict(s)
            seg["start"] = start
            seg["end"] = end
            seg["text"] = text
            valid.append(seg)
        return valid

    def _assign_speakers(
        self,
        segments: list[dict],
        annotation,
    ) -> list[dict]:
        """pyannote Annotation から各セグメントに話者を割り当てる。

        各セグメントの時間範囲と diarization 結果のオーバーラップを計算し、
        複数話者にまたがるセグメントは話者境界で分割する。
        """
        # Extract diarization segments as sorted list of (start, end, speaker)
        dia_segments = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            dia_segments.append((segment.start, segment.end, speaker))
        dia_segments.sort(key=lambda x: x[0])

        if not dia_segments:
            log.warning("Diarization returned no speaker segments")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
                seg["speaker_id"] = 0
            return segments

        log.info(f"Diarization found {len(dia_segments)} speaker segments")

        result = []
        for seg in segments:
            mapped = self._map_segment_to_speakers(seg, dia_segments)
            result.extend(mapped)

        return result

    def _map_segment_to_speakers(
        self,
        seg: dict,
        dia_segments: list[tuple],
    ) -> list[dict]:
        """1セグメントと diarization 結果のオーバーラップを計算し、
        話者を割り当てる。複数話者がある場合はセグメントを分割する。

        Returns:
            list of segment dicts (1つまたは分割された複数)
        """
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Calculate overlap with each diarization segment
        overlaps = []
        for dia_start, dia_end, speaker in dia_segments:
            overlap_start = max(seg_start, dia_start)
            overlap_end = min(seg_end, dia_end)
            overlap = overlap_end - overlap_start
            if overlap > 0:
                overlaps.append((overlap_start, overlap_end, speaker, overlap))

        if not overlaps:
            # No overlap found — assign default speaker
            new_seg = dict(seg)
            new_seg["speaker"] = "SPEAKER_00"
            new_seg["speaker_id"] = 0
            return [new_seg]

        # Sort overlaps by start time
        overlaps.sort(key=lambda x: x[0])

        # Check if all overlaps are from the same speaker
        unique_speakers = set(o[2] for o in overlaps)
        if len(unique_speakers) == 1:
            speaker = overlaps[0][2]
            new_seg = dict(seg)
            new_seg["speaker"] = speaker
            new_seg["speaker_id"] = self._parse_speaker_id(speaker)
            return [new_seg]

        # Multiple speakers — split the segment at speaker boundaries
        return self._split_at_speaker_boundaries(seg, overlaps)

    def _split_at_speaker_boundaries(
        self,
        seg: dict,
        overlaps: list[tuple],
    ) -> list[dict]:
        """テキストを話者境界の時間比で按分分割する。

        overlaps: sorted list of (overlap_start, overlap_end, speaker, overlap_duration)
        """
        text = seg["text"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_duration = seg_end - seg_start

        if seg_duration <= 0:
            new_seg = dict(seg)
            new_seg["speaker"] = overlaps[0][2]
            new_seg["speaker_id"] = self._parse_speaker_id(overlaps[0][2])
            return [new_seg]

        # Merge consecutive overlaps with the same speaker
        merged = []
        for overlap_start, overlap_end, speaker, _ in overlaps:
            if merged and merged[-1][2] == speaker:
                # Extend the previous entry
                merged[-1] = (merged[-1][0], overlap_end, speaker)
            else:
                merged.append((overlap_start, overlap_end, speaker))

        if len(merged) == 1:
            new_seg = dict(seg)
            new_seg["speaker"] = merged[0][2]
            new_seg["speaker_id"] = self._parse_speaker_id(merged[0][2])
            return [new_seg]

        # Split text proportionally by duration
        total_overlap_duration = sum(e - s for s, e, _ in merged)
        if total_overlap_duration <= 0:
            new_seg = dict(seg)
            new_seg["speaker"] = merged[0][2]
            new_seg["speaker_id"] = self._parse_speaker_id(merged[0][2])
            return [new_seg]

        result = []
        char_offset = 0
        text_len = len(text)

        for i, (m_start, m_end, speaker) in enumerate(merged):
            duration = m_end - m_start
            ratio = duration / total_overlap_duration

            if i == len(merged) - 1:
                # Last chunk gets remaining text
                chunk_text = text[char_offset:]
            else:
                char_count = round(text_len * ratio)
                chunk_text = text[char_offset:char_offset + char_count]
                char_offset += char_count

            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Skip very short fragments (< 3 chars)
            if len(chunk_text) < 3 and i < len(merged) - 1:
                continue

            new_seg = dict(seg)
            new_seg["start"] = round(m_start, 3)
            new_seg["end"] = round(m_end, 3)
            new_seg["text"] = chunk_text
            new_seg["speaker"] = speaker
            new_seg["speaker_id"] = self._parse_speaker_id(speaker)
            result.append(new_seg)

        if not result:
            # Fallback: assign dominant speaker
            best = max(overlaps, key=lambda x: x[3])
            new_seg = dict(seg)
            new_seg["speaker"] = best[2]
            new_seg["speaker_id"] = self._parse_speaker_id(best[2])
            return [new_seg]

        return result

    @staticmethod
    def _parse_speaker_id(speaker: str) -> int:
        """'SPEAKER_01' -> 1"""
        try:
            return int(speaker.split("_")[-1])
        except (ValueError, IndexError):
            return 0
