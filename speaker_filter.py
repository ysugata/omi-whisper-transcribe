"""マルチスピーカー声紋登録 + 話者タグ付けモジュール

pyannote/embedding (wespeaker-voxceleb-resnet34-LM, 256次元) を使い、
複数の話者の声紋を登録・比較する。

VRAM フロー (diarizer.py と同パターン):
  1. embedding モデルをロード
  2. 音声セグメントから声紋抽出
  3. モデルをアンロード (VRAM 解放)

必要パッケージ:
  whisperx の推移的依存で既にインストール済み。pip 追加不要。
  HF_AUTH_TOKEN が必要。
"""

import gc
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("omi-whisper.speaker-filter")

DEFAULT_THRESHOLD = 0.55
SPEAKERS_FILE = "speakers.json"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


class SpeakerFilter:
    """声紋ベースの話者フィルタ。"""

    def __init__(
        self,
        config_dir: str,
        hf_token: str,
        device: str = "cuda",
    ) -> None:
        self._config_dir = config_dir
        self._hf_token = hf_token
        self._device = device
        self._speakers_path = os.path.join(config_dir, SPEAKERS_FILE)
        self._enabled = bool(hf_token)

        # config の読み込み
        self._data = self._load()

        if self._enabled:
            log.info(f"SpeakerFilter initialized: {len(self._data['speakers'])} speaker(s)")
        else:
            log.warning("SpeakerFilter disabled: HF_AUTH_TOKEN not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        try:
            if os.path.exists(self._speakers_path):
                with open(self._speakers_path, encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"speakers.json load error: {e}")
        return {"threshold": DEFAULT_THRESHOLD, "model": EMBEDDING_MODEL, "speakers": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._speakers_path) or ".", exist_ok=True)
        with open(self._speakers_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # VRAM management
    # ------------------------------------------------------------------

    @staticmethod
    def _free_vram():
        gc.collect()
        try:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_model(self):
        """wespeaker-voxceleb-resnet34-LM モデルをロードして返す。"""
        import torch
        from pyannote.audio import Model, Inference
        model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=self._hf_token)
        device = torch.device(self._device)
        inference = Inference(model, window="whole", device=device)
        log.info("Embedding model loaded (device=%s)", device)
        return inference

    @staticmethod
    def _unload_model(inference):
        del inference
        gc.collect()
        try:
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("Embedding model unloaded")

    # ------------------------------------------------------------------
    # Speaker management
    # ------------------------------------------------------------------

    def add_speaker(self, name: str, is_user: bool = False) -> dict:
        speaker = {
            "id": "sp_" + uuid.uuid4().hex[:6],
            "name": name,
            "is_user": is_user,
            "created_at": datetime.now().isoformat(),
            "embedding": None,
            "samples": [],
        }
        self._data["speakers"].append(speaker)
        self._save()
        log.info(f"Speaker added: {speaker['id']} ({name}, is_user={is_user})")
        return speaker

    def remove_speaker(self, speaker_id: str) -> bool:
        before = len(self._data["speakers"])
        self._data["speakers"] = [
            s for s in self._data["speakers"] if s["id"] != speaker_id
        ]
        removed = len(self._data["speakers"]) < before
        if removed:
            self._save()
            log.info(f"Speaker removed: {speaker_id}")
        return removed

    def update_speaker(self, speaker_id: str, name: Optional[str] = None, is_user: Optional[bool] = None) -> Optional[dict]:
        speaker = self._find_speaker(speaker_id)
        if not speaker:
            return None
        if name is not None:
            speaker["name"] = name
        if is_user is not None:
            speaker["is_user"] = is_user
        self._save()
        return speaker

    def list_speakers(self) -> list:
        result = []
        for s in self._data["speakers"]:
            result.append({
                "id": s["id"],
                "name": s["name"],
                "is_user": s["is_user"],
                "created_at": s["created_at"],
                "sample_count": len(s.get("samples", [])),
                "total_duration": sum(
                    sample.get("duration", 0) for sample in s.get("samples", [])
                ),
                "has_embedding": s.get("embedding") is not None,
            })
        return result

    def list_speakers_detailed(self) -> list:
        """list_speakers() と同じ構造 + 各サンプルのメタデータ (embedding 除外)。"""
        result = []
        for s in self._data["speakers"]:
            samples_meta = []
            for sample in s.get("samples", []):
                meta = {
                    "id": sample.get("id"),
                    "added_at": sample.get("added_at"),
                    "source_session": sample.get("source_session", ""),
                    "segment_indices": sample.get("segment_indices", []),
                    "duration": sample.get("duration", 0),
                    "source": sample.get("source", "manual"),
                }
                if "audio_start" in sample:
                    meta["audio_start"] = sample["audio_start"]
                    meta["audio_end"] = sample["audio_end"]
                samples_meta.append(meta)
            result.append({
                "id": s["id"],
                "name": s["name"],
                "is_user": s["is_user"],
                "created_at": s["created_at"],
                "sample_count": len(s.get("samples", [])),
                "total_duration": sum(
                    sample.get("duration", 0) for sample in s.get("samples", [])
                ),
                "has_embedding": s.get("embedding") is not None,
                "samples": samples_meta,
            })
        return result

    def _find_speaker(self, speaker_id: str) -> Optional[dict]:
        for s in self._data["speakers"]:
            if s["id"] == speaker_id:
                return s
        return None

    # ------------------------------------------------------------------
    # Voiceprint sample management
    # ------------------------------------------------------------------

    def add_voiceprint(self, speaker_id: str, audio_path: str, segments: list, session_name: str = "") -> Optional[dict]:
        """指定話者に声紋サンプルを追加する。

        Args:
            speaker_id: 話者 ID
            audio_path: 音声ファイルパス
            segments: セグメントリスト (start, end を含む dict)
            session_name: セッション名 (表示用)

        Returns:
            追加されたサンプル情報、失敗時は None
        """
        speaker = self._find_speaker(speaker_id)
        if not speaker:
            log.warning(f"Speaker not found: {speaker_id}")
            return None

        if not segments:
            log.warning("No segments provided for voiceprint")
            return None

        try:
            inference = self._load_model()
            try:
                embedding, duration = self._extract_embedding(inference, audio_path, segments)
            finally:
                self._unload_model(inference)
        except Exception as e:
            log.error(f"Failed to extract voiceprint: {e}", exc_info=True)
            self._free_vram()
            return None

        if embedding is None:
            return None

        # Compute audio time range from segments
        audio_start = min(s.get("start", 0) for s in segments)
        audio_end = max(s.get("end", 0) for s in segments)

        sample = {
            "id": "s_" + uuid.uuid4().hex[:6],
            "added_at": datetime.now().isoformat(),
            "source_session": session_name,
            "segment_indices": list(range(len(segments))),
            "duration": round(duration, 1),
            "audio_start": round(audio_start, 2),
            "audio_end": round(audio_end, 2),
            "embedding": embedding.tolist(),
        }

        if "samples" not in speaker:
            speaker["samples"] = []
        speaker["samples"].append(sample)

        # 全サンプルの平均 embedding を再計算
        self._recalc_speaker_embedding(speaker)
        self._save()

        log.info(f"Voiceprint added: {speaker_id} ({session_name}, {duration:.1f}s)")
        return sample

    def remove_voiceprint(self, speaker_id: str, sample_id: str) -> Optional[dict]:
        speaker = self._find_speaker(speaker_id)
        if not speaker:
            return None

        before = len(speaker.get("samples", []))
        speaker["samples"] = [
            s for s in speaker.get("samples", []) if s["id"] != sample_id
        ]
        if len(speaker["samples"]) == before:
            return None

        self._recalc_speaker_embedding(speaker)
        self._save()

        log.info(f"Voiceprint removed: {speaker_id}/{sample_id}")
        return {"speaker_id": speaker_id, "sample_count": len(speaker["samples"])}

    def _recalc_speaker_embedding(self, speaker: dict):
        """全サンプルの embedding 平均を speaker['embedding'] にセット。"""
        samples = speaker.get("samples", [])
        if not samples:
            speaker["embedding"] = None
            return

        embeddings = [
            np.array(s["embedding"]) for s in samples if s.get("embedding")
        ]
        if not embeddings:
            speaker["embedding"] = None
            return

        avg = np.mean(embeddings, axis=0)
        # 正規化 (コサイン類似度のため)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        speaker["embedding"] = avg.tolist()

    @staticmethod
    def _vad_filter_audio(waveform_1d_np: np.ndarray, sr: int = 16000):
        """Apply Silero VAD to extract speech-only regions from audio.

        Args:
            waveform_1d_np: 1-D float32 numpy array (mono).
            sr: Sample rate.

        Returns:
            Filtered 1-D numpy array with only speech regions, or None if no speech found.
        """
        from faster_whisper.vad import get_speech_timestamps, VadOptions

        vad_opts = VadOptions(
            threshold=0.35,
            min_speech_duration_ms=250,
            speech_pad_ms=100,
        )
        timestamps = get_speech_timestamps(waveform_1d_np, vad_opts)
        if not timestamps:
            return None

        chunks = []
        for ts in timestamps:
            chunks.append(waveform_1d_np[ts["start"]:ts["end"]])
        return np.concatenate(chunks)

    def _extract_embedding(self, inference, audio_path: str, segments: list):
        """セグメントリストから embedding を抽出する。

        Returns:
            (embedding, total_duration) or (None, 0)
        """
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        embeddings = []
        total_duration = 0.0

        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            duration = end - start
            if duration < 0.5:
                continue

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            end_sample = min(end_sample, waveform.shape[1])
            if start_sample >= end_sample:
                continue

            segment_audio = waveform[:, start_sample:end_sample]

            # VAD filter: extract speech-only regions
            segment_np = segment_audio.squeeze(0).numpy()
            filtered = self._vad_filter_audio(segment_np, sr)
            if filtered is None:
                log.warning(
                    f"VAD: no speech in segment [{start:.1f}-{end:.1f}s], skipping"
                )
                continue

            filtered_waveform = torch.from_numpy(filtered).unsqueeze(0).float()
            total_duration += duration

            # pyannote Inference expects dict-like or file
            # We pass a dict with waveform and sample_rate
            emb = inference({"waveform": filtered_waveform, "sample_rate": sr})
            embeddings.append(emb)

        if not embeddings:
            log.warning("No valid segments for embedding extraction")
            return None, 0

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg, total_duration

    # ------------------------------------------------------------------
    # Realtime voice filter
    # ------------------------------------------------------------------

    def get_user_speakers(self) -> list:
        """Return speakers with is_user=True and a valid embedding."""
        return [
            s for s in self._data["speakers"]
            if s.get("is_user") and s.get("embedding") is not None
        ]

    def filter_segments_realtime(
        self,
        inference,
        audio_np: np.ndarray,
        segments: list,
        sample_rate: int = 16000,
        threshold: float = 0,
    ) -> list:
        """Filter out non-user segments using pre-loaded embedding model.

        Args:
            inference: Pre-loaded pyannote Inference instance.
            audio_np: float32 numpy array (mono, already at sample_rate).
            segments: List of segment dicts (start, end, text, ...).
            sample_rate: Sample rate of audio_np.
            threshold: Cosine similarity threshold. 0 = use speakers.json value.

        Returns:
            Filtered list of segments (only user speech retained).
        """
        import torch

        user_speakers = self.get_user_speakers()
        if not user_speakers:
            return segments

        if threshold <= 0:
            threshold = self._data.get("threshold", DEFAULT_THRESHOLD)

        # Build user embedding matrix (normalized)
        user_embs = []
        for sp in user_speakers:
            emb = np.array(sp["embedding"])
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            user_embs.append(emb)
        user_embs = np.array(user_embs)

        # Ensure audio is 2D: (1, samples) for pyannote
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
        else:
            waveform = torch.from_numpy(audio_np).float()

        kept = []
        filtered_count = 0

        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            duration = end - start

            # Too short to judge reliably — keep
            if duration < 0.5:
                kept.append(seg)
                continue

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            end_sample = min(end_sample, waveform.shape[1])
            if start_sample >= end_sample:
                kept.append(seg)
                continue

            try:
                segment_audio = waveform[:, start_sample:end_sample]
                emb = inference({"waveform": segment_audio, "sample_rate": sample_rate})
                emb = np.array(emb)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                # Max cosine similarity against all user speakers
                similarities = user_embs @ emb
                best_sim = float(np.max(similarities))

                if best_sim >= threshold:
                    kept.append(seg)
                else:
                    text_preview = seg.get("text", "")[:40]
                    log.info(
                        f"Voice filter: removed segment [{start:.1f}-{end:.1f}s] "
                        f"sim={best_sim:.3f} < {threshold:.2f} "
                        f"text=\"{text_preview}\""
                    )
                    filtered_count += 1
            except Exception as e:
                log.warning(f"Voice filter error at {start:.1f}-{end:.1f}s: {e}")
                kept.append(seg)  # On error, keep the segment

        if filtered_count > 0:
            log.info(f"Voice filter: kept {len(kept)}/{len(segments)} segments "
                     f"(removed {filtered_count})")

        return kept

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag_segments(self, audio_path: str, segments: list) -> list:
        """各セグメントの embedding を全登録話者と比較し、person_id / is_user をセット。

        Returns:
            更新されたセグメントリスト
        """
        speakers_with_emb = [
            s for s in self._data["speakers"]
            if s.get("embedding") is not None
        ]
        if not speakers_with_emb:
            log.warning("No speakers with embeddings registered")
            return segments

        threshold = self._data.get("threshold", DEFAULT_THRESHOLD)

        try:
            inference = self._load_model()
            try:
                return self._tag_segments_impl(inference, audio_path, segments, speakers_with_emb, threshold)
            finally:
                self._unload_model(inference)
        except Exception as e:
            log.error(f"Failed to tag segments: {e}", exc_info=True)
            self._free_vram()
            return segments

    def tag_segments_with_model(self, inference, audio_path: str, segments: list) -> list:
        """既にロード済みのモデルを使ってタグ付け (バッチ用)。"""
        speakers_with_emb = [
            s for s in self._data["speakers"]
            if s.get("embedding") is not None
        ]
        if not speakers_with_emb:
            return segments

        threshold = self._data.get("threshold", DEFAULT_THRESHOLD)
        return self._tag_segments_impl(inference, audio_path, segments, speakers_with_emb, threshold)

    def _tag_segments_impl(self, inference, audio_path: str, segments: list, speakers: list, threshold: float) -> list:
        """タグ付けの実装。"""
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        # 登録話者の embedding を numpy 配列に
        speaker_embs = []
        for sp in speakers:
            emb = np.array(sp["embedding"])
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            speaker_embs.append(emb)
        speaker_embs = np.array(speaker_embs)

        updated = []
        for seg in segments:
            seg = dict(seg)  # copy
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            duration = end - start

            if duration < 0.5:
                # 短すぎるセグメントはスキップ (既存値を維持)
                updated.append(seg)
                continue

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            end_sample = min(end_sample, waveform.shape[1])
            if start_sample >= end_sample:
                updated.append(seg)
                continue

            try:
                segment_audio = waveform[:, start_sample:end_sample]
                emb = inference({"waveform": segment_audio, "sample_rate": sr})
                emb = np.array(emb)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                # コサイン類似度 (正規化済みなので内積)
                similarities = speaker_embs @ emb
                best_idx = int(np.argmax(similarities))
                best_sim = float(similarities[best_idx])

                if best_sim >= threshold:
                    seg["person_id"] = speakers[best_idx]["id"]
                    seg["person_name"] = speakers[best_idx]["name"]
                    seg["is_user"] = speakers[best_idx]["is_user"]
                    seg["speaker_similarity"] = round(best_sim, 3)
                else:
                    seg["person_id"] = "unknown"
                    seg["person_name"] = "unknown"
                    seg["is_user"] = False
                    seg["speaker_similarity"] = round(best_sim, 3)
            except Exception as e:
                log.warning(f"Failed to tag segment at {start:.1f}-{end:.1f}: {e}")
                seg["person_id"] = "unknown"
                seg["person_name"] = "unknown"
                seg["is_user"] = False

            updated.append(seg)

        return updated

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        speakers = self._data.get("speakers", [])
        speakers_with_emb = sum(1 for s in speakers if s.get("embedding"))
        total_samples = sum(len(s.get("samples", [])) for s in speakers)
        return {
            "enabled": self._enabled,
            "speaker_count": len(speakers),
            "speakers_with_embedding": speakers_with_emb,
            "total_samples": total_samples,
            "threshold": self._data.get("threshold", DEFAULT_THRESHOLD),
            "model": EMBEDDING_MODEL,
        }

    def get_threshold(self) -> float:
        return self._data.get("threshold", DEFAULT_THRESHOLD)

    def update_threshold(self, threshold: float) -> None:
        threshold = max(0.3, min(0.9, threshold))
        self._data["threshold"] = round(threshold, 2)
        self._save()
        log.info(f"Threshold updated: {threshold}")

    def delete_all(self) -> None:
        self._data["speakers"] = []
        self._save()
        log.info("All speakers deleted")

    def reload(self) -> None:
        """ファイルから再読み込み。"""
        self._data = self._load()
