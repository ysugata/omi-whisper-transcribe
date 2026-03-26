"""
音声前処理モジュール — ラウドネス正規化

Whisper に渡す前に音量を一定レベルに正規化する。
胸元マイクなど録音レベルが不安定な環境で有効。
"""

import logging

import numpy as np

log = logging.getLogger("omi-whisper.preprocess")


class AudioPreprocessor:
    """ラウドネス正規化を適用する前処理。"""

    def __init__(
        self,
        normalize_enabled: bool = True,
        normalize_target_dbfs: float = -16.0,
    ):
        self.normalize_enabled = normalize_enabled
        self.normalize_target_dbfs = normalize_target_dbfs

        log.info(
            f"AudioPreprocessor: normalize={normalize_enabled} ({normalize_target_dbfs}dBFS)"
        )

    def process_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        float32 音声データを正規化して返す。

        Args:
            audio: float32 音声データ (-1.0 ~ 1.0)
            sample_rate: サンプリングレート (未使用だが将来の拡張用)

        Returns:
            正規化済み音声データ
        """
        if not self.normalize_enabled:
            return audio

        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio  # 無音はそのまま

        current_dbfs = 20 * np.log10(rms + 1e-10)
        gain_db = self.normalize_target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)

        normalized = audio * gain_linear
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32)
