"""
音声前処理モジュール — ノイズ除去 + ラウドネス正規化

Whisper に渡す前にノイズ除去・音量正規化を行う。
胸元マイクなど録音レベルが不安定な環境や、
車内・電車内などの定常騒音がある環境で有効。
"""

import logging

import numpy as np

log = logging.getLogger("omi-whisper.preprocess")


class AudioPreprocessor:
    """ノイズ除去 + ラウドネス正規化を適用する前処理。"""

    def __init__(
        self,
        normalize_enabled: bool = True,
        normalize_target_dbfs: float = -16.0,
        noise_reduce_enabled: bool = False,
        noise_reduce_prop_decrease: float = 0.8,
    ):
        self.normalize_enabled = normalize_enabled
        self.normalize_target_dbfs = normalize_target_dbfs
        self.noise_reduce_enabled = noise_reduce_enabled
        self.noise_reduce_prop_decrease = noise_reduce_prop_decrease

        features = []
        if noise_reduce_enabled:
            features.append(f"noise_reduce(prop={noise_reduce_prop_decrease})")
        if normalize_enabled:
            features.append(f"normalize({normalize_target_dbfs}dBFS)")

        log.info(f"AudioPreprocessor: {', '.join(features) if features else 'none'}")

    def process_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        float32 音声データを前処理して返す。

        処理順: ノイズ除去 → ラウドネス正規化

        Args:
            audio: float32 音声データ (-1.0 ~ 1.0)
            sample_rate: サンプリングレート

        Returns:
            前処理済み音声データ
        """
        # Step 1: ノイズ除去 (スペクトルゲーティング)
        if self.noise_reduce_enabled:
            audio = self._reduce_noise(audio, sample_rate)

        # Step 2: ラウドネス正規化
        if self.normalize_enabled:
            audio = self._normalize(audio)

        return audio

    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """noisereduce によるスペクトルゲーティング。

        stationary=True は定常騒音 (エンジン音、空調、電車走行音) に最適。
        CPU numpy ベースのため追加 VRAM は不要。
        """
        try:
            import noisereduce as nr
        except ImportError:
            log.warning("noisereduce not installed, skipping noise reduction")
            return audio

        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio  # 無音はそのまま

        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=self.noise_reduce_prop_decrease,
            use_torch=False,
        )

        return reduced.astype(np.float32)

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """ラウドネス正規化。"""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio  # 無音はそのまま

        current_dbfs = 20 * np.log10(rms + 1e-10)
        gain_db = self.normalize_target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)

        normalized = audio * gain_linear
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized.astype(np.float32)
