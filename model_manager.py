"""
Model Manager: Whisper モデルのライフサイクル管理

GPU 共有の知識を持たないシンプルなモデル管理。
外部オーケストレーター (gpu-shared) または単体デプロイで使う。

管理 REST API:
  POST /model/load    — モデルロード (冪等)
  POST /model/unload  — モデルアンロード (セッション中は 409)
  GET  /model/status  — 状態取得
"""

import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("omi-whisper.model")

_LOAD_TIMEOUT = 600  # モデルロードのタイムアウト (秒)


class ModelManager:
    """Whisper モデルのロード/アンロードとセッション管理"""

    def __init__(self, model_name: str, compute_type: str):
        self._model_name = model_name
        self._compute_type = compute_type
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self._session_count = 0
        self._lock = asyncio.Lock()
        self._gpu_lock = asyncio.Lock()  # diarization 等の排他 GPU 処理用

        log.info(f"ModelManager initialized: model={model_name}, compute={compute_type}")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def session_count(self) -> int:
        return self._session_count

    @property
    def gpu_lock(self) -> asyncio.Lock:
        """排他 GPU 処理用ロック。diarization 等で使用。"""
        return self._gpu_lock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> tuple:
        """
        WS 接続時に呼ぶ。未ロードなら自動ロード。
        GPU 排他ロック (diarization 等) が解放されるまで待機する。

        Returns: (model, executor)
        """
        async with self._gpu_lock:
            async with self._lock:
                self._session_count += 1
                log.info(f"acquire: session_count={self._session_count}")

                if self._model is not None:
                    return self._model, self._executor

                try:
                    await self._load_model()
                except BaseException:
                    self._session_count -= 1
                    raise

                return self._model, self._executor

    def release(self):
        """WS 切断時に呼ぶ。"""
        self._session_count = max(0, self._session_count - 1)
        log.info(f"release: session_count={self._session_count}")

    async def load_model(self):
        """REST API 用。冪等。"""
        async with self._lock:
            if self._model is not None:
                log.info("load_model: already loaded")
                return
            await self._load_model()

    async def unload_model(self) -> bool:
        """
        REST API 用。セッション中は False を返す。

        Returns: True if unloaded, False if sessions active
        """
        async with self._lock:
            if self._session_count > 0:
                log.warning(f"unload_model: rejected, session_count={self._session_count}")
                return False
            if self._model is None:
                log.info("unload_model: already unloaded")
                return True
            self._unload_model()
            return True

    async def force_unload(self) -> bool:
        """バッチ処理用。session_count を無視して強制アンロード。

        既存 WS セッションの model 参照は無効になるため、
        呼び出し元が gpu_lock を保持していることが前提。
        """
        async with self._lock:
            if self._model is None:
                return True
            log.warning(f"force_unload: unloading with session_count={self._session_count}")
            self._unload_model()
            return True

    def status(self) -> dict:
        """状態を返す。"""
        return {
            "loaded": self.loaded,
            "session_count": self._session_count,
            "model_name": self._model_name,
            "compute_type": self._compute_type,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _load_model(self):
        """Whisper モデルを GPU にロード"""
        log.info(f"Loading Whisper model: {self._model_name} ({self._compute_type})")

        loop = asyncio.get_running_loop()
        self._model = await asyncio.wait_for(
            loop.run_in_executor(None, self._load_model_sync),
            timeout=_LOAD_TIMEOUT,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

        log.info("Whisper model loaded successfully")

    def _load_model_sync(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._model_name,
            device="cuda",
            compute_type=self._compute_type,
        )

    def _unload_model(self):
        """Whisper モデルをアンロードして VRAM を解放"""
        log.info("Unloading Whisper model...")

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._model is not None:
            del self._model
            self._model = None

        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
            log.info("VRAM released (torch.cuda.empty_cache)")
        except ImportError:
            log.debug("torch not available, skipping cuda cache clear")

        log.info("Whisper model unloaded")
