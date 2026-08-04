"""
Model Manager: ASR モデルのライフサイクル管理

GPU 共有の知識を持たないシンプルなモデル管理。
外部オーケストレーター (gpu-shared) または単体デプロイで使う。

対応エンジン:
  - whisper: faster-whisper (デフォルト)
  - qwen3asr: Qwen3-ASR 1.7B

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
    """ASR モデルのロード/アンロードとセッション管理"""

    def __init__(self, model_name: str, compute_type: str, engine: str = "whisper"):
        self._model_name = model_name
        self._compute_type = compute_type
        self._engine = engine
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self._session_count = 0
        self._lock = asyncio.Lock()
        self._gpu_lock = asyncio.Lock()  # diarization 等の排他 GPU 処理用

        log.info(f"ModelManager initialized: engine={engine}, model={model_name}, compute={compute_type}")

    @property
    def engine(self) -> str:
        return self._engine

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

    async def acquire(self, timeout: float = 30) -> tuple:
        """
        WS 接続時に呼ぶ。未ロードなら自動ロード。
        GPU 排他ロック (diarization 等) が解放されるまで待機する。

        Args:
            timeout: gpu_lock 待機のタイムアウト (秒)。
                     バッチ処理中は gpu_lock が保持されるため、
                     無期限待ちを防ぐ。

        Returns: (model, executor)

        Raises:
            TimeoutError: gpu_lock 取得がタイムアウトした場合
        """
        # NOTE: asyncio.wait_for(lock.acquire()) は Python 3.10 でキャンセルと
        # 取得が競合するとロックを取得したままタイムアウトを送出しうる
        # (bpo-42130 系、3.12 で修正)。バッチが gpu_lock を長時間保持する本環境
        # では 30 秒タイムアウトが頻繁に発生するため、取得タスクを自前で
        # 監視し、競合時は必ず解放してからタイムアウトを返す。
        acquire_task = asyncio.ensure_future(self._gpu_lock.acquire())

        def _abort_acquire():
            """取得タスクを確実に始末する。取得が成立していたら解放する。"""
            acquire_task.cancel()

            def _drain(t: asyncio.Task):
                if not t.cancelled() and t.exception() is None:
                    # キャンセルが間に合わず取得が成立していた → 解放
                    self._gpu_lock.release()

            acquire_task.add_done_callback(_drain)

        try:
            done, _pending = await asyncio.wait({acquire_task}, timeout=timeout)
        except BaseException:
            # 呼び出し元のキャンセル (WS 切断等)。asyncio.wait は内側のタスクを
            # キャンセルしないため、放置すると孤児タスクが後からロックを取得し
            # 誰も解放しない — まさにこの関数が防ぎたいリークになる。
            _abort_acquire()
            raise
        if acquire_task not in done:
            _abort_acquire()
            raise TimeoutError(
                f"GPU busy (lock timeout {timeout}s) — batch operation in progress?"
            )
        # 取得完了 (例外があればここで送出)
        acquire_task.result()
        try:
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
        finally:
            self._gpu_lock.release()

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

    async def force_unload(self, wait_sessions: bool = False,
                           wait_timeout: float = 120) -> bool:
        """バッチ処理用アンロード。

        Args:
            wait_sessions: True の場合、アクティブセッション完了を待ってから
                           アンロードする (安全)。False は即座にアンロード (旧動作)。
            wait_timeout: wait_sessions=True 時の最大待機秒数。
                          タイムアウト後はアンロードせず False を返す。

        呼び出し元が gpu_lock を保持していることが前提。
        """
        if wait_sessions and self._session_count > 0:
            log.info(f"force_unload: waiting for {self._session_count} session(s) "
                     f"to finish (timeout={wait_timeout}s)")
            try:
                waited = 0.0
                interval = 1.0
                while self._session_count > 0 and waited < wait_timeout:
                    # gpu_lock を一時解放して WS セッションが完了できるようにする
                    self._gpu_lock.release()
                    try:
                        await asyncio.sleep(interval)
                        waited += interval
                    finally:
                        await self._gpu_lock.acquire()
                if self._session_count > 0:
                    log.warning(f"force_unload: timeout after {waited}s, "
                                f"session_count still {self._session_count}")
                    return False
                log.info(f"force_unload: all sessions ended after {waited}s")
            except Exception as e:
                log.error(f"force_unload: error waiting for sessions: {e}")
                return False

        async with self._lock:
            if self._model is None:
                return True
            if self._session_count > 0 and not wait_sessions:
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
            "engine": self._engine,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _load_model(self):
        """ASR モデルを GPU にロード"""
        log.info(f"Loading {self._engine} model: {self._model_name} ({self._compute_type})")

        loop = asyncio.get_running_loop()
        self._model = await asyncio.wait_for(
            loop.run_in_executor(None, self._load_model_sync),
            timeout=_LOAD_TIMEOUT,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._engine)

        log.info(f"{self._engine} model loaded successfully")

    def _load_model_sync(self):
        if self._engine == "qwen3asr":
            from transcriber_qwen3asr import Qwen3ASRTranscriber
            transcriber = Qwen3ASRTranscriber()
            transcriber.load_model()
            return transcriber
        else:
            from faster_whisper import WhisperModel
            return WhisperModel(
                self._model_name,
                device="cuda",
                compute_type=self._compute_type,
            )

    def _unload_model(self):
        """ASR モデルをアンロードして VRAM を解放"""
        log.info(f"Unloading {self._engine} model...")

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._model is not None:
            # Qwen3-ASR transcriber has its own unload method
            if self._engine == "qwen3asr" and hasattr(self._model, 'unload_model'):
                self._model.unload_model()
            else:
                del self._model
            self._model = None

        gc.collect()

        try:
            import torch
            torch.cuda.empty_cache()
            log.info("VRAM released (torch.cuda.empty_cache)")
        except ImportError:
            log.debug("torch not available, skipping cuda cache clear")

        log.info(f"{self._engine} model unloaded")
