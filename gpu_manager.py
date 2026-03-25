"""
GPU Manager: omi_whisper_bridge と qwen-tts-api 間の GPU 自動切り替え

RTX 3060 12GB を共有するため、WebSocket 接続の有無に応じて
Whisper モデルのロード/アンロードと qwen-tts コンテナの停止/起動を自動制御する。

状態遷移:
  IDLE → ACQUIRING → ACTIVE → IDLE_COUNTDOWN → RELEASING → IDLE
"""

import asyncio
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

log = logging.getLogger("omi-whisper.gpu")

# acquire_gpu の Lock 待機タイムアウト (秒)
_ACQUIRE_TIMEOUT = 600


class GpuState(Enum):
    IDLE = auto()           # GPU 空き、qwen-tts 稼働中
    ACQUIRING = auto()      # qwen-tts 停止 → Whisper ロード中
    ACTIVE = auto()         # 音声処理中 (session_count >= 1)
    IDLE_COUNTDOWN = auto() # 全セッション切断、タイマー動作中
    RELEASING = auto()      # Whisper アンロード → qwen-tts 起動中


class GpuManager:
    """GPU リソースの排他管理: Whisper モデルと qwen-tts コンテナの切り替え"""

    def __init__(
        self,
        whisper_model_name: str,
        compute_type: str,
        idle_timeout_minutes: float = 5.0,
        qwen_container: str = "qwen-tts-api-qwen-tts-api-1",
        docker_stop_timeout: int = 30,
    ):
        self._whisper_model_name = whisper_model_name
        self._compute_type = compute_type
        self._idle_timeout = idle_timeout_minutes * 60  # seconds
        self._qwen_container = qwen_container
        self._docker_stop_timeout = docker_stop_timeout

        self._state = GpuState.IDLE
        self._model = None
        self._executor: ThreadPoolExecutor | None = None
        self._session_count = 0
        self._lock = asyncio.Lock()
        self._idle_timer_task: asyncio.Task | None = None
        self._sessions_drained = asyncio.Event()
        self._sessions_drained.set()  # 初期状態はセッションなし

        log.info(
            f"GpuManager initialized: qwen_container={qwen_container}, "
            f"idle_timeout={idle_timeout_minutes}min, "
            f"docker_stop_timeout={docker_stop_timeout}s"
        )

    @property
    def state(self) -> GpuState:
        return self._state

    def get_model(self):
        """ロード済み Whisper モデルを返す (未ロード時は None)"""
        return self._model

    def get_executor(self) -> ThreadPoolExecutor | None:
        """GPU 推論直列化用エグゼキュータを返す"""
        return self._executor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire_gpu(self) -> tuple:
        """
        GPU を Whisper 用に確保する。

        - IDLE: qwen-tts 停止 → Whisper ロード
        - ACTIVE / IDLE_COUNTDOWN: セッション数インクリメントのみ
        - ACQUIRING / RELEASING: Lock で待機後に再確認

        タイムアウト (_ACQUIRE_TIMEOUT 秒) を超えたら TimeoutError。

        Returns: (model, executor)
        """
        if self._lock.locked():
            log.info(
                f"acquire_gpu: waiting for lock (state={self._state.name})... "
                f"another session is loading/releasing the model"
            )

        try:
            return await asyncio.wait_for(
                self._acquire_gpu_with_lock(), timeout=_ACQUIRE_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.error(f"acquire_gpu timed out after {_ACQUIRE_TIMEOUT}s")
            raise TimeoutError(
                f"GPU acquisition timed out after {_ACQUIRE_TIMEOUT}s"
            )

    async def _acquire_gpu_with_lock(self) -> tuple:
        """Lock を取得してから GPU 確保処理を実行"""
        async with self._lock:
            return await self._acquire_gpu_locked()

    async def _acquire_gpu_locked(self) -> tuple:
        """Lock 取得済みの状態で GPU 確保処理を実行"""
        self._cancel_idle_timer()
        self._session_count += 1
        self._sessions_drained.clear()
        log.info(f"acquire_gpu: session_count={self._session_count}, state={self._state.name}")

        if self._state in (GpuState.ACTIVE, GpuState.IDLE_COUNTDOWN):
            # モデルは既にロード済み
            self._state = GpuState.ACTIVE
            return self._model, self._executor

        # IDLE or post-RELEASING (モデル未ロード)
        self._state = GpuState.ACQUIRING
        log.info("State → ACQUIRING")

        try:
            await self._stop_qwen_tts()
            await self._load_whisper_model()
            self._state = GpuState.ACTIVE
            log.info("State → ACTIVE")
        except BaseException:
            # CancelledError (BaseException) も含めて状態を復元
            self._session_count -= 1
            if self._session_count == 0:
                self._state = GpuState.IDLE
            raise

        return self._model, self._executor

    def session_ended(self):
        """
        セッション終了を通知。全セッション切断でアイドルタイマー開始。
        """
        self._session_count = max(0, self._session_count - 1)
        log.info(f"session_ended: session_count={self._session_count}, state={self._state.name}")

        if self._session_count == 0:
            self._sessions_drained.set()
            if self._state == GpuState.ACTIVE:
                self._state = GpuState.IDLE_COUNTDOWN
                log.info(f"State → IDLE_COUNTDOWN (timeout={self._idle_timeout}s)")
                self._start_idle_timer()

    async def ensure_qwen_tts_running(self):
        """起動時に qwen-tts が稼働していることを保証する"""
        try:
            running = await self._is_qwen_tts_running()
            if not running:
                log.info("qwen-tts is not running at startup, starting it...")
                await self._start_qwen_tts()
            else:
                log.info("qwen-tts is already running")
        except Exception as e:
            log.warning(f"Could not check/start qwen-tts at startup: {e}")

    async def release_gpu_for_tts(self):
        """
        TTS 用に GPU を解放する。

        - IDLE: qwen-tts が動いていなければ起動
        - IDLE_COUNTDOWN: タイマーキャンセル → 即座に Whisper アンロード → qwen-tts 起動
        - ACTIVE: セッション終了を待ってから解放
        - ACQUIRING / RELEASING: Lock で待機後に再確認
        """
        log.info(f"release_gpu_for_tts: state={self._state.name}, sessions={self._session_count}")

        # ACTIVE の場合、Lock を取る前にセッション終了を待つ
        if self._state == GpuState.ACTIVE:
            log.info("release_gpu_for_tts: waiting for active sessions to drain...")
            self._sessions_drained.clear()
            await self._sessions_drained.wait()
            log.info("release_gpu_for_tts: sessions drained")

        async with self._lock:
            await self._release_gpu_for_tts_locked()

    async def _release_gpu_for_tts_locked(self):
        """Lock 取得済みの状態で TTS 用 GPU 解放処理を実行"""
        # Lock 待機中に状態が変わっている可能性があるため再チェック
        if self._state == GpuState.ACTIVE:
            # Lock 待機中に新しいセッションが来た場合
            log.info("release_gpu_for_tts: state became ACTIVE again, waiting for drain...")
            # Lock を解放してリトライさせるのではなく、ここでも待つ
            self._sessions_drained.clear()
            # Lock 内で長時間待つのは避けたいが、セッションが終わるのを待つ必要がある
            await self._sessions_drained.wait()

        if self._state == GpuState.IDLE:
            # Whisper 未ロード — qwen-tts が動いているか確認
            running = await self._is_qwen_tts_running()
            if not running:
                log.info("release_gpu_for_tts: IDLE but qwen-tts not running, starting it...")
                await self._start_qwen_tts()
            else:
                log.info("release_gpu_for_tts: IDLE and qwen-tts already running")
            return

        if self._state in (GpuState.IDLE_COUNTDOWN, GpuState.ACTIVE):
            # Whisper ロード済み、セッションなし → 即座に解放
            self._cancel_idle_timer()
            self._state = GpuState.RELEASING
            log.info("State → RELEASING (TTS requested)")

            try:
                self._unload_whisper_model()
                await self._start_qwen_tts()
                self._state = GpuState.IDLE
                log.info("State → IDLE (released for TTS)")
            except Exception as e:
                log.error(f"Error during GPU release for TTS: {e}", exc_info=True)
                self._state = GpuState.IDLE
            return

        # ACQUIRING / RELEASING — Lock で待機した後ここに来る場合
        # 既に遷移済みなので何もしない (IDLE であれば上で処理済み)
        log.info(f"release_gpu_for_tts: state={self._state.name}, no action needed")

    async def wait_for_qwen_tts_healthy(self, timeout: float = 600) -> bool:
        """
        qwen-tts コンテナが healthy になるまでポーリングで待機。

        Returns: True if healthy, False if timed out
        """
        log.info(f"Waiting for qwen-tts to become healthy (timeout={timeout}s)...")
        deadline = asyncio.get_event_loop().time() + timeout
        poll_interval = 3.0

        while asyncio.get_event_loop().time() < deadline:
            # STT リクエストが来て qwen-tts が停止された場合は中断
            if self._state == GpuState.ACQUIRING:
                log.warning("wait_for_qwen_tts_healthy: GPU is being acquired for STT, aborting")
                return False

            try:
                rc, stdout, stderr = await self._run_docker(
                    "inspect", "-f", "{{.State.Health.Status}}", self._qwen_container,
                    timeout=10,
                )
                status = stdout.strip().lower()
                if status == "healthy":
                    log.info("qwen-tts is healthy")
                    return True
                log.debug(f"qwen-tts health status: {status}")
            except Exception as e:
                log.debug(f"Health check failed: {e}")

            await asyncio.sleep(poll_interval)

        log.error(f"qwen-tts did not become healthy within {timeout}s")
        return False

    # ------------------------------------------------------------------
    # Idle timer
    # ------------------------------------------------------------------

    def _start_idle_timer(self):
        self._cancel_idle_timer()
        self._idle_timer_task = asyncio.ensure_future(self._idle_timer_coro())

    def _cancel_idle_timer(self):
        if self._idle_timer_task is not None and not self._idle_timer_task.done():
            self._idle_timer_task.cancel()
            log.debug("Idle timer cancelled")
        self._idle_timer_task = None

    async def _idle_timer_coro(self):
        """アイドルタイムアウト後に GPU を解放"""
        try:
            await asyncio.sleep(self._idle_timeout)
        except asyncio.CancelledError:
            return

        async with self._lock:
            if self._state != GpuState.IDLE_COUNTDOWN or self._session_count > 0:
                return

            self._state = GpuState.RELEASING
            log.info("State → RELEASING (idle timeout expired)")

            try:
                self._unload_whisper_model()
                await self._start_qwen_tts()
                self._state = GpuState.IDLE
                log.info("State → IDLE")
            except Exception as e:
                log.error(f"Error during GPU release: {e}", exc_info=True)
                self._state = GpuState.IDLE

    # ------------------------------------------------------------------
    # Whisper model management
    # ------------------------------------------------------------------

    async def _load_whisper_model(self):
        """Whisper モデルを GPU にロード (ブロッキング処理をスレッドで実行)"""
        log.info(f"Loading Whisper model: {self._whisper_model_name} ({self._compute_type})")

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model_sync)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

        log.info("Whisper model loaded successfully")

    def _load_model_sync(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._whisper_model_name,
            device="cuda",
            compute_type=self._compute_type,
        )

    def _unload_whisper_model(self):
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

    # ------------------------------------------------------------------
    # Docker control for qwen-tts (非同期サブプロセス)
    # ------------------------------------------------------------------

    async def _run_docker(self, *args: str, timeout: float = 30) -> tuple[int, str, str]:
        """
        docker コマンドを非同期で実行する。

        Returns: (returncode, stdout, stderr)
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return (
                proc.returncode,
                stdout.decode().strip(),
                stderr.decode().strip(),
            )
        except asyncio.TimeoutError:
            log.error(f"docker {' '.join(args)}: timed out after {timeout}s")
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise
        except FileNotFoundError:
            log.error("docker command not found — is docker.io installed in the container?")
            raise

    async def _is_qwen_tts_running(self) -> bool:
        """qwen-tts コンテナが稼働中かチェック"""
        try:
            rc, stdout, stderr = await self._run_docker(
                "inspect", "-f", "{{.State.Running}}", self._qwen_container,
                timeout=10,
            )
            return stdout.lower() == "true"
        except Exception as e:
            log.warning(f"Failed to inspect qwen-tts container: {e}")
            return False

    async def _stop_qwen_tts(self):
        """qwen-tts コンテナを停止。既に停止済みなら何もしない。"""
        running = await self._is_qwen_tts_running()

        if not running:
            log.info(f"qwen-tts container '{self._qwen_container}' is not running, skip stop")
            return

        log.info(f"Stopping qwen-tts container '{self._qwen_container}'...")
        try:
            rc, stdout, stderr = await self._run_docker(
                "stop", "-t", str(self._docker_stop_timeout), self._qwen_container,
                timeout=self._docker_stop_timeout + 15,
            )
            if rc == 0:
                log.info("qwen-tts container stopped")
            else:
                # "is not running" はエラーではない
                if "is not running" in stderr:
                    log.info("qwen-tts was already stopped")
                else:
                    log.warning(f"docker stop returned {rc}: {stderr}")
        except asyncio.TimeoutError:
            log.warning("docker stop timed out, proceeding anyway")
        except Exception as e:
            log.error(f"Failed to stop qwen-tts: {e}")
            # 停止失敗しても Whisper ロードは試みる (VRAM 不足なら CUDA エラーで捕まる)

    async def _start_qwen_tts(self):
        """qwen-tts コンテナを起動。既に稼働中なら何もしない。"""
        running = await self._is_qwen_tts_running()

        if running:
            log.info(f"qwen-tts container '{self._qwen_container}' is already running")
            return

        log.info(f"Starting qwen-tts container '{self._qwen_container}'...")
        try:
            rc, stdout, stderr = await self._run_docker(
                "start", self._qwen_container,
                timeout=30,
            )
            if rc == 0:
                log.info("qwen-tts container started")
            else:
                # コンテナが存在しない場合など
                if "No such container" in stderr:
                    log.warning(
                        f"Container '{self._qwen_container}' not found. "
                        f"Is the container name correct?"
                    )
                else:
                    log.warning(f"docker start returned {rc}: {stderr}")
        except asyncio.TimeoutError:
            log.warning("docker start timed out")
        except Exception as e:
            log.error(f"Failed to start qwen-tts: {e}")
