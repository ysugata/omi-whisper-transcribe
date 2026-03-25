"""
Omi × faster-whisper ブリッジアプリ

Omi デバイスから WebSocket 経由で受信した音声を
ローカルの faster-whisper で文字起こしし、
Omi バックエンドが期待する JSON 形式で返します。

音声保存機能:
  セッションごとに WAV ファイルと文字起こし JSON を保存。
  保存先: ./recordings/YYYY-MM-DD/HH-MM-SS/
  SAVE_AUDIO = False で無効化可能。

使い方:
  python omi_whisper_bridge.py

必要パッケージ:
  pip install faster-whisper websockets numpy
"""

import asyncio
import hmac
import json
import logging
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import websockets

# ============================================================
# 設定 — ここを環境に合わせて変更
# ============================================================

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")              # tiny, base, small, medium, large-v3
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8_float16")            # float16, int8_float16, int8
_lang = os.environ.get("LANGUAGE", "ja")
LANGUAGE = None if _lang.lower() == "auto" else _lang                    # "ja", "en", or "auto" (自動検出)
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "5"))
VAD_FILTER = os.environ.get("VAD_FILTER", "true").lower() in ("true", "1", "yes")  # 無音区間のハルシネーション対策
BUFFER_SECONDS = float(os.environ.get("BUFFER_SECONDS", "3.0"))          # 音声をバッファしてからまとめて処理する秒数
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))                # Omi は 16kHz PCM を送信
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8090"))
MANAGEMENT_PORT = int(os.environ.get("MANAGEMENT_PORT", "8091"))

# --- 録音保存設定 ---
SAVE_AUDIO = os.environ.get("SAVE_AUDIO", "true").lower() in ("true", "1", "yes")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "./recordings")        # 保存先ディレクトリ
MAX_RECORDING_DAYS = int(os.environ.get("MAX_RECORDING_DAYS", "30"))     # 自動削除: この日数より古い録音を起動時に削除 (0=無効)

# --- 無音検出設定 ---
SILENCE_THRESHOLD = int(os.environ.get("SILENCE_THRESHOLD", "200"))      # int16 RMS 閾値 (0=無効)
SILENCE_TIMEOUT = float(os.environ.get("SILENCE_TIMEOUT", "30.0"))       # 無音タイムアウト秒数

# --- 認証設定 (Cloudflare Tunnel 経由で公開する場合) ---
AUTH_TOKEN = os.environ.get("OMI_WHISPER_TOKEN", "")                     # 空文字 = 認証なし (LAN内専用)
# トークン生成例: python -c "import secrets; print(secrets.token_urlsafe(48))"

# ============================================================
# ログ設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("omi-whisper")

# ============================================================
# モデルマネージャ (Whisper モデルのライフサイクル管理)
# ============================================================

from model_manager import ModelManager

model_manager = ModelManager(
    model_name=WHISPER_MODEL,
    compute_type=COMPUTE_TYPE,
)

# ============================================================
# 古い録音の自動クリーンアップ
# ============================================================

def cleanup_old_recordings():
    """MAX_RECORDING_DAYS より古い日付ディレクトリを削除"""
    if MAX_RECORDING_DAYS <= 0 or not SAVE_AUDIO:
        return

    import shutil
    base = Path(RECORDINGS_DIR)
    if not base.exists():
        return

    cutoff = datetime.now().timestamp() - (MAX_RECORDING_DAYS * 86400)
    removed = 0

    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        try:
            day_date = datetime.strptime(day_dir.name, "%Y-%m-%d")
            if day_date.timestamp() < cutoff:
                shutil.rmtree(day_dir)
                removed += 1
        except ValueError:
            continue

    if removed:
        log.info(f"Cleaned up {removed} old recording dir(s) (>{MAX_RECORDING_DAYS} days)")


# ============================================================
# セッション録音マネージャ
# ============================================================

class SessionRecorder:
    """1セッション分の音声と文字起こしを管理・保存する"""

    def __init__(self):
        self.enabled = SAVE_AUDIO
        self.all_segments: list[dict] = []
        self.total_bytes = 0
        self._finalized = False

        if not self.enabled:
            return

        now = datetime.now()
        self.session_dir = Path(RECORDINGS_DIR) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.wav_path = self.session_dir / "audio.wav"
        self.json_path = self.session_dir / "transcript.json"

        # WAV ファイルを開く (PCM 16-bit, 16kHz, mono)
        self.wav_file = wave.open(str(self.wav_path), "wb")
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        self.wav_file.setframerate(SAMPLE_RATE)

        self.start_time = now

        log.info(f"Recording to: {self.session_dir}")

    def write_audio(self, pcm_bytes: bytes):
        """PCM 音声データを WAV に追記"""
        if not self.enabled:
            return
        self.wav_file.writeframes(pcm_bytes)
        self.total_bytes += len(pcm_bytes)

    def add_segments(self, segments: list[dict]):
        """文字起こしセグメントを蓄積"""
        if not self.enabled:
            return
        self.all_segments.extend(segments)

    def finalize(self):
        """セッション終了時に WAV を閉じ、JSON を書き出す"""
        if not self.enabled or self._finalized:
            return
        self._finalized = True

        self.wav_file.close()

        duration_sec = self.total_bytes / (SAMPLE_RATE * 2)
        file_size_mb = self.total_bytes / (1024 * 1024)

        # メタデータ付きで JSON 保存
        output = {
            "session": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_seconds": round(duration_sec, 1),
                "audio_file": "audio.wav",
                "audio_size_mb": round(file_size_mb, 1),
                "model": WHISPER_MODEL,
                "language": LANGUAGE,
            },
            "segments": self.all_segments,
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        log.info(
            f"Saved: {self.wav_path.name} ({file_size_mb:.1f}MB, {duration_sec:.0f}s) "
            f"+ {self.json_path.name} ({len(self.all_segments)} segments)"
        )

        # 空の録音 (音声なし) なら削除
        if self.total_bytes == 0:
            import shutil
            shutil.rmtree(self.session_dir, ignore_errors=True)
            log.info("Empty session removed.")


# ============================================================
# 音声バッファ → 文字起こし
# ============================================================

def calc_rms_int16(pcm_bytes: bytes) -> float:
    """PCM int16 バイト列の RMS (Root Mean Square) を計算"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def transcribe_audio(audio_bytes: bytes, model, session_offset: float = 0.0) -> list[dict]:
    """
    PCM 16-bit LE の音声バイト列を受け取り、
    Omi の custom_stt が期待する segments リストを返す。
    model: ロード済みの WhisperModel インスタンス
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < 1600:  # 0.1秒未満は無視
        return []

    segments_out = []

    segments, info = model.transcribe(
        audio_np,
        beam_size=BEAM_SIZE,
        language=LANGUAGE,
        vad_filter=VAD_FILTER,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
    )

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        segments_out.append({
            "text": text,
            "speaker": "SPEAKER_00",
            "speaker_id": 0,
            "start": round(seg.start + session_offset, 2),
            "end": round(seg.end + session_offset, 2),
            "is_user": True,
            "person_id": None,
        })

    return segments_out


# ============================================================
# WebSocket ハンドラ
# ============================================================

async def handle_connection(websocket):
    """
    1つの WebSocket 接続 = 1つの Omi セッション。
    バイナリ音声チャンクを受信し、バッファが溜まったら
    faster-whisper で文字起こしして JSON を返す。
    音声とテキストをセッションごとに保存する。
    """
    remote = websocket.remote_address

    # --- トークン認証 ---
    if AUTH_TOKEN:
        ws_path = websocket.request.path if hasattr(websocket, 'request') else websocket.path
        params = parse_qs(urlparse(ws_path).query)
        token = params.get("token", [None])[0]
        if not token or not hmac.compare_digest(token, AUTH_TOKEN):
            log.warning(f"Rejected: invalid token from {remote}")
            await websocket.close(4401, "Unauthorized")
            return
        log.info(f"Authenticated connection from {remote}")
    else:
        log.info(f"Connection from {remote} (no auth)")

    audio_buffer = bytearray()
    session_start = time.time()
    total_audio_seconds = 0.0
    recorder = SessionRecorder()
    silent_since = None  # 無音開始時刻 (None = 発話中)

    # モデル確保 (未ロードなら自動ロード)
    try:
        model, executor = await model_manager.acquire()
    except Exception as e:
        log.error(f"Failed to acquire model: {e}", exc_info=True)
        await websocket.close(1013, "Model unavailable")
        return

    try:
        async for message in websocket:
            # Keep-alive ping (2バイト以下)
            if isinstance(message, bytes) and len(message) <= 2:
                continue

            # テキストメッセージ (JSON コマンド) はログのみ
            if isinstance(message, str):
                log.debug(f"Text message: {message[:200]}")
                try:
                    data = json.loads(message)
                    log.info(f"Command: {data.get('type', 'unknown')}")
                except json.JSONDecodeError:
                    pass
                continue

            # バイナリ = 音声データ
            audio_buffer.extend(message)

            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * 2)

            # バッファが指定秒数に達したら処理
            if buffer_duration >= BUFFER_SECONDS:
                chunk = bytes(audio_buffer)
                audio_buffer.clear()

                # --- 無音検出 ---
                if SILENCE_THRESHOLD > 0:
                    rms = calc_rms_int16(chunk)

                    if rms < SILENCE_THRESHOLD:
                        # 無音バッファ
                        if silent_since is None:
                            silent_since = time.time()
                            log.debug(f"Silence started (RMS={rms:.0f})")

                        elapsed_silence = time.time() - silent_since

                        # タイムアウト → 録音セッション自動終了
                        if elapsed_silence >= SILENCE_TIMEOUT and recorder is not None:
                            recorder.finalize()
                            log.info(
                                f"Session finalized due to silence timeout "
                                f"({elapsed_silence:.0f}s silent)"
                            )
                            recorder = None

                        total_audio_seconds += buffer_duration
                        log.debug(
                            f"Silent buffer skipped (RMS={rms:.0f}, "
                            f"silent={elapsed_silence:.0f}s)"
                        )
                        continue

                    # 発話あり → silent_since リセット
                    if silent_since is not None:
                        log.info(
                            f"Speech resumed after "
                            f"{time.time() - silent_since:.0f}s of silence (RMS={rms:.0f})"
                        )
                        silent_since = None

                    # 録音セッションが終了していたら新規作成
                    if recorder is None:
                        recorder = SessionRecorder()
                        log.info("New recording session started (speech detected)")

                # 音声を録音ファイルに書き込み
                if recorder is not None:
                    recorder.write_audio(chunk)

                log.info(f"Transcribing {buffer_duration:.1f}s of audio...")
                t0 = time.time()

                loop = asyncio.get_running_loop()
                segments = await loop.run_in_executor(
                    executor,
                    transcribe_audio,
                    chunk,
                    model,
                    total_audio_seconds,
                )

                elapsed = time.time() - t0
                total_audio_seconds += buffer_duration

                if segments:
                    response = {"segments": segments}
                    await websocket.send(json.dumps(response))

                    # 文字起こし結果を録音ログに追加
                    if recorder is not None:
                        recorder.add_segments(segments)

                    log.info(
                        f"Sent {len(segments)} segment(s) "
                        f"[{elapsed:.2f}s processing, "
                        f"RTF={elapsed/buffer_duration:.2f}x]"
                    )
                else:
                    log.debug(f"No speech detected in buffer ({elapsed:.2f}s)")

    except websockets.exceptions.ConnectionClosed:
        log.info(f"Connection closed: {remote}")
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
    finally:
        # 残りバッファの処理
        if len(audio_buffer) > SAMPLE_RATE:
            # 無音検出が有効なら残りバッファもチェック
            remaining = bytes(audio_buffer)
            is_speech = (SILENCE_THRESHOLD <= 0 or
                         calc_rms_int16(remaining) >= SILENCE_THRESHOLD)

            if is_speech:
                if recorder is None:
                    recorder = SessionRecorder()
                recorder.write_audio(remaining)

                log.info("Processing remaining buffer...")
                try:
                    segments = transcribe_audio(remaining, model, total_audio_seconds)
                    if segments:
                        recorder.add_segments(segments)
                        try:
                            response = {"segments": segments}
                            await websocket.send(json.dumps(response))
                        except Exception:
                            pass
                except Exception as e:
                    log.error(f"Error processing remaining buffer: {e}", exc_info=True)

        # セッション録音を確定 (WAV クローズ + JSON 書き出し)
        if recorder is not None:
            recorder.finalize()

        duration = time.time() - session_start
        log.info(
            f"Session ended: {remote} "
            f"(duration={duration:.0f}s, audio={total_audio_seconds:.0f}s)"
        )

        # セッション終了通知
        model_manager.release()


# ============================================================
# メイン
# ============================================================

async def main():
    log.info(f"Starting Omi-Whisper bridge on ws://{HOST}:{PORT}")
    log.info(f"Model: {WHISPER_MODEL}, Compute: {COMPUTE_TYPE}, Language: {LANGUAGE}")
    log.info(f"Buffer: {BUFFER_SECONDS}s, VAD: {VAD_FILTER}, Beam: {BEAM_SIZE}")
    log.info(f"Audio saving: {'ON -> ' + RECORDINGS_DIR if SAVE_AUDIO else 'OFF'}")
    log.info(f"Silence: threshold={SILENCE_THRESHOLD}, timeout={SILENCE_TIMEOUT}s"
             f"{' (DISABLED)' if SILENCE_THRESHOLD <= 0 else ''}")
    log.info(f"Auth: {'TOKEN required' if AUTH_TOKEN else 'DISABLED (LAN only)'}")
    log.info(f"Management API: http://0.0.0.0:{MANAGEMENT_PORT}")
    log.info("Model will be loaded on-demand (GPU not used until first connection)")

    # 起動時に古い録音を削除
    cleanup_old_recordings()

    # 管理 REST API を起動
    from aiohttp import web

    async def handle_model_load(request):
        await model_manager.load_model()
        return web.json_response({"status": "loaded"})

    async def handle_model_unload(request):
        ok = await model_manager.unload_model()
        if not ok:
            return web.json_response(
                {"error": "active sessions", "session_count": model_manager.session_count},
                status=409,
            )
        return web.json_response({"status": "unloaded"})

    async def handle_model_status(request):
        return web.json_response(model_manager.status())

    mgmt_app = web.Application()
    mgmt_app.router.add_post("/model/load", handle_model_load)
    mgmt_app.router.add_post("/model/unload", handle_model_unload)
    mgmt_app.router.add_get("/model/status", handle_model_status)

    mgmt_runner = web.AppRunner(mgmt_app)
    await mgmt_runner.setup()
    mgmt_site = web.TCPSite(mgmt_runner, "0.0.0.0", MANAGEMENT_PORT)
    await mgmt_site.start()
    log.info(f"Management API listening on http://0.0.0.0:{MANAGEMENT_PORT}")

    async with websockets.serve(
        handle_connection,
        HOST,
        PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=10,
    ):
        log.info(f"Listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
