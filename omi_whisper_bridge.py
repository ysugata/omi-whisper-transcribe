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
import re
import subprocess
import time
import uuid
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

# --- Whisper 精度改善パラメータ ---
CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")                       # 設定ファイルディレクトリ
INITIAL_PROMPT = os.environ.get("INITIAL_PROMPT", "")                     # 固有名詞等を含むプロンプト (空=なし, 設定ファイル優先)
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "3"))   # 繰り返し防止 n-gram サイズ (0=無効)
LOG_PROB_THRESHOLD = float(os.environ.get("LOG_PROB_THRESHOLD", "-0.5"))  # セグメント平均対数確率の閾値
NO_SPEECH_THRESHOLD = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6")) # 無音確率の閾値
COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("COMPRESSION_RATIO_THRESHOLD", "2.4"))  # 圧縮率閾値

# --- Layer 3: LLM 後処理設定 ---
LLM_ENABLED = os.environ.get("LLM_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "10"))
LLM_MIN_TEXT_LENGTH = int(os.environ.get("LLM_MIN_TEXT_LENGTH", "5"))

# --- 話者分離設定 ---
DIARIZATION_ENABLED = os.environ.get("DIARIZATION_ENABLED", "false").lower() in ("true", "1", "yes")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN", "")

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
# Layer 1 + Layer 2: 固有名詞改善
# ============================================================

from prompt_builder import build_prompt
from term_corrector import TermCorrector
from llm_corrector import LLMCorrector
from diarizer import Diarizer

_PROMPT_SETTINGS_PATH = os.path.join(CONFIG_DIR, "prompt_settings.json")
_TERM_CORRECTIONS_PATH = os.path.join(CONFIG_DIR, "term_corrections.json")

term_corrector = TermCorrector(_TERM_CORRECTIONS_PATH)

# Layer 3: LLM 後処理 (API キー未設定なら自動的に無効)
llm_corrector: LLMCorrector | None = None
if LLM_ENABLED and LLM_API_KEY:
    llm_corrector = LLMCorrector(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        timeout=LLM_TIMEOUT,
    )

# 話者分離 (HF_AUTH_TOKEN があれば常に利用可能)
diarizer: Diarizer | None = None
if HF_AUTH_TOKEN:
    diarizer = Diarizer(
        hf_token=HF_AUTH_TOKEN,
        language=LANGUAGE or "ja",
    )


def _get_initial_prompt() -> str:
    """セッション開始時に initial_prompt を決定する。

    優先順位:
    1. prompt_settings.json がある場合、その内容から構築
    2. INITIAL_PROMPT 環境変数
    3. デフォルトの短いプロンプト
    """
    try:
        if os.path.exists(_PROMPT_SETTINGS_PATH):
            with open(_PROMPT_SETTINGS_PATH, encoding="utf-8") as f:
                settings = json.load(f)
            prompt = build_prompt(
                topic=settings.get("topic", ""),
                participants=settings.get("participants", []),
                terms=settings.get("terms", []),
            )
            if prompt:
                log.info(f"Prompt from settings: {prompt[:80]}...")
                return prompt
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"prompt_settings.json 読み込みエラー: {e}")

    if INITIAL_PROMPT:
        return INITIAL_PROMPT

    return ""


def _get_known_terms() -> list[str]:
    """LLM 補正用の参考用語リストを構築する。

    prompt_settings の terms + term_corrector の正解語をマージ。
    """
    terms: set[str] = set()

    # term_corrector の正解語 (辞書の values)
    for v in term_corrector.list_corrections().values():
        terms.add(v)

    # prompt_settings の terms
    try:
        if os.path.exists(_PROMPT_SETTINGS_PATH):
            with open(_PROMPT_SETTINGS_PATH, encoding="utf-8") as f:
                settings = json.load(f)
            for t in settings.get("terms", []):
                terms.add(t)
    except (json.JSONDecodeError, OSError):
        pass

    return sorted(terms)


_PENDING_SUGGESTIONS_PATH = os.path.join(CONFIG_DIR, "pending_suggestions.json")


def _save_llm_suggestions(originals: list[str], corrected_segments: list[dict]) -> None:
    """LLM が修正した差分を pending_suggestions.json に追記する。"""
    from llm_corrector import LLMCorrector

    new_suggestions = []
    for orig_text, seg in zip(originals, corrected_segments):
        corrected_text = seg.get("text", "")
        diffs = LLMCorrector.detect_corrections(orig_text, corrected_text)
        for d in diffs:
            new_suggestions.append({
                "id": str(uuid.uuid4())[:8],
                "wrong": d["wrong"],
                "correct": d["correct"],
                "context": orig_text,
                "detected_at": datetime.now().isoformat(),
                "status": "pending",
            })

    if not new_suggestions:
        return

    # 既存の提案を読み込み
    try:
        if os.path.exists(_PENDING_SUGGESTIONS_PATH):
            with open(_PENDING_SUGGESTIONS_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
    except (json.JSONDecodeError, OSError):
        existing = []

    # 重複チェック: 同じ wrong+correct ペアが pending にあればスキップ
    existing_pairs = {(s["wrong"], s["correct"]) for s in existing if s.get("status") == "pending"}
    added = 0
    for s in new_suggestions:
        pair = (s["wrong"], s["correct"])
        if pair not in existing_pairs:
            existing.append(s)
            existing_pairs.add(pair)
            added += 1

    if added > 0:
        os.makedirs(os.path.dirname(_PENDING_SUGGESTIONS_PATH) or ".", exist_ok=True)
        with open(_PENDING_SUGGESTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {added} new LLM suggestion(s) to pending_suggestions.json")


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
        self._audio_closed = False

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

    def close_audio(self):
        """WAV ファイルを閉じる (diarization 前に呼ぶ)"""
        if not self.enabled or self._audio_closed:
            return
        self.wav_file.close()
        self._audio_closed = True

    def _convert_to_opus(self) -> bool:
        """WAV → OGG Opus に変換し、成功したら WAV を削除"""
        ogg_path = self.session_dir / "audio.ogg"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(self.wav_path),
                    "-c:a", "libopus",
                    "-b:a", "32k",        # Omi と同じ 32kbps
                    "-ar", "16000",
                    "-ac", "1",
                    str(ogg_path),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log.error(f"ffmpeg failed: {result.stderr[:500]}")
                return False
            # 変換成功 → WAV 削除
            self.wav_path.unlink(missing_ok=True)
            return True
        except FileNotFoundError:
            log.warning("ffmpeg not found, keeping WAV")
            return False
        except subprocess.TimeoutExpired:
            log.error("ffmpeg timed out")
            return False

    def finalize(self):
        """セッション終了時に WAV を閉じ、OGG Opus に変換し、JSON を書き出す"""
        if not self.enabled or self._finalized:
            return
        self._finalized = True

        if not self._audio_closed:
            self.wav_file.close()
            self._audio_closed = True

        duration_sec = self.total_bytes / (SAMPLE_RATE * 2)

        # 空の録音 (音声なし) なら削除
        if self.total_bytes == 0:
            import shutil
            shutil.rmtree(self.session_dir, ignore_errors=True)
            log.info("Empty session removed.")
            return

        # WAV → OGG Opus 変換
        if self._convert_to_opus():
            audio_file = "audio.ogg"
            audio_path = self.session_dir / audio_file
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        else:
            audio_file = "audio.wav"
            file_size_mb = self.total_bytes / (1024 * 1024)

        # 話者数を集計
        speakers = set(s.get("speaker", "SPEAKER_00") for s in self.all_segments)
        num_speakers = len(speakers)

        # メタデータ付きで JSON 保存
        output = {
            "session": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_seconds": round(duration_sec, 1),
                "audio_file": audio_file,
                "audio_size_mb": round(file_size_mb, 2),
                "model": WHISPER_MODEL,
                "language": LANGUAGE,
                "num_speakers": num_speakers,
            },
            "segments": self.all_segments,
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        log.info(
            f"Saved: {audio_file} ({file_size_mb:.2f}MB, {duration_sec:.0f}s) "
            f"+ {self.json_path.name} ({len(self.all_segments)} segments)"
        )


# ============================================================
# 音声バッファ → 文字起こし
# ============================================================

def calc_rms_int16(pcm_bytes: bytes) -> float:
    """PCM int16 バイト列の RMS (Root Mean Square) を計算"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


# ハルシネーション除去用パターン (Whisper が生成しがちな定型句)
_HALLUCINATION_PATTERNS = [
    re.compile(r"ご視聴ありがとうございま(した|す)"),
    re.compile(r"チャンネル登録.*?お願いします"),
    re.compile(r"チャンネル登録"),
    re.compile(r"高評価"),
    re.compile(r"ご清聴ありがとうございま(した|す)"),
    re.compile(r"お(つかれさま|疲れ様)でした"),
    re.compile(r"(ではまた|また次回)"),
    re.compile(r"最後まで(ご覧|見て)いただき.*"),
    re.compile(r"ご覧いただきありがとうございま(した|す)"),
    re.compile(r"おやすみなさい"),
    re.compile(r"字幕.*?(視聴者|翻訳|提供)"),
    re.compile(r"Thanks? for watching\.?", re.IGNORECASE),
    re.compile(r"See you next time\.?", re.IGNORECASE),
    re.compile(r"Subscribe", re.IGNORECASE),
    re.compile(r"Bye[\.\s]*bye\.?", re.IGNORECASE),
    re.compile(r"♪+"),
    re.compile(r"^\s*\.+\s*$"),
    re.compile(r"(.{2,}?)\1{3,}"),  # 同一フレーズの4回以上繰り返し
]


def _remove_hallucinations(text: str) -> str:
    """既知のハルシネーションパターンを除去し、残りテキストを返す"""
    cleaned = text
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# プロンプトエコー検出用の区切り文字パターン
_PROMPT_SPLIT_RE = re.compile(r"[、。,.\s:：]+")


def _is_prompt_echo(text: str, prompt: str, threshold: float = 0.6) -> bool:
    """テキストが initial_prompt のエコー (プロンプト内容の誤出力) かどうかを判定する。

    テキスト中の単語の60%以上がプロンプトに含まれていればエコーと判定。
    """
    if not prompt:
        return False
    text_words = set(w for w in _PROMPT_SPLIT_RE.split(text) if len(w) >= 2)
    if not text_words:
        return False
    prompt_words = set(w for w in _PROMPT_SPLIT_RE.split(prompt) if len(w) >= 2)
    overlap = len(text_words & prompt_words)
    ratio = overlap / len(text_words)
    if ratio >= threshold:
        log.debug(f"Prompt echo detected ({ratio:.0%}): '{text[:60]}'")
        return True
    return False


def transcribe_audio(
    audio_bytes: bytes,
    model,
    session_offset: float = 0.0,
    initial_prompt: str = "",
    known_terms: list[str] | None = None,
) -> list[dict]:
    """PCM 16-bit LE の音声バイト列を受け取り、
    Omi の custom_stt が期待する segments リストを返す。

    Args:
        audio_bytes: PCM 16-bit LE 音声データ
        model: ロード済みの WhisperModel インスタンス
        session_offset: タイムスタンプオフセット (秒)
        initial_prompt: Layer 1 で構築した initial_prompt
        known_terms: Layer 3 LLM 補正用の参考用語リスト
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < 1600:  # 0.1秒未満は無視
        return []

    segments_out = []

    # transcribe パラメータ構築
    transcribe_kwargs = dict(
        beam_size=BEAM_SIZE,
        language=LANGUAGE,
        vad_filter=VAD_FILTER,
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=1000,
            speech_pad_ms=400,
            threshold=0.5,
        ),
        word_timestamps=False,
        condition_on_previous_text=True,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    segments, info = model.transcribe(audio_np, **transcribe_kwargs)

    for seg in segments:
        # 確信度フィルタ: 低確信度・無音確率高のセグメントを除去
        if seg.avg_logprob < -0.7 or seg.no_speech_prob > 0.5:
            log.debug(
                f"Segment filtered: avg_logprob={seg.avg_logprob:.2f}, "
                f"no_speech_prob={seg.no_speech_prob:.2f}, text='{seg.text.strip()[:50]}'"
            )
            continue

        text = seg.text.strip()
        if not text:
            continue

        # ハルシネーション除去
        text = _remove_hallucinations(text)
        if not text:
            continue

        # プロンプトエコー検出
        if initial_prompt and _is_prompt_echo(text, initial_prompt):
            continue

        # Layer 2: 用語補正
        text = term_corrector.correct(text)

        segments_out.append({
            "text": text,
            "speaker": "SPEAKER_00",
            "speaker_id": 0,
            "start": round(seg.start + session_offset, 2),
            "end": round(seg.end + session_offset, 2),
            "is_user": True,
            "person_id": None,
        })

    # Layer 3: LLM 後処理
    if llm_corrector and segments_out:
        originals = [seg["text"] for seg in segments_out]
        segments_out = llm_corrector.correct_segments(
            segments_out,
            known_terms=known_terms,
            min_text_length=LLM_MIN_TEXT_LENGTH,
        )
        try:
            _save_llm_suggestions(originals, segments_out)
        except Exception as e:
            log.warning(f"Failed to save LLM suggestions: {e}")

    return segments_out


def _offset_segments(segments: list[dict], offset: float) -> list[dict]:
    """セグメントのタイムスタンプにオフセットを加算したコピーを返す"""
    if offset == 0:
        return segments
    return [
        {**seg, "start": round(seg["start"] + offset, 2),
                "end": round(seg["end"] + offset, 2)}
        for seg in segments
    ]


# ============================================================
# バックグラウンド話者分離
# ============================================================

async def _run_diarization_bg(recorder: SessionRecorder) -> None:
    """セッション終了後にバックグラウンドで話者分離 → finalize を実行する。

    gpu_lock を取得してから実行するため、diarization 中に新セッションの
    acquire() が呼ばれても、diarization 完了まで待機する。
    """
    try:
        async with model_manager.gpu_lock:
            unloaded = await model_manager.unload_model()
            if not unloaded:
                log.info("Skipping diarization (other sessions active)")
                recorder.finalize()
                return

            log.info("Running post-session diarization (background)...")
            loop = asyncio.get_running_loop()
            updated = await loop.run_in_executor(
                None,
                lambda: diarizer.diarize(
                    str(recorder.wav_path), recorder.all_segments
                ),
            )
            recorder.all_segments = updated
    except Exception as e:
        log.error(f"Diarization error: {e}", exc_info=True)
    finally:
        recorder.finalize()


# ============================================================
# バッチ話者分離
# ============================================================

_batch_state = {
    "running": False,
    "total": 0,
    "processed": 0,
    "failed": 0,
    "skipped": 0,
    "current_session": None,
    "errors": [],
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
}


def _find_undiarized_sessions() -> list[dict]:
    """RECORDINGS_DIR 配下の未 diarize セッションを古い順に返す。"""
    base = Path(RECORDINGS_DIR)
    if not base.exists():
        return []

    sessions = []
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        for time_dir in sorted(day_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            session_info = data.get("session", {})
            if session_info.get("diarized"):
                continue

            # 音声ファイル特定
            audio_file = session_info.get("audio_file", "audio.ogg")
            audio_path = time_dir / audio_file
            if not audio_path.exists():
                # fallback
                for ext in ("audio.ogg", "audio.wav"):
                    alt = time_dir / ext
                    if alt.exists():
                        audio_path = alt
                        break
                else:
                    continue

            segments = data.get("segments", [])
            if not segments:
                continue

            sessions.append({
                "path": f"{day_dir.name}/{time_dir.name}",
                "audio": audio_path,
                "transcript_path": transcript_path,
                "data": data,
            })

    return sessions


async def _batch_diarize_task():
    """バッチ話者分離タスク。asyncio.create_task で起動する。"""
    _batch_state["running"] = True
    _batch_state["processed"] = 0
    _batch_state["failed"] = 0
    _batch_state["skipped"] = 0
    _batch_state["current_session"] = None
    _batch_state["errors"] = []
    _batch_state["cancel_requested"] = False
    _batch_state["started_at"] = datetime.now().isoformat()
    _batch_state["finished_at"] = None

    try:
        sessions = _find_undiarized_sessions()
        _batch_state["total"] = len(sessions)
        log.info(f"Batch diarization: {len(sessions)} session(s) to process")

        if not sessions:
            return

        loop = asyncio.get_running_loop()

        for sess in sessions:
            if _batch_state["cancel_requested"]:
                log.info("Batch diarization cancelled by user")
                break

            _batch_state["current_session"] = sess["path"]
            log.info(f"Batch diarize: {sess['path']}")

            try:
                async with model_manager.gpu_lock:
                    await model_manager.unload_model()

                    updated = await loop.run_in_executor(
                        None,
                        lambda s=sess: diarizer.diarize(
                            str(s["audio"]), s["data"]["segments"]
                        ),
                    )

                # transcript.json を更新
                data = sess["data"]
                data["segments"] = updated
                data["session"]["diarized"] = True
                speakers = set(s.get("speaker", "SPEAKER_00") for s in updated)
                data["session"]["num_speakers"] = len(speakers)

                with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                _batch_state["processed"] += 1
                log.info(f"Batch diarize done: {sess['path']} ({len(speakers)} speakers)")

            except Exception as e:
                _batch_state["failed"] += 1
                err_msg = f"{sess['path']}: {e}"
                _batch_state["errors"].append(err_msg)
                log.error(f"Batch diarize error: {err_msg}", exc_info=True)

    except Exception as e:
        log.error(f"Batch diarization task error: {e}", exc_info=True)
    finally:
        _batch_state["running"] = False
        _batch_state["current_session"] = None
        _batch_state["finished_at"] = datetime.now().isoformat()
        log.info(
            f"Batch diarization finished: "
            f"processed={_batch_state['processed']}, "
            f"failed={_batch_state['failed']}, "
            f"total={_batch_state['total']}"
        )


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
    recorder_audio_seconds = 0.0  # WAV に実際に書き込んだ秒数 (録音用オフセット)
    recorder = SessionRecorder()
    silent_since = None  # 無音開始時刻 (None = 発話中)

    # Layer 1: セッション開始時に initial_prompt を構築
    session_prompt = _get_initial_prompt()
    # Layer 3 用の参考用語リスト
    session_known_terms = _get_known_terms() if llm_corrector else None

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
                        recorder_audio_seconds = 0.0
                        log.info("New recording session started (speech detected)")

                # オフセットを退避 (transcribe 前の値を使う)
                ws_offset = total_audio_seconds
                rec_offset = recorder_audio_seconds

                # 音声を録音ファイルに書き込み
                if recorder is not None:
                    recorder.write_audio(chunk)

                log.info(f"Transcribing {buffer_duration:.1f}s of audio...")
                t0 = time.time()

                loop = asyncio.get_running_loop()
                raw_segments = await loop.run_in_executor(
                    executor,
                    lambda: transcribe_audio(chunk, model, 0.0, session_prompt, session_known_terms),
                )

                elapsed = time.time() - t0
                total_audio_seconds += buffer_duration
                recorder_audio_seconds += buffer_duration

                if raw_segments:
                    # WebSocket 応答: 接続全体のオフセット (Omi アプリ用)
                    ws_segments = _offset_segments(raw_segments, ws_offset)
                    response = {"segments": ws_segments}
                    await websocket.send(json.dumps(response))

                    # 録音ログ: WAV ファイル内の位置に合わせたオフセット
                    if recorder is not None:
                        rec_segments = _offset_segments(raw_segments, rec_offset)
                        recorder.add_segments(rec_segments)

                    log.info(
                        f"Sent {len(raw_segments)} segment(s) "
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
                    recorder_audio_seconds = 0.0
                rec_offset = recorder_audio_seconds
                recorder.write_audio(remaining)

                log.info("Processing remaining buffer...")
                try:
                    raw_segments = transcribe_audio(remaining, model, 0.0, session_prompt, session_known_terms)
                    if raw_segments:
                        rec_segments = _offset_segments(raw_segments, rec_offset)
                        recorder.add_segments(rec_segments)
                        try:
                            ws_segments = _offset_segments(raw_segments, total_audio_seconds)
                            await websocket.send(json.dumps({"segments": ws_segments}))
                        except Exception:
                            pass
                except Exception as e:
                    log.error(f"Error processing remaining buffer: {e}", exc_info=True)

        # セッション終了: whisper モデルを解放
        model_manager.release()

        # 話者分離 (バックグラウンドで実行、新セッションをブロックしない)
        if (DIARIZATION_ENABLED and diarizer and diarizer.enabled
                and recorder is not None
                and recorder.enabled
                and not recorder._finalized
                and recorder.total_bytes > 0):
            recorder.close_audio()
            asyncio.create_task(_run_diarization_bg(recorder))
        else:
            # diarization なし → 即 finalize
            if recorder is not None:
                recorder.finalize()

        duration = time.time() - session_start
        log.info(
            f"Session ended: {remote} "
            f"(duration={duration:.0f}s, audio={total_audio_seconds:.0f}s)"
        )


# ============================================================
# メイン
# ============================================================

async def main():
    log.info(f"Starting Omi-Whisper bridge on ws://{HOST}:{PORT}")
    log.info(f"Model: {WHISPER_MODEL}, Compute: {COMPUTE_TYPE}, Language: {LANGUAGE}")
    log.info(f"Buffer: {BUFFER_SECONDS}s, VAD: {VAD_FILTER}, Beam: {BEAM_SIZE}")
    log.info(f"Quality: no_repeat_ngram={NO_REPEAT_NGRAM_SIZE}, "
             f"log_prob_thresh={LOG_PROB_THRESHOLD}, "
             f"no_speech_thresh={NO_SPEECH_THRESHOLD}, "
             f"compression_ratio_thresh={COMPRESSION_RATIO_THRESHOLD}")
    log.info(f"Config dir: {CONFIG_DIR}")
    log.info(f"Term corrections: {_TERM_CORRECTIONS_PATH} ({len(term_corrector.list_corrections())} entries)")
    startup_prompt = _get_initial_prompt()
    if startup_prompt:
        log.info(f"Initial prompt: {startup_prompt[:100]}{'...' if len(startup_prompt) > 100 else ''}")
    else:
        log.info("Initial prompt: (none — set via viewer or INITIAL_PROMPT env)")
    if llm_corrector:
        log.info(f"Layer 3 LLM: {LLM_MODEL} (min_text={LLM_MIN_TEXT_LENGTH}, timeout={LLM_TIMEOUT}s)")
    else:
        reason = "DISABLED" if not LLM_ENABLED else "no GEMINI_API_KEY"
        log.info(f"Layer 3 LLM: {reason}")
    if diarizer and diarizer.enabled:
        auto = "auto-ON" if DIARIZATION_ENABLED else "manual-only"
        log.info(f"Diarization: AVAILABLE ({auto})")
    else:
        log.info("Diarization: UNAVAILABLE (no HF_AUTH_TOKEN)")
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

    async def handle_batch_start(request):
        if diarizer is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        if _batch_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409,
            )
        asyncio.create_task(_batch_diarize_task())
        return web.json_response({"status": "started"})

    async def handle_batch_status(request):
        return web.json_response(_batch_state)

    async def handle_batch_cancel(request):
        if _batch_state["running"]:
            _batch_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    mgmt_app = web.Application()
    mgmt_app.router.add_post("/model/load", handle_model_load)
    mgmt_app.router.add_post("/model/unload", handle_model_unload)
    mgmt_app.router.add_get("/model/status", handle_model_status)
    mgmt_app.router.add_post("/diarize/start", handle_batch_start)
    mgmt_app.router.add_get("/diarize/status", handle_batch_status)
    mgmt_app.router.add_post("/diarize/cancel", handle_batch_cancel)

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
