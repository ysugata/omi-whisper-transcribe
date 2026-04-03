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
BUFFER_SECONDS = float(os.environ.get("BUFFER_SECONDS", "5.0"))          # 音声をバッファしてからまとめて処理する秒数
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))                # Omi は 16kHz PCM を送信

# --- Whisper 精度改善パラメータ ---
CONFIG_DIR = os.environ.get("CONFIG_DIR", "./config")                       # 設定ファイルディレクトリ

NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE", "5"))   # 繰り返し防止 n-gram サイズ (0=無効)
PATIENCE = float(os.environ.get("PATIENCE", "2"))                         # ビームサーチ patience (探索深度、0=無効)
LOG_PROB_THRESHOLD = float(os.environ.get("LOG_PROB_THRESHOLD", "-0.5"))  # セグメント平均対数確率の閾値
NO_SPEECH_THRESHOLD = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6")) # 無音確率の閾値
COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("COMPRESSION_RATIO_THRESHOLD", "2.4"))  # 圧縮率閾値

# --- Layer 3: LLM 後処理設定 ---
LLM_ENABLED = os.environ.get("LLM_ENABLED", "true").lower() in ("true", "1", "yes")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_API_KEY = GEMINI_API_KEY  # Default LLM uses Gemini
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "10"))
LLM_MIN_TEXT_LENGTH = int(os.environ.get("LLM_MIN_TEXT_LENGTH", "20"))
LLM_PIPELINE_STAGES = os.environ.get("LLM_PIPELINE_STAGES", "B")        # Pipeline stages to run (A/B/C)
LLM_SCREENING_ENABLED = os.environ.get("LLM_SCREENING_ENABLED", "true").lower() in ("true", "1", "yes")
STAGE_C_AUTO = os.environ.get("STAGE_C_AUTO", "true").lower() in ("true", "1", "yes")  # Auto-trigger 3-C
LLM_BATCH_MODE = os.environ.get("LLM_BATCH_MODE", "sync")              # "sync" (standard API) or "batch" (Gemini Batch API, slow)
RECOVERY_SCAN_DAYS = int(os.environ.get("RECOVERY_SCAN_DAYS", "7"))     # Startup scan: recover unprocessed sessions from last N days (0=disabled)

# --- セッションタグ付け設定 ---
TAGGING_ENABLED = os.environ.get("TAGGING_ENABLED", "true").lower() in ("true", "1", "yes")
TAGGING_MODEL = os.environ.get("TAGGING_MODEL", "gemini-2.5-flash-lite")

# --- 音響シーン分類設定 ---
AUDIO_SCENE_ENABLED = os.environ.get("AUDIO_SCENE_ENABLED", "true").lower() in ("true", "1", "yes")
AUDIO_SCENE_DEVICE = os.environ.get("AUDIO_SCENE_DEVICE", "cpu")
AUDIO_SCENE_THRESHOLD = float(os.environ.get("AUDIO_SCENE_THRESHOLD", "0.3"))

# --- 深夜スケジューラ設定 ---
NIGHTLY_SCHEDULE_HOUR = int(os.environ.get("NIGHTLY_SCHEDULE_HOUR", "4"))  # 0=無効, 1-23=実行時刻

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

# --- 音声前処理設定 ---
NORMALIZE_ENABLED = os.environ.get("NORMALIZE_ENABLED", "true").lower() in ("true", "1", "yes")
NORMALIZE_TARGET_DBFS = float(os.environ.get("NORMALIZE_TARGET_DBFS", "-16.0"))
NOISE_REDUCE_ENABLED = os.environ.get("NOISE_REDUCE_ENABLED", "false").lower() in ("true", "1", "yes")
NOISE_REDUCE_PROP_DECREASE = float(os.environ.get("NOISE_REDUCE_PROP_DECREASE", "0.8"))

# --- セグメント分割設定 ---
SEGMENT_MAX_DURATION = float(os.environ.get("SEGMENT_MAX_DURATION", "15.0"))  # 秒 (0=無効)

# --- VAD セグメンテーション設定 ---
VAD_SEG_ENABLED = os.environ.get("VAD_SEG_ENABLED", "true").lower() in ("true", "1", "yes")
VAD_CHECK_INTERVAL = float(os.environ.get("VAD_CHECK_INTERVAL", "1.0"))       # VAD チェック間隔 (秒)
VAD_SILENCE_TRIGGER = float(os.environ.get("VAD_SILENCE_TRIGGER", "0.8"))     # 発話後この無音で文字起こし実行
VAD_MAX_BUFFER = float(os.environ.get("VAD_MAX_BUFFER", "30.0"))              # バッファ上限 (秒)
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))                 # Silero VAD 閾値
VAD_MIN_SILENCE_MS = int(os.environ.get("VAD_MIN_SILENCE_MS", "500"))         # 発話間の最小無音 (ms)
VAD_SPEECH_PAD_MS = int(os.environ.get("VAD_SPEECH_PAD_MS", "300"))           # 発話前後のパディング (ms)

# --- 無音検出設定 ---
SILENCE_THRESHOLD = int(os.environ.get("SILENCE_THRESHOLD", "200"))      # int16 RMS 閾値 (0=無効)
SILENCE_TIMEOUT = float(os.environ.get("SILENCE_TIMEOUT", "30.0"))       # 無音タイムアウト秒数

# --- Hotwords 設定 ---
HOTWORDS_ENABLED = os.environ.get("HOTWORDS_ENABLED", "true").lower() in ("true", "1", "yes")

# --- リアルタイム声紋フィルタ設定 ---
VOICE_FILTER_ENABLED = os.environ.get("VOICE_FILTER_ENABLED", "false").lower() in ("true", "1", "yes")
VOICE_FILTER_THRESHOLD = float(os.environ.get("VOICE_FILTER_THRESHOLD", "0"))  # 0 = use speakers.json value

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

from term_corrector import TermCorrector
from llm_corrector import LLMCorrector
from llm_pipeline import LLMPipeline
from batch_corrector import BatchCorrector
from diarizer import Diarizer
from audio_preprocessor import AudioPreprocessor
from hotword_loader import HotwordLoader
from speaker_filter import SpeakerFilter
from session_tagger import SessionTagger
from training_sanitizer import TrainingSanitizer
from responses_sanitizer import ResponsesSanitizer
from segment_screener import SegmentScreener
from suggestion_learner import SuggestionLearner
from faster_whisper.vad import get_speech_timestamps, VadOptions

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

# Layer 3 Pipeline: 3-stage LLM correction (batch post-processing)
llm_pipeline: LLMPipeline | None = None
if LLM_ENABLED and LLM_API_KEY:
    llm_pipeline = LLMPipeline(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        timeout=LLM_TIMEOUT,
        stages=LLM_PIPELINE_STAGES,
    )

# Batch corrector: post-session async LLM correction
batch_corrector: BatchCorrector | None = None

# 話者分離 (HF_AUTH_TOKEN があれば常に利用可能)
diarizer: Diarizer | None = None
if HF_AUTH_TOKEN:
    diarizer = Diarizer(
        hf_token=HF_AUTH_TOKEN,
        language=LANGUAGE or "ja",
    )

# 音声前処理 (ノイズ除去 + ラウドネス正規化)
audio_preprocessor: AudioPreprocessor | None = None
if NORMALIZE_ENABLED or NOISE_REDUCE_ENABLED:
    audio_preprocessor = AudioPreprocessor(
        normalize_enabled=NORMALIZE_ENABLED,
        normalize_target_dbfs=NORMALIZE_TARGET_DBFS,
        noise_reduce_enabled=NOISE_REDUCE_ENABLED,
        noise_reduce_prop_decrease=NOISE_REDUCE_PROP_DECREASE,
    )

# Hotwords ローダー
hotword_loader = HotwordLoader(CONFIG_DIR)

# 声紋ベースの話者フィルタ (HF_AUTH_TOKEN があれば常に利用可能)
speaker_filter: SpeakerFilter | None = None
if HF_AUTH_TOKEN:
    speaker_filter = SpeakerFilter(
        config_dir=CONFIG_DIR,
        hf_token=HF_AUTH_TOKEN,
    )

# セッションタグ付け (LLM API キーがあれば利用可能)
_USER_TAGS_PATH = os.path.join(CONFIG_DIR, "user_tags.json")
TAG_TAXONOMY_PATH = os.path.join(CONFIG_DIR, "tag_taxonomy.json")

session_tagger: SessionTagger | None = None
if TAGGING_ENABLED and LLM_API_KEY:
    session_tagger = SessionTagger(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=TAGGING_MODEL,
        user_tags_path=_USER_TAGS_PATH,
        taxonomy_path=TAG_TAXONOMY_PATH,
    )

from audio_scene_classifier import AudioSceneClassifier
audio_scene_classifier = AudioSceneClassifier(
    enabled=AUDIO_SCENE_ENABLED,
    device=AUDIO_SCENE_DEVICE,
    threshold=AUDIO_SCENE_THRESHOLD,
)

# Training data sanitizer (LLM API キーがあれば利用可能)
# SANITIZE_MODEL supports both Gemini and OpenAI models:
#   gemini-*  → TrainingSanitizer (Chat Completions API, GEMINI_API_KEY)
#   gpt-*     → ResponsesSanitizer (Responses API + Structured Outputs, OPENAI_API_KEY)
SANITIZE_MODEL = os.environ.get("SANITIZE_MODEL", "gemini-2.5-flash")
SANITIZE_USE_RESPONSES_API = os.environ.get(
    "SANITIZE_USE_RESPONSES_API", "auto"
).lower()  # auto | true | false

# Responses API 用の追加設定
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1800"))
OPENAI_TIMEOUT_MS = int(os.environ.get("OPENAI_TIMEOUT_MS", "60000"))

training_sanitizer: TrainingSanitizer | None = None
responses_sanitizer: ResponsesSanitizer | None = None

# Responses API を使うかどうか判定
_use_responses = False
if SANITIZE_USE_RESPONSES_API == "true":
    _use_responses = True
elif SANITIZE_USE_RESPONSES_API == "auto":
    _use_responses = SANITIZE_MODEL.startswith("gpt-")
# else: false → 常に Chat Completions

if _use_responses and OPENAI_API_KEY:
    responses_sanitizer = ResponsesSanitizer(
        api_key=OPENAI_API_KEY,
        model=SANITIZE_MODEL,
        reasoning_effort=OPENAI_REASONING_EFFORT,
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        timeout_ms=OPENAI_TIMEOUT_MS,
    )
else:
    # Gemini or fallback Chat Completions API
    if SANITIZE_MODEL.startswith("gpt-"):
        _sanitize_key = OPENAI_API_KEY
        _sanitize_url = "https://api.openai.com/v1/"
    else:
        _sanitize_key = GEMINI_API_KEY
        _sanitize_url = LLM_BASE_URL
    if _sanitize_key:
        training_sanitizer = TrainingSanitizer(
            api_key=_sanitize_key,
            base_url=_sanitize_url,
            model=SANITIZE_MODEL,
        )


def _build_raw_text_from_transcript(transcript: dict) -> str:
    """transcript.json から Responses API 用の raw_text を構築する."""
    parts = []
    session = transcript.get("session", {})
    tags = session.get("tags", {})
    if tags:
        summary = tags.get("summary", "")
        if summary:
            parts.append(f"[要約] {summary}")

    segments = transcript.get("segments", [])
    for seg in segments[:80]:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("person_name") or seg.get("speaker", "")
        if speaker:
            parts.append(f"[{speaker}] {text}")
        else:
            parts.append(text)

    return "\n".join(parts)


def _get_known_terms() -> list[str]:
    """LLM 補正用の参考用語リストを構築する。

    term_corrector の正解語 + hotwords カスタム語をマージ。
    """
    terms: set[str] = set()

    # term_corrector の正解語 (辞書の values)
    for v in term_corrector.list_corrections().values():
        terms.add(v)

    # hotwords のカスタム語
    for w in hotword_loader.get_custom_words():
        terms.add(w)

    return sorted(terms)


def _build_session_glossary(
    segments: list[dict],
    known_terms: list[str] | None,
    max_glossary: int = 50,
) -> list[str]:
    """Build a session-specific glossary filtered by segment text relevance.

    1. Extract katakana words from segment text
    2. Include known_terms that substring-match segment text
    3. Include hotwords that substring-match segment text
    4. Cap at max_glossary
    """
    import re as _re
    combined_text = " ".join(seg.get("text", "") for seg in segments)
    if not combined_text.strip():
        return []

    # Extract katakana words from text (3+ chars)
    katakana_words = set(_re.findall(r'[\u30A0-\u30FF]{3,}', combined_text))

    glossary: list[str] = []
    seen: set[str] = set()

    # Add known_terms that appear as substring in text
    if known_terms:
        for term in known_terms:
            if term in combined_text and term not in seen:
                glossary.append(term)
                seen.add(term)

    # Add hotwords custom words matching text
    for word in hotword_loader.get_custom_words():
        if word in combined_text and word not in seen:
            glossary.append(word)
            seen.add(word)

    # Add extracted katakana words not already included
    for kw in sorted(katakana_words):
        if kw not in seen:
            glossary.append(kw)
            seen.add(kw)

    # Add remaining known_terms (not yet included) up to limit
    if known_terms:
        for term in known_terms:
            if term not in seen:
                glossary.append(term)
                seen.add(term)
            if len(glossary) >= max_glossary:
                break

    return glossary[:max_glossary]


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
        self.session_tags: dict | None = None  # set by _tag_session() before finalize

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

        # 空の録音 (音声なし or セグメント0件) なら削除
        if self.total_bytes == 0 or len(self.all_segments) == 0:
            import shutil
            shutil.rmtree(self.session_dir, ignore_errors=True)
            reason = "no audio" if self.total_bytes == 0 else "no segments (all filtered)"
            log.info(f"Empty session removed: {reason}")
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
        session_meta = {
            "start": self.start_time.isoformat(),
            "end": datetime.now().isoformat(),
            "duration_seconds": round(duration_sec, 1),
            "audio_file": audio_file,
            "audio_size_mb": round(file_size_mb, 2),
            "model": WHISPER_MODEL,
            "language": LANGUAGE,
            "num_speakers": num_speakers,
        }
        if self.session_tags:
            session_meta["tags"] = self.session_tags
            session_meta["session_tagged"] = True
        output = {
            "session": session_meta,
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
    re.compile(r"最後まで(ご覧|見て)いただき.*"),
    re.compile(r"ご覧いただきありがとうございま(した|す)"),
    re.compile(r"字幕.*?(視聴者|翻訳|提供)"),
    re.compile(r"Thanks? for watching\.?", re.IGNORECASE),
    re.compile(r"See you next time\.?", re.IGNORECASE),
    re.compile(r"Subscribe", re.IGNORECASE),
    re.compile(r"Bye[\.\s]*bye\.?", re.IGNORECASE),
    re.compile(r"♪+"),
    re.compile(r"^\s*\.+\s*$"),
    re.compile(r"(.{2,}?)\1{3,}"),  # 同一フレーズの4回以上繰り返し
    re.compile(r"(F\d+[・,、]\s*){3,}"),  # F-key 暴走パターン (F1・F2・F3...)
    re.compile(r"^[。、.,!?・…―\s]+$"),  # 句読点・記号のみ
    re.compile(r"\ufffd"),  # Unicode replacement character (音声デコードエラー)
]

# 全文一致専用パターン: セグメント全体がこのフレーズのみの場合だけ除去
# (実際の会話で使われうるが、セグメント全体がこれだけなら Whisper のハルシネーション)
_HALLUCINATION_FULL_MATCH = [
    re.compile(r"^お(つかれさま|疲れ様)でした[。.!]*$"),
    re.compile(r"^(ではまた|また次回)[。.!]*$"),
    re.compile(r"^おやすみなさい[。.!]*$"),
]


def _remove_hallucinations(text: str) -> str:
    """既知のハルシネーションパターンを除去し、残りテキストを返す"""
    stripped = text.strip()
    # Full-match patterns: only remove if entire segment matches
    for pattern in _HALLUCINATION_FULL_MATCH:
        if pattern.match(stripped):
            return ""
    cleaned = text
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# 末尾カンマ断片クリーナー: 1-6文字の断片がカンマ区切りで2回以上末尾に連続
_TRAILING_FRAGMENT_RE = re.compile(
    r"([,，、]\s*[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fffa-zA-Z]{1,6}){2,}\s*$"
)


def _clean_trailing_fragments(text: str) -> str:
    """セグメント末尾のカンマ区切り断片ノイズを除去する。

    例: "全部で, でも多いな, カマイブレもらいたいから" → "全部で"
    全部消えた場合は元テキストを保持。
    """
    cleaned = _TRAILING_FRAGMENT_RE.sub("", text).strip()
    if not cleaned:
        return text
    return cleaned


# ギブリッシュ仮名フィルタ: Whisper が非音声を無理やりカタカナ/ひらがなに変換したノイズを除去
# 文法的助詞パターン (前後に仮名がある助詞)
_PARTICLE_PATTERNS = re.compile(
    r'[ぁ-ん\u30a0-\u30ff](は|が|を|へ|から|まで|より|ばかり|さえ|すら|だけ|ほど|の|で|と|に|も|や)[ぁ-ん\u30a0-\u30ff]'
)
# 文法的語尾パターン (ギブリッシュには現れない日本語特有の活用形)
_GRAMMATICAL_ENDINGS = re.compile(
    r'(です|ます|ました|ません|でした|ている|ていた|ていない|ないで|なかった'
    r'|ことが|ものが|ように|ようだ|だろう|でしょう|ですか|ますか|ますよ|ですね'
    r'|ですよ|ましょう|ください|ながら|けれど|ったら|ったり|のです|んです'
    r'|よかった|じゃない|なさい|だよ|だね|だな|てる|てた|ちゃう|ちゃった'
    r'|って|かな|かも|でしょ|っけ|じゃん|んだ|のに|けど)'
)

_GIBBERISH_LOGPROB_THRESHOLD = -0.7


def _is_gibberish_kana(text: str, avg_logprob: float) -> bool:
    """テキストが意味不明なカタカナ/ひらがな羅列 (ギブリッシュ) かどうかを判定する。

    Whisper が非音声 (電車の走行音、空調ノイズ等) を無理やり文字化した場合に
    生成される、意味のないカタカナ/ひらがなの連続を検出する。

    偽陽性防止のため、以下の場合はスキップ:
    - 高い信頼度 (avg_logprob >= -0.3)
    - 漢字を含む (通常の日本語文)
    - 15文字以下 (口語は仮名のみ8〜14文字が普通)
    - 文法的な助詞パターンを含む (文法的に正しい文)
    """
    # High confidence → skip
    if avg_logprob >= -0.3:
        return False

    # Strip punctuation, commas, spaces to get the "body"
    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)

    if len(body) <= 15:
        return False

    # Contains kanji → legitimate Japanese
    if re.search(r'[\u4e00-\u9fff]', body):
        return False

    # Count kana ratios
    katakana_count = len(re.findall(r'[\u30a0-\u30ff]', body))
    hiragana_count = len(re.findall(r'[\u3040-\u309f]', body))
    body_len = len(body)

    katakana_ratio = katakana_count / body_len if body_len else 0
    hiragana_ratio = hiragana_count / body_len if body_len else 0

    # Must be kana-dominant (70%+)
    if katakana_ratio < 0.7 and hiragana_ratio < 0.7:
        return False

    # Low confidence required
    if avg_logprob >= _GIBBERISH_LOGPROB_THRESHOLD:
        return False

    # Contains grammatical particle patterns → likely real Japanese
    if _PARTICLE_PATTERNS.search(body):
        return False

    # Contains grammatical endings (です/ます/ました etc.) → real Japanese
    if _GRAMMATICAL_ENDINGS.search(body):
        return False

    return True


# 句読点正規化
def _normalize_punctuation(text: str) -> str:
    """句読点を正規化する。

    1. 日本語文字に囲まれた `.` → `。` (数字の小数点は除外)
    2. `- ` で始まる発話区切りマーカーを除去
    3. 末尾のカンマ `,` `、` を除去
    4. 連続する同一句読点を1つにまとめる
    """
    # 1. Period after Japanese text → 。 (exclude decimal points like 3.14)
    #    Matches: period preceded by non-digit, non-space AND followed by non-digit or end of string
    result = re.sub(
        r'(?<=[^\d\s])\.(?=[^\d\s]|$)',
        '。',
        text,
    )

    # 2. Remove dash speech delimiters: "- 楽しかった?- 美味しかった" → "楽しかった? 美味しかった"
    #    Match dash followed by space (dialogue marker style)
    result = re.sub(r'\s*-\s+', ' ', result).strip()

    # 3. Remove trailing commas
    result = re.sub(r'[,，、]+\s*$', '', result)

    # 4. Collapse consecutive identical punctuation
    result = re.sub(r'([。、!?！？])\1+', r'\1', result)

    return result.strip()


# セグメント分割: 長いセグメントを文境界で分割
_SEGMENT_SPLIT_RE = re.compile(
    r'(?<=[。.!?？！])|'    # Sentence-ending punctuation
    r'(?<=けど\s)|'         # Conjunction + space
    r'(?<=から\s)|'
    r'(?<=のに\s)|'
    r'(?<=ので\s)|'
    r'(?<=けれど\s)'
)


def _split_long_segment(seg_dict: dict, max_duration: float) -> list[dict]:
    """長いセグメントを文境界で分割する。

    タイムスタンプは文字数比で按分。
    5文字未満の断片は隣接セグメントにマージ。
    分割不能な場合は元のまま返す。
    """
    text = seg_dict["text"]
    start = seg_dict["start"]
    end = seg_dict["end"]
    duration = end - start

    if duration <= max_duration:
        return [seg_dict]

    # Split by sentence boundaries
    parts = _SEGMENT_SPLIT_RE.split(text)
    parts = [p for p in parts if p]  # remove empty strings

    if len(parts) <= 1:
        return [seg_dict]

    # Merge short fragments (< 5 chars) into neighbors
    merged = []
    for p in parts:
        if merged and len(p.strip()) < 5:
            merged[-1] += p
        else:
            merged.append(p)

    # If last part is too short, merge into previous
    if len(merged) > 1 and len(merged[-1].strip()) < 5:
        merged[-2] += merged[-1]
        merged.pop()

    if len(merged) <= 1:
        return [seg_dict]

    # Compute timestamps proportionally by character count
    total_chars = sum(len(p) for p in merged)
    if total_chars == 0:
        return [seg_dict]

    result = []
    current_start = start
    for i, part in enumerate(merged):
        ratio = len(part) / total_chars
        part_duration = duration * ratio
        part_end = current_start + part_duration
        if i == len(merged) - 1:
            part_end = end  # last segment gets exact end time

        new_seg = dict(seg_dict)
        new_seg["text"] = part.strip()
        new_seg["start"] = round(current_start, 2)
        new_seg["end"] = round(part_end, 2)
        if new_seg["text"]:  # skip empty
            result.append(new_seg)

        current_start = part_end

    return result if result else [seg_dict]


def transcribe_audio(
    audio_bytes: bytes,
    model,
    session_offset: float = 0.0,
    hotwords: str = "",
) -> list[dict]:
    """PCM 16-bit LE の音声バイト列を受け取り、
    Omi の custom_stt が期待する segments リストを返す。

    Args:
        audio_bytes: PCM 16-bit LE 音声データ
        model: ロード済みの WhisperModel インスタンス
        session_offset: タイムスタンプオフセット (秒)
        hotwords: カンマ区切りの hotwords 文字列 (デコーダバイアス用)
    """
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < 1600:  # 0.1秒未満は無視
        return []

    # 音声前処理 (ラウドネス正規化)
    if audio_preprocessor is not None:
        audio_np = audio_preprocessor.process_array(audio_np, SAMPLE_RATE)

    segments_out = []

    chunk_duration = len(audio_np) / SAMPLE_RATE

    # transcribe パラメータ構築
    transcribe_kwargs = dict(
        beam_size=BEAM_SIZE,
        language=LANGUAGE,
        vad_filter=VAD_FILTER,
        vad_parameters=dict(
            min_speech_duration_ms=500,   # 500ms (was 250ms) — reject short noise bursts
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            threshold=0.5,                # 0.5 (was 0.4) — stricter speech detection
        ),
        word_timestamps=False,
        condition_on_previous_text=False,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        patience=PATIENCE,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
    )
    if hotwords:
        transcribe_kwargs["hotwords"] = hotwords

    segments, info = model.transcribe(audio_np, **transcribe_kwargs)

    # auto モード時に検出言語をログ出力
    if LANGUAGE is None:
        log.info(
            f"Language detected: {info.language} "
            f"(probability={info.language_probability:.2f})"
        )

    for seg in segments:
        # 確信度フィルタ 1: 高い無音確率 → logprob も考慮して除去
        # logprob が高い (>= -0.3) 場合は Whisper が自信を持って文字起こし
        # できているので、VAD の no_speech 判定を無視して保持する
        if seg.no_speech_prob > 0.8 and seg.avg_logprob < -0.3:
            log.info(
                f"Segment FILTERED (high no_speech): no_speech_prob={seg.no_speech_prob:.2f}, "
                f"avg_logprob={seg.avg_logprob:.2f}, text='{seg.text.strip()[:80]}'"
            )
            continue

        # 確信度フィルタ 2: 両方悪い場合に除去
        if seg.avg_logprob < -0.6 and seg.no_speech_prob > 0.4:
            log.info(
                f"Segment FILTERED (low quality): avg_logprob={seg.avg_logprob:.2f}, "
                f"no_speech_prob={seg.no_speech_prob:.2f}, text='{seg.text.strip()[:80]}'"
            )
            continue

        text = seg.text.strip()
        if not text:
            continue

        # Unicode replacement character (音声デコードエラー由来)
        if '\ufffd' in text:
            log.debug(f"Segment with replacement char dropped: '{text[:40]}'")
            continue

        # 短すぎるセグメントでかつ信頼度が低い場合は除去
        if len(text) <= 1:
            log.debug(f"Single-char segment dropped: '{text}'")
            continue
        if len(text) <= 3 and seg.avg_logprob < -0.4:
            log.debug(f"Short low-confidence segment dropped: '{text}' (logprob={seg.avg_logprob:.2f})")
            continue

        # ギブリッシュ仮名フィルタ (意味不明なカタカナ/ひらがな羅列を除去)
        if _is_gibberish_kana(text, seg.avg_logprob):
            log.info(f"Gibberish kana filtered: '{text[:80]}' (logprob={seg.avg_logprob:.2f})")
            continue

        # タイムスタンプをチャンク長にクランプ
        seg_start = max(0.0, min(seg.start, chunk_duration))
        seg_end = max(0.0, min(seg.end, chunk_duration))
        if seg_start >= seg_end:
            seg_end = min(seg_start + 0.5, chunk_duration)  # 最小 0.5s

        # 文字速度フィルタ (日本語会話は 5〜10 文字/秒、25 は早口でも不可能)
        duration = seg_end - seg_start
        if duration > 0 and len(text) / duration > 25:
            log.info(f"Segment FILTERED (impossible rate {len(text)/duration:.0f} chars/s): '{text[:80]}'")
            continue

        # ハルシネーション除去
        cleaned = _remove_hallucinations(text)
        if not cleaned:
            log.info(f"Hallucination removed: '{text[:80]}'")
            continue
        text = cleaned

        # 末尾カンマ断片クリーナー
        text = _clean_trailing_fragments(text)

        # 句読点正規化
        text = _normalize_punctuation(text)
        if not text:
            continue

        # Layer 2: 用語補正
        text = term_corrector.correct(text)

        seg_dict = {
            "text": text,
            "speaker": "SPEAKER_00",
            "speaker_id": 0,
            "start": round(seg_start + session_offset, 2),
            "end": round(seg_end + session_offset, 2),
            "is_user": True,
            "person_id": None,
            "avg_logprob": round(seg.avg_logprob, 3),
        }
        if LANGUAGE is None:
            seg_dict["language"] = info.language

        # セグメント分割 (長いセグメントを文境界で分割)
        if SEGMENT_MAX_DURATION > 0 and (seg_end - seg_start) > SEGMENT_MAX_DURATION:
            split_segs = _split_long_segment(seg_dict, SEGMENT_MAX_DURATION)
            segments_out.extend(split_segs)
        else:
            segments_out.append(seg_dict)

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
            updated, success = await loop.run_in_executor(
                None,
                lambda: diarizer.diarize(
                    str(recorder.wav_path), recorder.all_segments
                ),
            )
            if success:
                recorder.all_segments = updated
                log.info("Post-session diarization succeeded")
            else:
                log.warning("Post-session diarization failed, keeping original segments")

    except Exception as e:
        log.error(f"Diarization error: {e}", exc_info=True)
    finally:
        recorder.finalize()
        # Enqueue for batch LLM correction + tagging
        if (batch_corrector and recorder.enabled
                and hasattr(recorder, 'session_dir') and recorder.session_dir
                and len(recorder.all_segments) > 0):
            session_id = str(recorder.session_dir.relative_to(RECORDINGS_DIR))
            batch_corrector.enqueue(session_id)


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


def _find_undiarized_sessions(force: bool = False) -> list[dict]:
    """RECORDINGS_DIR 配下の未 diarize セッションを古い順に返す。

    force=True の場合、既に diarized=True のセッションも含める (再実行用)。
    """
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
            if not force and session_info.get("diarized"):
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


async def _batch_diarize_task(force: bool = False):
    """バッチ話者分離タスク。asyncio.create_task で起動する。"""
    import shutil

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
        sessions = _find_undiarized_sessions(force=force)
        _batch_state["total"] = len(sessions)
        log.info(f"Batch diarization: {len(sessions)} session(s) to process (force={force})")

        if not sessions:
            return

        if _batch_state["cancel_requested"]:
            log.info("Batch diarization cancelled by user (before GPU lock)")
            return

        loop = asyncio.get_running_loop()

        for sess in sessions:
            if _batch_state["cancel_requested"]:
                log.info("Batch diarization cancelled by user")
                break

            _batch_state["current_session"] = sess["path"]
            log.info(f"Batch diarize: {sess['path']}")

            try:
                # Pyannote (~2GB) + Whisper (~5GB) = ~7GB — fits in RTX 3060 12GB.
                # No need to unload Whisper. gpu_lock blocks new WS sessions.
                async with model_manager.gpu_lock:
                    updated, success = await loop.run_in_executor(
                        None,
                        lambda s=sess: diarizer.diarize(
                            str(s["audio"]), s["data"]["segments"]
                        ),
                    )

                if not success:
                    _batch_state["failed"] += 1
                    err_msg = f"{sess['path']}: diarization failed (see logs)"
                    _batch_state["errors"].append(err_msg)
                    log.warning(f"Batch diarize failed: {sess['path']}")
                    continue

                # バックアップを作成してから transcript.json を更新
                bak_path = str(sess["transcript_path"]) + ".bak"
                shutil.copy2(sess["transcript_path"], bak_path)

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

            # Yield to event loop between sessions
            await asyncio.sleep(0)

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
# バッチ話者タグ付け
# ============================================================

_batch_tag_state = {
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


def _find_untagged_sessions(force: bool = False) -> list[dict]:
    """RECORDINGS_DIR 配下の未タグ付けセッションを古い順に返す。"""
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
            if not force and session_info.get("speaker_tagged"):
                continue

            # 音声ファイル特定
            audio_file = session_info.get("audio_file", "audio.ogg")
            audio_path = time_dir / audio_file
            if not audio_path.exists():
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


async def _batch_tag_task(force: bool = False):
    """バッチ話者タグ付けタスク。asyncio.create_task で起動する。"""
    import shutil

    _batch_tag_state["running"] = True
    _batch_tag_state["processed"] = 0
    _batch_tag_state["failed"] = 0
    _batch_tag_state["skipped"] = 0
    _batch_tag_state["current_session"] = None
    _batch_tag_state["errors"] = []
    _batch_tag_state["cancel_requested"] = False
    _batch_tag_state["started_at"] = datetime.now().isoformat()
    _batch_tag_state["finished_at"] = None

    try:
        sessions = _find_untagged_sessions(force=force)
        _batch_tag_state["total"] = len(sessions)
        log.info(f"Batch speaker tagging: {len(sessions)} session(s) to process (force={force})")

        if not sessions:
            return

        if _batch_tag_state["cancel_requested"]:
            log.info("Batch speaker tagging cancelled by user (before GPU lock)")
            return

        loop = asyncio.get_running_loop()

        for sess in sessions:
            if _batch_tag_state["cancel_requested"]:
                log.info("Batch speaker tagging cancelled by user")
                break

            _batch_tag_state["current_session"] = sess["path"]
            log.info(f"Batch tag: {sess['path']}")

            try:
                # Voice embedding model (~50MB) coexists with Whisper (~5GB),
                # so no need to unload Whisper. gpu_lock blocks new WS sessions.
                async with model_manager.gpu_lock:
                    inference = await loop.run_in_executor(
                        None, speaker_filter._load_model
                    )
                    try:
                        original_segments = sess["data"]["segments"]
                        updated = await loop.run_in_executor(
                            None,
                            lambda s=sess: speaker_filter.tag_segments_with_model(
                                inference, str(s["audio"]), s["data"]["segments"]
                            ),
                        )
                    finally:
                        await loop.run_in_executor(
                            None, lambda: speaker_filter._unload_model(inference)
                        )

                # Preserve manually_tagged segments
                for i, orig in enumerate(original_segments):
                    if orig.get("manually_tagged") and i < len(updated):
                        updated[i] = orig

                # バックアップを作成してから transcript.json を更新
                bak_path = str(sess["transcript_path"]) + ".bak"
                shutil.copy2(sess["transcript_path"], bak_path)

                data = sess["data"]
                data["segments"] = updated
                data["session"]["speaker_tagged"] = True

                with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                _batch_tag_state["processed"] += 1
                log.info(f"Batch tag done: {sess['path']}")

            except Exception as e:
                _batch_tag_state["failed"] += 1
                err_msg = f"{sess['path']}: {e}"
                _batch_tag_state["errors"].append(err_msg)
                log.error(f"Batch tag error: {err_msg}", exc_info=True)

            # Yield to event loop between sessions
            await asyncio.sleep(0)

    except Exception as e:
        log.error(f"Batch speaker tagging task error: {e}", exc_info=True)
    finally:
        _batch_tag_state["running"] = False
        _batch_tag_state["current_session"] = None
        _batch_tag_state["finished_at"] = datetime.now().isoformat()
        log.info(
            f"Batch speaker tagging finished: "
            f"processed={_batch_tag_state['processed']}, "
            f"failed={_batch_tag_state['failed']}, "
            f"total={_batch_tag_state['total']}"
        )


# ============================================================
# バッチセッションタグ付け (LLM)
# ============================================================

_batch_session_tag_state = {
    "running": False,
    "total": 0,
    "processed": 0,
    "failed": 0,
    "current_session": None,
    "errors": [],
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
}


def _find_untagged_session_tag_sessions(force: bool = False) -> list[dict]:
    """タグ付け対象のセッションを探す。"""
    sessions = []
    recordings = Path(RECORDINGS_DIR)
    if not recordings.is_dir():
        return sessions

    for day_dir in sorted(recordings.iterdir()):
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

            if not data.get("segments"):
                continue

            if not force and data.get("session", {}).get("session_tagged"):
                continue

            sessions.append({
                "path": f"{day_dir.name}/{time_dir.name}",
                "transcript_path": transcript_path,
                "data": data,
            })

    return sessions


def _find_unscene_classified_sessions(force: bool = False) -> list[tuple]:
    """音響シーン未分類のセッションを探す。(session_dir, session_data) のリストを返す。"""
    results = []
    recordings = Path(RECORDINGS_DIR)
    if not recordings.is_dir():
        return results

    for day_dir in sorted(recordings.iterdir()):
        if not day_dir.is_dir():
            continue
        for time_dir in sorted(day_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue

            session_json = time_dir / "session.json"
            sess_data = {}
            if session_json.exists():
                try:
                    with open(session_json, encoding="utf-8") as f:
                        sess_data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            if not force and "audio_scene" in sess_data:
                continue

            results.append((str(time_dir), sess_data))

    return results


async def _batch_session_tag_task(force: bool = False):
    """バッチセッションタグ付けタスク。"""
    import shutil

    _batch_session_tag_state["running"] = True
    _batch_session_tag_state["processed"] = 0
    _batch_session_tag_state["failed"] = 0
    _batch_session_tag_state["current_session"] = None
    _batch_session_tag_state["errors"] = []
    _batch_session_tag_state["cancel_requested"] = False
    _batch_session_tag_state["started_at"] = datetime.now().isoformat()
    _batch_session_tag_state["finished_at"] = None

    try:
        sessions = _find_untagged_session_tag_sessions(force=force)
        _batch_session_tag_state["total"] = len(sessions)
        log.info(f"Batch session tagging: {len(sessions)} session(s) to process (force={force})")

        if not sessions:
            return

        loop = asyncio.get_running_loop()

        for sess in sessions:
            if _batch_session_tag_state["cancel_requested"]:
                log.info("Batch session tagging cancelled by user")
                break

            _batch_session_tag_state["current_session"] = sess["path"]
            log.info(f"Batch session tag: {sess['path']}")

            try:
                segments = sess["data"].get("segments", [])
                tags = await loop.run_in_executor(
                    None,
                    lambda segs=segments: session_tagger.tag_session(segs),
                )

                if tags:
                    # Backup then update transcript.json
                    bak_path = str(sess["transcript_path"]) + ".bak"
                    shutil.copy2(sess["transcript_path"], bak_path)

                    data = sess["data"]
                    data["session"]["tags"] = tags
                    data["session"]["session_tagged"] = True

                    with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                    _batch_session_tag_state["processed"] += 1
                    log.info(f"Batch session tag done: {sess['path']} "
                             f"({tags.get('activity', '?')}, {tags.get('mood', '?')})")
                else:
                    _batch_session_tag_state["failed"] += 1
                    _batch_session_tag_state["errors"].append(f"{sess['path']}: tagger returned None")

            except Exception as e:
                _batch_session_tag_state["failed"] += 1
                err_msg = f"{sess['path']}: {e}"
                _batch_session_tag_state["errors"].append(err_msg)
                log.error(f"Batch session tag error: {err_msg}", exc_info=True)

            await asyncio.sleep(0)

    except Exception as e:
        log.error(f"Batch session tagging task error: {e}", exc_info=True)
    finally:
        _batch_session_tag_state["running"] = False
        _batch_session_tag_state["current_session"] = None
        _batch_session_tag_state["finished_at"] = datetime.now().isoformat()
        log.info(
            f"Batch session tagging finished: "
            f"processed={_batch_session_tag_state['processed']}, "
            f"failed={_batch_session_tag_state['failed']}, "
            f"total={_batch_session_tag_state['total']}"
        )


# ============================================================
# 統合タグ付けバッチ (Voice Tag → Session Tag)
# ============================================================

_batch_combined_tag_state = {
    "running": False,
    "phase": None,            # "voice_tag" | "audio_scene" | "session_tag"
    "total": 0,
    "processed": 0,
    "failed": 0,
    "current_session": None,
    "errors": [],
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
    # Phase details
    "session_tag_total": 0,
    "session_tag_processed": 0,
    "session_tag_failed": 0,
    "voice_tag_total": 0,
    "voice_tag_processed": 0,
    "voice_tag_failed": 0,
    "audio_scene_total": 0,
    "audio_scene_processed": 0,
    "audio_scene_failed": 0,
}


async def _batch_combined_tag_task(force: bool = False):
    """統合タグ付けバッチ: Voice Tag → Session Tag の順に実行する。

    声紋タグを先に実行することで、LLM セッションタグ付け時に
    話者情報 (is_user, person_name) を参照できる。
    """
    import shutil

    st = _batch_combined_tag_state
    st["running"] = True
    st["phase"] = None
    st["processed"] = 0
    st["failed"] = 0
    st["current_session"] = None
    st["errors"] = []
    st["cancel_requested"] = False
    st["started_at"] = datetime.now().isoformat()
    st["finished_at"] = None
    st["session_tag_total"] = 0
    st["session_tag_processed"] = 0
    st["session_tag_failed"] = 0
    st["voice_tag_total"] = 0
    st["voice_tag_processed"] = 0
    st["voice_tag_failed"] = 0
    st["audio_scene_total"] = 0
    st["audio_scene_processed"] = 0
    st["audio_scene_failed"] = 0

    try:
        # --- Phase 1: Voice Tag (Speaker) ---
        # 声紋タグを先に実行し、セグメントに is_user / person_name を付与する。
        # これにより Phase 2 の LLM タグ付けで話者情報を活用できる。
        st["phase"] = "voice_tag"
        if speaker_filter is not None:
            voice_sessions = _find_untagged_sessions(force=force)
            st["voice_tag_total"] = len(voice_sessions)
            st["total"] = len(voice_sessions)
            log.info(f"Combined tag Phase 1 (voice tag): {len(voice_sessions)} session(s) (force={force})")

            if voice_sessions:
                loop = asyncio.get_running_loop()

                for sess in voice_sessions:
                    if st["cancel_requested"]:
                        log.info("Combined tag cancelled by user (voice tag phase)")
                        break

                    st["current_session"] = sess["path"]
                    try:
                        original_segments = sess["data"]["segments"]
                        # Voice embedding model (~50MB) coexists with Whisper (~5GB)
                        # on RTX 3060 12GB, so no need to unload Whisper.
                        # gpu_lock blocks new WS sessions during processing.
                        async with model_manager.gpu_lock:
                            inference = await loop.run_in_executor(
                                None, speaker_filter._load_model
                            )
                            try:
                                updated = await loop.run_in_executor(
                                    None,
                                    lambda s=sess: speaker_filter.tag_segments_with_model(
                                        inference, str(s["audio"]), s["data"]["segments"]
                                    ),
                                )
                            finally:
                                await loop.run_in_executor(
                                    None, lambda: speaker_filter._unload_model(inference)
                                )

                        # Preserve manually_tagged segments
                        for i, orig in enumerate(original_segments):
                            if orig.get("manually_tagged") and i < len(updated):
                                updated[i] = orig

                        bak_path = str(sess["transcript_path"]) + ".bak"
                        shutil.copy2(sess["transcript_path"], bak_path)

                        data = sess["data"]
                        data["segments"] = updated
                        data["session"]["speaker_tagged"] = True

                        with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        st["voice_tag_processed"] += 1
                        st["processed"] += 1

                    except Exception as e:
                        st["voice_tag_failed"] += 1
                        st["failed"] += 1
                        err_msg = f"{sess['path']}: {e}"
                        st["errors"].append(err_msg)
                        log.error(f"Combined tag voice-tag error: {err_msg}", exc_info=True)

                    await asyncio.sleep(0)
        else:
            log.info("Combined tag Phase 1 skipped: no speaker_filter (HF_AUTH_TOKEN not set)")

        # --- Phase 1.5: Audio Scene Classification (PANNs CNN14) ---
        if st["cancel_requested"]:
            return

        if AUDIO_SCENE_ENABLED:
            st["phase"] = "audio_scene"
            scene_sessions = _find_unscene_classified_sessions(force=force)
            st["audio_scene_total"] = len(scene_sessions)
            log.info(f"Combined tag Phase 1.5 (audio scene): {len(scene_sessions)} session(s)")
            for sess_dir, sess_data in scene_sessions:
                if st["cancel_requested"]:
                    log.info("Combined tag cancelled by user (audio scene phase)")
                    break
                try:
                    wav_path = os.path.join(sess_dir, "full_audio.wav")
                    if not os.path.exists(wav_path):
                        log.warning("No full_audio.wav in %s, skipping scene classification", sess_dir)
                        st["audio_scene_failed"] += 1
                        continue
                    result = audio_scene_classifier.classify(wav_path)
                    if result:
                        sess_data["audio_scene"] = result
                        session_json = os.path.join(sess_dir, "session.json")
                        with open(session_json, "w") as f:
                            json.dump(sess_data, f, ensure_ascii=False, indent=2)
                        log.info("Audio scene for %s: %s", os.path.basename(sess_dir), result.get("scene_label", "unknown"))
                    st["audio_scene_processed"] += 1
                except Exception as e:
                    log.error("Audio scene classification failed for %s: %s", sess_dir, e)
                    st["audio_scene_failed"] += 1
                await asyncio.sleep(0)

        # --- Phase 2: Session Tag (LLM) ---
        # 声紋タグ済みのセグメントを使って LLM タグ付けを実行する。
        if st["cancel_requested"]:
            return

        st["phase"] = "session_tag"
        session_tag_sessions = _find_untagged_session_tag_sessions(force=force)
        st["session_tag_total"] = len(session_tag_sessions)
        st["total"] = st["voice_tag_total"] + st["audio_scene_total"] + len(session_tag_sessions)
        st["processed"] = st["voice_tag_processed"] + st["voice_tag_failed"]
        st["failed"] = 0  # Reset failed count for combined progress
        log.info(f"Combined tag Phase 2 (session tag): {len(session_tag_sessions)} session(s) (force={force})")

        if session_tag_sessions and session_tagger is not None:
            loop = asyncio.get_running_loop()
            for sess in session_tag_sessions:
                if st["cancel_requested"]:
                    log.info("Combined tag cancelled by user (session tag phase)")
                    break

                st["current_session"] = sess["path"]
                try:
                    # Re-read transcript to pick up voice tags from Phase 1
                    with open(sess["transcript_path"], "r", encoding="utf-8") as f:
                        fresh_data = json.load(f)
                    segments = fresh_data.get("segments", [])

                    # Load audio_scene from session.json if available
                    _sess_json_path = Path(sess["transcript_path"]).parent / "session.json"
                    _audio_scene = None
                    if _sess_json_path.exists():
                        try:
                            with open(_sess_json_path, encoding="utf-8") as _sf:
                                _audio_scene = json.load(_sf).get("audio_scene")
                        except (json.JSONDecodeError, OSError):
                            pass

                    tags = await loop.run_in_executor(
                        None,
                        lambda segs=segments, asc=_audio_scene: session_tagger.tag_session(segs, audio_scene=asc),
                    )

                    if tags:
                        bak_path = str(sess["transcript_path"]) + ".bak"
                        shutil.copy2(sess["transcript_path"], bak_path)

                        fresh_data["session"]["tags"] = tags
                        fresh_data["session"]["session_tagged"] = True

                        with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                            json.dump(fresh_data, f, ensure_ascii=False, indent=2)

                        st["session_tag_processed"] += 1
                        st["processed"] += 1
                    else:
                        st["session_tag_failed"] += 1
                        st["failed"] += 1
                        st["errors"].append(f"{sess['path']}: tagger returned None")

                except Exception as e:
                    st["session_tag_failed"] += 1
                    st["failed"] += 1
                    err_msg = f"{sess['path']}: {e}"
                    st["errors"].append(err_msg)
                    log.error(f"Combined tag session-tag error: {err_msg}", exc_info=True)

                await asyncio.sleep(0)

    except Exception as e:
        log.error(f"Combined tag task error: {e}", exc_info=True)
    finally:
        st["running"] = False
        st["current_session"] = None
        st["finished_at"] = datetime.now().isoformat()
        log.info(
            f"Combined tag finished: "
            f"session_tag={st['session_tag_processed']}/{st['session_tag_total']}, "
            f"voice_tag={st['voice_tag_processed']}/{st['voice_tag_total']}, "
            f"failed={st['session_tag_failed'] + st['voice_tag_failed']}"
        )


# ============================================================
# 深夜スケジューラ (話者分離 + タグ付け一括実行)
# ============================================================

_nightly_state = {
    "enabled": False,
    "hour": 0,
    "running": False,
    "phase": None,            # "diarize" | "tag"
    "last_run": None,
    "last_result": None,      # "success" | "skipped" | "error"
    "next_run": None,
    "error": None,
}


def _calc_next_nightly(hour: int) -> str:
    """Calculate the next run datetime for the given hour (local time)."""
    now = datetime.now()
    today_target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= today_target:
        from datetime import timedelta
        next_run = today_target + timedelta(days=1)
    else:
        next_run = today_target
    return next_run.isoformat()


async def _nightly_scheduler():
    """Background task: run diarize + combined tag at NIGHTLY_SCHEDULE_HOUR."""
    hour = NIGHTLY_SCHEDULE_HOUR
    if hour <= 0 or hour > 23:
        return

    _nightly_state["enabled"] = True
    _nightly_state["hour"] = hour
    _nightly_state["next_run"] = _calc_next_nightly(hour)
    log.info("Nightly scheduler: enabled (hour=%d, next=%s)", hour, _nightly_state["next_run"])

    while True:
        # Sleep until next target time
        now = datetime.now()
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if now >= target:
            from datetime import timedelta
            target = target + timedelta(days=1)

        _nightly_state["next_run"] = target.isoformat()
        sleep_seconds = (target - now).total_seconds()
        log.debug("Nightly scheduler: sleeping %.0f seconds until %s", sleep_seconds, target.isoformat())

        await asyncio.sleep(sleep_seconds)

        # Check if any batch is already running
        if (_batch_state.get("running")
                or _batch_combined_tag_state.get("running")
                or _batch_session_tag_state.get("running")):
            log.info("Nightly scheduler: skipped (another batch is running)")
            _nightly_state["last_run"] = datetime.now().isoformat()
            _nightly_state["last_result"] = "skipped"
            continue

        log.info("Nightly scheduler: starting (hour=%d)", hour)
        _nightly_state["running"] = True
        _nightly_state["error"] = None

        try:
            # Phase 1: Diarize
            if diarizer is not None:
                _nightly_state["phase"] = "diarize"
                log.info("Nightly scheduler: Phase 1 — diarize")
                await _batch_diarize_task(force=False)

            # Phase 2: Combined tag (voice tag + session tag)
            _nightly_state["phase"] = "tag"
            log.info("Nightly scheduler: Phase 2 — combined tag (voice → session)")
            await _batch_combined_tag_task(force=False)

            _nightly_state["last_result"] = "success"
        except Exception as e:
            log.error("Nightly scheduler: error: %s", e, exc_info=True)
            _nightly_state["last_result"] = "error"
            _nightly_state["error"] = str(e)
        finally:
            _nightly_state["running"] = False
            _nightly_state["phase"] = None
            _nightly_state["last_run"] = datetime.now().isoformat()
            _nightly_state["next_run"] = _calc_next_nightly(hour)
            log.info(
                "Nightly scheduler: finished (result=%s, next=%s)",
                _nightly_state["last_result"], _nightly_state["next_run"],
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
    last_speech_time = time.time()  # 最後に発話を検出した時刻 (セッション管理用)
    last_vad_check_len = 0  # 前回 VAD チェック時のバッファ長 (bytes)

    # VAD オプション (Silero VAD)
    vad_options = VadOptions(
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=250,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )

    # Hotwords (デコーダバイアス)
    session_hotwords = ""
    if HOTWORDS_ENABLED and hotword_loader.get_settings().get("enabled", True):
        session_hotwords = hotword_loader.get_hotwords_string()
        if session_hotwords:
            log.info(f"Hotwords: {hotword_loader.get_count()} terms active")

    # Realtime voice filter: load embedding model if enabled + user speakers exist
    voice_filter_inference = None
    if VOICE_FILTER_ENABLED and speaker_filter and speaker_filter.enabled:
        user_speakers = speaker_filter.get_user_speakers()
        if user_speakers:
            try:
                voice_filter_inference = speaker_filter._load_model()
                log.info(f"Voice filter: ENABLED ({len(user_speakers)} user speaker(s), "
                         f"threshold={VOICE_FILTER_THRESHOLD or speaker_filter.get_threshold()})")
            except Exception as e:
                log.warning(f"Voice filter: model load failed, continuing without filter: {e}")
                voice_filter_inference = None
        else:
            log.warning("Voice filter: ENABLED but no user speakers with embeddings registered")

    # モデル確保 (未ロードなら自動ロード)
    # バッチ処理中 (diarize/speaker-tag) は gpu_lock が保持されるため
    # タイムアウトで早期にリジェクトする
    try:
        model, executor = await model_manager.acquire(timeout=30)
    except TimeoutError:
        log.warning(f"WebSocket rejected: GPU busy (batch operation in progress) [{remote}]")
        await websocket.close(1013, "GPU busy — batch operation in progress")
        return
    except Exception as e:
        log.error(f"Failed to acquire model: {e}", exc_info=True)
        await websocket.close(1013, "Model unavailable")
        return

    async def _transcribe_chunk(chunk: bytes):
        """チャンクを文字起こしして結果を送信する共通処理。"""
        nonlocal total_audio_seconds, recorder_audio_seconds, recorder

        chunk_duration = len(chunk) / (SAMPLE_RATE * 2)

        # 録音セッションがなければ新規作成
        if recorder is None:
            recorder = SessionRecorder()
            recorder_audio_seconds = 0.0
            log.info("New recording session started (speech detected)")

        # オフセットを退避 (transcribe 前の値を使う)
        ws_offset = total_audio_seconds
        rec_offset = recorder_audio_seconds

        # 音声を録音ファイルに書き込み
        recorder.write_audio(chunk)

        log.info(f"Transcribing {chunk_duration:.1f}s of audio...")
        t0 = time.time()

        loop = asyncio.get_running_loop()
        raw_segments = await loop.run_in_executor(
            executor,
            lambda: transcribe_audio(chunk, model, 0.0, session_hotwords),
        )

        elapsed = time.time() - t0
        total_audio_seconds += chunk_duration
        recorder_audio_seconds += chunk_duration

        # Realtime voice filter: remove non-user segments
        if raw_segments and voice_filter_inference:
            audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            raw_segments = speaker_filter.filter_segments_realtime(
                voice_filter_inference, audio_np, raw_segments,
                sample_rate=SAMPLE_RATE, threshold=VOICE_FILTER_THRESHOLD,
            )

        if raw_segments:
            # 録音ログ: WAV ファイル内の位置に合わせたオフセット (先に保存)
            rec_segments = _offset_segments(raw_segments, rec_offset)
            recorder.add_segments(rec_segments)

            log.info(
                f"Transcribed {len(raw_segments)} segment(s) "
                f"[{elapsed:.2f}s processing, "
                f"RTF={elapsed/chunk_duration:.2f}x]"
            )

            # WebSocket 応答: 接続全体のオフセット (Omi アプリ用)
            ws_segments = _offset_segments(raw_segments, ws_offset)
            response = {"segments": ws_segments}
            await websocket.send(json.dumps(response))
        else:
            log.debug(f"No speech detected in chunk ({elapsed:.2f}s)")

    async def _finalize_and_enqueue(rec):
        """Finalize recorder and enqueue for batch LLM correction."""
        rec.finalize()

        # Enqueue for async LLM correction + tagging (Layer 3)
        if batch_corrector and rec.enabled and not rec._finalized:
            pass  # finalize failed or empty session
        elif (batch_corrector and rec.enabled
              and hasattr(rec, 'session_dir') and rec.session_dir
              and len(rec.all_segments) > 0):
            session_id = str(rec.session_dir.relative_to(RECORDINGS_DIR))
            batch_corrector.enqueue(session_id)

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

            if VAD_SEG_ENABLED:
                # ============================================================
                # VAD ベースのセグメンテーション
                # ============================================================

                # VAD チェック間隔: 前回チェックから十分な新規音声が溜まったか
                new_audio_sec = (len(audio_buffer) - last_vad_check_len) / (SAMPLE_RATE * 2)
                if new_audio_sec < VAD_CHECK_INTERVAL:
                    continue
                last_vad_check_len = len(audio_buffer)

                # バッファを float32 に変換して VAD 実行
                audio_float = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
                speech_timestamps = get_speech_timestamps(audio_float, vad_options)

                if not speech_timestamps:
                    # 発話なし — セッションタイムアウト判定
                    elapsed_silence = time.time() - last_speech_time
                    if elapsed_silence >= SILENCE_TIMEOUT and recorder is not None:
                        await _finalize_and_enqueue(recorder)
                        log.info(f"Session finalized due to silence timeout ({elapsed_silence:.0f}s)")
                        recorder = None

                    # バッファが上限を超えたら破棄 (無音が溜まり続けるのを防ぐ)
                    if buffer_duration >= VAD_MAX_BUFFER:
                        total_audio_seconds += buffer_duration
                        audio_buffer.clear()
                        last_vad_check_len = 0
                        log.debug(f"Silent buffer discarded ({buffer_duration:.1f}s)")
                    continue

                # 発話あり — タイムアウトリセット
                last_speech_time = time.time()

                # 末尾の無音時間を計算
                last_speech_end_sample = speech_timestamps[-1]["end"]
                last_speech_end_sec = last_speech_end_sample / SAMPLE_RATE
                trailing_silence = buffer_duration - last_speech_end_sec

                # 発話が完了した (十分な末尾無音がある) またはバッファ上限
                if trailing_silence >= VAD_SILENCE_TRIGGER or buffer_duration >= VAD_MAX_BUFFER:
                    # 発話終了地点 + パディングまで切り出し
                    cut_sec = min(last_speech_end_sec + 0.3, buffer_duration)
                    cut_bytes = int(cut_sec * SAMPLE_RATE) * 2
                    # 偶数バイトにアラインメント (int16)
                    cut_bytes = cut_bytes - (cut_bytes % 2)

                    chunk = bytes(audio_buffer[:cut_bytes])
                    remaining = audio_buffer[cut_bytes:]
                    audio_buffer = bytearray(remaining)
                    last_vad_check_len = 0

                    if buffer_duration >= VAD_MAX_BUFFER:
                        log.info(f"VAD: Buffer limit reached ({buffer_duration:.1f}s), force transcribing")
                    else:
                        log.info(f"VAD: Utterance complete ({last_speech_end_sec:.1f}s speech, "
                                 f"{trailing_silence:.1f}s trailing silence)")

                    await _transcribe_chunk(chunk)

            else:
                # ============================================================
                # フォールバック: 固定時間チャンク (旧方式)
                # ============================================================
                if buffer_duration >= BUFFER_SECONDS:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()

                    # RMS 無音検出
                    if SILENCE_THRESHOLD > 0:
                        rms = calc_rms_int16(chunk)
                        if rms < SILENCE_THRESHOLD:
                            elapsed_silence = time.time() - last_speech_time
                            if elapsed_silence >= SILENCE_TIMEOUT and recorder is not None:
                                await _finalize_and_enqueue(recorder)
                                log.info(f"Session finalized due to silence timeout ({elapsed_silence:.0f}s)")
                                recorder = None
                            total_audio_seconds += buffer_duration
                            continue
                        last_speech_time = time.time()

                    await _transcribe_chunk(chunk)

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
                    raw_segments = transcribe_audio(remaining, model, 0.0, session_hotwords)
                    # Realtime voice filter on remaining buffer
                    if raw_segments and voice_filter_inference:
                        audio_np = np.frombuffer(remaining, dtype=np.int16).astype(np.float32) / 32768.0
                        raw_segments = speaker_filter.filter_segments_realtime(
                            voice_filter_inference, audio_np, raw_segments,
                            sample_rate=SAMPLE_RATE, threshold=VOICE_FILTER_THRESHOLD,
                        )
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

        # Voice filter embedding model unload
        if voice_filter_inference is not None:
            speaker_filter._unload_model(voice_filter_inference)
            voice_filter_inference = None

        # 話者分離 (バックグラウンドで実行、新セッションをブロックしない)
        if (DIARIZATION_ENABLED and diarizer and diarizer.enabled
                and recorder is not None
                and recorder.enabled
                and not recorder._finalized
                and recorder.total_bytes > 0):
            recorder.close_audio()
            asyncio.create_task(_run_diarization_bg(recorder))
        else:
            # diarization なし → タグ付け + finalize
            if recorder is not None:
                await _finalize_and_enqueue(recorder)

        duration = time.time() - session_start
        log.info(
            f"Session ended: {remote} "
            f"(duration={duration:.0f}s, audio={total_audio_seconds:.0f}s)"
        )


# ============================================================
# メイン
# ============================================================

async def main():
    global batch_corrector
    log.info(f"Starting Omi-Whisper bridge on ws://{HOST}:{PORT}")
    log.info(f"Model: {WHISPER_MODEL}, Compute: {COMPUTE_TYPE}, Language: {LANGUAGE}")
    log.info(f"Buffer: {BUFFER_SECONDS}s, VAD: {VAD_FILTER}, Beam: {BEAM_SIZE}")
    log.info(f"Quality: no_repeat_ngram={NO_REPEAT_NGRAM_SIZE}, patience={PATIENCE}, "
             f"log_prob_thresh={LOG_PROB_THRESHOLD}, "
             f"no_speech_thresh={NO_SPEECH_THRESHOLD}, "
             f"compression_ratio_thresh={COMPRESSION_RATIO_THRESHOLD}")
    log.info(f"Config dir: {CONFIG_DIR}")
    log.info(f"Term corrections: {_TERM_CORRECTIONS_PATH} ({len(term_corrector.list_corrections())} entries)")
    if batch_corrector:
        log.info(f"Layer 3 LLM Batch: {LLM_MODEL} (stages={LLM_PIPELINE_STAGES}, "
                 f"mode={LLM_BATCH_MODE}, screening={LLM_SCREENING_ENABLED}, "
                 f"auto_c={STAGE_C_AUTO}, "
                 f"min_text={LLM_MIN_TEXT_LENGTH}, timeout={LLM_TIMEOUT}s)")
    elif llm_pipeline:
        log.info(f"Layer 3 LLM Pipeline: {LLM_MODEL} (stages={LLM_PIPELINE_STAGES}, no batch)")
    else:
        reason = "DISABLED" if not LLM_ENABLED else "no GEMINI_API_KEY"
        log.info(f"Layer 3 LLM: {reason}")
    if diarizer and diarizer.enabled:
        auto = "auto-ON" if DIARIZATION_ENABLED else "manual-only"
        log.info(f"Diarization: AVAILABLE ({auto})")
    else:
        log.info("Diarization: UNAVAILABLE (no HF_AUTH_TOKEN)")
    log.info(f"Normalize: {'ON (' + str(NORMALIZE_TARGET_DBFS) + 'dBFS)' if NORMALIZE_ENABLED else 'OFF'}")
    log.info(f"Noise reduce: {'ON (prop=' + str(NOISE_REDUCE_PROP_DECREASE) + ')' if NOISE_REDUCE_ENABLED else 'OFF'}")
    if HOTWORDS_ENABLED:
        hw_settings = hotword_loader.get_settings()
        if hw_settings.get("enabled", True):
            log.info(f"Hotwords: {hotword_loader.get_count()} terms loaded "
                     f"({len(hotword_loader.get_custom_words())} custom)")
        else:
            log.info("Hotwords: DISABLED (settings)")
    else:
        log.info("Hotwords: DISABLED (env)")
    if VOICE_FILTER_ENABLED:
        if speaker_filter and speaker_filter.enabled:
            user_count = len(speaker_filter.get_user_speakers())
            if user_count > 0:
                log.info(f"Voice filter: ENABLED ({user_count} user speaker(s), "
                         f"threshold={VOICE_FILTER_THRESHOLD or speaker_filter.get_threshold()})")
            else:
                log.info("Voice filter: ENABLED but no user speakers with embeddings")
        else:
            log.info("Voice filter: ENABLED but speaker filter unavailable")
    else:
        log.info("Voice filter: DISABLED")
    if speaker_filter and speaker_filter.enabled:
        sf_status = speaker_filter.get_status()
        log.info(f"Speaker filter: AVAILABLE ({sf_status['speaker_count']} speakers, "
                 f"threshold={sf_status['threshold']})")
    else:
        log.info("Speaker filter: UNAVAILABLE (no HF_AUTH_TOKEN)")
    if session_tagger and session_tagger.enabled:
        log.info(f"Session tagger: ON (model={TAGGING_MODEL})")
    else:
        reason = "DISABLED" if not TAGGING_ENABLED else "no GEMINI_API_KEY"
        log.info(f"Session tagger: {reason}")
    if VAD_SEG_ENABLED:
        log.info(f"VAD segmentation: ON (check={VAD_CHECK_INTERVAL}s, "
                 f"silence_trigger={VAD_SILENCE_TRIGGER}s, max_buffer={VAD_MAX_BUFFER}s)")
    else:
        log.info(f"VAD segmentation: OFF (fixed {BUFFER_SECONDS}s chunks)")
    log.info(f"Audio saving: {'ON -> ' + RECORDINGS_DIR if SAVE_AUDIO else 'OFF'}")
    log.info(f"Silence: threshold={SILENCE_THRESHOLD}, timeout={SILENCE_TIMEOUT}s"
             f"{' (DISABLED)' if SILENCE_THRESHOLD <= 0 else ''}")
    log.info(f"Auth: {'TOKEN required' if AUTH_TOKEN else 'DISABLED (LAN only)'}")
    log.info(f"Management API: http://0.0.0.0:{MANAGEMENT_PORT}")
    log.info("Model will be loaded on-demand (GPU not used until first connection)")

    # 起動時に古い録音を削除
    cleanup_old_recordings()

    # Batch corrector initialization (needs known_terms which depend on term_corrector + hotword_loader)
    if llm_pipeline:
        known_terms = _get_known_terms()

        # Segment screener: deterministic pre-processing to minimize LLM calls
        segment_screener = None
        if LLM_SCREENING_ENABLED:
            suggestions_path = os.path.join(CONFIG_DIR, "pending_suggestions.json")
            suggestion_learner = SuggestionLearner(suggestions_path)
            segment_screener = SegmentScreener(
                term_corrector=term_corrector,
                suggestion_learner=suggestion_learner,
                hotword_loader=hotword_loader,
                min_text_length=LLM_MIN_TEXT_LENGTH,
            )
            log.info(
                "Segment screener: ON (learned_patterns=%d)",
                suggestion_learner.pattern_count,
            )

        batch_corrector = BatchCorrector(
            recordings_dir=RECORDINGS_DIR,
            config_dir=CONFIG_DIR,
            pipeline=llm_pipeline,
            known_terms=known_terms,
            hotword_loader=hotword_loader,
            stage_c_auto=STAGE_C_AUTO,
            min_text_length=LLM_MIN_TEXT_LENGTH,
            mode=LLM_BATCH_MODE,
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=LLM_MODEL,
            segment_screener=segment_screener,
            recovery_scan_days=RECOVERY_SCAN_DAYS,
        )
        await batch_corrector.start()
        log.info("Batch LLM corrector: worker started (mode=%s, screening=%s)",
                 LLM_BATCH_MODE, LLM_SCREENING_ENABLED)

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

        # force パラメータ: 既に diarized なセッションも再処理
        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass

        # 対象セッション数を事前チェック
        sessions = _find_undiarized_sessions(force=force)
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already diarized",
            })

        asyncio.create_task(_batch_diarize_task(force=force))
        return web.json_response({"status": "started", "total": len(sessions)})

    async def handle_batch_status(request):
        return web.json_response(_batch_state)

    async def handle_batch_cancel(request):
        if _batch_state["running"]:
            _batch_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    # --- Hotwords 管理 API ---

    async def handle_hotwords_status(request):
        settings = hotword_loader.get_settings()
        return web.json_response({
            "enabled": HOTWORDS_ENABLED and settings.get("enabled", True),
            "env_enabled": HOTWORDS_ENABLED,
            "count": hotword_loader.get_count(),
            "custom": hotword_loader.get_custom_words(),
            "custom_count": len(hotword_loader.get_custom_words()),
            "categories": hotword_loader.get_category_stats(),
        })

    async def handle_hotwords_settings_get(request):
        return web.json_response(hotword_loader.get_settings())

    async def handle_hotwords_settings_put(request):
        body = await request.json()
        hotword_loader.update_settings(body)
        return web.json_response({"status": "ok", "count": hotword_loader.get_count()})

    async def handle_hotwords_custom_post(request):
        body = await request.json()
        word = str(body.get("word", "")).strip()
        if not word:
            return web.json_response({"error": "word is required"}, status=400)
        hotword_loader.add_custom(word)
        return web.json_response({"status": "ok", "custom": hotword_loader.get_custom_words()})

    async def handle_hotwords_custom_delete(request):
        body = await request.json()
        word = str(body.get("word", "")).strip()
        if not word:
            return web.json_response({"error": "word is required"}, status=400)
        hotword_loader.remove_custom(word)
        return web.json_response({"status": "ok", "custom": hotword_loader.get_custom_words()})

    # --- Speaker 管理 API ---

    async def handle_speakers_status(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        return web.json_response(speaker_filter.get_status())

    async def handle_speakers_list(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        return web.json_response(speaker_filter.list_speakers())

    async def handle_speakers_list_detailed(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speakers = speaker_filter.list_speakers_detailed()

        # Enrich legacy samples that lack audio_start/audio_end
        for sp in speakers:
            for sample in sp.get("samples", []):
                if "audio_start" in sample:
                    continue
                source = sample.get("source_session", "")
                # Strip " (correction)" suffix
                source = source.replace(" (correction)", "").strip()
                if not source or "/" not in source:
                    continue
                parts = source.split("/")
                if len(parts) != 2:
                    continue
                transcript_path = Path(RECORDINGS_DIR) / parts[0] / parts[1] / "transcript.json"
                try:
                    with open(transcript_path, encoding="utf-8") as f:
                        data = json.load(f)
                    segments = data.get("segments", [])
                    if not segments:
                        continue
                    # Try to find segment(s) matching the sample duration
                    target_dur = sample.get("duration", 0)
                    if target_dur > 0:
                        # Single-segment: find best match by duration
                        best, best_diff = None, float("inf")
                        for seg in segments:
                            seg_dur = seg.get("end", 0) - seg.get("start", 0)
                            diff = abs(seg_dur - target_dur)
                            if diff < best_diff:
                                best_diff = diff
                                best = seg
                        if best and best_diff < 1.0:
                            sample["audio_start"] = best["start"]
                            sample["audio_end"] = best["end"]
                            continue
                    # Fallback: entire session
                    sample["audio_start"] = segments[0].get("start", 0)
                    sample["audio_end"] = segments[-1].get("end", 0)
                except Exception:
                    pass

        return web.json_response(speakers)

    async def handle_speakers_tag_correction(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        body = await request.json()
        session_path = str(body.get("session", "")).strip()
        segment_index = body.get("segment_index")
        new_speaker_id = str(body.get("speaker_id", "")).strip()

        if not session_path or segment_index is None or not new_speaker_id:
            return web.json_response(
                {"error": "session, segment_index, and speaker_id are required"}, status=400,
            )

        parts = session_path.split("/")
        if len(parts) != 2:
            return web.json_response({"error": "invalid session path"}, status=400)

        session_dir = Path(RECORDINGS_DIR) / parts[0] / parts[1]
        transcript_path = session_dir / "transcript.json"
        if not transcript_path.exists():
            return web.json_response({"error": "session not found"}, status=404)

        try:
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return web.json_response({"error": "transcript read error"}, status=500)

        segments = data.get("segments", [])
        if segment_index < 0 or segment_index >= len(segments):
            return web.json_response({"error": "segment_index out of range"}, status=400)

        seg = segments[segment_index]

        if new_speaker_id == "unknown":
            seg["person_id"] = "unknown"
            seg["person_name"] = "unknown"
            seg["is_user"] = False
            seg["speaker_similarity"] = 0
            seg["manually_tagged"] = True
        else:
            sp = speaker_filter._find_speaker(new_speaker_id)
            if sp is None:
                return web.json_response({"error": "speaker not found"}, status=404)
            seg["person_id"] = sp["id"]
            seg["person_name"] = sp["name"]
            seg["is_user"] = sp.get("is_user", False)
            seg["speaker_similarity"] = 1.0
            seg["manually_tagged"] = True

        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        result = {
            "status": "ok",
            "segment_index": segment_index,
            "person_id": seg["person_id"],
            "person_name": seg["person_name"],
            "is_user": seg["is_user"],
            "manually_tagged": True,
        }

        # Auto-learn: add voiceprint from correction (best-effort)
        if new_speaker_id != "unknown":
            seg_duration = float(seg.get("end", 0)) - float(seg.get("start", 0))
            if seg_duration >= 0.5:
                try:
                    audio_file = data.get("session", {}).get("audio_file", "audio.ogg")
                    audio_path = session_dir / audio_file
                    if not audio_path.exists():
                        for ext in ("audio.ogg", "audio.wav"):
                            alt = session_dir / ext
                            if alt.exists():
                                audio_path = alt
                                break

                    if audio_path.exists():
                        loop = asyncio.get_running_loop()
                        async with model_manager.gpu_lock:
                            await model_manager.unload_model()
                            sample = await loop.run_in_executor(
                                None,
                                lambda: speaker_filter.add_voiceprint(
                                    new_speaker_id, str(audio_path),
                                    [{"start": seg["start"], "end": seg["end"]}],
                                    session_path + " (correction)",
                                ),
                            )
                        if sample:
                            result["auto_added_sample"] = {
                                "id": sample["id"],
                                "duration": sample.get("duration", 0),
                            }
                            log.info(f"Auto-added voiceprint from correction: {new_speaker_id}")
                except Exception as e:
                    log.warning(f"Auto-add voiceprint from correction failed (non-fatal): {e}")

        log.info(f"Tag correction: {session_path}[{segment_index}] → {seg['person_name']}")
        return web.json_response(result)

    async def handle_speakers_add(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        body = await request.json()
        name = str(body.get("name", "")).strip()
        if not name:
            return web.json_response({"error": "name is required"}, status=400)
        is_user = bool(body.get("is_user", False))
        speaker = speaker_filter.add_speaker(name, is_user=is_user)
        return web.json_response(speaker, status=201)

    async def handle_speakers_update(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speaker_id = request.match_info["id"]
        body = await request.json()
        name = body.get("name")
        is_user = body.get("is_user")
        result = speaker_filter.update_speaker(speaker_id, name=name, is_user=is_user)
        if result is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "ok"})

    async def handle_speakers_delete(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speaker_id = request.match_info["id"]
        ok = speaker_filter.remove_speaker(speaker_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "ok"})

    async def handle_speakers_add_voiceprint(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speaker_id = request.match_info["id"]
        body = await request.json()
        session_path = str(body.get("session", "")).strip()
        segment_indices = body.get("segment_indices", [])

        if not session_path:
            return web.json_response({"error": "session is required"}, status=400)
        if not segment_indices:
            return web.json_response({"error": "segment_indices is required"}, status=400)

        # セッションの音声・トランスクリプトを特定
        parts = session_path.split("/")
        if len(parts) != 2:
            return web.json_response({"error": "invalid session path"}, status=400)

        session_dir = Path(RECORDINGS_DIR) / parts[0] / parts[1]
        transcript_path = session_dir / "transcript.json"
        if not transcript_path.exists():
            return web.json_response({"error": "session not found"}, status=404)

        try:
            with open(transcript_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return web.json_response({"error": "transcript read error"}, status=500)

        audio_file = data.get("session", {}).get("audio_file", "audio.ogg")
        audio_path = session_dir / audio_file
        if not audio_path.exists():
            for ext in ("audio.ogg", "audio.wav"):
                alt = session_dir / ext
                if alt.exists():
                    audio_path = alt
                    break
            else:
                return web.json_response({"error": "audio not found"}, status=404)

        all_segments = data.get("segments", [])
        selected_segments = []
        for idx in segment_indices:
            if 0 <= idx < len(all_segments):
                selected_segments.append(all_segments[idx])

        if not selected_segments:
            return web.json_response({"error": "no valid segments selected"}, status=400)

        loop = asyncio.get_running_loop()
        try:
            async with model_manager.gpu_lock:
                await model_manager.unload_model()
                sample = await loop.run_in_executor(
                    None,
                    lambda: speaker_filter.add_voiceprint(
                        speaker_id, str(audio_path), selected_segments, session_path
                    ),
                )
        except Exception as e:
            log.error(f"Add voiceprint error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

        if sample is None:
            return web.json_response({"error": "failed to extract voiceprint"}, status=500)
        return web.json_response(sample, status=201)

    async def handle_speakers_remove_voiceprint(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speaker_id = request.match_info["id"]
        sample_id = request.match_info["sample_id"]
        result = speaker_filter.remove_voiceprint(speaker_id, sample_id)
        if result is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(result)

    async def handle_speakers_threshold(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        body = await request.json()
        threshold = float(body.get("threshold", 0.55))
        speaker_filter.update_threshold(threshold)
        return web.json_response({"status": "ok", "threshold": speaker_filter.get_threshold()})

    async def handle_speakers_tag_start(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        if _batch_tag_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409,
            )

        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass

        sessions = _find_untagged_sessions(force=force)
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        asyncio.create_task(_batch_tag_task(force=force))
        return web.json_response({"status": "started", "total": len(sessions)})

    async def handle_speakers_tag_status(request):
        return web.json_response(_batch_tag_state)

    async def handle_speakers_tag_cancel(request):
        if _batch_tag_state["running"]:
            _batch_tag_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    async def handle_speakers_delete_all(request):
        if speaker_filter is None:
            return web.json_response(
                {"error": "HF_AUTH_TOKEN not configured"}, status=503,
            )
        speaker_filter.delete_all()
        return web.json_response({"status": "ok"})

    # --- User Tags reload API ---

    async def handle_user_tags_reload(request):
        if session_tagger is not None:
            session_tagger.reload_user_tags()
        return web.json_response({"status": "ok"})

    # --- Taxonomy reload API ---

    async def handle_taxonomy_reload(request):
        if session_tagger is not None:
            session_tagger.reload_taxonomy()
        return web.json_response({"status": "ok"})

    # --- Session Tagging バッチ API ---

    async def handle_session_tag_start(request):
        if session_tagger is None:
            return web.json_response(
                {"error": "GEMINI_API_KEY not configured or tagging disabled"}, status=503,
            )
        if _batch_session_tag_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409,
            )

        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass

        sessions = _find_untagged_session_tag_sessions(force=force)
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        asyncio.create_task(_batch_session_tag_task(force=force))
        return web.json_response({"status": "started", "total": len(sessions)})

    async def handle_session_tag_status(request):
        return web.json_response(_batch_session_tag_state)

    async def handle_session_tag_cancel(request):
        if _batch_session_tag_state["running"]:
            _batch_session_tag_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    # --- Combined Tag バッチ API (Voice Tag → Session Tag) ---

    async def handle_combined_tag_start(request):
        if _batch_combined_tag_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409,
            )
        # At least one tagger must be available
        if session_tagger is None and speaker_filter is None:
            return web.json_response(
                {"error": "No tagger available (need GEMINI_API_KEY or HF_AUTH_TOKEN)"}, status=503,
            )

        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass

        # Check if there's anything to do
        st_sessions = _find_untagged_session_tag_sessions(force=force) if session_tagger else []
        vt_sessions = _find_untagged_sessions(force=force) if speaker_filter else []

        if not st_sessions and not vt_sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        asyncio.create_task(_batch_combined_tag_task(force=force))
        return web.json_response({
            "status": "started",
            "total": len(st_sessions) + len(vt_sessions),
        })

    async def handle_combined_tag_status(request):
        return web.json_response(_batch_combined_tag_state)

    async def handle_combined_tag_cancel(request):
        if _batch_combined_tag_state["running"]:
            _batch_combined_tag_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    # ============================================================
    # Batch sanitize (training data anonymization)
    # ============================================================

    _batch_sanitize_state = {
        "running": False,
        "total": 0,
        "processed": 0,
        "failed": 0,
        "dropped": 0,
        "records_generated": 0,
        "current_session": None,
        "errors": [],
        "cancel_requested": False,
        "started_at": None,
        "finished_at": None,
        "output_file": None,
    }

    async def _batch_sanitize_task(session_paths: list):
        """Batch sanitize sessions for safe training data export."""
        _batch_sanitize_state["running"] = True
        _batch_sanitize_state["total"] = len(session_paths)
        _batch_sanitize_state["processed"] = 0
        _batch_sanitize_state["failed"] = 0
        _batch_sanitize_state["dropped"] = 0
        _batch_sanitize_state["records_generated"] = 0
        _batch_sanitize_state["current_session"] = None
        _batch_sanitize_state["errors"] = []
        _batch_sanitize_state["cancel_requested"] = False
        _batch_sanitize_state["started_at"] = datetime.now().isoformat()
        _batch_sanitize_state["finished_at"] = None
        _batch_sanitize_state["output_file"] = None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"safe_training_{ts}.jsonl"
        output_path = Path(CONFIG_DIR) / output_filename
        _batch_sanitize_state["output_file"] = output_filename

        loop = asyncio.get_running_loop()
        try:
            with open(output_path, "w", encoding="utf-8") as out_f:
                for sess_path in session_paths:
                    if _batch_sanitize_state["cancel_requested"]:
                        log.info("Batch sanitize cancelled by user")
                        break

                    _batch_sanitize_state["current_session"] = sess_path
                    parts = sess_path.split("/")
                    if len(parts) != 2:
                        _batch_sanitize_state["failed"] += 1
                        _batch_sanitize_state["errors"].append(
                            f"Invalid session path: {sess_path}"
                        )
                        continue

                    transcript_path = (
                        Path(RECORDINGS_DIR) / parts[0] / parts[1] / "transcript.json"
                    )
                    if not transcript_path.exists():
                        _batch_sanitize_state["failed"] += 1
                        _batch_sanitize_state["errors"].append(
                            f"transcript.json not found: {sess_path}"
                        )
                        continue

                    try:
                        with open(transcript_path, encoding="utf-8") as f:
                            transcript = json.load(f)
                    except (json.JSONDecodeError, OSError) as e:
                        _batch_sanitize_state["failed"] += 1
                        _batch_sanitize_state["errors"].append(
                            f"{sess_path}: {e}"
                        )
                        continue

                    try:
                        if responses_sanitizer is not None:
                            # Responses API: transcript → raw_text を構築
                            raw_text = _build_raw_text_from_transcript(transcript)
                            rs_result = await loop.run_in_executor(
                                None,
                                responses_sanitizer.sanitize,
                                sess_path, raw_text, "ja",
                            )
                            if rs_result is None:
                                _batch_sanitize_state["failed"] += 1
                                continue
                            # Responses API の出力を JSONL に書き込み
                            out_rec = {"session_id": sess_path, **rs_result.data}
                            if rs_result.data.get("risk") == "high":
                                _batch_sanitize_state["dropped"] += 1
                                _batch_sanitize_state["processed"] += 1
                                log.info(f"Sanitize: {sess_path} → high risk, dropped")
                                continue
                            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                            _batch_sanitize_state["records_generated"] += 1
                            _batch_sanitize_state["processed"] += 1
                            log.info(
                                f"Sanitize: {sess_path} → {rs_result.data.get('risk', '?')}"
                            )
                        else:
                            # Chat Completions API (既存)
                            result = await loop.run_in_executor(
                                None, training_sanitizer.sanitize_session, transcript
                            )
                            if result is None:
                                _batch_sanitize_state["failed"] += 1
                                continue
                            if result["risk_level"] == "high":
                                _batch_sanitize_state["dropped"] += 1
                                _batch_sanitize_state["processed"] += 1
                                log.info(f"Sanitize: {sess_path} → high risk, dropped")
                                continue
                            for rec in result["records"]:
                                rec["session_id"] = sess_path
                                rec["risk_level"] = result["risk_level"]
                                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                _batch_sanitize_state["records_generated"] += 1
                            _batch_sanitize_state["processed"] += 1
                            log.info(
                                f"Sanitize: {sess_path} → {result['risk_level']}, "
                                f"{len(result['records'])} records"
                            )
                    except Exception as e:
                        _batch_sanitize_state["failed"] += 1
                        _batch_sanitize_state["errors"].append(
                            f"{sess_path}: {e}"
                        )
                        log.error(f"Sanitize error for {sess_path}: {e}")
                        continue
                    await asyncio.sleep(0)

        except Exception as e:
            log.error(f"Batch sanitize error: {e}", exc_info=True)
            _batch_sanitize_state["errors"].append(str(e))
        finally:
            _batch_sanitize_state["running"] = False
            _batch_sanitize_state["current_session"] = None
            _batch_sanitize_state["finished_at"] = datetime.now().isoformat()
            log.info(
                f"Batch sanitize finished: "
                f"{_batch_sanitize_state['processed']} processed, "
                f"{_batch_sanitize_state['failed']} failed, "
                f"{_batch_sanitize_state['dropped']} dropped, "
                f"{_batch_sanitize_state['records_generated']} records"
            )

    async def handle_sanitize_start(request):
        if training_sanitizer is None and responses_sanitizer is None:
            return web.json_response(
                {"error": "No sanitizer configured (set GEMINI_API_KEY or OPENAI_API_KEY)"}, status=503
            )
        if _batch_sanitize_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409
            )

        try:
            body = await request.json()
            sessions = body.get("sessions", [])
        except Exception:
            return web.json_response(
                {"error": "invalid request body"}, status=400
            )

        if not sessions or not isinstance(sessions, list):
            return web.json_response(
                {"error": "sessions list required"}, status=400
            )

        asyncio.create_task(_batch_sanitize_task(sessions))
        return web.json_response({
            "status": "started",
            "total": len(sessions),
        })

    async def handle_audio_scene_status(request):
        return web.json_response({
            "enabled": AUDIO_SCENE_ENABLED,
            "device": AUDIO_SCENE_DEVICE,
            "threshold": AUDIO_SCENE_THRESHOLD,
            "model": "PANNs-CNN14",
            "batch_state": {
                "audio_scene_total": _batch_combined_tag_state.get("audio_scene_total", 0),
                "audio_scene_processed": _batch_combined_tag_state.get("audio_scene_processed", 0),
                "audio_scene_failed": _batch_combined_tag_state.get("audio_scene_failed", 0),
            }
        })

    async def handle_sanitize_status(request):
        return web.json_response(_batch_sanitize_state)

    async def handle_sanitize_cancel(request):
        if _batch_sanitize_state["running"]:
            _batch_sanitize_state["cancel_requested"] = True
        return web.json_response({"status": "cancel_requested"})

    async def handle_sanitize_download(request):
        filename = request.query.get("file", "")
        # Security: only allow safe_training_*.jsonl pattern
        if not re.match(r'^safe_training_\d{8}_\d{6}\.jsonl$', filename):
            return web.json_response(
                {"error": "invalid filename"}, status=400
            )
        filepath = Path(CONFIG_DIR) / filename
        if not filepath.exists():
            return web.json_response(
                {"error": "file not found"}, status=404
            )
        return web.FileResponse(
            filepath,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    # --- Batch LLM correction API ---
    async def handle_batch_llm_status(request):
        if not batch_corrector:
            return web.json_response({"error": "LLM pipeline not configured"}, status=503)
        return web.json_response(batch_corrector.get_status())

    async def handle_batch_llm_jobs(request):
        if not batch_corrector:
            return web.json_response({"error": "LLM pipeline not configured"}, status=503)
        status_filter = request.query.get("status")
        limit = int(request.query.get("limit", "50"))
        return web.json_response(batch_corrector.get_jobs(status=status_filter, limit=limit))

    async def handle_batch_llm_retry(request):
        if not batch_corrector:
            return web.json_response({"error": "LLM pipeline not configured"}, status=503)
        session_id = None
        try:
            body = await request.json()
            session_id = body.get("session_id")
        except Exception:
            pass
        count = batch_corrector.retry_failed(session_id)
        return web.json_response({"retried": count})

    async def handle_batch_llm_process(request):
        if not batch_corrector:
            return web.json_response({"error": "LLM pipeline not configured"}, status=503)
        try:
            body = await request.json()
            session_id = body.get("session_id", "")
        except Exception:
            return web.json_response({"error": "session_id required"}, status=400)
        if not session_id:
            return web.json_response({"error": "session_id required"}, status=400)
        job = batch_corrector.process_now(session_id)
        return web.json_response(job.to_dict())

    # --- Nightly scheduler API ---

    async def handle_nightly_status(request):
        return web.json_response(_nightly_state)

    async def handle_nightly_trigger(request):
        if _nightly_state.get("running"):
            return web.json_response({"error": "already running"}, status=409)
        if (_batch_state.get("running")
                or _batch_combined_tag_state.get("running")
                or _batch_session_tag_state.get("running")):
            return web.json_response({"error": "another batch is running"}, status=409)

        async def _run_nightly_now():
            _nightly_state["running"] = True
            _nightly_state["error"] = None
            _nightly_state["phase"] = None
            try:
                if diarizer is not None:
                    _nightly_state["phase"] = "diarize"
                    log.info("Nightly trigger: Phase 1 — diarize")
                    await _batch_diarize_task(force=False)

                _nightly_state["phase"] = "tag"
                log.info("Nightly trigger: Phase 2 — combined tag (voice → session)")
                await _batch_combined_tag_task(force=False)

                _nightly_state["last_result"] = "success"
            except Exception as e:
                log.error("Nightly trigger: error: %s", e, exc_info=True)
                _nightly_state["last_result"] = "error"
                _nightly_state["error"] = str(e)
            finally:
                _nightly_state["running"] = False
                _nightly_state["phase"] = None
                _nightly_state["last_run"] = datetime.now().isoformat()
                if NIGHTLY_SCHEDULE_HOUR > 0:
                    _nightly_state["next_run"] = _calc_next_nightly(NIGHTLY_SCHEDULE_HOUR)

        asyncio.create_task(_run_nightly_now())
        return web.json_response({"status": "started"})

    mgmt_app = web.Application()
    mgmt_app.router.add_post("/model/load", handle_model_load)
    mgmt_app.router.add_post("/model/unload", handle_model_unload)
    mgmt_app.router.add_get("/model/status", handle_model_status)
    mgmt_app.router.add_post("/diarize/start", handle_batch_start)
    mgmt_app.router.add_get("/diarize/status", handle_batch_status)
    mgmt_app.router.add_post("/diarize/cancel", handle_batch_cancel)
    mgmt_app.router.add_get("/hotwords/status", handle_hotwords_status)
    mgmt_app.router.add_get("/hotwords/settings", handle_hotwords_settings_get)
    mgmt_app.router.add_put("/hotwords/settings", handle_hotwords_settings_put)
    mgmt_app.router.add_post("/hotwords/custom", handle_hotwords_custom_post)
    mgmt_app.router.add_delete("/hotwords/custom", handle_hotwords_custom_delete)
    # Speaker 管理 API (固定パスを {id} より先に登録)
    mgmt_app.router.add_get("/speakers/status", handle_speakers_status)
    mgmt_app.router.add_get("/speakers/list", handle_speakers_list)
    mgmt_app.router.add_get("/speakers/list-detailed", handle_speakers_list_detailed)
    mgmt_app.router.add_post("/speakers/add", handle_speakers_add)
    mgmt_app.router.add_put("/speakers/tag-correction", handle_speakers_tag_correction)
    mgmt_app.router.add_put("/speakers/threshold", handle_speakers_threshold)
    mgmt_app.router.add_post("/speakers/tag/start", handle_speakers_tag_start)
    mgmt_app.router.add_get("/speakers/tag/status", handle_speakers_tag_status)
    mgmt_app.router.add_post("/speakers/tag/cancel", handle_speakers_tag_cancel)
    mgmt_app.router.add_delete("/speakers/all", handle_speakers_delete_all)
    mgmt_app.router.add_put("/speakers/{id}", handle_speakers_update)
    mgmt_app.router.add_delete("/speakers/{id}", handle_speakers_delete)
    mgmt_app.router.add_post("/speakers/{id}/voiceprint", handle_speakers_add_voiceprint)
    mgmt_app.router.add_delete("/speakers/{id}/voiceprint/{sample_id}", handle_speakers_remove_voiceprint)
    # User tags reload
    mgmt_app.router.add_post("/user-tags/reload", handle_user_tags_reload)
    # Taxonomy reload
    mgmt_app.router.add_post("/taxonomy/reload", handle_taxonomy_reload)
    # Session tagging batch API
    mgmt_app.router.add_post("/session-tags/start", handle_session_tag_start)
    mgmt_app.router.add_get("/session-tags/status", handle_session_tag_status)
    mgmt_app.router.add_post("/session-tags/cancel", handle_session_tag_cancel)
    # Combined tag batch API (Voice Tag → Session Tag)
    mgmt_app.router.add_post("/tags/start", handle_combined_tag_start)
    mgmt_app.router.add_get("/tags/status", handle_combined_tag_status)
    mgmt_app.router.add_post("/tags/cancel", handle_combined_tag_cancel)
    # Audio scene classification API
    mgmt_app.router.add_get("/audio-scene/status", handle_audio_scene_status)
    # Sanitize batch API (training data anonymization)
    mgmt_app.router.add_post("/sanitize/start", handle_sanitize_start)
    mgmt_app.router.add_get("/sanitize/status", handle_sanitize_status)
    mgmt_app.router.add_post("/sanitize/cancel", handle_sanitize_cancel)
    mgmt_app.router.add_get("/sanitize/download", handle_sanitize_download)
    # Batch LLM correction API
    mgmt_app.router.add_get("/batch/status", handle_batch_llm_status)
    mgmt_app.router.add_get("/batch/jobs", handle_batch_llm_jobs)
    mgmt_app.router.add_post("/batch/retry", handle_batch_llm_retry)
    mgmt_app.router.add_post("/batch/process", handle_batch_llm_process)
    # Nightly scheduler API
    mgmt_app.router.add_get("/nightly/status", handle_nightly_status)
    mgmt_app.router.add_post("/nightly/trigger", handle_nightly_trigger)

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

        # Start nightly scheduler
        if NIGHTLY_SCHEDULE_HOUR > 0:
            asyncio.create_task(_nightly_scheduler())
        else:
            log.info("Nightly scheduler: DISABLED (NIGHTLY_SCHEDULE_HOUR=0)")

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
