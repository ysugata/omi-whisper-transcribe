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
import hashlib
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
PATIENCE = float(os.environ.get("PATIENCE", "1.5"))                       # ビームサーチ patience (探索深度、0=無効)
LOG_PROB_THRESHOLD = float(os.environ.get("LOG_PROB_THRESHOLD", "-0.5"))  # セグメント平均対数確率の閾値
NO_SPEECH_THRESHOLD = float(os.environ.get("NO_SPEECH_THRESHOLD", "0.6")) # 無音確率の閾値
COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("COMPRESSION_RATIO_THRESHOLD", "2.4"))  # 圧縮率閾値
CONDITION_ON_PREVIOUS_TEXT = os.environ.get("CONDITION_ON_PREVIOUS_TEXT", "true").lower() in ("true", "1", "yes")  # 前文脈を使った認識精度向上
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.1"))  # 繰り返しペナルティ (1.0=無効)
PROMPT_CARRY_ENABLED = os.environ.get("PROMPT_CARRY_ENABLED", "false").lower() in ("true", "1", "yes")  # チャンク間コンテキスト引き継ぎ
_halluc_thresh = os.environ.get("HALLUCINATION_SILENCE_THRESHOLD", "1.0")
HALLUCINATION_SILENCE_THRESHOLD = float(_halluc_thresh) if _halluc_thresh else None  # 無音区間ハルシネーション検出閾値 (秒、空文字=無効)

# --- Whisper 内部 VAD パラメータ ---
# Silero VAD (VAD_SEG_ENABLED) で前分割済みの場合はデフォルトを緩和し、
# 短い発話 (相槌、応答等) が Whisper 側で再フィルタされるのを防ぐ。
# 未前分割時はノイズバースト除去のため 500ms をデフォルトにする。
# NOTE: VAD_SEG_ENABLED は後方 (line ~136) で定義されるため、ここでは env を直接参照
_vad_seg_on = os.environ.get("VAD_SEG_ENABLED", "true").lower() in ("true", "1", "yes")
_default_whisper_vad_min_speech = "250" if _vad_seg_on else "500"
WHISPER_VAD_MIN_SPEECH_MS = int(os.environ.get("WHISPER_VAD_MIN_SPEECH_MS") or _default_whisper_vad_min_speech)
WHISPER_VAD_MIN_SILENCE_MS = int(os.environ.get("WHISPER_VAD_MIN_SILENCE_MS", "500"))
WHISPER_VAD_SPEECH_PAD_MS = int(os.environ.get("WHISPER_VAD_SPEECH_PAD_MS", "200"))
WHISPER_VAD_THRESHOLD = float(os.environ.get("WHISPER_VAD_THRESHOLD", "0.5"))

# --- Layer 3: LLM 後処理設定 ---
LLM_ENABLED = os.environ.get("LLM_ENABLED", "true").lower() in ("true", "1", "yes")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")
# gpt-5 系 Chat Completions 用: 思考トークンは出力課金されるため既定で最小化
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "minimal")


def resolve_llm_endpoint(model: str) -> tuple[str, str]:
    """モデル名から (api_key, base_url) を解決する。

    以前は LLM_API_KEY=GEMINI_API_KEY がハードコードされており、補正・
    タグ付けを OpenAI モデルに切り替えられなかった。プレフィックスで
    自動判定する: gpt-* → OpenAI、それ以外 → Gemini (OpenAI 互換 endpoint)。
    """
    if model.startswith("gpt-"):
        return OPENAI_API_KEY, OPENAI_BASE_URL
    return GEMINI_API_KEY, LLM_BASE_URL
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "10"))
LLM_MIN_TEXT_LENGTH = int(os.environ.get("LLM_MIN_TEXT_LENGTH", "20"))
LLM_PIPELINE_STAGES = os.environ.get("LLM_PIPELINE_STAGES", "B")        # Pipeline stages to run (A/B/C)
LLM_SCREENING_ENABLED = os.environ.get("LLM_SCREENING_ENABLED", "true").lower() in ("true", "1", "yes")
STAGE_C_AUTO = os.environ.get("STAGE_C_AUTO", "true").lower() in ("true", "1", "yes")  # Auto-trigger 3-C
LLM_BATCH_MODE = os.environ.get("LLM_BATCH_MODE", "sync")              # "sync" (standard API) or "batch" (Gemini Batch API, 50% cheaper)
RECOVERY_SCAN_DAYS = int(os.environ.get("RECOVERY_SCAN_DAYS", "7"))     # Startup scan: recover unprocessed sessions from last N days (0=disabled)

# --- セッションタグ付け設定 ---
TAGGING_ENABLED = os.environ.get("TAGGING_ENABLED", "true").lower() in ("true", "1", "yes")
TAGGING_MODEL = os.environ.get("TAGGING_MODEL", "gemini-3.1-flash-lite")
DOMAIN_PREFILTER_ENABLED = os.environ.get("DOMAIN_PREFILTER_ENABLED", "true").lower() in ("true", "1", "yes")
DOMAIN_PREFILTER_MIN_MATCHES = int(os.environ.get("DOMAIN_PREFILTER_MIN_MATCHES", "1"))

# --- RAG 分類設定 ---
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() in ("true", "1", "yes")
RAG_K = int(os.environ.get("RAG_K", "7"))
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_CONFIDENCE_THRESHOLD", "0.5"))
RAG_MIN_SESSIONS = int(os.environ.get("RAG_MIN_SESSIONS", "20"))

# --- タクソノミー自動精緻化設定 ---
TAXONOMY_REFINE_ENABLED = os.environ.get("TAXONOMY_REFINE_ENABLED", "true").lower() in ("true", "1", "yes")
TAXONOMY_SAMPLE_RATIO = float(os.environ.get("TAXONOMY_SAMPLE_RATIO", "0.2"))
TAXONOMY_PROPOSAL_MODEL = os.environ.get("TAXONOMY_PROPOSAL_MODEL", "gemini-2.5-flash")
TAXONOMY_VALIDATION_MODEL = os.environ.get("TAXONOMY_VALIDATION_MODEL", "gemini-3.1-flash-lite")

# --- 音響シーン分類設定 ---
AUDIO_SCENE_ENABLED = os.environ.get("AUDIO_SCENE_ENABLED", "true").lower() in ("true", "1", "yes")
AUDIO_SCENE_DEVICE = os.environ.get("AUDIO_SCENE_DEVICE", "cpu")
AUDIO_SCENE_THRESHOLD = float(os.environ.get("AUDIO_SCENE_THRESHOLD", "0.3"))
AUDIO_SCENE_MEDIA_THRESHOLD = float(os.environ.get("AUDIO_SCENE_MEDIA_THRESHOLD", "0.6"))

# --- ASR エンジン選択 ---
ENGINE = os.environ.get("ENGINE", "whisper").lower()  # "whisper" | "qwen3asr"

# --- VibeVoice-ASR 設定 ---
VIBEVOICE_ENABLED = os.environ.get("VIBEVOICE_ENABLED", "false").lower() in ("true", "1", "yes")

# --- 深夜スケジューラ設定 ---
NIGHTLY_SCHEDULE_HOUR = int(os.environ.get("NIGHTLY_SCHEDULE_HOUR", "4"))  # 0=無効, 1-23=実行時刻
BATCH_MAX_DAYS = int(os.environ.get("BATCH_MAX_DAYS", "7"))  # batch scan range in days (0=unlimited)

# --- 夜間自動 sanitize (Safe Training Data 自動生成) ---
# タグ付け完了後、未 sanitize の高価値セッションを自動で匿名化する。
# 生成された safe_training_*.jsonl は KC エクスポートと Wiki ビルドの材料。
NIGHTLY_SANITIZE_ENABLED = os.environ.get("NIGHTLY_SANITIZE_ENABLED", "true").lower() in ("true", "1", "yes")
NIGHTLY_SANITIZE_MAX_SESSIONS = int(os.environ.get("NIGHTLY_SANITIZE_MAX_SESSIONS", "50"))  # 1晩の上限 (LLM コスト制御)
NIGHTLY_SANITIZE_HIGH_VALUE_ONLY = os.environ.get("NIGHTLY_SANITIZE_HIGH_VALUE_ONLY", "true").lower() in ("true", "1", "yes")
NIGHTLY_SANITIZE_MAX_ATTEMPTS = int(os.environ.get("NIGHTLY_SANITIZE_MAX_ATTEMPTS", "2"))  # 失敗セッションの再試行上限

# --- KC 品質ゲート ---
# sanitize が抽出したナレッジカードを LLM で採点し quality (1-5) を付与する。
# Wiki ビルドとエクスポートは KC_MIN_QUALITY 未満のカードを除外する。
KC_QUALITY_GATE_ENABLED = os.environ.get("KC_QUALITY_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
KC_JUDGE_MODEL = os.environ.get("KC_JUDGE_MODEL", "gpt-5-nano")

# --- メディア音声の早期ゲート ---
# セッション確定直後に音響シーン分類を実行し、TV・ラジオ等のメディア音声が
# 主体のセッションは LLM 補正キューに入れない (ゴミデータへの課金防止)。
MEDIA_EARLY_GATE_ENABLED = os.environ.get("MEDIA_EARLY_GATE_ENABLED", "true").lower() in ("true", "1", "yes")

# --- デイリー・エリシテーション (能動的な聞き出し) ---
# 夜間にその日の意思決定へ係留した質問を生成し、Viewer で経営者に提示。
# 回答は sanitize を経てナレッジカード・Wiki へ還流する。
ELICITATION_ENABLED = os.environ.get("ELICITATION_ENABLED", "true").lower() in ("true", "1", "yes")
ELICITATION_MODEL = os.environ.get("ELICITATION_MODEL", "gpt-5.6-luna")  # 質問の質が命、量は少ない
ELICITATION_QUESTIONS_PER_NIGHT = int(os.environ.get("ELICITATION_QUESTIONS_PER_NIGHT", "3"))
ELICITATION_MAX_PENDING = int(os.environ.get("ELICITATION_MAX_PENDING", "9"))

# --- RAG ギャップログ ---
# 事業相談が答えられなかった質問を記録し、翌晩のエリシテーション質問の種にする
SUCCESSION_GAP_SIMILARITY = float(os.environ.get("SUCCESSION_GAP_SIMILARITY", "0.45"))

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

# --- SD カードアップロード設定 ---
# デバイス時計はリセットで大きく狂うことがある (79日遅れの実例あり)。
# timestamp がこの日数より古い/未来なら受信時刻にフォールバックする (0=無効)
SDCARD_TS_MAX_AGE_DAYS = int(os.environ.get("SDCARD_TS_MAX_AGE_DAYS", "7"))
# バックグラウンド処理待ちの同時アップロード上限 (PCM をメモリ保持するため)
SDCARD_MAX_PENDING = int(os.environ.get("SDCARD_MAX_PENDING", "4"))

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
VAD_SILENCE_TRIGGER = float(os.environ.get("VAD_SILENCE_TRIGGER", "1.2"))     # 発話後この無音で文字起こし実行 (longer = more context for Whisper)
VAD_MAX_BUFFER = float(os.environ.get("VAD_MAX_BUFFER", "20.0"))              # バッファ上限 (秒)
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

# 管理 API (8091) のトークン認証。空文字 = 認証なし (従来どおり LAN 内前提)。
# 設定すると全管理エンドポイントで Authorization: Bearer <token>
# または ?token=<token> が必須になる。
MANAGEMENT_TOKEN = os.environ.get("MANAGEMENT_TOKEN", "")

# ============================================================
# ログ設定
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("omi-whisper")


# ============================================================
# 管理 API 認証
# ============================================================

def _extract_mgmt_token(headers, query) -> str:
    """リクエストから管理 API トークンを取り出す (Bearer ヘッダー優先)。"""
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):]
    return query.get("token", "")


def _mgmt_auth_ok(supplied: str, expected: str) -> bool:
    """タイミング攻撃耐性のあるトークン比較。expected 空 = 認証なし。"""
    if not expected:
        return True
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def make_mgmt_auth_middleware(token: str):
    """MANAGEMENT_TOKEN が設定されている場合の認証 middleware を返す。"""
    from aiohttp import web as _web

    @_web.middleware
    async def _auth_middleware(request, handler):
        supplied = _extract_mgmt_token(request.headers, request.query)
        if not _mgmt_auth_ok(supplied, token):
            return _web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    return _auth_middleware

# ============================================================
# Layer 1 + Layer 2: 固有名詞改善
# ============================================================

from sdcard_registry import SdcardUploadRegistry, make_key as sdcard_make_key, resolve_recorded_at
from term_corrector import TermCorrector
from llm_corrector import LLMCorrector
from llm_pipeline import LLMPipeline
from batch_corrector import BatchCorrector
from diarizer import Diarizer
from audio_preprocessor import AudioPreprocessor
from hotword_loader import HotwordLoader
from speaker_filter import SpeakerFilter
from session_tagger import SessionTagger
from rag_classifier import RagClassifier
from taxonomy_refiner import TaxonomyRefiner
from training_sanitizer import TrainingSanitizer
from responses_sanitizer import ResponsesSanitizer
from succession_rag import SuccessionRAG
from segment_screener import SegmentScreener
from suggestion_learner import SuggestionLearner
from keyword_learner import KeywordLearner
from kc_judge import KCJudge
from elicitation import ElicitationEngine, ElicitationStore
from opus_decoder import OmiOpusStreamDecoder, OMI_HEADER_SIZE
from faster_whisper.vad import get_speech_timestamps, VadOptions

_TERM_CORRECTIONS_PATH = os.path.join(CONFIG_DIR, "term_corrections.json")

term_corrector = TermCorrector(_TERM_CORRECTIONS_PATH)

# Layer 3 Pipeline: 3-stage LLM correction (batch post-processing)
# (レガシーの LLMCorrector インスタンスは廃止。detect_corrections の
#  静的メソッドのみ _save_llm_suggestions で使用)
llm_pipeline: LLMPipeline | None = None
_llm_key, _llm_base_url = resolve_llm_endpoint(LLM_MODEL)
if LLM_ENABLED and _llm_key:
    llm_pipeline = LLMPipeline(
        api_key=_llm_key,
        base_url=_llm_base_url,
        model=LLM_MODEL,
        timeout=LLM_TIMEOUT,
        stages=LLM_PIPELINE_STAGES,
        reasoning_effort=LLM_REASONING_EFFORT,
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

# RAG 分類器 (Embedding ベース k-NN)
rag_classifier: RagClassifier | None = None
if RAG_ENABLED and GEMINI_API_KEY:
    rag_classifier = RagClassifier(
        api_key=GEMINI_API_KEY,
        recordings_dir=RECORDINGS_DIR,
        index_dir=CONFIG_DIR,
        k=RAG_K,
        confidence_threshold=RAG_CONFIDENCE_THRESHOLD,
        min_sessions=RAG_MIN_SESSIONS,
    )

# タクソノミー自動精緻化
taxonomy_refiner: TaxonomyRefiner | None = None
if TAXONOMY_REFINE_ENABLED and GEMINI_API_KEY and rag_classifier:
    taxonomy_refiner = TaxonomyRefiner(
        api_key=GEMINI_API_KEY,
        recordings_dir=RECORDINGS_DIR,
        config_dir=CONFIG_DIR,
        rag_classifier=rag_classifier,
        sample_ratio=TAXONOMY_SAMPLE_RATIO,
        proposal_model=TAXONOMY_PROPOSAL_MODEL,
        validation_model=TAXONOMY_VALIDATION_MODEL,
    )

# セッションタグ付け (LLM API キーがあれば利用可能)
TAG_TAXONOMY_PATH = os.path.join(CONFIG_DIR, "tag_taxonomy.json")

session_tagger: SessionTagger | None = None
_tag_key, _tag_base_url = resolve_llm_endpoint(TAGGING_MODEL)
if TAGGING_ENABLED and _tag_key:
    session_tagger = SessionTagger(
        api_key=_tag_key,
        base_url=_tag_base_url,
        model=TAGGING_MODEL,
        reasoning_effort=LLM_REASONING_EFFORT,
        taxonomy_path=TAG_TAXONOMY_PATH,
        prefilter_enabled=DOMAIN_PREFILTER_ENABLED,
        prefilter_min_matches=DOMAIN_PREFILTER_MIN_MATCHES,
        rag_classifier=rag_classifier,
    )

keyword_learner = KeywordLearner(
    taxonomy_path=TAG_TAXONOMY_PATH,
    suggestions_path=os.path.join(CONFIG_DIR, "pending_keyword_suggestions.json"),
    recordings_dir=RECORDINGS_DIR,
)

from audio_scene_classifier import AudioSceneClassifier
from vibevoice_transcriber import VibeVoiceTranscriber, load_audio as vv_load_audio

audio_scene_classifier = AudioSceneClassifier(
    enabled=AUDIO_SCENE_ENABLED,
    device=AUDIO_SCENE_DEVICE,
    threshold=AUDIO_SCENE_THRESHOLD,
    media_threshold=AUDIO_SCENE_MEDIA_THRESHOLD,
)


def _is_media_scene(scene: dict | None) -> bool:
    """音響シーン分類の結果が「TV・ラジオ等のメディア音声主体」かを判定する。"""
    return bool(scene) and scene.get("scene_label_en") == "media"


async def _early_media_gate(rec) -> bool:
    """セッション確定直後にメディア音声を検出し、LLM 補正のスキップを判断する。

    noisy セッションの根本原因は TV/アニメ音声の誤録音 (実測で判明済み)。
    従来は夜間の音響シーン分類まで判定できず、ゴミデータにも LLM 補正の
    課金が発生していた。ここで検出した scene は transcript.json に保存する
    ので、夜間のシーン分類フェーズはこのセッションをスキップできる。

    Returns True if the session is media-dominant (correction should be skipped).
    """
    if not (MEDIA_EARLY_GATE_ENABLED and AUDIO_SCENE_ENABLED
            and audio_scene_classifier.enabled
            and rec.enabled and getattr(rec, "session_dir", None)
            and rec.session_dir.exists()):
        return False
    audio_path = None
    for name in ("audio.ogg", "audio.wav"):
        cand = rec.session_dir / name
        if cand.exists():
            audio_path = cand
            break
    if audio_path is None:
        return False
    try:
        loop = asyncio.get_running_loop()
        scene = await loop.run_in_executor(
            None, audio_scene_classifier.classify, str(audio_path)
        )
        if scene is None:
            return False
        is_media = _is_media_scene(scene)
        # scene 結果と判定を transcript に永続化 (アトミック書き込み)
        transcript_path = rec.session_dir / "transcript.json"
        if transcript_path.exists():
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    data = json.load(f)
                session = data.setdefault("session", {})
                session["audio_scene"] = scene
                if is_media:
                    session["media_suspect"] = True
                tmp_path = str(transcript_path) + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, str(transcript_path))
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Early media gate: transcript update failed: {e}")
        if is_media:
            log.info(
                f"Early media gate: {rec.session_dir.name} is media-dominant "
                f"(scene={scene.get('scene_label')}, "
                f"conf={scene.get('scene_confidence')}) — skipping LLM correction"
            )
        return is_media
    except Exception as e:
        log.warning(f"Early media gate failed (non-fatal): {e}")
        return False

vibevoice_transcriber = VibeVoiceTranscriber()

# Text normalizer (half/full-width, duplicate chars) — CPU only
text_normalizer_fn = None
try:
    from text_normalizer import normalize as _tn_normalize
    text_normalizer_fn = _tn_normalize
    log.info("Text normalizer: loaded")
except ImportError:
    log.debug("Text normalizer: not available (neologdn not installed)")

# Phonetic corrector (pronunciation-based ASR error correction) — CPU only
phonetic_corrector_instance = None
try:
    from phonetic_corrector import PhoneticCorrector
    _pc_glossary = []
    # Build glossary from hotwords custom + term corrections
    _pc_glossary.extend(hotword_loader.get_custom_words())
    for v in term_corrector.list_corrections().values():
        if v not in _pc_glossary:
            _pc_glossary.append(v)
    if _pc_glossary:
        phonetic_corrector_instance = PhoneticCorrector(
            glossary=_pc_glossary, threshold=0.75, min_word_length=3,
        )
        log.info("Phonetic corrector: %d glossary terms", len(_pc_glossary))
except ImportError:
    log.debug("Phonetic corrector: not available (pykakasi not installed)")

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
succession_rag: SuccessionRAG | None = None

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
    _sanitize_key, _sanitize_url = resolve_llm_endpoint(SANITIZE_MODEL)
    if _sanitize_key:
        training_sanitizer = TrainingSanitizer(
            api_key=_sanitize_key,
            base_url=_sanitize_url,
            model=SANITIZE_MODEL,
        )

# KC 品質ゲート: sanitize 直後にナレッジカードを採点する
kc_judge: KCJudge | None = None
if KC_QUALITY_GATE_ENABLED:
    _judge_key, _judge_url = resolve_llm_endpoint(KC_JUDGE_MODEL)
    if _judge_key:
        kc_judge = KCJudge(
            api_key=_judge_key,
            base_url=_judge_url,
            model=KC_JUDGE_MODEL,
            reasoning_effort=LLM_REASONING_EFFORT,
        )


def _attach_kc_quality(rec: dict) -> None:
    """Judge the KCs in a ResponsesSanitizer record and attach scores in-place.

    採点失敗はスコアなし (=消費側では合格扱い) として続行する。
    """
    if kc_judge is None or not kc_judge.enabled:
        return
    kcs = []
    if isinstance(rec.get("knowledge_cards"), list):
        kcs.extend(k for k in rec["knowledge_cards"] if isinstance(k, dict))
    if isinstance(rec.get("knowledge_card"), dict):
        kcs.append(rec["knowledge_card"])
    if not kcs:
        return
    scores = kc_judge.judge(kcs)
    for kc, s in zip(kcs, scores):
        if s:
            kc["quality"] = s["score"]
            kc["quality_reason"] = s["reason"]


def _attach_kc_quality_chat_records(records: list) -> None:
    """Same as _attach_kc_quality but for TrainingSanitizer's flat records."""
    if kc_judge is None or not kc_judge.enabled:
        return
    kc_recs = [
        r for r in records
        if isinstance(r, dict) and r.get("type") == "knowledge_card"
    ]
    if not kc_recs:
        return
    scores = kc_judge.judge([
        {"question": r.get("input", ""), "answer": r.get("output", "")}
        for r in kc_recs
    ])
    for r, s in zip(kc_recs, scores):
        if s:
            r["quality"] = s["score"]
            r["quality_reason"] = s["reason"]


# デイリー・エリシテーション
elicitation_store = ElicitationStore(os.path.join(CONFIG_DIR, "elicitation_questions.json"))
elicitation_engine: ElicitationEngine | None = None
if ELICITATION_ENABLED:
    _el_key, _el_url = resolve_llm_endpoint(ELICITATION_MODEL)
    if _el_key:
        elicitation_engine = ElicitationEngine(
            api_key=_el_key, base_url=_el_url, model=ELICITATION_MODEL,
            reasoning_effort="low",
        )

_RAG_GAP_LOG_PATH = os.path.join(CONFIG_DIR, "rag_gap_log.json")


def _load_rag_gaps() -> list:
    try:
        data = json.loads(Path(_RAG_GAP_LOG_PATH).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_rag_gaps(gaps: list) -> None:
    tmp = _RAG_GAP_LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(gaps[-100:], f, ensure_ascii=False, indent=1)
    os.replace(tmp, _RAG_GAP_LOG_PATH)


def _append_rag_gap(query: str, top_similarity: float, reason: str) -> None:
    """事業相談が答えられなかった質問を記録する (エリシテーションの種)。"""
    try:
        gaps = _load_rag_gaps()
        if any(g.get("query") == query for g in gaps):
            return
        gaps.append({
            "query": query,
            "top_similarity": round(top_similarity, 3),
            "reason": reason,
            "at": datetime.now().isoformat(timespec="seconds"),
            "consumed": False,
        })
        _save_rag_gaps(gaps)
        log.info(f"RAG gap logged: {query[:60]} ({reason})")
    except OSError as e:
        log.warning(f"RAG gap log write failed: {e}")


def _mark_rag_gaps_consumed(queries: list) -> None:
    try:
        gaps = _load_rag_gaps()
        qset = set(queries)
        changed = False
        for g in gaps:
            if g.get("query") in qset and not g.get("consumed"):
                g["consumed"] = True
                changed = True
        if changed:
            _save_rag_gaps(gaps)
    except OSError:
        pass


def _is_rag_gap(answer_text: str, sources: list, warnings: list,
                threshold: float) -> tuple:
    """回答がナレッジ空白を示しているかを判定する。

    Abstention 研究の知見: 「記録がない」と正直に言えた質問こそが最も
    価値のある信号 — それが次に聞き出すべきことを教えてくれる。
    Returns (is_gap, reason).
    """
    if "no_sources" in (warnings or []):
        return True, "no_sources"
    text = answer_text or ""
    if "記録はありません" in text or "記録がありません" in text:
        return True, "abstained"
    top = 0.0
    for s in sources or []:
        try:
            top = max(top, float(s.get("similarity", 0) or 0))
        except (TypeError, ValueError):
            continue
    if top and top < threshold:
        return True, "low_similarity"
    return False, ""


def _collect_decision_contexts(max_sessions: int = 3, scan_days: int = 7) -> list:
    """直近の意思決定価値のあるセッションを、未使用のものから集める。

    Critical Decision Method: 質問は必ず実在の意思決定に係留する。
    """
    base = Path(RECORDINGS_DIR)
    if not base.is_dir():
        return []
    used = elicitation_store.used_source_refs()
    contexts: list = []
    day_dirs = sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)
    for day_dir in day_dirs[:scan_days * 4]:
        if len(contexts) >= max_sessions:
            break
        for time_dir in sorted(day_dir.iterdir(), reverse=True):
            if len(contexts) >= max_sessions:
                break
            if not time_dir.is_dir():
                continue
            sid = f"{day_dir.name}/{time_dir.name}"
            if sid in used:
                continue
            tpath = time_dir / "transcript.json"
            if not tpath.exists():
                continue
            try:
                with open(tpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            tags = data.get("session", {}).get("tags", {}) or {}
            outcomes = set(tags.get("outcomes") or [])
            decisionish = (
                "意思決定あり" in outcomes
                or "方針転換" in outcomes
                or bool(tags.get("decision_memo"))
                or tags.get("importance") == "高"
            )
            if not decisionish:
                continue
            segments = data.get("segments", [])
            if len(segments) < 3:
                continue
            excerpt = " ".join(s.get("text", "") for s in segments)[:1500]
            tags_line = "{} / {} / 重要度{}".format(
                tags.get("conversation_type", ""),
                ",".join(tags.get("outcomes") or []),
                tags.get("importance", "?"),
            )
            contexts.append({
                "session_id": sid,
                "summary": (tags.get("summary") or "")[:300],
                "tags_line": tags_line,
                "excerpt": excerpt,
            })
    return contexts


def _sanitize_elicitation_answers_sync() -> int:
    """回答済みの質問を Safe Training レコードに変換する (同期・executor 用)。"""
    answers = elicitation_store.unprocessed_answers()
    if not answers or responses_sanitizer is None:
        return 0
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(CONFIG_DIR) / f"safe_training_{ts}.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in answers:
            sid = "elicitation/" + q["id"]
            raw = "【継承者からの質問】{}\n【経営者本人の回答】{}".format(
                q.get("question", ""), q.get("answer", ""),
            )
            try:
                rs = responses_sanitizer.sanitize(sid, raw, "ja")
            except Exception as e:
                log.warning(f"Elicitation sanitize failed for {sid}: {e}")
                rs = None
            if rs is None:
                elicitation_store.mark_processed(q["id"], "failed")
                continue
            if rs.data.get("risk") == "high":
                elicitation_store.mark_processed(q["id"], "dropped")
                continue
            rec = {"session_id": sid, **rs.data}
            _attach_kc_quality(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            elicitation_store.mark_processed(q["id"], "done")
            log.info(f"Elicitation answer converted to KC: {sid}")
    if written == 0:
        try:
            out_path.unlink()
        except OSError:
            pass
    return written


async def _run_elicitation_phase() -> None:
    """夜間 Phase 2.6: 回答の取り込み + 新しい質問の生成。"""
    loop = asyncio.get_running_loop()

    # a) 回答済みの取り込み (経営者が昨日答えたもの → KC 化)
    if responses_sanitizer is not None:
        n = await loop.run_in_executor(None, _sanitize_elicitation_answers_sync)
        if n:
            log.info(f"Elicitation: {n} answered question(s) converted to KCs")

    # b) 新しい質問の生成
    if elicitation_engine is None or not elicitation_engine.enabled:
        return
    pending = len(elicitation_store.pending())
    budget = min(ELICITATION_QUESTIONS_PER_NIGHT,
                 ELICITATION_MAX_PENDING - pending)
    if budget <= 0:
        log.info("Elicitation: pending cap reached, no new questions")
        return
    contexts = await loop.run_in_executor(None, _collect_decision_contexts)
    gaps = [g["query"] for g in _load_rag_gaps() if not g.get("consumed")][:5]
    if not contexts and not gaps:
        log.info("Elicitation: no decision contexts or gaps to ask about")
        return
    questions = await loop.run_in_executor(
        None,
        lambda: elicitation_engine.generate(
            contexts, gaps,
            elicitation_store.recent_question_texts(), budget,
        ),
    )
    added = elicitation_store.add_questions(questions, ELICITATION_MAX_PENDING)
    if added and gaps:
        _mark_rag_gaps_consumed(gaps)
    log.info(
        f"Elicitation: generated {added} question(s) "
        f"(contexts={len(contexts)}, gaps={len(gaps)})"
    )


# Succession RAG (retrieval-augmented Q&A over sanitized knowledge cards)
SUCCESSION_ANSWER_MODEL = os.environ.get("SUCCESSION_ANSWER_MODEL", "gpt-5-nano")
SUCCESSION_EMBED_MODEL = os.environ.get("SUCCESSION_EMBED_MODEL", "gemini-embedding-001")
# Phase 2.5: parent-child retrieval reads the LLM Wiki built by host-side
# tools/build_wiki.py. The container mounts ~/knowledge read-only at
# /app/knowledge (see docker-compose.yml). When the wiki directory is not
# present, SuccessionRAG transparently falls back to KC-only Phase 1.5
# behavior.
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", "/app/knowledge")
if GEMINI_API_KEY and OPENAI_API_KEY:
    succession_rag = SuccessionRAG(
        config_dir=CONFIG_DIR,
        gemini_api_key=GEMINI_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        embed_model=SUCCESSION_EMBED_MODEL,
        answer_model=SUCCESSION_ANSWER_MODEL,
        reasoning_effort=OPENAI_REASONING_EFFORT,
        timeout_sec=OPENAI_TIMEOUT_MS / 1000.0,
        knowledge_dir=KNOWLEDGE_DIR if os.path.isdir(KNOWLEDGE_DIR) else None,
    )
    # Try to load the persisted index synchronously; rebuild is on-demand via API.
    try:
        if not succession_rag.load_index():
            log.info("SuccessionRAG: no persisted index found; call /succession/rebuild to build it")
    except Exception as e:
        log.warning("SuccessionRAG: load_index failed: %s", e)
else:
    log.info("SuccessionRAG disabled: GEMINI_API_KEY and OPENAI_API_KEY both required")


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
    model_name=WHISPER_MODEL if ENGINE == "whisper" else "Qwen/Qwen3-ASR-1.7B",
    compute_type=COMPUTE_TYPE,
    engine=ENGINE,
)

# ============================================================
# バックグラウンドタスク管理
# ============================================================

# asyncio はタスクを弱参照でしか保持しないため、bare create_task した
# タスクは GC で回収されて途中で消えることがある (CPython 公式ドキュメント
# 記載の挙動)。夜間スケジューラや diarization がこれで無言停止すると
# 致命的なので、強参照を保持し、例外もログに出す。
_background_tasks: set = set()


def _spawn_bg(coro, name: str = "") -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _done(t: asyncio.Task):
        _background_tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error(f"Background task {name or coro!r} failed: {t.exception()!r}")

    task.add_done_callback(_done)
    return task


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
                # Old-dated dirs can hold freshly ingested sessions: SD card
                # uploads carry the (possibly wrong) device clock as the dir
                # name.  The dir mtime reflects when sessions were last
                # added, so keep recently written dirs.
                try:
                    if day_dir.stat().st_mtime >= cutoff:
                        log.info(
                            f"Cleanup skipped {day_dir.name}: "
                            f"recently written (mtime)"
                        )
                        continue
                except OSError:
                    pass
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

    def __init__(self, recorded_at: datetime | None = None, source: str | None = None,
                 extra_meta: dict | None = None):
        self.enabled = SAVE_AUDIO
        self.all_segments: list[dict] = []
        self.total_bytes = 0
        self._finalized = False
        self._audio_closed = False
        self.session_tags: dict | None = None  # set by _tag_session() before finalize
        self.source = source  # e.g. "sdcard" - written to JSON metadata
        self.extra_meta = extra_meta  # e.g. device_timestamp/uploaded_at for sdcard

        if not self.enabled:
            return

        now = recorded_at if recorded_at is not None else datetime.now()
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
                # stderr の末尾を取る: 先頭は ffmpeg の version/configuration
                # ヘッダで埋まっていて実エラーメッセージが見えないため。
                log.error("ffmpeg failed (rc=%d): %s", result.returncode, result.stderr[-800:])
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
            "engine": ENGINE,
            "model": WHISPER_MODEL if ENGINE == "whisper" else "Qwen/Qwen3-ASR-1.7B",
            "language": LANGUAGE,
            "num_speakers": num_speakers,
        }
        if self.session_tags:
            session_meta["tags"] = self.session_tags
            session_meta["session_tagged"] = True
        if self.source:
            session_meta["source"] = self.source
        if self.extra_meta:
            session_meta.update(self.extra_meta)
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
    re.compile(r"ご視聴いただき.*"),  # 「ご視聴いただき有難う御座いました」等
    re.compile(r"チャンネル登録.*?お願いします"),
    re.compile(r"チャンネル登録"),
    re.compile(r"高評価"),
    re.compile(r"ご清聴ありがとうございま(した|す)"),
    re.compile(r"最後まで(ご覧|見て)いただき.*"),
    re.compile(r"ご覧いただきありがとうございま(した|す)"),
    re.compile(r"字幕.*?(視聴者|翻訳|提供)"),
    re.compile(r"サブタイトル.*?(製作|制作)"),  # 「サブタイトル&字幕製作by...」
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

# 店名・ブランド名の暴走検出: hotwords に登録された語がノイズから連続生成される
# 2個以上の固有名詞がカンマ等で連続する場合、ハルシネーションとみなして全文除去
_STORE_BRAND_NAMES = [
    "いすゞ", "Komeda", "コメダ", "丸亀", "ガスト", "スシロー", "Sushiro",
    "FamilyMart", "ファミマ", "ジョイフル", "Joyfull", "すき家", "Sukiya",
    "吉野家", "なだ万", "ASMO", "アスモ", "Adobe", "アドビ", "ActiveMeds",
    "みかか", "アジ研", "カツとじ", "セブンイレブン",
]
_STORE_BRAND_RE = re.compile(
    r"(?:" + "|".join(re.escape(s) for s in _STORE_BRAND_NAMES) + r")",
    re.IGNORECASE,
)

# 全文一致専用パターン: セグメント全体がこのフレーズのみの場合だけ除去
# (実際の会話で使われうるが、セグメント全体がこれだけなら Whisper のハルシネーション)
_HALLUCINATION_FULL_MATCH = [
    re.compile(r"^お(つかれさま|疲れ様)でした[。.!]*$"),
    re.compile(r"^(ではまた|また次回)[。.!]*$"),
    re.compile(r"^おやすみなさい[。.!]*$"),
    re.compile(r"^ごちそうさまでした[。.!]*$"),
    re.compile(r"^いらっしゃいませ[。.!]*$"),
    re.compile(r"^とりあえず[、,]*(これで)?終わりです[。.!]*$"),
    re.compile(r"^これが今日の(映像|動画)です[。.!]*$"),
]


def _remove_hallucinations(text: str) -> str:
    """既知のハルシネーションパターンを除去し、残りテキストを返す"""
    stripped = text.strip()
    # Full-match patterns: only remove if entire segment matches
    for pattern in _HALLUCINATION_FULL_MATCH:
        if pattern.match(stripped):
            return ""
    # Store/brand name hallucination: 2+ known names in one segment → full remove
    if len(_STORE_BRAND_RE.findall(stripped)) >= 2:
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


def _has_repetitive_kana(text: str) -> bool:
    """仮名の過度な繰り返しを検出する。

    ノイズ由来の「オォォォオォゥオォオゥ」「ンンンンンンン」等を検出。
    ユニーク仮名文字数に対する全仮名文字数の比率で判定する。
    """
    body = re.sub(r'[\s,，、。.!?？！・…―「」\-\'\"]+', '', text)
    kana_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', body)
    if len(kana_chars) < 6:
        return False
    unique_count = len(set(kana_chars))
    unique_ratio = unique_count / len(kana_chars)
    # ユニーク率 30% 未満 = 極端に少ない種類の仮名が繰り返されている
    # 例: "オォォォオォゥオォオゥ" → 3種/11文字 = 0.27 → True
    # 例: "ンンンンンンン" → 1種/7文字 = 0.14 → True
    # 例: "テレビテレビテレビ" → 3種/9文字 = 0.33 → False (正常な繰り返し)
    if unique_ratio < 0.30:
        return True
    # 同一仮名が 60% 以上を占める場合もノイズ
    from collections import Counter
    counts = Counter(kana_chars)
    most_common_ratio = counts.most_common(1)[0][1] / len(kana_chars)
    return most_common_ratio >= 0.6


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
    # 繰り返し仮名は logprob に関係なく検出 (信頼度チェック前)
    if _has_repetitive_kana(text):
        return True

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
    prompt_carry: str = "",
) -> list[dict]:
    """PCM 16-bit LE の音声バイト列を受け取り、
    Omi の custom_stt が期待する segments リストを返す。

    Args:
        audio_bytes: PCM 16-bit LE 音声データ
        model: ロード済みの WhisperModel インスタンス
        session_offset: タイムスタンプオフセット (秒)
        hotwords: カンマ区切りの hotwords 文字列 (デコーダバイアス用)
        prompt_carry: 前チャンクの出力テキスト (チャンク間コンテキスト引き継ぎ用)
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
            min_speech_duration_ms=WHISPER_VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS,
            speech_pad_ms=WHISPER_VAD_SPEECH_PAD_MS,
            threshold=WHISPER_VAD_THRESHOLD,
        ),
        word_timestamps=False,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        patience=PATIENCE,
        compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=LOG_PROB_THRESHOLD,
        no_speech_threshold=NO_SPEECH_THRESHOLD,
        repetition_penalty=REPETITION_PENALTY,
    )
    if HALLUCINATION_SILENCE_THRESHOLD is not None:
        transcribe_kwargs["hallucination_silence_threshold"] = HALLUCINATION_SILENCE_THRESHOLD
    if hotwords:
        transcribe_kwargs["hotwords"] = hotwords
    if prompt_carry:
        # 前チャンクの出力をinitial_promptとして渡し、チャンク間の文脈を維持。
        # Whisperのpromptトークン上限(224)に合わせて末尾200文字に制限。
        transcribe_kwargs["initial_prompt"] = prompt_carry[-200:]

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

        # Layer 2: テキスト正規化のみ (半角/全角統一)
        # 用語補正 (term_corrector) と音韻補正 (phonetic_corrector) はLLM前から
        # 削除。文脈なしの辞書置換はLLMの入力を汚染し、ドメイン推論を妨害する。
        # 用語情報は session_glossary としてLLMプロンプトに渡され、LLMが文脈を
        # 見て適用するかどうかを判断する。
        if text_normalizer_fn is not None:
            text = text_normalizer_fn(text)

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


def _split_pcm_on_silence(
    pcm_data: bytes,
    sample_rate: int,
    target_sec: float = 25.0,
    max_sec: float = 45.0,
) -> list[tuple[int, int]]:
    """
    PCM16 を silero VAD で無音境界分割し、(start_byte, end_byte) のリストを返す。

    realtime 経路の VAD_SEG ロジックをバッチ向けに書き換えたもの:
    - 25 秒以上溜まったら、次の無音区間で切る (文の途中切りを回避)
    - 45 秒を超えたら無音が無くても強制分割 (Whisper 内部 window に任せる)
    - 全体が無音 or 単一 speech の場合は 1 チャンクのみ

    タイムスタンプはサンプル単位。int16 mono なので byte offset = sample * 2。
    """
    if len(pcm_data) == 0:
        return []

    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    vad_opts = VadOptions(
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=250,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )
    speech_ts = get_speech_timestamps(audio_np, vad_opts)

    total_bytes = len(pcm_data)
    target_samples = int(target_sec * sample_rate)
    max_samples = int(max_sec * sample_rate)

    if not speech_ts:
        return [(0, total_bytes)]

    chunks: list[tuple[int, int]] = []
    current_start_sample = 0

    for i in range(1, len(speech_ts)):
        prev_end = speech_ts[i - 1]["end"]
        curr_start = speech_ts[i]["start"]
        chunk_len = prev_end - current_start_sample

        # target 到達 → 無音中間点で切る
        if chunk_len >= target_samples:
            cut_sample = (prev_end + curr_start) // 2
            chunks.append((current_start_sample * 2, cut_sample * 2))
            current_start_sample = cut_sample
        # max 超過 → 無音が無くても 次の speech 前で切る
        elif chunk_len >= max_samples:
            chunks.append((current_start_sample * 2, curr_start * 2))
            current_start_sample = curr_start

    # 最後の chunk: 残り全部
    chunks.append((current_start_sample * 2, total_bytes))

    return chunks


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
                and recorder.session_dir.exists()
                and len(recorder.all_segments) > 0):
            if not await _early_media_gate(recorder):
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


def _batch_cutoff_date() -> str | None:
    """Return the earliest date dir name to scan for batch processing.

    Returns None if BATCH_MAX_DAYS is 0 (unlimited).
    """
    if BATCH_MAX_DAYS <= 0:
        return None
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=BATCH_MAX_DAYS)
    return cutoff.strftime("%Y-%m-%d")


def _find_undiarized_sessions(force: bool = False) -> list[dict]:
    """RECORDINGS_DIR 配下の未 diarize セッションを古い順に返す。

    force=True の場合、既に diarized=True のセッションも含める (再実行用)。
    """
    base = Path(RECORDINGS_DIR)
    if not base.exists():
        return []

    cutoff = _batch_cutoff_date()
    sessions = []
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        if cutoff and day_dir.name < cutoff:
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

            # 恒久的に失敗するセッションを除外: 1秒未満の音声は diarizer が
            # 必ず ValueError を投げるため、毎晩リトライしても無駄。
            # 失敗マーカー (diarize_attempts) が規定回数に達したものも除外。
            duration = session_info.get("duration_seconds") or 0
            if not force and 0 < duration < 1.0:
                continue
            if not force and session_info.get("diarize_attempts", 0) >= 3:
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
    import gc
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
        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_undiarized_sessions(force=force)
        )
        _batch_state["total"] = len(sessions)
        log.info(f"Batch diarization: {len(sessions)} session(s) to process (force={force})")

        if not sessions:
            return

        if _batch_state["cancel_requested"]:
            log.info("Batch diarization cancelled by user (before GPU lock)")
            return

        loop = asyncio.get_running_loop()

        # Load pyannote pipeline once for entire batch
        async with model_manager.gpu_lock:
            # OOM対策: バッチ実行中は Whisper を unload して RAM/VRAM を解放。
            # gpu_lock 保持中なので新規セッションは入ってこない。失敗
            # (アクティブセッションあり) は無視して続行する。
            try:
                await model_manager.unload_model()
                gc.collect()
                log.info("Batch diarize: Whisper unloaded to free memory")
            except Exception as e:
                log.warning("Batch diarize: Whisper unload skipped: %s", e)

            pipeline = await loop.run_in_executor(None, diarizer.load_pipeline)
            try:
                for sess in sessions:
                    if _batch_state["cancel_requested"]:
                        log.info("Batch diarization cancelled by user")
                        break

                    _batch_state["current_session"] = sess["path"]
                    log.info(f"Batch diarize: {sess['path']}")

                    try:
                        updated, success = await loop.run_in_executor(
                            None,
                            lambda s=sess: diarizer.diarize_with_pipeline(
                                pipeline, str(s["audio"]), s["data"]["segments"]
                            ),
                        )

                        if not success:
                            _batch_state["failed"] += 1
                            err_msg = f"{sess['path']}: diarization failed"
                            _batch_state["errors"].append(err_msg)
                            log.warning(f"Batch diarize failed: {sess['path']}")
                            # 失敗を transcript に永続化しないと、翌晩また同じ
                            # セッションを選択して永遠にリトライし続ける。
                            try:
                                with open(sess["transcript_path"], encoding="utf-8") as f:
                                    fresh = json.load(f)
                                si = fresh.setdefault("session", {})
                                si["diarize_attempts"] = int(si.get("diarize_attempts", 0)) + 1
                                tmp_path = str(sess["transcript_path"]) + ".tmp"
                                with open(tmp_path, "w", encoding="utf-8") as f:
                                    json.dump(fresh, f, ensure_ascii=False, indent=2)
                                os.replace(tmp_path, str(sess["transcript_path"]))
                            except Exception:
                                log.warning(
                                    "Failed to persist diarize_attempts for %s",
                                    sess["path"],
                                )
                            continue

                        # Safe write: tmp + rename. Re-read the file first —
                        # the snapshot in sess["data"] is hours old by now and
                        # writing it back would erase tags / LLM corrections
                        # made since the batch started (lost update).
                        data = sess["data"]
                        try:
                            with open(sess["transcript_path"], encoding="utf-8") as f:
                                fresh = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            fresh = None
                        if fresh is not None:
                            fresh_segments = fresh.get("segments", [])
                            if len(fresh_segments) == len(updated):
                                # Graft only the diarization result onto the
                                # current segments (texts may have been
                                # corrected meanwhile).
                                for fs, us in zip(fresh_segments, updated):
                                    if "speaker" in us:
                                        fs["speaker"] = us["speaker"]
                            else:
                                fresh["segments"] = updated
                            data = fresh
                        else:
                            data["segments"] = updated
                        data.setdefault("session", {})["diarized"] = True
                        speakers = set(s.get("speaker", "SPEAKER_00") for s in updated)
                        data["session"]["num_speakers"] = len(speakers)

                        tmp_path = str(sess["transcript_path"]) + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, str(sess["transcript_path"]))

                        _batch_state["processed"] += 1
                        log.info(f"Batch diarize done: {sess['path']} ({len(speakers)} speakers)")

                    except Exception as e:
                        _batch_state["failed"] += 1
                        err_msg = f"{sess['path']}: {e}"
                        _batch_state["errors"].append(err_msg)
                        log.error(f"Batch diarize error: {err_msg}", exc_info=True)

                    # OOM対策: 各セッションで読み込んだ音声バッファ等を即解放
                    sess["data"] = None
                    gc.collect()
                    await asyncio.sleep(0)
            finally:
                await loop.run_in_executor(None, lambda: diarizer.unload_pipeline(pipeline))
                # This frame's reference kept the ~2GB model alive through the
                # empty_cache() above — drop it and flush again for real.
                pipeline = None
                await loop.run_in_executor(None, diarizer._free_vram)
                gc.collect()

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

    cutoff = _batch_cutoff_date()
    sessions = []
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        if cutoff and day_dir.name < cutoff:
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
        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_sessions(force=force)
        )
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

                # Re-read before write (lost-update guard): the snapshot in
                # sess["data"] predates any corrections made while this batch
                # was running (same pattern as the combined-tag voice phase).
                data = sess["data"]
                try:
                    with open(sess["transcript_path"], encoding="utf-8") as f:
                        fresh = json.load(f)
                except (json.JSONDecodeError, OSError):
                    fresh = None
                if fresh is not None and len(fresh.get("segments", [])) == len(updated):
                    fresh_segments = fresh["segments"]
                    for i, us in enumerate(updated):
                        fs = fresh_segments[i]
                        if fs.get("manually_tagged"):
                            continue
                        for key in ("person_id", "person_name",
                                    "is_user", "speaker_similarity"):
                            if key in us:
                                fs[key] = us[key]
                    data = fresh
                else:
                    # Preserve manually_tagged segments (snapshot view)
                    for i, orig in enumerate(original_segments):
                        if orig.get("manually_tagged") and i < len(updated):
                            updated[i] = orig
                    data["segments"] = updated
                data.setdefault("session", {})["speaker_tagged"] = True

                tmp_path = str(sess["transcript_path"]) + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, str(sess["transcript_path"]))

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

    cutoff = _batch_cutoff_date()
    for day_dir in sorted(recordings.iterdir()):
        if not day_dir.is_dir():
            continue
        if cutoff and day_dir.name < cutoff:
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

    cutoff = _batch_cutoff_date()
    for day_dir in sorted(recordings.iterdir()):
        if not day_dir.is_dir():
            continue
        if cutoff and day_dir.name < cutoff:
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

            # Legacy sessions: already classified before session.json was introduced
            if not force and not session_json.exists():
                try:
                    with open(transcript_path, encoding="utf-8") as f:
                        tdata = json.load(f)
                    tags = tdata.get("session", {}).get("tags", {})
                    if tags.get("external_audio_ratio") is not None:
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

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
        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_session_tag_sessions(force=force)
        )
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
                    data = sess["data"]
                    data["session"]["tags"] = tags
                    data["session"]["session_tagged"] = True

                    tmp_path = str(sess["transcript_path"]) + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, str(sess["transcript_path"]))

                    # Update RAG index
                    _update_rag_index(sess["path"], segments, tags)

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

    async def _run_voice_tag_phase():
        """Phase 1: Voice Tag (GPU) — runs in parallel with PANNs."""
        st["phase"] = "voice_tag"
        if speaker_filter is None:
            log.info("Combined tag Phase 1 skipped: no speaker_filter (HF_AUTH_TOKEN not set)")
            return

        voice_sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_sessions(force=force)
        )
        st["voice_tag_total"] = len(voice_sessions)
        st["total"] = len(voice_sessions)
        log.info(f"Combined tag Phase 1 (voice tag): {len(voice_sessions)} session(s) (force={force})")

        if not voice_sessions:
            return

        loop = asyncio.get_running_loop()

        # Load model once for entire batch
        async with model_manager.gpu_lock:
            inference = await loop.run_in_executor(
                None, speaker_filter._load_model
            )
            try:
                for sess in voice_sessions:
                    if st["cancel_requested"]:
                        log.info("Combined tag cancelled by user (voice tag phase)")
                        break

                    st["current_session"] = sess["path"]
                    try:
                        original_segments = sess["data"]["segments"]
                        updated = await loop.run_in_executor(
                            None,
                            lambda s=sess: speaker_filter.tag_segments_with_model(
                                inference, str(s["audio"]), s["data"]["segments"]
                            ),
                        )

                        # Re-read before write (lost-update guard): the batch
                        # snapshot is loaded up-front, so writing it back would
                        # erase corrections made while the batch was running —
                        # including manually_tagged set from the viewer.
                        data = sess["data"]
                        try:
                            with open(sess["transcript_path"], encoding="utf-8") as f:
                                fresh = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            fresh = None
                        if fresh is not None and len(fresh.get("segments", [])) == len(updated):
                            fresh_segments = fresh["segments"]
                            for i, us in enumerate(updated):
                                fs = fresh_segments[i]
                                if fs.get("manually_tagged"):
                                    continue
                                for key in ("person_id", "person_name",
                                            "is_user", "speaker_similarity"):
                                    if key in us:
                                        fs[key] = us[key]
                            data = fresh
                        else:
                            # Preserve manually_tagged segments (snapshot view)
                            for i, orig in enumerate(original_segments):
                                if orig.get("manually_tagged") and i < len(updated):
                                    updated[i] = orig
                            data["segments"] = updated
                        data.setdefault("session", {})["speaker_tagged"] = True

                        tmp_path = str(sess["transcript_path"]) + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, str(sess["transcript_path"]))

                        st["voice_tag_processed"] += 1
                        st["processed"] += 1

                    except Exception as e:
                        st["voice_tag_failed"] += 1
                        st["failed"] += 1
                        err_msg = f"{sess['path']}: {e}"
                        st["errors"].append(err_msg)
                        log.error(f"Combined tag voice-tag error: {err_msg}", exc_info=True)

                    await asyncio.sleep(0)
            finally:
                await loop.run_in_executor(
                    None, lambda: speaker_filter._unload_model(inference)
                )

    async def _run_audio_scene_phase():
        """Phase 1.5: Audio Scene (CPU) — runs in parallel with voice tag."""
        if not AUDIO_SCENE_ENABLED:
            return

        scene_sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_unscene_classified_sessions(force=force)
        )
        st["audio_scene_total"] = len(scene_sessions)
        log.info(f"Combined tag Phase 1.5 (audio scene): {len(scene_sessions)} session(s)")

        loop = asyncio.get_running_loop()
        for sess_dir, sess_data in scene_sessions:
            if st["cancel_requested"]:
                log.info("Combined tag cancelled by user (audio scene phase)")
                break
            try:
                audio_path = os.path.join(sess_dir, "audio.ogg")
                if not os.path.exists(audio_path):
                    audio_path = os.path.join(sess_dir, "audio.wav")
                if not os.path.exists(audio_path):
                    log.warning("No audio file in %s, skipping scene classification", sess_dir)
                    st["audio_scene_failed"] += 1
                    continue
                # Run PANNs in executor (CPU-bound)
                result = await loop.run_in_executor(
                    None, lambda p=audio_path: audio_scene_classifier.classify(p)
                )
                if result:
                    sess_data["audio_scene"] = result
                    session_json = os.path.join(sess_dir, "session.json")
                    tmp_path = session_json + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(sess_data, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, session_json)
                    log.info("Audio scene for %s: %s", os.path.basename(sess_dir), result.get("scene_label", "unknown"))
                    st["audio_scene_processed"] += 1
                else:
                    log.warning("Audio scene for %s: no classification result", os.path.basename(sess_dir))
                    st["audio_scene_failed"] += 1
            except Exception as e:
                log.error("Audio scene classification failed for %s: %s", sess_dir, e)
                st["audio_scene_failed"] += 1
            await asyncio.sleep(0)

    try:
        # --- Phase 1 + 1.5: Voice Tag (GPU) and Audio Scene (CPU) in parallel ---
        # PANNs runs on CPU, voice tag on GPU — no resource conflict.
        st["phase"] = "voice_tag+audio_scene"
        log.info("Combined tag: starting voice tag (GPU) + audio scene (CPU) in parallel")
        # return_exceptions: gather のデフォルトは最初の例外で即座に伝播する
        # が、もう一方のフェーズはキャンセルされず走り続ける。その状態で
        # finally が running=False を立てると、二重バッチ起動を許してしまう。
        results = await asyncio.gather(
            _run_voice_tag_phase(),
            _run_audio_scene_phase(),
            return_exceptions=True,
        )
        for name, r in zip(("voice_tag", "audio_scene"), results):
            if isinstance(r, BaseException):
                log.error(f"Combined tag: {name} phase failed: {r!r}")
                st["errors"].append(f"{name}: {r}")

        # --- Phase 2: Session Tag (LLM) ---
        # 声紋タグ済みのセグメントを使って LLM タグ付けを実行する。
        # Gemini Batch API が利用可能なら一括投入し、なければ sync で逐次処理。
        if st["cancel_requested"]:
            return

        st["phase"] = "session_tag"
        session_tag_sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_session_tag_sessions(force=force)
        )
        st["session_tag_total"] = len(session_tag_sessions)
        st["total"] = st["voice_tag_total"] + st["audio_scene_total"] + len(session_tag_sessions)
        st["processed"] = st["voice_tag_processed"] + st["voice_tag_failed"]
        st["failed"] = 0  # Reset failed count for combined progress
        log.info(f"Combined tag Phase 2 (session tag): {len(session_tag_sessions)} session(s) (force={force})")

        if session_tag_sessions and session_tagger is not None:
            loop = asyncio.get_running_loop()

            # Prepare: load segments/audio_scene, handle RAG-confident sessions sync
            batch_candidates = []  # (sess, fresh_data, segments, audio_scene)
            for sess in session_tag_sessions:
                if st["cancel_requested"]:
                    break
                st["current_session"] = sess["path"]
                try:
                    with open(sess["transcript_path"], "r", encoding="utf-8") as f:
                        fresh_data = json.load(f)
                    segments = fresh_data.get("segments", [])

                    _sess_json_path = Path(sess["transcript_path"]).parent / "session.json"
                    _audio_scene = None
                    if _sess_json_path.exists():
                        try:
                            with open(_sess_json_path, encoding="utf-8") as _sf:
                                _audio_scene = json.load(_sf).get("audio_scene")
                        except (json.JSONDecodeError, OSError):
                            pass

                    # Try RAG first (sync, no API cost)
                    tags = await loop.run_in_executor(
                        None,
                        lambda segs=segments, asc=_audio_scene: session_tagger.tag_session_rag_only(segs, audio_scene=asc),
                    )
                    if tags:
                        # RAG confident — apply immediately
                        fresh_data["session"]["tags"] = tags
                        fresh_data["session"]["session_tagged"] = True
                        tmp_path = str(sess["transcript_path"]) + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(fresh_data, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, str(sess["transcript_path"]))
                        _update_rag_index(sess["path"], segments, tags)
                        st["session_tag_processed"] += 1
                        st["processed"] += 1
                    else:
                        batch_candidates.append((sess, fresh_data, segments, _audio_scene))
                except Exception as e:
                    st["session_tag_failed"] += 1
                    st["failed"] += 1
                    st["errors"].append(f"{sess['path']}: {e}")
                    log.error(f"Combined tag session-tag prep error: {sess['path']}: {e}")
                await asyncio.sleep(0)

            # Try Gemini Batch API for remaining LLM-needed sessions.
            # TAGGING_MODEL が gemini 以外 (gpt-* 等) の場合は Gemini Batch に
            # 送れないため、最初から同期パス (OpenAI 互換クライアント) に回す。
            genai_client = None
            if batch_candidates and GEMINI_API_KEY and TAGGING_MODEL.startswith("gemini"):
                try:
                    from google import genai
                    genai_client = genai.Client(api_key=GEMINI_API_KEY)
                except Exception as e:
                    log.warning(f"Gemini genai init failed for session tag batch, falling back to sync: {e}")

            needs_sync_fallback = False
            batch_tagged_paths: set = set()  # Batch API で処理済み (sync 二重処理防止)
            if batch_candidates and genai_client and not st["cancel_requested"]:
                # Build batch requests. Each request carries the session_id in
                # metadata (google-genai InlinedRequest.metadata) so results
                # can be matched by key instead of trusting response order.
                batch_requests = []
                batch_session_map = []  # parallel list: (sess, fresh_data, segments, audio_scene)
                batch_by_key = {}       # session_id -> same tuple, for keyed matching
                for sess, fresh_data, segments, audio_scene in batch_candidates:
                    req = session_tagger.build_batch_request(segments, audio_scene=audio_scene)
                    if req:
                        req["metadata"] = {"session_id": sess["path"]}
                        batch_requests.append(req)
                        entry = (sess, fresh_data, segments, audio_scene)
                        batch_session_map.append(entry)
                        batch_by_key[sess["path"]] = entry

                if batch_requests:
                    log.info(f"Combined tag: submitting {len(batch_requests)} session(s) to Batch API")
                    try:
                        batch_result = await loop.run_in_executor(
                            None,
                            lambda: genai_client.batches.create(
                                model=TAGGING_MODEL,
                                src=batch_requests,
                                config={"display_name": f"omi-tag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"},
                            ),
                        )
                        log.info(f"Combined tag: Batch API submitted → {batch_result.name}")

                        # Poll for completion (timeout 2h → sync fallback)
                        _batch_tag_timeout = 2 * 60 * 60  # 2 hours
                        _batch_tag_start = asyncio.get_event_loop().time()
                        _batch_tag_done = False
                        while not st["cancel_requested"]:
                            await asyncio.sleep(30)
                            if asyncio.get_event_loop().time() - _batch_tag_start > _batch_tag_timeout:
                                log.warning("Combined tag: Batch API timeout (2h), falling back to sync")
                                break
                            batch_job = await loop.run_in_executor(
                                None,
                                lambda: genai_client.batches.get(name=batch_result.name),
                            )
                            state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
                            if state == "JOB_STATE_SUCCEEDED":
                                log.info(f"Combined tag: Batch API completed → {batch_result.name}")
                                responses = list(genai_client.batches.list_results(name=batch_result.name))

                                # Prefer keyed matching via echoed request
                                # metadata; only if the SDK/API did not echo
                                # keys, fall back to positional mapping with
                                # the strict count guard.
                                def _resp_key(r):
                                    md = getattr(r, "metadata", None)
                                    if isinstance(md, dict):
                                        return md.get("session_id")
                                    return None

                                keyed = [(_resp_key(r), r) for r in responses]
                                # bool(keyed): all([]) は真なので、空レスポンス
                                # (quota 切れ等) が「キー突合成功・全件処理済み」
                                # と誤判定され sync フォールバックが走らなくなる
                                use_keys = (
                                    bool(keyed)
                                    and all(k for k, _ in keyed)
                                    and len({k for k, _ in keyed}) == len(keyed)
                                )
                                if use_keys:
                                    matched = [
                                        (batch_by_key[k], r)
                                        for k, r in keyed if k in batch_by_key
                                    ]
                                    unknown = [k for k, _ in keyed if k not in batch_by_key]
                                    if unknown:
                                        log.warning(
                                            "Combined tag: %d batch response(s) with "
                                            "unknown session keys ignored", len(unknown),
                                        )
                                elif len(responses) != len(batch_session_map):
                                    # No keys AND count mismatch: positional
                                    # mapping would write tags to the WRONG
                                    # sessions silently — abort to sync.
                                    log.error(
                                        "Combined tag: Batch API returned %d responses "
                                        "for %d requests without usable keys — "
                                        "falling back to sync",
                                        len(responses), len(batch_session_map),
                                    )
                                    break
                                else:
                                    log.info("Combined tag: no metadata keys echoed; using positional mapping")
                                    matched = list(zip(batch_session_map, responses))

                                for entry, resp in matched:
                                    sess, fresh_data, segments, audio_scene = entry
                                    try:
                                        raw_text = resp.response.candidates[0].content.parts[0].text
                                        tags = session_tagger.parse_batch_response(raw_text, segments, audio_scene=audio_scene)
                                        if tags:
                                            fresh_data["session"]["tags"] = tags
                                            fresh_data["session"]["session_tagged"] = True
                                            tmp_path = str(sess["transcript_path"]) + ".tmp"
                                            with open(tmp_path, "w", encoding="utf-8") as f:
                                                json.dump(fresh_data, f, ensure_ascii=False, indent=2)
                                            os.replace(tmp_path, str(sess["transcript_path"]))
                                            _update_rag_index(sess["path"], segments, tags)
                                            st["session_tag_processed"] += 1
                                            st["processed"] += 1
                                            batch_tagged_paths.add(sess["path"])
                                        else:
                                            st["session_tag_failed"] += 1
                                            st["failed"] += 1
                                            st["errors"].append(f"{sess['path']}: batch response parse failed")
                                    except Exception as e:
                                        st["session_tag_failed"] += 1
                                        st["failed"] += 1
                                        st["errors"].append(f"{sess['path']}: {e}")
                                # 部分的な結果 (一部セッションに応答なし) は残りを
                                # sync フォールバックへ回す
                                _batch_tag_done = len(matched) == len(batch_session_map)
                                if not _batch_tag_done:
                                    log.warning(
                                        "Combined tag: batch covered %d/%d session(s); "
                                        "rest go to sync fallback",
                                        len(matched), len(batch_session_map),
                                    )
                                break
                            elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                                log.error(f"Combined tag: Batch API failed ({state})")
                                break
                            # Still pending — continue polling

                        # Sync fallback: timeout, failed, or cancelled
                        if not _batch_tag_done and not st["cancel_requested"]:
                            log.info(f"Combined tag: sync fallback for {len(batch_session_map)} session(s)")
                            # genai_client is still set here, so the flag is the
                            # only way the fallback below actually runs (it was
                            # dead code before: timeout/JOB_STATE_FAILED left
                            # sessions untagged and re-submitted every night).
                            needs_sync_fallback = True
                    except Exception as e:
                        log.error(f"Combined tag: Batch API submit failed: {e}, falling back to sync")
                        genai_client = None  # fall through to sync

            # Sync fallback for sessions that weren't batch-processed
            if (needs_sync_fallback or not genai_client) and batch_candidates and not st["cancel_requested"]:
                for sess, fresh_data, segments, audio_scene in batch_candidates:
                    if st["cancel_requested"]:
                        break
                    if sess["path"] in batch_tagged_paths:
                        continue  # Batch API で処理済み — 二重タグ付け/課金を防ぐ
                    st["current_session"] = sess["path"]
                    try:
                        tags = await loop.run_in_executor(
                            None,
                            lambda segs=segments, asc=audio_scene: session_tagger.tag_session(segs, audio_scene=asc),
                        )
                        if tags:
                            fresh_data["session"]["tags"] = tags
                            fresh_data["session"]["session_tagged"] = True
                            tmp_path = str(sess["transcript_path"]) + ".tmp"
                            with open(tmp_path, "w", encoding="utf-8") as f:
                                json.dump(fresh_data, f, ensure_ascii=False, indent=2)
                            os.replace(tmp_path, str(sess["transcript_path"]))
                            _update_rag_index(sess["path"], segments, tags)
                            st["session_tag_processed"] += 1
                            st["processed"] += 1
                        else:
                            st["session_tag_failed"] += 1
                            st["failed"] += 1
                    except Exception as e:
                        st["session_tag_failed"] += 1
                        st["failed"] += 1
                        st["errors"].append(f"{sess['path']}: {e}")
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
# VibeVoice-ASR バッチ文字起こし
# ============================================================

_batch_vibevoice_state = {
    "running": False,
    "phase": None,            # "vibevoice" | "voice_tag"
    "total": 0,
    "processed": 0,
    "failed": 0,
    "current_session": None,
    "errors": [],
    "cancel_requested": False,
    "started_at": None,
    "finished_at": None,
    "voice_tag_total": 0,
    "voice_tag_processed": 0,
    "voice_tag_failed": 0,
}


def _find_unvibevoiced_sessions(force: bool = False) -> list[dict]:
    """Find sessions that haven't been transcribed by VibeVoice-ASR."""
    sessions = []
    rec_root = Path(RECORDINGS_DIR)
    if not rec_root.exists():
        return sessions

    for date_dir in sorted(rec_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            try:
                data = json.loads(transcript_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sess = data.get("session", {})
            if not force and sess.get("vibevoice_transcribed"):
                continue
            # Need audio file
            audio = time_dir / "audio.ogg"
            if not audio.exists():
                audio = time_dir / "audio.wav"
            if not audio.exists():
                continue
            sessions.append({
                "path": f"{date_dir.name}/{time_dir.name}",
                "audio": audio,
                "transcript_path": transcript_path,
                "data": data,
            })
    return sessions


def _prefetch_audio(audio_path: str) -> tuple:
    """Convert audio to WAV and load into memory (CPU-only, no GPU).

    Used by the prefetch pipeline to overlap I/O with GPU inference.
    Returns (audio_data, sample_rate).
    """
    wav_path = VibeVoiceTranscriber.convert_to_wav(audio_path)
    try:
        audio_data, sr = vv_load_audio(wav_path)
        return audio_data, sr
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _find_vibevoice_untagged_sessions(force: bool = False) -> list[dict]:
    """Find sessions with vibevoice_segments that need voice tagging."""
    sessions = []
    rec_root = Path(RECORDINGS_DIR)
    if not rec_root.exists():
        return sessions

    for date_dir in sorted(rec_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir()):
            if not time_dir.is_dir():
                continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            try:
                data = json.loads(transcript_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sess = data.get("session", {})
            if not sess.get("vibevoice_transcribed"):
                continue
            if not force and sess.get("vibevoice_voice_tagged"):
                continue
            if not data.get("vibevoice_segments"):
                continue
            audio = time_dir / "audio.ogg"
            if not audio.exists():
                audio = time_dir / "audio.wav"
            if not audio.exists():
                continue
            sessions.append({
                "path": f"{date_dir.name}/{time_dir.name}",
                "audio": audio,
                "transcript_path": transcript_path,
                "data": data,
            })
    return sessions


async def _batch_vibevoice_task(force: bool = False):
    """VibeVoice-ASR バッチ: Phase 1 文字起こし → Phase 2 声紋タグ付け"""
    import shutil

    st = _batch_vibevoice_state
    st["running"] = True
    st["phase"] = None
    st["processed"] = 0
    st["failed"] = 0
    st["current_session"] = None
    st["errors"] = []
    st["cancel_requested"] = False
    st["started_at"] = datetime.now().isoformat()
    st["finished_at"] = None
    st["voice_tag_total"] = 0
    st["voice_tag_processed"] = 0
    st["voice_tag_failed"] = 0

    try:
        # --- Phase 1: VibeVoice-ASR transcription ---
        st["phase"] = "vibevoice"
        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_unvibevoiced_sessions(force=force)
        )
        # Sort by audio file size (ascending) — short sessions first for
        # better GPU utilization (less OOM risk, KV cache warmup).
        sessions.sort(key=lambda s: s["audio"].stat().st_size)
        st["total"] = len(sessions)
        log.info(f"VibeVoice batch Phase 1: {len(sessions)} session(s) (force={force})")

        if sessions:
            loop = asyncio.get_running_loop()

            # Acquire GPU lock, unload Whisper, load VibeVoice
            async with model_manager.gpu_lock:
                if not await model_manager.force_unload(wait_sessions=True):
                    # Whisper がまだ VRAM に残ったまま VibeVoice をロードすると
                    # OOM 一直線なので、このフェーズは中止して次回に回す。
                    log.error("VibeVoice batch: Whisper unload failed (active sessions) — aborting phase")
                    st["errors"].append("whisper unload failed; phase aborted")
                    return
                try:
                    await loop.run_in_executor(None, vibevoice_transcriber.load_model)

                    # Build Japanese context prompt with hotwords (once before loop)
                    hotwords_str = ""
                    if HOTWORDS_ENABLED and hotword_loader.get_settings().get("enabled", True):
                        custom = hotword_loader.get_custom_words()
                        hw_list = hotword_loader.get_hotwords_list()
                        words = list(custom)
                        for w in hw_list:
                            if w not in words:
                                words.append(w)
                            if len(words) >= 100:
                                break
                        if words:
                            hotwords_str = ", ".join(words)

                    vv_prompt = "日本語の会話を書き起こしてください。"
                    if hotwords_str:
                        vv_prompt += f"\nHotwords: {hotwords_str}"
                    log.info("VibeVoice prompt: %s", vv_prompt[:100])

                    # --- Prefetch pipeline: overlap CPU I/O with GPU inference ---
                    # Pre-load first session's audio
                    audio_data, audio_sr = None, None
                    try:
                        audio_data, audio_sr = await loop.run_in_executor(
                            None, _prefetch_audio, str(sessions[0]["audio"])
                        )
                    except Exception as e:
                        log.warning("Prefetch failed for first session: %s", e)

                    for i, sess in enumerate(sessions):
                        if st["cancel_requested"]:
                            log.info("VibeVoice batch cancelled by user (transcription phase)")
                            break

                        st["current_session"] = sess["path"]

                        # Start prefetching next session's audio (CPU) while GPU runs
                        next_future = None
                        if i + 1 < len(sessions):
                            next_future = loop.run_in_executor(
                                None, _prefetch_audio, str(sessions[i + 1]["audio"])
                            )

                        try:
                            if audio_data is not None:
                                # Use pre-loaded audio (skip file I/O)
                                segments = await loop.run_in_executor(
                                    None, vibevoice_transcriber.transcribe_audio,
                                    audio_data, audio_sr, vv_prompt
                                )
                            else:
                                # Fallback: prefetch failed, use original path
                                wav_path = await loop.run_in_executor(
                                    None, vibevoice_transcriber.convert_to_wav, str(sess["audio"])
                                )
                                try:
                                    segments = await loop.run_in_executor(
                                        None, vibevoice_transcriber.transcribe, wav_path, vv_prompt
                                    )
                                finally:
                                    try:
                                        os.unlink(wav_path)
                                    except OSError:
                                        pass

                            if segments is None:
                                raise RuntimeError("OOM: skipped")

                            # Save vibevoice_segments to transcript.json
                            data = sess["data"]
                            data["vibevoice_segments"] = segments
                            data["session"]["vibevoice_transcribed"] = True

                            with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)

                            st["processed"] += 1
                            log.info(f"VibeVoice: {sess['path']} -> {len(segments)} segments")

                        except Exception as e:
                            st["failed"] += 1
                            err_msg = f"{sess['path']}: {e}"
                            st["errors"].append(err_msg)
                            log.error(f"VibeVoice transcription error: {err_msg}", exc_info=True)

                        # Collect prefetched audio for next iteration
                        audio_data, audio_sr = None, None
                        if next_future is not None:
                            try:
                                audio_data, audio_sr = await next_future
                            except Exception as e:
                                log.warning("Prefetch failed for %s: %s", sessions[i + 1]["path"], e)

                        await asyncio.sleep(0)

                finally:
                    await loop.run_in_executor(None, vibevoice_transcriber.unload_model)

        # --- Phase 2: Voice Tag (speaker identification on vibevoice_segments) ---
        if st["cancel_requested"]:
            return

        st["phase"] = "voice_tag"
        if speaker_filter is not None:
            vt_sessions = _find_vibevoice_untagged_sessions(force=force)
            st["voice_tag_total"] = len(vt_sessions)
            log.info(f"VibeVoice batch Phase 2 (voice tag): {len(vt_sessions)} session(s)")

            if vt_sessions:
                loop = asyncio.get_running_loop()

                for sess in vt_sessions:
                    if st["cancel_requested"]:
                        log.info("VibeVoice batch cancelled by user (voice tag phase)")
                        break

                    st["current_session"] = sess["path"]
                    try:
                        async with model_manager.gpu_lock:
                            inference = await loop.run_in_executor(
                                None, speaker_filter._load_model
                            )
                            try:
                                updated = await loop.run_in_executor(
                                    None,
                                    lambda s=sess: speaker_filter.tag_segments_with_model(
                                        inference, str(s["audio"]), s["data"]["vibevoice_segments"]
                                    ),
                                )
                            finally:
                                await loop.run_in_executor(
                                    None, lambda: speaker_filter._unload_model(inference)
                                )

                        data = sess["data"]
                        data["vibevoice_segments"] = updated
                        data["session"]["vibevoice_voice_tagged"] = True

                        with open(sess["transcript_path"], "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        st["voice_tag_processed"] += 1

                    except Exception as e:
                        st["voice_tag_failed"] += 1
                        err_msg = f"{sess['path']}: {e}"
                        st["errors"].append(err_msg)
                        log.error(f"VibeVoice voice-tag error: {err_msg}", exc_info=True)

                    await asyncio.sleep(0)
        else:
            log.info("VibeVoice Phase 2 skipped: no speaker_filter")

    except Exception as e:
        log.error(f"VibeVoice batch error: {e}", exc_info=True)
        st["errors"].append(str(e))
    finally:
        st["running"] = False
        st["current_session"] = None
        st["finished_at"] = datetime.now().isoformat()
        log.info(f"VibeVoice batch finished: {st['processed']} processed, {st['failed']} failed")


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

# Persistent state path — survives container restart so missed nightly
# batches can be detected and caught up.
_NIGHTLY_STATE_PATH = os.path.join(CONFIG_DIR, "nightly_state.json")


def _load_nightly_persisted_state() -> dict:
    """Load persisted nightly state from disk. Returns {} if missing/invalid."""
    try:
        with open(_NIGHTLY_STATE_PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_nightly_persisted_state() -> None:
    """Persist last_run / last_result / last_completed_date to disk."""
    payload = {
        "last_run": _nightly_state.get("last_run"),
        "last_result": _nightly_state.get("last_result"),
        "last_completed_date": _nightly_state.get("last_completed_date"),
        "error": _nightly_state.get("error"),
    }
    try:
        os.makedirs(os.path.dirname(_NIGHTLY_STATE_PATH), exist_ok=True)
        tmp = _NIGHTLY_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _NIGHTLY_STATE_PATH)
    except OSError as e:
        log.warning("Nightly scheduler: failed to persist state: %s", e)


def _calc_next_nightly(hour: int, now: datetime | None = None) -> str:
    """Calculate the next run datetime for the given hour (local time)."""
    if now is None:
        now = datetime.now()
    today_target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= today_target:
        from datetime import timedelta
        next_run = today_target + timedelta(days=1)
    else:
        next_run = today_target
    return next_run.isoformat()


def _should_catch_up_nightly(hour: int, persisted: dict, now: datetime | None = None) -> bool:
    """Return True if today's nightly window has passed but no successful run
    is recorded for today — i.e., we missed it (e.g., due to crash/restart).

    A run counts as "done for today" if `last_completed_date` equals today's
    date in YYYY-MM-DD form.
    """
    if now is None:
        now = datetime.now()
    today_target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now < today_target:
        return False  # window hasn't opened yet
    today_str = now.strftime("%Y-%m-%d")
    return persisted.get("last_completed_date") != today_str


# ============================================================
# 夜間自動 sanitize: 選定ロジックと試行台帳
# ============================================================

# main() 内で定義されるクロージャ (_batch_sanitize_task / state) への参照。
# 夜間スケジューラ (モジュールレベル) から呼ぶために登録する。
_sanitize_hooks: dict = {}

_SANITIZE_LEDGER_PATH = os.path.join(CONFIG_DIR, "sanitize_ledger.json")

# 事業継承価値の判定 (viewer indexer の high_value_or と同じ基準)
_SANITIZE_USEFUL_CONV_TYPES = {
    "ミーティング", "社内ミーティング", "アイデア出し", "商談",
    "顧客ヒアリング", "1on1", "外部セミナー",
}
_SANITIZE_USEFUL_OUTCOMES = {
    "意思決定あり", "TODO発生", "方針転換", "アイデア創出", "問題提起",
}


def _load_sanitize_ledger() -> dict:
    try:
        with open(_SANITIZE_LEDGER_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _record_sanitize_outcome(session_id: str, outcome: str) -> None:
    """Persist per-session sanitize outcome (done/dropped/failed).

    high-risk dropped や失敗したセッションは safe_training には現れないため、
    この台帳がないと毎晩再選定されて LLM コストを払い直すことになる。
    """
    try:
        ledger = _load_sanitize_ledger()
        entry = ledger.get(session_id) or {"attempts": 0}
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["outcome"] = outcome
        entry["at"] = datetime.now().isoformat(timespec="seconds")
        ledger[session_id] = entry
        tmp = _SANITIZE_LEDGER_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _SANITIZE_LEDGER_PATH)
    except OSError as e:
        log.warning(f"Sanitize ledger write failed: {e}")


def _load_sanitized_session_ids() -> set:
    """Collect session_ids already present in config/safe_training_*.jsonl."""
    ids: set = set()
    for fp in sorted(Path(CONFIG_DIR).glob("safe_training_*.jsonl")):
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sid = json.loads(line).get("session_id")
                    except json.JSONDecodeError:
                        continue
                    if sid:
                        ids.add(sid)
        except OSError:
            continue
    return ids


def _find_unsanitized_sessions() -> list:
    """夜間自動 sanitize の対象セッションを古い順に返す (上限つき)。

    対象: タグ付け済み (V2) で、まだ sanitize されておらず、
    台帳上も未処理 (失敗は NIGHTLY_SANITIZE_MAX_ATTEMPTS 回まで再試行)。
    個人のみ / 外部音声 80%+ は除外。NIGHTLY_SANITIZE_HIGH_VALUE_ONLY の
    場合は事業継承価値のあるセッションに絞る。
    """
    base = Path(RECORDINGS_DIR)
    if not base.exists():
        return []

    sanitized = _load_sanitized_session_ids()
    ledger = _load_sanitize_ledger()

    # Prune: retention で削除されたセッションの台帳エントリを落とす
    # (放置すると無限成長し、毎セッションの全書き換えが重くなる)
    stale = [sid for sid in ledger if not (base / sid).is_dir()]
    if stale:
        for sid in stale:
            ledger.pop(sid, None)
        try:
            tmp = _SANITIZE_LEDGER_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(ledger, f, ensure_ascii=False, indent=1)
            os.replace(tmp, _SANITIZE_LEDGER_PATH)
            log.info("Sanitize ledger: pruned %d stale entrie(s)", len(stale))
        except OSError:
            pass

    result: list = []

    for day_dir in sorted(base.iterdir()):
        if len(result) >= NIGHTLY_SANITIZE_MAX_SESSIONS:
            break
        if not day_dir.is_dir():
            continue
        for time_dir in sorted(day_dir.iterdir()):
            if len(result) >= NIGHTLY_SANITIZE_MAX_SESSIONS:
                break
            if not time_dir.is_dir():
                continue
            sid = f"{day_dir.name}/{time_dir.name}"
            if sid in sanitized:
                continue
            entry = ledger.get(sid)
            if entry:
                if entry.get("outcome") in ("done", "dropped"):
                    continue
                if int(entry.get("attempts", 0)) >= NIGHTLY_SANITIZE_MAX_ATTEMPTS:
                    continue
            transcript_path = time_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            segments = data.get("segments", [])
            if len(segments) < 3:
                continue
            tags = data.get("session", {}).get("tags", {}) or {}
            if not tags.get("conversation_type"):
                continue  # V2 タグ待ち (今夜の Phase 2 後、明晩以降に対象化)

            domains = tags.get("domains") or []
            if domains == ["個人"]:
                continue
            ext = tags.get("external_audio_ratio")
            if ext is not None and ext >= 0.8:
                continue

            if NIGHTLY_SANITIZE_HIGH_VALUE_ONLY:
                outcomes = set(tags.get("outcomes") or [])
                high_value = (
                    tags.get("importance") in ("中", "高")
                    or tags.get("conversation_type") in _SANITIZE_USEFUL_CONV_TYPES
                    or bool(tags.get("decision_memo"))
                    or bool(outcomes & _SANITIZE_USEFUL_OUTCOMES)
                )
                if not high_value:
                    continue

            result.append(sid)

    return result


async def _nightly_scheduler():
    """Background task: run diarize + combined tag at NIGHTLY_SCHEDULE_HOUR.

    Catch-up: on startup, if today's window has already passed and the
    persisted state shows no completion for today, run immediately. This
    guarantees that a missed nightly (e.g., container OOM-killed mid-run)
    is recovered as soon as the bridge comes back up.
    """
    hour = NIGHTLY_SCHEDULE_HOUR
    if hour <= 0 or hour > 23:
        return

    persisted = _load_nightly_persisted_state()
    _nightly_state["enabled"] = True
    _nightly_state["hour"] = hour
    _nightly_state["last_run"] = persisted.get("last_run")
    _nightly_state["last_result"] = persisted.get("last_result")
    _nightly_state["last_completed_date"] = persisted.get("last_completed_date")
    _nightly_state["next_run"] = _calc_next_nightly(hour)

    catch_up = _should_catch_up_nightly(hour, persisted)
    log.info(
        "Nightly scheduler: enabled (hour=%d, batch_max_days=%d, next=%s, catch_up=%s, last_completed_date=%s)",
        hour, BATCH_MAX_DAYS, _nightly_state["next_run"], catch_up,
        persisted.get("last_completed_date"),
    )

    first_iteration = True
    while True:
        if first_iteration and catch_up:
            log.info(
                "Nightly scheduler: catch-up triggered (last_completed_date=%s, now=%s)",
                persisted.get("last_completed_date"),
                datetime.now().isoformat(),
            )
            sleep_seconds = 0.0
        else:
            # Sleep until next target time
            now = datetime.now()
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if now >= target:
                from datetime import timedelta
                target = target + timedelta(days=1)

            _nightly_state["next_run"] = target.isoformat()
            sleep_seconds = (target - now).total_seconds()
            log.debug("Nightly scheduler: sleeping %.0f seconds until %s", sleep_seconds, target.isoformat())

        first_iteration = False
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

        # Check if any batch is already running
        if (_batch_state.get("running")
                or _batch_combined_tag_state.get("running")
                or _batch_session_tag_state.get("running")
                or _batch_vibevoice_state.get("running")):
            log.info("Nightly scheduler: skipped (another batch is running)")
            _nightly_state["last_run"] = datetime.now().isoformat()
            _nightly_state["last_result"] = "skipped"
            continue

        log.info("Nightly scheduler: starting (hour=%d)", hour)
        _nightly_state["running"] = True
        _nightly_state["error"] = None

        phase_errors = []
        try:
            # Phase 1: Diarize
            if diarizer is not None:
                try:
                    _nightly_state["phase"] = "diarize"
                    log.info("Nightly scheduler: Phase 1 — diarize")
                    await _batch_diarize_task(force=False)
                except Exception as e:
                    log.error("Nightly scheduler: Phase 1 (diarize) error: %s", e, exc_info=True)
                    phase_errors.append(f"diarize: {e}")

            # Phase 2: Combined tag (voice tag + audio scene + session tag)
            try:
                _nightly_state["phase"] = "tag"
                log.info("Nightly scheduler: Phase 2 — combined tag (voice → audio scene → session)")
                await _batch_combined_tag_task(force=False)
            except Exception as e:
                log.error("Nightly scheduler: Phase 2 (combined tag) error: %s", e, exc_info=True)
                phase_errors.append(f"combined_tag: {e}")

            # Phase 2.5: Auto-sanitize (Safe Training Data 自動生成)
            # タグ付け直後に実行するので、その夜タグが付いたセッションも対象。
            # 生成物は KC エクスポートと翌朝の Wiki ビルドの材料になる。
            if (NIGHTLY_SANITIZE_ENABLED
                    and _sanitize_hooks.get("task") is not None
                    and (training_sanitizer is not None
                         or responses_sanitizer is not None)):
                try:
                    _nightly_state["phase"] = "sanitize"
                    san_state = _sanitize_hooks.get("state") or {}
                    if san_state.get("running"):
                        log.info("Nightly scheduler: Phase 2.5 skipped (sanitize already running)")
                    else:
                        # 選定スキャン (数秒〜) の間に手動 /sanitize/start が
                        # 409 を素通りして二重実行にならないよう、先に立てる
                        san_state["running"] = True
                        try:
                            paths = await asyncio.get_running_loop().run_in_executor(
                                None, _find_unsanitized_sessions
                            )
                            if paths:
                                log.info(
                                    "Nightly scheduler: Phase 2.5 — auto-sanitize %d session(s) "
                                    "(max %d/night)", len(paths), NIGHTLY_SANITIZE_MAX_SESSIONS,
                                )
                                await _sanitize_hooks["task"](paths)
                            else:
                                log.info("Nightly scheduler: Phase 2.5 — no sessions to sanitize")
                        finally:
                            # タスク自身の finally でも下ろすが、paths が空で
                            # タスクを呼ばなかった場合はここで下ろす
                            san_state["running"] = False
                except Exception as e:
                    log.error("Nightly scheduler: Phase 2.5 (sanitize) error: %s", e, exc_info=True)
                    phase_errors.append(f"sanitize: {e}")

            # Phase 2.6: デイリー・エリシテーション (回答取り込み + 質問生成)
            if ELICITATION_ENABLED:
                try:
                    _nightly_state["phase"] = "elicitation"
                    await _run_elicitation_phase()
                except Exception as e:
                    log.error("Nightly scheduler: Phase 2.6 (elicitation) error: %s", e, exc_info=True)
                    phase_errors.append(f"elicitation: {e}")

            # Phase 2.7: Retention cleanup — 自動 sanitize の後に実行する。
            # 先に消すと「上限50件/晩の sanitize が追いつく前に高価値セッション
            # が保持期限で消える」raceになる (起動時 cleanup は従来通り別途実行)。
            try:
                _nightly_state["phase"] = "cleanup"
                await asyncio.get_running_loop().run_in_executor(
                    None, cleanup_old_recordings
                )
            except Exception as e:
                log.error("Nightly scheduler: cleanup phase error: %s", e)
                phase_errors.append(f"cleanup: {e}")

            # Phase 3: Monthly taxonomy refinement (1st of month only)
            now_after_tag = datetime.now()
            if taxonomy_refiner and now_after_tag.day == 1:
                try:
                    _nightly_state["phase"] = "taxonomy_refine"
                    log.info("Nightly scheduler: Phase 3 — monthly taxonomy refinement")
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, taxonomy_refiner.run)
                except Exception as e:
                    log.error("Nightly scheduler: Phase 3 (taxonomy) error: %s", e, exc_info=True)
                    phase_errors.append(f"taxonomy: {e}")

            _nightly_state["last_result"] = "error" if phase_errors else "success"
            if phase_errors:
                _nightly_state["error"] = "; ".join(phase_errors)
        except Exception as e:
            log.error("Nightly scheduler: fatal error: %s", e, exc_info=True)
            _nightly_state["last_result"] = "error"
            _nightly_state["error"] = str(e)
        finally:
            _nightly_state["running"] = False
            _nightly_state["phase"] = None
            now_done = datetime.now()
            _nightly_state["last_run"] = now_done.isoformat()
            # Mark today as completed regardless of phase_errors — partial
            # success still counts so we don't loop catch-up forever on a
            # persistent failure (next day's run will retry).
            _nightly_state["last_completed_date"] = now_done.strftime("%Y-%m-%d")
            _nightly_state["next_run"] = _calc_next_nightly(hour, now_done)
            _save_nightly_persisted_state()
            log.info(
                "Nightly scheduler: finished (result=%s, next=%s, completed_date=%s)",
                _nightly_state["last_result"], _nightly_state["next_run"],
                _nightly_state["last_completed_date"],
            )


# ============================================================
# RAG インデックス自動構築
# ============================================================

async def _auto_build_rag_index():
    """Startup hook: build RAG index in background if not present."""
    if not rag_classifier:
        return
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, rag_classifier.build_index)
        log.info("RAG index built: %d sessions (embedded=%d, errors=%d)",
                 result.get("total", 0), result.get("embedded", 0), result.get("errors", 0))
    except Exception as e:
        log.error("RAG auto-build failed: %s", e)


def _update_rag_index(session_id: str, segments: list[dict], tags: dict) -> None:
    """Add a newly tagged session to the RAG index (called after tag_session)."""
    if not rag_classifier:
        return
    try:
        plain_text = RagClassifier._extract_plain_text(segments)
        if len(plain_text) < 10:
            return
        embedding = rag_classifier.embed_text(plain_text)
        rag_classifier.add_session(session_id, embedding, {
            "conversation_type": tags.get("conversation_type", ""),
            "outcomes": tags.get("outcomes", []),
            "domains": tags.get("domains", []),
            "importance": tags.get("importance", "低"),
            "quality": tags.get("quality", "clean"),
        })
    except Exception as e:
        log.debug("RAG index update failed for %s: %s", session_id, e)


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

    # --- クエリパラメータ解析 (認証 + コーデック判定で共用) ---
    ws_path = websocket.request.path if hasattr(websocket, 'request') else websocket.path
    params = parse_qs(urlparse(ws_path).query)

    # --- トークン認証 ---
    if AUTH_TOKEN:
        token = params.get("token", [None])[0]
        if not token or not hmac.compare_digest(token, AUTH_TOKEN):
            log.warning(f"Rejected: invalid token from {remote}")
            await websocket.close(4401, "Unauthorized")
            return
        log.info(f"Authenticated connection from {remote}")
    else:
        log.info(f"Connection from {remote} (no auth)")

    # --- コーデック判定 (pcm16 = 既存デフォルト / opus = Omi CV1 互換) ---
    codec = params.get("codec", ["pcm16"])[0].lower()
    opus_decoder_inst = None
    if codec == "opus":
        frame_header = params.get("frame_header", [""])[0]
        if frame_header != "3byte":
            log.warning(f"Rejected: codec=opus requires frame_header=3byte [{remote}]")
            await websocket.close(4400, "frame_header=3byte required for opus")
            return
        try:
            opus_decoder_inst = OmiOpusStreamDecoder(
                sample_rate=SAMPLE_RATE,
                channels=1,
            )
        except Exception as e:
            log.error(f"Failed to init OPUS decoder: {e}", exc_info=True)
            await websocket.close(1011, "OPUS decoder init failed")
            return
        log.info(f"OPUS mode enabled for {remote}")

    audio_buffer = bytearray()
    session_start = time.time()
    total_audio_seconds = 0.0
    recorder_audio_seconds = 0.0  # WAV に実際に書き込んだ秒数 (録音用オフセット)
    # 遅延生成: コンストラクタがディレクトリ作成 + WAV オープンを行うため、
    # GPU busy で早期リターンする接続ごとに空ディレクトリと open fd が
    # リークしていた。最初の発話チャンクで _transcribe_chunk が生成する。
    recorder = None
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

    # Prompt carry-over: 前チャンクの出力を次チャンクの initial_prompt に渡す
    # セッション内で文脈を維持し、用語の一貫性・漢字変換精度を向上させる
    prompt_carry_text = ""  # PROMPT_CARRY_ENABLED=false の場合は常に空のまま

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

    # Realtime voice filter: load embedding model if enabled + user speakers
    # exist. acquire() の後にロードする — 以前は acquire 前に同期ロードして
    # いたため、(a) バッチ実行中でも VRAM にモデルを積んで OOM を誘発し、
    # (b) イベントループをブロックし、(c) GPU busy リジェクト時にモデルが
    # アンロードされずリークしていた。
    voice_filter_inference = None
    if VOICE_FILTER_ENABLED and speaker_filter and speaker_filter.enabled:
        user_speakers = speaker_filter.get_user_speakers()
        if user_speakers:
            try:
                voice_filter_inference = await asyncio.get_running_loop().run_in_executor(
                    None, speaker_filter._load_model
                )
                log.info(f"Voice filter: ENABLED ({len(user_speakers)} user speaker(s), "
                         f"threshold={VOICE_FILTER_THRESHOLD or speaker_filter.get_threshold()})")
            except Exception as e:
                log.warning(f"Voice filter: model load failed, continuing without filter: {e}")
                voice_filter_inference = None
        else:
            log.warning("Voice filter: ENABLED but no user speakers with embeddings registered")

    async def _transcribe_chunk(chunk: bytes):
        """チャンクを文字起こしして結果を送信する共通処理。"""
        nonlocal total_audio_seconds, recorder_audio_seconds, recorder, prompt_carry_text

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

        log.info(f"Transcribing {chunk_duration:.1f}s of audio ({ENGINE})...")
        t0 = time.time()

        loop = asyncio.get_running_loop()
        _carry = prompt_carry_text  # capture for lambda
        if ENGINE == "qwen3asr":
            raw_segments = await loop.run_in_executor(
                executor,
                lambda: model.transcribe_audio(
                    chunk, SAMPLE_RATE, 0.0, session_hotwords,
                    LANGUAGE, term_corrector, SEGMENT_MAX_DURATION,
                    audio_preprocessor,
                ),
            )
        else:
            raw_segments = await loop.run_in_executor(
                executor,
                lambda: transcribe_audio(chunk, model, 0.0, session_hotwords, _carry),
            )

        elapsed = time.time() - t0
        total_audio_seconds += chunk_duration
        recorder_audio_seconds += chunk_duration

        # Prompt carry-over: 今回の出力を次チャンク用に保持 (Whisper only)
        # Qwen3-ASR の context パラメータはホットワード用で、文脈引き継ぎには使えない
        # logprob が低いチャンクはエラー伝播を防ぐためリセット
        if PROMPT_CARRY_ENABLED and ENGINE == "whisper" and raw_segments:
            avg_lp = sum(s.get("avg_logprob", -0.5) for s in raw_segments) / len(raw_segments)
            if avg_lp >= -0.5:
                prompt_carry_text = " ".join(s["text"] for s in raw_segments[-3:])
            else:
                prompt_carry_text = ""  # low confidence → reset to avoid error propagation

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

        # Enqueue for async LLM correction + tagging (Layer 3).
        # finalize() は空セッション (total_bytes==0) のディレクトリを削除する
        # ことがあるため、session_dir.exists() を確認しないと存在しないパスを
        # キューに積んで毎回失敗する (SDカード側 handler と同じガード)。
        if (batch_corrector and rec.enabled
                and hasattr(rec, 'session_dir') and rec.session_dir
                and rec.session_dir.exists()
                and len(rec.all_segments) > 0):
            if not await _early_media_gate(rec):
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
            if opus_decoder_inst is not None:
                # OPUSパス: Omi 3バイトヘッダ剥がし + デコード → PCM16 追加
                pcm = opus_decoder_inst.feed(message)
                if pcm:
                    audio_buffer.extend(pcm)
            else:
                # PCMパス (既存): そのまま追加
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

                log.info(f"Processing remaining buffer ({ENGINE})...")
                try:
                    if ENGINE == "qwen3asr":
                        raw_segments = model.transcribe_audio(
                            remaining, SAMPLE_RATE, 0.0, session_hotwords,
                            LANGUAGE, term_corrector, SEGMENT_MAX_DURATION,
                            audio_preprocessor,
                        )
                    else:
                        raw_segments = transcribe_audio(remaining, model, 0.0, session_hotwords, prompt_carry_text)
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
            _spawn_bg(_run_diarization_bg(recorder), name='diarization_bg')
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
    log.info(f"Engine: {ENGINE}")
    if ENGINE == "qwen3asr":
        log.info(f"Model: Qwen/Qwen3-ASR-1.7B, Language: {LANGUAGE}")
    else:
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
        log.info(f"Session tagger: ON (model={TAGGING_MODEL}, prefilter={DOMAIN_PREFILTER_ENABLED})")
    else:
        reason = "DISABLED" if not TAGGING_ENABLED else "no GEMINI_API_KEY"
        log.info(f"Session tagger: {reason}")
    if rag_classifier:
        log.info(f"RAG classifier: ON (k={RAG_K}, threshold={RAG_CONFIDENCE_THRESHOLD}, min={RAG_MIN_SESSIONS})")
    else:
        log.info("RAG classifier: DISABLED")
    if VAD_SEG_ENABLED:
        log.info(f"VAD segmentation: ON (check={VAD_CHECK_INTERVAL}s, "
                 f"silence_trigger={VAD_SILENCE_TRIGGER}s, max_buffer={VAD_MAX_BUFFER}s)")
    else:
        log.info(f"VAD segmentation: OFF (fixed {BUFFER_SECONDS}s chunks)")
    log.info(f"Audio saving: {'ON -> ' + RECORDINGS_DIR if SAVE_AUDIO else 'OFF'}")
    log.info(f"Silence: threshold={SILENCE_THRESHOLD}, timeout={SILENCE_TIMEOUT}s"
             f"{' (DISABLED)' if SILENCE_THRESHOLD <= 0 else ''}")
    log.info(f"Auth: {'TOKEN required' if AUTH_TOKEN else 'DISABLED (LAN only)'}")
    log.info(f"VibeVoice-ASR: {'ENABLED (batch only)' if VIBEVOICE_ENABLED else 'DISABLED'}")
    log.info(f"Management API: http://0.0.0.0:{MANAGEMENT_PORT}")
    log.info("Model will be loaded on-demand (GPU not used until first connection)")

    # 起動時に古い録音を削除
    cleanup_old_recordings()

    # RAG index: load or build in background
    if rag_classifier:
        if rag_classifier.has_index():
            rag_classifier.load_index()
            log.info("RAG index loaded: %d sessions", rag_classifier.session_count())
        else:
            log.info("RAG index not found, building in background...")
            _spawn_bg(_auto_build_rag_index(), name='auto_build_rag')

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

        # Gemini Batch モードは gemini モデル限定。gpt-* に切り替えたまま
        # batch モードだと全ジョブが submit 失敗するため sync に落とす。
        _corrector_mode = LLM_BATCH_MODE
        if _corrector_mode == "batch" and not LLM_MODEL.startswith("gemini"):
            log.warning(
                "LLM_BATCH_MODE=batch is Gemini-only; model %s → forcing sync mode",
                LLM_MODEL,
            )
            _corrector_mode = "sync"
        batch_corrector = BatchCorrector(
            recordings_dir=RECORDINGS_DIR,
            config_dir=CONFIG_DIR,
            pipeline=llm_pipeline,
            known_terms=known_terms,
            hotword_loader=hotword_loader,
            stage_c_auto=STAGE_C_AUTO,
            min_text_length=LLM_MIN_TEXT_LENGTH,
            mode=_corrector_mode,
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

    async def handle_whisper_config(request):
        config = {
            "engine": ENGINE,
            "model_name": WHISPER_MODEL if ENGINE == "whisper" else "Qwen/Qwen3-ASR-1.7B",
            "compute_type": COMPUTE_TYPE,
            "language": LANGUAGE or "auto",
            "beam_size": BEAM_SIZE,
            "vad_filter": VAD_FILTER,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "patience": PATIENCE,
            "log_prob_threshold": LOG_PROB_THRESHOLD,
            "segment_max_duration": SEGMENT_MAX_DURATION,
            "normalize_enabled": NORMALIZE_ENABLED,
            "noise_reduce_enabled": NOISE_REDUCE_ENABLED,
            "loaded": model_manager.loaded,
            "session_count": model_manager.session_count,
        }
        if ENGINE == "qwen3asr":
            config["engine_info"] = {
                "vram_estimate": "~3.4 GB (bfloat16)",
                "context_boost": HOTWORDS_ENABLED,
                "prompt_carry": False,
            }
        return web.json_response(config)

    async def handle_engine_switch(request):
        """Switch ASR engine (requires restart to take effect)."""
        global ENGINE
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        new_engine = body.get("engine", "").lower()
        if new_engine not in ("whisper", "qwen3asr"):
            return web.json_response(
                {"error": f"Invalid engine: {new_engine}. Use 'whisper' or 'qwen3asr'"},
                status=400,
            )

        if new_engine == ENGINE:
            return web.json_response({
                "status": "no_change",
                "engine": ENGINE,
                "message": f"Already using {ENGINE}",
            })

        if model_manager.session_count > 0:
            return web.json_response(
                {"error": "Cannot switch engine while sessions are active"},
                status=409,
            )

        old_engine = ENGINE

        # Unload current model
        if model_manager.loaded:
            unloaded = await model_manager.unload_model()
            if not unloaded:
                return web.json_response(
                    {"error": "Failed to unload current model"},
                    status=500,
                )

        # Switch engine
        ENGINE = new_engine
        model_manager._engine = new_engine
        if new_engine == "whisper":
            model_manager._model_name = WHISPER_MODEL
        else:
            model_manager._model_name = "Qwen/Qwen3-ASR-1.7B"

        log.info(f"Engine switched: {old_engine} -> {new_engine}")

        return web.json_response({
            "status": "switched",
            "engine": ENGINE,
            "message": f"Switched from {old_engine} to {new_engine}. "
                       "Model will load on next connection.",
        })

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
        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_undiarized_sessions(force=force)
        )
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already diarized",
            })

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_state["running"] = True
        _spawn_bg(_batch_diarize_task(force=force), name='batch_diarize')
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

        # tmp + rename: transcript.json は source of truth なので、書き込み
        # 途中のクラッシュで truncate されると復旧不能になる
        tmp_path = str(transcript_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(transcript_path))

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

        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_sessions(force=force)
        )
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_tag_state["running"] = True
        _spawn_bg(_batch_tag_task(force=force), name='batch_tag')
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

    # --- Taxonomy reload API ---

    async def handle_taxonomy_reload(request):
        if session_tagger is not None:
            session_tagger.reload_taxonomy()
        return web.json_response({"status": "ok"})

    # --- Keyword learning API ---

    async def handle_keyword_learn(request):
        """POST /keywords/learn — extract keyword candidates from a tag correction."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        date = body.get("date", "")
        time_str = body.get("time", "")
        old_domains = body.get("old_domains", [])
        new_domains = body.get("new_domains", [])
        result = keyword_learner.learn_from_correction(date, time_str, old_domains, new_domains)
        return web.json_response(result)

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

        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_session_tag_sessions(force=force)
        )
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_session_tag_state["running"] = True
        _spawn_bg(_batch_session_tag_task(force=force), name='batch_session_tag')
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
        st_sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_session_tag_sessions(force=force) if session_tagger else []
        )
        vt_sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_untagged_sessions(force=force) if speaker_filter else []
        )

        if not st_sessions and not vt_sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already tagged",
            })

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_combined_tag_state["running"] = True
        _spawn_bg(_batch_combined_tag_task(force=force), name='batch_combined_tag')
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

    # --- VibeVoice-ASR バッチ API ---

    async def handle_vibevoice_start(request):
        if not VIBEVOICE_ENABLED:
            return web.json_response(
                {"error": "VibeVoice-ASR is disabled"}, status=503,
            )
        if _batch_vibevoice_state["running"]:
            return web.json_response(
                {"error": "already running"}, status=409,
            )

        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass

        sessions = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _find_unvibevoiced_sessions(force=force)
        )
        if not sessions and not force:
            return web.json_response({
                "status": "no_sessions",
                "message": "All sessions already transcribed by VibeVoice",
            })

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_vibevoice_state["running"] = True
        _spawn_bg(_batch_vibevoice_task(force=force), name='batch_vibevoice')
        return web.json_response({
            "status": "started",
            "total": len(sessions),
        })

    async def handle_vibevoice_status(request):
        return web.json_response(_batch_vibevoice_state)

    async def handle_vibevoice_cancel(request):
        if _batch_vibevoice_state["running"]:
            _batch_vibevoice_state["cancel_requested"] = True
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
                                await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "failed")
                                continue
                            # Responses API の出力を JSONL に書き込み
                            out_rec = {"session_id": sess_path, **rs_result.data}
                            if rs_result.data.get("risk") == "high":
                                _batch_sanitize_state["dropped"] += 1
                                _batch_sanitize_state["processed"] += 1
                                await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "dropped")
                                log.info(f"Sanitize: {sess_path} → high risk, dropped")
                                continue
                            # KC 品質ゲート: 書き込み前に採点を付与
                            await loop.run_in_executor(None, _attach_kc_quality, out_rec)
                            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                            out_f.flush()
                            _batch_sanitize_state["records_generated"] += 1
                            _batch_sanitize_state["processed"] += 1
                            await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "done")
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
                                await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "failed")
                                continue
                            if result["risk_level"] == "high":
                                _batch_sanitize_state["dropped"] += 1
                                _batch_sanitize_state["processed"] += 1
                                await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "dropped")
                                log.info(f"Sanitize: {sess_path} → high risk, dropped")
                                continue
                            # KC 品質ゲート: 書き込み前に採点を付与
                            await loop.run_in_executor(
                                None, _attach_kc_quality_chat_records, result["records"]
                            )
                            for rec in result["records"]:
                                rec["session_id"] = sess_path
                                rec["risk_level"] = result["risk_level"]
                                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                _batch_sanitize_state["records_generated"] += 1
                            out_f.flush()
                            _batch_sanitize_state["processed"] += 1
                            await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "done")
                            log.info(
                                f"Sanitize: {sess_path} → {result['risk_level']}, "
                                f"{len(result['records'])} records"
                            )
                    except Exception as e:
                        _batch_sanitize_state["failed"] += 1
                        _batch_sanitize_state["errors"].append(
                            f"{sess_path}: {e}"
                        )
                        await loop.run_in_executor(None, _record_sanitize_outcome, sess_path, "failed")
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

    # 夜間スケジューラ (モジュールレベル) から自動 sanitize を呼べるよう登録
    _sanitize_hooks["task"] = _batch_sanitize_task
    _sanitize_hooks["state"] = _batch_sanitize_state

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

        # 二重起動ガード: タスク内で running=True になるまでの間に
        # もう一度 start が呼ばれると素通りするため、ここで立てる
        _batch_sanitize_state["running"] = True
        _spawn_bg(_batch_sanitize_task(sessions), name='batch_sanitize')
        return web.json_response({
            "status": "started",
            "total": len(sessions),
        })

    async def handle_audio_scene_status(request):
        return web.json_response({
            "enabled": AUDIO_SCENE_ENABLED,
            "device": AUDIO_SCENE_DEVICE,
            "threshold": AUDIO_SCENE_THRESHOLD,
            "media_threshold": AUDIO_SCENE_MEDIA_THRESHOLD,
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

    # ============================================================
    # Succession RAG (retrieval-augmented Q&A over sanitized KCs)
    # ============================================================

    async def handle_succession_status(request):
        if succession_rag is None:
            return web.json_response({
                "enabled": False,
                "index_size": 0,
                "ready": False,
            })
        return web.json_response(succession_rag.status())

    async def handle_succession_rebuild(request):
        if succession_rag is None:
            return web.json_response(
                {"error": "SuccessionRAG disabled"}, status=503,
            )
        force = False
        try:
            body = await request.json()
            force = bool(body.get("force", False))
        except Exception:
            pass
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, succession_rag.build_index, force
            )
        except Exception as e:
            log.exception("SuccessionRAG: rebuild failed")
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response(result)

    async def handle_succession_reload_wiki(request):
        """Reload the wiki index from disk after host-side build_wiki.py runs."""
        if succession_rag is None:
            return web.json_response(
                {"error": "SuccessionRAG disabled"}, status=503,
            )
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, succession_rag.reload_wiki)
        except Exception as e:
            log.exception("SuccessionRAG: wiki reload failed")
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response(result)

    async def handle_succession_ask(request):
        if succession_rag is None:
            return web.json_response(
                {"error": "SuccessionRAG disabled"}, status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        query = (body.get("query") or "").strip()
        if not query:
            return web.json_response({"error": "query is required"}, status=400)
        k = int(body.get("k", 5))
        if k < 1:
            k = 5
        if k > 20:
            k = 20

        # Optional multi-turn conversation history.
        # Format: [{"role": "user"|"assistant", "content": str}, ...]
        # The current `query` is NOT included in history; pass it via `query`.
        # Server will validate and cap to MAX_HISTORY_TURNS.
        history = body.get("history")
        if history is not None and not isinstance(history, list):
            return web.json_response(
                {"error": "history must be a list"}, status=400,
            )

        loop = asyncio.get_running_loop()
        try:
            answer = await loop.run_in_executor(
                None,
                lambda: succession_rag.ask(query, k=k, history=history),
            )
        except Exception as e:
            log.exception("SuccessionRAG: ask failed")
            return web.json_response({"error": str(e)}, status=500)
        payload = answer.to_dict()
        # ギャップ検出: 答えられなかった質問は「次に聞き出すべきこと」
        try:
            is_gap, reason = _is_rag_gap(
                payload.get("answer", ""), payload.get("sources", []),
                payload.get("warnings", []), SUCCESSION_GAP_SIMILARITY,
            )
            if is_gap:
                top_sim = 0.0
                for s in payload.get("sources", []):
                    try:
                        top_sim = max(top_sim, float(s.get("similarity", 0) or 0))
                    except (TypeError, ValueError):
                        pass
                await loop.run_in_executor(
                    None, _append_rag_gap, query, top_sim, reason,
                )
        except Exception:
            log.debug("RAG gap detection failed", exc_info=True)
        return web.json_response(payload)

    # --- エリシテーション API ---
    async def handle_elicitation_questions(request):
        return web.json_response({"questions": elicitation_store.pending()})

    async def handle_elicitation_answer(request):
        qid = request.match_info["id"]
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        q = elicitation_store.answer(qid, str(body.get("answer", "")))
        if q is None:
            return web.json_response(
                {"error": "question not found or empty answer"}, status=404,
            )
        return web.json_response(q)

    async def handle_elicitation_skip(request):
        qid = request.match_info["id"]
        q = elicitation_store.skip(qid)
        if q is None:
            return web.json_response({"error": "question not found"}, status=404)
        return web.json_response(q)

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

    # --- RAG classifier API ---

    async def handle_rag_status(request):
        if not rag_classifier:
            return web.json_response({"ready": False, "enabled": False, "total_sessions": 0})
        status = rag_classifier.get_status()
        status["enabled"] = True
        return web.json_response(status)

    async def handle_rag_rebuild(request):
        if not rag_classifier:
            return web.json_response({"error": "RAG classifier not enabled"}, status=503)
        # Delete existing index files and rebuild
        import os as _os
        npz = os.path.join(CONFIG_DIR, "rag_embeddings.npz")
        meta = os.path.join(CONFIG_DIR, "rag_metadata.json")
        for p in (npz, meta):
            if _os.path.exists(p):
                _os.remove(p)
        # Reset in-memory index
        rag_classifier._embeddings = None
        rag_classifier._metadata = []
        _spawn_bg(_auto_build_rag_index(), name='auto_build_rag')
        return web.json_response({"status": "rebuild_started"})

    # --- Taxonomy Refiner API ---

    async def handle_taxonomy_refine_run(request):
        if not taxonomy_refiner:
            return web.json_response({"error": "Taxonomy refiner not enabled"}, status=503)
        if taxonomy_refiner._running:
            return web.json_response({"error": "Already running"}, status=409)

        async def _run_refine():
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, taxonomy_refiner.run)
            except Exception as e:
                log.error("Taxonomy refine task error: %s", e)

        _spawn_bg(_run_refine(), name='taxonomy_refine')
        return web.json_response({"status": "started"})

    async def handle_taxonomy_refine_status(request):
        if not taxonomy_refiner:
            return web.json_response({"enabled": False, "running": False, "has_pending": False})
        status = taxonomy_refiner.get_status()
        status["enabled"] = True
        status["reports"] = taxonomy_refiner.get_reports()
        return web.json_response(status)

    async def handle_taxonomy_refine_pending(request):
        if not taxonomy_refiner:
            return web.json_response({"pending": None})
        pending = taxonomy_refiner.get_pending_proposal()
        return web.json_response({"pending": pending})

    async def handle_taxonomy_refine_approve(request):
        if not taxonomy_refiner:
            return web.json_response({"error": "Taxonomy refiner not enabled"}, status=503)
        # Support selective approval via accepted_indices
        accepted_indices = None
        try:
            body = await request.json()
            if "accepted_indices" in body:
                accepted_indices = [int(i) for i in body["accepted_indices"]]
        except Exception:
            pass  # No body or invalid JSON → approve all
        result = taxonomy_refiner.approve_proposal(accepted_indices=accepted_indices)
        if "error" in result:
            return web.json_response(result, status=400)
        # Reload taxonomy in session tagger
        if session_tagger:
            session_tagger.reload_taxonomy()
        return web.json_response(result)

    async def handle_taxonomy_refine_reject(request):
        if not taxonomy_refiner:
            return web.json_response({"error": "Taxonomy refiner not enabled"}, status=503)
        result = taxonomy_refiner.reject_proposal()
        if "error" in result:
            return web.json_response(result, status=400)
        return web.json_response(result)

    # --- Nightly scheduler API ---

    async def handle_nightly_status(request):
        return web.json_response(_nightly_state)

    async def handle_nightly_trigger(request):
        if _nightly_state.get("running"):
            return web.json_response({"error": "already running"}, status=409)
        if (_batch_state.get("running")
                or _batch_combined_tag_state.get("running")
                or _batch_session_tag_state.get("running")
                or _batch_vibevoice_state.get("running")):
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

        _spawn_bg(_run_nightly_now(), name='nightly_now')
        return web.json_response({"status": "started"})

    # --- SD カード一括アップロード ---
    MAX_SDCARD_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB

    def _parse_length_prefixed(data: bytes) -> list[bytes]:
        frames = []
        offset = 0
        while offset + 4 <= len(data):
            length = int.from_bytes(data[offset:offset + 4], "little")
            offset += 4
            if length == 0 or offset + length > len(data):
                break
            frames.append(data[offset:offset + length])
            offset += length
        return frames

    def _strip_omi_header(frames: list[bytes]) -> list[bytes]:
        return [f[OMI_HEADER_SIZE:] for f in frames if len(f) > OMI_HEADER_SIZE]

    # 再送重複排除レジストリ (recordings/ 直下に永続化、日付 dir と混ざらない)
    sdcard_registry = SdcardUploadRegistry(
        Path(RECORDINGS_DIR) / ".sdcard_registry.json"
    )
    sdcard_tasks: set[asyncio.Task] = set()

    async def _process_sdcard_upload(
        key: str,
        pcm_data: bytes,
        recorded_at: datetime,
        device_recorded_at: datetime | None,
        device_id: str,
    ) -> None:
        """Transcribe one SD upload in the background (client already ACKed)."""
        try:
            model, executor = await model_manager.acquire(timeout=600)
        except Exception as e:
            log.error(f"SD background: model acquire failed: {e}")
            sdcard_registry.mark_failed(key, f"model acquire: {e}")
            return

        try:
            extra_meta = {
                "uploaded_at": datetime.now().isoformat(),
                "device_id": device_id,
            }
            if device_recorded_at is not None:
                # Device clock was rejected as implausible; keep it for reference
                extra_meta["device_timestamp"] = device_recorded_at.isoformat()
            recorder = SessionRecorder(
                recorded_at=recorded_at, source="sdcard", extra_meta=extra_meta,
            )

            # Hotwords (既存ロジック流用)
            session_hotwords = ""
            if HOTWORDS_ENABLED and hotword_loader.get_settings().get("enabled", True):
                session_hotwords = hotword_loader.get_hotwords_string()

            # 録音書き込み
            recorder.write_audio(pcm_data)

            loop = asyncio.get_running_loop()

            # 文字起こし: realtime 経路と同等の品質を出すため、silero VAD で
            # 無音境界分割 → 各 chunk に prompt_carry を引き継ぎながら transcribe。
            # 固定長 chunk だと文の途中で切れて語尾欠落・幻覚が出やすいため、
            # 発話の切れ目で分割する。(30分音声の VAD は数秒かかるので executor で)
            chunk_ranges = await loop.run_in_executor(
                None, _split_pcm_on_silence, pcm_data, SAMPLE_RATE,
            )

            prompt_carry_text = ""
            total_segments = 0

            for start_byte, end_byte in chunk_ranges:
                chunk = pcm_data[start_byte:end_byte]
                if len(chunk) == 0:
                    continue
                chunk_offset_sec = start_byte / (SAMPLE_RATE * 2)

                if ENGINE == "qwen3asr":
                    raw_segments = await loop.run_in_executor(
                        executor,
                        lambda c=chunk: model.transcribe_audio(
                            c, SAMPLE_RATE, 0.0, session_hotwords,
                            LANGUAGE, term_corrector, SEGMENT_MAX_DURATION,
                            audio_preprocessor,
                        ),
                    )
                else:
                    _carry = prompt_carry_text
                    raw_segments = await loop.run_in_executor(
                        executor,
                        lambda c=chunk, p=_carry: transcribe_audio(
                            c, model, 0.0, session_hotwords, p,
                        ),
                    )

                if raw_segments:
                    rec_segments = _offset_segments(raw_segments, chunk_offset_sec)
                    recorder.add_segments(rec_segments)
                    total_segments += len(raw_segments)
                    if ENGINE != "qwen3asr":
                        prompt_carry_text = " ".join(
                            s.get("text", "") for s in raw_segments
                        )[-200:]

            log.info(
                f"SD transcribe: {total_segments} segment(s) across "
                f"{len(chunk_ranges)} VAD-split chunk(s)"
            )

            # Finalize + batch 投入 (既存 _finalize_and_enqueue と同じ流れ)
            recorder.finalize()
            session_id = None
            if (batch_corrector and recorder.enabled
                    and hasattr(recorder, 'session_dir') and recorder.session_dir
                    and len(recorder.all_segments) > 0
                    and recorder.session_dir.exists()):
                session_id = str(recorder.session_dir.relative_to(RECORDINGS_DIR))
                batch_corrector.enqueue(session_id)

            sdcard_registry.mark_done(key, session_id, total_segments)
            log.info(f"SD processed: session={session_id}, {total_segments} segment(s)")

        except Exception as e:
            log.error(f"SD upload processing failed: {e}", exc_info=True)
            sdcard_registry.mark_failed(key, str(e))
        finally:
            model_manager.release()

    async def handle_sdcard_upload(request: web.Request) -> web.Response:
        # 認証 (既存 WebSocket と同じ ?token= クエリ)
        if AUTH_TOKEN:
            token = request.query.get("token", "")
            if not token or not hmac.compare_digest(token, AUTH_TOKEN):
                return web.json_response({"error": "unauthorized"}, status=401)

        # ヘッダ検証
        headers = request.headers
        try:
            device_id = headers["X-Device-Id"]
            upload_codec = headers["X-Codec"]
            upload_sr = int(headers["X-Sample-Rate"])
            upload_ch = int(headers.get("X-Channels", "1"))
            timestamp_start = int(headers["X-Timestamp-Start"])
            frame_format = headers.get("X-Frame-Format", "length-prefixed")
            frame_header = headers.get("X-Frame-Header", "none")
        except (KeyError, ValueError) as e:
            return web.json_response({"error": f"bad headers: {e}"}, status=400)

        if upload_codec != "opus":
            return web.json_response(
                {"error": "only codec=opus is supported"}, status=400,
            )
        if upload_sr != SAMPLE_RATE:
            return web.json_response(
                {"error": f"sample_rate must be {SAMPLE_RATE}"}, status=400,
            )
        if upload_ch != 1:
            return web.json_response({"error": "only mono supported"}, status=400)
        if frame_format != "length-prefixed":
            return web.json_response(
                {"error": "only frame_format=length-prefixed supported"}, status=400,
            )

        # Content-Length で早期拒否 (本体読み込み前)
        if request.content_length is not None and request.content_length > MAX_SDCARD_UPLOAD_BYTES:
            return web.json_response({"error": "payload too large"}, status=413)

        body = await request.read()
        if len(body) == 0:
            return web.json_response({"error": "empty body"}, status=400)
        if len(body) > MAX_SDCARD_UPLOAD_BYTES:
            return web.json_response({"error": "payload too large"}, status=413)

        # 同一ファイルの再送は即 ACK。文字起こしはバックグラウンドで進むため、
        # アプリ側の HTTP タイムアウト → 再送でも二重処理しない。
        key = sdcard_make_key(
            device_id, timestamp_start, hashlib.sha256(body).hexdigest(),
        )
        existing = sdcard_registry.lookup(key)
        if existing is not None:
            log.info(f"SD upload duplicate ({existing['state']}): {key}")
            return web.json_response({
                "status": "ok",
                "duplicate": True,
                "state": existing["state"],
                "session_id": existing.get("session_id"),
            })

        # 未処理ファイルの受け入れは同時 SDCARD_MAX_PENDING 件まで
        # (30分ファイル1件で PCM 約58MB をメモリに保持するため)。
        # 503 を返せばアプリが後で再送してくる。
        if len(sdcard_tasks) >= SDCARD_MAX_PENDING:
            return web.json_response(
                {"error": "too many pending uploads, retry later"}, status=503,
            )

        raw_frames = _parse_length_prefixed(body)
        if not raw_frames:
            return web.json_response({"error": "no valid frames"}, status=400)

        if frame_header == "3byte":
            opus_frames = _strip_omi_header(raw_frames)
        elif frame_header == "none":
            opus_frames = raw_frames
        else:
            return web.json_response(
                {"error": f"unknown frame_header: {frame_header}"}, status=400,
            )

        try:
            decoder = OmiOpusStreamDecoder(sample_rate=upload_sr, channels=upload_ch)
            pcm_data = decoder.decode_batch(opus_frames)
        except Exception as e:
            log.error(f"OPUS decode failed: {e}", exc_info=True)
            return web.json_response({"error": "decode failed"}, status=500)

        duration = len(pcm_data) / (upload_sr * upload_ch * 2)

        # デバイス時計の妥当性チェック (大きく狂っていたら受信時刻で保存)
        recorded_at, device_recorded_at = resolve_recorded_at(
            timestamp_start, SDCARD_TS_MAX_AGE_DAYS or 36500,
        )
        session_id_hint = (
            recorded_at.strftime("%Y-%m-%d/%H-%M-%S") if SAVE_AUDIO else None
        )
        log.info(
            f"SD upload from {device_id}: {len(opus_frames)} frames, "
            f"{duration:.1f}s, codec={upload_codec}"
            + (f" (device clock {device_recorded_at} rejected, using now)"
               if device_recorded_at else "")
        )

        sdcard_registry.mark_processing(
            key, device_id=device_id,
            duration_seconds=round(duration, 2),
            session_id=session_id_hint,
        )
        task = asyncio.create_task(_process_sdcard_upload(
            key, pcm_data, recorded_at, device_recorded_at, device_id,
        ))
        sdcard_tasks.add(task)
        task.add_done_callback(sdcard_tasks.discard)

        return web.json_response({
            "status": "ok",
            "queued": True,
            "session_id": session_id_hint,
            "duration_seconds": round(duration, 2),
            "segments_count": 0,
            "enqueued_for_batch": False,
        })

    mgmt_middlewares = []
    if MANAGEMENT_TOKEN:
        mgmt_middlewares.append(make_mgmt_auth_middleware(MANAGEMENT_TOKEN))
        log.info("Management API: token auth enabled")
    mgmt_app = web.Application(
        client_max_size=MAX_SDCARD_UPLOAD_BYTES, middlewares=mgmt_middlewares
    )
    mgmt_app.router.add_post("/model/load", handle_model_load)
    mgmt_app.router.add_post("/model/unload", handle_model_unload)
    mgmt_app.router.add_get("/model/status", handle_model_status)
    mgmt_app.router.add_get("/whisper/config", handle_whisper_config)
    mgmt_app.router.add_post("/engine/switch", handle_engine_switch)
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
    # Taxonomy reload
    mgmt_app.router.add_post("/taxonomy/reload", handle_taxonomy_reload)
    # Keyword learning
    mgmt_app.router.add_post("/keywords/learn", handle_keyword_learn)
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
    # VibeVoice-ASR batch API
    mgmt_app.router.add_post("/vibevoice/start", handle_vibevoice_start)
    mgmt_app.router.add_get("/vibevoice/status", handle_vibevoice_status)
    mgmt_app.router.add_post("/vibevoice/cancel", handle_vibevoice_cancel)
    # Sanitize batch API (training data anonymization)
    mgmt_app.router.add_post("/sanitize/start", handle_sanitize_start)
    mgmt_app.router.add_get("/sanitize/status", handle_sanitize_status)
    mgmt_app.router.add_post("/sanitize/cancel", handle_sanitize_cancel)
    mgmt_app.router.add_get("/sanitize/download", handle_sanitize_download)
    # Succession RAG API (retrieval-augmented Q&A)
    mgmt_app.router.add_get("/succession/status", handle_succession_status)
    mgmt_app.router.add_post("/succession/rebuild", handle_succession_rebuild)
    mgmt_app.router.add_post("/succession/reload_wiki", handle_succession_reload_wiki)
    mgmt_app.router.add_post("/succession/ask", handle_succession_ask)
    # エリシテーション API
    mgmt_app.router.add_get("/elicitation/questions", handle_elicitation_questions)
    mgmt_app.router.add_post("/elicitation/questions/{id}/answer", handle_elicitation_answer)
    mgmt_app.router.add_post("/elicitation/questions/{id}/skip", handle_elicitation_skip)
    # Batch LLM correction API
    mgmt_app.router.add_get("/batch/status", handle_batch_llm_status)
    mgmt_app.router.add_get("/batch/jobs", handle_batch_llm_jobs)
    mgmt_app.router.add_post("/batch/retry", handle_batch_llm_retry)
    mgmt_app.router.add_post("/batch/process", handle_batch_llm_process)
    # RAG classifier API
    mgmt_app.router.add_get("/rag/status", handle_rag_status)
    mgmt_app.router.add_post("/rag/rebuild", handle_rag_rebuild)
    # Taxonomy Refiner API
    mgmt_app.router.add_post("/taxonomy/refine/run", handle_taxonomy_refine_run)
    mgmt_app.router.add_get("/taxonomy/refine/status", handle_taxonomy_refine_status)
    mgmt_app.router.add_get("/taxonomy/refine/pending", handle_taxonomy_refine_pending)
    mgmt_app.router.add_post("/taxonomy/refine/approve", handle_taxonomy_refine_approve)
    mgmt_app.router.add_post("/taxonomy/refine/reject", handle_taxonomy_refine_reject)
    # Nightly scheduler API
    mgmt_app.router.add_get("/nightly/status", handle_nightly_status)
    mgmt_app.router.add_post("/nightly/trigger", handle_nightly_trigger)
    # SD card batch upload (OPUS → 既存 SessionRecorder パイプライン合流)
    mgmt_app.router.add_post("/sdcard/upload", handle_sdcard_upload)

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
            _spawn_bg(_nightly_scheduler(), name='nightly_scheduler')
        else:
            log.info("Nightly scheduler: DISABLED (NIGHTLY_SCHEDULE_HOUR=0)")

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
