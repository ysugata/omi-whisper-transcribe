"""Omi Recordings Viewer — aiohttp web server."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiohttp
import aiohttp_jinja2
import jinja2
from aiohttp import web

from indexer import Indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "/app/recordings")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/app/config")
DB_PATH = os.environ.get("DB_PATH", "/app/data/index.db")
PORT = int(os.environ.get("PORT", "8092"))
BRIDGE_MGMT_URL = os.environ.get("BRIDGE_MGMT_URL", "")

_TERM_CORRECTIONS_PATH = Path(CONFIG_DIR) / "term_corrections.json"
_PROMPT_SETTINGS_PATH = Path(CONFIG_DIR) / "prompt_settings.json"
_PENDING_SUGGESTIONS_PATH = Path(CONFIG_DIR) / "pending_suggestions.json"
_PROMPT_PRESETS_PATH = Path(CONFIG_DIR) / "prompt_presets.json"
_AUTO_DETECTED_PATH = Path(CONFIG_DIR) / "auto_detected_terms.json"

# 初期辞書 (term_corrector.py と同じもの、viewer 単体でも初期生成できるように)
_INITIAL_DICT = {
    "クーバネティス": "Kubernetes", "クバネティス": "Kubernetes",
    "クーバネテス": "Kubernetes", "クーバーネティス": "Kubernetes",
    "クーベネティス": "Kubernetes", "クーバネテイス": "Kubernetes",
    "ドッカー": "Docker", "ドッカ": "Docker", "ヘルム": "Helm",
    "テラフォーム": "Terraform", "テラホーム": "Terraform",
    "シーアイシーディー": "CI/CD",
    "ギットハブ": "GitHub", "ギットハブアクションズ": "GitHub Actions",
    "ギットラブ": "GitLab", "ギットラボ": "GitLab",
    "アルゴシーディー": "ArgoCD", "ジェンキンス": "Jenkins",
    "アンシブル": "Ansible",
    "プロメテウス": "Prometheus", "プロメシウス": "Prometheus",
    "グラファーナ": "Grafana", "グラファナ": "Grafana",
    "データドッグ": "Datadog", "データドック": "Datadog",
    "リアクト": "React", "ネクストジェーエス": "Next.js",
    "タイプスクリプト": "TypeScript", "ジャバスクリプト": "JavaScript",
    "パイソン": "Python", "ファストエーピーアイ": "FastAPI",
    "ジャンゴ": "Django", "フラスコ": "Flask",
    "ジーアールピーシー": "gRPC", "エンボイ": "Envoy",
    "イスティオ": "Istio", "ウェブソケット": "WebSocket",
    "ニンクス": "Nginx", "エヌジンクス": "Nginx",
    "エンジンエックス": "Nginx",
    "レディス": "Redis", "レデス": "Redis",
    "ポストグレス": "PostgreSQL", "ポストグレ": "PostgreSQL",
    "マイエスキューエル": "MySQL",
    "エーダブリューエス": "AWS", "ジーシーピー": "GCP",
    "アジュール": "Azure",
    "ウィスパー": "Whisper", "ウイスパー": "Whisper",
}


def _read_json(path: Path, default):
    """JSON ファイルを読み込む。存在しない・壊れている場合はデフォルトを返す。"""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"JSON read error ({path}): {e}")
    return default


def _write_json(path: Path, data) -> None:
    """JSON ファイルに書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 既存ハンドラ
# ============================================================

async def handle_index(request: web.Request) -> web.Response:
    return aiohttp_jinja2.render_template("index.html", request, {})


async def handle_api_sessions(request: web.Request) -> web.Response:
    indexer: Indexer = request.app["indexer"]
    page = int(request.query.get("page", "1"))
    per_page = int(request.query.get("per_page", "20"))
    query = request.query.get("q", "").strip()

    page = max(1, page)
    per_page = max(1, min(100, per_page))

    try:
        result = await indexer.get_sessions(page, per_page, query)
    except Exception as e:
        logger.exception("Failed to query sessions")
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response(result)


async def handle_api_session_detail(request: web.Request) -> web.Response:
    indexer: Indexer = request.app["indexer"]
    date = request.match_info["date"]
    time = request.match_info["time"]

    session = await indexer.get_session(date, time)
    if not session:
        return web.json_response({"error": "not found"}, status=404)

    transcript = await indexer.get_transcript(date, time)
    return web.json_response({"session": session, "transcript": transcript})


_AUDIO_MIME = {".ogg": "audio/ogg", ".wav": "audio/wav"}


async def handle_audio(request: web.Request) -> web.FileResponse:
    date = request.match_info["date"]
    time = request.match_info["time"]
    filename = request.match_info["filename"]

    session_dir = Path(RECORDINGS_DIR) / date / time
    audio_path = session_dir / filename
    if not audio_path.exists():
        raise web.HTTPNotFound(text="audio not found")

    mime = _AUDIO_MIME.get(audio_path.suffix, "application/octet-stream")
    return web.FileResponse(audio_path, headers={"Content-Type": mime})


# ============================================================
# 設定 API: プロンプト設定
# ============================================================

async def handle_get_prompt_settings(request: web.Request) -> web.Response:
    default = {"topic": "", "participants": [], "terms": []}
    data = _read_json(_PROMPT_SETTINGS_PATH, default)
    return web.json_response(data)


async def handle_put_prompt_settings(request: web.Request) -> web.Response:
    body = await request.json()
    settings = {
        "topic": str(body.get("topic", "")),
        "participants": list(body.get("participants", [])),
        "terms": list(body.get("terms", [])),
    }
    _write_json(_PROMPT_SETTINGS_PATH, settings)
    logger.info(f"Prompt settings updated: topic={settings['topic'][:50]}")
    return web.json_response({"status": "ok"})


# ============================================================
# 設定 API: 用語補正辞書
# ============================================================

async def handle_get_terms(request: web.Request) -> web.Response:
    data = _read_json(_TERM_CORRECTIONS_PATH, None)
    if data is None:
        data = dict(_INITIAL_DICT)
        _write_json(_TERM_CORRECTIONS_PATH, data)
    return web.json_response(data)


async def handle_post_term(request: web.Request) -> web.Response:
    body = await request.json()
    wrong = str(body.get("wrong", "")).strip()
    correct = str(body.get("correct", "")).strip()
    if not wrong or not correct:
        return web.json_response({"error": "wrong and correct are required"}, status=400)

    data = _read_json(_TERM_CORRECTIONS_PATH, dict(_INITIAL_DICT))
    data[wrong] = correct
    _write_json(_TERM_CORRECTIONS_PATH, data)
    return web.json_response({"status": "ok", "count": len(data)})


async def handle_delete_term(request: web.Request) -> web.Response:
    body = await request.json()
    wrong = str(body.get("wrong", "")).strip()
    if not wrong:
        return web.json_response({"error": "wrong is required"}, status=400)

    data = _read_json(_TERM_CORRECTIONS_PATH, dict(_INITIAL_DICT))
    data.pop(wrong, None)
    _write_json(_TERM_CORRECTIONS_PATH, data)
    return web.json_response({"status": "ok", "count": len(data)})


# ============================================================
# 設定 API: 用語補正辞書 (追加エンドポイント)
# ============================================================

async def handle_update_term(request: web.Request) -> web.Response:
    wrong = request.match_info["wrong"]
    body = await request.json()
    new_wrong = str(body.get("wrong", "")).strip()
    new_correct = str(body.get("correct", "")).strip()
    if not new_wrong or not new_correct:
        return web.json_response({"error": "wrong and correct are required"}, status=400)

    data = _read_json(_TERM_CORRECTIONS_PATH, dict(_INITIAL_DICT))
    # 旧キー削除 + 新キー追加
    data.pop(wrong, None)
    data[new_wrong] = new_correct
    _write_json(_TERM_CORRECTIONS_PATH, data)
    return web.json_response({"status": "ok", "count": len(data)})


async def handle_get_auto_detected(request: web.Request) -> web.Response:
    data = _read_json(_AUTO_DETECTED_PATH, [])
    return web.json_response(data)


# ============================================================
# 設定 API: LLM 提案
# ============================================================

async def handle_get_suggestions(request: web.Request) -> web.Response:
    data = _read_json(_PENDING_SUGGESTIONS_PATH, [])
    pending = [s for s in data if s.get("status") == "pending"]
    return web.json_response(pending)


async def handle_approve_suggestion(request: web.Request) -> web.Response:
    sid = request.match_info["id"]
    suggestions = _read_json(_PENDING_SUGGESTIONS_PATH, [])

    target = None
    for s in suggestions:
        if s["id"] == sid:
            target = s
            break
    if not target:
        return web.json_response({"error": "not found"}, status=404)

    # 辞書に追加
    terms = _read_json(_TERM_CORRECTIONS_PATH, dict(_INITIAL_DICT))
    terms[target["wrong"]] = target["correct"]
    _write_json(_TERM_CORRECTIONS_PATH, terms)

    # auto_detected に記録
    auto_detected = _read_json(_AUTO_DETECTED_PATH, [])
    if target["wrong"] not in auto_detected:
        auto_detected.append(target["wrong"])
        _write_json(_AUTO_DETECTED_PATH, auto_detected)

    # 提案を削除
    suggestions = [s for s in suggestions if s["id"] != sid]
    _write_json(_PENDING_SUGGESTIONS_PATH, suggestions)

    logger.info(f"Suggestion approved: {target['wrong']} → {target['correct']}")
    return web.json_response({"status": "ok"})


async def handle_reject_suggestion(request: web.Request) -> web.Response:
    sid = request.match_info["id"]
    suggestions = _read_json(_PENDING_SUGGESTIONS_PATH, [])
    suggestions = [s for s in suggestions if s["id"] != sid]
    _write_json(_PENDING_SUGGESTIONS_PATH, suggestions)
    return web.json_response({"status": "ok"})


async def handle_update_suggestion(request: web.Request) -> web.Response:
    sid = request.match_info["id"]
    body = await request.json()
    new_correct = str(body.get("correct", "")).strip()
    if not new_correct:
        return web.json_response({"error": "correct is required"}, status=400)

    suggestions = _read_json(_PENDING_SUGGESTIONS_PATH, [])
    found = False
    for s in suggestions:
        if s["id"] == sid:
            s["correct"] = new_correct
            found = True
            break
    if not found:
        return web.json_response({"error": "not found"}, status=404)

    _write_json(_PENDING_SUGGESTIONS_PATH, suggestions)
    return web.json_response({"status": "ok"})


# ============================================================
# 設定 API: プリセット
# ============================================================

async def handle_get_presets(request: web.Request) -> web.Response:
    data = _read_json(_PROMPT_PRESETS_PATH, [])
    return web.json_response(data)


async def handle_post_preset(request: web.Request) -> web.Response:
    body = await request.json()
    now = datetime.now().isoformat()
    preset = {
        "id": str(uuid.uuid4())[:8],
        "name": str(body.get("name", "")).strip() or "Untitled",
        "topic": str(body.get("topic", "")),
        "participants": list(body.get("participants", [])),
        "terms": list(body.get("terms", [])),
        "created_at": now,
        "updated_at": now,
    }

    presets = _read_json(_PROMPT_PRESETS_PATH, [])
    presets.append(preset)
    _write_json(_PROMPT_PRESETS_PATH, presets)
    logger.info(f"Preset created: {preset['name']} ({preset['id']})")
    return web.json_response(preset, status=201)


async def handle_put_preset(request: web.Request) -> web.Response:
    pid = request.match_info["id"]
    body = await request.json()

    presets = _read_json(_PROMPT_PRESETS_PATH, [])
    found = False
    for p in presets:
        if p["id"] == pid:
            if "name" in body:
                p["name"] = str(body["name"]).strip() or p["name"]
            if "topic" in body:
                p["topic"] = str(body["topic"])
            if "participants" in body:
                p["participants"] = list(body["participants"])
            if "terms" in body:
                p["terms"] = list(body["terms"])
            p["updated_at"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        return web.json_response({"error": "not found"}, status=404)

    _write_json(_PROMPT_PRESETS_PATH, presets)
    return web.json_response({"status": "ok"})


async def handle_delete_preset(request: web.Request) -> web.Response:
    pid = request.match_info["id"]
    presets = _read_json(_PROMPT_PRESETS_PATH, [])
    presets = [p for p in presets if p["id"] != pid]
    _write_json(_PROMPT_PRESETS_PATH, presets)

    # current_prompt をクリア (プリセット削除時)
    settings = _read_json(_PROMPT_SETTINGS_PATH, {"topic": "", "participants": [], "terms": []})
    _write_json(_PROMPT_SETTINGS_PATH, settings)

    return web.json_response({"status": "ok"})


# ============================================================
# Diarization プロキシ API (Bridge へ中継)
# ============================================================

async def _proxy_bridge(request: web.Request, method: str, path: str) -> web.Response:
    """Bridge 管理 API へリクエストをプロキシする共通ロジック。"""
    if not BRIDGE_MGMT_URL:
        return web.json_response({"error": "bridge not configured"}, status=503)

    session: aiohttp.ClientSession = request.app["bridge_session"]
    url = f"{BRIDGE_MGMT_URL}{path}"
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        if method == "GET":
            resp = await session.get(url, timeout=timeout)
        else:
            resp = await session.post(url, timeout=timeout)
        body = await resp.read()
        return web.Response(
            body=body,
            status=resp.status,
            content_type=resp.content_type or "application/json",
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Bridge proxy error ({path}): {e}")
        return web.json_response({"error": "bridge unavailable"}, status=502)


async def handle_diarize_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/diarize/start")


async def handle_diarize_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/diarize/status")


async def handle_diarize_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/diarize/cancel")


# ============================================================
# アプリケーション構築
# ============================================================

async def on_startup(app: web.Application):
    indexer = Indexer(DB_PATH, RECORDINGS_DIR)
    await indexer.open()
    n = await indexer.sync()
    logger.info("Initial sync: indexed %d new sessions", n)

    app["indexer"] = indexer
    app["sync_task"] = asyncio.create_task(indexer.periodic_sync())
    app["bridge_session"] = aiohttp.ClientSession()


async def on_cleanup(app: web.Application):
    app["sync_task"].cancel()
    await app["indexer"].close()
    await app["bridge_session"].close()


def create_app() -> web.Application:
    app = web.Application()

    base_dir = Path(__file__).parent
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(base_dir / "templates")))

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/sessions", handle_api_sessions)
    app.router.add_get("/api/sessions/{date}/{time}", handle_api_session_detail)
    app.router.add_get("/audio/{date}/{time}/{filename}", handle_audio)

    # 設定 API
    app.router.add_get("/api/prompt-settings", handle_get_prompt_settings)
    app.router.add_put("/api/prompt-settings", handle_put_prompt_settings)
    app.router.add_get("/api/terms", handle_get_terms)
    app.router.add_post("/api/terms", handle_post_term)
    app.router.add_delete("/api/terms", handle_delete_term)
    # auto-detected は {wrong} より先に登録
    app.router.add_get("/api/terms/auto-detected", handle_get_auto_detected)
    app.router.add_put("/api/terms/{wrong}", handle_update_term)

    # 提案 API
    app.router.add_get("/api/suggestions", handle_get_suggestions)
    app.router.add_post("/api/suggestions/{id}/approve", handle_approve_suggestion)
    app.router.add_post("/api/suggestions/{id}/reject", handle_reject_suggestion)
    app.router.add_put("/api/suggestions/{id}", handle_update_suggestion)

    # プリセット API
    app.router.add_get("/api/presets", handle_get_presets)
    app.router.add_post("/api/presets", handle_post_preset)
    app.router.add_put("/api/presets/{id}", handle_put_preset)
    app.router.add_delete("/api/presets/{id}", handle_delete_preset)

    # Diarization プロキシ API
    app.router.add_post("/api/diarize/start", handle_diarize_start)
    app.router.add_get("/api/diarize/status", handle_diarize_status)
    app.router.add_post("/api/diarize/cancel", handle_diarize_cancel)

    app.router.add_static("/static/", path=str(base_dir / "static"), name="static")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
