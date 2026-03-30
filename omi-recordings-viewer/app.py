"""Omi Recordings Viewer — aiohttp web server."""

import asyncio
import json
import logging
import os


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
_PENDING_SUGGESTIONS_PATH = Path(CONFIG_DIR) / "pending_suggestions.json"
_AUTO_DETECTED_PATH = Path(CONFIG_DIR) / "auto_detected_terms.json"
_HOTWORDS_CUSTOM_PATH = Path(CONFIG_DIR) / "hotwords_custom.json"
_USER_TAGS_PATH = Path(CONFIG_DIR) / "user_tags.json"

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
    import time
    return aiohttp_jinja2.render_template("index.html", request, {"v": int(time.time())})


async def handle_api_sessions(request: web.Request) -> web.Response:
    indexer: Indexer = request.app["indexer"]
    page = int(request.query.get("page", "1"))
    per_page = int(request.query.get("per_page", "20"))
    query = request.query.get("q", "").strip()

    page = max(1, page)
    per_page = max(1, min(100, per_page))

    activity = request.query.get("activity", "").strip()
    mood = request.query.get("mood", "").strip()
    user_tag = request.query.get("user_tag", "").strip()

    try:
        result = await indexer.get_sessions(page, per_page, query, activity=activity, mood=mood, user_tag=user_tag)
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

    # hotwords custom にも正解語を追加 (デコーダバイアスで認識改善)
    custom_hw = _read_json(_HOTWORDS_CUSTOM_PATH, [])
    if target["correct"] not in custom_hw:
        custom_hw.append(target["correct"])
        _write_json(_HOTWORDS_CUSTOM_PATH, custom_hw)

    # auto_detected に記録
    auto_detected = _read_json(_AUTO_DETECTED_PATH, [])
    if target["wrong"] not in auto_detected:
        auto_detected.append(target["wrong"])
        _write_json(_AUTO_DETECTED_PATH, auto_detected)

    # 提案を削除
    suggestions = [s for s in suggestions if s["id"] != sid]
    _write_json(_PENDING_SUGGESTIONS_PATH, suggestions)

    logger.info(f"Suggestion approved: {target['wrong']} → {target['correct']} (also added to hotwords)")
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
# ユーザータグ CRUD API
# ============================================================

def _generate_tag_id() -> str:
    """Generate a short unique tag ID."""
    import uuid
    return "ut_" + uuid.uuid4().hex[:6]


async def handle_get_user_tags(request: web.Request) -> web.Response:
    data = _read_json(_USER_TAGS_PATH, [])
    return web.json_response(data)


async def handle_post_user_tag(request: web.Request) -> web.Response:
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    color = str(body.get("color", "#6c9bff")).strip()
    description = str(body.get("description", "")).strip()

    tags = _read_json(_USER_TAGS_PATH, [])
    new_tag = {
        "id": _generate_tag_id(),
        "name": name,
        "color": color,
        "description": description,
    }
    tags.append(new_tag)
    _write_json(_USER_TAGS_PATH, tags)

    # Notify Bridge to reload user tags
    await _notify_bridge_user_tags_reload(request)

    return web.json_response(new_tag, status=201)


async def handle_put_user_tag(request: web.Request) -> web.Response:
    tag_id = request.match_info["id"]
    body = await request.json()

    tags = _read_json(_USER_TAGS_PATH, [])
    target = None
    for t in tags:
        if t.get("id") == tag_id:
            target = t
            break
    if not target:
        return web.json_response({"error": "not found"}, status=404)

    if "name" in body:
        target["name"] = str(body["name"]).strip()
    if "color" in body:
        target["color"] = str(body["color"]).strip()
    if "description" in body:
        target["description"] = str(body["description"]).strip()

    _write_json(_USER_TAGS_PATH, tags)
    await _notify_bridge_user_tags_reload(request)
    return web.json_response({"status": "ok"})


async def handle_delete_user_tag(request: web.Request) -> web.Response:
    tag_id = request.match_info["id"]
    tags = _read_json(_USER_TAGS_PATH, [])
    new_tags = [t for t in tags if t.get("id") != tag_id]
    if len(new_tags) == len(tags):
        return web.json_response({"error": "not found"}, status=404)
    _write_json(_USER_TAGS_PATH, new_tags)
    await _notify_bridge_user_tags_reload(request)
    return web.json_response({"status": "ok"})


async def _notify_bridge_user_tags_reload(request: web.Request) -> None:
    """Notify Bridge to reload user tag definitions."""
    if not BRIDGE_MGMT_URL:
        return
    session: aiohttp.ClientSession = request.app["bridge_session"]
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        await session.post(f"{BRIDGE_MGMT_URL}/user-tags/reload", timeout=timeout)
    except Exception as e:
        logger.debug(f"Bridge user-tags reload notification failed: {e}")


# ============================================================
# セッションユーザータグ更新 API
# ============================================================

async def handle_put_session_user_tags(request: web.Request) -> web.Response:
    """Update user_tags on a session's transcript.json and re-index."""
    date = request.match_info["date"]
    time = request.match_info["time"]
    body = await request.json()
    user_tags = body.get("user_tags", [])
    if not isinstance(user_tags, list):
        return web.json_response({"error": "user_tags must be a list"}, status=400)

    transcript_path = Path(RECORDINGS_DIR) / date / time / "transcript.json"
    if not transcript_path.exists():
        return web.json_response({"error": "session not found"}, status=404)

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return web.json_response({"error": "transcript read error"}, status=500)

    session = data.setdefault("session", {})
    tags = session.setdefault("tags", {})
    tags["user_tags"] = user_tags
    transcript_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Trigger re-index
    indexer: Indexer = request.app["indexer"]
    try:
        info = indexer._parse_transcript(date, time)
        info["last_modified"] = transcript_path.stat().st_mtime
        await indexer._db.execute(
            """INSERT OR REPLACE INTO sessions
               (date, time, start_at, end_at, duration_seconds,
                audio_size_mb, model, language, transcript_text, segment_count,
                tags_topics, tags_activity, tags_intent, tags_mood,
                tags_quality, tags_entities, tags_summary, session_tagged,
                tags_user, last_modified)
               VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                       :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                       :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                       :tags_quality, :tags_entities, :tags_summary, :session_tagged,
                       :tags_user, :last_modified)""",
            info,
        )
        await indexer._db.commit()
    except Exception as e:
        logger.warning(f"Re-index after user-tags update failed: {e}")

    return web.json_response({"status": "ok", "user_tags": user_tags})


# ============================================================
# Diarization プロキシ API (Bridge へ中継)
# ============================================================

async def _proxy_bridge(
    request: web.Request, method: str, path: str, timeout_sec: float = 5,
) -> web.Response:
    """Bridge 管理 API へリクエストをプロキシする共通ロジック。"""
    if not BRIDGE_MGMT_URL:
        return web.json_response({"error": "bridge not configured"}, status=503)

    session: aiohttp.ClientSession = request.app["bridge_session"]
    url = f"{BRIDGE_MGMT_URL}{path}"
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_sec)
        if method == "GET":
            resp = await session.get(url, timeout=timeout)
        else:
            # POST/PUT/DELETE ボディを転送
            req_body = None
            req_headers = {}
            try:
                req_body = await request.read()
                if req_body:
                    req_headers["Content-Type"] = request.content_type or "application/json"
            except Exception:
                pass
            if method == "PUT":
                resp = await session.put(url, data=req_body, headers=req_headers, timeout=timeout)
            elif method == "DELETE":
                resp = await session.delete(url, data=req_body, headers=req_headers, timeout=timeout)
            else:
                resp = await session.post(url, data=req_body, headers=req_headers, timeout=timeout)
        body = await resp.read()
        return web.Response(
            body=body,
            status=resp.status,
            content_type=resp.content_type or "application/json",
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Bridge proxy error ({path}): {e}")
        return web.json_response({"error": "bridge unavailable"}, status=502)
    except Exception as e:
        logger.error(f"Bridge proxy unexpected error ({path}): {e}", exc_info=True)
        return web.json_response({"error": str(e)}, status=502)


async def handle_diarize_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/diarize/start")


async def handle_diarize_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/diarize/status")


async def handle_diarize_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/diarize/cancel")


# ============================================================
# Hotwords プロキシ API (Bridge へ中継)
# ============================================================

async def handle_hotwords_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/hotwords/status")


async def handle_hotwords_settings_get(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/hotwords/settings")


async def handle_hotwords_settings_put(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "PUT", "/hotwords/settings")


async def handle_hotwords_custom_post(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/hotwords/custom")


async def handle_hotwords_custom_delete(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "DELETE", "/hotwords/custom")


# ============================================================
# Speaker プロキシ API (Bridge へ中継)
# ============================================================

async def handle_speakers_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/speakers/status")


async def handle_speakers_list(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/speakers/list")


async def handle_speakers_add(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/speakers/add")


async def handle_speakers_update(request: web.Request) -> web.Response:
    speaker_id = request.match_info["id"]
    return await _proxy_bridge(request, "PUT", f"/speakers/{speaker_id}")


async def handle_speakers_delete(request: web.Request) -> web.Response:
    speaker_id = request.match_info["id"]
    return await _proxy_bridge(request, "DELETE", f"/speakers/{speaker_id}")


async def handle_speakers_add_voiceprint(request: web.Request) -> web.Response:
    speaker_id = request.match_info["id"]
    # Voiceprint extraction involves GPU lock + model load/unload — needs longer timeout
    return await _proxy_bridge(request, "POST", f"/speakers/{speaker_id}/voiceprint", timeout_sec=120)


async def handle_speakers_remove_voiceprint(request: web.Request) -> web.Response:
    speaker_id = request.match_info["id"]
    sample_id = request.match_info["sample_id"]
    return await _proxy_bridge(request, "DELETE", f"/speakers/{speaker_id}/voiceprint/{sample_id}")


async def handle_speakers_threshold(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "PUT", "/speakers/threshold")


async def handle_speakers_tag_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/speakers/tag/start")


async def handle_speakers_tag_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/speakers/tag/status")


async def handle_speakers_tag_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/speakers/tag/cancel")


async def handle_speakers_delete_all(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "DELETE", "/speakers/all")


# ============================================================
# Session Tagging プロキシ API (Bridge へ中継)
# ============================================================

async def handle_session_tag_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/session-tags/start")


async def handle_session_tag_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/session-tags/status")


async def handle_session_tag_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/session-tags/cancel")


# ============================================================
# Combined Tag プロキシ API (Session Tag → Voice Tag)
# ============================================================

async def handle_combined_tag_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/tags/start")


async def handle_combined_tag_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/tags/status")


async def handle_combined_tag_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/tags/cancel")


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


    # User tags CRUD API
    app.router.add_get("/api/user-tags", handle_get_user_tags)
    app.router.add_post("/api/user-tags", handle_post_user_tag)
    app.router.add_put("/api/user-tags/{id}", handle_put_user_tag)
    app.router.add_delete("/api/user-tags/{id}", handle_delete_user_tag)

    # Session user-tags update
    app.router.add_put("/api/sessions/{date}/{time}/user-tags", handle_put_session_user_tags)

    # Diarization プロキシ API
    app.router.add_post("/api/diarize/start", handle_diarize_start)
    app.router.add_get("/api/diarize/status", handle_diarize_status)
    app.router.add_post("/api/diarize/cancel", handle_diarize_cancel)

    # Hotwords プロキシ API
    app.router.add_get("/api/hotwords/status", handle_hotwords_status)
    app.router.add_get("/api/hotwords/settings", handle_hotwords_settings_get)
    app.router.add_put("/api/hotwords/settings", handle_hotwords_settings_put)
    app.router.add_post("/api/hotwords/custom", handle_hotwords_custom_post)
    app.router.add_delete("/api/hotwords/custom", handle_hotwords_custom_delete)

    # Speaker プロキシ API (固定パスを {id} より先に登録)
    app.router.add_get("/api/speakers/status", handle_speakers_status)
    app.router.add_get("/api/speakers/list", handle_speakers_list)
    app.router.add_post("/api/speakers/add", handle_speakers_add)
    app.router.add_put("/api/speakers/threshold", handle_speakers_threshold)
    app.router.add_post("/api/speakers/tag/start", handle_speakers_tag_start)
    app.router.add_get("/api/speakers/tag/status", handle_speakers_tag_status)
    app.router.add_post("/api/speakers/tag/cancel", handle_speakers_tag_cancel)
    app.router.add_delete("/api/speakers/all", handle_speakers_delete_all)
    app.router.add_put("/api/speakers/{id}", handle_speakers_update)
    app.router.add_delete("/api/speakers/{id}", handle_speakers_delete)
    app.router.add_post("/api/speakers/{id}/voiceprint", handle_speakers_add_voiceprint)
    app.router.add_delete("/api/speakers/{id}/voiceprint/{sample_id}", handle_speakers_remove_voiceprint)

    # Session tagging プロキシ API
    app.router.add_post("/api/session-tags/start", handle_session_tag_start)
    app.router.add_get("/api/session-tags/status", handle_session_tag_status)
    app.router.add_post("/api/session-tags/cancel", handle_session_tag_cancel)

    # Combined tag プロキシ API (Session Tag → Voice Tag)
    app.router.add_post("/api/tags/start", handle_combined_tag_start)
    app.router.add_get("/api/tags/status", handle_combined_tag_status)
    app.router.add_post("/api/tags/cancel", handle_combined_tag_cancel)

    app.router.add_static("/static/", path=str(base_dir / "static"), name="static")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
