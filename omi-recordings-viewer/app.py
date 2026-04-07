"""Omi Recordings Viewer — aiohttp web server."""

import asyncio
import json
import logging
import os
from datetime import datetime

from pathlib import Path

import aiohttp
import aiohttp_jinja2
import jinja2
from aiohttp import web

from exporter import export_session
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
_TAG_TAXONOMY_PATH = Path(CONFIG_DIR) / "tag_taxonomy.json"

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
    business_domain = request.query.get("business_domain", "").strip()
    importance = request.query.get("importance", "").strip()
    source_type = request.query.get("source_type", "").strip()
    speaker = request.query.get("speaker", "").strip()

    # V2 (3-axis) filters
    conversation_type = request.query.get("conversation_type", "").strip()
    outcome = request.query.get("outcome", "").strip()
    domain = request.query.get("domain", "").strip()
    importance_v2 = request.query.get("importance_v2", "").strip()
    audio_scene = request.query.get("audio_scene", "").strip()
    audio_speaker_count = request.query.get("audio_speaker_count", "").strip()

    try:
        result = await indexer.get_sessions(
            page, per_page, query, activity=activity, mood=mood,
            speaker=speaker,
            business_domain=business_domain, importance=importance,
            source_type=source_type,
            conversation_type=conversation_type, outcome=outcome,
            domain=domain, importance_v2=importance_v2,
            audio_scene=audio_scene,
            audio_speaker_count=audio_speaker_count,
        )
    except Exception as e:
        logger.exception("Failed to query sessions")
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response(result)


async def handle_api_speaker_names(request: web.Request) -> web.Response:
    indexer: Indexer = request.app["indexer"]
    try:
        names = await indexer.get_speaker_names()
    except Exception as e:
        logger.exception("Failed to get speaker names")
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(names)


async def handle_audio_scene_labels(request: web.Request) -> web.Response:
    """Get distinct audio scene labels for filter dropdown."""
    indexer: Indexer = request.app["indexer"]
    try:
        labels = await indexer.get_audio_scene_labels()
    except Exception as e:
        logger.exception("Failed to get audio scene labels")
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(labels)


async def handle_audio_scene_status(request: web.Request) -> web.Response:
    """Proxy audio scene status from bridge."""
    bridge_url = request.app["bridge_url"]
    try:
        async with aiohttp.ClientSession() as cs:
            async with cs.get(f"{bridge_url}/audio-scene/status") as resp:
                data = await resp.json()
                return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=502)


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


async def handle_audio_auto(request: web.Request) -> web.FileResponse:
    """Auto-detect audio file (ogg preferred, wav fallback)."""
    date = request.match_info["date"]
    time = request.match_info["time"]

    session_dir = Path(RECORDINGS_DIR) / date / time
    for name in ("audio.ogg", "audio.wav"):
        audio_path = session_dir / name
        if audio_path.exists():
            mime = _AUDIO_MIME.get(audio_path.suffix, "application/octet-stream")
            return web.FileResponse(audio_path, headers={"Content-Type": mime})

    raise web.HTTPNotFound(text="audio not found")


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


async def handle_speakers_list_detailed(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/speakers/list-detailed")


async def handle_speakers_tag_correction(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "PUT", "/speakers/tag-correction", timeout_sec=120)


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
# Tag Taxonomy API
# ============================================================

async def handle_get_tag_taxonomy(request: web.Request) -> web.Response:
    """GET /api/tag-taxonomy — return current taxonomy."""
    data = _read_json(_TAG_TAXONOMY_PATH, None)
    if data is None:
        # Return default taxonomy
        from session_tagger import _DEFAULT_TAXONOMY
        data = dict(_DEFAULT_TAXONOMY)
        _write_json(_TAG_TAXONOMY_PATH, data)
    return web.json_response(data)


async def handle_put_tag_taxonomy(request: web.Request) -> web.Response:
    """PUT /api/tag-taxonomy — update taxonomy + optionally retag affected sessions."""
    body = await request.json()
    new_taxonomy = body.get("taxonomy")
    if not new_taxonomy or not isinstance(new_taxonomy, dict):
        return web.json_response({"error": "taxonomy object required"}, status=400)

    # Validate required fields
    for key in ("conversation_type", "outcomes", "domains", "importance"):
        if key not in new_taxonomy or not isinstance(new_taxonomy[key], list):
            return web.json_response({"error": f"{key} must be a list"}, status=400)

    # Check impact: count sessions using removed values
    old_taxonomy = _read_json(_TAG_TAXONOMY_PATH, None)
    impact = {}
    if old_taxonomy and isinstance(old_taxonomy, dict):
        indexer: Indexer = request.app["indexer"]
        for axis, col in [
            ("conversation_type", "tags_conversation_type"),
            ("outcomes", "tags_outcomes"),
            ("domains", "tags_domains"),
            ("importance", "tags_importance_v2"),
        ]:
            old_vals = set(old_taxonomy.get(axis, []))
            new_vals = set(new_taxonomy.get(axis, []))
            removed = old_vals - new_vals
            if removed:
                for val in removed:
                    try:
                        if axis in ("outcomes", "domains"):
                            cursor = await indexer._db.execute(
                                f"SELECT COUNT(*) as cnt FROM sessions WHERE {col} LIKE ?",
                                ["%" + val + "%"],
                            )
                        else:
                            cursor = await indexer._db.execute(
                                f"SELECT COUNT(*) as cnt FROM sessions WHERE {col} = ?",
                                [val],
                            )
                        row = await cursor.fetchone()
                        cnt = row["cnt"] if row else 0
                        if cnt > 0:
                            impact[val] = cnt
                    except Exception:
                        pass

    # Save new taxonomy
    new_taxonomy["version"] = (old_taxonomy or {}).get("version", 0) + 1
    _write_json(_TAG_TAXONOMY_PATH, new_taxonomy)

    # Notify Bridge to reload taxonomy
    await _notify_bridge_taxonomy_reload(request)

    retag = body.get("retag", False)
    retag_started = False
    if retag and BRIDGE_MGMT_URL:
        try:
            session_obj: aiohttp.ClientSession = request.app["bridge_session"]
            timeout = aiohttp.ClientTimeout(total=5)
            await session_obj.post(
                f"{BRIDGE_MGMT_URL}/session-tags/start",
                json={"force": True},
                timeout=timeout,
            )
            retag_started = True
        except Exception as e:
            logger.warning(f"Failed to start retag: {e}")

    return web.json_response({
        "status": "ok",
        "impact": impact,
        "retag_started": retag_started,
    })


async def _notify_bridge_taxonomy_reload(request: web.Request) -> None:
    """Notify Bridge to reload taxonomy."""
    if not BRIDGE_MGMT_URL:
        return
    session_obj: aiohttp.ClientSession = request.app["bridge_session"]
    try:
        timeout = aiohttp.ClientTimeout(total=3)
        await session_obj.post(f"{BRIDGE_MGMT_URL}/taxonomy/reload", timeout=timeout)
    except Exception as e:
        logger.debug(f"Bridge taxonomy reload notification failed: {e}")


# ============================================================
# Session Tag 修正 API + Keyword Suggestion API
# ============================================================

_PENDING_KW_SUGGESTIONS_PATH = Path(CONFIG_DIR) / "pending_keyword_suggestions.json"


async def handle_put_session_tags(request: web.Request) -> web.Response:
    """PUT /api/sessions/{date}/{time}/session-tags — update V2 session tags (partial)."""
    date = request.match_info["date"]
    time = request.match_info["time"]
    body = await request.json()

    # Load taxonomy for validation
    taxonomy = _read_json(_TAG_TAXONOMY_PATH, None)
    if not taxonomy:
        return web.json_response({"error": "taxonomy not loaded"}, status=500)

    valid_conv = set(taxonomy.get("conversation_type", []))
    valid_outcomes = set(taxonomy.get("outcomes", []))
    valid_domains = set(taxonomy.get("domains", []))
    valid_importance = set(taxonomy.get("importance", []))

    conv_type = body.get("conversation_type")
    outcomes = body.get("outcomes", [])
    domains = body.get("domains", [])
    importance = body.get("importance")

    if conv_type and conv_type not in valid_conv:
        return web.json_response({"error": f"invalid conversation_type: {conv_type}"}, status=400)
    for o in outcomes:
        if o not in valid_outcomes:
            return web.json_response({"error": f"invalid outcome: {o}"}, status=400)
    for d in domains:
        if d not in valid_domains:
            return web.json_response({"error": f"invalid domain: {d}"}, status=400)
    if importance and importance not in valid_importance:
        return web.json_response({"error": f"invalid importance: {importance}"}, status=400)

    transcript_path = Path(RECORDINGS_DIR) / date / time / "transcript.json"
    if not transcript_path.exists():
        return web.json_response({"error": "session not found"}, status=404)

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return web.json_response({"error": "transcript read error"}, status=500)

    session = data.setdefault("session", {})
    tags = session.setdefault("tags", {})

    # Remember old state for keyword learning and feedback
    old_domains = list(tags.get("domains", []))
    was_already_corrected = tags.get("manually_corrected", False)

    # Partial update: only provided fields (empty lists are valid)
    if conv_type is not None:
        tags["conversation_type"] = conv_type
    if "outcomes" in body:
        tags["outcomes"] = outcomes
    if "domains" in body:
        tags["domains"] = domains
    if importance is not None:
        tags["importance"] = importance

    tags["manually_corrected"] = True
    tags["corrected_at"] = datetime.now().isoformat(timespec="seconds")
    if not was_already_corrected:
        tags["review_outcome"] = "corrected"

    transcript_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Re-index + feedback
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
                tags_user, speakers, last_modified,
                tags_source_type, tags_business_domain, tags_importance,
                tags_confidentiality, tags_decision_required,
                tags_decision_summary, tags_action_items, tags_outcome,
                tags_conversation_type, tags_outcomes, tags_domains,
                tags_decision_memo, tags_importance_v2, tags_version,
                tags_external_audio,
                tags_audio_scene, tags_audio_scene_spatial, tags_audio_scene_confidence,
                tags_audio_noise_sources, tags_audio_events,
                tags_audio_music_detected, tags_audio_speaker_count,
                tags_tag_source, tags_rag_similarity, tags_manually_corrected, tags_corrected_at,
                tags_review_outcome)
               VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                       :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                       :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                       :tags_quality, :tags_entities, :tags_summary, :session_tagged,
                       :tags_user, :speakers, :last_modified,
                       :tags_source_type, :tags_business_domain, :tags_importance,
                       :tags_confidentiality, :tags_decision_required,
                       :tags_decision_summary, :tags_action_items, :tags_outcome,
                       :tags_conversation_type, :tags_outcomes, :tags_domains,
                       :tags_decision_memo, :tags_importance_v2, :tags_version,
                       :tags_external_audio,
                       :tags_audio_scene, :tags_audio_scene_spatial, :tags_audio_scene_confidence,
                       :tags_audio_noise_sources, :tags_audio_events,
                       :tags_audio_music_detected, :tags_audio_speaker_count,
                       :tags_tag_source, :tags_rag_similarity, :tags_manually_corrected, :tags_corrected_at,
                       :tags_review_outcome)""",
            info,
        )
        await indexer._db.commit()
        if not was_already_corrected:
            indexer.record_review_feedback(info, was_corrected=True)
    except Exception as e:
        logger.warning(f"Re-index after session-tags update failed: {e}")

    # Notify Bridge for keyword learning if domains changed
    new_domains = list(tags.get("domains", []))
    if set(new_domains) != set(old_domains) and BRIDGE_MGMT_URL:
        try:
            session_obj: aiohttp.ClientSession = request.app["bridge_session"]
            timeout = aiohttp.ClientTimeout(total=5)
            await session_obj.post(
                f"{BRIDGE_MGMT_URL}/keywords/learn",
                json={
                    "date": date,
                    "time": time,
                    "old_domains": old_domains,
                    "new_domains": new_domains,
                },
                timeout=timeout,
            )
        except Exception as e:
            logger.debug(f"Keyword learn notification failed: {e}")

    return web.json_response({"status": "ok", "tags": tags})


async def handle_get_keyword_suggestions(request: web.Request) -> web.Response:
    """GET /api/keyword-suggestions — list pending keyword suggestions."""
    suggestions = _read_json(_PENDING_KW_SUGGESTIONS_PATH, [])
    pending = [s for s in suggestions if s.get("status") == "pending"]
    pending.sort(key=lambda s: s.get("occurrences", 0), reverse=True)
    return web.json_response({"suggestions": pending})


async def handle_approve_keyword_suggestion(request: web.Request) -> web.Response:
    """POST /api/keyword-suggestions/{id}/approve — approve and add to keyword_anchors."""
    suggestion_id = request.match_info["id"]
    suggestions = _read_json(_PENDING_KW_SUGGESTIONS_PATH, [])

    target = None
    for s in suggestions:
        if s.get("id") == suggestion_id:
            target = s
            break
    if not target:
        return web.json_response({"error": "suggestion not found"}, status=404)

    target["status"] = "approved"
    target["updated_at"] = datetime.now().isoformat(timespec="seconds")

    # Add to keyword_anchors in taxonomy
    taxonomy = _read_json(_TAG_TAXONOMY_PATH, {})
    anchors = taxonomy.setdefault("keyword_anchors", {})
    domain_kws = anchors.setdefault(target["domain"], [])
    if target["keyword"] not in domain_kws:
        domain_kws.append(target["keyword"])
    _write_json(_TAG_TAXONOMY_PATH, taxonomy)

    # Save suggestions
    _write_json(_PENDING_KW_SUGGESTIONS_PATH, suggestions)

    # Notify Bridge to reload taxonomy
    await _notify_bridge_taxonomy_reload(request)

    return web.json_response({"status": "ok", "suggestion": target})


async def handle_reject_keyword_suggestion(request: web.Request) -> web.Response:
    """POST /api/keyword-suggestions/{id}/reject — reject a keyword suggestion."""
    suggestion_id = request.match_info["id"]
    suggestions = _read_json(_PENDING_KW_SUGGESTIONS_PATH, [])

    target = None
    for s in suggestions:
        if s.get("id") == suggestion_id:
            target = s
            break
    if not target:
        return web.json_response({"error": "suggestion not found"}, status=404)

    target["status"] = "rejected"
    target["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(_PENDING_KW_SUGGESTIONS_PATH, suggestions)

    return web.json_response({"status": "ok", "suggestion": target})


# ============================================================
# Combined Tag プロキシ API (Voice Tag → Session Tag)
# ============================================================

async def handle_combined_tag_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/tags/start")


async def handle_combined_tag_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/tags/status")


async def handle_combined_tag_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/tags/cancel")


# ============================================================
# VibeVoice-ASR プロキシ API (Bridge へ中継)
# ============================================================

async def handle_vibevoice_start(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/vibevoice/start")


async def handle_vibevoice_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/vibevoice/status")


async def handle_vibevoice_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/vibevoice/cancel")


# ============================================================
# Whisper Config プロキシ API
# ============================================================

async def handle_whisper_config(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/whisper/config")


# ============================================================
# RAG Classifier プロキシ API
# ============================================================

async def handle_rag_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/rag/status")


async def handle_rag_rebuild(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/rag/rebuild", timeout_sec=10)


# ============================================================
# Taxonomy Refiner プロキシ API
# ============================================================

async def handle_taxonomy_refine_run(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/taxonomy/refine/run", timeout_sec=10)


async def handle_taxonomy_refine_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/taxonomy/refine/status")


async def handle_taxonomy_refine_pending(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/taxonomy/refine/pending")


async def handle_taxonomy_refine_approve(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/taxonomy/refine/approve", timeout_sec=10)


async def handle_taxonomy_refine_reject(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/taxonomy/refine/reject", timeout_sec=10)


# ============================================================
# Review Queue API
# ============================================================

async def handle_review_queue(request: web.Request) -> web.Response:
    """GET /api/review/queue — get next items to review, ordered by priority."""
    indexer: Indexer = request.app["indexer"]
    limit = max(1, min(20, int(request.query.get("limit", "1"))))
    offset = max(0, int(request.query.get("offset", "0")))

    try:
        items = await indexer.get_review_queue(limit=limit, offset=offset)
    except Exception as e:
        logger.exception("Review queue query failed")
        return web.json_response({"error": str(e)}, status=500)

    # Attach transcript for each item
    results = []
    for item in items:
        transcript = await indexer.get_transcript(item["date"], item["time"])
        results.append({
            "session": item,
            "transcript": transcript,
        })

    return web.json_response({"items": results})


async def handle_review_stats(request: web.Request) -> web.Response:
    """GET /api/review/stats — review queue statistics."""
    indexer: Indexer = request.app["indexer"]
    try:
        stats = await indexer.get_review_stats()
    except Exception as e:
        logger.exception("Review stats query failed")
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response(stats)


async def handle_review_accept(request: web.Request) -> web.Response:
    """POST /api/review/accept — mark session as reviewed (accept current tags).

    Body: {"date": "YYYY-MM-DD", "time": "HH-MM-SS"}
    Writes manually_corrected=true + corrected_at to transcript.json and re-indexes.
    Idempotent: accepting an already-accepted session is a no-op (200 OK).
    """
    body = await request.json()
    date = body.get("date", "").strip()
    time_val = body.get("time", "").strip()

    if not date or not time_val:
        return web.json_response({"error": "date and time are required"}, status=400)

    transcript_path = Path(RECORDINGS_DIR) / date / time_val / "transcript.json"
    if not transcript_path.exists():
        return web.json_response({"error": "session not found"}, status=404)

    try:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return web.json_response({"error": "transcript read error"}, status=500)

    session = data.setdefault("session", {})
    tags = session.setdefault("tags", {})

    tags["manually_corrected"] = True
    tags["corrected_at"] = datetime.now().isoformat(timespec="seconds")
    tags["review_outcome"] = "accepted"

    transcript_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Re-index + feedback
    indexer: Indexer = request.app["indexer"]
    try:
        info = indexer._parse_transcript(date, time_val)
        info["last_modified"] = transcript_path.stat().st_mtime
        await indexer._db.execute(
            """INSERT OR REPLACE INTO sessions
               (date, time, start_at, end_at, duration_seconds,
                audio_size_mb, model, language, transcript_text, segment_count,
                tags_topics, tags_activity, tags_intent, tags_mood,
                tags_quality, tags_entities, tags_summary, session_tagged,
                tags_user, speakers, last_modified,
                tags_source_type, tags_business_domain, tags_importance,
                tags_confidentiality, tags_decision_required,
                tags_decision_summary, tags_action_items, tags_outcome,
                tags_conversation_type, tags_outcomes, tags_domains,
                tags_decision_memo, tags_importance_v2, tags_version,
                tags_external_audio,
                tags_audio_scene, tags_audio_scene_spatial, tags_audio_scene_confidence,
                tags_audio_noise_sources, tags_audio_events,
                tags_audio_music_detected, tags_audio_speaker_count,
                tags_tag_source, tags_rag_similarity, tags_manually_corrected, tags_corrected_at,
                tags_review_outcome)
               VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                       :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                       :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                       :tags_quality, :tags_entities, :tags_summary, :session_tagged,
                       :tags_user, :speakers, :last_modified,
                       :tags_source_type, :tags_business_domain, :tags_importance,
                       :tags_confidentiality, :tags_decision_required,
                       :tags_decision_summary, :tags_action_items, :tags_outcome,
                       :tags_conversation_type, :tags_outcomes, :tags_domains,
                       :tags_decision_memo, :tags_importance_v2, :tags_version,
                       :tags_external_audio,
                       :tags_audio_scene, :tags_audio_scene_spatial, :tags_audio_scene_confidence,
                       :tags_audio_noise_sources, :tags_audio_events,
                       :tags_audio_music_detected, :tags_audio_speaker_count,
                       :tags_tag_source, :tags_rag_similarity, :tags_manually_corrected, :tags_corrected_at,
                       :tags_review_outcome)""",
            info,
        )
        await indexer._db.commit()
        indexer.record_review_feedback(info, was_corrected=False)
    except Exception as e:
        logger.warning(f"Re-index after review accept failed: {e}")

    return web.json_response({"status": "ok"})


async def handle_review_skip(request: web.Request) -> web.Response:
    """POST /api/review/skip — skip a review item (no persistence, 200 OK)."""
    return web.json_response({"status": "ok"})


# ============================================================
# Export API
# ============================================================

_VALID_EXPORT_FORMATS = {
    "comprehensive", "openai", "conversation",
    "knowledge", "decision_log", "sop",
    "safe_training",
}


def _parse_export_params(request: web.Request) -> dict:
    """Parse common export query parameters."""
    fmt = request.query.get("format", "comprehensive")
    if fmt not in _VALID_EXPORT_FORMATS:
        raise web.HTTPBadRequest(
            text="format must be one of: " + ", ".join(sorted(_VALID_EXPORT_FORMATS))
        )

    quality_raw = request.query.get("quality", "").strip()
    quality = [q.strip() for q in quality_raw.split(",") if q.strip()] or None

    activity_raw = request.query.get("activity", "").strip()
    activity = [a.strip() for a in activity_raw.split(",") if a.strip()] or None

    business_domain = request.query.get("business_domain", "").strip()

    # V2 taxonomy filters
    conversation_type = request.query.get("conversation_type", "").strip()
    domain = request.query.get("domain", "").strip()
    importance_v2 = request.query.get("importance_v2", "").strip()

    # include_monologue: default False for openai/conversation, True otherwise
    include_monologue_raw = request.query.get("include_monologue", "").strip().lower()
    if include_monologue_raw in ("true", "1", "yes"):
        include_monologue = True
    elif include_monologue_raw in ("false", "0", "no"):
        include_monologue = False
    else:
        include_monologue = fmt not in ("openai", "conversation")

    return {
        "format": fmt,
        "date_from": request.query.get("date_from", "").strip(),
        "date_to": request.query.get("date_to", "").strip(),
        "quality": quality,
        "activity": activity,
        "business_domain": business_domain,
        "conversation_type": conversation_type,
        "domain": domain,
        "importance_v2": importance_v2,
        "speaker_mode": request.query.get("speaker_mode", "conversation"),
        "min_segments": int(request.query.get("min_segments", "3")),
        "include_monologue": include_monologue,
    }


async def handle_export_stats(request: web.Request) -> web.Response:
    """GET /api/export/stats — preview statistics for export."""
    try:
        params = _parse_export_params(request)
    except web.HTTPBadRequest as e:
        return web.json_response({"error": e.text}, status=400)

    indexer: Indexer = request.app["indexer"]
    try:
        stats = await indexer.get_export_stats(
            date_from=params["date_from"],
            date_to=params["date_to"],
            quality=params["quality"],
            activity=params["activity"],
            min_segments=params["min_segments"],
            business_domain=params["business_domain"],
            conversation_type=params["conversation_type"],
            domain=params["domain"],
            importance_v2=params["importance_v2"],
        )
    except Exception as e:
        logger.exception("Export stats failed")
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response(stats)


async def handle_export_training(request: web.Request) -> web.Response:
    """GET /api/export/training — stream JSONL download."""
    try:
        params = _parse_export_params(request)
    except web.HTTPBadRequest as e:
        return web.json_response({"error": e.text}, status=400)

    indexer: Indexer = request.app["indexer"]
    try:
        session_pairs = await indexer.get_export_sessions(
            date_from=params["date_from"],
            date_to=params["date_to"],
            quality=params["quality"],
            activity=params["activity"],
            min_segments=params["min_segments"],
            business_domain=params["business_domain"],
            conversation_type=params["conversation_type"],
            domain=params["domain"],
            importance_v2=params["importance_v2"],
        )
    except Exception as e:
        logger.exception("Export query failed")
        return web.json_response({"error": str(e)}, status=500)

    fmt = params["format"]
    speaker_mode = params["speaker_mode"]
    include_monologue = params["include_monologue"]

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "application/x-ndjson",
            "Content-Disposition": f'attachment; filename="training_{fmt}.jsonl"',
        },
    )
    await response.prepare(request)

    for date, time in session_pairs:
        try:
            transcript = await indexer.get_transcript(date, time)
            if not transcript:
                continue
            session_id = f"{date}/{time}"
            line = export_session(fmt, session_id, transcript, speaker_mode,
                                  include_monologue=include_monologue)
            if line:
                await response.write((line + "\n").encode("utf-8"))
        except Exception:
            logger.exception("Export error for %s/%s", date, time)
            continue

    await response.write_eof()
    return response


# ============================================================
# Sanitize API (safe training data generation via Bridge LLM)
# ============================================================

def _parse_sanitize_params(request: web.Request) -> dict:
    """Parse query parameters for sanitize (reuses export filter params)."""
    quality_raw = request.query.get("quality", "").strip()
    quality = [q.strip() for q in quality_raw.split(",") if q.strip()] or None

    activity_raw = request.query.get("activity", "").strip()
    activity = [a.strip() for a in activity_raw.split(",") if a.strip()] or None

    business_domain = request.query.get("business_domain", "").strip()

    return {
        "date_from": request.query.get("date_from", "").strip(),
        "date_to": request.query.get("date_to", "").strip(),
        "quality": quality,
        "activity": activity,
        "business_domain": business_domain,
        "conversation_type": request.query.get("conversation_type", "").strip(),
        "domain": request.query.get("domain", "").strip(),
        "importance_v2": request.query.get("importance_v2", "").strip(),
        "min_segments": int(request.query.get("min_segments", "3")),
    }


async def handle_sanitize_start(request: web.Request) -> web.Response:
    """POST /api/sanitize/start — query DB for sessions, send list to Bridge."""
    if not BRIDGE_MGMT_URL:
        return web.json_response({"error": "bridge not configured"}, status=503)

    params = _parse_sanitize_params(request)

    indexer: Indexer = request.app["indexer"]
    try:
        session_pairs = await indexer.get_export_sessions(
            date_from=params["date_from"],
            date_to=params["date_to"],
            quality=params["quality"],
            activity=params["activity"],
            min_segments=params["min_segments"],
            business_domain=params["business_domain"],
            conversation_type=params.get("conversation_type", ""),
            domain=params.get("domain", ""),
            importance_v2=params.get("importance_v2", ""),
        )
    except Exception as e:
        logger.exception("Sanitize session query failed")
        return web.json_response({"error": str(e)}, status=500)

    if not session_pairs:
        return web.json_response({
            "status": "no_sessions",
            "message": "No sessions match the filters",
        })

    sessions = [f"{date}/{time}" for date, time in session_pairs]

    # Forward to Bridge
    session: aiohttp.ClientSession = request.app["bridge_session"]
    url = f"{BRIDGE_MGMT_URL}/sanitize/start"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        resp = await session.post(
            url,
            json={"sessions": sessions},
            timeout=timeout,
        )
        body = await resp.read()
        return web.Response(
            body=body,
            status=resp.status,
            content_type=resp.content_type or "application/json",
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Bridge proxy error (sanitize/start): {e}")
        return web.json_response({"error": "bridge unavailable"}, status=502)


async def handle_sanitize_status(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "GET", "/sanitize/status")


async def handle_sanitize_cancel(request: web.Request) -> web.Response:
    return await _proxy_bridge(request, "POST", "/sanitize/cancel")


async def handle_sanitize_download(request: web.Request) -> web.Response:
    """GET /api/sanitize/download — serve file from shared config dir."""
    import re as _re
    filename = request.query.get("file", "")
    # Security: only allow safe_training_*.jsonl pattern + path traversal check
    if not _re.match(r'^safe_training_\d{8}_\d{6}\.jsonl$', filename):
        return web.json_response({"error": "invalid filename"}, status=400)

    filepath = Path(CONFIG_DIR) / filename
    # Ensure resolved path stays within CONFIG_DIR
    try:
        filepath.resolve().relative_to(Path(CONFIG_DIR).resolve())
    except ValueError:
        return web.json_response({"error": "invalid path"}, status=400)

    if not filepath.exists():
        return web.json_response({"error": "file not found"}, status=404)

    return web.FileResponse(
        filepath,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


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
    app["bridge_url"] = BRIDGE_MGMT_URL


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
    app.router.add_get("/api/speaker-names", handle_api_speaker_names)
    app.router.add_get("/api/audio-scene-labels", handle_audio_scene_labels)
    app.router.add_get("/api/audio-scene/status", handle_audio_scene_status)
    app.router.add_get("/api/whisper/config", handle_whisper_config)
    app.router.add_get("/api/sessions/{date}/{time}", handle_api_session_detail)
    app.router.add_get("/audio/{date}/{time}/", handle_audio_auto)
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
    app.router.add_get("/api/speakers/list-detailed", handle_speakers_list_detailed)
    app.router.add_put("/api/speakers/tag-correction", handle_speakers_tag_correction)
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

    # Tag taxonomy API
    app.router.add_get("/api/tag-taxonomy", handle_get_tag_taxonomy)
    app.router.add_put("/api/tag-taxonomy", handle_put_tag_taxonomy)

    # Session tag correction + keyword suggestion API
    app.router.add_put("/api/sessions/{date}/{time}/session-tags", handle_put_session_tags)
    app.router.add_get("/api/keyword-suggestions", handle_get_keyword_suggestions)
    app.router.add_post("/api/keyword-suggestions/{id}/approve", handle_approve_keyword_suggestion)
    app.router.add_post("/api/keyword-suggestions/{id}/reject", handle_reject_keyword_suggestion)

    # Combined tag プロキシ API (Voice Tag → Session Tag)
    app.router.add_post("/api/tags/start", handle_combined_tag_start)
    app.router.add_get("/api/tags/status", handle_combined_tag_status)
    app.router.add_post("/api/tags/cancel", handle_combined_tag_cancel)

    # VibeVoice-ASR プロキシ API
    app.router.add_post("/api/vibevoice/start", handle_vibevoice_start)
    app.router.add_get("/api/vibevoice/status", handle_vibevoice_status)
    app.router.add_post("/api/vibevoice/cancel", handle_vibevoice_cancel)

    # Review queue API
    app.router.add_get("/api/review/queue", handle_review_queue)
    app.router.add_get("/api/review/stats", handle_review_stats)
    app.router.add_post("/api/review/accept", handle_review_accept)
    app.router.add_post("/api/review/skip", handle_review_skip)

    # RAG classifier プロキシ API
    app.router.add_get("/api/rag/status", handle_rag_status)
    app.router.add_post("/api/rag/rebuild", handle_rag_rebuild)
    app.router.add_post("/api/taxonomy/refine/run", handle_taxonomy_refine_run)
    app.router.add_get("/api/taxonomy/refine/status", handle_taxonomy_refine_status)
    app.router.add_get("/api/taxonomy/refine/pending", handle_taxonomy_refine_pending)
    app.router.add_post("/api/taxonomy/refine/approve", handle_taxonomy_refine_approve)
    app.router.add_post("/api/taxonomy/refine/reject", handle_taxonomy_refine_reject)

    # Export API
    app.router.add_get("/api/export/stats", handle_export_stats)
    app.router.add_get("/api/export/training", handle_export_training)

    # Sanitize batch API (training data anonymization)
    app.router.add_post("/api/sanitize/start", handle_sanitize_start)
    app.router.add_get("/api/sanitize/status", handle_sanitize_status)
    app.router.add_post("/api/sanitize/cancel", handle_sanitize_cancel)
    app.router.add_get("/api/sanitize/download", handle_sanitize_download)

    app.router.add_static("/static/", path=str(base_dir / "static"), name="static")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
