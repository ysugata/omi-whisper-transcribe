"""Omi Recordings Viewer — aiohttp web server."""

import logging
import os
from pathlib import Path

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
DB_PATH = os.environ.get("DB_PATH", "/app/data/index.db")
PORT = int(os.environ.get("PORT", "8092"))


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


async def handle_audio(request: web.Request) -> web.FileResponse:
    date = request.match_info["date"]
    time = request.match_info["time"]

    audio_path = Path(RECORDINGS_DIR) / date / time / "audio.wav"
    if not audio_path.exists():
        raise web.HTTPNotFound(text="audio not found")

    return web.FileResponse(audio_path, headers={"Content-Type": "audio/wav"})


async def on_startup(app: web.Application):
    indexer = Indexer(DB_PATH, RECORDINGS_DIR)
    await indexer.open()
    n = await indexer.sync()
    logger.info("Initial sync: indexed %d new sessions", n)

    app["indexer"] = indexer

    import asyncio
    app["sync_task"] = asyncio.create_task(indexer.periodic_sync())


async def on_cleanup(app: web.Application):
    app["sync_task"].cancel()
    await app["indexer"].close()


def create_app() -> web.Application:
    app = web.Application()

    base_dir = Path(__file__).parent
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(base_dir / "templates")))

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/sessions", handle_api_sessions)
    app.router.add_get("/api/sessions/{date}/{time}", handle_api_session_detail)
    app.router.add_get("/audio/{date}/{time}/audio.wav", handle_audio)
    app.router.add_static("/static/", path=str(base_dir / "static"), name="static")

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=PORT)
