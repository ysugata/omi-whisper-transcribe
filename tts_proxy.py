"""
TTS リバースプロキシ: qwen-tts-api への HTTP リクエストを中継する。

GPU が Whisper に使われている場合は自動的に解放し、
qwen-tts が healthy になるまで待ってからリクエストを転送する。

omi_whisper_bridge.py の既存イベントループ上で動作する。
"""

import asyncio
import logging
import time

import aiohttp
from aiohttp import web

log = logging.getLogger("omi-whisper.tts-proxy")

# 転送しないホップバイホップヘッダー
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


class TtsProxy:
    """aiohttp ベースの TTS リバースプロキシ"""

    def __init__(self, gpu_manager, qwen_tts_url: str, port: int = 8000):
        self._gpu = gpu_manager
        self._qwen_tts_url = qwen_tts_url.rstrip("/")
        self._port = port
        self._runner: web.AppRunner | None = None

    async def start(self):
        """プロキシサーバーを起動 (既存イベントループ上)"""
        app = web.Application()
        app.router.add_route("*", "/proxy-status", self._handle_proxy_status)
        app.router.add_route("*", "/{path_info:.*}", self._handle_proxy)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        log.info(f"TTS proxy listening on http://0.0.0.0:{self._port}")
        log.info(f"Forwarding to {self._qwen_tts_url}")

    async def stop(self):
        """プロキシサーバーを停止"""
        if self._runner:
            await self._runner.cleanup()
            log.info("TTS proxy stopped")

    async def _handle_proxy_status(self, request: web.Request) -> web.Response:
        """プロキシ自体のステータス"""
        return web.json_response({
            "proxy": "running",
            "gpu_state": self._gpu.state.name,
            "session_count": self._gpu._session_count,
            "qwen_tts_url": self._qwen_tts_url,
        })

    async def _handle_proxy(self, request: web.Request) -> web.Response:
        """全リクエストを qwen-tts に転送"""
        path = request.match_info.get("path_info", "")
        target_url = f"{self._qwen_tts_url}/{path}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        t0 = time.time()
        log.info(f"Proxy request: {request.method} /{path} (gpu_state={self._gpu.state.name})")

        # GPU を TTS 用に解放
        try:
            await self._gpu.release_gpu_for_tts()
        except Exception as e:
            log.error(f"Failed to release GPU for TTS: {e}", exc_info=True)
            return web.json_response(
                {"error": "Failed to release GPU for TTS", "detail": str(e)},
                status=503,
            )

        # qwen-tts が healthy になるまで待機
        healthy = await self._gpu.wait_for_qwen_tts_healthy(timeout=600)
        if not healthy:
            # STT リクエストが来て qwen-tts が停止された場合
            if self._gpu.state.name == "ACQUIRING":
                return web.json_response(
                    {"error": "GPU acquired by STT, TTS unavailable", "gpu_state": "ACQUIRING"},
                    status=503,
                )
            return web.json_response(
                {"error": "qwen-tts did not become healthy in time"},
                status=504,
            )

        waited = time.time() - t0
        if waited > 1.0:
            log.info(f"GPU switch took {waited:.1f}s")

        # リクエストを転送
        try:
            body = await request.read()
            headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
            }

            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp_body = await resp.read()
                    resp_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in _HOP_BY_HOP
                    }
                    total = time.time() - t0
                    log.info(
                        f"Proxy response: {resp.status} ({total:.1f}s total, "
                        f"{total - waited:.1f}s upstream)"
                    )
                    return web.Response(
                        status=resp.status,
                        headers=resp_headers,
                        body=resp_body,
                    )
        except aiohttp.ClientError as e:
            log.error(f"Proxy upstream error: {e}")
            return web.json_response(
                {"error": "Failed to connect to qwen-tts", "detail": str(e)},
                status=502,
            )
        except Exception as e:
            log.error(f"Proxy error: {e}", exc_info=True)
            return web.json_response(
                {"error": "Internal proxy error", "detail": str(e)},
                status=500,
            )
