"""Tests for management API token auth helpers (omi_whisper_bridge)."""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mock_bridge_deps():
    """Stub heavy imports so omi_whisper_bridge can be imported host-side."""
    stubs = {}
    for mod_name in [
        "faster_whisper", "faster_whisper.vad",
        "torch", "torch.cuda",
        "whisperx", "whisperx.diarize",
        "pyannote", "pyannote.audio",
        "torchaudio", "torchaudio.functional",
        "noisereduce",
        "openai",
    ]:
        if mod_name not in sys.modules:
            stubs[mod_name] = MagicMock()
    if "faster_whisper.vad" in stubs:
        stubs["faster_whisper.vad"].get_speech_timestamps = MagicMock()
        stubs["faster_whisper.vad"].VadOptions = MagicMock()
    sys.modules.update(stubs)


_mock_bridge_deps()

from omi_whisper_bridge import (  # noqa: E402
    _extract_mgmt_token,
    _mgmt_auth_ok,
    make_mgmt_auth_middleware,
)


class TestExtractToken:
    def test_bearer_header(self):
        assert _extract_mgmt_token({"Authorization": "Bearer abc"}, {}) == "abc"

    def test_query_param(self):
        assert _extract_mgmt_token({}, {"token": "xyz"}) == "xyz"

    def test_bearer_wins_over_query(self):
        assert _extract_mgmt_token(
            {"Authorization": "Bearer abc"}, {"token": "xyz"}
        ) == "abc"

    def test_non_bearer_auth_ignored(self):
        assert _extract_mgmt_token({"Authorization": "Basic abc"}, {}) == ""

    def test_empty(self):
        assert _extract_mgmt_token({}, {}) == ""


class TestAuthOk:
    def test_no_expected_token_allows_all(self):
        assert _mgmt_auth_ok("", "") is True
        assert _mgmt_auth_ok("anything", "") is True

    def test_matching_token(self):
        assert _mgmt_auth_ok("secret", "secret") is True

    def test_wrong_token(self):
        assert _mgmt_auth_ok("wrong", "secret") is False

    def test_missing_token(self):
        assert _mgmt_auth_ok("", "secret") is False


@pytest.mark.asyncio
async def test_middleware_blocks_and_allows():
    mw = make_mgmt_auth_middleware("secret")

    async def handler(request):
        return "OK"

    # Allowed: correct bearer token
    req = MagicMock()
    req.headers = {"Authorization": "Bearer secret"}
    req.query = {}
    assert await mw(req, handler) == "OK"

    # Allowed: correct query token
    req.headers = {}
    req.query = {"token": "secret"}
    assert await mw(req, handler) == "OK"

    # Blocked: wrong token
    req.headers = {"Authorization": "Bearer nope"}
    req.query = {}
    resp = await mw(req, handler)
    assert resp.status == 401

    # Blocked: no token
    req.headers = {}
    req.query = {}
    resp = await mw(req, handler)
    assert resp.status == 401
