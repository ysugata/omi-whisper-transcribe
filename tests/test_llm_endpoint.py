"""Tests for model-prefix based LLM endpoint resolution and gpt-5 param handling."""
import sys
import types
from pathlib import Path
from unittest import mock

# Mock heavy dependencies before importing omi_whisper_bridge
_MOCK_MODULES = [
    "faster_whisper", "faster_whisper.vad",
    "torch", "torchaudio",
    "whisperx",
    "noisereduce",
    "pyannote", "pyannote.audio",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

sys.modules["faster_whisper.vad"].get_speech_timestamps = mock.MagicMock()
sys.modules["faster_whisper.vad"].VadOptions = mock.MagicMock()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import omi_whisper_bridge as bridge  # noqa: E402
from llm_pipeline import LLMPipeline  # noqa: E402


def test_resolve_gpt_model_uses_openai(monkeypatch):
    monkeypatch.setattr(bridge, "OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(bridge, "GEMINI_API_KEY", "gm-key")
    key, url = bridge.resolve_llm_endpoint("gpt-5-nano")
    assert key == "sk-openai"
    assert "api.openai.com" in url


def test_resolve_gemini_model_uses_gemini(monkeypatch):
    monkeypatch.setattr(bridge, "OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(bridge, "GEMINI_API_KEY", "gm-key")
    key, url = bridge.resolve_llm_endpoint("gemini-3.1-flash-lite")
    assert key == "gm-key"
    assert "generativelanguage.googleapis.com" in url


def _capture_call_kwargs(model: str) -> dict:
    p = LLMPipeline(api_key="k", model=model, reasoning_effort="minimal")
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            resp = mock.MagicMock()
            resp.choices[0].message.content = "{}"
            return resp

    fake = mock.MagicMock()
    fake.chat.completions = _FakeCompletions()
    p._client = fake
    p._call_api("sys", "user", {"type": "json_object"}, max_tokens=512)
    return captured


def test_gpt5_call_uses_completion_tokens_and_effort():
    kw = _capture_call_kwargs("gpt-5-nano")
    # gpt-5 rejects max_tokens/temperature on Chat Completions
    assert "max_tokens" not in kw
    assert "temperature" not in kw
    assert kw["max_completion_tokens"] == 512
    assert kw["reasoning_effort"] == "minimal"


def test_gemini_call_keeps_legacy_params():
    kw = _capture_call_kwargs("gemini-3.1-flash-lite")
    assert kw["max_tokens"] == 512
    assert kw["temperature"] == 0
    assert kw["reasoning_effort"] == "low"  # gemini-3.x thinking minimized


def test_filter_strips_line_number_prefix():
    """gpt-5-nano returns snippets with the '[n] ' prefix — must still match."""
    corrections = [{
        "line": 1,
        "original_snippet": "[1] クーバネティスのデプロイ",
        "corrected": "[1] Kubernetesのデプロイ",
        "source": "glossary",
    }]
    out = LLMPipeline._filter_corrections(
        corrections,
        original_texts={1: "クーバネティスのデプロイ"},
    )
    assert len(out) == 1
    assert out[0]["original_snippet"] == "クーバネティスのデプロイ"
    assert out[0]["corrected"] == "Kubernetesのデプロイ"
