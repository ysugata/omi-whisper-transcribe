"""Unit tests for succession_rag.py (mocked embeddings + LLM)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from succession_rag import KCSource, SuccessionAnswer, SuccessionRAG, _build_user_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sanitized_file(tmp_path: Path, sessions: list[dict]) -> Path:
    """Write a safe_training_*.jsonl fixture with the given session records."""
    p = tmp_path / "safe_training_test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in sessions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def _make_session(session_id: str, cards: list[dict]) -> dict:
    return {
        "session_id": session_id,
        "risk": "low",
        "summary": {"input": "", "output": f"{session_id} summary"},
        "keypoints": ["k1", "k2", "k3"],
        "safe_conversation": [
            {"speaker": "user", "text": "x"},
            {"speaker": "assistant", "text": "y"},
        ],
        "issues_detected": [],
        "knowledge_cards": cards,
        "knowledge_card": cards[0] if cards else None,
    }


def _kc(q: str, a: str, conf: str = "high") -> dict:
    return {
        "question": q,
        "answer": a,
        "evidence": [f"ev for {q[:10]}"],
        "confidence": conf,
    }


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_includes_query_and_sources(self):
        sources = [
            KCSource(
                session_id="sid1",
                question="FBAと自己配送の方針は?",
                answer="FBA統一+自己配送併用。",
                evidence=["FBA議論", "方針決定"],
                confidence="high",
                similarity=0.95,
            ),
        ]
        out = _build_user_prompt("物販の方針を教えて", sources)
        assert "物販の方針を教えて" in out
        assert "[#1]" in out
        assert "sid1" in out
        assert "FBAと自己配送" in out
        assert "根拠:" in out
        assert "0.950" in out


# ---------------------------------------------------------------------------
# SuccessionRAG — index build / retrieval
# ---------------------------------------------------------------------------


def _fake_embed_batch(self: SuccessionRAG, texts: list[str]) -> np.ndarray:
    """Deterministic 8-dim hash-based embeddings for tests."""
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(8).astype(np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        vecs.append(v)
    return np.stack(vecs) if vecs else np.zeros((0, 8), dtype=np.float32)


class TestBuildAndRetrieve:
    def test_build_empty_dir(self, tmp_path):
        rag = SuccessionRAG(tmp_path, "gkey", "okey")
        result = rag.build_index()
        assert result["status"] == "empty"
        assert rag.index_size == 0

    def test_build_from_sanitized_jsonl(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [
                _kc("FBAと自己配送の方針は?", "FBA統一+自己配送併用で利益最大化。"),
                _kc("Keepa APIで何を分析する?", "波形で将来の購買機会を見込む。"),
            ]),
            _make_session("s2", [
                _kc("Claude Codeのインストール方針は?", "Nano Banana経由で隔離して進める。"),
            ]),
        ])

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            result = rag.build_index()

        assert result["status"] == "built"
        assert result["size"] == 3
        assert rag.index_size == 3

    def test_persist_and_reload(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問", "A1 具体的な回答が入る長めの答え")])
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag1 = SuccessionRAG(tmp_path, "gkey", "okey")
            rag1.build_index()

            # Reload into a fresh instance
            rag2 = SuccessionRAG(tmp_path, "gkey", "okey")
            assert rag2.load_index() is True
            assert rag2.index_size == 1
            assert rag2.status()["fingerprint"] == rag1.status()["fingerprint"]

    def test_retrieve_returns_similarity_sorted(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [
                _kc("Q1 とても具体的な質問について", "A1 十分な長さの回答。"),
                _kc("Q2 別の角度からの具体的質問", "A2 別の具体的な回答文。"),
                _kc("Q3 さらに別の質問テキスト", "A3 さらに別の回答テキスト。"),
            ]),
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            results = rag.retrieve("Q1 とても具体的な質問について", k=2)

        assert len(results) == 2
        # Top result should be the exact-match question with similarity 1.0
        assert results[0].question == "Q1 とても具体的な質問について"
        assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
        assert results[0].similarity >= results[1].similarity

    def test_fingerprint_unchanged_skips_rebuild(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問", "A1 十分な長さの回答。")])
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            second = rag.build_index()
        assert second["status"] == "unchanged"


# ---------------------------------------------------------------------------
# ask()
# ---------------------------------------------------------------------------


class TestAsk:
    def test_empty_index_returns_warning(self, tmp_path):
        rag = SuccessionRAG(tmp_path, "gkey", "okey")
        ans = rag.ask("なんでもいい")
        assert ans.index_size == 0
        assert "empty_index" in ans.warnings

    def test_ask_full_flow_mocked(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問について", "A1 具体的な回答文が入ります。")])
        ])

        fake_response = MagicMock()
        fake_response.status = "completed"
        fake_response.output_text = "モックされた LLM 回答です [#1]"
        fake_response.incomplete_details = None

        fake_client = MagicMock()
        fake_client.responses.create.return_value = fake_response

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            rag._openai_client = fake_client  # inject mock

            ans = rag.ask("Q1 とても具体的な質問について", k=1)

        assert ans.answer == "モックされた LLM 回答です [#1]"
        assert len(ans.sources) == 1
        assert ans.sources[0].session_id == "s1"
        assert ans.warnings == []
        fake_client.responses.create.assert_called_once()

    def test_ask_llm_error_returns_sources(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問について", "A1 具体的な回答文が入ります。")])
        ])

        fake_client = MagicMock()
        fake_client.responses.create.side_effect = RuntimeError("rate limited")

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            rag._openai_client = fake_client

            ans = rag.ask("Q1 とても具体的な質問について", k=1)

        assert "llm_error" in ans.warnings[0]
        assert len(ans.sources) == 1  # Sources still returned on failure
