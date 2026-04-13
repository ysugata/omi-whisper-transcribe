"""Unit tests for succession_rag.py with Contextual Retrieval + hybrid fusion."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from succession_rag import (
    KCSource,
    SuccessionAnswer,
    SuccessionRAG,
    _build_user_prompt,
    _tokenize_cjk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sanitized_file(tmp_path: Path, sessions: list[dict]) -> Path:
    p = tmp_path / "safe_training_test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in sessions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def _make_session(session_id: str, cards: list[dict], summary: str = "") -> dict:
    return {
        "session_id": session_id,
        "risk": "low",
        "summary": {"input": "", "output": summary or f"{session_id} summary"},
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


def _fake_embed_batch(self: SuccessionRAG, texts: list[str]) -> np.ndarray:
    """Deterministic 8-dim hash-based embeddings for tests."""
    vecs = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.standard_normal(8).astype(np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        vecs.append(v)
    return np.stack(vecs) if vecs else np.zeros((0, 8), dtype=np.float32)


def _fake_contextualize(
    self: SuccessionRAG,
    sources,
    session_summaries,
    reuse=None,
) -> None:
    """No-op contextualize: set a trivial context derived from the question."""
    reuse = reuse or {}
    for s in sources:
        cached = reuse.get((s.session_id, s.question))
        s.context = cached if cached else f"ctx:{s.question[:20]}"


# ---------------------------------------------------------------------------
# _tokenize_cjk
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_ascii_words(self):
        toks = _tokenize_cjk("FBA と Keepa API")
        assert "fba" in toks
        assert "keepa" in toks
        assert "api" in toks
        assert "と" in toks

    def test_japanese_bigrams(self):
        toks = _tokenize_cjk("物販方針")
        # unigrams
        assert "物" in toks
        assert "販" in toks
        assert "方" in toks
        assert "針" in toks
        # bigrams
        assert "物販" in toks
        assert "販方" in toks
        assert "方針" in toks

    def test_empty(self):
        assert _tokenize_cjk("") == []

    def test_mixed(self):
        toks = _tokenize_cjk("FBA と自己配送の使い分け")
        assert "fba" in toks
        assert "自己" in toks
        assert "配送" in toks


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_includes_context_and_sources(self):
        sources = [
            KCSource(
                session_id="sid1",
                question="FBAと自己配送の方針は?",
                answer="FBA統一+自己配送併用。",
                evidence=["FBA議論", "方針決定"],
                confidence="high",
                context="物販事業における配送チャネル判断",
                similarity=0.95,
                rrf_score=0.032,
            ),
        ]
        out = _build_user_prompt("物販の方針を教えて", sources)
        assert "物販の方針を教えて" in out
        assert "[#1]" in out
        assert "sid1" in out
        assert "FBAと自己配送" in out
        assert "根拠:" in out
        assert "コンテキスト: 物販事業における配送チャネル判断" in out
        assert "rrf: 0.032" in out


# ---------------------------------------------------------------------------
# SuccessionRAG — build / retrieve
# ---------------------------------------------------------------------------


class TestBuildAndRetrieve:
    def test_build_empty_dir(self, tmp_path):
        rag = SuccessionRAG(tmp_path, "gkey", "okey")
        result = rag.build_index()
        assert result["status"] == "empty"
        assert rag.index_size == 0

    def test_build_from_sanitized_jsonl(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [
                _kc("FBAと自己配送の方針は具体的にどう決めたか?",
                    "FBA統一+自己配送併用で利益最大化を図る方針。"),
                _kc("Keepa APIで何を分析する設計にした?",
                    "波形で将来の購買機会を見込む基準作り。"),
            ], summary="物販 FBA 方針と Keepa API 設計の議論"),
            _make_session("s2", [
                _kc("Claude Codeのインストール方針はどう決めた?",
                    "Nano Banana経由で隔離して進める方針。"),
            ], summary="Claude Code 開発環境の議論"),
        ])

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            result = rag.build_index()

        assert result["status"] == "built"
        assert result["size"] == 3
        assert result["retrieval_mode"] == "hybrid_rrf"
        assert rag.index_size == 3
        assert rag._bm25 is not None

    def test_contexts_populated(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 ある程度長い具体的な質問文",
                                      "A1 十分な長さの具体的な回答です。")])
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
        assert rag._sources[0].context.startswith("ctx:")

    def test_persist_and_reload(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問文章",
                                      "A1 具体的な回答内容が入る長めの答え")])
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag1 = SuccessionRAG(tmp_path, "gkey", "okey")
            rag1.build_index()

            rag2 = SuccessionRAG(tmp_path, "gkey", "okey")
            assert rag2.load_index() is True
            assert rag2.index_size == 1
            assert rag2._sources[0].context == rag1._sources[0].context
            # BM25 rebuilt on load
            assert rag2._bm25 is not None

    def test_retrieve_hybrid_returns_sources(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [
                _kc("物販のFBA方針について具体的にどう決めた?",
                    "FBA統一+自己配送併用の方針で進める。"),
                _kc("Keepa APIで何を分析する設計か?",
                    "波形分析で将来の購買機会を予測する設計。"),
                _kc("Claude Code導入の判断基準は何か?",
                    "隔離環境で安全に実験できる点を重視。"),
            ]),
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            results = rag.retrieve("物販 FBA の方針", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, KCSource) for r in results)
        # All results should have rrf_score set
        assert all(r.rrf_score > 0 for r in results)

    def test_retrieve_bm25_finds_keyword_match(self, tmp_path):
        """BM25 should surface the FBA document when the query mentions FBA,
        even with hash-random dense embeddings that would otherwise miss it.
        """
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [
                _kc("物販のFBA方針について具体的にどう決めた?",
                    "FBA統一+自己配送併用の方針で進める。"),
                _kc("メキシコシティのメトロポリタン大聖堂について",
                    "世界遺産として登録されている歴史地区。"),
                _kc("Claude Code のインストール先は Mac mini?",
                    "Nano Banana 経由で隔離してインストールする。"),
                _kc("イスタンブールのボスポラス海峡は?",
                    "ヨーロッパとアジアを隔てる海峡として有名。"),
                _kc("スポーツ栄養士の課題は何?",
                    "認知度不足と売上向上が課題。"),
                _kc("Keepa API で何を分析する設計?",
                    "波形で将来の購買機会を見込む基準を作る。"),
            ]),
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            results = rag.retrieve("FBA 方針", k=3)

        assert len(results) == 3
        # The FBA document should be among the top 3 via BM25 (query keyword match)
        questions = [r.question for r in results]
        assert any("FBA" in q for q in questions), (
            f"Expected FBA doc in top-3, got: {questions}"
        )

    def test_fingerprint_unchanged_skips_rebuild(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問の文章",
                                      "A1 十分な長さの具体的な回答テキスト")])
        ])
        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
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
        ans = rag.ask("何でもいい")
        assert ans.index_size == 0
        assert "empty_index" in ans.warnings

    def test_ask_full_flow_mocked(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問の文章",
                                      "A1 具体的な回答文が入る長めの答え")])
        ])

        fake_response = MagicMock()
        fake_response.status = "completed"
        fake_response.output_text = "モックされた LLM 回答です [#1]"
        fake_response.incomplete_details = None

        fake_client = MagicMock()
        fake_client.responses.create.return_value = fake_response

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            rag._openai_client = fake_client

            ans = rag.ask("Q1 とても具体的な質問の文章", k=1)

        assert ans.answer == "モックされた LLM 回答です [#1]"
        assert len(ans.sources) == 1
        assert ans.retrieval_mode == "hybrid"
        assert ans.sources[0].session_id == "s1"
        fake_client.responses.create.assert_called_once()

    def test_ask_llm_error_returns_sources(self, tmp_path):
        _make_sanitized_file(tmp_path, [
            _make_session("s1", [_kc("Q1 とても具体的な質問の文章",
                                      "A1 具体的な回答文が入る長めの答え")])
        ])

        fake_client = MagicMock()
        fake_client.responses.create.side_effect = RuntimeError("rate limited")

        with patch.object(SuccessionRAG, "_embed_batch", _fake_embed_batch), \
             patch.object(SuccessionRAG, "_contextualize_all", _fake_contextualize):
            rag = SuccessionRAG(tmp_path, "gkey", "okey")
            rag.build_index()
            rag._openai_client = fake_client

            ans = rag.ask("Q1 とても具体的な質問の文章", k=1)

        assert "llm_error" in ans.warnings[0]
        assert len(ans.sources) == 1
