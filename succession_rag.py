"""Succession RAG — retrieval-augmented Q&A over sanitized knowledge cards.

Loads all config/safe_training_*.jsonl records via SanitizedIndex,
embeds the knowledge_card questions via Gemini gemini-embedding-001,
and answers user queries by:

1. Embedding the query
2. Cosine top-K retrieval against the KC index
3. Building a grounded prompt (system instructions + top-K evidence)
4. Calling the answer LLM (gpt-5-nano via Responses API) and returning
   the answer plus the source KCs for citation.

Design goals:
- Evidence-grounded: every answer cites the KCs it came from.
- Incremental: new sanitized sessions are picked up by rebuilding the index
  (cheap: embedding-only, no training).
- Cost-minimal: gemini-embedding-001 + gpt-5-nano ≈ $0.70-1.00 / month
  for ~50 queries/day.
- Fail-safe: if the LLM call fails, still return the retrieved KCs so the
  user can read the evidence directly.

Persistence:
- Index is saved to config/succession_index.npz (embeddings)
  and config/succession_index.json (metadata). Reloaded on process start.
- Metadata includes a content fingerprint so rebuilds are idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Allow the viewer's exporter module to be imported for SanitizedIndex reuse.
_VIEWER_DIR = Path(__file__).resolve().parent / "omi-recordings-viewer"
if str(_VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(_VIEWER_DIR))

from exporter import SanitizedIndex  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class KCSource:
    """A single sanitized knowledge card with session metadata."""

    session_id: str
    question: str
    answer: str
    evidence: list[str]
    confidence: str
    similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuccessionAnswer:
    """Response to a succession query."""

    query: str
    answer: str
    sources: list[KCSource]
    model: str
    elapsed_sec: float
    index_size: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "model": self.model,
            "elapsed_sec": self.elapsed_sec,
            "index_size": self.index_size,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
あなたは事業経営者の過去の発言・判断を記録した「事業継承ナレッジベース」を
参照して質問に答えるアシスタントです。

## 回答ルール

1. **提供された KC (Knowledge Card) の内容のみ**に基づいて答えてください。
   外部知識や一般論で補完しないこと。
2. KC に直接的な答えがない場合は「この質問に対応する記録はありません」と
   率直に答え、一般論で埋めないでください。
3. 複数の KC を組み合わせて回答してもよいが、**どの KC の情報を使ったか**を
   回答の末尾で [#1] [#2] のように参照してください。
4. 本人の語り口・判断パターンを尊重してください。KC の answer に書かれている
   表現・価値観をそのまま継承すること。
5. 回答は簡潔に (2〜5 文)。冗長な前置きは不要。
6. **推測で語らない**: 「たぶん」「おそらく」で始まる文章は書かない。

## KC の参照形式

入力として以下の形式で KC が与えられます:

```
[#1] (セッション: 2026-03-28/10-00-00, confidence: high)
Q: ...
A: ...
(根拠: ...)
```

## 注意

ユーザーの質問が KC の範囲外の場合、無理に答えず次のように返答してください:
「この質問に関する記録はありません。関連する過去の発言を挙げるなら [#1] [#2] です。」
"""


def _build_user_prompt(query: str, sources: list[KCSource]) -> str:
    """Build the user message containing the query and top-K KC context."""
    parts = [f"## 質問\n{query}\n\n## 参照可能な KC"]
    for i, s in enumerate(sources, 1):
        block = [
            f"\n[#{i}] (セッション: {s.session_id}, confidence: {s.confidence}, "
            f"類似度: {s.similarity:.3f})",
            f"Q: {s.question}",
            f"A: {s.answer}",
        ]
        if s.evidence:
            block.append("根拠:")
            for ev in s.evidence[:3]:
                block.append(f"  - {ev}")
        parts.append("\n".join(block))
    parts.append("\n\n## 回答\n上記ルールに従い、参照可能な KC のみを使って回答してください。")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SuccessionRAG
# ---------------------------------------------------------------------------


class SuccessionRAG:
    """Retrieval-augmented Q&A over sanitized knowledge cards.

    Thread-safety: build_index() and ask() each acquire an internal lock.
    Index data is swapped atomically so readers never see a partially
    updated state.
    """

    def __init__(
        self,
        config_dir: str | Path,
        gemini_api_key: str,
        openai_api_key: str,
        embed_model: str = "gemini-embedding-001",
        answer_model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        max_output_tokens: int = 800,
        timeout_sec: float = 60.0,
    ) -> None:
        self._config_dir = Path(config_dir)
        self._gemini_key = gemini_api_key
        self._openai_key = openai_api_key
        self._embed_model = embed_model
        self._answer_model = answer_model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._timeout_sec = timeout_sec

        # Index state
        self._embeddings: Optional[np.ndarray] = None  # (N, D), L2-normalized
        self._sources: list[KCSource] = []
        self._fingerprint: str = ""

        # Lazy clients
        self._genai_client = None
        self._openai_client = None

        # Persistent files
        self._npz_path = self._config_dir / "succession_index.npz"
        self._meta_path = self._config_dir / "succession_index.json"

        import threading
        self._lock = threading.Lock()

        self._enabled = bool(gemini_api_key and openai_api_key)
        if not self._enabled:
            log.warning(
                "SuccessionRAG disabled: missing GEMINI_API_KEY or OPENAI_API_KEY"
            )

    # -- lifecycle -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def index_size(self) -> int:
        return len(self._sources)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "index_size": self.index_size,
            "embed_model": self._embed_model,
            "answer_model": self._answer_model,
            "fingerprint": self._fingerprint,
            "index_path": str(self._npz_path),
            "ready": self._embeddings is not None and len(self._sources) > 0,
        }

    # -- client helpers ------------------------------------------------------

    def _get_genai_client(self):
        if self._genai_client is None:
            from google import genai
            self._genai_client = genai.Client(api_key=self._gemini_key)
        return self._genai_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._openai_key,
                timeout=self._timeout_sec,
            )
        return self._openai_client

    # -- index build / persistence ------------------------------------------

    @staticmethod
    def _compute_fingerprint(sources: list[KCSource]) -> str:
        """Stable content hash of the (session_id, question, answer) tuples."""
        h = hashlib.sha256()
        for s in sources:
            h.update(s.session_id.encode("utf-8"))
            h.update(b"\x1f")
            h.update(s.question.encode("utf-8"))
            h.update(b"\x1e")
            h.update(s.answer.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _collect_sources(self) -> list[KCSource]:
        """Walk config/safe_training_*.jsonl via SanitizedIndex and flatten
        into one KCSource per knowledge_card."""
        idx = SanitizedIndex(self._config_dir)
        sources: list[KCSource] = []
        seen: set[tuple[str, str]] = set()

        for sid, bucket in idx._by_session.items():  # noqa: SLF001 — internal traversal
            for c in bucket.get("knowledge_cards") or []:
                q = (c.get("question") or "").strip()
                a = (c.get("answer") or "").strip()
                if not q or not a:
                    continue
                key = (sid, q)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(KCSource(
                    session_id=sid,
                    question=q,
                    answer=a,
                    evidence=list(c.get("evidence") or []),
                    confidence=str(c.get("confidence") or "medium"),
                ))
        return sources

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts and return an (N, D) array of L2-normalized vectors."""
        client = self._get_genai_client()
        all_vecs: list[list[float]] = []
        BATCH = 100
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            result = client.models.embed_content(
                model=self._embed_model,
                contents=chunk,
            )
            for e in result.embeddings:
                all_vecs.append(list(e.values))
            log.info(
                "SuccessionRAG: embedded %d / %d",
                min(i + BATCH, len(texts)),
                len(texts),
            )
        arr = np.asarray(all_vecs, dtype=np.float32)
        # L2-normalize so cosine == dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    def build_index(self, force: bool = False) -> dict[str, Any]:
        """Collect sources + embed + persist. Returns stats."""
        if not self._enabled:
            return {"status": "disabled"}

        t0 = time.time()
        sources = self._collect_sources()
        if not sources:
            with self._lock:
                self._sources = []
                self._embeddings = None
                self._fingerprint = ""
            return {"status": "empty", "size": 0}

        new_fp = self._compute_fingerprint(sources)
        if not force and new_fp == self._fingerprint and self._embeddings is not None:
            return {
                "status": "unchanged",
                "size": len(sources),
                "fingerprint": new_fp,
                "elapsed_sec": 0.0,
            }

        log.info("SuccessionRAG: building index for %d KCs", len(sources))
        questions = [s.question for s in sources]
        embeddings = self._embed_batch(questions)

        with self._lock:
            self._sources = sources
            self._embeddings = embeddings
            self._fingerprint = new_fp

        self._save()
        elapsed = time.time() - t0
        log.info(
            "SuccessionRAG: index built (%d KCs, dim=%d, %.2fs)",
            len(sources), embeddings.shape[1], elapsed,
        )
        return {
            "status": "built",
            "size": len(sources),
            "dim": int(embeddings.shape[1]),
            "fingerprint": new_fp,
            "elapsed_sec": elapsed,
        }

    def _save(self) -> None:
        """Persist embeddings + metadata to disk."""
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            if self._embeddings is not None:
                np.savez_compressed(self._npz_path, embeddings=self._embeddings)
            meta = {
                "fingerprint": self._fingerprint,
                "embed_model": self._embed_model,
                "sources": [s.to_dict() for s in self._sources],
            }
            self._meta_path.write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:  # pragma: no cover — log-and-continue
            log.error("SuccessionRAG: failed to persist index: %s", e)

    def load_index(self) -> bool:
        """Load persisted index if available. Returns True on success."""
        if not (self._npz_path.exists() and self._meta_path.exists()):
            return False
        try:
            data = np.load(self._npz_path)
            embeddings = data["embeddings"]
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            sources = [
                KCSource(**{k: v for k, v in s.items() if k != "similarity"})
                for s in meta.get("sources", [])
            ]
            if len(sources) != embeddings.shape[0]:
                log.warning(
                    "SuccessionRAG: index mismatch (%d sources vs %d embeddings), rebuild required",
                    len(sources), embeddings.shape[0],
                )
                return False
            with self._lock:
                self._sources = sources
                self._embeddings = embeddings.astype(np.float32)
                self._fingerprint = meta.get("fingerprint", "")
            log.info("SuccessionRAG: loaded %d KCs from %s", len(sources), self._npz_path)
            return True
        except Exception as e:
            log.error("SuccessionRAG: failed to load index: %s", e)
            return False

    # -- retrieval -----------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[KCSource]:
        """Top-K cosine retrieval. Returns empty list if index is empty."""
        if self._embeddings is None or not self._sources:
            return []
        if not query.strip():
            return []

        q_vec = self._embed_batch([query])[0]  # (D,)
        sims = self._embeddings @ q_vec  # (N,)
        k = min(k, len(sims))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[KCSource] = []
        for i in top_idx:
            src = self._sources[int(i)]
            # Return a copy with similarity attached
            results.append(KCSource(
                session_id=src.session_id,
                question=src.question,
                answer=src.answer,
                evidence=list(src.evidence),
                confidence=src.confidence,
                similarity=float(sims[int(i)]),
            ))
        return results

    # -- answer --------------------------------------------------------------

    def ask(self, query: str, k: int = 5) -> SuccessionAnswer:
        """Run the full RAG pipeline and return an answer + sources."""
        t0 = time.time()
        warnings: list[str] = []

        if not self._enabled:
            return SuccessionAnswer(
                query=query,
                answer="SuccessionRAG が無効です (API キー未設定)。",
                sources=[],
                model=self._answer_model,
                elapsed_sec=0.0,
                index_size=0,
                warnings=["disabled"],
            )

        if self._embeddings is None or not self._sources:
            return SuccessionAnswer(
                query=query,
                answer="インデックスが空です。先に /succession/rebuild を実行してください。",
                sources=[],
                model=self._answer_model,
                elapsed_sec=0.0,
                index_size=0,
                warnings=["empty_index"],
            )

        sources = self.retrieve(query, k=k)
        if not sources:
            return SuccessionAnswer(
                query=query,
                answer="関連する KC が見つかりませんでした。",
                sources=[],
                model=self._answer_model,
                elapsed_sec=time.time() - t0,
                index_size=self.index_size,
                warnings=["no_sources"],
            )

        user_prompt = _build_user_prompt(query, sources)

        try:
            client = self._get_openai_client()
            response = client.responses.create(
                model=self._answer_model,
                instructions=_SYSTEM_PROMPT,
                input=[{"role": "user", "content": user_prompt}],
                reasoning={"effort": self._reasoning_effort},
                max_output_tokens=self._max_output_tokens,
                store=False,
            )
        except Exception as e:
            log.error("SuccessionRAG: LLM call failed: %s", e)
            return SuccessionAnswer(
                query=query,
                answer=(
                    "LLM 呼び出しに失敗しました。以下の関連 KC を直接ご確認ください。"
                ),
                sources=sources,
                model=self._answer_model,
                elapsed_sec=time.time() - t0,
                index_size=self.index_size,
                warnings=[f"llm_error:{type(e).__name__}"],
            )

        if response.status == "incomplete":
            reason = ""
            if response.incomplete_details:
                reason = getattr(response.incomplete_details, "reason", "unknown")
            warnings.append(f"incomplete:{reason}")

        answer_text = (response.output_text or "").strip()
        if not answer_text:
            answer_text = "(回答が空でした。以下の関連 KC をご確認ください。)"
            warnings.append("empty_answer")

        return SuccessionAnswer(
            query=query,
            answer=answer_text,
            sources=sources,
            model=self._answer_model,
            elapsed_sec=time.time() - t0,
            index_size=self.index_size,
            warnings=warnings,
        )
