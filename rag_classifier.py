"""RAG-based session classifier using embedding similarity and weighted k-NN voting.

Embeds session text via Gemini gemini-embedding-001 and classifies by comparing
against a stored index of previously-tagged sessions.  When the top vote
exceeds a confidence threshold the result is returned directly; otherwise
the caller falls back to the LLM tagger.
"""

import json
import logging
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Axes used for voting
_SINGLE_AXES = ("conversation_type", "importance")
_MULTI_AXES = ("outcomes", "domains")


class RagClassifier:
    """Embedding-based session classifier with k-NN weighted voting."""

    def __init__(
        self,
        api_key: str,
        recordings_dir: str,
        index_dir: str,
        model: str = "gemini-embedding-001",
        k: int = 7,
        confidence_threshold: float = 0.5,
        min_sessions: int = 20,
    ) -> None:
        self._api_key = api_key
        self._recordings_dir = recordings_dir
        self._index_dir = index_dir
        self._model = model
        self._k = k
        self._confidence_threshold = confidence_threshold
        self._min_sessions = min_sessions

        # In-memory index
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._metadata: list[dict] = []  # parallel to rows of _embeddings
        self._lock = threading.Lock()

        # Paths
        self._npz_path = os.path.join(index_dir, "rag_embeddings.npz")
        self._json_path = os.path.join(index_dir, "rag_metadata.json")

        # Lazy-init genai client
        self._genai_client = None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._genai_client is None:
            from google import genai
            self._genai_client = genai.Client(api_key=self._api_key)
        return self._genai_client

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string via Gemini embedding API."""
        client = self._get_client()
        result = client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_texts_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        client = self._get_client()
        result = client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        with self._lock:
            if self._embeddings is not None and len(self._metadata) > 0:
                np.savez_compressed(self._npz_path, embeddings=self._embeddings)
                with open(self._json_path, "w", encoding="utf-8") as f:
                    json.dump(self._metadata, f, ensure_ascii=False)

    def load_index(self) -> None:
        """Load existing index from disk."""
        if not os.path.exists(self._npz_path) or not os.path.exists(self._json_path):
            log.warning("RAG index files not found")
            return
        with self._lock:
            data = np.load(self._npz_path)
            self._embeddings = data["embeddings"]
            with open(self._json_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
        log.info("RAG index loaded: %d sessions", len(self._metadata))

    def has_index(self) -> bool:
        return os.path.exists(self._npz_path) and os.path.exists(self._json_path)

    def session_count(self) -> int:
        with self._lock:
            return len(self._metadata)

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._metadata) >= self._min_sessions

    def get_status(self) -> dict:
        with self._lock:
            count = len(self._metadata)
        return {
            "ready": count >= self._min_sessions,
            "total_sessions": count,
            "min_sessions": self._min_sessions,
            "model": self._model,
            "k": self._k,
            "confidence_threshold": self._confidence_threshold,
        }

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_session(self, session_id: str, embedding: list, tags: dict) -> None:
        """Add a single session to the in-memory index (incremental)."""
        vec = np.array(embedding, dtype=np.float32)
        with self._lock:
            # Skip duplicates
            for m in self._metadata:
                if m["session_id"] == session_id:
                    return
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])
            self._metadata.append({"session_id": session_id, "tags": tags})
        self._save_index()

    def build_index(self) -> dict:
        """Build full index from all tagged sessions on disk.

        Returns {"total": N, "embedded": M, "errors": E}.
        """
        recordings = Path(self._recordings_dir)
        if not recordings.is_dir():
            return {"total": 0, "embedded": 0, "errors": 0}

        # Collect existing session IDs to skip
        with self._lock:
            existing_ids = {m["session_id"] for m in self._metadata}

        sessions_to_embed = []
        for day_dir in sorted(recordings.iterdir()):
            if not day_dir.is_dir():
                continue
            for time_dir in sorted(day_dir.iterdir()):
                if not time_dir.is_dir():
                    continue
                transcript_path = time_dir / "transcript.json"
                if not transcript_path.exists():
                    continue

                session_id = f"{day_dir.name}/{time_dir.name}"
                if session_id in existing_ids:
                    continue

                try:
                    with open(transcript_path, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                if not data.get("session", {}).get("session_tagged"):
                    continue

                tags = data.get("session", {}).get("tags", {})
                if not tags.get("conversation_type"):
                    continue

                segments = data.get("segments", [])
                text = self._extract_plain_text(segments)
                if len(text) < 10:
                    continue

                sessions_to_embed.append({
                    "session_id": session_id,
                    "text": text,
                    "tags": {
                        "conversation_type": tags.get("conversation_type", ""),
                        "outcomes": tags.get("outcomes", []),
                        "domains": tags.get("domains", []),
                        "importance": tags.get("importance", "低"),
                        "quality": tags.get("quality", "clean"),
                    },
                })

        total = len(sessions_to_embed)
        if total == 0:
            log.info("RAG build_index: no new sessions to embed")
            return {"total": 0, "embedded": 0, "errors": 0}

        log.info("RAG build_index: embedding %d sessions...", total)

        embedded = 0
        errors = 0

        # Batch embed in chunks of 100 (API limit)
        batch_size = 100
        for i in range(0, total, batch_size):
            chunk = sessions_to_embed[i:i + batch_size]
            texts = [s["text"] for s in chunk]
            try:
                embeddings = self.embed_texts_batch(texts)
                for j, emb in enumerate(embeddings):
                    sess = chunk[j]
                    self.add_session(sess["session_id"], emb, sess["tags"])
                    embedded += 1
            except Exception as e:
                log.error("RAG embed batch error (offset=%d): %s", i, e)
                # Fall back to individual embedding
                for sess in chunk:
                    try:
                        emb = self.embed_text(sess["text"])
                        self.add_session(sess["session_id"], emb, sess["tags"])
                        embedded += 1
                    except Exception as e2:
                        log.debug("RAG embed error for %s: %s", sess["session_id"], e2)
                        errors += 1

        log.info("RAG build_index done: total=%d, embedded=%d, errors=%d", total, embedded, errors)
        return {"total": total, "embedded": embedded, "errors": errors}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict:
        """Classify text using k-NN weighted voting.

        Returns {"confident": bool, "tags": {...}, "embedding": [...],
                 "top_similarity": float, "vote_detail": {...}}.
        """
        if not self.is_ready():
            return {"confident": False, "tags": {}, "embedding": [],
                    "top_similarity": 0.0, "vote_detail": {}}

        embedding = self.embed_text(text)
        neighbors = self._search_similar(embedding, self._k)

        if not neighbors:
            return {"confident": False, "tags": {}, "embedding": embedding,
                    "top_similarity": 0.0, "vote_detail": {}}

        vote_result = self._weighted_vote(neighbors)
        confident = self._is_confident(vote_result)
        tags = self._votes_to_tags(vote_result)
        top_sim = neighbors[0]["similarity"] if neighbors else 0.0

        return {
            "confident": confident,
            "tags": tags,
            "embedding": embedding,
            "top_similarity": top_sim,
            "vote_detail": {k: dict(v) for k, v in vote_result.items()},
        }

    def _search_similar(self, embedding: list, k: int) -> list[dict]:
        """Find top-k most similar sessions by cosine similarity."""
        query = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        with self._lock:
            if self._embeddings is None or len(self._metadata) == 0:
                return []
            embs = self._embeddings
            meta = list(self._metadata)

        # Cosine similarity: dot(q, e) / (|q| * |e|)
        norms = np.linalg.norm(embs, axis=1)
        # Avoid division by zero
        valid = norms > 0
        similarities = np.zeros(len(embs))
        if valid.any():
            similarities[valid] = embs[valid] @ query / (norms[valid] * query_norm)

        # Top-k indices
        actual_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim <= 0:
                break
            results.append({
                "similarity": sim,
                "session_id": meta[idx]["session_id"],
                "tags": meta[idx]["tags"],
            })
        return results

    def _weighted_vote(self, neighbors: list[dict]) -> dict:
        """Weighted k-NN voting across all axes."""
        votes = {
            "conversation_type": defaultdict(float),
            "importance": defaultdict(float),
            "outcomes": defaultdict(float),
            "domains": defaultdict(float),
            "quality": defaultdict(float),
        }
        for n in neighbors:
            w = n["similarity"]
            tags = n["tags"]
            # Single-select axes
            if tags.get("conversation_type"):
                votes["conversation_type"][tags["conversation_type"]] += w
            if tags.get("importance"):
                votes["importance"][tags["importance"]] += w
            if tags.get("quality"):
                votes["quality"][tags["quality"]] += w
            # Multi-select axes
            for d in tags.get("domains", []):
                votes["domains"][d] += w
            for o in tags.get("outcomes", []):
                votes["outcomes"][o] += w
        return votes

    def _is_confident(self, vote_result: dict) -> bool:
        """Check if single-select axes have a clear winner."""
        for axis in _SINGLE_AXES:
            weights = vote_result.get(axis, {})
            if not weights:
                return False
            total = sum(weights.values())
            top = max(weights.values())
            if total > 0 and (top / total) < self._confidence_threshold:
                return False
        return True

    def _votes_to_tags(self, vote_result: dict) -> dict:
        """Convert vote tallies to final tag values."""
        tags = {}

        # Single-select: pick the highest-weighted option
        for axis in ("conversation_type", "importance", "quality"):
            weights = vote_result.get(axis, {})
            if weights:
                tags[axis] = max(weights, key=weights.get)
            else:
                tags[axis] = "その他" if axis == "conversation_type" else (
                    "低" if axis == "importance" else "clean"
                )

        # Multi-select: pick options above average weight
        for axis in ("outcomes", "domains"):
            weights = vote_result.get(axis, {})
            if not weights:
                tags[axis] = [] if axis == "domains" else ["なし"]
                continue
            avg = sum(weights.values()) / len(weights) if weights else 0
            selected = [k for k, v in sorted(weights.items(), key=lambda x: -x[1]) if v >= avg]
            if axis == "outcomes":
                tags[axis] = selected[:4] if selected else ["なし"]
            else:
                tags[axis] = selected[:3]

        return tags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_plain_text(segments: list[dict]) -> str:
        """Extract plain text from segments (no speaker labels)."""
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
