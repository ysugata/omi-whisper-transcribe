#!/usr/bin/env python3
"""Semantic deduplication of sanitized knowledge cards.

Reads one or more config/safe_training_*.jsonl files, embeds the KC
questions via Gemini text-embedding-004, groups near-duplicates by
cosine similarity, and writes a deduplicated JSONL where each session's
knowledge_cards array contains only the highest-confidence KC from each
cluster.

Usage:
    python tools/dedupe_sanitized.py \
        --input config/safe_training_20260413_113103.jsonl \
        --output config/safe_training_deduped_20260413.jsonl \
        --threshold 0.92

    # Or dedupe across multiple files:
    python tools/dedupe_sanitized.py \
        --input 'config/safe_training_*.jsonl' \
        --output config/safe_training_deduped_all.jsonl

Environment:
    GEMINI_API_KEY — required for the embedding API.
    RAG_EMBED_MODEL — optional (default: gemini-embedding-001).
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("dedupe_sanitized")


CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2, "none": 3}


def _confidence_rank(card: dict) -> int:
    return CONFIDENCE_RANK.get(card.get("confidence", "none"), 4)


def _load_records(paths: list[Path]) -> list[dict]:
    """Load all JSONL records from the given paths."""
    records: list[dict] = []
    for p in paths:
        if not p.exists():
            logger.warning("skip missing: %s", p)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("skip malformed line in %s: %s", p, e)
    return records


def _collect_kcs(records: list[dict]) -> list[tuple[int, int, dict]]:
    """Walk records and yield (record_idx, kc_idx, card) triples.

    Handles both new `knowledge_cards` array format and legacy single
    `knowledge_card` field.
    """
    out: list[tuple[int, int, dict]] = []
    for ri, rec in enumerate(records):
        cards = rec.get("knowledge_cards")
        if isinstance(cards, list) and cards:
            for ki, c in enumerate(cards):
                if isinstance(c, dict) and c.get("question") and c.get("answer"):
                    out.append((ri, ki, c))
            continue
        kc = rec.get("knowledge_card")
        if isinstance(kc, dict) and kc.get("question") and kc.get("answer"):
            out.append((ri, -1, kc))  # -1 sentinel = legacy single-card
    return out


def _embed_questions(questions: list[str], api_key: str, model: str) -> list[list[float]]:
    """Embed questions via Gemini API in batches."""
    from google import genai
    client = genai.Client(api_key=api_key)

    all_embeds: list[list[float]] = []
    # Gemini batch embedding accepts up to 100 items per call
    BATCH = 100
    for i in range(0, len(questions), BATCH):
        chunk = questions[i:i + BATCH]
        result = client.models.embed_content(model=model, contents=chunk)
        for e in result.embeddings:
            all_embeds.append(list(e.values))
        logger.info("embedded %d / %d", min(i + BATCH, len(questions)), len(questions))
    return all_embeds


def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product of unit vectors (caller must normalize)."""
    return sum(x * y for x, y in zip(a, b))


def _normalize(v: list[float]) -> list[float]:
    import math
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def _cluster_kcs(kcs: list[tuple[int, int, dict]],
                  embeddings: list[list[float]],
                  threshold: float) -> list[list[int]]:
    """Single-linkage clustering by cosine similarity threshold.

    Returns list of clusters, where each cluster is a list of indices
    into the kcs/embeddings lists.
    """
    n = len(kcs)
    normed = [_normalize(v) for v in embeddings]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine(normed[i], normed[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _select_representative(cluster: list[int],
                            kcs: list[tuple[int, int, dict]]) -> int:
    """Pick highest-confidence KC from a cluster.

    Ties broken by longer answer (more informative).
    """
    def key(idx: int) -> tuple:
        _, _, card = kcs[idx]
        return (_confidence_rank(card), -len(card.get("answer", "")))
    return min(cluster, key=key)


def _apply_dedup(records: list[dict],
                  kcs: list[tuple[int, int, dict]],
                  clusters: list[list[int]]) -> tuple[list[dict], dict]:
    """Build deduped records by filtering knowledge_cards.

    Returns (deduped_records, stats).
    """
    keep: set[tuple[int, int]] = set()
    for cluster in clusters:
        rep_idx = _select_representative(cluster, kcs)
        ri, ki, _ = kcs[rep_idx]
        keep.add((ri, ki))

    new_records = []
    total_before = 0
    total_after = 0
    for ri, rec in enumerate(records):
        new_rec = dict(rec)
        cards = rec.get("knowledge_cards")
        if isinstance(cards, list) and cards:
            total_before += len(cards)
            filtered = [c for ki, c in enumerate(cards) if (ri, ki) in keep]
            new_rec["knowledge_cards"] = filtered
            total_after += len(filtered)
            # Also update legacy knowledge_card if it was derived from the multi list
            if filtered:
                new_rec["knowledge_card"] = {
                    k: filtered[0][k] for k in ("question", "answer", "evidence", "confidence")
                }
            else:
                new_rec["knowledge_card"] = None
        else:
            kc = rec.get("knowledge_card")
            if isinstance(kc, dict) and kc.get("question"):
                total_before += 1
                if (ri, -1) in keep:
                    total_after += 1
                else:
                    new_rec["knowledge_card"] = None
        new_records.append(new_rec)

    return new_records, {
        "kcs_before": total_before,
        "kcs_after": total_after,
        "clusters": len(clusters),
        "dedup_ratio": (total_before - total_after) / total_before if total_before else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL path or glob pattern")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSONL path")
    parser.add_argument("--threshold", "-t", type=float, default=0.92,
                        help="Cosine similarity threshold (default: 0.92)")
    parser.add_argument("--model", default=os.environ.get("RAG_EMBED_MODEL", "gemini-embedding-001"),
                        help="Gemini embedding model (default: gemini-embedding-001)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats only, do not write output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is required")
        return 2

    input_paths = [Path(p) for p in glob.glob(args.input)]
    if not input_paths:
        logger.error("no files matched input pattern: %s", args.input)
        return 2

    logger.info("loading %d input file(s)", len(input_paths))
    records = _load_records(input_paths)
    logger.info("loaded %d records", len(records))

    kcs = _collect_kcs(records)
    logger.info("collected %d knowledge cards", len(kcs))
    if not kcs:
        logger.warning("no KCs to dedupe; output will be a copy of input")

    if len(kcs) >= 2:
        questions = [c["question"] for _, _, c in kcs]
        logger.info("embedding %d questions via %s", len(questions), args.model)
        embeddings = _embed_questions(questions, api_key, args.model)
        clusters = _cluster_kcs(kcs, embeddings, args.threshold)
        logger.info("clustered into %d groups (threshold=%.2f)", len(clusters), args.threshold)
    else:
        clusters = [[i] for i in range(len(kcs))]

    new_records, stats = _apply_dedup(records, kcs, clusters)
    logger.info(
        "dedup: before=%d, after=%d, removed=%d (%.1f%%), clusters=%d",
        stats["kcs_before"],
        stats["kcs_after"],
        stats["kcs_before"] - stats["kcs_after"],
        stats["dedup_ratio"] * 100,
        stats["clusters"],
    )

    if args.dry_run:
        logger.info("dry-run: not writing output")
        return 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d records to %s", len(new_records), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
