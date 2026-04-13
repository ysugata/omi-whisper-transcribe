#!/usr/bin/env python3
"""CLI: Build the Succession RAG index from sanitized JSONL files.

Standalone wrapper around succession_rag.SuccessionRAG.build_index() —
useful when you want to rebuild the index without going through the
Bridge HTTP API.

Usage:
    python tools/build_succession_index.py \
        --config-dir config \
        --force

Environment:
    GEMINI_API_KEY — required (embedding)
    OPENAI_API_KEY — required (answer model; not used at build time but
                      the class requires it for init)
    SUCCESSION_EMBED_MODEL — optional (default: gemini-embedding-001)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the bridge module importable when run from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from succession_rag import SuccessionRAG  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", default="config",
                   help="Directory containing safe_training_*.jsonl (default: config)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if fingerprint matches")
    p.add_argument("--embed-model",
                   default=os.environ.get("SUCCESSION_EMBED_MODEL", "gemini-embedding-001"))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or "placeholder"
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY is required", file=sys.stderr)
        return 2

    rag = SuccessionRAG(
        config_dir=args.config_dir,
        gemini_api_key=gemini_key,
        openai_api_key=openai_key,
        embed_model=args.embed_model,
    )

    # Attempt to load existing index so fingerprint comparison works
    rag.load_index()
    result = rag.build_index(force=args.force)
    print(result)
    return 0 if result.get("status") in ("built", "unchanged") else 1


if __name__ == "__main__":
    sys.exit(main())
