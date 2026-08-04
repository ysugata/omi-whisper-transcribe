#!/usr/bin/env python3
"""Build/update the Karpathy-style LLM Wiki under ~/knowledge/.

Implements Phase 2 of the 事業継承 AI roadmap: compile sanitized
knowledge cards into an atomic-topic markdown wiki maintained by
gpt-5-nano.

Usage:
    # Initial full build (reads all safe_training_*.jsonl):
    GEMINI_API_KEY=xxx OPENAI_API_KEY=xxx \\
        python tools/build_wiki.py --mode full --output ~/knowledge

    # Incremental update (only new KCs since last build):
    python tools/build_wiki.py --mode incremental --output ~/knowledge

    # Preview what would change without writing anything:
    python tools/build_wiki.py --mode incremental --dry-run

    # Limit to a single domain (useful for debugging prompts):
    python tools/build_wiki.py --mode full --domain 物販 --dry-run

Architecture:
    Stage 1 — load KCs via SanitizedIndex
    Stage 2 — cross-reference recordings/<session>/transcript.json for
              tags.domains, tags.entities, tags.decision_memo
    Stage 3 — group by domain (using tag_taxonomy.json as scaffolding)
    Stage 4 — LLM topic clustering within each domain
    Stage 5 — markdown generation (topics/, decisions/, journal/, index.md)
    Stage 6 — idempotent merge into ~/knowledge/

State is tracked in <output>/.build_state.json so incremental runs
only process new sessions / changed KCs.

This is a HOST-SIDE tool; it writes directly to ~/knowledge/. Docker
containers mount the wiki read-only, so the bridge must never invoke
this script — only the host crontab should.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Make the viewer's SanitizedIndex importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "omi-recordings-viewer"))

from exporter import SanitizedIndex, kc_passes_quality  # noqa: E402

log = logging.getLogger("build_wiki")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_taxonomy(config_dir: Path) -> dict:
    """Load tag_taxonomy.json; fall back to minimal defaults if missing."""
    path = config_dir / "tag_taxonomy.json"
    if not path.exists():
        log.warning("tag_taxonomy.json not found at %s, using defaults", path)
        return {
            "domains": ["AI", "経営", "個人"],
            "domain_descriptions": {},
            "keyword_anchors": {},
        }
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load taxonomy: %s", e)
        return {"domains": [], "domain_descriptions": {}, "keyword_anchors": {}}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EnrichedKC:
    """A knowledge card enriched with transcript-level metadata.

    SanitizedIndex gives us (session_id, question, answer, evidence,
    confidence). We cross-reference transcript.json to attach domains,
    entities, decision_memo, occurred_at, and session summary — which
    we need for topic clustering and for frontmatter provenance.
    """

    session_id: str
    question: str
    answer: str
    evidence: list[str]
    confidence: str
    context: str = ""              # from succession_index if available
    # From transcript.json
    domains: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    decision_memo: Optional[str] = None
    outcomes: list[str] = field(default_factory=list)
    conversation_type: str = ""
    importance: str = ""
    session_summary: str = ""
    occurred_at: Optional[str] = None  # ISO date of the session

    def primary_domain(self) -> str:
        """Return the first non-personal domain, or '個人' if only personal,
        or '未分類' if empty."""
        if not self.domains:
            return "未分類"
        non_personal = [d for d in self.domains if d != "個人"]
        return non_personal[0] if non_personal else "個人"


# ---------------------------------------------------------------------------
# KC loading + transcript cross-reference
# ---------------------------------------------------------------------------


def load_enriched_kcs(
    config_dir: Path,
    recordings_dir: Path,
) -> list[EnrichedKC]:
    """Load all sanitized KCs and enrich them with transcript.json tags.

    Missing transcript.json (e.g. recording deleted after sanitize) is
    a warning, not an error — the KC is still usable, just without
    domain hints.
    """
    idx = SanitizedIndex(config_dir)
    log.info("SanitizedIndex: %d sessions loaded", len(idx))

    # Also load succession_index.json for contexts (if it exists)
    contexts: dict[tuple[str, str], str] = {}
    si_path = config_dir / "succession_index.json"
    if si_path.exists():
        try:
            with open(si_path, encoding="utf-8") as f:
                si = json.load(f)
            for s in si.get("sources", []):
                key = (s.get("session_id", ""), s.get("question", ""))
                ctx = s.get("context", "")
                if ctx:
                    contexts[key] = ctx
            log.info("Loaded %d contexts from succession_index.json", len(contexts))
        except Exception as e:
            log.warning("Failed to load succession_index.json: %s", e)

    enriched: list[EnrichedKC] = []
    missing_transcript = 0

    for sid, bucket in idx._by_session.items():  # noqa: SLF001
        cards = bucket.get("knowledge_cards") or []
        # KC 品質ゲート: kc_judge の採点が閾値未満のカードは Wiki に載せない
        cards = [c for c in cards if kc_passes_quality(c)]
        if not cards:
            continue

        # Derive transcript.json path from session_id "YYYY-MM-DD/HH-MM-SS"
        try:
            date_part, time_part = sid.split("/", 1)
        except ValueError:
            log.warning("Skipping malformed session_id: %s", sid)
            continue
        transcript_path = recordings_dir / date_part / time_part / "transcript.json"

        # Defaults if transcript missing
        domains: list[str] = []
        entities: list[str] = []
        decision_memo: Optional[str] = None
        outcomes: list[str] = []
        conversation_type = ""
        importance = ""
        occurred_at = None

        if transcript_path.exists():
            try:
                with open(transcript_path, encoding="utf-8") as f:
                    transcript = json.load(f)
                session_meta = transcript.get("session", {})
                tags = session_meta.get("tags", {})
                domains = list(tags.get("domains") or [])
                entities = list(tags.get("entities") or [])
                decision_memo = tags.get("decision_memo") or None
                outcomes = list(tags.get("outcomes") or [])
                conversation_type = str(tags.get("conversation_type") or "")
                importance = str(tags.get("importance") or "")
                occurred_at = session_meta.get("start")
            except Exception as e:
                log.warning("Failed to read %s: %s", transcript_path, e)
                missing_transcript += 1
        else:
            missing_transcript += 1

        session_summary = bucket.get("summary_output") or ""

        for c in cards:
            q = (c.get("question") or "").strip()
            a = (c.get("answer") or "").strip()
            if not q or not a:
                continue
            enriched.append(EnrichedKC(
                session_id=sid,
                question=q,
                answer=a,
                evidence=list(c.get("evidence") or []),
                confidence=str(c.get("confidence") or "medium"),
                context=contexts.get((sid, q), ""),
                domains=domains,
                entities=entities,
                decision_memo=decision_memo,
                outcomes=outcomes,
                conversation_type=conversation_type,
                importance=importance,
                session_summary=session_summary,
                occurred_at=occurred_at,
            ))

    if missing_transcript:
        log.warning(
            "%d sessions had no readable transcript.json "
            "(KCs still loaded, domains will be empty)",
            missing_transcript,
        )

    log.info("Enriched KCs: %d", len(enriched))
    return enriched


# ---------------------------------------------------------------------------
# Build state management
# ---------------------------------------------------------------------------


@dataclass
class BuildState:
    """Persistent state in <output>/.build_state.json.

    Tracks which KCs have already been processed so incremental mode
    only touches new ones.
    """

    last_build_at: Optional[str] = None
    last_build_mode: str = ""
    processed_keys: list[list[str]] = field(default_factory=list)
    # topic_assignments: key = [session_id, question], value = topic slug
    topic_assignments: dict[str, str] = field(default_factory=dict)

    def processed_set(self) -> set[tuple[str, str]]:
        return {tuple(k) for k in self.processed_keys if len(k) == 2}

    @classmethod
    def load(cls, path: Path) -> "BuildState":
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                last_build_at=data.get("last_build_at"),
                last_build_mode=data.get("last_build_mode", ""),
                processed_keys=data.get("processed_keys", []),
                topic_assignments=data.get("topic_assignments", {}),
            )
        except Exception as e:
            log.warning("Failed to load build state from %s: %s", path, e)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "last_build_at": self.last_build_at,
                "last_build_mode": self.last_build_mode,
                "processed_keys": self.processed_keys,
                "topic_assignments": self.topic_assignments,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------


def _domain_breakdown(kcs: list[EnrichedKC]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for k in kcs:
        counts[k.primary_domain()] = counts.get(k.primary_domain(), 0) + 1
    return counts


def _report_dry_run(
    all_kcs: list[EnrichedKC],
    new_kcs: list[EnrichedKC],
    state: BuildState,
    taxonomy: dict,
) -> None:
    """Print a preview of what the build would do."""
    print("=" * 70)
    print("Build Wiki — Dry Run")
    print("=" * 70)
    print()
    print(f"Total KCs in sanitized data: {len(all_kcs)}")
    print(f"Already processed:           {len(state.processed_set())}")
    print(f"New KCs to process:          {len(new_kcs)}")
    print()

    if not new_kcs:
        print("No new KCs to process. Wiki is up to date.")
        return

    print("Domain breakdown of new KCs:")
    counts = _domain_breakdown(new_kcs)
    total = sum(counts.values()) or 1
    known_domains = set(taxonomy.get("domains", []))
    for domain, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = n / total * 100
        marker = " " if domain in known_domains else "*"
        print(f"  {marker} {domain:20s} {n:4d}  ({pct:5.1f}%)")
    if any(k.primary_domain() not in known_domains for k in new_kcs):
        print()
        print("  * = domain not in tag_taxonomy.json (will go under 未分類/)")
    print()

    # Show some sample KCs per domain
    by_domain: dict[str, list[EnrichedKC]] = {}
    for k in new_kcs:
        by_domain.setdefault(k.primary_domain(), []).append(k)

    print("Sample KCs (max 2 per domain):")
    for domain in sorted(by_domain.keys()):
        samples = by_domain[domain][:2]
        print(f"\n  [{domain}]")
        for s in samples:
            ctx = f" — {s.context[:60]}" if s.context else ""
            print(f"    • ({s.session_id}) {s.question[:60]}{ctx}")
            if s.decision_memo:
                print(f"      decision: {s.decision_memo[:70]}")

    print()
    print("Decision memos present:")
    with_decisions = [k for k in new_kcs if k.decision_memo]
    print(f"  {len(with_decisions)} / {len(new_kcs)} KCs have a decision_memo")
    print()

    print("Estimated cost (rough):")
    n_topics_est = max(len(new_kcs) // 5, 10)  # ~5 KCs per topic
    # topic clustering + markdown gen, each ~500 input + 400 output tokens
    cost_estimate = (n_topics_est * 0.001) + (len(new_kcs) * 0.0002)
    print(f"  ~{n_topics_est} new topics, ~${cost_estimate:.2f} in LLM calls")
    print()
    print("=" * 70)
    print("Stage B/C/D (LLM clustering, markdown gen, merge) not yet implemented.")
    print("Run without --dry-run once implementation lands to actually build.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Karpathy-style LLM wiki from sanitized KCs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config-dir",
        default=str(_REPO_ROOT / "config"),
        help="Directory with safe_training_*.jsonl and tag_taxonomy.json",
    )
    parser.add_argument(
        "--recordings-dir",
        default=str(_REPO_ROOT / "recordings"),
        help="Directory with <date>/<time>/transcript.json (for domain lookup)",
    )
    parser.add_argument(
        "--output",
        default=str(Path.home() / "knowledge"),
        help="Wiki output directory (default: ~/knowledge)",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        default="incremental",
        help="full = reprocess everything, incremental = only new KCs",
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Only process a single domain (debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview only; do not write files or call the LLM",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=10.0,
        help="Abort if estimated LLM cost exceeds this (USD)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    config_dir = Path(args.config_dir).resolve()
    recordings_dir = Path(args.recordings_dir).resolve()
    output_dir = Path(args.output).resolve()

    if not args.dry_run:
        # Real writes require these keys; dry-run does not
        if not os.environ.get("OPENAI_API_KEY"):
            log.error("OPENAI_API_KEY required (or use --dry-run)")
            return 2
        if not os.environ.get("GEMINI_API_KEY"):
            log.warning("GEMINI_API_KEY not set (embeddings may fail)")

    # Load taxonomy (used as scaffolding for topic clustering)
    taxonomy = _load_taxonomy(config_dir)
    log.info(
        "Loaded taxonomy: %d domains, %d descriptions, %d keyword anchors",
        len(taxonomy.get("domains", [])),
        len(taxonomy.get("domain_descriptions", {})),
        len(taxonomy.get("keyword_anchors", {})),
    )

    # Load KCs
    all_kcs = load_enriched_kcs(config_dir, recordings_dir)
    if not all_kcs:
        log.warning("No enriched KCs found; nothing to build")
        return 0

    # Filter by domain if requested
    if args.domain:
        before = len(all_kcs)
        all_kcs = [k for k in all_kcs if args.domain in k.domains]
        log.info("Filtered to domain=%s: %d → %d KCs", args.domain, before, len(all_kcs))

    # Load / compute new-KC delta
    state_path = output_dir / ".build_state.json"
    state = BuildState.load(state_path)

    if args.mode == "full":
        new_kcs = all_kcs
    else:
        processed = state.processed_set()
        new_kcs = [
            k for k in all_kcs
            if (k.session_id, k.question) not in processed
        ]

    log.info(
        "Mode=%s: %d total, %d new (vs %d already processed)",
        args.mode, len(all_kcs), len(new_kcs), len(state.processed_set()),
    )

    if args.dry_run:
        _report_dry_run(all_kcs, new_kcs, state, taxonomy)
        return 0

    if not new_kcs:
        log.info("No new KCs to process. Wiki is up to date.")
        return 0

    # ---- Stage B: cluster -------------------------------------------------
    from wiki_clustering import (
        LLMClient,
        build_topic_buckets,
        cluster_kcs_by_domain,
        generate_topic_markdown,
        write_topic_file,
    )

    llm = LLMClient(api_key=os.environ["OPENAI_API_KEY"])
    t_cluster = time.time()
    assignments = cluster_kcs_by_domain(
        enriched_kcs=new_kcs,
        taxonomy=taxonomy,
        llm=llm,
    )
    log.info(
        "Stage B: %d assignments produced in %.1fs",
        len(assignments), time.time() - t_cluster,
    )

    if not assignments:
        log.error("Stage B produced no assignments; aborting")
        return 4

    buckets = build_topic_buckets(assignments, new_kcs)
    log.info("Stage B: %d topic buckets built", len(buckets))

    # ---- Stage C: generate markdown --------------------------------------
    t_gen = time.time()
    created, updated, skipped = 0, 0, 0
    errors: list[str] = []
    for i, bucket in enumerate(buckets, start=1):
        try:
            body = generate_topic_markdown(bucket, llm)
        except Exception as e:
            errors.append(f"{bucket.domain}/{bucket.topic_slug}: {e}")
            log.error(
                "  [%d/%d] %s/%s generation failed: %s",
                i, len(buckets), bucket.domain, bucket.topic_slug, e,
            )
            continue
        try:
            path, action = write_topic_file(bucket, body, output_dir)
        except Exception as e:
            errors.append(f"write {bucket.domain}/{bucket.topic_slug}: {e}")
            log.error("  [%d/%d] write failed: %s", i, len(buckets), e)
            continue
        if action == "created":
            created += 1
        elif action == "updated":
            updated += 1
        elif action == "skipped":
            skipped += 1
        if i % 10 == 0 or i == len(buckets):
            log.info(
                "  [%d/%d] topics written (created=%d updated=%d skipped=%d)",
                i, len(buckets), created, updated, skipped,
            )
    log.info(
        "Stage C: %d created, %d updated, %d skipped, %d errors (%.1fs)",
        created, updated, skipped, len(errors), time.time() - t_gen,
    )

    # ---- Stage D: update build state -------------------------------------
    # Mark processed KCs so incremental runs skip them next time.
    processed_keys_list = list(state.processed_set())
    for k in new_kcs:
        key = (k.session_id, k.question)
        if key not in state.processed_set():
            processed_keys_list.append(key)
    state.processed_keys = [list(k) for k in processed_keys_list]

    # Record topic assignments for auditing / future incremental routing
    for a in assignments:
        key_str = f"{a.kc_key[0]}||{a.kc_key[1]}"
        state.topic_assignments[key_str] = f"{a.domain}/{a.topic_slug}"

    state.last_build_at = datetime.now().isoformat()
    state.last_build_mode = args.mode
    state.save(state_path)
    log.info("Build state saved to %s", state_path)

    if errors:
        log.warning("Build completed with %d errors:", len(errors))
        for e in errors[:10]:
            log.warning("  %s", e)
        return 1

    log.info("Build complete: %d new topic buckets", len(buckets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
