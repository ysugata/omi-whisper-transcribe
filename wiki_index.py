"""WikiIndex — read-only loader for the ~/knowledge wiki.

Walks `<knowledge_dir>/knowledge-{public,private}/topics/**/*.md`,
parses YAML frontmatter to extract `sources[].session_id`, and builds
a session_id → topic reverse index for parent-child retrieval.

Used by SuccessionRAG to look up parent wiki topics for a given KC,
implementing the LangChain Parent Document Retriever pattern (search
fine, return aggregated context).

The bridge container mounts ~/knowledge read-only, so this module
**only reads**. Wiki writes happen on the host via tools/build_wiki.py
or manual edits.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WikiTopic:
    """One topic markdown file from the wiki."""

    path: str                 # relative path under knowledge_dir, e.g. "knowledge-public/topics/物販/FBA-...md"
    visibility: str           # "public" or "private"
    title: str
    domain: str
    body: str                 # markdown body (without frontmatter)
    sources: list[str]        # session_id list from frontmatter
    confidence: str = "medium"
    status: str = "active"
    manual: bool = False
    # Populated during retrieval
    aggregate_score: float = 0.0
    matched_kc_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "visibility": self.visibility,
            "title": self.title,
            "domain": self.domain,
            "body": self.body,
            "sources": list(self.sources),
            "confidence": self.confidence,
            "status": self.status,
            "manual": self.manual,
            "aggregate_score": self.aggregate_score,
            "matched_kc_count": self.matched_kc_count,
        }


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


_FM_DELIM = "---"


def _split_frontmatter(content: str) -> tuple[str, str]:
    """Return (frontmatter_block, body). Empty fm if not present."""
    if not content.startswith(_FM_DELIM + "\n"):
        return "", content
    end = content.find("\n" + _FM_DELIM + "\n", len(_FM_DELIM) + 1)
    if end == -1:
        return "", content
    fm = content[len(_FM_DELIM) + 1:end]
    body = content[end + len(_FM_DELIM) + 2:]
    return fm, body


_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_LIST_ITEM_RE = re.compile(r"^\s*-\s*(.*)$")
_KV_INLINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")


def _parse_yaml_scalar(v: str) -> Any:
    v = v.strip()
    if not v or v == "null":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "[]":
        return []
    # Quoted string
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        try:
            import json
            return json.loads(v)
        except Exception:
            return v[1:-1]
    return v


def _parse_frontmatter(fm_block: str) -> dict[str, Any]:
    """Minimal YAML parser for our hand-rolled frontmatter format.

    Supports:
        key: value
        key: [list, of, values]
        key:
          - item
          - item
        key:
          - sub_key: value
            sub_key2: value
    """
    result: dict[str, Any] = {}
    lines = fm_block.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            m = _KEY_RE.match(line.rstrip())
            if not m:
                i += 1
                continue
            key, value = m.group(1), m.group(2)
            if value == "":
                # Block-style list or dict follows
                items: list = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if not next_line.strip():
                        i += 1
                        continue
                    if not next_line.startswith(" "):
                        break
                    item_match = _LIST_ITEM_RE.match(next_line)
                    if item_match:
                        item_value = item_match.group(1).strip()
                        if ":" in item_value and not item_value.startswith('"'):
                            # Dict item like "- session_id: ..."
                            sub: dict[str, Any] = {}
                            kv = _KV_INLINE_RE.match(item_value)
                            if kv:
                                sub[kv.group(1)] = _parse_yaml_scalar(kv.group(2))
                            # Look for sub-keys on indented continuation lines
                            i += 1
                            while i < len(lines):
                                cont = lines[i]
                                if not cont.strip():
                                    i += 1
                                    continue
                                if not cont.startswith("    "):
                                    break
                                stripped = cont.strip()
                                if stripped.startswith("- "):
                                    break
                                kv2 = _KV_INLINE_RE.match(stripped)
                                if kv2:
                                    sub[kv2.group(1)] = _parse_yaml_scalar(kv2.group(2))
                                    i += 1
                                else:
                                    break
                            items.append(sub)
                        else:
                            items.append(_parse_yaml_scalar(item_value))
                            i += 1
                    else:
                        break
                result[key] = items
            else:
                result[key] = _parse_yaml_scalar(value)
                i += 1
        else:
            i += 1

    return result


def _extract_sources(fm: dict[str, Any]) -> list[str]:
    """Extract session_id list from frontmatter sources field."""
    sources = fm.get("sources") or []
    out: list[str] = []
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, dict):
                sid = s.get("session_id")
                if isinstance(sid, str) and sid:
                    out.append(sid)
            elif isinstance(s, str) and s:
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# WikiIndex
# ---------------------------------------------------------------------------


class WikiIndex:
    """In-memory index over the ~/knowledge wiki for parent-child retrieval.

    Loads all topic markdown files from
        <knowledge_dir>/knowledge-{public,private}/topics/**/*.md
    on demand. Cheap reload via load() — the file count is small
    (typically 50-500 topics) and frontmatter parsing is fast.

    Provides:
        - topics_for_session(session_id) → list[WikiTopic]
        - all_topics() → iterator
        - load() / reload()
    """

    def __init__(self, knowledge_dir: str | Path | None) -> None:
        self._dir = Path(knowledge_dir) if knowledge_dir else None
        self._topics: list[WikiTopic] = []
        self._by_session: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._enabled = self._dir is not None and self._dir.exists()
        if self._dir and not self._enabled:
            log.warning(
                "WikiIndex disabled: knowledge_dir does not exist: %s",
                self._dir,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def topic_count(self) -> int:
        return len(self._topics)

    @property
    def session_count(self) -> int:
        return len(self._by_session)

    def load(self) -> dict[str, Any]:
        """Walk knowledge-public/topics/ and knowledge-private/topics/.

        Returns stats dict with topic_count, session_count, errors.
        """
        if not self._enabled or self._dir is None:
            return {"status": "disabled", "topic_count": 0, "session_count": 0}

        topics: list[WikiTopic] = []
        by_session: dict[str, list[int]] = {}
        errors: list[str] = []

        for visibility in ("public", "private"):
            topics_root = self._dir / f"knowledge-{visibility}" / "topics"
            if not topics_root.exists():
                continue
            for md_path in sorted(topics_root.rglob("*.md")):
                try:
                    topic = self._parse_topic(md_path, visibility)
                except Exception as e:
                    errors.append(f"{md_path.name}: {e}")
                    continue
                if topic is None:
                    continue
                topics.append(topic)
                topic_idx = len(topics) - 1
                for sid in topic.sources:
                    by_session.setdefault(sid, []).append(topic_idx)

        with self._lock:
            self._topics = topics
            self._by_session = by_session

        log.info(
            "WikiIndex: loaded %d topics covering %d sessions (%d errors)",
            len(topics), len(by_session), len(errors),
        )
        return {
            "status": "loaded",
            "topic_count": len(topics),
            "session_count": len(by_session),
            "errors": errors[:10],
        }

    def _parse_topic(self, md_path: Path, visibility: str) -> Optional[WikiTopic]:
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("WikiIndex: cannot read %s: %s", md_path, e)
            return None

        fm_block, body = _split_frontmatter(content)
        fm = _parse_frontmatter(fm_block)
        sources = _extract_sources(fm)
        rel_path = md_path.relative_to(self._dir).as_posix() if self._dir else md_path.name

        title = str(fm.get("title") or md_path.stem).strip()
        domain = str(fm.get("domain") or "未分類").strip()
        confidence = str(fm.get("confidence") or "medium").strip()
        status = str(fm.get("status") or "active").strip()
        manual = bool(fm.get("manual") or False)

        return WikiTopic(
            path=rel_path,
            visibility=visibility,
            title=title,
            domain=domain,
            body=body.strip(),
            sources=sources,
            confidence=confidence,
            status=status,
            manual=manual,
        )

    def topics_for_session(self, session_id: str) -> list[WikiTopic]:
        """Return all wiki topics whose frontmatter sources reference this session."""
        indices = self._by_session.get(session_id, [])
        return [self._topics[i] for i in indices]

    def topic_indices_for_session(self, session_id: str) -> list[int]:
        """Same as topics_for_session but returns indices (cheaper for aggregation)."""
        return list(self._by_session.get(session_id, []))

    def get_topic(self, idx: int) -> Optional[WikiTopic]:
        if 0 <= idx < len(self._topics):
            return self._topics[idx]
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "knowledge_dir": str(self._dir) if self._dir else None,
            "topic_count": len(self._topics),
            "session_count": len(self._by_session),
        }
