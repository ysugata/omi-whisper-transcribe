"""LLM topic clustering + markdown generation for build_wiki.

Stage B + C + D of the Phase 2 wiki build pipeline:

  Stage B: group EnrichedKCs into atomic topic buckets within each
           domain, using gpt-5-nano guided by tag_taxonomy's
           keyword_anchors as prior knowledge.
  Stage C: generate the markdown file for each topic (frontmatter +
           body + wikilinks).
  Stage D: merge into existing files idempotently, honoring the
           `manual: true` opt-out and `supersedes`/`superseded_by`.

Design notes:
- We use a Responses-API client (gpt-5-nano, reasoning effort "low")
  with JSON schema structured outputs. Empty reasoning-token blowouts
  (cf. succession_rag contextualize) are avoided by generous
  max_output_tokens budgets.
- Each domain's KCs are clustered in a single LLM call (bounded input
  size). Large domains are split into chunks of ~60 KCs.
- Topic slugs are ASCII-safe hashes of the topic name so filenames
  work on any filesystem; the human-readable name lives in the
  frontmatter `title`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("build_wiki.clustering")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TopicAssignment:
    """One KC's assignment to a topic bucket within a domain."""

    kc_key: tuple[str, str]        # (session_id, question)
    domain: str
    topic_name: str                 # human-readable, e.g. "FBA統一方針"
    topic_slug: str                 # filesystem-safe, e.g. "fba-unified"
    rationale: str = ""             # why the LLM put it here (for audit)


@dataclass
class TopicBucket:
    """All KCs that cluster together into a single wiki page."""

    domain: str
    topic_name: str
    topic_slug: str
    kcs: list[Any] = field(default_factory=list)  # list[EnrichedKC]

    def file_path(self, output_dir: Path, visibility: str = "public") -> Path:
        """Path for this topic's .md file under the output directory."""
        subdir = f"knowledge-{visibility}"
        return output_dir / subdir / "topics" / self.domain / f"{self.topic_slug}.md"


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_SLUG_INVALID = re.compile(r"[^\w\-]+", re.UNICODE)
_SLUG_MULTI_DASH = re.compile(r"-{2,}")


def make_slug(name: str, max_len: int = 40) -> str:
    """Produce a filesystem-safe slug.

    For Japanese names we preserve the characters (kanji/kana are valid
    \\w under unicode), but strip whitespace and punctuation and cap the
    length. A short hash suffix guarantees uniqueness across close names.
    """
    name = (name or "").strip()
    if not name:
        return "untitled"
    # Replace whitespace with -, drop punctuation
    s = re.sub(r"\s+", "-", name)
    s = _SLUG_INVALID.sub("", s)
    s = _SLUG_MULTI_DASH.sub("-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    # Add a short hash suffix for collision resistance
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{s}-{digest}" if s else f"topic-{digest}"


# ---------------------------------------------------------------------------
# Clustering prompt
# ---------------------------------------------------------------------------

_CLUSTER_INSTRUCTIONS = """\
あなたは事業経営者のナレッジベースを整理するアーキテクトです。
与えられた Knowledge Card (KC) 群を、**atomic なトピック**にクラスタリング
してください。

## 原則

- **1 topic = 1 idea** (Zettelkasten atomic notes)
- 「同じ判断・方針・知見について話している」KCs を 1 グループにまとめる
- 違う角度 (事実/理由/手段/比較/リスク) のペアは同じ topic にまとめて良い
- **5-10 個を超える巨大 topic は避ける**: 大きい場合はサブトピックに分割
- topic 名は具体的な名詞句 (**20 文字以内**)
- **「雑談」「その他」のような曖昧な topic 名は禁止**
- **関連するドメイン知識 (keyword_anchors) を参考**: 例えば 物販 なら FBA,
  Keepa, Amazon, 仕入れ 等がトピック命名のヒント

## 出力

`topics` 配列: 各 topic について `name`、その topic に属する KC の
`kc_ids` (0-indexed の元リストでのインデックス) と `rationale`
(1 文の命名根拠) を返す。

- 各 KC は**必ず 1 つ以上の topic に含まれる**こと
- 大半の KC は 1 topic のみに属する (重複は例外)
- `kc_ids` が空の topic は作らない
"""


_CLUSTER_SCHEMA = {
    "type": "json_schema",
    "name": "topic_clustering",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kc_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["name", "kc_ids", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["topics"],
        "additionalProperties": False,
    },
}


def _format_kcs_for_prompt(kcs: list[Any]) -> str:
    """Render a list of EnrichedKCs into a compact numbered block."""
    lines = []
    for i, k in enumerate(kcs):
        ctx = f" [{k.context}]" if getattr(k, "context", "") else ""
        dec = f" (決定: {k.decision_memo})" if k.decision_memo else ""
        lines.append(
            f"#{i}{ctx} Q: {k.question}\n"
            f"   A: {k.answer[:150]}{dec}"
        )
    return "\n".join(lines)


def _keyword_hint(domain: str, taxonomy: dict) -> str:
    """Build a short keyword hint from tag_taxonomy for the given domain."""
    anchors = taxonomy.get("keyword_anchors", {})
    keywords = anchors.get(domain) or []
    if not keywords:
        return ""
    return (
        f"\n\n## 参考 (このドメインの主要キーワード)\n"
        f"{', '.join(keywords[:15])}"
    )


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around OpenAI Responses API (gpt-5-nano)."""

    def __init__(self, api_key: str, model: str = "gpt-5-nano") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=120.0)
        self._model = model

    def structured(
        self,
        instructions: str,
        user_input: str,
        schema: dict,
        max_output_tokens: int = 4000,
    ) -> dict:
        """Call the Responses API expecting a JSON schema-strict object.

        Raises on incomplete/empty output.
        """
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=[{"role": "user", "content": user_input}],
            text={"format": schema},
            reasoning={"effort": "low"},
            max_output_tokens=max_output_tokens,
            store=False,
        )
        if response.status != "completed":
            reason = ""
            if getattr(response, "incomplete_details", None):
                reason = getattr(response.incomplete_details, "reason", "unknown")
            raise RuntimeError(
                f"LLM returned status={response.status} reason={reason}"
            )
        raw = response.output_text or ""
        if not raw.strip():
            raise RuntimeError("LLM returned empty output_text")
        return json.loads(raw)

    def text(
        self,
        instructions: str,
        user_input: str,
        max_output_tokens: int = 2000,
    ) -> str:
        """Call the Responses API expecting free-form text (for markdown gen)."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=[{"role": "user", "content": user_input}],
            reasoning={"effort": "low"},
            max_output_tokens=max_output_tokens,
            store=False,
        )
        if response.status != "completed":
            reason = ""
            if getattr(response, "incomplete_details", None):
                reason = getattr(response.incomplete_details, "reason", "unknown")
            raise RuntimeError(
                f"LLM returned status={response.status} reason={reason}"
            )
        return (response.output_text or "").strip()


# ---------------------------------------------------------------------------
# Stage B: cluster KCs by domain
# ---------------------------------------------------------------------------


def cluster_kcs_by_domain(
    enriched_kcs: list[Any],
    taxonomy: dict,
    llm: LLMClient,
    chunk_size: int = 60,
) -> list[TopicAssignment]:
    """Group enriched KCs into topic buckets via LLM.

    Returns a flat list of TopicAssignments; one KC may appear in
    multiple assignments if the LLM chose to cross-reference it.
    """
    # Group by primary_domain
    by_domain: dict[str, list[Any]] = {}
    for k in enriched_kcs:
        by_domain.setdefault(k.primary_domain(), []).append(k)

    assignments: list[TopicAssignment] = []

    for domain, kcs in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        log.info("Clustering domain=%s (%d KCs)", domain, len(kcs))
        keyword_hint = _keyword_hint(domain, taxonomy)

        # Chunk large domains so context stays manageable
        chunk_count = 0
        for i in range(0, len(kcs), chunk_size):
            chunk = kcs[i:i + chunk_size]
            chunk_count += 1
            user_input = (
                f"## ドメイン\n{domain}\n{keyword_hint}\n\n"
                f"## KCs (このチャンクで {len(chunk)} 件)\n"
                f"{_format_kcs_for_prompt(chunk)}"
            )
            try:
                t0 = time.time()
                resp = llm.structured(
                    instructions=_CLUSTER_INSTRUCTIONS,
                    user_input=user_input,
                    schema=_CLUSTER_SCHEMA,
                    max_output_tokens=4000,
                )
                elapsed = time.time() - t0
                topics = resp.get("topics") or []
                log.info(
                    "  chunk %d: %d topics (%.1fs)",
                    chunk_count, len(topics), elapsed,
                )
            except Exception as e:
                log.error(
                    "  clustering failed for %s chunk %d: %s",
                    domain, chunk_count, e,
                )
                continue

            for t in topics:
                name = str(t.get("name") or "").strip()
                if not name:
                    continue
                slug = make_slug(name)
                rationale = str(t.get("rationale") or "")
                for idx in t.get("kc_ids") or []:
                    if not isinstance(idx, int) or idx < 0 or idx >= len(chunk):
                        continue
                    kc = chunk[idx]
                    assignments.append(TopicAssignment(
                        kc_key=(kc.session_id, kc.question),
                        domain=domain,
                        topic_name=name,
                        topic_slug=slug,
                        rationale=rationale,
                    ))

    return assignments


def build_topic_buckets(
    assignments: list[TopicAssignment],
    enriched_kcs: list[Any],
) -> list[TopicBucket]:
    """Fold TopicAssignments + EnrichedKCs into TopicBuckets."""
    kc_by_key: dict[tuple[str, str], Any] = {
        (k.session_id, k.question): k for k in enriched_kcs
    }
    by_topic: dict[tuple[str, str], TopicBucket] = {}
    for a in assignments:
        key = (a.domain, a.topic_slug)
        bucket = by_topic.get(key)
        if bucket is None:
            bucket = TopicBucket(
                domain=a.domain,
                topic_name=a.topic_name,
                topic_slug=a.topic_slug,
            )
            by_topic[key] = bucket
        kc = kc_by_key.get(a.kc_key)
        if kc is not None and kc not in bucket.kcs:
            bucket.kcs.append(kc)
    return list(by_topic.values())


# ---------------------------------------------------------------------------
# Stage C: generate markdown for a topic bucket
# ---------------------------------------------------------------------------


_TOPIC_GEN_INSTRUCTIONS = """\
あなたは事業経営者の Knowledge Card (KC) 群を受け取り、**1 つの Atomic
Topic ページ**を markdown で生成するライターです。

## 原則

- 入力の KC に書かれている情報**のみ**を使う。外部知識で補完しない。
- 本人の語り口と表現を尊重し、KC の answer の言い回しを残す。
- **200-500 字**で簡潔に。セクション構成は下記テンプレート通り。
- wikilinks は関連する他のトピックや人物への参照があれば `[[name]]`
  で埋め込む (なければ省略)。
- 推測や一般論 (「〜かもしれません」「一般的には」) を書かない。
- 回答不能な部分は書かない (欠落させる) より、省略する。

## 出力テンプレート (この構造を厳守、markdown のみ、前置き/後置きなし)

```
## 概要
2-4 文で topic の要点を書く。何の話題か、何が決まったか or 何が分かったか。

## 要点
- 箇条書き 3-5 個で具体的な事実・判断・数値
- KC の answer を統合する

## 判断の理由
(decision_memo や reasoning がある場合のみ。なければこのセクション丸ごと省略)

## 関連
(関連 topic / 人物がある場合のみ、[[wikilink]] 形式。なければこのセクション省略)
```

ルール:
- 各 `##` 見出しの後に必ず本文を置く
- 空セクションや「該当なし」は書かず、セクション自体を削る
- 最後の行を空行にしない
"""


def generate_topic_markdown(
    bucket: TopicBucket,
    llm: LLMClient,
) -> str:
    """Generate the markdown body for a topic (excluding frontmatter)."""
    kc_block = _format_kcs_for_prompt(bucket.kcs)
    session_summaries = sorted({
        k.session_summary for k in bucket.kcs if k.session_summary
    })
    summary_block = "\n".join(f"- {s}" for s in session_summaries[:5])

    user_input = (
        f"## Topic: {bucket.topic_name}\n"
        f"## Domain: {bucket.domain}\n\n"
        f"## このトピックに関する KC\n{kc_block}\n\n"
        f"## 関連セッションの要約\n{summary_block or '(なし)'}"
    )

    return llm.text(
        instructions=_TOPIC_GEN_INSTRUCTIONS,
        user_input=user_input,
        max_output_tokens=2500,
    )


def render_frontmatter(bucket: TopicBucket) -> dict[str, Any]:
    """Build frontmatter dict for a topic file."""
    sources = []
    seen_sessions: set[str] = set()
    all_entities: set[str] = set()
    confidences: list[str] = []
    dates: list[str] = []
    for k in bucket.kcs:
        if k.session_id not in seen_sessions:
            sources.append({
                "type": "voice",
                "session_id": k.session_id,
                "confidence": k.confidence,
            })
            seen_sessions.add(k.session_id)
        for e in k.entities:
            all_entities.add(e)
        confidences.append(k.confidence)
        if k.occurred_at:
            dates.append(k.occurred_at[:10])

    rank = {"high": 0, "medium": 1, "low": 2, "none": 3}
    best = min(confidences, key=lambda c: rank.get(c, 4)) if confidences else "medium"

    return {
        "id": f"topic_{bucket.domain}_{bucket.topic_slug}",
        "title": bucket.topic_name,
        "domain": bucket.domain,
        "date": min(dates) if dates else None,
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "sources": sources,
        "entities": sorted(all_entities)[:20],
        "confidence": best,
        "status": "active",
        "supersedes": [],
        "superseded_by": None,
        "manual": False,
    }


def format_markdown_file(
    frontmatter: dict[str, Any],
    body: str,
) -> str:
    """Assemble the full .md file content with YAML frontmatter."""
    # Hand-roll YAML for predictable ordering and no extra deps
    lines = ["---"]
    for key in (
        "id", "title", "domain", "date", "updated", "confidence",
        "status", "supersedes", "superseded_by", "manual",
        "sources", "entities",
    ):
        if key not in frontmatter:
            continue
        value = frontmatter[key]
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        elif isinstance(value, list):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        first = True
                        for k2, v2 in item.items():
                            prefix = "  - " if first else "    "
                            first = False
                            lines.append(f"{prefix}{k2}: {_yaml_scalar(v2)}")
                    else:
                        lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    return "\n".join(lines)


def _yaml_scalar(v: Any) -> str:
    """Render a scalar with quoting when it contains YAML-unsafe chars."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v)
    # Quote if starts with special chars or contains : / # etc.
    if any(c in s for c in ":#[]{},'\"") or s != s.strip():
        return json.dumps(s, ensure_ascii=False)
    return s


# ---------------------------------------------------------------------------
# Stage D: merge new topic content into existing wiki files
# ---------------------------------------------------------------------------


def write_topic_file(
    bucket: TopicBucket,
    body: str,
    output_dir: Path,
    visibility: str = "public",
) -> tuple[Path, str]:
    """Write (or merge) a topic bucket to its .md file.

    Returns (path, action) where action is one of:
      - "created"  : new file
      - "updated"  : replaced content (manual=false)
      - "skipped"  : existing file had manual=true, left alone

    Merging strategy for existing auto-generated files: overwrite the
    body wholesale, but preserve any frontmatter fields the user may
    have edited (manual, supersedes, superseded_by, status).
    """
    path = bucket.file_path(output_dir, visibility=visibility)
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = render_frontmatter(bucket)

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        existing_fm, _ = parse_frontmatter(existing)
        if existing_fm.get("manual") is True:
            return path, "skipped"
        # Preserve user-editable fields
        for key in ("supersedes", "superseded_by", "status"):
            if key in existing_fm and existing_fm[key] not in (None, [], ""):
                frontmatter[key] = existing_fm[key]
        action = "updated"
    else:
        action = "created"

    path.write_text(format_markdown_file(frontmatter, body), encoding="utf-8")
    return path, action


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from a markdown file.

    Returns (frontmatter_dict, body). If no frontmatter found, returns
    ({}, content) unchanged. Uses a minimal line-based parser to avoid
    pulling in pyyaml; we only need to read scalar/list fields.
    """
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    fm_block = content[4:end]
    body = content[end + 5:]

    fm: dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[list] = None
    for line in fm_block.splitlines():
        if not line.strip():
            continue
        stripped = line.rstrip()
        if line.startswith("  - ") and current_list is not None:
            current_list.append(line[4:].strip())
            continue
        if line.startswith("    ") and current_list is not None and current_list:
            # sub-field of a dict list item — ignore for now
            continue
        if not line.startswith(" "):
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", stripped)
            if m:
                k, v = m.group(1), m.group(2)
                if v == "":
                    current_key = k
                    current_list = []
                    fm[k] = current_list
                else:
                    current_list = None
                    fm[k] = _parse_yaml_scalar(v)
    return fm, body


def _parse_yaml_scalar(v: str) -> Any:
    v = v.strip()
    if v == "null" or v == "":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if v == "[]":
        return []
    if v.startswith('"') and v.endswith('"'):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v
