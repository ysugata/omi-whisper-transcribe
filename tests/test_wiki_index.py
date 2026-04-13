"""Unit tests for wiki_index.py."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from wiki_index import (
    WikiIndex,
    WikiTopic,
    _extract_sources,
    _parse_frontmatter,
    _split_frontmatter,
)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestSplitFrontmatter:
    def test_basic(self):
        content = "---\ntitle: foo\n---\nbody here\nmore body\n"
        fm, body = _split_frontmatter(content)
        assert fm == "title: foo"
        assert body == "body here\nmore body\n"

    def test_no_frontmatter(self):
        content = "just some markdown\nno frontmatter"
        fm, body = _split_frontmatter(content)
        assert fm == ""
        assert body == content


class TestParseFrontmatter:
    def test_scalar_fields(self):
        fm = _parse_frontmatter(
            "title: hello world\n"
            "domain: 物販\n"
            "manual: true\n"
            "confidence: high\n"
        )
        assert fm["title"] == "hello world"
        assert fm["domain"] == "物販"
        assert fm["manual"] is True
        assert fm["confidence"] == "high"

    def test_empty_list(self):
        fm = _parse_frontmatter("supersedes: []\n")
        assert fm["supersedes"] == []

    def test_block_list_of_dicts(self):
        fm = _parse_frontmatter(
            "sources:\n"
            "  - type: voice\n"
            "    session_id: 2026-03-28/03-48-40\n"
            "    confidence: high\n"
            "  - type: voice\n"
            "    session_id: 2026-03-28/03-33-17\n"
            "    confidence: high\n"
        )
        assert isinstance(fm["sources"], list)
        assert len(fm["sources"]) == 2
        assert fm["sources"][0]["session_id"] == "2026-03-28/03-48-40"
        assert fm["sources"][1]["session_id"] == "2026-03-28/03-33-17"

    def test_block_list_of_scalars(self):
        fm = _parse_frontmatter(
            "entities:\n"
            "  - Amazon\n"
            "  - FBA\n"
            "  - Keepa\n"
        )
        assert fm["entities"] == ["Amazon", "FBA", "Keepa"]


class TestExtractSources:
    def test_dict_form(self):
        fm = {"sources": [
            {"type": "voice", "session_id": "2026-03-28/03-48-40"},
            {"type": "voice", "session_id": "2026-03-28/03-33-17"},
        ]}
        assert _extract_sources(fm) == [
            "2026-03-28/03-48-40",
            "2026-03-28/03-33-17",
        ]

    def test_string_form(self):
        fm = {"sources": ["sid1", "sid2"]}
        assert _extract_sources(fm) == ["sid1", "sid2"]

    def test_empty(self):
        assert _extract_sources({}) == []
        assert _extract_sources({"sources": []}) == []
        assert _extract_sources({"sources": None}) == []


# ---------------------------------------------------------------------------
# WikiIndex end-to-end
# ---------------------------------------------------------------------------


def _write_topic(
    knowledge_dir: Path,
    visibility: str,
    domain: str,
    slug: str,
    title: str,
    sources: list[str],
    body: str = "## 概要\nテスト本文\n",
    manual: bool = False,
) -> Path:
    p = knowledge_dir / f"knowledge-{visibility}" / "topics" / domain / f"{slug}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = [
        "---",
        f"id: topic_{domain}_{slug}",
        f"title: {title}",
        f"domain: {domain}",
        "confidence: high",
        "status: active",
        f"manual: {'true' if manual else 'false'}",
        "sources:",
    ]
    for sid in sources:
        fm_lines.append(f"  - type: voice")
        fm_lines.append(f"    session_id: {sid}")
        fm_lines.append(f"    confidence: high")
    fm_lines.append("---")
    fm_lines.append("")
    fm_lines.append(body)
    p.write_text("\n".join(fm_lines), encoding="utf-8")
    return p


class TestWikiIndex:
    def test_load_empty_dir(self, tmp_path):
        idx = WikiIndex(tmp_path)
        result = idx.load()
        assert result["status"] == "loaded"
        assert idx.topic_count == 0

    def test_disabled_when_dir_missing(self, tmp_path):
        idx = WikiIndex(tmp_path / "does_not_exist")
        assert idx.enabled is False
        result = idx.load()
        assert result["status"] == "disabled"

    def test_load_real_topics(self, tmp_path):
        _write_topic(
            tmp_path, "public", "物販", "fba-policy",
            title="FBA優先と自己配送併用",
            sources=["2026-03-28/03-48-40", "2026-03-28/03-33-17"],
        )
        _write_topic(
            tmp_path, "public", "AI", "claude-code-setup",
            title="Claude Code導入",
            sources=["2026-03-26/16-12-14"],
        )
        _write_topic(
            tmp_path, "private", "経営", "private-decision",
            title="Private Decision",
            sources=["2026-03-29/10-00-00"],
        )

        idx = WikiIndex(tmp_path)
        result = idx.load()
        assert result["status"] == "loaded"
        assert idx.topic_count == 3
        assert idx.session_count == 4

    def test_topics_for_session(self, tmp_path):
        _write_topic(
            tmp_path, "public", "物販", "fba-policy",
            title="FBA優先",
            sources=["sid_fba_01", "sid_fba_02"],
        )
        _write_topic(
            tmp_path, "public", "物販", "keepa-design",
            title="Keepa API設計",
            sources=["sid_fba_01"],   # crosses both topics
        )
        idx = WikiIndex(tmp_path)
        idx.load()
        topics = idx.topics_for_session("sid_fba_01")
        assert len(topics) == 2
        titles = sorted(t.title for t in topics)
        assert titles == ["FBA優先", "Keepa API設計"]

    def test_manual_flag_preserved(self, tmp_path):
        _write_topic(
            tmp_path, "public", "AI", "manual-topic",
            title="Manual Topic",
            sources=["sid1"],
            manual=True,
        )
        idx = WikiIndex(tmp_path)
        idx.load()
        topic = idx.topics_for_session("sid1")[0]
        assert topic.manual is True

    def test_body_extracted(self, tmp_path):
        _write_topic(
            tmp_path, "public", "AI", "with-body",
            title="With Body",
            sources=["sid1"],
            body="## 概要\n本文の内容\n\n## 要点\n- A\n- B\n",
        )
        idx = WikiIndex(tmp_path)
        idx.load()
        topic = idx.topics_for_session("sid1")[0]
        assert "## 概要" in topic.body
        assert "本文の内容" in topic.body
        assert "## 要点" in topic.body
