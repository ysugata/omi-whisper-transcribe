"""Tests for export WHERE-clause filters (indexer._build_export_where)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "omi-recordings-viewer"))

from indexer import Indexer


def _make_session(rec_dir, date, time, domains=None, segments=5):
    d = rec_dir / date / time
    d.mkdir(parents=True, exist_ok=True)
    session = {"start": f"{date}T{time.replace('-', ':')}", "duration_seconds": 60}
    if domains is not None:
        session["tags"] = {
            "conversation_type": "社内ミーティング",
            "outcomes": ["情報共有のみ"],
            "domains": domains,
        }
        session["session_tagged"] = True
    segs = [{"start": i, "end": i + 1, "text": f"seg{i}"} for i in range(segments)]
    (d / "transcript.json").write_text(
        json.dumps({"session": session, "segments": segs}, ensure_ascii=False)
    )


@pytest.mark.asyncio
async def test_exclude_personal_only(tmp_path):
    """exclude_personal_only drops only sessions whose domains == [個人].

    tags_domains is stored comma-joined ("個人" / "個人,経営"), so the filter
    must match the bare string, and untagged (NULL) sessions must be kept.
    """
    rec_dir = tmp_path / "recordings"
    _make_session(rec_dir, "2026-04-01", "10-00-00", domains=["個人"])
    _make_session(rec_dir, "2026-04-01", "11-00-00", domains=["個人", "経営"])
    _make_session(rec_dir, "2026-04-01", "12-00-00", domains=None)  # untagged

    indexer = Indexer(str(tmp_path / "test.db"), str(rec_dir))
    await indexer.open()
    await indexer.sync()

    kept = await indexer.get_export_sessions(
        min_segments=0, exclude_personal_only=True
    )
    times = sorted(t for _, t in kept)
    assert times == ["11-00-00", "12-00-00"], \
        f"personal-only session must be dropped, others kept; got {times}"

    # Without the filter all 3 are returned
    all_sessions = await indexer.get_export_sessions(min_segments=0)
    assert len(all_sessions) == 3

    await indexer.close()
