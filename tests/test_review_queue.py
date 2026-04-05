"""Tests for review queue priority ordering and statistics."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "omi-recordings-viewer"))

from indexer import Indexer


def _make_session(rec_dir, date, time, tags=None, segments=None, audio_scene=None):
    """Helper to create a session directory with transcript.json."""
    d = rec_dir / date / time
    d.mkdir(parents=True, exist_ok=True)
    session = {"start": f"{date}T{time.replace('-', ':')}", "duration_seconds": 60}
    if tags:
        session["tags"] = tags
        session["session_tagged"] = True
    if audio_scene:
        session["audio_scene"] = audio_scene
    segs = segments if segments is not None else [{"start": 0, "end": 5, "text": "hello"}]
    (d / "transcript.json").write_text(
        json.dumps({"session": session, "segments": segs}, ensure_ascii=False)
    )


@pytest.fixture
def tmp_env(tmp_path):
    rec_dir = tmp_path / "recordings"
    db_path = str(tmp_path / "test.db")
    return db_path, rec_dir


@pytest.mark.asyncio
async def test_review_queue_priority_ordering(tmp_env):
    """RAG low similarity should rank higher than LLM noisy, which ranks higher than RAG high similarity."""
    db_path, rec_dir = tmp_env

    # Session 1: RAG with low similarity (highest priority)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": ["情報共有"],
        "domains": [],
        "importance": "低",
        "tag_source": "rag",
        "rag_similarity": 0.3,
        "tag_version": 2,
    })

    # Session 2: LLM with noisy quality (medium priority)
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話",
        "outcomes": ["意思決定"],
        "domains": ["個人"],
        "importance": "中",
        "quality": "noisy",
        "tag_source": "llm",
        "tag_version": 2,
    })

    # Session 3: RAG with high similarity (lowest priority)
    _make_session(rec_dir, "2026-04-01", "12-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": ["情報共有"],
        "domains": [],
        "importance": "低",
        "tag_source": "rag",
        "rag_similarity": 0.85,
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 3

    # First should be RAG low similarity (highest priority_score)
    assert queue[0]["time"] == "10-00-00", f"Expected RAG low sim first, got {queue[0]['time']}"
    # priority_score of first should be > second
    assert queue[0]["priority_score"] > queue[1]["priority_score"]
    # Last should be RAG high similarity
    assert queue[2]["time"] == "12-00-00", f"Expected RAG high sim last, got {queue[2]['time']}"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_reviewed(tmp_env):
    """Sessions with manually_corrected=1 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: not reviewed
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_source": "llm",
        "tag_version": 2,
    })

    # Session 2: already reviewed
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話",
        "outcomes": [],
        "importance": "中",
        "tag_source": "llm",
        "manually_corrected": True,
        "corrected_at": "2026-04-01T12:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_v1_and_untagged(tmp_env):
    """Sessions with tags_version < 2 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: V2 tagged (should appear)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    })

    # Session 2: V1 tagged (should NOT appear)
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "activity": "meeting",
        "tag_version": 1,
    })

    # Session 3: untagged (should NOT appear)
    _make_session(rec_dir, "2026-04-01", "12-00-00")

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_excludes_empty_sessions(tmp_env):
    """Sessions with segment_count=0 should not appear in the queue."""
    db_path, rec_dir = tmp_env

    # Session 1: has segments
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    })

    # Session 2: no segments
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話",
        "outcomes": [],
        "importance": "低",
        "tag_version": 2,
    }, segments=[])

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    queue = await indexer.get_review_queue(limit=10, offset=0)
    assert len(queue) == 1
    assert queue[0]["time"] == "10-00-00"

    await indexer.close()


@pytest.mark.asyncio
async def test_review_stats(tmp_env):
    """Review stats should correctly count pending/reviewed/today."""
    db_path, rec_dir = tmp_env

    # 2 pending, 1 reviewed (today)
    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話", "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "業務会話", "tag_version": 2,
    })
    _make_session(rec_dir, "2026-04-01", "12-00-00", tags={
        "conversation_type": "日常会話",
        "manually_corrected": True,
        "corrected_at": "2099-01-01T00:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    stats = await indexer.get_review_stats()
    assert stats["total_pending"] == 2
    assert stats["total_reviewed"] == 1
    # reviewed_today depends on actual date, so just check type
    assert isinstance(stats["reviewed_today"], int)

    await indexer.close()


@pytest.mark.asyncio
async def test_review_queue_offset(tmp_env):
    """Offset should skip the first N items."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "その他", "tag_version": 2,
        "tag_source": "rag", "rag_similarity": 0.3,
    })
    _make_session(rec_dir, "2026-04-01", "11-00-00", tags={
        "conversation_type": "日常会話", "tag_version": 2,
        "tag_source": "llm",
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    # offset=0 should return highest priority
    q0 = await indexer.get_review_queue(limit=1, offset=0)
    assert len(q0) == 1

    # offset=1 should return the other one
    q1 = await indexer.get_review_queue(limit=1, offset=1)
    assert len(q1) == 1
    assert q1[0]["time"] != q0[0]["time"]

    # offset=2 should return nothing
    q2 = await indexer.get_review_queue(limit=1, offset=2)
    assert len(q2) == 0

    await indexer.close()


@pytest.mark.asyncio
async def test_new_columns_in_parse_transcript(tmp_env):
    """Verify that _parse_transcript correctly extracts new review columns."""
    db_path, rec_dir = tmp_env

    _make_session(rec_dir, "2026-04-01", "10-00-00", tags={
        "conversation_type": "日常会話",
        "tag_source": "rag",
        "rag_similarity": 0.72,
        "manually_corrected": True,
        "corrected_at": "2026-04-01T10:00:00",
        "tag_version": 2,
    })

    indexer = Indexer(db_path, str(rec_dir))
    await indexer.open()
    await indexer.sync()

    s = await indexer.get_session("2026-04-01", "10-00-00")
    assert s["tags_tag_source"] == "rag"
    assert abs(s["tags_rag_similarity"] - 0.72) < 0.001
    assert s["tags_manually_corrected"] == 1
    assert s["tags_corrected_at"] == "2026-04-01T10:00:00"

    await indexer.close()
