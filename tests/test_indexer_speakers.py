"""Tests for indexer speakers column and speaker filter."""

import asyncio
import json
import os
import tempfile

import pytest

# Ensure the viewer module is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "omi-recordings-viewer"))

from indexer import Indexer


@pytest.fixture
def tmp_env(tmp_path):
    """Create a temp recordings dir with sample transcript.json files."""
    rec_dir = tmp_path / "recordings"
    db_path = str(tmp_path / "test.db")

    # Session 1: has person_name tags
    s1 = rec_dir / "2026-03-30" / "10-00-00"
    s1.mkdir(parents=True)
    (s1 / "transcript.json").write_text(json.dumps({
        "session": {"start": "2026-03-30T10:00:00", "duration_seconds": 60},
        "segments": [
            {"start": 0, "end": 5, "text": "hello", "person_name": "Taro", "person_id": "sp1"},
            {"start": 5, "end": 10, "text": "hi", "person_name": "Hanako", "person_id": "sp2"},
            {"start": 10, "end": 15, "text": "...", "person_name": "unknown", "person_id": "unknown"},
        ],
    }, ensure_ascii=False))

    # Session 2: no person_name
    s2 = rec_dir / "2026-03-30" / "11-00-00"
    s2.mkdir(parents=True)
    (s2 / "transcript.json").write_text(json.dumps({
        "session": {"start": "2026-03-30T11:00:00", "duration_seconds": 30},
        "segments": [
            {"start": 0, "end": 5, "text": "test"},
        ],
    }, ensure_ascii=False))

    # Session 3: only Taro
    s3 = rec_dir / "2026-03-30" / "12-00-00"
    s3.mkdir(parents=True)
    (s3 / "transcript.json").write_text(json.dumps({
        "session": {"start": "2026-03-30T12:00:00", "duration_seconds": 20},
        "segments": [
            {"start": 0, "end": 5, "text": "solo", "person_name": "Taro", "person_id": "sp1"},
        ],
    }, ensure_ascii=False))

    return db_path, str(rec_dir)


@pytest.mark.asyncio
async def test_speakers_column_populated(tmp_env):
    db_path, rec_dir = tmp_env
    indexer = Indexer(db_path, rec_dir)
    await indexer.open()
    count = await indexer.sync()
    assert count == 3

    # Session 1 should have "Hanako,Taro" (sorted)
    s1 = await indexer.get_session("2026-03-30", "10-00-00")
    assert s1["speakers"] == "Hanako,Taro"

    # Session 2 should have no speakers
    s2 = await indexer.get_session("2026-03-30", "11-00-00")
    assert s2["speakers"] is None

    # Session 3 should have "Taro"
    s3 = await indexer.get_session("2026-03-30", "12-00-00")
    assert s3["speakers"] == "Taro"

    await indexer.close()


@pytest.mark.asyncio
async def test_get_speaker_names(tmp_env):
    db_path, rec_dir = tmp_env
    indexer = Indexer(db_path, rec_dir)
    await indexer.open()
    await indexer.sync()

    names = await indexer.get_speaker_names()
    assert names == ["Hanako", "Taro"]

    await indexer.close()


@pytest.mark.asyncio
async def test_filter_by_speaker(tmp_env):
    db_path, rec_dir = tmp_env
    indexer = Indexer(db_path, rec_dir)
    await indexer.open()
    await indexer.sync()

    # Filter by Taro: sessions 1 and 3
    result = await indexer.get_sessions(speaker="Taro")
    assert result["total"] == 2

    # Filter by Hanako: only session 1
    result = await indexer.get_sessions(speaker="Hanako")
    assert result["total"] == 1

    # No filter: all 3
    result = await indexer.get_sessions()
    assert result["total"] == 3

    await indexer.close()
