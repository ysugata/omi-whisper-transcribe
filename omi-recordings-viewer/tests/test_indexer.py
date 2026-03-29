"""Tests for the mtime-based indexer sync logic."""

import asyncio
import json
import os
import tempfile
import time

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from indexer import Indexer


@pytest.fixture
def tmp_env(tmp_path):
    """Create a temp recordings dir and DB path."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    db_path = str(tmp_path / "index.db")
    return recordings, db_path


def _create_session(recordings, date, time_str, segments=None, tags=None):
    """Helper to create a session directory with transcript.json."""
    session_dir = recordings / date / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "session": {
            "start": f"{date}T{time_str.replace('-', ':')}",
            "end": f"{date}T{time_str.replace('-', ':')}",
            "duration_seconds": 30.0,
            "model": "large-v3",
            "language": "ja",
        },
        "segments": segments or [{"text": "hello world", "start": 0, "end": 1}],
    }
    if tags:
        data["session"]["tags"] = tags
        if tags.get("activity"):
            data["session"]["session_tagged"] = True
    path = session_dir / "transcript.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_initial_sync_indexes_all(tmp_env):
    """First sync should index all sessions."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    _create_session(recordings, "2026-03-29", "11-00-00")
    _create_session(recordings, "2026-03-30", "09-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    n = await indexer.sync()
    assert n == 3
    await indexer.close()


@pytest.mark.asyncio
async def test_second_sync_skips_unchanged(tmp_env):
    """Second sync with no changes should index 0 sessions."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    _create_session(recordings, "2026-03-29", "11-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    n1 = await indexer.sync()
    assert n1 == 2

    # Second sync — nothing changed
    n2 = await indexer.sync()
    assert n2 == 0
    await indexer.close()


@pytest.mark.asyncio
async def test_sync_detects_modified_file(tmp_env):
    """Sync should re-index a session whose transcript.json was modified."""
    recordings, db_path = tmp_env
    path = _create_session(recordings, "2026-03-29", "10-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    n1 = await indexer.sync()
    assert n1 == 1

    # Modify the file (simulate diarization/tagging updating it)
    time.sleep(0.05)  # Ensure mtime changes
    data = json.loads(path.read_text())
    data["segments"].append({"text": "updated segment", "start": 1, "end": 2})
    path.write_text(json.dumps(data), encoding="utf-8")

    n2 = await indexer.sync()
    assert n2 == 1  # Should detect the change

    # Third sync — no more changes
    n3 = await indexer.sync()
    assert n3 == 0
    await indexer.close()


@pytest.mark.asyncio
async def test_sync_detects_new_session(tmp_env):
    """Sync should pick up newly created sessions."""
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    n1 = await indexer.sync()
    assert n1 == 1

    # Add a new session
    _create_session(recordings, "2026-03-29", "12-00-00")
    n2 = await indexer.sync()
    assert n2 == 1

    await indexer.close()


@pytest.mark.asyncio
async def test_untagged_sessions_not_reindexed(tmp_env):
    """Untagged sessions should NOT be re-indexed every cycle (the bug fix)."""
    recordings, db_path = tmp_env
    # Create 50 untagged sessions
    for i in range(50):
        _create_session(recordings, "2026-03-29", f"{i:02d}-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()
    n1 = await indexer.sync()
    assert n1 == 50

    # Subsequent syncs should index 0 (no file changes)
    for _ in range(5):
        n = await indexer.sync()
        assert n == 0, "Untagged sessions should not be re-indexed"

    await indexer.close()


@pytest.mark.asyncio
async def test_sync_does_not_block_event_loop(tmp_env):
    """Sync should yield to event loop, allowing concurrent tasks to run."""
    recordings, db_path = tmp_env
    # Create sessions
    for i in range(25):
        _create_session(recordings, "2026-03-29", f"{i:02d}-00-00")

    indexer = Indexer(db_path, str(recordings))
    await indexer.open()

    # Run sync and a concurrent task
    counter = {"ticks": 0}

    async def ticker():
        while True:
            counter["ticks"] += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    await indexer.sync()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The ticker should have run at least a few times during sync
    assert counter["ticks"] > 0, "Event loop was blocked during sync"
    await indexer.close()


@pytest.mark.asyncio
async def test_migration_adds_last_modified(tmp_env):
    """Migration should add last_modified column to existing DB."""
    recordings, db_path = tmp_env
    indexer = Indexer(db_path, str(recordings))
    await indexer.open()

    # Check column exists
    cursor = await indexer._db.execute("PRAGMA table_info(sessions)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "last_modified" in cols

    await indexer.close()
