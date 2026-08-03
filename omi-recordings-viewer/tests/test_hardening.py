"""Tests for write_tx transaction isolation and the SanitizedIndex cache."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exporter import get_sanitized_index
from indexer import Indexer


@pytest.fixture
def tmp_env(tmp_path):
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    db_path = str(tmp_path / "index.db")
    return recordings, db_path


def _create_session(recordings, date, time_str):
    session_dir = recordings / date / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "session": {"start": f"{date}T00:00:00", "duration_seconds": 30.0,
                    "language": "ja"},
        "segments": [{"text": "hello", "start": 0, "end": 1}],
    }
    (session_dir / "transcript.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_write_tx_commits_on_success(tmp_env):
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    ix = Indexer(db_path, str(recordings))
    await ix.open()
    await ix.sync()

    async with ix.write_tx() as db:
        await db.execute(
            "UPDATE sessions SET language = 'en' WHERE date = '2026-03-29'"
        )
    s = await ix.get_session("2026-03-29", "10-00-00")
    assert s["language"] == "en"
    await ix.close()


@pytest.mark.asyncio
async def test_write_tx_rolls_back_on_error(tmp_env):
    recordings, db_path = tmp_env
    _create_session(recordings, "2026-03-29", "10-00-00")
    ix = Indexer(db_path, str(recordings))
    await ix.open()
    await ix.sync()

    with pytest.raises(RuntimeError):
        async with ix.write_tx() as db:
            await db.execute(
                "UPDATE sessions SET language = 'en' WHERE date = '2026-03-29'"
            )
            raise RuntimeError("boom")

    s = await ix.get_session("2026-03-29", "10-00-00")
    assert s["language"] == "ja"  # the failed unit's UPDATE was rolled back

    # The connection is reusable after rollback
    async with ix.write_tx() as db:
        await db.execute(
            "UPDATE sessions SET language = 'fr' WHERE date = '2026-03-29'"
        )
    s = await ix.get_session("2026-03-29", "10-00-00")
    assert s["language"] == "fr"
    await ix.close()


def _write_safe_training(config_dir, name, session_id):
    rec = {
        "type": "summary",
        "input": "in",
        "output": "out",
        "session_id": session_id,
        "risk_level": "low",
    }
    (config_dir / name).write_text(
        json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def test_sanitized_index_cache_hit_and_invalidation(tmp_path):
    _write_safe_training(tmp_path, "safe_training_20260101_000000.jsonl", "a/b")

    idx1 = get_sanitized_index(tmp_path)
    idx2 = get_sanitized_index(tmp_path)
    assert idx1 is idx2  # unchanged directory → cached object reused

    # New file invalidates the cache
    _write_safe_training(tmp_path, "safe_training_20260102_000000.jsonl", "c/d")
    idx3 = get_sanitized_index(tmp_path)
    assert idx3 is not idx1
    assert "a/b" in idx3 and "c/d" in idx3
