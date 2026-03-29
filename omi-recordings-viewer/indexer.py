"""SQLite index manager for recording sessions."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    start_at TEXT,
    end_at TEXT,
    duration_seconds REAL,
    audio_size_mb REAL,
    model TEXT,
    language TEXT,
    transcript_text TEXT,
    segment_count INTEGER,
    tags_topics TEXT,
    tags_activity TEXT,
    tags_intent TEXT,
    tags_mood TEXT,
    tags_quality TEXT,
    tags_entities TEXT,
    tags_summary TEXT,
    session_tagged INTEGER DEFAULT 0,
    UNIQUE(date, time)
);

CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_at DESC);
"""

# Migration: add tag columns to existing DBs
MIGRATION_ADD_TAGS = [
    "ALTER TABLE sessions ADD COLUMN tags_topics TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_activity TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_intent TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_mood TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_quality TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_entities TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_summary TEXT",
    "ALTER TABLE sessions ADD COLUMN session_tagged INTEGER DEFAULT 0",
]

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    transcript_text, content=sessions, content_rowid=id
);
"""

FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, transcript_text)
    VALUES (new.id, new.transcript_text);
END;
CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, transcript_text)
    VALUES ('delete', old.id, old.transcript_text);
END;
CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, transcript_text)
    VALUES ('delete', old.id, old.transcript_text);
    INSERT INTO sessions_fts(rowid, transcript_text)
    VALUES (new.id, new.transcript_text);
END;
"""


class Indexer:
    def __init__(self, db_path: str, recordings_dir: str):
        self.db_path = db_path
        self.recordings_dir = Path(recordings_dir)
        self._db: aiosqlite.Connection | None = None

    async def open(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(SCHEMA)
        await self._db.executescript(FTS_SCHEMA)
        await self._db.executescript(FTS_TRIGGERS)
        # Migrate: add tag columns if missing
        await self._migrate_tags()
        await self._db.commit()

    async def _migrate_tags(self):
        """Add tag columns to existing DB if they don't exist."""
        cursor = await self._db.execute("PRAGMA table_info(sessions)")
        existing = {row[1] for row in await cursor.fetchall()}
        for stmt in MIGRATION_ADD_TAGS:
            # Extract column name from "ALTER TABLE sessions ADD COLUMN <name> ..."
            col = stmt.split("ADD COLUMN")[1].strip().split()[0]
            if col not in existing:
                try:
                    await self._db.execute(stmt)
                    logger.info("Migration: added column %s", col)
                except Exception:
                    pass  # Column already exists

    async def close(self):
        if self._db:
            await self._db.close()

    def _scan_sessions(self, since_date: str | None = None) -> list[tuple[str, str]]:
        """Scan recordings dir and return (date, time) pairs that have transcript.json.

        If since_date is given (e.g. "2026-03-25"), only date directories
        >= that value are scanned.  On first run (DB empty) since_date is
        None and a full scan is performed.
        """
        results = []
        if not self.recordings_dir.is_dir():
            return results
        for date_dir in sorted(self.recordings_dir.iterdir()):
            if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
                continue
            if since_date and date_dir.name < since_date:
                continue
            for time_dir in sorted(date_dir.iterdir()):
                if not time_dir.is_dir() or not TIME_RE.match(time_dir.name):
                    continue
                if (time_dir / "transcript.json").exists():
                    results.append((date_dir.name, time_dir.name))
        return results

    async def _get_latest_date(self) -> str | None:
        """Return the latest indexed date, or None if DB is empty."""
        cursor = await self._db.execute(
            "SELECT MAX(date) as max_date FROM sessions"
        )
        row = await cursor.fetchone()
        return row["max_date"] if row else None

    async def _get_indexed(self, since_date: str | None = None) -> set[tuple[str, str]]:
        """Get (date, time) pairs already in the DB.

        If since_date is given, only returns pairs >= that date.
        """
        if since_date:
            cursor = await self._db.execute(
                "SELECT date, time FROM sessions WHERE date >= ?",
                (since_date,),
            )
        else:
            cursor = await self._db.execute("SELECT date, time FROM sessions")
        rows = await cursor.fetchall()
        return {(r["date"], r["time"]) for r in rows}

    def _parse_transcript(self, date: str, time: str) -> dict:
        """Parse a transcript.json file and return session data."""
        path = self.recordings_dir / date / time / "transcript.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        session = data.get("session", {})
        segments = data.get("segments", [])
        transcript_text = " ".join(seg.get("text", "") for seg in segments).strip()

        # Extract tags if present
        tags = session.get("tags", {})
        tags_topics = ",".join(tags.get("topics", [])) if tags.get("topics") else None
        tags_entities = ",".join(tags.get("entities", [])) if tags.get("entities") else None

        return {
            "date": date,
            "time": time,
            "start_at": session.get("start"),
            "end_at": session.get("end"),
            "duration_seconds": session.get("duration_seconds"),
            "audio_size_mb": session.get("audio_size_mb"),
            "model": session.get("model"),
            "language": session.get("language"),
            "transcript_text": transcript_text,
            "segment_count": len(segments),
            "tags_topics": tags_topics,
            "tags_activity": tags.get("activity"),
            "tags_intent": tags.get("intent"),
            "tags_mood": tags.get("mood"),
            "tags_quality": tags.get("quality"),
            "tags_entities": tags_entities,
            "tags_summary": tags.get("summary"),
            "session_tagged": 1 if session.get("session_tagged") else 0,
        }

    async def _get_untagged(self, since_date: str | None = None) -> set[tuple[str, str]]:
        """Get (date, time) pairs that are indexed but not yet tagged."""
        if since_date:
            cursor = await self._db.execute(
                "SELECT date, time FROM sessions WHERE date >= ? AND session_tagged = 0",
                (since_date,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT date, time FROM sessions WHERE session_tagged = 0"
            )
        rows = await cursor.fetchall()
        return {(r["date"], r["time"]) for r in rows}

    async def sync(self) -> int:
        """Sync DB with recordings dir. Returns number of new/updated sessions."""
        since_date = await self._get_latest_date()
        on_disk = self._scan_sessions(since_date)
        indexed = await self._get_indexed(since_date)

        # New sessions from recent dates
        new = [(d, t) for d, t in on_disk if (d, t) not in indexed]

        # Also re-index ALL untagged sessions (batch tagging may update old dates)
        all_untagged = await self._get_untagged()
        new.extend((d, t) for d, t in all_untagged if (d, t) not in set(new))

        if not new:
            return 0

        count = 0
        for date, time in new:
            try:
                info = self._parse_transcript(date, time)
            except Exception:
                logger.exception("Failed to parse %s/%s", date, time)
                continue

            await self._db.execute(
                """INSERT OR REPLACE INTO sessions
                   (date, time, start_at, end_at, duration_seconds,
                    audio_size_mb, model, language, transcript_text, segment_count,
                    tags_topics, tags_activity, tags_intent, tags_mood,
                    tags_quality, tags_entities, tags_summary, session_tagged)
                   VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                           :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                           :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                           :tags_quality, :tags_entities, :tags_summary, :session_tagged)""",
                info,
            )
            count += 1

        await self._db.commit()
        if count:
            logger.info("Indexed %d new/updated sessions", count)
        return count

    async def get_sessions(
        self, page: int = 1, per_page: int = 20, query: str = "",
        activity: str = "", mood: str = "",
    ):
        """Get paginated sessions, optionally filtered by text query and tag filters."""
        offset = (page - 1) * per_page

        where_clauses = []
        params = []

        if query:
            where_clauses.append("transcript_text LIKE ?")
            params.append("%" + query + "%")

        if activity:
            where_clauses.append("tags_activity = ?")
            params.append(activity)

        if mood:
            where_clauses.append("tags_mood = ?")
            params.append(mood)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        count_row = await self._db.execute_fetchall(
            f"SELECT COUNT(*) as cnt FROM sessions{where_sql}",
            params,
        )
        total = count_row[0]["cnt"]

        cursor = await self._db.execute(
            f"SELECT * FROM sessions{where_sql} ORDER BY start_at DESC LIMIT ? OFFSET ?",
            params + [per_page, offset],
        )

        rows = await cursor.fetchall()
        return {
            "sessions": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if per_page else 0,
        }

    async def get_session(self, date: str, time: str):
        """Get a single session by date/time."""
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE date = ? AND time = ?",
            (date, time),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)

    async def get_transcript(self, date: str, time: str):
        """Load full transcript.json from disk."""
        path = self.recordings_dir / date / time / "transcript.json"
        if not path.exists():
            return None
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, path.read_text, "utf-8")
        return json.loads(data)

    async def periodic_sync(self, interval: int = 30):
        """Background task: sync every `interval` seconds."""
        while True:
            await asyncio.sleep(interval)
            try:
                n = await self.sync()
                if n:
                    logger.info("Periodic sync: added %d sessions", n)
            except Exception:
                logger.exception("Periodic sync failed")
