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
    tags_user TEXT,
    speakers TEXT,
    last_modified REAL DEFAULT 0,
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
    "ALTER TABLE sessions ADD COLUMN tags_user TEXT",
    "ALTER TABLE sessions ADD COLUMN speakers TEXT",
    "ALTER TABLE sessions ADD COLUMN last_modified REAL DEFAULT 0",
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

    def _scan_sessions_with_mtime(self) -> list[tuple[str, str, float]]:
        """Scan recordings dir and return (date, time, mtime) tuples.

        Always does a full scan.  The mtime is the modification time of
        transcript.json, used to detect changes by batch operations
        (diarize, tag, speaker tag) without re-reading unchanged files.
        """
        results = []
        if not self.recordings_dir.is_dir():
            return results
        for date_dir in sorted(self.recordings_dir.iterdir()):
            if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
                continue
            for time_dir in sorted(date_dir.iterdir()):
                if not time_dir.is_dir() or not TIME_RE.match(time_dir.name):
                    continue
                transcript = time_dir / "transcript.json"
                if transcript.exists():
                    try:
                        mtime = transcript.stat().st_mtime
                    except OSError:
                        continue
                    results.append((date_dir.name, time_dir.name, mtime))
        return results

    async def _get_indexed_mtimes(self) -> dict[tuple[str, str], float]:
        """Get {(date, time): last_modified} for all indexed sessions."""
        cursor = await self._db.execute(
            "SELECT date, time, last_modified FROM sessions"
        )
        rows = await cursor.fetchall()
        return {(r["date"], r["time"]): r["last_modified"] or 0.0 for r in rows}

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
        tags_user = ",".join(tags.get("user_tags", [])) if tags.get("user_tags") else None

        # Extract unique speaker names from segments
        speaker_names = set()
        for seg in segments:
            pn = seg.get("person_name")
            if pn and pn != "unknown":
                speaker_names.add(pn)
        speakers_csv = ",".join(sorted(speaker_names)) if speaker_names else None

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
            "tags_user": tags_user,
            "speakers": speakers_csv,
        }

    async def sync(self) -> int:
        """Sync DB with recordings dir. Returns number of new/updated sessions.

        Uses file modification time (mtime) to detect changes efficiently.
        Only sessions with new or changed transcript.json are re-indexed,
        avoiding the previous O(N) re-index of all untagged sessions.
        """
        loop = asyncio.get_event_loop()

        # Scan disk for all sessions with mtimes (in executor to avoid blocking)
        on_disk = await loop.run_in_executor(
            None, self._scan_sessions_with_mtime
        )

        # Get stored mtimes from DB
        indexed_mtimes = await self._get_indexed_mtimes()

        # Determine which sessions need (re-)indexing
        to_index = []
        for date, time, mtime in on_disk:
            key = (date, time)
            stored_mtime = indexed_mtimes.get(key)
            if stored_mtime is None or abs(mtime - stored_mtime) > 0.01:
                to_index.append((date, time, mtime))

        if not to_index:
            return 0

        count = 0
        for date, time, mtime in to_index:
            try:
                info = await loop.run_in_executor(
                    None, self._parse_transcript, date, time
                )
            except Exception:
                logger.exception("Failed to parse %s/%s", date, time)
                continue

            info["last_modified"] = mtime
            await self._db.execute(
                """INSERT OR REPLACE INTO sessions
                   (date, time, start_at, end_at, duration_seconds,
                    audio_size_mb, model, language, transcript_text, segment_count,
                    tags_topics, tags_activity, tags_intent, tags_mood,
                    tags_quality, tags_entities, tags_summary, session_tagged,
                    tags_user, speakers, last_modified)
                   VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                           :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                           :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                           :tags_quality, :tags_entities, :tags_summary, :session_tagged,
                           :tags_user, :speakers, :last_modified)""",
                info,
            )
            count += 1

            # Yield to event loop periodically to prevent blocking
            if count % 10 == 0:
                await asyncio.sleep(0)

        await self._db.commit()
        if count:
            logger.info("Indexed %d new/updated sessions", count)
        return count

    async def get_sessions(
        self, page: int = 1, per_page: int = 20, query: str = "",
        activity: str = "", mood: str = "", user_tag: str = "",
        speaker: str = "",
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

        if user_tag:
            where_clauses.append("(tags_user LIKE ? OR tags_user = ?)")
            params.append("%" + user_tag + "%")
            params.append(user_tag)

        if speaker:
            where_clauses.append("(speakers LIKE ? OR speakers = ?)")
            params.append("%" + speaker + "%")
            params.append(speaker)

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

    async def get_speaker_names(self) -> list[str]:
        """Return all unique speaker names from the speakers column."""
        cursor = await self._db.execute(
            "SELECT DISTINCT speakers FROM sessions WHERE speakers IS NOT NULL AND speakers != ''"
        )
        rows = await cursor.fetchall()
        names = set()
        for row in rows:
            for name in row["speakers"].split(","):
                name = name.strip()
                if name:
                    names.add(name)
        return sorted(names)

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
