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
    tags_source_type TEXT,
    tags_business_domain TEXT,
    tags_importance TEXT,
    tags_confidentiality TEXT,
    tags_decision_required INTEGER DEFAULT 0,
    tags_decision_summary TEXT,
    tags_action_items TEXT,
    tags_outcome TEXT,
    tags_conversation_type TEXT,
    tags_outcomes TEXT,
    tags_domains TEXT,
    tags_decision_memo TEXT,
    tags_importance_v2 TEXT,
    tags_version INTEGER DEFAULT 0,
    tags_audio_scene TEXT,
    tags_audio_scene_spatial TEXT,
    tags_audio_scene_confidence REAL,
    UNIQUE(date, time)
);

CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_at DESC);
"""

# Indexes on columns that may be added by migration — must run AFTER migration
POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(tags_activity);
CREATE INDEX IF NOT EXISTS idx_sessions_mood ON sessions(tags_mood);
CREATE INDEX IF NOT EXISTS idx_sessions_speakers ON sessions(speakers);
CREATE INDEX IF NOT EXISTS idx_sessions_business_domain ON sessions(tags_business_domain);
CREATE INDEX IF NOT EXISTS idx_sessions_importance ON sessions(tags_importance);
CREATE INDEX IF NOT EXISTS idx_sessions_decision_required ON sessions(tags_decision_required);
CREATE INDEX IF NOT EXISTS idx_sessions_conversation_type ON sessions(tags_conversation_type);
CREATE INDEX IF NOT EXISTS idx_sessions_importance_v2 ON sessions(tags_importance_v2);
CREATE INDEX IF NOT EXISTS idx_sessions_audio_scene ON sessions(tags_audio_scene);
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
    "ALTER TABLE sessions ADD COLUMN tags_source_type TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_business_domain TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_importance TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_confidentiality TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_decision_required INTEGER DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN tags_decision_summary TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_action_items TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_outcome TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_conversation_type TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_outcomes TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_domains TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_decision_memo TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_importance_v2 TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_version INTEGER DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN tags_external_audio REAL",
    "ALTER TABLE sessions ADD COLUMN tags_audio_scene TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_audio_scene_spatial TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_audio_scene_confidence REAL",
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
        # Create indexes on migrated columns (must be after migration)
        await self._db.executescript(POST_MIGRATION_INDEXES)
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

        # Extract action_items as comma-separated string
        action_items = tags.get("action_items", [])
        tags_action_items = ",".join(action_items) if isinstance(action_items, list) and action_items else None

        # V2 (3-axis) tag detection
        is_v2 = bool(tags.get("conversation_type"))
        tag_version = tags.get("tag_version", 0)
        if is_v2 and tag_version == 0:
            tag_version = 2
        elif not is_v2 and session.get("session_tagged") and tag_version == 0:
            tag_version = 1

        # V2 fields
        tags_conversation_type = tags.get("conversation_type") if is_v2 else None
        outcomes = tags.get("outcomes", [])
        tags_outcomes = ",".join(outcomes) if isinstance(outcomes, list) and outcomes else None
        domains = tags.get("domains", [])
        tags_domains = ",".join(domains) if isinstance(domains, list) and domains else None
        tags_decision_memo = tags.get("decision_memo") if is_v2 else None
        tags_importance_v2 = tags.get("importance") if is_v2 else None

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
            "tags_source_type": tags.get("source_type"),
            "tags_business_domain": tags.get("business_domain"),
            "tags_importance": tags.get("importance"),
            "tags_confidentiality": tags.get("confidentiality"),
            "tags_decision_required": 1 if tags.get("decision_required") else 0,
            "tags_decision_summary": tags.get("decision_summary"),
            "tags_action_items": tags_action_items,
            "tags_outcome": tags.get("outcome"),
            "tags_conversation_type": tags_conversation_type,
            "tags_outcomes": tags_outcomes,
            "tags_domains": tags_domains,
            "tags_decision_memo": tags_decision_memo,
            "tags_importance_v2": tags_importance_v2,
            "tags_version": tag_version,
            "tags_external_audio": tags.get("external_audio_ratio"),
            "tags_audio_scene": (session.get("audio_scene") or {}).get("scene_label"),
            "tags_audio_scene_spatial": (session.get("audio_scene") or {}).get("spatial_label"),
            "tags_audio_scene_confidence": (session.get("audio_scene") or {}).get("scene_confidence"),
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
                    tags_user, speakers, last_modified,
                    tags_source_type, tags_business_domain, tags_importance,
                    tags_confidentiality, tags_decision_required,
                    tags_decision_summary, tags_action_items, tags_outcome,
                    tags_conversation_type, tags_outcomes, tags_domains,
                    tags_decision_memo, tags_importance_v2, tags_version,
                    tags_external_audio,
                    tags_audio_scene, tags_audio_scene_spatial, tags_audio_scene_confidence)
                   VALUES (:date, :time, :start_at, :end_at, :duration_seconds,
                           :audio_size_mb, :model, :language, :transcript_text, :segment_count,
                           :tags_topics, :tags_activity, :tags_intent, :tags_mood,
                           :tags_quality, :tags_entities, :tags_summary, :session_tagged,
                           :tags_user, :speakers, :last_modified,
                           :tags_source_type, :tags_business_domain, :tags_importance,
                           :tags_confidentiality, :tags_decision_required,
                           :tags_decision_summary, :tags_action_items, :tags_outcome,
                           :tags_conversation_type, :tags_outcomes, :tags_domains,
                           :tags_decision_memo, :tags_importance_v2, :tags_version,
                           :tags_external_audio,
                           :tags_audio_scene, :tags_audio_scene_spatial, :tags_audio_scene_confidence)""",
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
        speaker: str = "", business_domain: str = "",
        importance: str = "", source_type: str = "",
        conversation_type: str = "", outcome: str = "",
        domain: str = "", importance_v2: str = "",
        audio_scene: str = "",
    ):
        """Get paginated sessions, optionally filtered by text query and tag filters."""
        offset = (page - 1) * per_page

        where_clauses = []
        params = []

        if query:
            where_clauses.append("transcript_text LIKE ?")
            params.append("%" + query + "%")

        # V1 filters (backward compat)
        if activity:
            where_clauses.append("tags_activity = ?")
            params.append(activity)

        if mood:
            where_clauses.append("tags_mood = ?")
            params.append(mood)

        if business_domain:
            where_clauses.append("tags_business_domain = ?")
            params.append(business_domain)

        if importance:
            where_clauses.append("tags_importance = ?")
            params.append(importance)

        if source_type:
            where_clauses.append("tags_source_type = ?")
            params.append(source_type)

        # V2 filters (3-axis)
        if conversation_type:
            where_clauses.append("tags_conversation_type = ?")
            params.append(conversation_type)

        if outcome:
            where_clauses.append("(tags_outcomes LIKE ? OR tags_outcomes = ?)")
            params.append("%" + outcome + "%")
            params.append(outcome)

        if domain:
            where_clauses.append("(tags_domains LIKE ? OR tags_domains = ?)")
            params.append("%" + domain + "%")
            params.append(domain)

        if importance_v2:
            where_clauses.append("tags_importance_v2 = ?")
            params.append(importance_v2)

        if user_tag:
            where_clauses.append("(tags_user LIKE ? OR tags_user = ?)")
            params.append("%" + user_tag + "%")
            params.append(user_tag)

        if speaker:
            where_clauses.append("(speakers LIKE ? OR speakers = ?)")
            params.append("%" + speaker + "%")
            params.append(speaker)

        if audio_scene:
            where_clauses.append("tags_audio_scene = ?")
            params.append(audio_scene)

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

    async def get_audio_scene_labels(self) -> list[str]:
        """Get distinct audio scene labels."""
        cursor = await self._db.execute(
            "SELECT DISTINCT tags_audio_scene FROM sessions WHERE tags_audio_scene IS NOT NULL ORDER BY tags_audio_scene"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

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

    async def get_export_stats(
        self, date_from: str = "", date_to: str = "",
        quality: list[str] | None = None, activity: list[str] | None = None,
        min_segments: int = 3, business_domain: str = "",
        conversation_type: str = "", domain: str = "", importance_v2: str = "",
    ) -> dict:
        """Get stats for export preview (session count, segment count, date range, est size)."""
        where, params = self._build_export_where(
            date_from, date_to, quality, activity, min_segments,
            business_domain=business_domain,
            conversation_type=conversation_type, domain=domain,
            importance_v2=importance_v2,
        )
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        rows = await self._db.execute_fetchall(
            f"SELECT COUNT(*) as cnt, COALESCE(SUM(segment_count), 0) as segs,"
            f" MIN(date) as min_date, MAX(date) as max_date,"
            f" COALESCE(SUM(LENGTH(transcript_text)), 0) as text_bytes"
            f" FROM sessions{where_sql}",
            params,
        )
        r = rows[0]
        return {
            "sessions": r["cnt"],
            "segments": r["segs"],
            "date_from": r["min_date"] or "",
            "date_to": r["max_date"] or "",
            "est_size_bytes": int(r["text_bytes"] * 1.5),  # rough estimate with JSON overhead
        }

    async def get_export_sessions(
        self, date_from: str = "", date_to: str = "",
        quality: list[str] | None = None, activity: list[str] | None = None,
        min_segments: int = 3, business_domain: str = "",
        conversation_type: str = "", domain: str = "", importance_v2: str = "",
    ) -> list[tuple[str, str]]:
        """Get (date, time) pairs matching export filters."""
        where, params = self._build_export_where(
            date_from, date_to, quality, activity, min_segments,
            business_domain=business_domain,
            conversation_type=conversation_type, domain=domain,
            importance_v2=importance_v2,
        )
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        cursor = await self._db.execute(
            f"SELECT date, time FROM sessions{where_sql} ORDER BY date, time",
            params,
        )
        rows = await cursor.fetchall()
        return [(r["date"], r["time"]) for r in rows]

    def _build_export_where(
        self, date_from: str, date_to: str,
        quality: list[str] | None, activity: list[str] | None,
        min_segments: int, business_domain: str = "",
        conversation_type: str = "", domain: str = "", importance_v2: str = "",
    ) -> tuple[list[str], list]:
        """Build WHERE clauses and params for export queries."""
        where: list[str] = []
        params: list = []

        if min_segments > 0:
            where.append("segment_count >= ?")
            params.append(min_segments)

        if date_from:
            where.append("date >= ?")
            params.append(date_from)
        if date_to:
            where.append("date <= ?")
            params.append(date_to)

        if quality:
            placeholders = ",".join("?" for _ in quality)
            where.append(f"tags_quality IN ({placeholders})")
            params.extend(quality)

        if activity:
            placeholders = ",".join("?" for _ in activity)
            where.append(f"tags_activity IN ({placeholders})")
            params.extend(activity)

        if business_domain:
            where.append("tags_business_domain = ?")
            params.append(business_domain)

        # V2 taxonomy filters
        if conversation_type:
            where.append("tags_conversation_type = ?")
            params.append(conversation_type)
        if domain:
            where.append("tags_domains LIKE ?")
            params.append(f"%{domain}%")
        if importance_v2:
            where.append("tags_importance_v2 = ?")
            params.append(importance_v2)

        return where, params

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
