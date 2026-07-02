"""SQLite index manager for recording sessions."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
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
    tags_external_audio REAL,
    tags_audio_scene TEXT,
    tags_audio_scene_spatial TEXT,
    tags_audio_scene_confidence REAL,
    tags_audio_noise_sources TEXT,
    tags_audio_events TEXT,
    tags_audio_music_detected INTEGER DEFAULT 0,
    tags_audio_speaker_count TEXT,
    tags_tag_source TEXT,
    tags_rag_similarity REAL,
    tags_manually_corrected INTEGER DEFAULT 0,
    tags_corrected_at TEXT,
    tags_review_outcome TEXT,
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
CREATE INDEX IF NOT EXISTS idx_sessions_audio_speaker_count ON sessions(tags_audio_speaker_count);
CREATE INDEX IF NOT EXISTS idx_sessions_manually_corrected ON sessions(tags_manually_corrected);
CREATE INDEX IF NOT EXISTS idx_sessions_tag_source ON sessions(tags_tag_source);
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
    "ALTER TABLE sessions ADD COLUMN tags_audio_noise_sources TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_audio_events TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_audio_music_detected INTEGER DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN tags_audio_speaker_count TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_tag_source TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_rag_similarity REAL",
    "ALTER TABLE sessions ADD COLUMN tags_manually_corrected INTEGER DEFAULT 0",
    "ALTER TABLE sessions ADD COLUMN tags_corrected_at TEXT",
    "ALTER TABLE sessions ADD COLUMN tags_review_outcome TEXT",
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
        tags_user = None  # user_tags feature removed

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
            "tags_audio_noise_sources": ",".join(
                ns["type"] for ns in (session.get("audio_scene") or {}).get("noise_sources", [])
            ) or None,
            "tags_audio_events": ",".join(
                ev["type"] for ev in (session.get("audio_scene") or {}).get("events", [])
            ) or None,
            "tags_audio_music_detected": 1 if (
                (session.get("audio_scene") or {}).get("music", {}).get("detected")
            ) else 0,
            "tags_audio_speaker_count": (
                (session.get("audio_scene") or {}).get("speaker_count", {}).get("estimate")
            ),
            "tags_tag_source": tags.get("tag_source"),
            "tags_rag_similarity": tags.get("rag_similarity"),
            "tags_manually_corrected": 1 if tags.get("manually_corrected") else 0,
            "tags_corrected_at": tags.get("corrected_at"),
            "tags_review_outcome": tags.get("review_outcome"),
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
                    tags_audio_scene, tags_audio_scene_spatial, tags_audio_scene_confidence,
                    tags_audio_noise_sources, tags_audio_events,
                    tags_audio_music_detected, tags_audio_speaker_count,
                    tags_tag_source, tags_rag_similarity, tags_manually_corrected, tags_corrected_at,
                    tags_review_outcome)
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
                           :tags_audio_scene, :tags_audio_scene_spatial, :tags_audio_scene_confidence,
                           :tags_audio_noise_sources, :tags_audio_events,
                           :tags_audio_music_detected, :tags_audio_speaker_count,
                           :tags_tag_source, :tags_rag_similarity, :tags_manually_corrected, :tags_corrected_at,
                           :tags_review_outcome)""",
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
        activity: str = "", mood: str = "",
        speaker: str = "", business_domain: str = "",
        importance: str = "", source_type: str = "",
        conversation_type: str = "", outcome: str = "",
        domain: str = "", importance_v2: str = "",
        audio_scene: str = "",
        audio_speaker_count: str = "",
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

        if speaker:
            where_clauses.append("(speakers LIKE ? OR speakers = ?)")
            params.append("%" + speaker + "%")
            params.append(speaker)

        if audio_scene:
            where_clauses.append("tags_audio_scene = ?")
            params.append(audio_scene)

        if audio_speaker_count:
            where_clauses.append("tags_audio_speaker_count = ?")
            params.append(audio_speaker_count)

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

    async def get_taxonomy_label_counts(self) -> dict:
        """Get session counts per taxonomy label for all 4 axes."""
        result = {}

        # Single-select axes: direct GROUP BY
        for axis, col in [
            ("conversation_type", "tags_conversation_type"),
            ("importance", "tags_importance_v2"),
        ]:
            cursor = await self._db.execute(
                f"SELECT {col}, COUNT(*) as cnt FROM sessions "
                f"WHERE {col} IS NOT NULL AND {col} != '' "
                f"GROUP BY {col}"
            )
            rows = await cursor.fetchall()
            result[axis] = {r[0]: r[1] for r in rows}

        # Multi-select axes: comma-separated, split in Python
        for axis, col in [
            ("outcomes", "tags_outcomes"),
            ("domains", "tags_domains"),
        ]:
            cursor = await self._db.execute(
                f"SELECT {col} FROM sessions "
                f"WHERE {col} IS NOT NULL AND {col} != ''"
            )
            rows = await cursor.fetchall()
            counts: dict[str, int] = {}
            for r in rows:
                for val in r[0].split(","):
                    val = val.strip()
                    if val:
                        counts[val] = counts.get(val, 0) + 1
            result[axis] = counts

        return result

    async def get_export_stats(
        self, date_from: str = "", date_to: str = "",
        quality: list[str] | None = None, activity: list[str] | None = None,
        min_segments: int = 3, business_domain: str = "",
        conversation_type: str = "", domain: str = "", importance_v2: str = "",
        max_per_class: int = 0,
        require_tagged: bool = False,
        require_summary: bool = False,
        exclude_personal_only: bool = False,
        external_audio_max: float | None = None,
        high_value_or: bool = False,
    ) -> dict:
        """Get stats for export preview (session count, segment count, date range, est size)."""
        where, params = self._build_export_where(
            date_from, date_to, quality, activity, min_segments,
            business_domain=business_domain,
            conversation_type=conversation_type, domain=domain,
            importance_v2=importance_v2,
            require_tagged=require_tagged,
            require_summary=require_summary,
            exclude_personal_only=exclude_personal_only,
            external_audio_max=external_audio_max,
            high_value_or=high_value_or,
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
        total_sessions = r["cnt"]

        # If balanced, compute effective session count
        balanced_sessions = total_sessions
        class_breakdown = {}
        if max_per_class > 0:
            cursor = await self._db.execute(
                f"SELECT COALESCE(tags_conversation_type, '_untagged') as cls,"
                f" COUNT(*) as cnt FROM sessions{where_sql}"
                f" GROUP BY cls",
                params,
            )
            cls_rows = await cursor.fetchall()
            balanced_sessions = 0
            for cr in cls_rows:
                actual = cr["cnt"]
                capped = min(actual, max_per_class)
                balanced_sessions += capped
                class_breakdown[cr["cls"]] = {"total": actual, "sampled": capped}

        result = {
            "sessions": total_sessions,
            "segments": r["segs"],
            "date_from": r["min_date"] or "",
            "date_to": r["max_date"] or "",
            "est_size_bytes": int(r["text_bytes"] * 1.5),
        }
        if max_per_class > 0:
            result["balanced_sessions"] = balanced_sessions
            result["class_breakdown"] = class_breakdown
        return result

    async def get_export_sessions(
        self, date_from: str = "", date_to: str = "",
        quality: list[str] | None = None, activity: list[str] | None = None,
        min_segments: int = 3, business_domain: str = "",
        conversation_type: str = "", domain: str = "", importance_v2: str = "",
        max_per_class: int = 0,
        require_tagged: bool = False,
        require_summary: bool = False,
        exclude_personal_only: bool = False,
        external_audio_max: float | None = None,
        high_value_or: bool = False,
    ) -> list[tuple[str, str]]:
        """Get (date, time) pairs matching export filters.

        New params (for KC extraction / sanitize targeting):
            require_tagged: session_tagged = 1
            require_summary: tags_summary non-empty
            exclude_personal_only: drop sessions whose domains = ["個人"] only
            external_audio_max: cap tags_external_audio (NULL allowed)
            high_value_or: add OR clause (importance≥中 OR useful conv_type
                OR decision_memo non-empty OR outcomes contains 意思決定/TODO/
                方針転換/アイデア創出/問題提起). Captures "事業継承価値" sessions.

        If max_per_class > 0, limit each conversation_type to at most
        that many sessions (stratified sampling for balanced training data).
        Sessions are selected randomly within each class.
        """
        where, params = self._build_export_where(
            date_from, date_to, quality, activity, min_segments,
            business_domain=business_domain,
            conversation_type=conversation_type, domain=domain,
            importance_v2=importance_v2,
            require_tagged=require_tagged,
            require_summary=require_summary,
            exclude_personal_only=exclude_personal_only,
            external_audio_max=external_audio_max,
            high_value_or=high_value_or,
        )
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        if max_per_class > 0:
            # Stratified: GROUP BY conversation_type, limit per group
            cursor = await self._db.execute(
                f"SELECT date, time, tags_conversation_type FROM sessions{where_sql}"
                f" ORDER BY RANDOM()",
                params,
            )
            rows = await cursor.fetchall()
            class_counts: dict[str, int] = {}
            result = []
            for r in rows:
                cls = r["tags_conversation_type"] or "_untagged"
                class_counts[cls] = class_counts.get(cls, 0) + 1
                if class_counts[cls] <= max_per_class:
                    result.append((r["date"], r["time"]))
            return result

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
        require_tagged: bool = False,
        require_summary: bool = False,
        exclude_personal_only: bool = False,
        external_audio_max: float | None = None,
        high_value_or: bool = False,
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

        if require_tagged:
            where.append("session_tagged = 1")

        if require_summary:
            where.append("tags_summary IS NOT NULL AND tags_summary != ''")

        if exclude_personal_only:
            # drop sessions where domains is only "個人"
            # (tags_domains is comma-joined; a lone 個人 means personal-only.
            #  NULL = untagged, which is kept.)
            where.append(
                "(tags_domains IS NULL OR tags_domains != '個人')"
            )

        if external_audio_max is not None:
            # NULL (not computed) is allowed, else must be below max
            where.append(
                "(tags_external_audio IS NULL OR tags_external_audio < ?)"
            )
            params.append(external_audio_max)

        if high_value_or:
            # 事業継承価値のあるセッションの OR 条件:
            #   importance 中+ / 有用 conv_type / decision_memo 非空 /
            #   outcomes に実質的アウトカムを含む
            useful_conv_types = (
                "ミーティング", "アイデア出し", "商談",
                "顧客ヒアリング", "1on1", "外部セミナー",
            )
            useful_outcomes = (
                "意思決定あり", "TODO発生", "方針転換",
                "アイデア創出", "問題提起",
            )
            conv_placeholders = ",".join("?" for _ in useful_conv_types)
            outcome_likes = " OR ".join(
                "tags_outcomes LIKE ?" for _ in useful_outcomes
            )
            where.append(
                "("
                " tags_importance_v2 IN ('中','高')"
                f" OR tags_conversation_type IN ({conv_placeholders})"
                " OR (tags_decision_memo IS NOT NULL AND tags_decision_memo != '')"
                f" OR ({outcome_likes})"
                ")"
            )
            params.extend(useful_conv_types)
            params.extend(f"%{o}%" for o in useful_outcomes)

        return where, params

    # Default review scoring weights (sum = 1.0 for 5 base axes)
    # importance is applied as a multiplier, not an additive axis
    _DEFAULT_REVIEW_WEIGHTS = {
        "model_uncertainty": 0.40,
        "novelty": 0.20,
        "content_complexity": 0.15,
        "audio_scene_ambiguity": 0.12,
        "temporal_freshness": 0.13,
    }

    def _get_review_weights(self) -> dict:
        """Load adjusted review weights from config, or return defaults."""
        weights_path = self.recordings_dir.parent / "config" / "review_weights.json"
        try:
            if weights_path.exists():
                data = json.loads(weights_path.read_text(encoding="utf-8"))
                adjusted = data.get("adjusted_weights")
                if adjusted and len(adjusted) == 5:
                    return adjusted
        except Exception:
            logger.warning("Failed to load review_weights.json, using defaults")
        return dict(self._DEFAULT_REVIEW_WEIGHTS)

    async def get_review_queue(self, limit: int = 1, offset: int = 0) -> list[dict]:
        """Get sessions ordered by review priority (highest uncertainty first).

        priority_score = base_score * importance_multiplier

        Base score (5 axes + interaction bonus):
        - A. Model uncertainty (0.40): continuous RAG similarity / LLM quality
        - B. Novelty (0.20): conversation_type その他 or NULL
        - C. Content complexity (0.15): segment count proportional
        - D. Audio scene ambiguity (0.12): external audio + low scene confidence
        - E. Temporal freshness (0.13): 90-day linear decay
        - F. Interaction bonus (+0.10 max): compound risk signals

        Importance multiplier: 高=1.5, 中=1.0, 低/NULL=0.8
        """
        w = self._get_review_weights()
        w_a = w["model_uncertainty"]
        w_b = w["novelty"]
        w_c = w["content_complexity"]
        w_d = w["audio_scene_ambiguity"]
        w_e = w["temporal_freshness"]

        sql = f"""
            SELECT *,
                (
                    (
                        -- A. Model uncertainty ({w_a}): continuous value
                        CASE
                            WHEN tags_tag_source = 'rag' THEN
                                (1.0 - COALESCE(tags_rag_similarity, 0.5)) * {w_a}
                            WHEN tags_quality IN ('noisy', 'fragmented') THEN {w_a} * 0.857
                            WHEN tags_quality = 'partial' THEN {w_a} * 0.571
                            ELSE {w_a} * 0.429
                        END
                        +
                        -- B. Novelty ({w_b})
                        CASE
                            WHEN tags_conversation_type = 'その他' THEN {w_b}
                            WHEN tags_conversation_type IS NULL THEN {w_b} * 0.9
                            WHEN tags_conversation_type IN ('独り言・メモ', '雑談・移動中メモ') THEN {w_b} * 0.25
                            ELSE 0.0
                        END
                        +
                        -- C. Content complexity ({w_c}): proportional to segment count
                        MIN(COALESCE(segment_count, 1), 50) * 1.0 / 50 * {w_c}
                        +
                        -- D. Audio scene ambiguity ({w_d})
                        CASE WHEN COALESCE(tags_external_audio, 0) >= 0.5 THEN {w_d} * 0.6 ELSE 0.0 END
                        +
                        CASE
                            WHEN tags_audio_scene_confidence IS NOT NULL AND tags_audio_scene_confidence < 0.7
                            THEN (1.0 - tags_audio_scene_confidence) * {w_d} * 0.4
                            ELSE 0.0
                        END
                        +
                        -- E. Temporal freshness ({w_e}): 90-day linear decay
                        CASE
                            WHEN start_at IS NOT NULL THEN
                                MAX(0.0, 1.0 - (julianday('now') - julianday(start_at)) / 90.0) * {w_e}
                            ELSE {w_e} * 0.5
                        END
                        +
                        -- F. Interaction bonus (+0.10 max)
                        CASE
                            WHEN tags_tag_source = 'rag' AND COALESCE(tags_rag_similarity, 0.5) < 0.5
                                 AND tags_quality IN ('noisy', 'fragmented') THEN 0.10
                            WHEN tags_tag_source = 'rag' AND COALESCE(tags_rag_similarity, 0.5) < 0.5
                                 AND COALESCE(tags_external_audio, 0) >= 0.5 THEN 0.08
                            ELSE 0.0
                        END
                    )
                    *
                    -- Importance multiplier
                    CASE
                        WHEN tags_importance_v2 = '高' THEN 1.5
                        WHEN tags_importance_v2 = '中' THEN 1.0
                        ELSE 0.8
                    END
                ) AS priority_score
            FROM sessions
            WHERE tags_version >= 2
              AND COALESCE(tags_manually_corrected, 0) = 0
              AND segment_count > 0
            ORDER BY priority_score DESC, start_at DESC
            LIMIT ? OFFSET ?
        """
        cursor = await self._db.execute(sql, (limit, offset))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_review_stats(self) -> dict:
        """Get review queue statistics."""
        rows = await self._db.execute_fetchall(
            """SELECT
                   SUM(CASE WHEN tags_version >= 2 AND COALESCE(tags_manually_corrected, 0) = 0 AND segment_count > 0 THEN 1 ELSE 0 END) AS total_pending,
                   SUM(CASE WHEN COALESCE(tags_manually_corrected, 0) = 1 THEN 1 ELSE 0 END) AS total_reviewed,
                   SUM(CASE WHEN COALESCE(tags_manually_corrected, 0) = 1 AND tags_corrected_at >= date('now') THEN 1 ELSE 0 END) AS reviewed_today
               FROM sessions"""
        )
        r = rows[0]
        return {
            "total_pending": r["total_pending"] or 0,
            "total_reviewed": r["total_reviewed"] or 0,
            "reviewed_today": r["reviewed_today"] or 0,
        }

    @staticmethod
    def classify_session_signals(session: dict) -> list[str]:
        """Classify session attributes into signal buckets for feedback tracking."""
        signals = []
        tag_source = session.get("tags_tag_source")
        rag_sim = session.get("tags_rag_similarity")
        quality = session.get("tags_quality")
        conv_type = session.get("tags_conversation_type")
        ext_audio = session.get("tags_external_audio") or 0

        # Axis A: model uncertainty
        if tag_source == "rag":
            if rag_sim is not None and rag_sim < 0.5:
                signals.append("rag_low_sim")
            else:
                signals.append("rag_high_sim")
        elif quality in ("noisy", "fragmented"):
            signals.append("llm_noisy")
        else:
            signals.append("llm_clean")

        # Axis B: novelty
        if conv_type == "その他":
            signals.append("sono_ta")

        # Axis D: audio scene ambiguity
        if ext_audio >= 0.5:
            signals.append("high_external_audio")

        return signals

    def record_review_feedback(self, session: dict, was_corrected: bool) -> None:
        """Record review outcome and update scoring weights via EMA feedback.

        Args:
            session: dict with session DB columns (tags_tag_source, etc.)
            was_corrected: True if user edited tags, False if accepted as-is
        """
        weights_path = self.recordings_dir.parent / "config" / "review_weights.json"
        ema_alpha = 0.15

        # Load or initialize
        try:
            if weights_path.exists():
                data = json.loads(weights_path.read_text(encoding="utf-8"))
            else:
                data = {}
        except Exception:
            data = {}

        if "version" not in data:
            data = {
                "version": 1,
                "updated_at": "",
                "base_weights": dict(self._DEFAULT_REVIEW_WEIGHTS),
                "correction_rates": {},
                "adjusted_weights": dict(self._DEFAULT_REVIEW_WEIGHTS),
            }

        correction_rates = data.setdefault("correction_rates", {})
        signals = self.classify_session_signals(session)

        for sig in signals:
            bucket = correction_rates.setdefault(
                sig, {"corrections": 0, "total": 0, "rate": 0.0}
            )
            bucket["total"] += 1
            if was_corrected:
                bucket["corrections"] += 1
            # EMA update of correction rate
            observed = 1.0 if was_corrected else 0.0
            bucket["rate"] = bucket["rate"] * (1 - ema_alpha) + observed * ema_alpha

        # Map signal buckets → axis keys
        signal_axis_map = {
            "rag_low_sim": "model_uncertainty",
            "rag_high_sim": "model_uncertainty",
            "llm_noisy": "model_uncertainty",
            "llm_clean": "model_uncertainty",
            "sono_ta": "novelty",
            "high_external_audio": "audio_scene_ambiguity",
        }

        # Compute max correction rate per axis
        axis_max_rate = {}
        for sig, bucket in correction_rates.items():
            axis = signal_axis_map.get(sig)
            if axis and bucket["total"] >= 3:  # Need minimum samples
                axis_max_rate[axis] = max(
                    axis_max_rate.get(axis, 0.0), bucket["rate"]
                )

        # Adjust weights: base * (1 + max_rate), then normalize to 1.0
        base = data.get("base_weights", dict(self._DEFAULT_REVIEW_WEIGHTS))
        adjusted = {}
        for key, base_w in base.items():
            boost = axis_max_rate.get(key, 0.0)
            adjusted[key] = base_w * (1.0 + boost)

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        data["adjusted_weights"] = adjusted
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")

        try:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.exception("Failed to save review_weights.json")

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
