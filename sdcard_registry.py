"""SD card upload dedup registry + device timestamp validation.

The Omi app retries /sdcard/upload when it does not receive a response
in time.  Transcribing a 30-min file synchronously takes several minutes,
so every large upload used to be retried forever (662 retries of the same
file were observed).  The registry remembers which uploads have already
been accepted so retries can be acknowledged instantly without
re-transcribing.

Device clocks also drift badly (a 79-day offset was observed), which used
to file sessions under months-old dates where the startup cleanup deleted
them.  resolve_recorded_at() falls back to the server clock when the
device timestamp is implausible.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("sdcard")

# Entry states
STATE_PROCESSING = "processing"
STATE_DONE = "done"
STATE_FAILED = "failed"

# Registry entries older than this are pruned on load
REGISTRY_RETENTION_DAYS = 60


def make_key(device_id: str, timestamp_start: int, content_sha256: str) -> str:
    """Stable identity of one SD file upload."""
    return f"{device_id}:{timestamp_start}:{content_sha256[:16]}"


def resolve_recorded_at(
    timestamp_start: int,
    max_age_days: int,
    now: datetime | None = None,
) -> tuple[datetime, datetime | None]:
    """Return (effective_recorded_at, device_recorded_at_or_None).

    device_recorded_at is non-None only when the device timestamp was
    rejected and the server clock was used instead; callers should record
    it in the session metadata so the original time is not lost.
    """
    now = now or datetime.now()
    try:
        device_dt = datetime.fromtimestamp(timestamp_start)
    except (ValueError, OSError, OverflowError):
        return now, None

    too_old = device_dt < now - timedelta(days=max_age_days)
    in_future = device_dt > now + timedelta(days=1)
    if too_old or in_future:
        return now, device_dt
    return device_dt, None


class SdcardUploadRegistry:
    """Persistent map of upload key -> processing state.

    All mutations happen on the event loop thread (no await between
    lookup and mark), so no locking is needed.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                self._entries = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("SD registry unreadable (%s), starting empty", e)
            self._entries = {}
            return
        self._prune()

    def _prune(self, drop_processing: bool = True) -> None:
        """Apply retention.

        drop_processing=True (load time): stale "processing" entries are from
        a crashed/restarted process; drop them so the file can be uploaded
        again. Runtime pruning must pass False — in-flight uploads are
        legitimately in STATE_PROCESSING.
        """
        cutoff = (
            datetime.now() - timedelta(days=REGISTRY_RETENTION_DAYS)
        ).isoformat()
        before = len(self._entries)
        self._entries = {
            k: v for k, v in self._entries.items()
            if v.get("uploaded_at", "") >= cutoff
            and (not drop_processing or v.get("state") != STATE_PROCESSING)
        }
        if len(self._entries) != before:
            self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("SD registry save failed: %s", e)

    def lookup(self, key: str) -> dict | None:
        """Return the entry if this upload is in-flight or already done.

        Failed uploads return None so the client's retry re-processes them.
        """
        entry = self._entries.get(key)
        if entry is None or entry.get("state") == STATE_FAILED:
            return None
        return entry

    def mark_processing(self, key: str, **meta) -> None:
        self._entries[key] = {
            "state": STATE_PROCESSING,
            "uploaded_at": datetime.now().isoformat(),
            **meta,
        }
        self._save()

    def mark_done(self, key: str, session_id: str | None, segments: int) -> None:
        entry = self._entries.setdefault(key, {})
        entry.update(
            state=STATE_DONE,
            session_id=session_id,
            segments=segments,
            finished_at=datetime.now().isoformat(),
        )
        entry.setdefault("uploaded_at", datetime.now().isoformat())
        self._maybe_prune()
        self._save()

    def mark_failed(self, key: str, error: str) -> None:
        entry = self._entries.setdefault(key, {})
        entry.update(
            state=STATE_FAILED,
            error=error[:200],
            finished_at=datetime.now().isoformat(),
        )
        entry.setdefault("uploaded_at", datetime.now().isoformat())
        self._maybe_prune()
        self._save()

    def _maybe_prune(self) -> None:
        """Runtime retention: _prune() was only ever called at load, so a
        long-lived container grew the registry (and its O(n) rewrites)
        without bound. Every 50 mutations is plenty."""
        self._mutations = getattr(self, "_mutations", 0) + 1
        if self._mutations >= 50:
            self._mutations = 0
            self._prune(drop_processing=False)

    def __len__(self) -> int:
        return len(self._entries)
