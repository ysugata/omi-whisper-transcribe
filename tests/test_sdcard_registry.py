"""Unit tests for sdcard_registry (dedup registry + timestamp validation)."""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sdcard_registry import (  # noqa: E402
    SdcardUploadRegistry,
    make_key,
    resolve_recorded_at,
    STATE_DONE,
    STATE_PROCESSING,
)


NOW = datetime(2026, 7, 15, 12, 0, 0)


# --- resolve_recorded_at ---

def test_recent_timestamp_is_kept():
    ts = int((NOW - timedelta(hours=5)).timestamp())
    recorded_at, device_dt = resolve_recorded_at(ts, max_age_days=7, now=NOW)
    assert recorded_at == datetime.fromtimestamp(ts)
    assert device_dt is None


def test_too_old_timestamp_falls_back_to_now():
    # Observed in production: device clock 79 days behind
    ts = int((NOW - timedelta(days=79)).timestamp())
    recorded_at, device_dt = resolve_recorded_at(ts, max_age_days=7, now=NOW)
    assert recorded_at == NOW
    assert device_dt == datetime.fromtimestamp(ts)


def test_future_timestamp_falls_back_to_now():
    ts = int((NOW + timedelta(days=3)).timestamp())
    recorded_at, device_dt = resolve_recorded_at(ts, max_age_days=7, now=NOW)
    assert recorded_at == NOW
    assert device_dt == datetime.fromtimestamp(ts)


def test_invalid_timestamp_falls_back_to_now():
    recorded_at, device_dt = resolve_recorded_at(-(2**62), max_age_days=7, now=NOW)
    assert recorded_at == NOW
    assert device_dt is None


def test_boundary_within_max_age_is_kept():
    ts = int((NOW - timedelta(days=6, hours=23)).timestamp())
    recorded_at, device_dt = resolve_recorded_at(ts, max_age_days=7, now=NOW)
    assert recorded_at == datetime.fromtimestamp(ts)
    assert device_dt is None


# --- make_key ---

def test_make_key_uses_content_hash_prefix():
    key = make_key("dev1", 1751500000, "a" * 64)
    assert key == f"dev1:1751500000:{'a' * 16}"


# --- SdcardUploadRegistry ---

def test_lookup_missing_returns_none(tmp_path):
    reg = SdcardUploadRegistry(tmp_path / "reg.json")
    assert reg.lookup("nope") is None


def test_processing_and_done_are_deduped(tmp_path):
    reg = SdcardUploadRegistry(tmp_path / "reg.json")
    reg.mark_processing("k1", device_id="dev1")
    assert reg.lookup("k1")["state"] == STATE_PROCESSING

    reg.mark_done("k1", "2026-07-15/12-00-00", 42)
    entry = reg.lookup("k1")
    assert entry["state"] == STATE_DONE
    assert entry["session_id"] == "2026-07-15/12-00-00"
    assert entry["segments"] == 42


def test_failed_upload_allows_retry(tmp_path):
    reg = SdcardUploadRegistry(tmp_path / "reg.json")
    reg.mark_processing("k1")
    reg.mark_failed("k1", "boom")
    assert reg.lookup("k1") is None  # retry re-processes


def test_done_persists_across_instances(tmp_path):
    path = tmp_path / "reg.json"
    reg = SdcardUploadRegistry(path)
    reg.mark_processing("k1")
    reg.mark_done("k1", "2026-07-15/12-00-00", 10)

    reg2 = SdcardUploadRegistry(path)
    assert reg2.lookup("k1")["state"] == STATE_DONE


def test_stale_processing_dropped_on_load(tmp_path):
    """A crash mid-processing must not block the retry forever."""
    path = tmp_path / "reg.json"
    reg = SdcardUploadRegistry(path)
    reg.mark_processing("k1")

    reg2 = SdcardUploadRegistry(path)  # simulates process restart
    assert reg2.lookup("k1") is None


def test_corrupt_registry_file_starts_empty(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text("{broken json", encoding="utf-8")
    reg = SdcardUploadRegistry(path)
    assert len(reg) == 0
    reg.mark_done("k1", None, 0)
    assert json.loads(path.read_text())["k1"]["state"] == STATE_DONE
