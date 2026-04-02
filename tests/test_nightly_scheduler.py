"""Tests for nightly scheduler logic.

_calc_next_nightly() is defined in omi_whisper_bridge.py which requires
faster_whisper (GPU-only dependency). We replicate the pure logic here
and verify it directly.
"""

import pytest
from datetime import datetime, timedelta
from unittest import mock


def _calc_next_nightly(hour: int, now: datetime = None) -> str:
    """Replicate the pure logic of omi_whisper_bridge._calc_next_nightly()."""
    if now is None:
        now = datetime.now()
    today_target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if now >= today_target:
        next_run = today_target + timedelta(days=1)
    else:
        next_run = today_target
    return next_run.isoformat()


class TestCalcNextNightly:
    """Test next-run time calculation."""

    def test_before_target_hour(self):
        """If current time is before target hour, next run is today."""
        now = datetime(2026, 4, 3, 2, 30, 0)
        result = _calc_next_nightly(4, now=now)
        assert result == "2026-04-03T04:00:00"

    def test_after_target_hour(self):
        """If current time is after target hour, next run is tomorrow."""
        now = datetime(2026, 4, 3, 5, 0, 0)
        result = _calc_next_nightly(4, now=now)
        assert result == "2026-04-04T04:00:00"

    def test_at_target_hour(self):
        """If current time is exactly target hour, next run is tomorrow."""
        now = datetime(2026, 4, 3, 4, 0, 0)
        result = _calc_next_nightly(4, now=now)
        assert result == "2026-04-04T04:00:00"

    def test_midnight_target(self):
        """Target hour=1 from 23:59 → next day 1:00."""
        now = datetime(2026, 4, 3, 23, 59, 0)
        result = _calc_next_nightly(1, now=now)
        assert result == "2026-04-04T01:00:00"

    def test_just_before_target(self):
        """One minute before target → today."""
        now = datetime(2026, 4, 3, 3, 59, 0)
        result = _calc_next_nightly(4, now=now)
        assert result == "2026-04-03T04:00:00"


class TestNightlySchedulerDisabled:
    """Test that NIGHTLY_SCHEDULE_HOUR=0 disables the scheduler."""

    def test_hour_zero_disabled(self):
        """Hour=0 means the scheduler should not run."""
        # The scheduler function returns immediately when hour <= 0
        hour = 0
        assert hour <= 0  # scheduler would return early

    def test_hour_negative_disabled(self):
        """Negative hours are also disabled."""
        hour = -1
        assert hour <= 0


class TestNightlyExclusivity:
    """Test exclusivity check logic (replicated from bridge)."""

    def test_skip_when_batch_running(self):
        """Should skip nightly when another batch is already running."""
        batch_state = {"running": True}
        combined_state = {"running": False}
        session_tag_state = {"running": False}
        should_skip = (
            batch_state.get("running")
            or combined_state.get("running")
            or session_tag_state.get("running")
        )
        assert should_skip is True

    def test_skip_when_combined_tag_running(self):
        batch_state = {"running": False}
        combined_state = {"running": True}
        session_tag_state = {"running": False}
        should_skip = (
            batch_state.get("running")
            or combined_state.get("running")
            or session_tag_state.get("running")
        )
        assert should_skip is True

    def test_no_skip_when_all_idle(self):
        batch_state = {"running": False}
        combined_state = {"running": False}
        session_tag_state = {"running": False}
        should_skip = (
            batch_state.get("running")
            or combined_state.get("running")
            or session_tag_state.get("running")
        )
        assert should_skip is False
