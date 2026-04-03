"""Tests for BatchCorrector — post-session async LLM correction."""

import asyncio
import json
import os
import tempfile
from unittest import mock

import pytest

from batch_corrector import (
    BatchCorrector,
    BatchJob,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_TAGGING,
    JOB_STATUS_TAG_SUBMITTED,
)
from llm_pipeline import LLMPipeline, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pipeline(corrected_texts=None, stages_run=None, api_calls=2):
    """Create a mock LLMPipeline that returns predefined results."""
    pipeline = mock.MagicMock(spec=LLMPipeline)
    pipeline.enabled = True

    def process_segments(segments, **kwargs):
        if corrected_texts:
            corrected = [dict(seg) for seg in segments]
            for i, text in enumerate(corrected_texts):
                if i < len(corrected) and text is not None:
                    corrected[i]["text"] = text
        else:
            corrected = [dict(seg) for seg in segments]
        return PipelineResult(
            corrected_segments=corrected,
            stages_run=stages_run or ["A", "B"],
            api_calls=api_calls,
        )

    pipeline.process_segments = process_segments
    return pipeline


def _create_test_session(tmpdir, session_id, segments, llm_corrected=False):
    """Create a test session directory with transcript.json."""
    session_dir = os.path.join(tmpdir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    session_meta = {
        "start": "2026-03-28T10:00:00",
        "duration_seconds": 60,
        "model": "large-v3",
        "language": "ja",
    }
    if llm_corrected:
        session_meta["llm_corrected"] = True
        session_meta["llm_corrected_at"] = "2026-03-28T10:05:00"
    transcript = {
        "session": session_meta,
        "segments": segments,
    }
    with open(os.path.join(session_dir, "transcript.json"), "w") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    return session_dir


# ---------------------------------------------------------------------------
# BatchJob tests
# ---------------------------------------------------------------------------

class TestBatchJob:
    def test_create(self):
        job = BatchJob(session_id="2026-03-28/10-00-00")
        assert job.status == JOB_STATUS_QUEUED
        assert job.created_at != ""
        assert job.session_id == "2026-03-28/10-00-00"

    def test_to_dict_from_dict(self):
        job = BatchJob(session_id="test", status=JOB_STATUS_SUCCEEDED, changes=5)
        d = job.to_dict()
        assert d["session_id"] == "test"
        assert d["changes"] == 5
        restored = BatchJob.from_dict(d)
        assert restored.session_id == "test"
        assert restored.changes == 5

    def test_legacy_fields_roundtrip(self):
        """Legacy tagging fields should survive serialization (backward compat)."""
        job = BatchJob(session_id="t1", tagged=True, tag_error="err")
        d = job.to_dict()
        assert d["tagged"] is True
        assert d["tag_error"] == "err"
        restored = BatchJob.from_dict(d)
        assert restored.tagged is True
        assert restored.tag_error == "err"


# ---------------------------------------------------------------------------
# BatchCorrector tests
# ---------------------------------------------------------------------------

class TestBatchCorrector:
    def test_enqueue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=["Docker"],
            )
            job = bc.enqueue("2026-03-28/10-00-00")
            assert job.status == JOB_STATUS_QUEUED

            status = bc.get_status()
            assert status["jobs"]["queued"] == 1
            assert status["total"] == 1

    def test_enqueue_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
            )
            bc.enqueue("session1")
            bc.enqueue("session1")  # duplicate
            assert bc.get_status()["total"] == 1

    def test_queue_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()

            bc1 = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            bc1.enqueue("session1")
            bc1.enqueue("session2")

            # Create new instance — should load from disk
            bc2 = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            assert bc2.get_status()["total"] == 2

    def test_running_job_reset_on_restart(self):
        """Running jobs should be reset to queued on restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)

            # Write a queue with a running job
            queue = [BatchJob(session_id="s1", status=JOB_STATUS_RUNNING).to_dict()]
            with open(os.path.join(config_dir, "batch_llm_queue.json"), "w") as f:
                json.dump(queue, f)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            jobs = bc.get_jobs()
            assert jobs[0]["status"] == JOB_STATUS_QUEUED

    def test_legacy_tagging_migrated_to_succeeded(self):
        """Legacy TAGGING/TAG_SUBMITTED jobs migrate to SUCCEEDED on restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)

            queue = [
                BatchJob(session_id="s1", status=JOB_STATUS_TAGGING).to_dict(),
                BatchJob(session_id="s2", status=JOB_STATUS_TAG_SUBMITTED).to_dict(),
            ]
            with open(os.path.join(config_dir, "batch_llm_queue.json"), "w") as f:
                json.dump(queue, f)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            jobs = bc.get_jobs()
            assert all(j["status"] == JOB_STATUS_SUCCEEDED for j in jobs)

    def test_retry_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            bc.enqueue("s1")
            bc._jobs["s1"].status = JOB_STATUS_FAILED
            bc._jobs["s1"].error = "test error"

            count = bc.retry_failed()
            assert count == 1
            assert bc._jobs["s1"].status == JOB_STATUS_QUEUED

    def test_get_jobs_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            bc.enqueue("s1")
            bc.enqueue("s2")
            bc._jobs["s2"].status = JOB_STATUS_SUCCEEDED

            queued = bc.get_jobs(status=JOB_STATUS_QUEUED)
            assert len(queued) == 1
            assert queued[0]["session_id"] == "s1"

    def test_get_status_no_tagging_keys(self):
        """get_status should not include tagging/tag_submitted counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir, config_dir=config_dir,
                pipeline=pipeline, known_terms=[],
            )
            status = bc.get_status()
            assert "tagging" not in status["jobs"]
            assert "tag_submitted" not in status["jobs"]


# ---------------------------------------------------------------------------
# Async processing tests
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    @pytest.mark.asyncio
    async def test_process_sync_success(self):
        """Process a job and verify transcript.json is updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [
                {"text": "ドッカーのコンテナ", "start": 0.0, "end": 2.0},
                {"text": "テストテキスト", "start": 2.0, "end": 4.0},
            ]
            _create_test_session(recordings_dir, "2026-03-28/10-00-00", segments)

            pipeline = _make_mock_pipeline(
                corrected_texts=["Dockerのコンテナ", None],
                stages_run=["A", "B"],
            )
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=["Docker"],
            )

            job = bc.enqueue("2026-03-28/10-00-00")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            assert job.changes == 1
            assert job.stages_run == ["A", "B"]

            # Verify transcript.json was updated
            with open(os.path.join(recordings_dir, "2026-03-28/10-00-00/transcript.json")) as f:
                data = json.load(f)
            assert data["segments"][0]["text"] == "Dockerのコンテナ"
            assert data["session"]["llm_corrected"] is True

    @pytest.mark.asyncio
    async def test_process_sync_no_changes(self):
        """No changes → job succeeds but transcript unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, "s1", segments)

            pipeline = _make_mock_pipeline()  # no changes
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            assert job.changes == 0

    @pytest.mark.asyncio
    async def test_process_sync_missing_transcript(self):
        """Missing transcript.json → job fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=tmpdir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
            )

            job = bc.enqueue("nonexistent/session")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_FAILED
            assert "not found" in job.error

    @pytest.mark.asyncio
    async def test_correction_goes_directly_to_succeeded(self):
        """Correction → SUCCEEDED directly (no TAGGING phase)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [
                {"text": "ドッカーのコンテナ", "start": 0.0, "end": 2.0},
            ]
            _create_test_session(recordings_dir, "s1", segments)

            pipeline = _make_mock_pipeline(
                corrected_texts=["Dockerのコンテナ"],
                stages_run=["A", "B"],
            )
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=["Docker"],
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            # Goes directly to SUCCEEDED, never to TAGGING
            assert job.status == JOB_STATUS_SUCCEEDED
            assert job.changes == 1

    @pytest.mark.asyncio
    async def test_skip_correction_if_already_done(self):
        """Sessions with llm_corrected=True skip correction → SUCCEEDED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [{"text": "Dockerのコンテナ", "start": 0.0, "end": 2.0}]
            _create_test_session(recordings_dir, "s1", segments, llm_corrected=True)

            pipeline = mock.MagicMock(spec=LLMPipeline)
            pipeline.enabled = True

            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            pipeline.process_segments.assert_not_called()


# ---------------------------------------------------------------------------
# Screening integration tests
# ---------------------------------------------------------------------------

class TestRecovery:
    """Tests for _recover_unprocessed_sessions()."""

    def test_recover_unprocessed_sessions(self):
        """Unprocessed sessions (no llm_corrected) are auto-enqueued on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            from datetime import datetime, timedelta
            today = datetime.now().strftime("%Y-%m-%d")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, f"{today}/10-00-00", segments)
            _create_test_session(recordings_dir, f"{today}/11-00-00", segments, llm_corrected=True)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=7,
            )

            # Only the uncorrected session should be in the queue
            assert bc.get_status()["total"] == 1
            jobs = bc.get_jobs()
            assert jobs[0]["session_id"] == f"{today}/10-00-00"

    def test_recovery_skips_already_queued(self):
        """Sessions already in queue are not duplicated by recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, f"{today}/10-00-00", segments)

            # Pre-populate queue with this session
            queue = [BatchJob(session_id=f"{today}/10-00-00").to_dict()]
            with open(os.path.join(config_dir, "batch_llm_queue.json"), "w") as f:
                json.dump(queue, f)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=7,
            )

            # Should NOT duplicate
            assert bc.get_status()["total"] == 1

    def test_recovery_requeues_failed(self):
        """Failed sessions are re-enqueued by recovery scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, f"{today}/10-00-00", segments)

            # Pre-populate with failed job
            queue = [BatchJob(session_id=f"{today}/10-00-00", status=JOB_STATUS_FAILED).to_dict()]
            with open(os.path.join(config_dir, "batch_llm_queue.json"), "w") as f:
                json.dump(queue, f)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=7,
            )

            jobs = bc.get_jobs()
            assert len(jobs) == 1
            assert jobs[0]["status"] == JOB_STATUS_QUEUED

    def test_recovery_skips_empty_sessions(self):
        """Sessions with 0 segments are not recovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")

            _create_test_session(recordings_dir, f"{today}/10-00-00", [])  # empty

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=7,
            )

            assert bc.get_status()["total"] == 0

    def test_recovery_skips_old_sessions(self):
        """Sessions older than recovery_scan_days are not recovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, "2020-01-01/10-00-00", segments)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=7,
            )

            assert bc.get_status()["total"] == 0

    def test_recovery_disabled_when_zero_days(self):
        """recovery_scan_days=0 disables recovery scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")

            segments = [{"text": "テスト", "start": 0.0, "end": 1.0}]
            _create_test_session(recordings_dir, f"{today}/10-00-00", segments)

            pipeline = _make_mock_pipeline()
            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                recovery_scan_days=0,
            )

            assert bc.get_status()["total"] == 0


class TestBatchScreening:
    @pytest.mark.asyncio
    async def test_all_clean_skips_llm(self):
        """When screener says all clean → LLM pipeline not called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [
                {"text": "今日は天気がいいですね、散歩に行きましょう", "start": 0.0, "end": 3.0},
            ]
            _create_test_session(recordings_dir, "s1", segments)

            pipeline = mock.MagicMock(spec=LLMPipeline)
            pipeline.enabled = True

            # Create a mock screener that returns all clean
            from segment_screener import SegmentScreener, ScreeningResult
            from correction_metrics import CorrectionMetrics
            screener = mock.MagicMock(spec=SegmentScreener)
            screener.screen.return_value = ScreeningResult(
                segments=[dict(seg) for seg in segments],
                dirty_indices=[],
                clean_indices=[0],
                metrics=CorrectionMetrics(total_segments=1, screened_out=1),
            )

            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                segment_screener=screener,
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            # Pipeline should NOT have been called (all clean)
            pipeline.process_segments.assert_not_called()
            assert job.segments_screened == 1
            assert job.segments_dirty == 0

    @pytest.mark.asyncio
    async def test_dirty_segments_sent_to_llm(self):
        """When screener finds dirty segments → only those sent to LLM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [
                {"text": "今日は天気がいいですね、散歩に行きましょう", "start": 0.0, "end": 3.0},
                {"text": "ドッカーのコンテナを起動してください", "start": 3.0, "end": 6.0},
            ]
            _create_test_session(recordings_dir, "s1", segments)

            pipeline = _make_mock_pipeline(
                corrected_texts=[None, "Dockerのコンテナを起動してください"],
                stages_run=["B"],
                api_calls=1,
            )

            from segment_screener import SegmentScreener, ScreeningResult
            from correction_metrics import CorrectionMetrics
            screener = mock.MagicMock(spec=SegmentScreener)
            screener.screen.return_value = ScreeningResult(
                segments=[dict(seg) for seg in segments],
                dirty_indices=[1],
                clean_indices=[0],
                metrics=CorrectionMetrics(
                    total_segments=2, screened_out=1, dirty_segments=1,
                ),
            )

            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=["Docker"],
                segment_screener=screener,
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            assert job.segments_screened == 2
            assert job.segments_dirty == 1

    @pytest.mark.asyncio
    async def test_no_screener_full_pipeline(self):
        """Without screener, all segments go to LLM (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir)
            recordings_dir = os.path.join(tmpdir, "recordings")

            segments = [
                {"text": "テストテキストです、確認してください", "start": 0.0, "end": 2.0},
            ]
            _create_test_session(recordings_dir, "s1", segments)

            pipeline = _make_mock_pipeline(stages_run=["B"], api_calls=1)

            bc = BatchCorrector(
                recordings_dir=recordings_dir,
                config_dir=config_dir,
                pipeline=pipeline,
                known_terms=[],
                segment_screener=None,  # No screener
            )

            job = bc.enqueue("s1")
            await bc._process_sync(job)

            assert job.status == JOB_STATUS_SUCCEEDED
            assert job.segments_screened == 0  # No screening
