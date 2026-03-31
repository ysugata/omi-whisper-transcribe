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


def _create_test_session(tmpdir, session_id, segments):
    """Create a test session directory with transcript.json."""
    session_dir = os.path.join(tmpdir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    transcript = {
        "session": {
            "start": "2026-03-28T10:00:00",
            "duration_seconds": 60,
            "model": "large-v3",
            "language": "ja",
        },
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
