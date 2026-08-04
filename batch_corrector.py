"""Batch LLM corrector: post-session async transcript correction.

Manages a job queue that processes sessions after finalization.
Jobs persist to JSON so they survive container restarts.

Supports two execution modes:
- "batch": Gemini Batch API (50% cost, async 24h turnaround)
- "sync": Standard synchronous API (OpenAI-compatible, immediate)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job data model
# ---------------------------------------------------------------------------

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_SUCCEEDED = "succeeded"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_SUBMITTED = "submitted"  # Gemini Batch API: waiting for results
JOB_STATUS_TAGGING = "tagging"  # sync: tagging in progress
JOB_STATUS_TAG_SUBMITTED = "tag_submitted"  # batch: tagging batch submitted


@dataclass
class BatchJob:
    session_id: str  # e.g. "2026-03-28/05-46-37"
    status: str = JOB_STATUS_QUEUED
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    stages_run: list = field(default_factory=list)
    api_calls: int = 0
    changes: int = 0
    retried: bool = False
    stage_c_triggered: bool = False
    stage_c_reason: str = ""
    batch_job_name: str = ""  # Gemini Batch API job name
    mode: str = "sync"  # "batch" or "sync"
    tag_batch_job_name: str = ""  # Gemini Batch API job name for tagging
    tagged: bool = False  # tagging completed
    tag_error: str = ""  # tagging error (correction succeeded but tagging failed)
    segments_screened: int = 0  # total segments screened
    segments_dirty: int = 0    # segments sent to LLM
    dict_corrections: int = 0  # dictionary/learned corrections applied

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BatchJob":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Batch corrector
# ---------------------------------------------------------------------------

class BatchCorrector:
    """Async job queue for post-session LLM transcript correction.

    Supports Gemini Batch API (mode="batch") and sync API (mode="sync").
    """

    def __init__(
        self,
        recordings_dir: str,
        config_dir: str,
        pipeline,  # LLMPipeline instance (used for sync mode + result processing)
        known_terms: list,
        hotword_loader=None,
        stage_c_auto: bool = True,
        min_text_length: int = 20,
        mode: str = "sync",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-3.1-flash-lite",
        batch_poll_interval: int = 60,
        segment_screener=None,
        recovery_scan_days: int = 7,
    ):
        self._recordings_dir = recordings_dir
        self._config_dir = config_dir
        self._pipeline = pipeline
        self._known_terms = known_terms
        self._hotword_loader = hotword_loader
        self._stage_c_auto = stage_c_auto
        self._min_text_length = min_text_length
        self._mode = mode
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model
        self._batch_poll_interval = batch_poll_interval
        self._segment_screener = segment_screener
        self._recovery_scan_days = recovery_scan_days

        self._queue_path = os.path.join(config_dir, "batch_llm_queue.json")
        self._jobs: dict[str, BatchJob] = {}  # session_id -> BatchJob
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._current_session: Optional[str] = None
        self._genai_client = None

        # Initialize Gemini genai client for batch mode
        if mode == "batch" and gemini_api_key:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=gemini_api_key)
                log.info("BatchCorrector: Gemini Batch API client initialized")
            except Exception as e:
                log.warning("BatchCorrector: Gemini genai init failed, falling back to sync: %s", e)
                self._mode = "sync"

        self._load_queue()
        self._recover_unprocessed_sessions()

    # ------------------------------------------------------------------
    # Queue persistence
    # ------------------------------------------------------------------

    def _load_queue(self):
        """Load job queue from disk."""
        if not os.path.exists(self._queue_path):
            return
        try:
            with open(self._queue_path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                job = BatchJob.from_dict(d)
                # Reset running jobs back to queued (container restart)
                if job.status == JOB_STATUS_RUNNING:
                    job.status = JOB_STATUS_QUEUED
                    job.started_at = ""
                # Migrate legacy TAGGING/TAG_SUBMITTED → SUCCEEDED (correction already done)
                elif job.status in (JOB_STATUS_TAGGING, JOB_STATUS_TAG_SUBMITTED):
                    job.status = JOB_STATUS_SUCCEEDED
                    if not job.finished_at:
                        job.finished_at = datetime.now().isoformat()
                self._jobs[job.session_id] = job

            # Prune terminal jobs whose recordings were deleted by retention:
            # without this the queue file grows forever (observed: 2341 jobs /
            # 1.4MB spanning 4 months) and every state change rewrites all of
            # it synchronously on the event loop. Only dir-less jobs are safe
            # to drop — existing dirs may lack the llm_corrected flag and
            # would be re-enqueued by _recover_unprocessed_sessions.
            terminal = (JOB_STATUS_SUCCEEDED, JOB_STATUS_FAILED)
            before = len(self._jobs)
            self._jobs = {
                sid: j for sid, j in self._jobs.items()
                if j.status not in terminal
                or os.path.isdir(os.path.join(self._recordings_dir, sid))
            }
            pruned = before - len(self._jobs)
            if pruned:
                log.info("BatchCorrector: pruned %d job(s) for deleted sessions", pruned)
                self._save_queue()

            queued = sum(1 for j in self._jobs.values() if j.status == JOB_STATUS_QUEUED)
            submitted = sum(1 for j in self._jobs.values() if j.status == JOB_STATUS_SUBMITTED)
            if queued or submitted:
                log.info(
                    "BatchCorrector: loaded %d queued, %d submitted job(s)",
                    queued, submitted,
                )
        except (json.JSONDecodeError, OSError) as e:
            log.warning("BatchCorrector: failed to load queue: %s", e)

    def _save_queue(self):
        """Persist job queue to disk."""
        try:
            data = [j.to_dict() for j in self._jobs.values()]
            tmp = self._queue_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._queue_path)
        except OSError as e:
            log.warning("BatchCorrector: failed to save queue: %s", e)

    def _recover_unprocessed_sessions(self):
        """Scan recent recordings and enqueue sessions that were never processed.

        Catches sessions that were created during container restarts
        (enqueue() was never called) or sessions whose jobs were lost.
        """
        if self._recovery_scan_days <= 0:
            return

        base = Path(self._recordings_dir)
        if not base.exists():
            return

        cutoff = datetime.now().timestamp() - (self._recovery_scan_days * 86400)
        recovered = 0

        for day_dir in sorted(base.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day_date = datetime.strptime(day_dir.name, "%Y-%m-%d")
            except ValueError:
                continue
            if day_date.timestamp() < cutoff:
                continue

            for session_dir in sorted(day_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                transcript_path = session_dir / "transcript.json"
                if not transcript_path.exists():
                    continue

                session_id = f"{day_dir.name}/{session_dir.name}"

                # Skip if already in queue and not failed
                if session_id in self._jobs:
                    job = self._jobs[session_id]
                    if job.status in (
                        JOB_STATUS_QUEUED, JOB_STATUS_RUNNING,
                        JOB_STATUS_SUBMITTED, JOB_STATUS_SUCCEEDED,
                    ):
                        continue

                # Check transcript for llm_corrected flag
                try:
                    with open(transcript_path, encoding="utf-8") as f:
                        data = json.load(f)
                    session_meta = data.get("session", {})
                    segments = data.get("segments", [])

                    if not segments:
                        continue
                    if session_meta.get("llm_corrected"):
                        continue

                    # Unprocessed session found — enqueue it
                    job = BatchJob(session_id=session_id, mode=self._mode)
                    self._jobs[session_id] = job
                    recovered += 1

                except (json.JSONDecodeError, OSError):
                    continue

        if recovered:
            self._save_queue()
            log.info(
                "BatchCorrector: recovered %d unprocessed session(s) "
                "from last %d day(s)",
                recovered, self._recovery_scan_days,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, session_id: str) -> BatchJob:
        """Add a session to the correction queue."""
        if session_id in self._jobs:
            existing = self._jobs[session_id]
            if existing.status in (
                JOB_STATUS_QUEUED, JOB_STATUS_RUNNING, JOB_STATUS_SUBMITTED,
            ):
                log.info("BatchCorrector: session %s already %s", session_id, existing.status)
                return existing
            # Re-queue failed/succeeded jobs
            log.info("BatchCorrector: re-queuing session %s (was %s)", session_id, existing.status)

        job = BatchJob(session_id=session_id, mode=self._mode)
        self._jobs[session_id] = job
        self._save_queue()
        log.info("BatchCorrector: enqueued session %s (mode=%s)", session_id, self._mode)
        return job

    def get_status(self) -> dict:
        """Get overall batch status."""
        counts = {JOB_STATUS_QUEUED: 0, JOB_STATUS_RUNNING: 0,
                  JOB_STATUS_SUCCEEDED: 0, JOB_STATUS_FAILED: 0,
                  JOB_STATUS_SUBMITTED: 0}
        for j in self._jobs.values():
            counts[j.status] = counts.get(j.status, 0) + 1
        return {
            "running": self._running,
            "mode": self._mode,
            "screening_enabled": self._segment_screener is not None,
            "current_session": self._current_session,
            "jobs": counts,
            "total": len(self._jobs),
        }

    def get_jobs(self, status: Optional[str] = None, limit: int = 50) -> list:
        """Get job list, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: (
            0 if j.status == JOB_STATUS_RUNNING else
            1 if j.status == JOB_STATUS_SUBMITTED else
            2 if j.status == JOB_STATUS_QUEUED else
            3 if j.status == JOB_STATUS_FAILED else 4,
            j.created_at,
        ))
        return [j.to_dict() for j in jobs[:limit]]

    def retry_failed(self, session_id: Optional[str] = None) -> int:
        """Re-queue failed jobs. Returns count of re-queued jobs."""
        count = 0
        for sid, job in self._jobs.items():
            if job.status == JOB_STATUS_FAILED:
                if session_id is None or sid == session_id:
                    job.status = JOB_STATUS_QUEUED
                    job.error = ""
                    job.started_at = ""
                    job.finished_at = ""
                    job.batch_job_name = ""
                    count += 1
        if count:
            self._save_queue()
            log.info("BatchCorrector: re-queued %d failed job(s)", count)
        return count

    def process_now(self, session_id: str) -> BatchJob:
        """Force-enqueue a session for immediate processing."""
        return self.enqueue(session_id)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def start(self):
        """Start the background worker loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._worker_loop())
        log.info("BatchCorrector: worker started (mode=%s)", self._mode)

    async def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        log.info("BatchCorrector: worker stopped")

    async def _worker_loop(self):
        """Main worker loop: poll for queued jobs and check submitted jobs.

        個々のジョブ処理外で例外が漏れるとワーカーが無言で死に、以降
        LLM 補正が一切走らなくなるため、ループ全体を防御する。
        """
        try:
            while self._running:
                try:
                    # 1. Process queued jobs (correction phase)
                    job = self._next_queued_job()
                    if job is not None:
                        if self._mode == "batch":
                            await self._submit_batch_job(job)
                        else:
                            await self._process_sync(job)
                        continue

                    # 2. Check submitted batch jobs for completion
                    if self._mode == "batch":
                        submitted = [j for j in self._jobs.values()
                                     if j.status == JOB_STATUS_SUBMITTED]
                        for sjob in submitted:
                            await self._check_batch_result(sjob)
                except Exception:
                    log.exception("BatchCorrector: worker loop error (continuing)")

                await asyncio.sleep(5 if self._mode == "sync" else self._batch_poll_interval)
        finally:
            self._running = False

    def _next_queued_job(self) -> Optional[BatchJob]:
        """Get the next queued job (FIFO by created_at)."""
        queued = [
            j for j in self._jobs.values()
            if j.status == JOB_STATUS_QUEUED
        ]
        if not queued:
            return None
        queued.sort(key=lambda j: j.created_at)
        return queued[0]

    # ------------------------------------------------------------------
    # Sync mode: process using LLMPipeline directly
    # ------------------------------------------------------------------

    async def _process_sync(self, job: BatchJob):
        """Process a single job using synchronous API calls."""
        session_id = job.session_id
        self._current_session = session_id
        job.status = JOB_STATUS_RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_queue()

        log.info("BatchCorrector [sync]: processing session %s", session_id)

        try:
            segments, transcript_data, transcript_path = self._load_transcript(session_id)
            if segments is None:
                job.status = JOB_STATUS_SUCCEEDED
                job.changes = 0
                job.finished_at = datetime.now().isoformat()
                self._save_queue()
                self._current_session = None
                return

            # Skip correction if already done (restart recovery)
            already_corrected = transcript_data.get("session", {}).get("llm_corrected", False)
            if already_corrected:
                log.info("BatchCorrector [sync]: session %s already corrected, skipping", session_id)
            else:
                glossary = self._build_session_glossary(segments)

                # Screening: deterministic pre-processing + classification
                dirty_indices = None
                screened_segments = segments
                if self._segment_screener:
                    screening_result = self._segment_screener.screen(segments)
                    screened_segments = screening_result.segments
                    dirty_indices = screening_result.dirty_indices
                    job.segments_screened = screening_result.metrics.total_segments
                    job.segments_dirty = len(dirty_indices)
                    job.dict_corrections = (
                        screening_result.metrics.dict_corrections
                        + screening_result.metrics.learned_corrections
                    )

                    if not dirty_indices:
                        # All segments clean → skip LLM entirely
                        log.info(
                            "BatchCorrector [sync]: session %s all clean, skipping LLM",
                            session_id,
                        )
                        self._apply_and_save(
                            job, segments, transcript_data, transcript_path,
                            screened_segments,
                            stages_run=["screened"],
                            api_calls=0,
                            retried=False,
                        )
                        job.status = JOB_STATUS_SUCCEEDED
                        job.finished_at = datetime.now().isoformat()
                        self._save_queue()
                        self._current_session = None
                        return

                loop = asyncio.get_running_loop()
                pipeline_result = await loop.run_in_executor(
                    None,
                    lambda: self._pipeline.process_segments(
                        screened_segments,
                        known_terms=self._known_terms,
                        session_glossary=glossary,
                        min_text_length=self._min_text_length,
                        auto_c=self._stage_c_auto,
                        segment_indices=dirty_indices,
                    ),
                )

                self._apply_and_save(
                    job, segments, transcript_data, transcript_path,
                    pipeline_result.corrected_segments,
                    pipeline_result.stages_run,
                    pipeline_result.api_calls,
                    pipeline_result.retried,
                )

            job.status = JOB_STATUS_SUCCEEDED
            job.finished_at = datetime.now().isoformat()

        except Exception as e:
            log.error("BatchCorrector [sync]: session %s failed: %s", session_id, e)
            job.status = JOB_STATUS_FAILED
            job.error = str(e)
            job.finished_at = datetime.now().isoformat()

        self._save_queue()
        self._current_session = None

    # ------------------------------------------------------------------
    # Batch mode: submit to Gemini Batch API
    # ------------------------------------------------------------------

    async def _submit_batch_job(self, job: BatchJob):
        """Submit a session to Gemini Batch API."""
        session_id = job.session_id
        self._current_session = session_id
        job.status = JOB_STATUS_RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_queue()

        log.info("BatchCorrector [batch]: submitting session %s", session_id)

        try:
            segments, transcript_data, transcript_path = self._load_transcript(session_id)
            if segments is None:
                job.status = JOB_STATUS_SUCCEEDED
                job.changes = 0
                job.finished_at = datetime.now().isoformat()
                self._save_queue()
                self._current_session = None
                return

            glossary = self._build_session_glossary(segments)
            requests = self._build_batch_requests(segments, glossary)

            if not requests:
                job.status = JOB_STATUS_SUCCEEDED
                job.changes = 0
                job.finished_at = datetime.now().isoformat()
                self._save_queue()
                self._current_session = None
                return

            loop = asyncio.get_running_loop()
            batch_result = await loop.run_in_executor(
                None,
                lambda: self._genai_client.batches.create(
                    model=self._gemini_model,
                    src=requests,
                    config={"display_name": f"omi-llm-{session_id.replace('/', '-')}"},
                ),
            )

            job.batch_job_name = batch_result.name
            job.status = JOB_STATUS_SUBMITTED
            log.info(
                "BatchCorrector [batch]: submitted %s → %s (%d request(s))",
                session_id, batch_result.name, len(requests),
            )

        except Exception as e:
            log.error("BatchCorrector [batch]: submit failed for %s: %s", session_id, e)
            job.status = JOB_STATUS_FAILED
            job.error = str(e)
            job.finished_at = datetime.now().isoformat()

        self._save_queue()
        self._current_session = None

    async def _check_batch_result(self, job: BatchJob):
        """Check if a submitted Gemini Batch job has completed."""
        if not self._genai_client or not job.batch_job_name:
            return

        try:
            loop = asyncio.get_running_loop()
            batch_job = await loop.run_in_executor(
                None,
                lambda: self._genai_client.batches.get(name=job.batch_job_name),
            )

            state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

            if state == "JOB_STATE_SUCCEEDED":
                log.info(
                    "BatchCorrector [batch]: %s completed → %s",
                    job.session_id, job.batch_job_name,
                )
                await self._process_batch_result(job, batch_job)

            elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                error_msg = getattr(batch_job, 'error', state)
                log.error(
                    "BatchCorrector [batch]: %s failed (%s): %s",
                    job.session_id, state, error_msg,
                )
                job.status = JOB_STATUS_FAILED
                job.error = f"Gemini Batch {state}: {error_msg}"
                job.finished_at = datetime.now().isoformat()
                self._save_queue()

            else:
                # Still pending/running
                log.debug(
                    "BatchCorrector [batch]: %s state=%s",
                    job.session_id, state,
                )

        except Exception as e:
            log.warning("BatchCorrector [batch]: poll error for %s: %s", job.session_id, e)

    async def _process_batch_result(self, job: BatchJob, batch_job):
        """Process completed Gemini Batch API result and update transcript."""
        session_id = job.session_id

        try:
            segments, transcript_data, transcript_path = self._load_transcript(session_id)
            if segments is None:
                job.status = JOB_STATUS_SUCCEEDED
                job.changes = 0
                job.finished_at = datetime.now().isoformat()
                self._save_queue()
                return

            # Extract responses from batch result
            responses = []
            if hasattr(batch_job, 'dest') and batch_job.dest:
                if hasattr(batch_job.dest, 'inlined_responses') and batch_job.dest.inlined_responses:
                    for inline_resp in batch_job.dest.inlined_responses:
                        if hasattr(inline_resp, 'response') and inline_resp.response:
                            text = None
                            if hasattr(inline_resp.response, 'text'):
                                text = inline_resp.response.text
                            elif hasattr(inline_resp.response, 'candidates'):
                                cands = inline_resp.response.candidates
                                if cands and cands[0].content and cands[0].content.parts:
                                    text = cands[0].content.parts[0].text
                            responses.append(text)
                        else:
                            responses.append(None)

            if not responses:
                log.warning("BatchCorrector [batch]: no responses for %s", session_id)
                job.status = JOB_STATUS_FAILED
                job.error = "No responses from Gemini Batch API"
                job.finished_at = datetime.now().isoformat()
                self._save_queue()
                return

            # Parse responses and apply corrections using pipeline logic
            corrected_segments = self._apply_batch_responses(segments, responses)

            stages = ["A", "B"] if len(responses) >= 2 else ["B"]
            self._apply_and_save(
                job, segments, transcript_data, transcript_path,
                corrected_segments, stages, len(responses), False,
            )

            job.status = JOB_STATUS_SUCCEEDED
            job.finished_at = datetime.now().isoformat()

        except Exception as e:
            log.error("BatchCorrector [batch]: result processing failed for %s: %s", session_id, e)
            job.status = JOB_STATUS_FAILED
            job.error = str(e)
            job.finished_at = datetime.now().isoformat()

        self._save_queue()

    @staticmethod
    def _strip_additional_properties(schema: dict) -> dict:
        """Recursively strip additionalProperties from a JSON Schema.

        Gemini Batch API does not support this keyword and rejects it
        with 400 INVALID_ARGUMENT.
        """
        import copy
        s = copy.deepcopy(schema)

        def _strip(obj):
            if isinstance(obj, dict):
                obj.pop("additionalProperties", None)
                for v in obj.values():
                    _strip(v)
            elif isinstance(obj, list):
                for v in obj:
                    _strip(v)

        _strip(s)
        return s

    def _build_batch_requests(self, segments: list, glossary: list) -> list:
        """Build inline requests for Gemini Batch API from pipeline stages."""
        from llm_pipeline_prompt import CANDIDATE_SYSTEM_PROMPT, CORRECTION_SYSTEM_PROMPT
        from llm_pipeline_schema import CANDIDATE_OBJECT_SCHEMA, CORRECTION_OBJECT_SCHEMA

        numbered_text, original_by_line, _ = self._pipeline._build_numbered_input(
            segments, self._min_text_length,
        )
        if not numbered_text:
            return []

        requests = []

        # Build combined terms list
        all_terms = list(glossary)
        if self._known_terms:
            for t in self._known_terms:
                if t not in all_terms:
                    all_terms.append(t)
        terms_str = ", ".join(all_terms[:50]) if all_terms else ""

        stages = self._pipeline._stages

        # Stage 3-A: candidate extraction
        if "A" in stages:
            parts = []
            if terms_str:
                parts.append(f"参考用語リスト: {terms_str}")
            parts.append(numbered_text)
            user_msg_a = "\n\n".join(parts)

            requests.append({
                "contents": [{
                    "parts": [{"text": user_msg_a}],
                    "role": "user",
                }],
                "config": {
                    "system_instruction": {"parts": [{"text": CANDIDATE_SYSTEM_PROMPT}]},
                    "response_mime_type": "application/json",
                    "response_schema": self._strip_additional_properties(CANDIDATE_OBJECT_SCHEMA),
                },
            })

        # Stage 3-B: notation correction
        if "B" in stages:
            parts = []
            if terms_str:
                parts.append(f"参考用語リスト: {terms_str}")
            parts.append(numbered_text)
            user_msg_b = "\n\n".join(parts)

            requests.append({
                "contents": [{
                    "parts": [{"text": user_msg_b}],
                    "role": "user",
                }],
                "config": {
                    "system_instruction": {"parts": [{"text": CORRECTION_SYSTEM_PROMPT}]},
                    "response_mime_type": "application/json",
                    "response_schema": self._strip_additional_properties(CORRECTION_OBJECT_SCHEMA),
                },
            })

        return requests

    def _apply_batch_responses(self, segments: list, responses: list) -> list:
        """Parse batch API responses and apply corrections."""
        corrected = [dict(seg) for seg in segments]
        stages = self._pipeline._stages

        _, original_by_line, line_to_seg_idx = self._pipeline._build_numbered_input(
            segments, self._min_text_length,
        )

        resp_idx = 0

        # Parse 3-A response (candidates — informational only for now)
        candidates_text = None
        if "A" in stages and resp_idx < len(responses):
            candidates_text = responses[resp_idx]
            resp_idx += 1
            if candidates_text:
                try:
                    data = json.loads(candidates_text)
                    num_candidates = len(data.get("candidates", []))
                    log.info("BatchCorrector [batch] 3-A: %d candidate(s)", num_candidates)
                except (json.JSONDecodeError, TypeError):
                    log.warning("BatchCorrector [batch] 3-A: invalid JSON response")

        # Parse 3-B response (corrections)
        if "B" in stages and resp_idx < len(responses):
            corrections_text = responses[resp_idx]
            resp_idx += 1
            if corrections_text:
                try:
                    data = json.loads(corrections_text)
                    corrections = data.get("corrections", [])
                    corrections = self._pipeline._filter_corrections(corrections, original_by_line)
                    corrected = self._pipeline._apply_corrections(
                        segments, corrections, line_to_seg_idx, original_by_line,
                    )
                    log.info("BatchCorrector [batch] 3-B: %d correction(s) applied", len(corrections))
                except (json.JSONDecodeError, TypeError) as e:
                    log.warning("BatchCorrector [batch] 3-B: parse error: %s", e)

        return corrected

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _load_transcript(self, session_id: str):
        """Load transcript.json for a session. Returns (segments, data, path) or (None,None,None)."""
        transcript_path = os.path.join(
            self._recordings_dir, session_id, "transcript.json",
        )
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"transcript.json not found: {transcript_path}")

        with open(transcript_path, encoding="utf-8") as f:
            transcript_data = json.load(f)
        segments = transcript_data.get("segments", [])

        if not segments:
            log.info("BatchCorrector: session %s has no segments, skipping", session_id)
            return None, None, None

        return segments, transcript_data, transcript_path

    def _apply_and_save(
        self, job, segments, transcript_data, transcript_path,
        corrected_segments, stages_run, api_calls, retried,
    ):
        """Apply corrections to transcript and save to disk."""
        changed_idx = [
            i for i, (orig, corr) in enumerate(zip(segments, corrected_segments))
            if orig.get("text", "") != corr.get("text", "")
        ]
        changes = len(changed_idx)

        job.stages_run = stages_run
        job.api_calls = api_calls
        job.changes = changes
        job.retried = retried
        job.stage_c_triggered = "C" in stages_run

        # Apply to the snapshot lists (used by _save_suggestions below)
        for i in changed_idx:
            segments[i]["text"] = corrected_segments[i].get("text", "")

        # Re-read before write (lost-update guard): transcript_data was loaded
        # BEFORE the LLM round-trip (30s〜数分)。その間に tagging バッチや
        # 話者修正が書いた内容をスナップショットで上書きしないよう、
        # 変更したフィールドだけを fresh に接ぎ木する。
        out_data = transcript_data
        try:
            with open(transcript_path, encoding="utf-8") as f:
                fresh = json.load(f)
        except (json.JSONDecodeError, OSError):
            fresh = None
        if fresh is not None and len(fresh.get("segments", [])) == len(segments):
            for i in changed_idx:
                fresh["segments"][i]["text"] = corrected_segments[i].get("text", "")
            out_data = fresh

        # llm_corrected は変更ゼロでも必ず書く: このフラグが無いと
        # _recover_unprocessed_sessions が再起動のたびに同じセッションを
        # 再エンキューし、LLM 課金を払い直す (queue 履歴だけが再処理を
        # 防いでいたため、履歴の prune もできなかった)。
        session_meta = out_data.setdefault("session", {})
        session_meta["llm_corrected"] = True
        session_meta["llm_corrected_at"] = datetime.now().isoformat()
        session_meta["llm_stages"] = stages_run
        session_meta["llm_api_calls"] = api_calls

        tmp_path = transcript_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, transcript_path)

        if changes > 0:
            log.info(
                "BatchCorrector: session %s corrected %d segment(s), "
                "stages=%s, api_calls=%d",
                job.session_id, changes,
                "+".join(stages_run), api_calls,
            )
        else:
            log.info("BatchCorrector: session %s no changes needed", job.session_id)

        # Save suggestions
        try:
            self._save_suggestions(segments, corrected_segments)
        except Exception as e:
            log.warning("BatchCorrector: failed to save suggestions: %s", e)

    def _build_session_glossary(
        self, segments: list, max_glossary: int = 50,
    ) -> list:
        """Build a session-specific glossary filtered by segment text relevance."""
        combined_text = " ".join(seg.get("text", "") for seg in segments)
        if not combined_text.strip():
            return []

        katakana_words = set(re.findall(r'[\u30A0-\u30FF]{3,}', combined_text))

        glossary = []
        seen = set()

        if self._known_terms:
            for term in self._known_terms:
                if term in combined_text and term not in seen:
                    glossary.append(term)
                    seen.add(term)

        if self._hotword_loader:
            for word in self._hotword_loader.get_custom_words():
                if word in combined_text and word not in seen:
                    glossary.append(word)
                    seen.add(word)

        for kw in sorted(katakana_words):
            if kw not in seen:
                glossary.append(kw)
                seen.add(kw)

        return glossary[:max_glossary]

    def _save_suggestions(self, originals: list, corrected: list):
        """Save LLM correction diffs to pending_suggestions.json."""
        from llm_corrector import LLMCorrector

        suggestions_path = os.path.join(self._config_dir, "pending_suggestions.json")
        existing = []
        if os.path.exists(suggestions_path):
            try:
                with open(suggestions_path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        new_suggestions = LLMCorrector.detect_corrections(
            [s.get("text", "") for s in originals],
            [s.get("text", "") for s in corrected],
        )
        if new_suggestions:
            existing.extend(new_suggestions)
            existing = existing[-200:]
            tmp = suggestions_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            os.replace(tmp, suggestions_path)

    def update_known_terms(self, known_terms: list):
        """Update known terms list (called when term_corrections.json changes)."""
        self._known_terms = known_terms
