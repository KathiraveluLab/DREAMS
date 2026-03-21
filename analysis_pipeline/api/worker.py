"""
Background pipeline worker.

Sleeps when the queue is empty.  Wakes up **only** when:
1. The API signals it (``worker.wake()``) after a new upload, or
2. A safety-net poll fires every ``POLL_INTERVAL`` seconds.

This means **zero CPU usage** when no images are being uploaded.

The worker runs each pipeline step in order.  Each step function
internally queries for ALL pending records, so multiple uploads
that arrive while the worker is asleep are batched automatically.
"""

import logging
import threading

from ..db import init_db, get_db
from . import queue

from ..config import SQL_CHUNK_SIZE
from collections import defaultdict

logger = logging.getLogger(__name__)

# Pipeline steps to run for each uploaded record.
# "ingest" is handled by the upload endpoint; "manifest" is read-only.
_WORKER_STEPS = [
    "caption",
    "image_embeddings",
    "caption_embeddings",
    "emotions",
    "temporal",
]

# How often (seconds) the worker checks the queue even without a signal.
# This is a safety net — the primary mechanism is the threading.Event.
POLL_INTERVAL = 30


class PipelineWorker:
    """Daemon thread that processes the ingest queue."""

    def __init__(self) -> None:
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="pipeline-worker",
        )
        self._processing = False

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def is_processing(self) -> bool:
        return self._processing

    def start(self) -> None:
        logger.info("Pipeline worker starting …")
        self._thread.start()

    def wake(self) -> None:
        """Signal the worker to check the queue immediately."""
        self._wake.set()

    def stop(self) -> None:
        """Gracefully stop the worker (blocks up to 10 s)."""
        logger.info("Pipeline worker stopping …")
        self._stop.set()
        self._wake.set()           # unblock the .wait()
        self._thread.join(timeout=10)

    # ── Internal loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        # One-time setup
        init_db()
        queue.init_queue()

        # Recover any jobs that were stuck in "processing" from a crash
        recovered = queue.recover_stale_jobs()
        if recovered:
            logger.info("Recovered %d stale job(s) from previous run.", recovered)

        logger.info(
            "Pipeline worker ready — sleeping until /api/ingest is called "
            "(safety-net poll every %ds).",
            POLL_INTERVAL,
        )

        while not self._stop.is_set():
            # Block here — zero CPU.
            # Unblocked by: wake(), stop(), or POLL_INTERVAL timeout.
            self._wake.wait(timeout=POLL_INTERVAL)
            self._wake.clear()

            if self._stop.is_set():
                break

            self._process_all()

    # ── Batch processing ──────────────────────────────────────────────────

    def _process_all(self) -> None:
        """Dequeue all waiting jobs, run every step, update job statuses."""
        jobs = queue.dequeue_batch()
        if not jobs:
            return

        self._processing = True
        logger.info("Processing %d queued job(s) …", len(jobs))

        # Run every step once.  Each step's run() function internally
        # fetches ALL pending records, so the whole batch is covered.
        for step_name in _WORKER_STEPS:
            # Update each job to show which step is active
            for job in jobs:
                queue.update_step(job["job_id"], step_name)

            logger.info("Running step '%s' …", step_name)
            try:
                step_fn = self._get_step_fn(step_name)
                step_fn(log=logger)
            except Exception as exc:
                logger.error("Step '%s' raised: %s", step_name, exc,
                             exc_info=True)
                # Don't abort — continue to next step and diagnose per-record
                # in the finalization below.

        # Finalize: check per-record processing_state and update each job.
        self._finalize_jobs(jobs)
        self._processing = False

    def _finalize_jobs(self, jobs: list[dict]) -> None:
        """Mark each job as *done* or *error* based on processing_state."""
        if not jobs:
            return
        
        conn = get_db()
        try:
            memory_ids = [job["memory_id"] for job in jobs]
            
            # Fetch all processing_state rows for the entire batch in chunks
            all_states = []
            for i in range(0, len(memory_ids), SQL_CHUNK_SIZE):
                chunk = memory_ids[i:i + SQL_CHUNK_SIZE]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT memory_id, step_name, status, error_msg FROM processing_state "
                    f"WHERE memory_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                all_states.extend(rows)

            # Group states by memory_id locally
            states_by_mid = defaultdict(list)
            for r in all_states:
                states_by_mid[r["memory_id"]].append(r)

            jobs_done = []
            jobs_error = []

            for job in jobs:
                memory_id = job["memory_id"]
                job_id = job["job_id"]
                
                states = states_by_mid.get(memory_id, [])
                
                done_steps = {s["step_name"] for s in states if s["status"] == "done"}
                error_rows = [s for s in states if s["status"] == "error"]

                missing_steps = [
                    step_name
                    for step_name in _WORKER_STEPS
                    if step_name not in done_steps
                ]

                if error_rows:
                    msgs = "; ".join(
                        f"{r['step_name']}: {r['error_msg']}" for r in error_rows
                    )
                    jobs_error.append((job_id, msgs))
                    logger.warning("Job %s completed with errors: %s", job_id, msgs)
                elif missing_steps:
                    msg = "Incomplete processing: missing steps " + ", ".join(missing_steps)
                    jobs_error.append((job_id, msg))
                    logger.warning("Job %s incomplete: %s", job_id, msg)
                else:
                    jobs_done.append(job_id)
                    logger.info("Job %s completed successfully (%d steps done).", 
                                job_id, len(done_steps))
            
            # Finalize queue statuses in bulk
            if jobs_done:
                queue.mark_jobs_done(jobs_done)
            if jobs_error:
                queue.mark_jobs_error(jobs_error)

        finally:
            conn.close()

    # ── Lazy step imports ─────────────────────────────────────────────────

    @staticmethod
    def _get_step_fn(step_name: str):
        """Import a step module lazily (avoids loading ML models at startup)."""
        if step_name == "caption":
            from ..steps.caption import run
        elif step_name == "image_embeddings":
            from ..steps.image_embeddings import run
        elif step_name == "caption_embeddings":
            from ..steps.caption_embeddings import run
        elif step_name == "emotions":
            from ..steps.emotions import run
        elif step_name == "temporal":
            from ..steps.temporal import run
        else:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        return run


# Module-level singleton — imported by app.py
worker = PipelineWorker()
