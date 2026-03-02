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

logger = logging.getLogger(__name__)

# Pipeline steps to run for each uploaded record.
# "ingest" is handled by the upload endpoint; "manifest" is read-only.
_WORKER_STEPS = [
    "caption",
    "image_embeddings",
    "caption_embeddings",
    "emotions",
    "location",
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
        conn = get_db()
        try:
            for job in jobs:
                memory_id = job["memory_id"]
                job_id = job["job_id"]

                done_rows = conn.execute(
                    "SELECT step_name FROM processing_state "
                    "WHERE memory_id = ? AND status = 'done'",
                    (memory_id,),
                ).fetchall()
                done_steps = {r["step_name"] for r in done_rows}

                error_rows = conn.execute(
                    "SELECT step_name, error_msg FROM processing_state "
                    "WHERE memory_id = ? AND status = 'error'",
                    (memory_id,),
                ).fetchall()

                if error_rows:
                    msgs = "; ".join(
                        f"{r['step_name']}: {r['error_msg']}" for r in error_rows
                    )
                    queue.mark_error(job_id, msgs)
                    logger.warning("Job %s completed with errors: %s",
                                   job_id, msgs)
                else:
                    queue.mark_done(job_id)
                    logger.info("Job %s completed successfully "
                                "(%d steps done).", job_id, len(done_steps))
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
        elif step_name == "location":
            from ..steps.location import run
        elif step_name == "temporal":
            from ..steps.temporal import run
        else:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        return run


# Module-level singleton — imported by app.py
worker = PipelineWorker()
