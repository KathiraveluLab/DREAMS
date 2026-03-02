"""
SQLite-backed job queue for the analysis pipeline API.

Uses the same ``pipeline.db`` database.  The worker thread sleeps when
the queue is empty and wakes only when a new job is enqueued (signalled
via ``threading.Event`` in the worker module).

All functions create their own DB connections so they are safe to call
from any thread.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from ..db import get_db

# ── Schema ────────────────────────────────────────────────────────────────────

_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_queue (
    job_id          TEXT PRIMARY KEY,
    memory_id       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    current_step    TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);
CREATE INDEX IF NOT EXISTS idx_iq_status ON ingest_queue(status);
CREATE INDEX IF NOT EXISTS idx_iq_memory ON ingest_queue(memory_id);
"""


def init_queue() -> None:
    """Create the ``ingest_queue`` table if it does not exist."""
    with get_db() as conn:
        conn.executescript(_QUEUE_SCHEMA)


# ── Enqueue / dequeue ─────────────────────────────────────────────────────────

def enqueue(memory_id: str) -> str:
    """Add a record to the processing queue.  Returns the ``job_id``.

    If a *queued* or *processing* job already exists for the same
    ``memory_id``, the existing ``job_id`` is returned (idempotent).
    """
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT job_id FROM ingest_queue "
            "WHERE memory_id = ? AND status IN ('queued', 'processing')",
            (memory_id,),
        ).fetchone()
        if existing:
            return existing["job_id"]

        conn.execute(
            "INSERT INTO ingest_queue "
            "(job_id, memory_id, status, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?)",
            (job_id, memory_id, now, now),
        )
        conn.commit()
        return job_id
    finally:
        conn.close()


def dequeue_batch() -> list[dict]:
    """Atomically move all *queued* jobs to *processing* and return them.

    Returns a list of ``{"job_id": ..., "memory_id": ...}`` dicts.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        jobs = conn.execute(
            "SELECT job_id, memory_id FROM ingest_queue "
            "WHERE status = 'queued' ORDER BY created_at",
        ).fetchall()
        if not jobs:
            return []

        job_ids = [j["job_id"] for j in jobs]
        placeholders = ",".join("?" for _ in job_ids)
        conn.execute(
            f"UPDATE ingest_queue SET status = 'processing', "
            f"updated_at = ? WHERE job_id IN ({placeholders})",
            [now] + job_ids,
        )
        conn.commit()
        return [{"job_id": j["job_id"], "memory_id": j["memory_id"]}
                for j in jobs]
    finally:
        conn.close()


# ── Status updates ────────────────────────────────────────────────────────────

def update_step(job_id: str, step_name: str) -> None:
    """Record which pipeline step is currently running for *job_id*."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            "UPDATE ingest_queue SET current_step = ?, updated_at = ? "
            "WHERE job_id = ?",
            (step_name, now, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_done(job_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            "UPDATE ingest_queue SET status = 'done', current_step = NULL, "
            "updated_at = ? WHERE job_id = ?",
            (now, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_error(job_id: str, error_msg: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        conn.execute(
            "UPDATE ingest_queue SET status = 'error', error_message = ?, "
            "updated_at = ? WHERE job_id = ?",
            (error_msg, now, job_id),
        )
        conn.commit()
    finally:
        conn.close()


# ── Queries ───────────────────────────────────────────────────────────────────

def get_job(job_id: str) -> Optional[dict]:
    """Return the full row for a job, or ``None``."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM ingest_queue WHERE job_id = ?", (job_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_job_by_memory(memory_id: str) -> Optional[dict]:
    """Return the *latest* job for a given ``memory_id``."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM ingest_queue WHERE memory_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (memory_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def recover_stale_jobs() -> int:
    """Reset any jobs stuck in *processing* back to *queued*.

    Called once on startup to recover from a previous crash.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cur = conn.execute(
            "UPDATE ingest_queue SET status = 'queued', current_step = NULL, "
            "updated_at = ? WHERE status = 'processing'",
            (now,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
