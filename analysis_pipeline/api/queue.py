"""
SQLite-backed job queue for the analysis pipeline API.

Uses the same ``pipeline.db`` database.  The worker thread sleeps when
the queue is empty and wakes only when a new job is enqueued (signalled
via ``threading.Event`` in the worker module).

All functions create their own DB connections so they are safe to call
from any thread.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from ..config import SQL_CHUNK_SIZE
from ..db import get_db

# ── Schema ────────────────────────────────────────────────────────────────────

_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_queue (
    job_id          TEXT PRIMARY KEY,
    batch_id        TEXT,
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
        # Migrate existing tables
        try:
            conn.execute("ALTER TABLE ingest_queue ADD COLUMN batch_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_iq_batch ON ingest_queue(batch_id)")
        except sqlite3.OperationalError as e:
            logger.debug("Migration skipped (already applied): %s", e)


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


def enqueue_batch(memory_ids: list[str], batch_id: str) -> list[str]:
    """Bulk-insert a batch of memories into the processing queue.

    Returns the list of ``job_id``s that were actually created
    (excludes duplicates that were already queued/processing).
    """
    if not memory_ids:
        return []
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    created_job_ids: list[str] = []
    try:
        # Process in chunks to avoid SQLite max variable limits
        for i in range(0, len(memory_ids), SQL_CHUNK_SIZE):
            chunk = memory_ids[i:i + SQL_CHUNK_SIZE]

            # Avoid creating duplicates if idempotency is needed
            placeholders = ",".join("?" for _ in chunk)
            existing = conn.execute(
                f"SELECT memory_id FROM ingest_queue "
                f"WHERE memory_id IN ({placeholders}) AND status IN ('queued', 'processing')",
                chunk,
            ).fetchall()
            existing_ids = {row["memory_id"] for row in existing}

            insert_data = []
            for mid in chunk:
                if mid not in existing_ids:
                    jid = uuid.uuid4().hex
                    insert_data.append((jid, batch_id, mid, "queued", now, now))
                    created_job_ids.append(jid)

            if insert_data:
                conn.executemany(
                    "INSERT INTO ingest_queue "
                    "(job_id, batch_id, memory_id, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    insert_data,
                )
        conn.commit()
        return created_job_ids
    except sqlite3.Error:
        conn.rollback()
        raise
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
        for i in range(0, len(job_ids), SQL_CHUNK_SIZE):
            chunk = job_ids[i:i + SQL_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(
                f"UPDATE ingest_queue SET status = 'processing', "
                f"updated_at = ? WHERE job_id IN ({placeholders})",
                [now] + chunk,
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


def get_batch_status(batch_id: str) -> dict:
    """Return counts of jobs in different states for a given batch."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM ingest_queue WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        ).fetchall()
        status_counts = {r["status"]: r["cnt"] for r in rows}
        total = sum(status_counts.values())
        return {
            "batch_id": batch_id,
            "total": total,
            "queued": status_counts.get("queued", 0),
            "processing": status_counts.get("processing", 0),
            "done": status_counts.get("done", 0),
            "error": status_counts.get("error", 0),
        }
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
