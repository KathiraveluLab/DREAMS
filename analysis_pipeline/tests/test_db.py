"""
Unit tests for analysis_pipeline.db

Covers:
- init_db           — schema creation is idempotent
- get_pending_ids   — only returns records without a 'done' state
- mark_record_done  — updates processing_state correctly
- mark_record_error — records error message and status
"""

import pytest
from analysis_pipeline.db import (
    init_db,
    get_db,
    get_pending_ids,
    mark_record_done,
    mark_record_error,
    mark_records_done,
    mark_records_error,
)
from analysis_pipeline.api.queue import init_queue, enqueue_batch, get_batch_status
from analysis_pipeline.utils import make_memory_id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_memory(conn, memory_id: str, user_id: str = "u1") -> None:
    """Insert a minimal memory row for testing."""
    conn.execute(
        """INSERT INTO memories
           (memory_id, user_id, image_path, captured_at, is_duplicate)
           VALUES (?, ?, ?, ?, 0)""",
        (memory_id, user_id, "/fake/path.jpg", "2026-01-01T00:00:00"),
    )
    conn.commit()


# ── init_db ───────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_memories_table(self):
        init_db()
        conn = get_db()
        try:
            tables = {row[0] for row in
                      conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            assert "memories" in tables
            assert "processing_state" in tables
            assert "emotion_scores" in tables
            assert "temporal_features" in tables
        finally:
            conn.close()

    def test_idempotent(self):
        """Calling init_db twice must not raise."""
        init_db()
        init_db()


# ── get_pending_ids ───────────────────────────────────────────────────────────

class TestGetPendingIds:
    def test_new_record_is_pending(self, db_conn):
        mid = make_memory_id("u1", "pending.jpg")
        _insert_memory(db_conn, mid)
        pending = get_pending_ids("image_embeddings", db_conn)
        assert mid in pending

    def test_done_record_not_pending(self, db_conn):
        mid = make_memory_id("u1", "done.jpg")
        _insert_memory(db_conn, mid)
        mark_record_done(mid, "image_embeddings", db_conn)
        db_conn.commit()
        pending = get_pending_ids("image_embeddings", db_conn)
        assert mid not in pending

    def test_error_record_still_pending(self, db_conn):
        mid = make_memory_id("u1", "error.jpg")
        _insert_memory(db_conn, mid)
        mark_record_error(mid, "image_embeddings", "something broke", db_conn)
        db_conn.commit()
        pending = get_pending_ids("image_embeddings", db_conn)
        assert mid in pending

    def test_duplicate_excluded(self, db_conn):
        mid = make_memory_id("u1", "dup.jpg")
        db_conn.execute(
            """INSERT INTO memories
               (memory_id, user_id, image_path, captured_at, is_duplicate)
               VALUES (?, ?, ?, ?, 1)""",
            (mid, "u1", "/fake/dup.jpg", "2026-01-01"),
        )
        db_conn.commit()
        pending = get_pending_ids("image_embeddings", db_conn)
        assert mid not in pending


# ── mark_record_done ──────────────────────────────────────────────────────────

class TestMarkRecordDone:
    def test_sets_status_done(self, db_conn):
        mid = make_memory_id("u1", "m_done.jpg")
        _insert_memory(db_conn, mid)
        mark_record_done(mid, "caption", db_conn)
        db_conn.commit()
        row = db_conn.execute(
            "SELECT status FROM processing_state WHERE memory_id=? AND step_name='caption'",
            (mid,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "done"

    def test_upsert_overwrites_error(self, db_conn):
        mid = make_memory_id("u1", "overwrite.jpg")
        _insert_memory(db_conn, mid)
        mark_record_error(mid, "caption", "oops", db_conn)
        db_conn.commit()
        mark_record_done(mid, "caption", db_conn)
        db_conn.commit()
        row = db_conn.execute(
            "SELECT status FROM processing_state WHERE memory_id=? AND step_name='caption'",
            (mid,),
        ).fetchone()
        assert row["status"] == "done"

    def test_bulk_sets_status_done(self, db_conn):
        mids = [make_memory_id("u1", "bulk_done1.jpg"), make_memory_id("u1", "bulk_done2.jpg")]
        for mid in mids:
            _insert_memory(db_conn, mid)
        mark_records_done(mids, "caption", db_conn)
        db_conn.commit()
        rows = db_conn.execute(
            "SELECT memory_id, status FROM processing_state WHERE step_name='caption'"
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["status"] == "done"


# ── mark_record_error ─────────────────────────────────────────────────────────

class TestMarkRecordError:
    def test_sets_status_error(self, db_conn):
        mid = make_memory_id("u1", "m_err.jpg")
        _insert_memory(db_conn, mid)
        mark_record_error(mid, "emotions", "Model failed", db_conn)
        db_conn.commit()
        row = db_conn.execute(
            "SELECT status, error_msg FROM processing_state WHERE memory_id=? AND step_name='emotions'",
            (mid,),
        ).fetchone()
        assert row["status"] == "error"
        assert "Model failed" in row["error_msg"]

    def test_bulk_sets_status_error(self, db_conn):
        mids = [make_memory_id("u1", "bulk_err1.jpg"), make_memory_id("u1", "bulk_err2.jpg")]
        for mid in mids:
            _insert_memory(db_conn, mid)
        mark_records_error(mids, "emotions", "Batch failure", db_conn)
        db_conn.commit()
        rows = db_conn.execute(
            "SELECT memory_id, status, error_msg FROM processing_state WHERE step_name='emotions'"
        ).fetchall()
        assert len(rows) == 2
        for r in rows:
            assert r["status"] == "error"
            assert "Batch failure" in r["error_msg"]


# ── Batch Queueing ────────────────────────────────────────────────────────────

class TestBatchQueue:
    def test_enqueue_batch_and_status(self, db_conn):
        init_queue()
        mids = [make_memory_id("u1", "q1.jpg"), make_memory_id("u1", "q2.jpg")]
        for mid in mids:
            _insert_memory(db_conn, mid)
            
        batch_id = "batch_001"
        job_ids = enqueue_batch(mids, batch_id)
        assert len(job_ids) == 2
        
        status = get_batch_status(batch_id)
        assert status["batch_id"] == batch_id
        assert status["total"] == 2
        assert status["queued"] == 2
        
        # Test idempotency
        duplicate_job_ids = enqueue_batch(mids, batch_id)
        assert len(duplicate_job_ids) == 0
        
        status_after = get_batch_status(batch_id)
        assert status_after["total"] == 2
