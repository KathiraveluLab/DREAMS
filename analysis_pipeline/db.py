"""
Database helpers for SQLite (structured data) and ChromaDB (vector store).

Key improvement over prior pipeline: per-record processing state tracking
and duplicate detection via perceptual hashing.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import chromadb

from .config import SQLITE_DB_PATH, CHROMA_DB_DIR

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SQLite                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_SCHEMA_SQL = """
-- Core table: one row per memory (photo + metadata)
CREATE TABLE IF NOT EXISTS memories (
    memory_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    image_path      TEXT,
    category        TEXT,
    caption         TEXT,
    generated_caption TEXT,
    captured_at     TEXT,
    imported_at     TEXT NOT NULL DEFAULT (datetime('now')),
    perceptual_hash TEXT,
    is_duplicate    INTEGER NOT NULL DEFAULT 0,
    duplicate_of    TEXT,
    FOREIGN KEY (duplicate_of) REFERENCES memories(memory_id)
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_phash ON memories(perceptual_hash);

-- Per-record processing state (the key improvement)
CREATE TABLE IF NOT EXISTS processing_state (
    memory_id   TEXT NOT NULL,
    step_name   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | done | error
    error_msg   TEXT,
    started_at  TEXT,
    finished_at TEXT,
    PRIMARY KEY (memory_id, step_name),
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);
CREATE INDEX IF NOT EXISTS idx_ps_step ON processing_state(step_name, status);

-- Emotion scores (richer than prior: 7 discrete + valence/arousal + sentiment + CHIME)
CREATE TABLE IF NOT EXISTS emotion_scores (
    memory_id       TEXT PRIMARY KEY,
    -- 7 discrete emotions (probabilities)
    anger           REAL,
    disgust         REAL,
    fear            REAL,
    joy             REAL,
    neutral         REAL,
    sadness         REAL,
    surprise        REAL,
    dominant_emotion TEXT,
    -- valence / arousal (continuous)
    valence         REAL,
    arousal         REAL,
    -- sentiment (pos/neg/neutral)
    sentiment_label TEXT,
    sentiment_pos   REAL,
    sentiment_neg   REAL,
    sentiment_neu   REAL,
    -- CHIME recovery category
    chime_category  TEXT,
    chime_confidence REAL,
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);

-- Temporal features
CREATE TABLE IF NOT EXISTS temporal_features (
    memory_id       TEXT PRIMARY KEY,
    hour            INTEGER,
    day_of_week     INTEGER,
    month           INTEGER,
    year            INTEGER,
    season          TEXT,
    time_of_day     TEXT,          -- morning / afternoon / evening / night
    sin_hour        REAL,
    cos_hour        REAL,
    sin_dow         REAL,
    cos_dow         REAL,
    sin_month       REAL,
    cos_month       REAL,
    relative_day    REAL,          -- days since user's first memory
    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);

-- Pipeline run log
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id      TEXT NOT NULL,
    step_name   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | done | error
    records     INTEGER DEFAULT 0,
    error_msg   TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    PRIMARY KEY (run_id, step_name)
);
"""

_MASTER_MANIFEST_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS master_manifest AS
SELECT
    m.memory_id,
    m.user_id,
    m.image_path,
    m.category,
    m.caption,
    m.generated_caption,
    m.captured_at,
    m.is_duplicate,
    e.dominant_emotion,
    e.valence,
    e.arousal,
    e.sentiment_label,
    e.chime_category,
    t.hour,
    t.time_of_day,
    t.season,
    t.sin_hour,
    t.cos_hour,
    t.relative_day
FROM memories m
LEFT JOIN emotion_scores  e ON m.memory_id = e.memory_id
LEFT JOIN temporal_features t ON m.memory_id = t.memory_id
WHERE m.is_duplicate = 0;
"""

_MEMORIES_TABLE_SQL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    memory_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    image_path      TEXT,
    category        TEXT,
    caption         TEXT,
    generated_caption TEXT,
    captured_at     TEXT,
    imported_at     TEXT NOT NULL DEFAULT (datetime('now')),
    perceptual_hash TEXT,
    is_duplicate    INTEGER NOT NULL DEFAULT 0,
    duplicate_of    TEXT,
    FOREIGN KEY (duplicate_of) REFERENCES memories(memory_id)
)
"""


def get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection."""
    path = db_path or SQLITE_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_memories_drop_coordinate_columns(conn: sqlite3.Connection) -> None:
    """Rebuild the memories table if legacy coordinate columns are still present."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(memories)").fetchall()
    }
    if "latitude" not in columns and "longitude" not in columns:
        return

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DROP TABLE IF EXISTS memories_new")
    conn.execute(_MEMORIES_TABLE_SQL_TEMPLATE.format(table_name="memories_new"))
    conn.execute(
        """
        INSERT INTO memories_new (
            memory_id, user_id, image_path, category, caption,
            generated_caption, captured_at, imported_at,
            perceptual_hash, is_duplicate, duplicate_of
        )
        SELECT
            memory_id, user_id, image_path, category, caption,
            generated_caption, captured_at, imported_at,
            perceptual_hash, is_duplicate, duplicate_of
        FROM memories
        """
    )
    conn.execute("DROP TABLE memories")
    conn.execute("ALTER TABLE memories_new RENAME TO memories")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_phash ON memories(perceptual_hash)")
    conn.execute("PRAGMA foreign_keys=ON")


def init_db() -> None:
    """Create all tables, indices, and views."""
    with get_db() as conn:
        conn.execute("DROP VIEW IF EXISTS master_manifest")
        conn.execute("DROP TABLE IF EXISTS location_info")
        conn.executescript(_SCHEMA_SQL)
        conn.execute("DROP VIEW IF EXISTS master_manifest")
        _migrate_memories_drop_coordinate_columns(conn)
        conn.execute(_MASTER_MANIFEST_VIEW_SQL)


# ── Per-record processing state helpers ───────────────────────────────────────

def get_pending_ids(step_name: str, conn: sqlite3.Connection | None = None) -> list[str]:
    """Return memory_ids that have NOT been successfully processed for *step_name*."""
    close = False
    if conn is None:
        conn = get_db()
        close = True
    try:
        rows = conn.execute(
            """
            SELECT m.memory_id
            FROM memories m
            WHERE m.is_duplicate = 0
              AND m.memory_id NOT IN (
                  SELECT ps.memory_id
                  FROM processing_state ps
                  WHERE ps.step_name = ? AND ps.status = 'done'
              )
            ORDER BY m.captured_at
            """,
            (step_name,),
        ).fetchall()
        return [r["memory_id"] for r in rows]
    finally:
        if close:
            conn.close()


def mark_record_done(memory_id: str, step_name: str, conn: sqlite3.Connection) -> None:
    """Mark a single record as successfully processed for a step."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO processing_state
            (memory_id, step_name, status, finished_at)
        VALUES (?, ?, 'done', ?)
        """,
        (memory_id, step_name, now),
    )


def mark_record_error(memory_id: str, step_name: str, error: str, conn: sqlite3.Connection) -> None:
    """Mark a single record as failed for a step."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO processing_state
            (memory_id, step_name, status, error_msg, finished_at)
        VALUES (?, ?, 'error', ?, ?)
        """,
        (memory_id, step_name, error, now),
    )


# ── Pipeline run tracking ─────────────────────────────────────────────────────

def record_step_start(run_id: str, step_name: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO pipeline_runs (run_id, step_name, status, started_at) VALUES (?,?,?,?)",
            (run_id, step_name, "running", now),
        )


def record_step_done(run_id: str, step_name: str, records: int, error: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    status = "error" if error else "done"
    with get_db() as conn:
        conn.execute(
            """UPDATE pipeline_runs
               SET status=?, records=?, error_msg=?, finished_at=?
               WHERE run_id=? AND step_name=?""",
            (status, records, error, now, run_id, step_name),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ChromaDB                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_chroma_client = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Singleton persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _chroma_client


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine distance."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
