import sqlite3
import sys
from pathlib import Path
import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from config import DREAMS_DB_PATH, CHROMA_DB_DIR, PROCESSED_DIR

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, caption TEXT, timestamp TEXT, lat REAL, lon REAL, image_url TEXT, local_image TEXT, snapshot_id TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS emotion_scores (id INTEGER PRIMARY KEY REFERENCES memories(id), user_id TEXT NOT NULL, valence REAL, arousal REAL, anger REAL DEFAULT 0, disgust REAL DEFAULT 0, fear REAL DEFAULT 0, joy REAL DEFAULT 0, neutral REAL DEFAULT 0, sadness REAL DEFAULT 0, surprise REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS temporal_features (id INTEGER PRIMARY KEY REFERENCES memories(id), user_id TEXT NOT NULL, absolute_utc TEXT, relative_day INTEGER, sin_hour REAL, cos_hour REAL);
CREATE TABLE IF NOT EXISTS location_descriptions (id INTEGER PRIMARY KEY REFERENCES memories(id), user_id TEXT NOT NULL, description TEXT, geocode_display_name TEXT, image_caption TEXT);
CREATE TABLE IF NOT EXISTS pipeline_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running', records_processed INTEGER DEFAULT 0, started_at TEXT, finished_at TEXT, error_message TEXT);
CREATE VIEW IF NOT EXISTS master_manifest AS SELECT m.*, e.valence, e.arousal, e.anger, e.disgust, e.fear, e.joy, e.neutral, e.sadness, e.surprise, t.absolute_utc, t.relative_day, t.sin_hour, t.cos_hour, l.description, l.geocode_display_name, l.image_caption FROM memories m LEFT JOIN emotion_scores e ON m.id = e.id LEFT JOIN temporal_features t ON m.id = t.id LEFT JOIN location_descriptions l ON m.id = l.id;
"""

def get_db():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DREAMS_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn

_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _chroma_client

def get_collection(name):
    return get_chroma_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

# --------------- Pipeline run tracking ---------------

def record_step_start(run_id: str, step: str):
    from datetime import datetime, timezone
    conn = get_db()
    conn.execute(
        "INSERT INTO pipeline_runs (run_id, step, status, started_at) VALUES (?, ?, 'running', ?)",
        (run_id, step, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

def record_step_done(run_id: str, step: str, records: int = 0, error: str | None = None):
    from datetime import datetime, timezone
    status = "error" if error else "done"
    conn = get_db()
    conn.execute(
        "UPDATE pipeline_runs SET status = ?, records_processed = ?, finished_at = ?, error_message = ? "
        "WHERE run_id = ? AND step = ? AND status = 'running'",
        (status, records, datetime.now(timezone.utc).isoformat(), error, run_id, step),
    )
    conn.commit()
    conn.close()

def get_last_run_steps():
    """Return set of step names that completed successfully in the most recent run."""
    conn = get_db()
    row = conn.execute(
        "SELECT run_id FROM pipeline_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        conn.close()
        return set()
    last_run_id = row[0]
    rows = conn.execute(
        "SELECT step FROM pipeline_runs WHERE run_id = ? AND status = 'done'",
        (last_run_id,),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}
