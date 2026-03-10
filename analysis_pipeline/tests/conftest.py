"""
Shared pytest fixtures for the analysis_pipeline test suite.

Key design decisions:
- All tests run against an in-memory / tmp-dir SQLite DB so they never
  touch the real pipeline.db.
- ChromaDB is pointed at a fresh tmp dir per test session.
- The Flask test client is provided as a session-scoped fixture to avoid
  spinning up the app for every single test.
- ML models (CLIP, BLIP, MiniLM, etc.) are NOT loaded during tests —
  individual step tests mock those out.
"""

import io
import os
import pytest
from pathlib import Path
from PIL import Image


# ── Minimal in-memory JPEG factory ───────────────────────────────────────────

@pytest.fixture
def make_jpeg():
    """Fixture: returns a function that generates raw JPEG bytes."""
    def _make(width: int = 64, height: int = 64,
              color: tuple = (120, 180, 80)) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()
    return _make


@pytest.fixture
def make_jpeg_file(make_jpeg):
    """Fixture: returns a function that saves a JPEG to a path."""
    def _make_file(path: Path, **kwargs) -> Path:
        path.write_bytes(make_jpeg(**kwargs))
        return path
    return _make_file


# ── Redirect all pipeline paths to tmp dirs ───────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _redirect_pipeline_paths(tmp_path_factory):
    """
    Point every path in analysis_pipeline.config at temporary directories
    for the entire test session.  This prevents tests from creating real DB
    files or writing processed images to the project tree.
    """
    base = tmp_path_factory.mktemp("pipeline_data")

    # Patch config before any pipeline module is imported in tests
    import analysis_pipeline.config as cfg
    cfg.DATA_DIR       = base
    cfg.RAW_DIR        = base / "raw"
    cfg.PROCESSED_DIR  = base / "processed"
    cfg.SNAPSHOT_DIR   = base / "snapshots"
    cfg.LOG_DIR        = base / "logs"
    cfg.CACHE_DIR      = base / "cache"
    cfg.SQLITE_DB_PATH = base / "test_pipeline.db"
    cfg.CHROMA_DB_DIR  = base / "chromadb"
    cfg.GEOCODE_CACHE_PATH = base / "cache" / "geocode_cache.db"

    for d in (cfg.RAW_DIR, cfg.PROCESSED_DIR, cfg.SNAPSHOT_DIR,
              cfg.LOG_DIR, cfg.CACHE_DIR, cfg.CHROMA_DB_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Patch db module directly because `from .config import X` creates local copies
    import analysis_pipeline.db as db_mod
    db_mod.SQLITE_DB_PATH = cfg.SQLITE_DB_PATH
    db_mod.CHROMA_DB_DIR = cfg.CHROMA_DB_DIR
    db_mod.GEOCODE_CACHE_PATH = cfg.GEOCODE_CACHE_PATH

    # Reset ChromaDB singleton so it picks up the new path
    db_mod._chroma_client = None

    yield base


# ── SQLite — fresh schema per test ────────────────────────────────────────────

@pytest.fixture
def db_conn(_redirect_pipeline_paths):
    """Return an open SQLite connection with the schema initialised."""
    from analysis_pipeline.db import init_db, get_db
    init_db()
    conn = get_db()
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _clear_state(db_conn):
    """Clear all SQLite tables and ChromaDB collections before each test."""
    # Temporarily disable foreign keys to allow bulk wiping
    db_conn.execute("PRAGMA foreign_keys = OFF")
    # Clear main pipeline tables (geocode_cache is in a separate DB file, skip it)
    for table in ("processing_state", "emotion_scores",
                  "temporal_features", "location_info", "pipeline_runs", "memories"):
        db_conn.execute(f"DELETE FROM {table}")
    db_conn.commit()
    db_conn.execute("PRAGMA foreign_keys = ON")

    # ChromaDB — delete all items from every collection
    from analysis_pipeline.db import get_chroma_client
    chroma = get_chroma_client()
    for coll in chroma.list_collections():
        coll_obj = chroma.get_collection(coll.name)
        all_ids = coll_obj.get()["ids"]
        if all_ids:
            coll_obj.delete(ids=all_ids)



# ── Flask test client ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app(_redirect_pipeline_paths):
    """Create the Flask app once per session with TESTING=True."""
    from analysis_pipeline.api.app import create_app
    application = create_app()
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    """Fresh test client per test (shares the session-level app)."""
    with app.test_client() as c:
        yield c
