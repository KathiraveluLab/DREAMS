"""
Step 8 — Manifest: Build quality report and optionally export data.

Validates alignment across all tables and ChromaDB collections,
reports completeness, flags problems, and can export the master
manifest as CSV/Parquet for downstream analysis.
"""

import logging
import os

from ..db import get_db, get_collection

logger = logging.getLogger(__name__)

# Whitelist of valid table names to prevent SQL injection via f-string
_VALID_TABLES = frozenset({
    "memories", "emotion_scores", "temporal_features",
    "location_info", "processing_state", "pipeline_runs",
})


def _count_table(conn, table: str) -> int:
    if table not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table!r}")
    return conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]


def _count_collection(name: str) -> int:
    try:
        return get_collection(name).count()
    except Exception:
        return 0


def run(log: logging.Logger | None = None, export_path: str | None = None) -> int:
    """Print a quality report and optionally export the manifest."""
    _log = log or logger
    conn = get_db()

    try:
        total = _count_table(conn, "memories")
        non_dup = conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE is_duplicate = 0"
        ).fetchone()["c"]
        duplicates = total - non_dup

        emotions = _count_table(conn, "emotion_scores")
        temporal = _count_table(conn, "temporal_features")
        location = _count_table(conn, "location_info")
        img_vectors = _count_collection("image_embeddings")
        cap_vectors = _count_collection("caption_embeddings")

        # processing state summary
        errors = conn.execute(
            "SELECT step_name, COUNT(*) as cnt FROM processing_state WHERE status='error' GROUP BY step_name"
        ).fetchall()
        error_summary = {r["step_name"]: r["cnt"] for r in errors}

        _log.info("")
        _log.info("=" * 60)
        _log.info("  PIPELINE QUALITY REPORT")
        _log.info("=" * 60)
        _log.info("  Memories (total):       %d", total)
        _log.info("  Memories (unique):      %d", non_dup)
        _log.info("  Duplicates detected:    %d", duplicates)
        _log.info("-" * 60)
        _log.info("  Emotion scores:         %d / %d", emotions, non_dup)
        _log.info("  Temporal features:      %d / %d", temporal, non_dup)
        _log.info("  Location info:          %d / %d", location, non_dup)
        _log.info("  Image embeddings:       %d / %d", img_vectors, non_dup)
        _log.info("  Caption embeddings:     %d / %d", cap_vectors, non_dup)
        _log.info("-" * 60)

        if error_summary:
            _log.info("  ERRORS BY STEP:")
            for step, cnt in error_summary.items():
                _log.info("    %-25s %d errors", step, cnt)
        else:
            _log.info("  No processing errors!")

        # completeness percentage
        if non_dup > 0:
            completeness = min(emotions, temporal, location, img_vectors, cap_vectors) / non_dup * 100
            _log.info("-" * 60)
            _log.info("  Overall completeness:   %.1f%%", completeness)

        # check for NULL captions
        null_caps = conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE is_duplicate=0 AND caption IS NULL AND generated_caption IS NULL"
        ).fetchone()["c"]
        if null_caps:
            _log.warning("  ⚠ %d records have no caption (user or generated)", null_caps)

        # check for NULL coordinates
        null_coords = conn.execute(
            "SELECT COUNT(*) as c FROM memories WHERE is_duplicate=0 AND (latitude IS NULL OR longitude IS NULL)"
        ).fetchone()["c"]
        if null_coords:
            _log.warning("  ⚠ %d records have no GPS coordinates", null_coords)

        _log.info("=" * 60)

        # optional export
        if export_path:
            from ..utils import validate_safe_path
            from ..config import DATA_DIR, PROJECT_ROOT
            safe_export = validate_safe_path(
                export_path,
                allowed_roots=[DATA_DIR, PROJECT_ROOT],
            )
            _log.info("Exporting master manifest to %s ...", safe_export)
            rows = conn.execute("SELECT * FROM master_manifest ORDER BY user_id, captured_at").fetchall()
            if rows:
                import csv as csvmod
                columns = rows[0].keys()
                with open(str(safe_export), "w", newline="", encoding="utf-8") as f:
                    writer = csvmod.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(dict(r))
                _log.info("Exported %d rows to %s", len(rows), export_path)

        return non_dup
    finally:
        conn.close()
