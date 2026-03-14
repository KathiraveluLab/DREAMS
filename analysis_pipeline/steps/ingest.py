"""
Step 1 — Ingest: Import data from CSV / image folder into SQLite.

Improvements:
- Deterministic memory_id (re-importing same data is idempotent)
- Perceptual hashing for image deduplication

- Supports user-supplied captions alongside BLIP-generated ones
- Snapshot of raw data for reproducibility
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone

from ..config import RAW_DIR, PROCESSED_DIR, SNAPSHOT_DIR, PROJECT_ROOT, DATA_DIR
from ..db import get_db, init_db
from ..utils import (
    load_image,
    is_image_file,
    perceptual_hash,
    hamming_distance,
    parse_timestamp,
    make_memory_id,
    validate_safe_path,
)

logger = logging.getLogger(__name__)

# images with a hamming distance ≤ this threshold are considered duplicates
_DUPLICATE_THRESHOLD = 10


def _find_image(filename: str, search_dirs: list[Path]) -> Path | None:
    """Search for an image file across multiple directories."""
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate
        # also try case-insensitive match
        for f in d.iterdir():
            if f.name.lower() == filename.lower() and f.is_file():
                return f
    return None


def _copy_image_to_processed(src: Path, memory_id: str) -> Path:
    """Copy image to processed dir with memory_id prefix for easy lookup."""
    dest = (PROCESSED_DIR / f"{memory_id}{src.suffix.lower()}").resolve()
    # Ensure the destination stays within PROCESSED_DIR
    if not str(dest).startswith(str(PROCESSED_DIR.resolve())):
        raise ValueError(f"Destination escapes PROCESSED_DIR: {dest}")
    if not dest.exists():
        shutil.copy2(str(src), str(dest))
    return dest


def _detect_duplicates(conn, memory_id: str, phash: str) -> tuple[bool, str | None]:
    """Check if a perceptual hash is too close to an existing one."""
    rows = conn.execute(
        "SELECT memory_id, perceptual_hash FROM memories WHERE perceptual_hash IS NOT NULL AND memory_id != ?",
        (memory_id,),
    ).fetchall()
    for row in rows:
        dist = hamming_distance(phash, row["perceptual_hash"])
        if dist <= _DUPLICATE_THRESHOLD:
            return True, row["memory_id"]
    return False, None


def run(source_path: str, logger: logging.Logger | None = None) -> int:
    """Ingest data from a CSV file.

    Expected CSV columns:
        id, user_id, image_filename, category, date, caption

    Returns the number of records imported.
    """
    _log = logger or logging.getLogger(__name__)

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # ensure DB schema exists
    init_db()

    # determine image search directories
    image_dirs = [
        source.parent,                    # same dir as CSV
        source.parent / "images",         # images/ subfolder
        RAW_DIR,
        PROCESSED_DIR,
    ]

    # create versioned snapshot
    snapshot_name = f"ingest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    snap_dir = SNAPSHOT_DIR / snapshot_name
    snap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(snap_dir / source.name))
    _log.info("Snapshot saved to %s", snap_dir)

    imported = 0
    skipped_dup = 0
    skipped_exists = 0
    errors = 0

    with open(source, encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)

        conn = get_db()
        try:
            for row in reader:
                try:
                    user_id = row.get("user_id", "").strip()
                    image_filename = row.get("image_filename", "").strip()
                    if not user_id or not image_filename:
                        _log.warning("Row missing user_id or image_filename, skipping: %s", row)
                        errors += 1
                        continue

                    memory_id = make_memory_id(user_id, image_filename)

                    # idempotent: skip if already imported
                    existing = conn.execute(
                        "SELECT memory_id FROM memories WHERE memory_id=?", (memory_id,)
                    ).fetchone()
                    if existing:
                        skipped_exists += 1
                        continue

                    # locate image file
                    img_path = _find_image(image_filename, image_dirs)
                    processed_path = None
                    phash = None
                    is_dup = False
                    dup_of = None

                    if img_path and img_path.exists():
                        # compute perceptual hash for deduplication
                        try:
                            img = load_image(img_path)
                            phash = perceptual_hash(img)
                            is_dup, dup_of = _detect_duplicates(conn, memory_id, phash)
                            if is_dup:
                                _log.info(
                                    "Duplicate detected: %s is near-duplicate of %s",
                                    image_filename, dup_of,
                                )
                                skipped_dup += 1
                        except Exception as e:
                            _log.warning("Could not hash image %s: %s", image_filename, e)

                        # copy to processed dir
                        try:
                            processed_path = _copy_image_to_processed(img_path, memory_id)
                        except Exception as e:
                            _log.warning("Could not copy image %s: %s", image_filename, e)
                    else:
                        _log.warning("Image not found: %s", image_filename)

                    # parse fields
                    captured_at = row.get("date", "").strip() or None
                    caption = row.get("caption", "").strip() or None
                    category = row.get("category", "").strip() or None

                    conn.execute(
                        """INSERT INTO memories
                           (memory_id, user_id, image_path, category, caption,
                            captured_at,
                            perceptual_hash, is_duplicate, duplicate_of)
                           VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            memory_id,
                            user_id,
                            str(processed_path) if processed_path else None,
                            category,
                            caption,
                            captured_at,
                            phash,
                            1 if is_dup else 0,
                            dup_of,
                        ),
                    )
                    imported += 1

                except Exception as e:
                    _log.error("Error ingesting row %s: %s", row.get("id", "?"), e)
                    errors += 1

            conn.commit()
        finally:
            conn.close()

    _log.info(
        "Ingestion complete: %d imported, %d duplicates, %d already existed, %d errors",
        imported, skipped_dup, skipped_exists, errors,
    )
    return imported
