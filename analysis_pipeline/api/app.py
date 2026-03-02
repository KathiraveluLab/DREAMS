"""
Flask application for the analysis pipeline REST API.

Endpoints
---------
POST /api/ingest              Upload an image for processing
GET  /api/status/<job_id>     Poll job status
GET  /api/analysis/<memory_id> Full analysis JSON for one record
GET  /api/analysis             Paginated list of completed records
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Blueprint, request, jsonify
from PIL import Image

from ..config import IMAGE_EXTENSIONS, PROCESSED_DIR, DATA_DIR
from ..db import get_db, get_collection, init_db
from ..utils import (
    make_memory_id,
    perceptual_hash,
    hamming_distance,
    extract_gps_from_exif,
    safe_float,
    validate_safe_path,
)
from . import queue

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_UPLOAD_MB = 10
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

# Perceptual-hash Hamming-distance threshold (same as steps/ingest.py)
_DUPLICATE_THRESHOLD = 10

# Pipeline steps that the worker runs for each uploaded record
_ALL_STEPS = [
    "caption", "image_embeddings", "caption_embeddings",
    "emotions", "location", "temporal",
]

# ── Blueprint ─────────────────────────────────────────────────────────────────

bp = Blueprint("pipeline_api", __name__, url_prefix="/api")


# ── POST /api/ingest ──────────────────────────────────────────────────────────

@bp.route("/ingest", methods=["POST"])
def ingest():
    """Accept an image upload, insert into the DB, and enqueue for processing.

    Form fields
    -----------
    image      : file   (required)  Image file (.jpg, .png, etc.)
    user_id    : str    (required)  Owner of the memory
    caption    : str    (optional)  User-provided caption text
    latitude   : float  (optional)  GPS latitude  (falls back to EXIF)
    longitude  : float  (optional)  GPS longitude (falls back to EXIF)
    category   : str    (optional)  e.g. "park", "hospital"
    timestamp  : str    (optional)  ISO-8601 capture time (default: now)

    Returns 202 with ``{job_id, memory_id, status}`` on success.
    """
    # ── 1. Validate required fields ──────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "Missing required field: image"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    user_id = (request.form.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400

    # ── 2. Validate file extension ───────────────────────────────────────
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        return jsonify({
            "error": f"Invalid file type '{ext}'. Allowed: {allowed}",
        }), 400

    # ── 3. Check file size (enforced by Flask MAX_CONTENT_LENGTH too) ────
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > _MAX_UPLOAD_BYTES:
        return jsonify({
            "error": f"File too large ({size / 1024 / 1024:.1f} MB). "
                     f"Maximum: {_MAX_UPLOAD_MB} MB",
        }), 413

    # ── 4. Generate deterministic memory_id ──────────────────────────────
    memory_id = make_memory_id(user_id, file.filename)

    # ── 5. Check for existing record with same memory_id ─────────────────
    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT memory_id, is_duplicate, duplicate_of "
            "FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
    finally:
        conn.close()

    if existing:
        job = queue.get_job_by_memory(memory_id)
        return jsonify({
            "memory_id": memory_id,
            "status": "already_exists",
            "job_id": job["job_id"] if job else None,
        }), 200

    # ── 6. Save image to PROCESSED_DIR ───────────────────────────────────
    safe_name = f"{memory_id}{ext}"
    save_path = PROCESSED_DIR / safe_name
    try:
        validate_safe_path(save_path, allowed_roots=[PROCESSED_DIR])
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    file.save(str(save_path))

    # ── 7. Verify it is a valid image ────────────────────────────────────
    try:
        img = Image.open(str(save_path))
        img.verify()
        # Re-open (verify() closes the file pointer)
        img = Image.open(str(save_path)).convert("RGB")
    except Exception:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Invalid or corrupted image file"}), 400

    # ── 8. Compute perceptual hash & check duplicates ────────────────────
    phash = perceptual_hash(img)
    is_dup, dup_of = _check_duplicate(memory_id, phash)

    # ── 9. Extract optional metadata ─────────────────────────────────────
    caption = (request.form.get("caption") or "").strip() or None
    lat = safe_float(request.form.get("latitude"))
    lon = safe_float(request.form.get("longitude"))
    category = (request.form.get("category") or "").strip() or None
    captured_at = (
        (request.form.get("timestamp") or "").strip()
        or datetime.now(timezone.utc).isoformat()
    )

    # ── 10. EXIF GPS fallback ────────────────────────────────────────────
    if lat is None or lon is None:
        exif_lat, exif_lon = extract_gps_from_exif(save_path)
        if lat is None:
            lat = exif_lat
        if lon is None:
            lon = exif_lon

    # ── 11. Insert into memories table ───────────────────────────────────
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO memories
               (memory_id, user_id, image_path, category, caption,
                latitude, longitude, captured_at,
                perceptual_hash, is_duplicate, duplicate_of)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory_id, user_id, str(save_path), category, caption,
                lat, lon, captured_at,
                phash, 1 if is_dup else 0, dup_of,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # ── 12. If duplicate, don't enqueue ──────────────────────────────────
    if is_dup:
        logger.info("Upload %s is a duplicate of %s — skipping queue.",
                     memory_id, dup_of)
        return jsonify({
            "memory_id": memory_id,
            "status": "duplicate",
            "duplicate_of": dup_of,
        }), 200

    # ── 13. Enqueue for background processing ────────────────────────────
    job_id = queue.enqueue(memory_id)
    logger.info("Enqueued job %s for memory %s", job_id, memory_id)

    # Wake the worker (import here to avoid circular import)
    from .worker import worker
    worker.wake()

    return jsonify({
        "job_id": job_id,
        "memory_id": memory_id,
        "status": "queued",
    }), 202


# ── GET /api/status/<job_id> ──────────────────────────────────────────────────

@bp.route("/status/<job_id>")
def status(job_id: str):
    """Return the current processing status for a queued job."""
    job = queue.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200


# ── GET /api/analysis/<memory_id> ─────────────────────────────────────────────

@bp.route("/analysis/<memory_id>")
def analysis(memory_id: str):
    """Return the full analysis JSON for a single processed record.

    Query params
    ------------
    include_embeddings : bool  If ``true``, include raw embedding vectors
                               (adds ~4 KB per record).  Default ``false``.
    """
    include_emb = request.args.get("include_embeddings", "false").lower() == "true"

    result = _build_analysis(memory_id, include_embeddings=include_emb)
    if result is None:
        return jsonify({"error": "Record not found"}), 404
    return jsonify(result), 200


# ── GET /api/analysis ─────────────────────────────────────────────────────────

@bp.route("/analysis")
def analysis_list():
    """Paginated list of analysis records.

    Query params
    ------------
    user_id  : str   Filter by user (optional)
    page     : int   Page number (default 1)
    per_page : int   Records per page (default 20, max 100)
    """
    user_id = request.args.get("user_id")
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))
    offset = (page - 1) * per_page

    conn = get_db()
    try:
        # Count
        if user_id:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM memories "
                "WHERE is_duplicate = 0 AND user_id = ?",
                (user_id,),
            ).fetchone()["c"]
            rows = conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE is_duplicate = 0 AND user_id = ? "
                "ORDER BY captured_at LIMIT ? OFFSET ?",
                (user_id, per_page, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE is_duplicate = 0",
            ).fetchone()["c"]
            rows = conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE is_duplicate = 0 "
                "ORDER BY captured_at LIMIT ? OFFSET ?",
                (per_page, offset),
            ).fetchall()
    finally:
        conn.close()

    records = []
    for r in rows:
        obj = _build_analysis(r["memory_id"], include_embeddings=False)
        if obj:
            records.append(obj)

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "records": records,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check_duplicate(memory_id: str, phash: str) -> tuple[bool, str | None]:
    """Compare a perceptual hash against all existing hashes in the DB."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT memory_id, perceptual_hash FROM memories "
            "WHERE perceptual_hash IS NOT NULL AND memory_id != ?",
            (memory_id,),
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        dist = hamming_distance(phash, row["perceptual_hash"])
        if dist <= _DUPLICATE_THRESHOLD:
            return True, row["memory_id"]
    return False, None


def _build_analysis(
    memory_id: str,
    include_embeddings: bool = False,
) -> dict | None:
    """Assemble the full analysis JSON for one record from all DB tables."""
    conn = get_db()
    try:
        mem = conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,),
        ).fetchone()
        if not mem:
            return None

        emo = conn.execute(
            "SELECT * FROM emotion_scores WHERE memory_id = ?", (memory_id,),
        ).fetchone()

        temp = conn.execute(
            "SELECT * FROM temporal_features WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()

        loc = conn.execute(
            "SELECT * FROM location_info WHERE memory_id = ?", (memory_id,),
        ).fetchone()

        steps = conn.execute(
            "SELECT step_name, status, error_msg FROM processing_state "
            "WHERE memory_id = ?",
            (memory_id,),
        ).fetchall()
    finally:
        conn.close()

    # ── Base fields ──────────────────────────────────────────────────────
    result: dict = {
        "memory_id": mem["memory_id"],
        "user_id": mem["user_id"],
        "image_path": mem["image_path"],
        "caption": mem["caption"],
        "generated_caption": mem["generated_caption"],
        "caption_source": "user" if mem["caption"] else "generated",
        "category": mem["category"],
        "captured_at": mem["captured_at"],
        "is_duplicate": bool(mem["is_duplicate"]),
    }

    # ── Emotions ─────────────────────────────────────────────────────────
    if emo:
        result["emotions"] = {
            "discrete": {
                "anger": emo["anger"],
                "disgust": emo["disgust"],
                "fear": emo["fear"],
                "joy": emo["joy"],
                "neutral": emo["neutral"],
                "sadness": emo["sadness"],
                "surprise": emo["surprise"],
            },
            "dominant_emotion": emo["dominant_emotion"],
            "valence": emo["valence"],
            "arousal": emo["arousal"],
            "sentiment": {
                "label": emo["sentiment_label"],
                "positive": emo["sentiment_pos"],
                "negative": emo["sentiment_neg"],
                "neutral": emo["sentiment_neu"],
            },
            "chime": {
                "category": emo["chime_category"],
                "confidence": emo["chime_confidence"],
            },
        }
    else:
        result["emotions"] = None

    # ── Location ─────────────────────────────────────────────────────────
    if loc:
        result["location"] = {
            "latitude": mem["latitude"],
            "longitude": mem["longitude"],
            "display_name": loc["display_name"],
            "place_type": loc["place_type"],
            "address": {
                "road": loc["address_road"],
                "city": loc["address_city"],
                "state": loc["address_state"],
                "country": loc["address_country"],
            },
        }
    elif mem["latitude"] is not None and mem["longitude"] is not None:
        result["location"] = {
            "latitude": mem["latitude"],
            "longitude": mem["longitude"],
            "display_name": None,
            "place_type": None,
            "address": None,
        }
    else:
        result["location"] = None

    # ── Temporal ─────────────────────────────────────────────────────────
    if temp:
        result["temporal"] = {
            "hour": temp["hour"],
            "day_of_week": temp["day_of_week"],
            "month": temp["month"],
            "year": temp["year"],
            "season": temp["season"],
            "time_of_day": temp["time_of_day"],
            "recovery_day": temp["relative_day"],
            "cyclical": {
                "sin_hour": temp["sin_hour"],
                "cos_hour": temp["cos_hour"],
                "sin_dow": temp["sin_dow"],
                "cos_dow": temp["cos_dow"],
                "sin_month": temp["sin_month"],
                "cos_month": temp["cos_month"],
            },
        }
    else:
        result["temporal"] = None

    # ── Embeddings ───────────────────────────────────────────────────────
    if include_embeddings:
        result["embeddings"] = {}
        try:
            img_coll = get_collection("image_embeddings")
            img_data = img_coll.get(ids=[memory_id], include=["embeddings"])
            if img_data["ids"]:
                result["embeddings"]["image"] = {
                    "vector": img_data["embeddings"][0],
                    "dimensions": len(img_data["embeddings"][0]),
                }
        except Exception:
            pass
        try:
            cap_coll = get_collection("caption_embeddings")
            cap_data = cap_coll.get(ids=[memory_id], include=["embeddings"])
            if cap_data["ids"]:
                result["embeddings"]["caption"] = {
                    "vector": cap_data["embeddings"][0],
                    "dimensions": len(cap_data["embeddings"][0]),
                }
        except Exception:
            pass
    else:
        result["embeddings"] = {
            "image": {"collection": "image_embeddings", "dimensions": 512},
            "caption": {"collection": "caption_embeddings", "dimensions": 384},
        }

    # ── Processing progress ──────────────────────────────────────────────
    steps_dict = {s["step_name"]: s["status"] for s in steps}
    completed = [s for s in _ALL_STEPS if steps_dict.get(s) == "done"]
    errors = {
        s["step_name"]: s["error_msg"]
        for s in steps
        if s["status"] == "error"
    }

    if len(completed) == len(_ALL_STEPS):
        result["processing_status"] = "complete"
    elif errors:
        result["processing_status"] = "partial"
        result["processing_errors"] = errors
        result["steps_completed"] = completed
    elif completed:
        result["processing_status"] = "processing"
        result["steps_completed"] = completed
        result["steps_remaining"] = [
            s for s in _ALL_STEPS
            if s not in steps_dict or steps_dict[s] != "done"
        ]
    else:
        result["processing_status"] = "pending"

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  App factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES

    # Register the API blueprint
    app.register_blueprint(bp)

    # Health-check root
    @app.route("/")
    def index():
        from .worker import worker
        return jsonify({
            "service": "DREAMS Analysis Pipeline API",
            "version": "0.1.0",
            "worker_active": worker.is_processing,
        })

    # Ensure DB schema exists (idempotent)
    with app.app_context():
        init_db()
        queue.init_queue()

    return app
