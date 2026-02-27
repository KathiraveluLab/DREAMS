import json
import logging
import os
import shutil
import sys
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db

D1_API_TOKEN = os.getenv("D1_API_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
D1_DATABASE_ID = os.getenv("D1_DATABASE_ID", "")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_PATH = RAW_DIR / "metadata.json"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"


def run(logger: logging.Logger | None = None) -> dict:
    """Pull data from Cloudflare D1, download images, and ingest into SQLite.

    Returns dict with keys: records_processed, status.
    """
    log = logger or logging.getLogger(__name__)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not all([D1_API_TOKEN, CF_ACCOUNT_ID, D1_DATABASE_ID]):
        log.error("Missing D1 credentials (D1_API_TOKEN, CF_ACCOUNT_ID, D1_DATABASE_ID)")
        return {"records_processed": 0, "status": "error"}

    try:
        resp = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query",
            headers={"Authorization": f"Bearer {D1_API_TOKEN}", "Content-Type": "application/json"},
            json={"sql": "SELECT id, user_id, caption, timestamp, lat, lon, image_url, processed, processing_version, created_at FROM memories WHERE processed = 0 OR processed IS NULL"},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            log.error("D1 API returned unsuccessful response")
            return {"records_processed": 0, "status": "error"}
        result = data.get("result")
        if not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], dict):
            log.error("D1 API returned unexpected payload shape")
            return {"records_processed": 0, "status": "error"}
        records = result[0].get("results", [])
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        log.error("D1 API request/parse failed: %s", e)
        return {"records_processed": 0, "status": "error"}

    if not records:
        log.info("No unprocessed records in D1")
        return {"records_processed": 0, "status": "skipped"}

    processed, stats, now = [], {}, datetime.utcnow()
    iso_time = now.isoformat() + "Z"

    for r in records:
        uid = r.get("user_id", "unknown")
        stats.setdefault(uid, {"count": 0, "images": 0})["count"] += 1

        local_img = None
        if url := r.get("image_url"):
            clean_uid = Path(uid).name.replace("..", "_").replace("/", "_").replace("\\", "_") or "unknown"
            (user_dir := IMAGES_DIR / clean_uid).mkdir(parents=True, exist_ok=True)

            try:
                img_resp = requests.get(url, timeout=30)
                img_resp.raise_for_status()
                dst = user_dir / f"img_{r.get('id', 0):03d}{Path(urlparse(url).path).suffix or '.jpg'}"
                dst.write_bytes(img_resp.content)
                local_img = f"{clean_uid}/{dst.name}"
                stats[uid]["images"] += 1
            except requests.RequestException:
                log.warning("Failed to download image for record %s", r.get("id"))

        processed.append({
            "id": r.get("id", 0), "user_id": uid, "caption": r.get("caption", ""),
            "timestamp": r.get("timestamp", ""), "lat": r.get("lat"), "lon": r.get("lon"),
            "image_url": url, "local_image": local_img, "processed": r.get("processed", 0),
            "processing_version": r.get("processing_version"), "created_at": r.get("created_at", "")
        })

    snapshot = {"snapshot_id": now.strftime("snapshot_%Y_%m_%d"), "created_at": iso_time, "record_count": len(processed), "user_count": len(stats), "user_stats": stats, "records": processed}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    conn = init_db()
    conn.execute("DELETE FROM memories")
    for r in processed:
        conn.execute("INSERT OR REPLACE INTO memories (id, user_id, caption, timestamp, lat, lon, image_url, local_image, snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (r["id"], r["user_id"], r["caption"], r["timestamp"], r["lat"], r["lon"], r["image_url"], r["local_image"], snapshot["snapshot_id"], r["created_at"]))
    conn.commit()
    conn.close()

    snap_dir = SNAPSHOTS_DIR / (snapshot["snapshot_id"] if not (SNAPSHOTS_DIR / snapshot["snapshot_id"]).exists() else f"{snapshot['snapshot_id']}_{now.strftime('%H%M%S')}")
    snap_dir.mkdir(parents=True, exist_ok=True)

    if IMAGES_DIR.exists():
        shutil.copytree(IMAGES_DIR, snap_dir / "images", dirs_exist_ok=True)

    with open(snap_dir / "metadata.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    log.info("Pulled %d records (%d users) from D1", len(processed), len(stats))
    return {"records_processed": len(processed), "status": "ok"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
