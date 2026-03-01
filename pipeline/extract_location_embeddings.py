import asyncio
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, RAW_IMAGES_DIR, LOCATION_TEXT_COLLECTION_NAME, LOCATION_IMAGE_COLLECTION_NAME
from db import init_db, get_collection
from geocoder import reverse_geocode


def get_nominatim_user_agent():
    return "dreams-research/1.0 (contact@dreams-research.org)"


def format_geocode_data(geocode_data: dict) -> str:
    """Format geocode data deterministically for CLIP encode"""
    if not geocode_data or not geocode_data.get("raw"):
        return "A specific geographic location without known features."

    raw = geocode_data["raw"]
    address = geocode_data.get("address", {})

    locality = address.get("city") or address.get("town") or address.get("village") or address.get("county") or ""
    sublocality = address.get("suburb") or address.get("neighbourhood") or ""
    
    place_category = raw.get("category", "").replace("_", " ")
    place_type = raw.get("type", "").replace("_", " ")

    feature = address.get("amenity") or address.get("building") or address.get("leisure") or address.get("natural") or ""
    if feature: feature = feature.replace("_", " ")

    desc = "A photo of"
    if feature and place_category:
        desc += f" a {feature} which is a {place_category}"
    elif feature:
        desc += f" a {feature}"
    elif place_category and place_type:
        desc += f" a {place_type} ({place_category})"
    elif place_type:
        desc += f" a {place_type}"
    else:
        desc += " a specific location"

    if sublocality and locality:
        desc += f", located in {sublocality}, {locality}."
    elif locality:
        desc += f", located in {locality}."
    elif sublocality:
        desc += f", located in {sublocality}."
    else:
        desc += "."

    return desc


async def process_metadata(rec, ua, log):
    if (lat := rec.get("lat")) is None or (lon := rec.get("lon")) is None: return None
    lat, lon = float(lat), float(lon)

    img_path = next((p for p in [RAW_IMAGES_DIR / rec.get("local_image", ""), RAW_IMAGES_DIR / f"{rec.get('id')}.jpg", RAW_IMAGES_DIR / f"{rec.get('id')}.png"] if p.exists()), None)
    if not img_path: return None

    try: geocode = await reverse_geocode(lat, lon, user_agent=ua)
    except Exception as e:
        log.warning("Geocoding failed for record %s: %s", rec.get("id"), e)
        geocode = {"display_name": None, "address": None, "raw": None}

    desc = format_geocode_data(geocode)

    return {
        "id": str(rec["id"]),
        "user_id": rec.get("user_id"),
        "lat": lat,
        "lon": lon,
        "caption": rec.get("caption", ""),
        "timestamp": rec.get("timestamp", ""),
        "img_path": img_path,
        "geocode_display_name": geocode.get("display_name") or "(unknown)",
        "description": desc
    }


def run(logger: logging.Logger | None = None) -> dict:
    """Geocode locations and compute CLIP multi-modal embeddings.

    Returns dict with keys: records_processed, status.
    """
    log = logger or logging.getLogger(__name__)

    if not RAW_METADATA_PATH.exists():
        log.error("Metadata not found: %s", RAW_METADATA_PATH)
        return {"records_processed": 0, "status": "error"}

    with open(RAW_METADATA_PATH) as f:
        records = json.load(f).get("records", [])

    ua = get_nominatim_user_agent()
    log.info("Loaded %d records. Reverse geocoding...", len(records))

    async def _geocode_all():
        return [res for rec in records if (res := await process_metadata(rec, ua, log))]

    results = asyncio.run(_geocode_all())

    if not results:
        log.warning("No valid records found for location embedding")
        return {"records_processed": 0, "status": "skipped"}

    log.info("Loading CLIP model and computing %d multi-modal embeddings...", len(results))
    clip_model = SentenceTransformer("clip-ViT-B-32")

    # Batch encode texts
    texts = [r["description"] for r in results]
    text_embs = clip_model.encode(texts, convert_to_numpy=True)

    # Batch encode images
    images = [Image.open(r["img_path"]).convert("RGB") for r in results]
    img_embs = clip_model.encode(images, convert_to_numpy=True)

    # Normalization (Crucial for cosine similarity in ChromaDB)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)

    log.info("Upserting text embeddings to ChromaDB...")
    get_collection(LOCATION_TEXT_COLLECTION_NAME).upsert(
        ids=[r["id"] for r in results],
        embeddings=text_embs.tolist(),
        documents=[r["description"] for r in results],
        metadatas=[{"user_id": r["user_id"], "lat": r["lat"], "lon": r["lon"], "geocode_display_name": r["geocode_display_name"], "caption": r["caption"], "timestamp": r["timestamp"]} for r in results]
    )

    log.info("Upserting image embeddings to ChromaDB...")
    get_collection(LOCATION_IMAGE_COLLECTION_NAME).upsert(
        ids=[r["id"] for r in results],
        embeddings=img_embs.tolist(),
        documents=[r["description"] for r in results],
        metadatas=[{"user_id": r["user_id"], "lat": r["lat"], "lon": r["lon"], "geocode_display_name": r["geocode_display_name"], "caption": r["caption"], "timestamp": r["timestamp"]} for r in results]
    )

    log.info("Saving text representations to SQLite...")
    conn = init_db()
    for r in results:
        conn.execute("INSERT OR REPLACE INTO location_descriptions (id, user_id, description, geocode_display_name, image_caption) VALUES (?, ?, ?, ?, ?)", (int(r["id"]), r["user_id"], r["description"], r["geocode_display_name"], ""))
    conn.commit()
    conn.close()

    log.info("Location embeddings: %d records processed", len(results))
    return {"records_processed": len(results), "status": "ok"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
