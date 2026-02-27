import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_IMAGES_DIR, RAW_METADATA_PATH, LOCATION_COLLECTION_NAME
from db import get_collection
from geocoder import reverse_geocode

def get_nominatim_user_agent():
    return "dreams-research/1.0 (contact@dreams-research.org)"

def format_geocode_data(geocode_data: dict) -> str:
    if not geocode_data or not geocode_data.get("raw"):
        return "Unknown location without specific geographic features"
    
    raw = geocode_data["raw"]
    address = geocode_data.get("address", {})
    
    parts = []
    if "city" in address:
        parts.append(f"City: {address['city']}")
    elif "town" in address:
        parts.append(f"Town: {address['town']}")
    elif "village" in address:
        parts.append(f"Village: {address['village']}")
    elif "county" in address:
        parts.append(f"County: {address['county']}")

    if "suburb" in address:
        parts.append(f"Suburb: {address['suburb']}")
    elif "neighbourhood" in address:
        parts.append(f"Neighbourhood: {address['neighbourhood']}")

    place_type = raw.get("type", "").replace("_", " ")
    place_category = raw.get("category", "").replace("_", " ")
    if place_type:
        parts.append(f"Type: {place_type}")
    if place_category:
        parts.append(f"Category: {place_category}")

    if "amenity" in address:
        parts.append(f"Amenity: {address['amenity'].replace('_', ' ')}")
    if "building" in address:
        parts.append(f"Building: {address['building'].replace('_', ' ')}")
    if "leisure" in address:
        parts.append(f"Leisure: {address['leisure'].replace('_', ' ')}")
    if "natural" in address:
        parts.append(f"Natural: {address['natural'].replace('_', ' ')}")

    seen = set()
    unique_parts = [x for x in parts if not (x in seen or seen.add(x))]
    return "Location characteristics: " + ", ".join(unique_parts) if unique_parts else "Unknown location without specific geographic features"

async def process(rec, ua):
    if (lat := rec.get("lat")) is None or (lon := rec.get("lon")) is None:
        return None
    lat, lon = float(lat), float(lon)

    img_path = next((p for p in [RAW_IMAGES_DIR / rec.get("local_image", ""), RAW_IMAGES_DIR / f"{rec.get('id')}.jpg", RAW_IMAGES_DIR / f"{rec.get('id')}.png"] if p.exists()), None)
    if not img_path:
        return None

    try:
        geo = await reverse_geocode(lat, lon, user_agent=ua)
    except Exception:
        geo = {"display_name": None, "address": None, "raw": None}

    desc = format_geocode_data(geo)

    return {"id": str(rec["id"]), "lat": lat, "lon": lon, "img_path": img_path, "geocode_display_name": geo.get("display_name") or "(unknown)", "description": desc}

def encode_multimodal(clip_model, text: str, img_path: str) -> np.ndarray:
    text_emb = clip_model.encode([text], convert_to_numpy=True)
    img_emb = clip_model.encode([Image.open(img_path).convert("RGB")], convert_to_numpy=True)
    multi_emb = (text_emb + img_emb) / 2.0
    return multi_emb / np.linalg.norm(multi_emb, axis=1, keepdims=True)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--batch", action="store_true")
    args.add_argument("--query", type=str)
    args.add_argument("--image", type=str)
    args.add_argument("--lat", type=float)
    args.add_argument("--lon", type=float)
    args.add_argument("--eps", type=float, default=0.3)
    args.add_argument("--min-samples", type=int, default=2)
    args = args.parse_args()

    if args.query:
        clip_model = SentenceTransformer("clip-ViT-B-32")
        coll = get_collection(LOCATION_COLLECTION_NAME)
        if coll.count() == 0: sys.exit(1)
        res = coll.query(query_embeddings=clip_model.encode([args.query], normalize_embeddings=True).tolist(), n_results=min(3, coll.count()), include=["documents", "metadatas", "distances"])
        for d, m in zip(res["documents"][0], res["metadatas"][0]): print(f"{m.get('geocode_display_name')}: {d}")
        return

    ua = get_nominatim_user_agent()

    if args.image and args.lat is not None and args.lon is not None:
        clip_model = SentenceTransformer("clip-ViT-B-32")
        rid = hashlib.md5(f"{args.lat}:{args.lon}:{time.time()}".encode()).hexdigest()[:12]
        geo = asyncio.run(reverse_geocode(args.lat, args.lon, user_agent=ua))
        desc = format_geocode_data(geo)
        
        emb = encode_multimodal(clip_model, desc, args.image)
        
        get_collection(LOCATION_COLLECTION_NAME).upsert(ids=[rid], embeddings=emb.tolist(), metadatas=[{"lat": args.lat, "lon": args.lon, "geocode_display_name": geo.get("display_name")}])
        return

    if not RAW_METADATA_PATH.exists(): sys.exit(1)
    with open(RAW_METADATA_PATH) as f: records = json.load(f).get("records", [])

    results = [res for rec in records if (res := asyncio.run(process(rec, ua)))]
    if not results: sys.exit(1)

    clip_model = SentenceTransformer("clip-ViT-B-32")
    texts = [r["description"] for r in results]
    images = [Image.open(r["img_path"]).convert("RGB") for r in results]
    
    text_embs = clip_model.encode(texts, convert_to_numpy=True)
    img_embs = clip_model.encode(images, convert_to_numpy=True)
    multi_embs = (text_embs + img_embs) / 2.0
    multi_embs = multi_embs / np.linalg.norm(multi_embs, axis=1, keepdims=True)

    get_collection(LOCATION_COLLECTION_NAME).upsert(ids=[r["id"] for r in results], embeddings=multi_embs.tolist(), documents=[r["description"] for r in results], metadatas=[{"lat": r["lat"], "lon": r["lon"], "geocode_display_name": r["geocode_display_name"]} for r in results])

    dist = np.clip(1.0 - np.dot(multi_embs, multi_embs.T), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed").fit_predict(dist)

if __name__ == "__main__":
    main()
