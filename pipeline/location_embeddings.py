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
import hdbscan

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_IMAGES_DIR, RAW_METADATA_PATH, LOCATION_TEXT_COLLECTION_NAME, LOCATION_IMAGE_COLLECTION_NAME
from db import get_collection
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

    return {
        "id": str(rec["id"]),
        "lat": lat,
        "lon": lon,
        "caption": rec.get("caption", ""),
        "timestamp": rec.get("timestamp", ""),
        "img_path": img_path,
        "geocode_display_name": geo.get("display_name") or "(unknown)",
        "description": desc
    }

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
        text_coll = get_collection(LOCATION_TEXT_COLLECTION_NAME)
        img_coll = get_collection(LOCATION_IMAGE_COLLECTION_NAME)
        
        if text_coll.count() == 0 or img_coll.count() == 0: 
            print("Collections are empty. Run pipeline first.")
            sys.exit(1)
            
        q_emb = clip_model.encode([args.query], normalize_embeddings=True).tolist()
        n_res = min(20, text_coll.count())
        
        # Search both modalities
        t_res = text_coll.query(query_embeddings=q_emb, n_results=n_res, include=["documents", "metadatas", "distances"])
        i_res = img_coll.query(query_embeddings=q_emb, n_results=n_res, include=["documents", "metadatas", "distances"])
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        metadata_map = {}
        doc_map = {}
        
        for k, rank_factor in [("text", t_res), ("image", i_res)]:
            if rank_factor and "ids" in rank_factor and rank_factor["ids"]:
                for rank, uid in enumerate(rank_factor["ids"][0]):
                    if uid not in rrf_scores:
                        rrf_scores[uid] = 0
                        metadata_map[uid] = rank_factor["metadatas"][0][rank]
                        doc_map[uid] = rank_factor["documents"][0][rank]
                    # RRF formula: 1 / (k + rank)
                    rrf_scores[uid] += 1.0 / (60 + rank)
                    
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top results for '{args.query}':")
        for uid, score in sorted_results[:min(10, len(sorted_results))]:
            m = metadata_map[uid]
            d = doc_map[uid]
            caption = m.get("caption", "N/A")
            timestamp = m.get("timestamp", "N/A")
            print(f"RRF={score:.4f} | {m.get('geocode_display_name')}")
            print(f"  Description: {d}")
            print(f"  Caption: {caption}")
            print(f"  Timestamp: {timestamp}")
            print("-" * 40)
        return

    ua = get_nominatim_user_agent()

    if args.image and args.lat is not None and args.lon is not None:
        clip_model = SentenceTransformer("clip-ViT-B-32")
        rid = hashlib.md5(f"{args.lat}:{args.lon}:{time.time()}".encode()).hexdigest()[:12]
        geo = asyncio.run(reverse_geocode(args.lat, args.lon, user_agent=ua))
        desc = format_geocode_data(geo)
        
        text_emb = clip_model.encode([desc], convert_to_numpy=True)
        img_emb = clip_model.encode([Image.open(args.image).convert("RGB")], convert_to_numpy=True)
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
        
        meta = {"lat": args.lat, "lon": args.lon, "geocode_display_name": geo.get("display_name"), "caption": "N/A", "timestamp": "N/A"}
        get_collection(LOCATION_TEXT_COLLECTION_NAME).upsert(ids=[rid], embeddings=text_emb.tolist(), metadatas=[meta])
        get_collection(LOCATION_IMAGE_COLLECTION_NAME).upsert(ids=[rid], embeddings=img_emb.tolist(), metadatas=[meta])
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
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)

    metas = [{"lat": r["lat"], "lon": r["lon"], "geocode_display_name": r["geocode_display_name"], "caption": r.get("caption", ""), "timestamp": r.get("timestamp", "")} for r in results]
    get_collection(LOCATION_TEXT_COLLECTION_NAME).upsert(ids=[r["id"] for r in results], embeddings=text_embs.tolist(), documents=texts, metadatas=metas)
    get_collection(LOCATION_IMAGE_COLLECTION_NAME).upsert(ids=[r["id"] for r in results], embeddings=img_embs.tolist(), documents=texts, metadatas=metas)

    # Compute joint distance matrix for clustering
    dist_txt = np.clip(1.0 - np.dot(text_embs, text_embs.T), 0.0, 2.0)
    dist_img = np.clip(1.0 - np.dot(img_embs, img_embs.T), 0.0, 2.0)
    joint_dist = (dist_txt + dist_img) / 2.0
    np.fill_diagonal(joint_dist, 0.0)
    
    # HDBSCAN Cython bindings requires float64 (double_t) for distance matrices.
    joint_dist = joint_dist.astype(np.float64)
    
    print("Running HDBSCAN clustering on joint multi-modal distance space...\n")
    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_samples=args.min_samples, min_cluster_size=args.min_samples)
    labels = clusterer.fit_predict(joint_dist)
    
    # Group results by cluster label
    from collections import defaultdict
    clusters = defaultdict(list)
    for res, label in zip(results, labels):
        clusters[label].append(res)
        
    for label in sorted(clusters.keys()):
        if label == -1:
            print(f"=== Noise (Unclustered: {len(clusters[label])} records) ===")
        else:
            print(f"=== Cluster {label} ({len(clusters[label])} records) ===")
            
        for r in clusters[label]:
            print(f"  [{r['id']}] {r['geocode_display_name']}")
            print(f"    Caption: {r.get('caption', 'N/A')[:60]}...")
            
        print()

if __name__ == "__main__":
    main()
