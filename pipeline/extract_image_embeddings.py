import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, RAW_IMAGES_DIR, CLIP_MODEL, IMAGE_COLLECTION_NAME
from db import get_collection


def run(logger: logging.Logger | None = None) -> dict:
    """Encode images with CLIP and upsert to ChromaDB.

    Returns dict with keys: records_processed, status.
    """
    log = logger or logging.getLogger(__name__)

    if not RAW_METADATA_PATH.exists():
        log.error("Metadata not found: %s", RAW_METADATA_PATH)
        return {"records_processed": 0, "status": "error"}

    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading CLIP model (%s) on %s...", CLIP_MODEL, device)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    embeddings, record_infos = [], []

    records = metadata.get("records", [])
    for record in records:
        if not (local_image := record.get("local_image")):
            continue

        image_path = RAW_IMAGES_DIR / local_image
        if not image_path.exists():
            found = list(RAW_IMAGES_DIR.rglob(Path(local_image).name))
            if not found:
                continue
            image_path = found[0]

        try:
            with Image.open(image_path) as img:
                image_input = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image_input).cpu().numpy().flatten()
            rid = record.get("id")
            if rid is None or str(rid).strip() == "":
                log.warning("Skipping record with missing ID (image: %s)", local_image)
                continue
            embeddings.append(emb)
            record_infos.append({"id": str(rid), "user_id": record.get("user_id"), "local_image": local_image})
        except Exception as exc:
            log.warning("Failed to encode image for record %s: %s", record.get("id"), exc)

    if not embeddings:
        log.warning("No image embeddings extracted")
        return {"records_processed": 0, "status": "skipped"}

    log.info("Encoding %d images...", len(embeddings))
    get_collection(IMAGE_COLLECTION_NAME).upsert(
        ids=[r["id"] for r in record_infos],
        embeddings=np.stack(embeddings).tolist(),
        metadatas=[{"user_id": r["user_id"], "local_image": r["local_image"]} for r in record_infos]
    )

    log.info("Image embeddings: %d records processed", len(embeddings))
    return {"records_processed": len(embeddings), "status": "ok"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
