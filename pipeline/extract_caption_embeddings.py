import json
import logging
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, SENTENCE_BERT_MODEL, CAPTION_COLLECTION_NAME
from db import get_collection


def run(logger: logging.Logger | None = None) -> dict:
    """Encode captions with Sentence-BERT and upsert to ChromaDB.

    Returns dict with keys: records_processed, status.
    """
    log = logger or logging.getLogger(__name__)

    if not RAW_METADATA_PATH.exists():
        log.error("Metadata not found: %s", RAW_METADATA_PATH)
        return {"records_processed": 0, "status": "error"}

    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    model = SentenceTransformer(SENTENCE_BERT_MODEL)

    captions = []
    record_infos = []

    for record in metadata.get("records", []):
        raw_caption = record.get("caption", "")
        if not raw_caption or raw_caption == "[REDACTED]":
            continue

        caption = unicodedata.normalize("NFC", raw_caption).strip()
        if not caption:
            continue

        rid = record.get("id")
        if rid is None or str(rid).strip() == "":
            continue

        captions.append(caption)
        record_infos.append({
            "id": str(rid),
            "user_id": record.get("user_id"),
            "caption": caption,
        })

    if not captions:
        log.warning("No valid captions found")
        return {"records_processed": 0, "status": "skipped"}

    log.info("Encoding %d captions with Sentence-BERT...", len(captions))
    embeddings = model.encode(captions, convert_to_numpy=True, normalize_embeddings=True)

    collection = get_collection(CAPTION_COLLECTION_NAME)
    collection.upsert(
        ids=[r["id"] for r in record_infos],
        embeddings=embeddings.tolist(),
        documents=[r["caption"] for r in record_infos],
        metadatas=[{"user_id": r["user_id"]} for r in record_infos],
    )

    log.info("Caption embeddings: %d records processed", len(captions))
    return {"records_processed": len(captions), "status": "ok"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
