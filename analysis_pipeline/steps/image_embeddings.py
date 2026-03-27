"""
Step 3 — Image Embeddings: Compute CLIP image vectors → ChromaDB.

Uses HuggingFace CLIPModel (openai/clip-vit-base-patch32) to encode
each photo into a 512-dimensional vector.  Stores in ChromaDB
collection 'image_embeddings'.

Improvement: per-record checkpointing — only processes images not yet
in ChromaDB, so a crash mid-way doesn't require reprocessing.

NOTE: Uses transformers.CLIPModel instead of OpenAI's `clip` package
so that installation works via plain `pip install transformers` without
needing `git` or a GitHub clone.
"""

import gc
import logging
from pathlib import Path

from ..config import BATCH_SIZE, PROCESSED_DIR, DATA_DIR
from ..db import (
    get_db, get_collection,
    get_pending_ids, mark_record_done, mark_record_error,
)
from ..utils import validate_safe_path

logger = logging.getLogger(__name__)

# HuggingFace model ID (equivalent to OpenAI CLIP ViT-B/32)
_HF_CLIP_MODEL = "openai/clip-vit-base-patch32"


def run(log: logging.Logger | None = None) -> int:
    """Encode images with CLIP and upsert into ChromaDB."""
    _log = log or logger

    pending = get_pending_ids("image_embeddings")
    if not pending:
        _log.info("All images already have embeddings.")
        return 0

    # lazy-load heavy dependencies
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log.info("Loading CLIP model '%s' on %s", _HF_CLIP_MODEL, device)

    model = None
    processor = None
    conn = get_db()
    try:
        model = CLIPModel.from_pretrained(_HF_CLIP_MODEL).to(device)
        processor = CLIPProcessor.from_pretrained(_HF_CLIP_MODEL)
        model.eval()
        _log.info("CLIP loaded.")

        collection = get_collection("image_embeddings")
        # fetch image paths for pending records
        placeholders = ",".join("?" for _ in pending)
        rows = conn.execute(
            f"SELECT memory_id, image_path, user_id FROM memories WHERE memory_id IN ({placeholders})",
            pending,
        ).fetchall()

        rows_by_id = {r["memory_id"]: r for r in rows}
        _log.info("Computing image embeddings for %d records...", len(pending))
        processed = 0

        for i in range(0, len(pending), BATCH_SIZE):
            batch_ids = pending[i : i + BATCH_SIZE]
            batch_images = []
            batch_valid_ids = []
            batch_metas = []

            for mid in batch_ids:
                row = rows_by_id.get(mid)
                if not row or not row["image_path"]:
                    mark_record_error(mid, "image_embeddings", "No image path", conn)
                    continue
                try:
                    img_path = validate_safe_path(
                        row["image_path"],
                        allowed_roots=[PROCESSED_DIR, DATA_DIR],
                    )
                except ValueError:
                    mark_record_error(mid, "image_embeddings", "Image path outside allowed dirs", conn)
                    continue
                if not img_path.exists():
                    mark_record_error(mid, "image_embeddings", "Image file missing", conn)
                    continue
                try:
                    img = Image.open(str(img_path)).convert("RGB")
                    batch_images.append(img)
                    batch_valid_ids.append(mid)
                    batch_metas.append({"user_id": row["user_id"]})
                except Exception as e:
                    _log.warning("Image load failed %s: %s", mid, e)
                    mark_record_error(mid, "image_embeddings", str(e), conn)

            if not batch_images:
                continue

            try:
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    embeddings = model.get_image_features(**inputs)
                    # L2-normalise so cosine similarity = dot product
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

                emb_list = embeddings.cpu().numpy().tolist()

                # upsert to ChromaDB
                collection.upsert(
                    ids=batch_valid_ids,
                    embeddings=emb_list,
                    metadatas=batch_metas,
                )

                for mid in batch_valid_ids:
                    mark_record_done(mid, "image_embeddings", conn)
                    processed += 1

            except Exception as e:
                _log.error("Batch embedding failed: %s", e)
                for mid in batch_valid_ids:
                    mark_record_error(mid, "image_embeddings", str(e), conn)

            conn.commit()
            _log.info("  Embedded %d / %d", min(i + BATCH_SIZE, len(pending)), len(pending))

        _log.info("Image embeddings complete: %d processed.", processed)
        return processed
    finally:
        conn.close()
        # Free CLIP from RAM
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log.info("CLIP model unloaded or skipped.")
