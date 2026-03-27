"""
Step 4 — Caption Embeddings: Compute MiniLM sentence vectors → ChromaDB.

Embeds the best available caption (user-written preferred over BLIP-generated)
using all-MiniLM-L6-v2 into a 384-dimensional vector.
"""

import gc
import logging

from ..config import SENTENCE_MODEL_NAME, BATCH_SIZE
from ..db import (
    get_db, get_collection,
    get_pending_ids, mark_record_done, mark_record_error,
)
from ..utils import clean_text

logger = logging.getLogger(__name__)


def run(log: logging.Logger | None = None) -> int:
    """Encode captions with MiniLM and upsert into ChromaDB."""
    _log = log or logger

    pending = get_pending_ids("caption_embeddings")
    if not pending:
        _log.info("All captions already have embeddings.")
        return 0

    from sentence_transformers import SentenceTransformer

    _log.info("Loading sentence model: %s", SENTENCE_MODEL_NAME)
    model = None
    conn = None
    try:
        conn = get_db()
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
        _log.info("Sentence model loaded.")

        collection = get_collection("caption_embeddings")
        placeholders = ",".join("?" for _ in pending)
        rows = conn.execute(
            f"""SELECT memory_id, caption, generated_caption, user_id
                FROM memories WHERE memory_id IN ({placeholders})""",
            pending,
        ).fetchall()
        rows_by_id = {r["memory_id"]: r for r in rows}
        _log.info("Computing caption embeddings for %d records...", len(pending))
        processed = 0

        for i in range(0, len(pending), BATCH_SIZE):
            batch_ids = pending[i : i + BATCH_SIZE]
            texts = []
            valid_ids = []
            metas = []

            for mid in batch_ids:
                row = rows_by_id.get(mid)
                if not row:
                    continue
                # prefer user caption, fall back to generated
                text = row["caption"] or row["generated_caption"]
                if not text or not text.strip():
                    mark_record_error(mid, "caption_embeddings", "No caption available", conn)
                    continue

                texts.append(clean_text(text))
                valid_ids.append(mid)
                metas.append({
                    "user_id": row["user_id"],
                    "caption_source": "user" if row["caption"] else "generated",
                })

            if not texts:
                continue

            try:
                embeddings = model.encode(texts, normalize_embeddings=True)
                emb_list = embeddings.tolist()

                collection.upsert(
                    ids=valid_ids,
                    embeddings=emb_list,
                    metadatas=metas,
                    documents=texts,
                )

                for mid in valid_ids:
                    mark_record_done(mid, "caption_embeddings", conn)
                    processed += 1

            except Exception as e:
                _log.error("Caption embedding batch failed: %s", e)
                for mid in valid_ids:
                    mark_record_error(mid, "caption_embeddings", str(e), conn)

            conn.commit()
            _log.info("  Embedded %d / %d", min(i + BATCH_SIZE, len(pending)), len(pending))

        _log.info("Caption embeddings complete: %d processed.", processed)
        return processed
    finally:
        conn.close()
        # Free MiniLM from RAM
        if model is not None:
            del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            _log.warning("Failed to empty CUDA cache: %s", e)
        _log.info("Sentence model unloaded or skipped.")
