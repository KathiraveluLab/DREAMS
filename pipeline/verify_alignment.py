import logging
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from config import SENTENCE_BERT_MODEL, IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME
from db import init_db, get_collection


def run(logger: logging.Logger | None = None) -> dict:
    """Verify alignment between SQLite tables and ChromaDB collections.

    Returns dict with keys: records_processed, status, details.
    """
    log = logger or logging.getLogger(__name__)

    conn = init_db()

    try:
        mem_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        man_count = conn.execute("SELECT count(*) FROM master_manifest").fetchone()[0]
        log.info("Memories: %d rows | Master Manifest: %d rows", mem_count, man_count)

        if mem_count == 0:
            log.warning("No memories in database, nothing to verify")
            return {"records_processed": 0, "status": "skipped"}

        details = {"memories": mem_count, "master_manifest": man_count}

        for t in ("emotion_scores", "temporal_features"):
            count = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            log.info("  %s: %d rows", t, count)
            details[t] = count

        for c in (IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME):
            count = get_collection(c).count()
            log.info("  %s: %d vectors", c, count)
            details[c] = count

        cap_coll = get_collection(CAPTION_COLLECTION_NAME)
        if cap_coll.count() > 0:
            if row := conn.execute("SELECT id, caption FROM memories WHERE caption IS NOT NULL AND caption != '' LIMIT 1").fetchone():
                rid, cap = str(row[0]), row[1]
                stored = cap_coll.get(ids=[rid], include=["embeddings"])["embeddings"]
                if stored is not None and len(stored) > 0:
                    sim = cosine_similarity([stored[0]], [SentenceTransformer(SENTENCE_BERT_MODEL).encode([cap], normalize_embeddings=True)[0]])[0][0]
                    log.info("  Embedding consistency check: cosine similarity = %.4f", sim)
                    details["embedding_cosine_similarity"] = round(float(sim), 4)
    finally:
        conn.close()

    log.info("Verification complete")
    return {"records_processed": mem_count, "status": "ok", "details": details}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run()
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
