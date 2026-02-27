import logging
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME
from db import init_db, get_collection


def run(logger: logging.Logger | None = None, export: bool = False) -> dict:
    """Report master manifest statistics and optionally export to parquet.

    Returns dict with keys: records_processed, status.
    """
    log = logger or logging.getLogger(__name__)

    conn = init_db()
    counts = {}

    try:
        for table in ["memories", "emotion_scores", "temporal_features", "location_descriptions", "master_manifest"]:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            log.info("  %s: %d rows", table, count)
            counts[table] = count

        for col in ["valence", "arousal", "sin_hour"]:
            nulls = conn.execute(f"SELECT count(*) FROM master_manifest WHERE {col} IS NULL").fetchone()[0]
            if nulls > 0:
                log.info("  %s NULLs: %d", col, nulls)

        for coll in [IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME]:
            count = get_collection(coll).count()
            log.info("  %s: %d vectors", coll, count)
            counts[coll] = count

        row = conn.execute("SELECT * FROM master_manifest LIMIT 1").fetchone()
        if row:
            cols = [d[0] for d in conn.execute("SELECT * FROM master_manifest LIMIT 0").description]
            non_null = sum(1 for v in row if v is not None)
            log.info("  Sample row: %d / %d columns populated", non_null, len(cols))

        if export:
            import pandas as pd
            out_path = PROCESSED_DIR / "master_manifest.parquet"
            pd.read_sql_query("SELECT * FROM master_manifest", conn).to_parquet(out_path, index=False)
            log.info("Exported manifest to %s", out_path)

    except Exception as e:
        log.error("Manifest report failed: %s", e)
        return {"records_processed": counts.get("master_manifest", 0), "status": "error", "error": str(e)}
    finally:
        conn.close()

    log.info("Manifest report complete")
    return {"records_processed": counts.get("master_manifest", 0), "status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(export=args.export)
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
