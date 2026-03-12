"""
Hard reset — wipes ALL pipeline state so you can start completely fresh.
Deletes: SQLite DB, ChromaDB vectors, uploaded images, snapshots, logs, manifests.

Usage
-----
    python reset_pipeline.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PIPELINE = ROOT / "analysis_pipeline"

TARGETS = [
    # SQLite database (all jobs, memories, processing state)
    PIPELINE / "data" / "pipeline.db",

    # ChromaDB vector store (all embeddings)
    PIPELINE / "data" / "chromadb",

    # Processed images (renamed copies stored by memory_id)
    PIPELINE / "data" / "processed",

    # Raw uploaded images saved by the API
    PIPELINE / "data" / "raw" / "images",

    # Pipeline snapshots
    PIPELINE / "data" / "snapshots",

    # Log files
    PIPELINE / "logs",
]

def reset():
    print("\n" + "=" * 60)
    print("  DREAMS PIPELINE — HARD RESET")
    print("=" * 60)

    confirm = input("\n  This will delete ALL records, embeddings, images,\n"
                    "  logs and job history.\n\n"
                    "  Type  YES  to confirm: ").strip()

    if confirm != "YES":
        print("\n  Aborted — nothing was deleted.\n")
        sys.exit(0)

    print()
    deleted   = []
    not_found = []

    for target in TARGETS:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target.relative_to(ROOT)))
            print(f"  ✓  Deleted  {target.relative_to(ROOT)}")
        else:
            not_found.append(str(target.relative_to(ROOT)))
            print(f"  –  Missing  {target.relative_to(ROOT)}  (skipped)")

    # Re-create the empty directory structure the pipeline expects
    dirs_to_recreate = [
        PIPELINE / "data" / "processed",
        PIPELINE / "data" / "chromadb",
        PIPELINE / "data" / "cache",
        PIPELINE / "data" / "raw" / "images",
        PIPELINE / "data" / "snapshots",
        PIPELINE / "logs",
    ]

    print()
    for d in dirs_to_recreate:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓  Recreated  {d.relative_to(ROOT)}")

    print()
    print("=" * 60)
    print(f"  Reset complete.")
    print(f"  Deleted  : {len(deleted)} item(s)")
    print(f"  Skipped  : {len(not_found)} item(s) (already missing)")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("  1.  python -m analysis_pipeline.api       ← start the server")
    print("  2.  POST http://localhost:5001/api/ingest  ← upload a fresh image")
    print("  3.  GET  http://localhost:5001/api/analysis/<memory_id>")
    print()

if __name__ == "__main__":
    reset()