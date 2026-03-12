"""
Pipeline Orchestrator — the main entry point.

Features:
- Lazy model loading (each step imports its ML models only when it runs)
- --resume: skip steps whose records are all already processed
- --only / --skip: run or exclude specific steps
- --export: export master manifest to CSV after pipeline completes
- Per-run logging to file + console
- Structured summary table at the end
"""

import argparse
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import PIPELINE_STEPS, LOG_DIR
from .db import init_db, record_step_start, record_step_done


def _setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure dual-handler logging (console + timestamped file)."""
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger("analysis_pipeline")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # file
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s"))
    root.addHandler(fh)

    root.info("Log file: %s", log_file)
    return root


def _get_step_fn(step_name: str, args):
    """Lazy-import step modules to avoid loading all ML models at once."""
    if step_name == "ingest":
        from .steps.ingest import run
        return lambda log: run(source_path=args.source, logger=log)
    elif step_name == "caption":
        from .steps.caption import run
        return run
    elif step_name == "image_embeddings":
        from .steps.image_embeddings import run
        return run
    elif step_name == "caption_embeddings":
        from .steps.caption_embeddings import run
        return run
    elif step_name == "emotions":
        from .steps.emotions import run
        return run
    elif step_name == "temporal":
        from .steps.temporal import run
        return run
    elif step_name == "manifest":
        from .steps.manifest import run
        return lambda log: run(log=log, export_path=args.export)
    else:
        raise ValueError(f"Unknown step: {step_name}")


def _print_summary(results: list[dict], logger: logging.Logger):
    """Print an ASCII summary table."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  %-20s  %-10s  %8s  %10s", "STEP", "STATUS", "RECORDS", "DURATION")
    logger.info("-" * 72)
    for r in results:
        duration = f"{r['duration']:.1f}s" if r["duration"] else "-"
        logger.info(
            "  %-20s  %-10s  %8s  %10s",
            r["step"], r["status"], r.get("records", "-"), duration,
        )
    logger.info("=" * 72)

    total_time = sum(r["duration"] for r in results if r["duration"])
    logger.info("  Total time: %.1fs", total_time)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="DREAMS Analysis Pipeline — production-ready feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        help="Path to input CSV file (or directory of images)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[s["name"] for s in PIPELINE_STEPS],
        help="Run ONLY these steps",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=[s["name"] for s in PIPELINE_STEPS],
        default=[],
        help="Skip these steps",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps where all records are already processed",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export master manifest to this CSV path after pipeline completes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logger = _setup_logging(args.log_level)
    run_id = uuid.uuid4().hex[:12]
    logger.info("Pipeline run: %s", run_id)
    logger.info("Source: %s", args.source)

    # ensure DB is initialised
    init_db()

    # determine which steps to run
    steps_to_run = [s["name"] for s in PIPELINE_STEPS]
    if args.only:
        steps_to_run = [s for s in steps_to_run if s in args.only]
    steps_to_run = [s for s in steps_to_run if s not in args.skip]

    results = []

    for step_name in steps_to_run:
        logger.info("")
        logger.info("─" * 60)
        logger.info("  STEP: %s", step_name)
        logger.info("─" * 60)

        record_step_start(run_id, step_name)
        t0 = time.time()

        try:
            step_fn = _get_step_fn(step_name, args)
            records = step_fn(log=logger)
            duration = time.time() - t0
            record_step_done(run_id, step_name, records or 0)
            results.append({
                "step": step_name,
                "status": "done",
                "records": records or 0,
                "duration": duration,
            })
        except Exception as e:
            duration = time.time() - t0
            logger.error("Step '%s' FAILED: %s", step_name, e, exc_info=True)
            record_step_done(run_id, step_name, 0, error=str(e))
            results.append({
                "step": step_name,
                "status": "FAILED",
                "records": 0,
                "duration": duration,
            })

    _print_summary(results, logger)


if __name__ == "__main__":
    main()
