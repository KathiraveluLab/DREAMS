#!/usr/bin/env python3
"""
DREAMS Research Pipeline — Single-command orchestrator.

Usage:
    python pipeline/run_pipeline.py dataset.csv              # Full pipeline from CSV
    python pipeline/run_pipeline.py --source d1               # Pull from Cloudflare D1
    python pipeline/run_pipeline.py dataset.csv --only emotions,temporal
    python pipeline/run_pipeline.py dataset.csv --skip location_embeddings
    python pipeline/run_pipeline.py dataset.csv --resume      # Skip already-done steps
    python pipeline/run_pipeline.py dataset.csv --export      # Export parquet at the end
"""

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PIPELINE_STEPS, LOG_DIR, PROCESSED_DIR
from db import init_db, record_step_start, record_step_done, get_last_run_steps

# ── Step registry: maps step name → callable(logger) -> dict ──────────────

def _get_step_fn(step_name: str, cli_args):
    """Lazy-import each step to avoid loading all heavy models at startup."""

    if step_name == "import":
        from import_local_dataset import run as import_run
        from pull_data import run as pull_run
        if cli_args.source == "d1":
            return lambda log: pull_run(logger=log)
        else:
            return lambda log: import_run(Path(cli_args.csv_file), logger=log)

    if step_name == "emotions":
        from extract_emotions import run as fn
        return lambda log: fn(logger=log)

    if step_name == "temporal":
        from extract_temporal_features import run as fn
        return lambda log: fn(logger=log)

    if step_name == "location_embeddings":
        from extract_location_embeddings import run as fn
        return lambda log: fn(logger=log)

    if step_name == "caption_embeddings":
        from extract_caption_embeddings import run as fn
        return lambda log: fn(logger=log)

    if step_name == "image_embeddings":
        from extract_image_embeddings import run as fn
        return lambda log: fn(logger=log)

    if step_name == "verify":
        from verify_alignment import run as fn
        return lambda log: fn(logger=log)

    if step_name == "manifest":
        from create_master_manifest import run as fn
        return lambda log: fn(logger=log, export=cli_args.export)

    raise ValueError(f"Unknown step: {step_name}")


# ── Logging setup ─────────────────────────────────────────────────────────

def setup_logging(level_name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

    logger = logging.getLogger("dreams_pipeline")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))

    # Console handler — concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("  %(message)s"))
    logger.addHandler(ch)

    # File handler — detailed
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    logger.info("Log file: %s", log_file)
    return logger


# ── Summary table ─────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print a clean ASCII summary table."""
    W = 25 + 1 + 10 + 1 + 10 + 1 + 10  # total inner width
    col_step, col_status, col_records, col_time = 25, 10, 10, 10
    top = f"  ┌{'─' * col_step}┬{'─' * col_status}┬{'─' * col_records}┬{'─' * col_time}┐"
    mid = f"  ├{'─' * col_step}┼{'─' * col_status}┼{'─' * col_records}┼{'─' * col_time}┤"
    bot = f"  └{'─' * col_step}┴{'─' * col_status}┴{'─' * col_records}┴{'─' * col_time}┘"
    hdr = f"  │{'Step':<{col_step}}│{'Status':^{col_status}}│{'Records':^{col_records}}│{'Duration':^{col_time}}│"

    print(f"\n{top}\n{hdr}\n{mid}")
    total_time = 0.0
    for r in results:
        status_icon = {"ok": "✓ done", "skipped": "⊘ skip", "error": "✗ fail"}.get(r["status"], r["status"])
        records_str = str(r.get("records", "")) if r.get("records") else "—"
        dur = r.get("duration", 0.0)
        total_time += dur
        dur_str = f"{dur:.1f}s"
        print(f"  │{r['step']:<{col_step}}│{status_icon:^{col_status}}│{records_str:^{col_records}}│{dur_str:^{col_time}}│")
    print(mid)
    print(f"  │{'Total':<{col_step}}│{'':^{col_status}}│{'':^{col_records}}│{f'{total_time:.1f}s':^{col_time}}│")
    print(f"{bot}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DREAMS Research Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="Path to CSV dataset (required unless --source d1)")
    parser.add_argument("--source", choices=["csv", "d1"], default="csv",
                        help="Data source: 'csv' (default) or 'd1' (Cloudflare)")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of steps to run (e.g. 'emotions,temporal')")
    parser.add_argument("--skip", type=str, default=None,
                        help="Comma-separated list of steps to skip")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps that completed in the last run")
    parser.add_argument("--export", action="store_true",
                        help="Export master manifest to parquet")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args()

    # Validate args
    if args.source == "csv" and not args.csv_file:
        parser.error("csv_file is required when --source is csv")

    if args.csv_file and not Path(args.csv_file).exists():
        parser.error(f"CSV file not found: {args.csv_file}")

    # Setup
    logger = setup_logging(args.log_level)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]

    # Ensure DB schema exists (creates pipeline_runs table too)
    init_db().close()

    # Determine which steps to run
    all_steps = [name for name, _ in PIPELINE_STEPS]
    step_descriptions = dict(PIPELINE_STEPS)

    if args.only:
        selected = [s.strip() for s in args.only.split(",")]
        for s in selected:
            if s not in all_steps:
                parser.error(f"Unknown step '{s}'. Valid: {', '.join(all_steps)}")
        steps_to_run = selected
    else:
        steps_to_run = list(all_steps)

    if args.skip:
        skipped = {s.strip() for s in args.skip.split(",")}
        steps_to_run = [s for s in steps_to_run if s not in skipped]

    if args.resume:
        done_steps = get_last_run_steps()
        if done_steps:
            logger.info("Resuming: skipping %d already-completed steps", len(done_steps))
            steps_to_run = [s for s in steps_to_run if s not in done_steps]

    if not steps_to_run:
        logger.info("No steps to run.")
        return

    # Header
    print(f"\n  DREAMS Research Pipeline")
    print(f"  Run ID: {run_id}")
    print(f"  Steps:  {' → '.join(steps_to_run)}")
    print(f"  Source: {args.source}\n")

    # Execute
    summary = []
    had_error = False

    for step_name in steps_to_run:
        desc = step_descriptions.get(step_name, step_name)
        logger.info("━━━ [%s] %s ━━━", step_name, desc)

        record_step_start(run_id, step_name)
        t0 = time.time()

        try:
            fn = _get_step_fn(step_name, args)
            result = fn(logger)
            elapsed = time.time() - t0

            records = result.get("records_processed", 0)
            if result.get("status") == "error":
                record_step_done(run_id, step_name, records=records, error=result.get("error", "step returned error"))
                had_error = True
                logger.error("Step '%s' failed", step_name)
            else:
                record_step_done(run_id, step_name, records=records)

            summary.append({
                "step": step_name,
                "status": result.get("status", "ok"),
                "records": records,
                "duration": elapsed,
            })

        except Exception as exc:
            elapsed = time.time() - t0
            record_step_done(run_id, step_name, error=str(exc))
            summary.append({
                "step": step_name,
                "status": "error",
                "records": 0,
                "duration": elapsed,
            })
            had_error = True
            logger.error("Step '%s' raised an exception: %s", step_name, exc, exc_info=True)

    # Summary
    print_summary(summary)

    if had_error:
        logger.warning("Pipeline completed with errors. Use --resume to retry failed steps.")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
