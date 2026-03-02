"""
Step 7 — Temporal: Extract cyclical temporal features from timestamps.

Encodes time information as cyclical sin/cos features (so that 23:00 and
01:00 are close together, December and January are close, etc.) plus
human-readable buckets (time_of_day, season).

Also computes relative_day: days since the user's first memory, which is
critical for tracking recovery trajectory.
"""

import logging
import math

from ..db import get_db, get_pending_ids, mark_record_done, mark_record_error
from ..utils import parse_timestamp

logger = logging.getLogger(__name__)


def _time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "night"


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    return "fall"


def run(log: logging.Logger | None = None) -> int:
    """Compute temporal features for all pending records."""
    _log = log or logger

    pending = get_pending_ids("temporal")
    if not pending:
        _log.info("All records already have temporal features.")
        return 0

    conn = get_db()
    try:
        placeholders = ",".join("?" for _ in pending)
        rows = conn.execute(
            f"SELECT memory_id, user_id, captured_at FROM memories WHERE memory_id IN ({placeholders})",
            pending,
        ).fetchall()

        # compute each user's earliest timestamp for relative_day
        user_first = {}
        all_user_rows = conn.execute(
            "SELECT user_id, MIN(captured_at) as first_ts FROM memories GROUP BY user_id"
        ).fetchall()
        for r in all_user_rows:
            dt = parse_timestamp(r["first_ts"])
            if dt:
                user_first[r["user_id"]] = dt

        _log.info("Computing temporal features for %d records...", len(rows))
        processed = 0

        for row in rows:
            mid = row["memory_id"]
            ts_str = row["captured_at"]
            dt = parse_timestamp(ts_str)

            if dt is None:
                mark_record_error(mid, "temporal", "Unparseable timestamp", conn)
                continue

            try:
                hour = dt.hour
                dow = dt.weekday()
                month = dt.month
                year = dt.year

                # cyclical encoding
                sin_hour = math.sin(2 * math.pi * hour / 24)
                cos_hour = math.cos(2 * math.pi * hour / 24)
                sin_dow = math.sin(2 * math.pi * dow / 7)
                cos_dow = math.cos(2 * math.pi * dow / 7)
                sin_month = math.sin(2 * math.pi * (month - 1) / 12)
                cos_month = math.cos(2 * math.pi * (month - 1) / 12)

                # relative day (days since user's first memory)
                # Strip timezone info before subtracting to avoid
                # "can't subtract offset-naive and offset-aware datetimes"
                first = user_first.get(row["user_id"])
                try:
                    if first:
                        dt_naive = dt.replace(tzinfo=None)
                        first_naive = first.replace(tzinfo=None)
                        relative_day = (dt_naive - first_naive).total_seconds() / 86400.0
                    else:
                        relative_day = 0.0
                except Exception:
                    relative_day = 0.0

                conn.execute(
                    """INSERT OR REPLACE INTO temporal_features
                       (memory_id, hour, day_of_week, month, year,
                        season, time_of_day,
                        sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month,
                        relative_day)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mid, hour, dow, month, year,
                        _season(month), _time_of_day(hour),
                        sin_hour, cos_hour, sin_dow, cos_dow, sin_month, cos_month,
                        relative_day,
                    ),
                )
                mark_record_done(mid, "temporal", conn)
                processed += 1

            except Exception as e:
                _log.warning("Temporal extraction failed for %s: %s", mid, e)
                mark_record_error(mid, "temporal", str(e), conn)

        conn.commit()
        _log.info("Temporal features complete: %d processed.", processed)
        return processed
    finally:
        conn.close()
