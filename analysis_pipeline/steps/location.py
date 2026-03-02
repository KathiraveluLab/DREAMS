"""
Step 6 — Location: Reverse-geocode GPS coordinates → place labels + metadata.

Key improvements:
- Response caching in a dedicated SQLite DB (never hits Nominatim twice for
  the same spot) 
- Extracts structured place metadata (place_type, road, city, state, country)
  rather than just embedding the raw address string
- Graceful rate-limiting with exponential back-off
"""

import asyncio
import logging
import time
import json

import aiohttp

from ..config import GEOCODE_RATE_LIMIT, GEOCODE_USER_AGENT, BATCH_SIZE
from ..db import (
    get_db, get_pending_ids,
    mark_record_done, mark_record_error,
    get_cached_geocode, set_cached_geocode,
)

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


async def _reverse_geocode_single(
    lat: float, lon: float, session: aiohttp.ClientSession, _log: logging.Logger
) -> dict | None:
    """Call Nominatim reverse-geocode with rate limiting and retry."""
    # check cache first
    cached = get_cached_geocode(lat, lon)
    if cached is not None:
        return cached

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
    }
    headers = {"User-Agent": GEOCODE_USER_AGENT}

    for attempt in range(3):
        try:
            await asyncio.sleep(GEOCODE_RATE_LIMIT)  # respect 1 req/s
            async with session.get(_NOMINATIM_URL, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # cache for future runs
                    set_cached_geocode(lat, lon, data)
                    return data
                elif resp.status == 429:
                    wait = (2 ** attempt) * 2
                    _log.warning("Nominatim rate-limited, waiting %ds", wait)
                    await asyncio.sleep(wait)
                else:
                    _log.warning("Nominatim returned %d for (%.4f, %.4f)", resp.status, lat, lon)
                    return None
        except Exception as e:
            _log.warning("Geocode attempt %d failed for (%.4f, %.4f): %s", attempt + 1, lat, lon, e)
            await asyncio.sleep(2 ** attempt)

    return None


def _extract_place_metadata(geo_data: dict) -> dict:
    """Pull structured fields from a Nominatim response."""
    address = geo_data.get("address", {})
    # determine place type from the 'category' or 'type' field
    place_type = geo_data.get("type", "")
    if not place_type:
        place_type = geo_data.get("category", "")

    return {
        "display_name": geo_data.get("display_name", ""),
        "place_type": place_type,
        "address_road": address.get("road", ""),
        "address_city": (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or ""
        ),
        "address_state": address.get("state", ""),
        "address_country": address.get("country", ""),
        "raw_response": json.dumps(geo_data),
    }


async def _process_batch(batch_rows: list, _log: logging.Logger, conn) -> int:
    """Geocode a batch of records asynchronously."""
    processed = 0
    async with aiohttp.ClientSession() as session:
        for row in batch_rows:
            mid = row["memory_id"]
            lat, lon = row["latitude"], row["longitude"]

            if lat is None or lon is None:
                mark_record_error(mid, "location", "Missing coordinates", conn)
                continue

            try:
                geo_data = await _reverse_geocode_single(lat, lon, session, _log)
                if geo_data is None:
                    mark_record_error(mid, "location", "Geocode returned no data", conn)
                    continue

                meta = _extract_place_metadata(geo_data)
                conn.execute(
                    """INSERT OR REPLACE INTO location_info
                       (memory_id, display_name, place_type,
                        address_road, address_city, address_state, address_country,
                        raw_response)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        meta["display_name"],
                        meta["place_type"],
                        meta["address_road"],
                        meta["address_city"],
                        meta["address_state"],
                        meta["address_country"],
                        meta["raw_response"],
                    ),
                )
                mark_record_done(mid, "location", conn)
                processed += 1

            except Exception as e:
                _log.warning("Location processing failed for %s: %s", mid, e)
                mark_record_error(mid, "location", str(e), conn)

    return processed


def run(log: logging.Logger | None = None) -> int:
    """Reverse-geocode all pending records with caching."""
    _log = log or logger

    pending = get_pending_ids("location")
    if not pending:
        _log.info("All records already have location info.")
        return 0

    conn = get_db()
    try:
        placeholders = ",".join("?" for _ in pending)
        rows = conn.execute(
            f"SELECT memory_id, latitude, longitude FROM memories WHERE memory_id IN ({placeholders})",
            pending,
        ).fetchall()

        _log.info("Geocoding %d records (cached lookups will be instant)...", len(rows))
        total_processed = 0

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            count = asyncio.run(_process_batch(list(batch), _log, conn))
            total_processed += count
            conn.commit()
            _log.info("  Geocoded %d / %d", min(i + BATCH_SIZE, len(rows)), len(rows))

        _log.info("Location processing complete: %d processed.", total_processed)
        return total_processed
    finally:
        conn.close()
