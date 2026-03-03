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

# OSM place_types that are too generic or infrastructure-level to be
# meaningful for place-class inference.  When place_type is in this set,
# place_category is set to None rather than guessing the wrong class.
_MEANINGLESS_PLACE_TYPES: frozenset[str] = frozenset({
    "post_box", "postbox", "street", "road", "path", "footway",
    "cycleway", "service", "bench", "waste_basket", "bollard",
    "fire_hydrant", "manhole", "drain", "tree", "street_lamp",
    "telephone", "atm", "vending_machine", "parking", "parking_space",
    "building", "yes", "construction", "boundary", "administrative",
    "county", "state", "country", "neighbourhood", "suburb",
    "quarter", "city", "town", "village", "hamlet",
})

# Maps raw OSM place_type → broader DREAMS place_category.
# Categories are designed to align with the proximity framework:
# visual-scene grouping + emotional trajectory analysis per place-class.
_PLACE_CATEGORY_MAP: dict[str, str] = {
    # Faith / spiritual spaces
    "church": "faith_space",
    "cathedral": "faith_space",
    "chapel": "faith_space",
    "mosque": "faith_space",
    "temple": "faith_space",
    "synagogue": "faith_space",
    "place_of_worship": "faith_space",
    "shrine": "faith_space",
    "wayside_shrine": "faith_space",
    # Outdoor / nature
    "park": "outdoor_nature",
    "garden": "outdoor_nature",
    "nature_reserve": "outdoor_nature",
    "forest": "outdoor_nature",
    "beach": "outdoor_nature",
    "trail": "outdoor_nature",
    "recreation_ground": "outdoor_nature",
    "national_park": "outdoor_nature",
    "playground": "outdoor_nature",
    "grassland": "outdoor_nature",
    "wetland": "outdoor_nature",
    "water": "outdoor_nature",
    # Clinical / health
    "hospital": "clinical_setting",
    "clinic": "clinical_setting",
    "pharmacy": "clinical_setting",
    "doctors": "clinical_setting",
    "dentist": "clinical_setting",
    "health_centre": "clinical_setting",
    "nursing_home": "clinical_setting",
    "optician": "clinical_setting",
    "physiotherapist": "clinical_setting",
    "psychotherapist": "clinical_setting",
    # Support services / social care
    "shelter": "support_service",
    "social_facility": "support_service",
    "community_centre": "support_service",
    "social_centre": "support_service",
    "food_bank": "support_service",
    "refugee": "support_service",
    # Food / social
    "restaurant": "food_social",
    "cafe": "food_social",
    "fast_food": "food_social",
    "bar": "food_social",
    "pub": "food_social",
    "food_court": "food_social",
    "ice_cream": "food_social",
    "biergarten": "food_social",
    "bakery": "food_social",
    # Education
    "school": "education",
    "university": "education",
    "college": "education",
    "library": "education",
    "kindergarten": "education",
    "language_school": "education",
    "driving_school": "education",
    # Work / commercial
    "office": "work_space",
    "workplace": "work_space",
    "commercial": "work_space",
    "industrial": "work_space",
    # Fitness / sport
    "gym": "fitness_sport",
    "sports_centre": "fitness_sport",
    "swimming_pool": "fitness_sport",
    "stadium": "fitness_sport",
    "fitness_centre": "fitness_sport",
    "pitch": "fitness_sport",
    "track": "fitness_sport",
    "sports_hall": "fitness_sport",
    # Transit
    "bus_stop": "transit",
    "bus_station": "transit",
    "train_station": "transit",
    "subway_entrance": "transit",
    "ferry_terminal": "transit",
    "airport": "transit",
    "halt": "transit",
    "tram_stop": "transit",
    "taxi": "transit",
    # Retail / shopping
    "supermarket": "retail",
    "convenience": "retail",
    "department_store": "retail",
    "mall": "retail",
    "marketplace": "retail",
    "clothes": "retail",
    "electronics": "retail",
    "hardware": "retail",
    # Residential
    "house": "residential",
    "apartments": "residential",
    "dormitory": "residential",
    "detached": "residential",
}


def _get_place_category(place_type: str) -> str | None:
    """Map a raw OSM place_type to a DREAMS place_category.

    Returns None for meaningless/infrastructure types (e.g. 'post_box'),
    the mapped category string for known types, or None for unknown types
    (so downstream code can rely on the image vector instead).
    """
    if not place_type:
        return None
    pt = place_type.lower().strip()
    if pt in _MEANINGLESS_PLACE_TYPES:
        return None
    return _PLACE_CATEGORY_MAP.get(pt)  # None for unknown but non-junk types


def _nominatim_deep_parse(geo_data: dict) -> str | None:
    """Look beyond the top-level OSM type into the address sub-fields.

    Nominatim's address object can contain keys like 'amenity', 'leisure',
    'shop', 'tourism' whose *values* map directly to place categories.
    This handles the common case where the top-level type is 'parking'
    but the address has amenity='place_of_worship' because the GPS
    snapped to a parking lot adjacent to the actual building.

    Priority order searched: amenity > leisure > shop > tourism > building.
    """
    address = geo_data.get("address", {})
    for key in ("amenity", "leisure", "shop", "tourism", "building"):
        val = address.get(key, "")
        if val:
            cat = _get_place_category(val)
            if cat is not None:
                return cat
    # Also check the OSM 'name' field of the object — e.g. "St. Patrick's Church"
    # If the name contains a place keyword, use that.
    name = (geo_data.get("name") or "").lower()
    for keyword, category in _CAPTION_KEYWORD_MAP.items():
        if keyword in name:
            return category
    return None


# Maps plain-language keywords → place_category.
# Used for both caption text and OSM name field scanning.
_CAPTION_KEYWORD_MAP: dict[str, str] = {
    # Faith
    "church": "faith_space",
    "chapel": "faith_space",
    "cathedral": "faith_space",
    "mosque": "faith_space",
    "temple": "faith_space",
    "synagogue": "faith_space",
    "prayer": "faith_space",
    "worship": "faith_space",
    "mass ": "faith_space",       # trailing space avoids 'massive'
    "sermon": "faith_space",
    # Outdoor / nature
    "park": "outdoor_nature",
    "garden": "outdoor_nature",
    "trail": "outdoor_nature",
    "beach": "outdoor_nature",
    "forest": "outdoor_nature",
    "lake": "outdoor_nature",
    "river": "outdoor_nature",
    "mountain": "outdoor_nature",
    "nature": "outdoor_nature",
    "water": "outdoor_nature",
    "waterfront": "outdoor_nature",
    "riverside": "outdoor_nature",
    "lakeside": "outdoor_nature",
    "outdoor": "outdoor_nature",
    "outside": "outdoor_nature",
    "hiking": "outdoor_nature",
    "campsite": "outdoor_nature",
    # Clinical
    "hospital": "clinical_setting",
    "clinic": "clinical_setting",
    "pharmacy": "clinical_setting",
    "doctor": "clinical_setting",
    "therapy": "clinical_setting",
    "therapist": "clinical_setting",
    "counseling": "clinical_setting",
    "counselling": "clinical_setting",
    "treatment": "clinical_setting",
    "recovery center": "clinical_setting",
    "recovery centre": "clinical_setting",
    # Support services
    "shelter": "support_service",
    "food bank": "support_service",
    "community centre": "support_service",
    "community center": "support_service",
    "social service": "support_service",
    "halfway house": "support_service",
    # Food / social
    "restaurant": "food_social",
    "cafe": "food_social",
    "coffee": "food_social",
    "bar ": "food_social",         # trailing space avoids 'barely'
    "pub ": "food_social",
    "diner": "food_social",
    "bakery": "food_social",
    "pizza": "food_social",
    "dinner": "food_social",
    "lunch": "food_social",
    "eating": "food_social",
    # Education
    "school": "education",
    "university": "education",
    "college": "education",
    "library": "education",
    "classroom": "education",
    "campus": "education",
    # Fitness
    "gym": "fitness_sport",
    "workout": "fitness_sport",
    "fitness": "fitness_sport",
    "exercise": "fitness_sport",
    "swimming": "fitness_sport",
    "running": "fitness_sport",
    "sport": "fitness_sport",
    # Transit
    "bus stop": "transit",
    "train station": "transit",
    "airport": "transit",
    "bus station": "transit",
    # Retail
    "store": "retail",
    "shop": "retail",
    "supermarket": "retail",
    "mall": "retail",
    "market": "retail",
    # Residential
    "home": "residential",
    "house": "residential",
    "apartment": "residential",
    "flat ": "residential",        # trailing space avoids 'flat screen'
    "my room": "residential",
    "bedroom": "residential",
}


def _caption_place_category(caption: str) -> str | None:
    """Scan a caption for place-type keywords and return a place_category.

    This is the final fallback when reverse geocoding returns a meaningless
    type (e.g. 'parking' for a GPS point next to a church).  The user's own
    words are the most reliable signal — the mentor confirmed that text/caption
    is the primary data source in DREAMS.

    Searches longest phrases first to avoid 'bar' matching inside 'bakery'.
    Returns the first match found, or None.
    """
    if not caption:
        return None
    text = caption.lower()
    # Sort by keyword length descending so multi-word phrases are checked first
    for keyword in sorted(_CAPTION_KEYWORD_MAP, key=len, reverse=True):
        if keyword in text:
            return _CAPTION_KEYWORD_MAP[keyword]
    return None


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


def _extract_place_metadata(geo_data: dict, caption: str | None = None) -> dict:
    """Pull structured fields from a Nominatim response.

    place_category resolution uses three levels in priority order:
      1. Direct OSM type/category field  (e.g. type='church' → 'faith_space')
      2. Deep Nominatim address parse    (e.g. address.amenity='place_of_worship'
                                          when type='parking' due to GPS snap)
      3. Caption keyword scan            (e.g. 'at the church' → 'faith_space')
    This ensures GPS snap errors (parking next to church) are corrected.
    """
    address = geo_data.get("address", {})
    # determine place type from the 'category' or 'type' field
    place_type = geo_data.get("type", "")
    if not place_type:
        place_type = geo_data.get("category", "")

    # Level 1: direct OSM type
    place_category = _get_place_category(place_type)

    # Level 2: deep address parse (handles GPS snap to adjacent parking/road)
    if place_category is None:
        place_category = _nominatim_deep_parse(geo_data)

    # Level 3: caption keyword scan (user's own words are the most reliable signal)
    if place_category is None and caption:
        place_category = _caption_place_category(caption)

    return {
        "display_name": geo_data.get("display_name", ""),
        "place_type": place_type,
        "place_category": place_category,
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
            # Prefer user caption; fall back to generated caption for Level 3
            caption: str | None = row["caption"] or row["generated_caption"]

            if lat is None or lon is None:
                mark_record_error(mid, "location", "Missing coordinates", conn)
                continue

            try:
                geo_data = await _reverse_geocode_single(lat, lon, session, _log)
                if geo_data is None:
                    mark_record_error(mid, "location", "Geocode returned no data", conn)
                    continue

                meta = _extract_place_metadata(geo_data, caption=caption)
                conn.execute(
                    """INSERT OR REPLACE INTO location_info
                       (memory_id, display_name, place_type, place_category,
                        address_road, address_city, address_state, address_country,
                        raw_response)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        mid,
                        meta["display_name"],
                        meta["place_type"],
                        meta["place_category"],
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
            f"SELECT memory_id, latitude, longitude, caption, generated_caption "
            f"FROM memories WHERE memory_id IN ({placeholders})",
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

        # Always backfill any records (old or new) where place_category is still
        # null — this corrects existing records affected by GPS snap errors.
        backfilled = backfill_place_categories(conn, _log)
        if backfilled:
            _log.info("Backfilled place_category for %d existing record(s).", backfilled)

        return total_processed
    finally:
        conn.close()


def backfill_place_categories(
    conn=None, log: logging.Logger | None = None
) -> int:
    """Fix place_category for already-processed records where it is NULL.

    Applies the three-level resolution (deep Nominatim parse on stored
    raw_response, then caption keyword scan) without hitting the geocoding
    API again.  Safe to call repeatedly — only touches rows with NULL
    place_category.

    Returns the number of records updated.
    """
    _log = log or logger
    close = conn is None
    if conn is None:
        conn = get_db()

    try:
        rows = conn.execute(
            """SELECT li.memory_id, li.place_type, li.raw_response,
                      m.caption, m.generated_caption
               FROM location_info li
               JOIN memories m ON m.memory_id = li.memory_id
               WHERE li.place_category IS NULL""",
        ).fetchall()

        updated = 0
        for row in rows:
            caption = row["caption"] or row["generated_caption"]

            # Level 1: direct OSM type (catches records processed before the fix)
            cat = _get_place_category(row["place_type"] or "")

            # Level 2: deep Nominatim address parse from stored raw_response
            if cat is None and row["raw_response"]:
                try:
                    geo_data = json.loads(row["raw_response"])
                    cat = _nominatim_deep_parse(geo_data)
                except Exception:
                    pass

            # Level 3: caption keyword scan
            if cat is None and caption:
                cat = _caption_place_category(caption)

            if cat is not None:
                conn.execute(
                    "UPDATE location_info SET place_category = ? WHERE memory_id = ?",
                    (cat, row["memory_id"]),
                )
                _log.debug(
                    "Backfilled place_category=%r for memory %s (was null)",
                    cat, row["memory_id"],
                )
                updated += 1

        if updated:
            conn.commit()
        return updated
    finally:
        if close:
            conn.close()
