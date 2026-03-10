"""Location extraction and semantic enrichment for DREAMS photo memories.

Pipeline:
    1. extract_gps_from_image()  — raw GPS + timestamp from EXIF
    2. reverse_geocode()         — coords → place metadata (OSM Nominatim)
    3. format_location_text()    — metadata → semantic text (no geography)
    4. get_location_embedding()  — text → 384-dim vector (all-MiniLM-L6-v2)
    5. enrich_location()         — orchestrates 2-4 in one call
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_USER_AGENT = (
    "DREAMS-Research/1.0 (https://github.com/KathiraveluLab/DREAMS)"
)
_MIN_REQUEST_INTERVAL = 1.1   # Nominatim policy: max 1 req/s
_CACHE_PRECISION = 5          # decimal places ≈ 1 m resolution
_MAX_CACHE_SIZE = 10_000      # evict oldest entry when exceeded
_CITY_KEYS = ("city", "town", "village", "hamlet")

# ── Module-level state ──────────────────────────────────────────────────

_geocode_cache: Dict[tuple, dict] = {}
_last_request_time: float = 0.0
_nominatim_lock = threading.Lock()
_embedding_model = None


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_embedding_model():
    """Lazily load the SentenceTransformer singleton."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-V2")
    return _embedding_model


def _dms_to_decimal(dms_value) -> float:
    """Convert EXIF GPS DMS (degrees/minutes/seconds) to decimal degrees."""
    if not isinstance(dms_value, (tuple, list)) or len(dms_value) != 3:
        raise ValueError(f"Expected 3-element DMS sequence, got: {dms_value}")

    decimal = 0.0
    for idx, component in enumerate(dms_value):
        if isinstance(component, tuple):
            if component[1] == 0:
                raise ValueError(f"Zero denominator in DMS: {component}")
            value = component[0] / component[1]
        else:
            value = float(component)
        decimal += value / (60 ** idx)
    return decimal


def _parse_gps_timestamp(gps_info: dict) -> Optional[str]:
    """Build ISO-8601 timestamp from GPSDateStamp + GPSTimeStamp, or None."""
    if "GPSDateStamp" not in gps_info or "GPSTimeStamp" not in gps_info:
        return None
    try:
        year, month, day = map(int, gps_info["GPSDateStamp"].split(":"))
        h, m, s_raw = (float(p) for p in gps_info["GPSTimeStamp"])
        s_int = int(s_raw)
        return datetime(
            year, month, day, int(h), int(m), s_int,
            int((s_raw - s_int) * 1_000_000), tzinfo=timezone.utc,
        ).isoformat()
    except (ValueError, TypeError, IndexError):
        logger.warning("Could not parse GPSDateStamp / GPSTimeStamp")
        return None


def _parse_exif_datetime(raw: str) -> Optional[str]:
    """Parse EXIF DateTimeOriginal to ISO-8601, or None."""
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").isoformat()
    except (ValueError, TypeError):
        logger.warning("Could not parse EXIF DateTimeOriginal: '%s'", raw)
        return None


# ── Public API ───────────────────────────────────────────────────────────

def extract_gps_from_image(image_path: str) -> Optional[Dict[str, Any]]:
    """Extract GPS coordinates and timestamp from EXIF metadata.

    Returns ``{"lat": float, "lon": float}`` with optional ``"timestamp"``,
    or ``None`` if no GPS data is available.
    """
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None

            gps_info = None
            datetime_original = None
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    gps_info = {GPSTAGS.get(t, t): value[t] for t in value}
                elif tag_name == "DateTimeOriginal":
                    datetime_original = value
                if gps_info is not None and datetime_original is not None:
                    break

            if not gps_info:
                return None
            if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
                return None

            lat = _dms_to_decimal(gps_info["GPSLatitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat

            lon = _dms_to_decimal(gps_info["GPSLongitude"])
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            result: Dict[str, Any] = {"lat": lat, "lon": lon}
            timestamp = _parse_gps_timestamp(gps_info)
            if not timestamp and datetime_original:
                timestamp = _parse_exif_datetime(datetime_original)
            if timestamp:
                result["timestamp"] = timestamp
            return result

    except (AttributeError, KeyError, IndexError, TypeError, ValueError, IOError) as exc:
        logger.error("Failed to extract GPS from '%s': %s", image_path, exc)
        return None


def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Reverse-geocode coordinates via OSM Nominatim.

    Thread-safe, rate-limited (1.1 s), with bounded in-memory cache.
    Returns dict with ``display_name``, ``place_category``, ``place_type``,
    ``address`` — or ``None`` on failure.
    """
    global _last_request_time

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        logger.warning("Invalid coordinates: lat=%s, lon=%s", lat, lon)
        return None

    with _nominatim_lock:
        cache_key = (round(lat, _CACHE_PRECISION), round(lon, _CACHE_PRECISION))
        if cache_key in _geocode_cache:
            # Move item to end to implement LRU
            value = _geocode_cache.pop(cache_key)
            _geocode_cache[cache_key] = value
            return value

        # Rate limit — inline to avoid extra function overhead inside lock
        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        try:
            response = requests.get(
                _NOMINATIM_URL,
                params={
                    "lat": lat, "lon": lon,
                    "format": "jsonv2", "addressdetails": 1,
                    "accept-language": "en",
                },
                headers={"User-Agent": _NOMINATIM_USER_AGENT},
                timeout=10,
            )
            _last_request_time = time.time()
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.warning("Nominatim error: %s", data["error"])
                return None

            addr = data.get("address", {})
            place_type = data.get("category", "")

            result = {
                "display_name": data.get("display_name", ""),
                "place_category": data.get("type", ""),
                "place_type": place_type,
                "address": {
                    "road": addr.get("road"),
                    "city": next((addr[k] for k in _CITY_KEYS if addr.get(k)), None),
                    "state": addr.get("state"),
                    "country": addr.get("country"),
                    "country_code": addr.get("country_code"),
                },
            }

            _geocode_cache[cache_key] = result
            if len(_geocode_cache) > _MAX_CACHE_SIZE:
                _geocode_cache.pop(next(iter(_geocode_cache)))
            return result

        except (requests.RequestException, ValueError, KeyError) as exc:
            _last_request_time = time.time()
            logger.error("Reverse geocoding failed for (%s, %s): %s", lat, lon, exc)
            return None


def format_location_text(
    geocode_result: Optional[Dict[str, Any]],
    lat: float,
    lon: float,
) -> str:
    """Build a semantic location string for embedding.

    Contains place-type semantics only (no city/state/country) so that
    two churches in different cities produce similar embeddings.
    """
    fallback = f"unknown location at coordinates {lat} {lon}"
    if geocode_result is None:
        return fallback

    category = geocode_result.get("place_category", "").replace("_", " ")
    place_type = geocode_result.get("place_type", "").replace("_", " ")

    parts: List[str] = []
    if category:
        parts.append(category)
    if place_type and place_type != category:
        parts.append(place_type)

    # Resolve place name: try address[place_type] key, else first segment
    # of display_name (e.g. address["amenity"] = "St. Mary's Church")
    pt_raw = geocode_result.get("place_type", "")
    addr = geocode_result.get("address", {})
    if pt_raw and addr.get(pt_raw):
        parts.append(addr[pt_raw])
    elif geocode_result.get("display_name"):
        parts.append(geocode_result["display_name"].split(",")[0].strip())

    return " ".join(parts) if parts else fallback


def get_location_embedding(location_text: str, model: Any = None) -> List[float]:
    """Encode location text into a 384-dim semantic vector."""
    if model is None:
        model = _get_embedding_model()
    return model.encode(location_text).tolist()


def enrich_location(
    lat: float, lon: float, model: Any = None,
) -> Dict[str, Any]:
    """Run the full pipeline: geocode → text → embedding.

    Always returns a dict (never None). Falls back gracefully if
    geocoding fails.
    """
    geocode = reverse_geocode(lat, lon)
    text = format_location_text(geocode, lat, lon)
    embedding = get_location_embedding(text, model=model)

    result: Dict[str, Any] = {
        "location_text": text,
        "location_embedding": embedding,
    }
    if geocode:
        for key in ("display_name", "place_category", "place_type", "address"):
            result[key] = geocode[key]
    return result
