import asyncio
import json
import time
import httpx
from typing import Any

NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"

# Client-side rate limiter: at most 1 request per second (Nominatim policy)
_last_request_time = 0.0
_rate_lock = asyncio.Lock()

_EMPTY = {"display_name": None, "address": None, "raw": None}
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


async def reverse_geocode(
    lat: float, lon: float, user_agent: str
) -> dict[str, Any]:
    """Reverse-geocode coordinates using OpenStreetMap Nominatim.

    Returns a dict with keys: display_name, address, raw.
    Includes rate limiting (1 req/s), exponential backoff on 429/5xx,
    and safe JSON parsing.
    """
    global _last_request_time

    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
    }
    headers = {"User-Agent": user_agent}

    for attempt in range(_MAX_RETRIES):
        # Rate limit: enforce >= 1 second between requests
        async with _rate_lock:
            elapsed = time.monotonic() - _last_request_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            _last_request_time = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{NOMINATIM_BASE_URL}/reverse", params=params, headers=headers
                )
        except httpx.RequestError:
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            return dict(_EMPTY)

        # Retry on 429 or 5xx with exponential backoff
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            return dict(_EMPTY)

        if response.status_code == 404:
            return dict(_EMPTY)

        # Safe JSON parse
        try:
            data = response.json()
            if not isinstance(data, dict):
                return dict(_EMPTY)
        except (json.JSONDecodeError, ValueError):
            return dict(_EMPTY)

        if data.get("error"):
            return dict(_EMPTY)

        return {
            "display_name": data.get("display_name"),
            "address": data.get("address", {}),
            "raw": data,
        }

    return dict(_EMPTY)
