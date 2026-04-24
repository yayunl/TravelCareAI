"""Server-side Google Maps Geocoding + Places (Nearby + Details). Use a server/API key (not browser-referrer-only)."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

import httpx


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


async def geocode_address(address: str, api_key: str) -> tuple[float, float] | None:
    if not (address or "").strip() or not api_key:
        return None
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address.strip(), "key": api_key},
        )
        r.raise_for_status()
        data = r.json()
    status = data.get("status")
    if status != "OK":
        logger.warning("geocode Maps status=%s (not OK)", status)
        return None
    results = data.get("results") or []
    if not results:
        return None
    loc = results[0].get("geometry", {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


async def fetch_destination_local_context(
    latitude: float,
    longitude: float,
    api_key: str,
) -> dict[str, Any] | None:
    """
    Approximate local civil time at the trip coordinates via Google Time Zone API + zoneinfo.

    Returns a small dict for LLM/research payloads (not a substitute for facility opening hours).
    """
    if not (api_key or "").strip():
        return None
    ts = int(time.time())
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/timezone/json",
            params={
                "location": f"{latitude},{longitude}",
                "timestamp": ts,
                "key": api_key.strip(),
            },
        )
        r.raise_for_status()
        data = r.json()
    if data.get("status") != "OK":
        logger.warning("Google Time Zone API status=%s", data.get("status"))
        return None
    tz_id = str(data.get("timeZoneId") or "").strip()
    if not tz_id:
        return None
    try:
        zi = ZoneInfo(tz_id)
        now_local = datetime.now(zi)
    except Exception:
        logger.warning("Could not load ZoneInfo for %r", tz_id[:80])
        return None
    raw_off = int(data.get("rawOffset") or 0)
    dst_off = int(data.get("dstOffset") or 0)
    return {
        "timezone_id": tz_id,
        "timezone_name": (str(data.get("timeZoneName") or "").strip()[:120] or None),
        "local_datetime_iso": now_local.replace(microsecond=0).isoformat(),
        "local_weekday_en": now_local.strftime("%A"),
        "local_hour_24": now_local.hour,
        "utc_offset_seconds": raw_off + dst_off,
    }


async def nearby_medical_places(
    *,
    latitude: float,
    longitude: float,
    api_key: str,
    radius_meters: int = 5000,
    place_type: str = "hospital",
    keyword: str = "",
    max_results: int = 6,
    include_review_snippets: bool = True,
    max_detail_places: int = 3,
) -> dict[str, Any]:
    """Legacy Places Nearby Search + optional Details for short review snippets."""
    if not api_key:
        return {"error": "Google Maps server key not configured", "places": []}
    params: dict[str, Any] = {
        "location": f"{latitude},{longitude}",
        "radius": max(200, min(radius_meters, 50000)),
        "type": (place_type or "hospital").strip() or "hospital",
        "key": api_key,
    }
    kw = (keyword or "").strip()
    if kw:
        params["keyword"] = kw

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params=params,
        )
        r.raise_for_status()
        data = r.json()
    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        logger.warning("Places Nearby Search status=%s", status)
        return {"error": f"Places Nearby status: {status}", "places": [], "raw_status": status}

    out: list[dict[str, Any]] = []
    for p in (data.get("results") or [])[:max_results]:
        loc = p.get("geometry", {}).get("location") or {}
        plat, plng = loc.get("lat"), loc.get("lng")
        dist = None
        if plat is not None and plng is not None:
            dist = round(_haversine_m(latitude, longitude, float(plat), float(plng)))
        entry: dict[str, Any] = {
            "name": p.get("name"),
            "vicinity": p.get("vicinity"),
            "place_id": p.get("place_id"),
            "types": p.get("types") or [],
            "rating": p.get("rating"),
            "user_ratings_total": p.get("user_ratings_total"),
            "price_level": p.get("price_level"),
            "business_status": p.get("business_status"),
            "straight_line_distance_m": dist,
            "open_now": (p.get("opening_hours") or {}).get("open_now"),
        }
        out.append(entry)

    if include_review_snippets and api_key and out:
        async with httpx.AsyncClient(timeout=25.0) as dc:
            for entry in out[:max_detail_places]:
                pid = entry.get("place_id")
                if not pid:
                    continue
                details = await _place_details_reviews(client=dc, place_id=pid, api_key=api_key)
                if details:
                    entry["review_snippets"] = details

    return {"places": out, "origin": {"lat": latitude, "lng": longitude}}


async def _place_details_reviews(
    *,
    client: httpx.AsyncClient | None,
    place_id: str,
    api_key: str,
    max_reviews: int = 2,
    max_chars: int = 220,
) -> list[dict[str, Any]]:
    fields = "name,rating,user_ratings_total,reviews,price_level"
    params = {"place_id": place_id, "fields": fields, "key": api_key}
    close = False
    if client is None:
        client = httpx.AsyncClient(timeout=20.0)
        close = True
    try:
        r = await client.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params=params,
        )
        r.raise_for_status()
        data = r.json()
    finally:
        if close:
            await client.aclose()
    if data.get("status") != "OK":
        logger.debug("Place Details status=%s place_id=%s…", data.get("status"), (place_id or "")[:12])
        return []
    res = data.get("result") or {}
    snippets: list[dict[str, Any]] = []
    for rev in (res.get("reviews") or [])[:max_reviews]:
        text = (rev.get("text") or "")[:max_chars]
        snippets.append(
            {
                "rating": rev.get("rating"),
                "relative_time": rev.get("relative_time_description"),
                "text": text,
            }
        )
    return snippets
