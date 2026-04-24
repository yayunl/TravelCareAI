from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from app import places_client, web_search
from app.llm import openai_chat_temperature_for_model, raise_for_openai_response

logger = logging.getLogger(__name__)

_PLACES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_nearby_medical_places",
        "description": (
            "Find nearby medical facilities using Google Places (ratings, review snippets, "
            "approximate straight-line distance in meters from the search origin). "
            "Does not provide reliable medical prices—use web_search for cost ballparks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Search origin latitude"},
                "longitude": {"type": "number", "description": "Search origin longitude"},
                "radius_meters": {
                    "type": "integer",
                    "description": "Search radius 200–50000",
                    "default": 4500,
                },
                "place_type": {
                    "type": "string",
                    "description": "Places type, e.g. hospital, doctor, pharmacy, dentist",
                    "default": "hospital",
                },
                "keyword": {
                    "type": "string",
                    "description": "Extra keyword e.g. emergency, urgent care, 24 hour",
                    "default": "",
                },
                "include_review_snippets": {
                    "type": "boolean",
                    "default": True,
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
}

_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web. Prioritize official local or regional government and public-health sources "
            "when travel_location is known. Also usable for typical costs or clinic types, but treat costs as uncertain. "
            "When the server uses SerpAPI (SERPAPI_API_KEY), you may pass optional SerpAPI parameters below to bias "
            "locale and geography (see SerpAPI Google Search API docs). Do not combine uule with location or lat/lon. "
            "If you omit them, the server may fill lat/lon from the map pin when available (preferred for SerpAPI), "
            "else a location string from travel_location (SerpAPI accepts only certain canonical location names)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused English search query"},
                "location": {
                    "type": "string",
                    "description": "SerpAPI only: plain-text origin (e.g. city, country) to bias results.",
                },
                "gl": {"type": "string", "description": "SerpAPI only: two-letter country code (e.g. us, fr)."},
                "hl": {"type": "string", "description": "SerpAPI only: interface language code (e.g. en, es)."},
                "google_domain": {
                    "type": "string",
                    "description": "SerpAPI only: Google domain host (default google.com).",
                },
                "lat": {"type": "number", "description": "SerpAPI only: latitude; must be paired with lon."},
                "lon": {"type": "number", "description": "SerpAPI only: longitude; must be paired with lat."},
                "radius": {
                    "type": "integer",
                    "description": "SerpAPI only: bias radius in meters (1–1000 per SerpAPI limits).",
                },
                "start": {
                    "type": "integer",
                    "description": "SerpAPI only: pagination offset (e.g. 0, 10, 20).",
                },
                "safe": {
                    "type": "string",
                    "description": "SerpAPI only: active or off for SafeSearch-style filtering.",
                },
                "tbm": {
                    "type": "string",
                    "description": "SerpAPI only: vertical (e.g. nws news, isch images); default web search if omitted.",
                },
                "device": {
                    "type": "string",
                    "description": "SerpAPI only: desktop, tablet, or mobile.",
                },
                "cr": {"type": "string", "description": "SerpAPI only: country restrict string per SerpAPI docs."},
                "lr": {"type": "string", "description": "SerpAPI only: language restrict string per SerpAPI docs."},
                "tbs": {"type": "string", "description": "SerpAPI only: advanced tbs parameter string."},
                "nfpr": {"type": "integer", "description": "SerpAPI only: 1 to exclude auto-corrected query results."},
                "filter": {"type": "integer", "description": "SerpAPI only: 0 to disable similar/omitted filters."},
                "uule": {"type": "string", "description": "SerpAPI only: Google-encoded location; not with location/lat."},
                "no_cache": {"type": "boolean", "description": "SerpAPI only: force fresh fetch when true."},
            },
            "required": ["query"],
        },
    },
}


def _build_tools(*, maps_key: str, serper_key: str, serpapi_key: str, has_coords: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if serper_key.strip() or serpapi_key.strip():
        tools.append(_WEB_TOOL)
    if maps_key.strip() and has_coords:
        tools.append(_PLACES_TOOL)
    return tools


async def _dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    maps_key: str,
    serper_key: str,
    serpapi_key: str,
    travel_location: str | None,
    default_lat: float | None,
    default_lng: float | None,
) -> dict[str, Any]:
    if name == "search_nearby_medical_places":
        lat = arguments.get("latitude", default_lat)
        lng = arguments.get("longitude", default_lng)
        try:
            lat_f = float(lat) if lat is not None else None
            lng_f = float(lng) if lng is not None else None
        except (TypeError, ValueError):
            return {"error": "invalid_lat_lng"}
        if lat_f is None or lng_f is None:
            return {
                "error": (
                    "latitude and longitude required. Send map_latitude/map_longitude from the client "
                    "or ensure travel_location geocodes on the server."
                ),
            }
        if not maps_key.strip():
            return {"error": "GOOGLE_MAPS_SERVER_KEY not configured"}
        return await places_client.nearby_medical_places(
            latitude=lat_f,
            longitude=lng_f,
            api_key=maps_key,
            radius_meters=int(arguments.get("radius_meters") or 4500),
            place_type=str(arguments.get("place_type") or "hospital"),
            keyword=str(arguments.get("keyword") or ""),
            include_review_snippets=bool(arguments.get("include_review_snippets", True)),
        )

    if name == "web_search":
        q = str(arguments.get("query") or "").strip()
        if not q:
            return {"error": "empty_query"}
        if not serper_key.strip() and not serpapi_key.strip():
            return {"error": "SERPER_API_KEY or SERPAPI_API_KEY not configured"}
        wa = dict(arguments)
        tl = (travel_location or "").strip()
        # SerpAPI `location` must match supported canonical names; free text like "Austin, TX, USA" often 400s.
        # When the assist request has map coordinates, prefer lat/lon for geo bias (same as SerpAPI docs).
        if (
            default_lat is not None
            and default_lng is not None
            and "location" not in wa
            and "lat" not in wa
            and "uule" not in wa
        ):
            wa["lat"] = default_lat
            wa["lon"] = default_lng
        elif tl and "location" not in wa and "lat" not in wa and "uule" not in wa:
            wa["location"] = tl[:240]
        opts = web_search.sanitize_serpapi_options(wa)
        return await web_search.run_google_web_search(
            q,
            serper_key=serper_key,
            serpapi_key=serpapi_key,
            serpapi_options=opts,
        )

    return {"error": f"unknown_tool:{name}"}


def _trace_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Bounded tool arguments for API/UI activity logs (no secrets)."""
    if name == "web_search":
        out: dict[str, Any] = {"query": str(arguments.get("query") or "").strip()[:500]}
        for k in sorted(web_search.SERPAPI_OPTIONAL_PARAM_KEYS):
            if k not in arguments:
                continue
            v = arguments[k]
            if v is None or v == "":
                continue
            if isinstance(v, (int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v).strip()[:200]
        return out
    if name == "search_nearby_medical_places":
        return {
            "place_type": str(arguments.get("place_type") or "")[:80],
            "keyword": str(arguments.get("keyword") or "")[:120],
            "radius_meters": arguments.get("radius_meters"),
            "include_review_snippets": arguments.get("include_review_snippets"),
        }
    out: dict[str, Any] = {}
    for k, v in list(arguments.items())[:14]:
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, int, float)):
            out[k] = v
        else:
            out[k] = str(v)[:240]
    return out


def _activity_transparency_char_limit() -> int:
    return int(os.getenv("LLM_ACTIVITY_TRUNCATE_CHARS", "0") or "0")


def _clip_for_activity_log(s: str) -> str:
    lim = _activity_transparency_char_limit()
    if lim <= 0 or len(s) <= lim:
        return s
    return s[:lim] + "\n… [truncated: LLM_ACTIVITY_TRUNCATE_CHARS]"


def _simplify_tool_calls_for_activity_log(tcalls: Any) -> list[dict[str, Any]]:
    if not isinstance(tcalls, list):
        return []
    simplified: list[dict[str, Any]] = []
    for x in tcalls:
        if not isinstance(x, dict):
            continue
        fn = x.get("function") or {}
        args_raw = fn.get("arguments") or "{}"
        args_s = _clip_for_activity_log(str(args_raw))
        simplified.append(
            {
                "id": str(x.get("id") or "")[:80],
                "name": str(fn.get("name") or ""),
                "arguments": args_s,
            }
        )
    return simplified


def _openai_messages_for_activity_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-safe copy of chat messages for UI (full content unless LLM_ACTIVITY_TRUNCATE_CHARS is set)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        entry: dict[str, Any] = {"role": role}
        content = m.get("content")
        if content is None:
            entry["content"] = None
        elif isinstance(content, str):
            entry["content"] = _clip_for_activity_log(content)
        else:
            entry["content"] = _clip_for_activity_log(str(content))
        if role == "assistant" and m.get("tool_calls"):
            entry["tool_calls"] = _simplify_tool_calls_for_activity_log(m.get("tool_calls"))
        if role == "tool" and m.get("tool_call_id"):
            entry["tool_call_id"] = str(m.get("tool_call_id"))[:80]
        out.append(entry)
    return out


def _research_round_reasoning_summary(msg: dict[str, Any], batch: list[dict[str, Any]]) -> str:
    """Human-readable plan for the activity UI: assistant prose when present + explicit tool intents."""
    lines: list[str] = []
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        lines.append(content.strip())
    if batch:
        if lines:
            lines.append("")
        lines.append("Tool requests this round:")
        for b in batch:
            name = str(b.get("name") or "tool").strip()
            args = b.get("arguments") if isinstance(b.get("arguments"), dict) else {}
            if name == "web_search":
                q = str(args.get("query") or "").strip()
                lines.append(f"- web_search: {q}")
            elif name == "search_nearby_medical_places":
                pt = str(args.get("place_type") or "")
                kw = str(args.get("keyword") or "")
                rad = args.get("radius_meters")
                snip = args.get("include_review_snippets")
                bits: list[str] = []
                if pt:
                    bits.append(f"place_type={pt}")
                if kw:
                    bits.append(f"keyword={kw}")
                if rad is not None:
                    bits.append(f"radius_m={rad}")
                if snip is not None:
                    bits.append(f"include_review_snippets={snip}")
                lines.append("- search_nearby_medical_places: " + (", ".join(bits) if bits else "(defaults)"))
            else:
                try:
                    blob = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    blob = str(args)
                lines.append(f"- {name}: {blob}")
    if not lines:
        return (
            "No assistant-visible message text this round — the model went straight to tool calls. "
            "See the tool list below and the raw assistant turn for parameters."
        )
    return _clip_for_activity_log("\n".join(lines))


def _openai_assistant_response_for_activity_log(msg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": str(msg.get("role") or "assistant")}
    content = msg.get("content")
    if content is not None and content != "":
        c = str(content)
        out["content"] = _clip_for_activity_log(c)
    tc = msg.get("tool_calls") or []
    if isinstance(tc, list) and tc:
        out["tool_calls"] = _simplify_tool_calls_for_activity_log(tc)
    return out


def _summarize_tool_result_for_trace(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"note": "non-object tool result"}
    err = result.get("error")
    if err:
        return {"error": str(err)[:500]}
    if name == "web_search":
        results = result.get("results") or []
        if not isinstance(results, list):
            results = []
        titles: list[str] = []
        for r in results[:6]:
            if isinstance(r, dict) and r.get("title"):
                titles.append(str(r["title"]).strip()[:160])
        out: dict[str, Any] = {
            "result_kind": "web_search",
            "result_count": len(results),
            "top_titles": titles,
            "query_echo": str(result.get("query") or "")[:400],
        }
        prov = result.get("provider")
        if prov:
            out["provider"] = str(prov)[:32]
        spu = result.get("serpapi_params_used")
        if isinstance(spu, dict) and spu:
            out["serpapi_params_used"] = dict(list(spu.items())[:16])
        return out
    if name == "search_nearby_medical_places":
        places = result.get("places") or []
        if not isinstance(places, list):
            places = []
        names = [str((p or {}).get("name") or "").strip()[:120] for p in places[:8] if p]
        return {
            "result_kind": "places",
            "place_count": len(places),
            "sample_names": [n for n in names if n],
        }
    return {"keys": list(result.keys())[:14]}


def _safe_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _extract_review_snippets(place: dict[str, Any]) -> list[str]:
    snippets: list[str] = []

    raw = place.get("review_snippets")
    if isinstance(raw, list):
        for item in raw[:2]:
            if item is None:
                continue
            if isinstance(item, dict):
                s = str(item.get("text") or item.get("snippet") or "").strip()
            else:
                s = str(item).strip()
            if s:
                snippets.append(s[:300])

    if snippets:
        return snippets

    # fallback shapes if places_client returns differently
    for key in ("reviews", "editorial_summary", "summary"):
        value = place.get(key)
        if isinstance(value, list):
            for item in value[:2]:
                if isinstance(item, dict):
                    s = str(item.get("text") or item.get("snippet") or "").strip()
                else:
                    s = str(item).strip()
                if s:
                    snippets.append(s[:300])
        elif isinstance(value, str) and value.strip():
            snippets.append(value.strip()[:300])

    return snippets[:2]


def _normalize_place(place: dict[str, Any], default_type: str = "") -> dict[str, Any]:
    name = str(place.get("name") or "").strip()
    address = str(
        place.get("address")
        or place.get("formatted_address")
        or place.get("vicinity")
        or ""
    ).strip()

    maps_url = str(
        place.get("maps_url")
        or place.get("google_maps_url")
        or place.get("url")
        or ""
    ).strip() or None

    distance_m = _safe_int(
        place.get("straight_line_distance_m")
        or place.get("distance_m")
        or place.get("straight_line_distance")
    )

    rating = _safe_float(place.get("rating"))
    review_count = _safe_int(
        place.get("user_ratings_total")
        or place.get("review_count")
        or place.get("ratings_count")
    )

    open_now_raw = place.get("open_now")
    open_now = open_now_raw if isinstance(open_now_raw, bool) else None

    ptype = str(
        place.get("primary_type")
        or place.get("type")
        or default_type
        or ""
    ).strip()

    return {
        "name": name,
        "type": ptype,
        "address": address or None,
        "distance_m": distance_m,
        "rating": rating,
        "review_count": review_count,
        "open_now": open_now,
        "review_snippets": _extract_review_snippets(place),
        "maps_url": maps_url,
    }


def _extract_places_from_result(result: dict[str, Any], fallback_type: str = "") -> list[dict[str, Any]]:
    # Support several possible shapes from places_client.nearby_medical_places
    candidates: list[dict[str, Any]] = []

    for key in ("places", "results", "items"):
        raw = result.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    norm = _normalize_place(item, default_type=fallback_type)
                    if norm["name"]:
                        candidates.append(norm)

    # If tool returned a single place-like object
    if not candidates and isinstance(result, dict) and result.get("name"):
        norm = _normalize_place(result, default_type=fallback_type)
        if norm["name"]:
            candidates.append(norm)

    return candidates


def _dedupe_and_rank_places(places: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []

    for p in places:
        key = (
            (p.get("name") or "").strip().lower(),
            (p.get("address") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    def sort_key(p: dict[str, Any]) -> tuple[float, float]:
        dist = p.get("distance_m")
        rating = p.get("rating")
        dist_val = float(dist) if isinstance(dist, (int, float)) else 10**9
        rating_val = float(rating) if isinstance(rating, (int, float)) else -1.0
        return (dist_val, -rating_val)

    deduped.sort(key=sort_key)
    return deduped[:limit]


def _build_digest_text(structured: dict[str, Any]) -> str:
    lines: list[str] = []

    official = structured.get("official_local_system_summary") or []
    if official:
        lines.append("Official/local-system orientation:")
        for item in official[:4]:
            lines.append(f"- {item}")

    cost = structured.get("cost_signals") or []
    if cost:
        lines.append("Cost signals (uncertain):")
        for row in cost[:4]:
            title = str(row.get("source_title") or "Source").strip()
            note = str(row.get("note") or "").strip()
            low = row.get("low_amount")
            high = row.get("high_amount")
            currency = str(row.get("currency") or "").strip()
            amt = ""
            if low is not None or high is not None:
                amt = f" ({currency} {low}–{high})" if currency else f" ({low}–{high})"
            lines.append(f"- {title}{amt}: {note}")

    facilities = structured.get("nearby_facilities") or []
    if facilities:
        lines.append("Nearby facilities:")
        for f in facilities[:5]:
            bits = [str(f.get("name") or "").strip()]
            if f.get("distance_m") is not None:
                bits.append(f'{f["distance_m"]} m')
            if f.get("rating") is not None:
                rc = f.get("review_count")
                if rc is not None:
                    bits.append(f'rating {f["rating"]} ({rc} reviews)')
                else:
                    bits.append(f'rating {f["rating"]}')
            snippet_bits = f.get("review_snippets") or []
            joined = " | ".join(str(x) for x in snippet_bits[:2] if x)
            line = " — ".join(x for x in bits if x)
            if joined:
                line += f": {joined}"
            lines.append(f"- {line}")

    other = structured.get("other_web_findings") or []
    if other:
        lines.append("Other web findings:")
        for row in other[:4]:
            title = str(row.get("title") or "").strip()
            snippet = str(row.get("snippet") or "").strip()
            url = str(row.get("url") or "").strip()
            suffix = f" ({url})" if url else ""
            lines.append(f"- {title}{suffix}: {snippet}")

    notes = structured.get("research_notes") or []
    if notes:
        lines.append("Research notes:")
        for item in notes[:3]:
            lines.append(f"- {item}")

    out = "\n".join(lines).strip()
    return out[:10000]


async def run_research_tool_loop(
    *,
    combined_user_text: str,
    travel_location: str | None,
    home_country: str | None,
    latitude: float | None,
    longitude: float | None,
    language: str,
    openai_key: str,
    maps_server_key: str,
    serper_key: str,
    serpapi_key: str = "",
    destination_local_context: dict[str, Any] | None = None,
    model: str | None = None,
    max_rounds: int = 6,
    max_tool_calls: int = 10,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """
    Returns (structured_research, number_of_tool_messages_appended, activity_trace_for_ui).

    structured_research schema:
    {
      "digest_text": str,
      "official_local_system_summary": list[str],
      "cost_signals": list[dict],
      "nearby_facilities": list[dict],
      "other_web_findings": list[dict],
      "research_notes": list[str],
    }
    """
    has_coords = latitude is not None and longitude is not None
    loc_set = bool((travel_location or "").strip())
    tools = _build_tools(
        maps_key=maps_server_key,
        serper_key=serper_key,
        serpapi_key=serpapi_key,
        has_coords=has_coords,
    )
    if not tools:
        logger.info(
            "run_research_tool_loop: no tools (serper=%s serpapi=%s maps=%s coords=%s)",
            bool(serper_key.strip()),
            bool(serpapi_key.strip()),
            bool(maps_server_key.strip()),
            has_coords,
        )
        skip_trace: list[dict[str, Any]] = [
            {
                "phase": "research",
                "kind": "skipped",
                "message": (
                    "Research tools were not available (set SERPER_API_KEY and/or SERPAPI_API_KEY for web search, "
                    "and/or Google Maps server key with map coordinates for nearby Places)."
                ),
            }
        ]
        return (
            {
                "digest_text": "",
                "official_local_system_summary": [],
                "cost_signals": [],
                "nearby_facilities": [],
                "other_web_findings": [],
                "research_notes": ["No tools configured."],
            },
            0,
            skip_trace,
        )

    model_name = (model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
    logger.info(
        "run_research_tool_loop start tools=%d has_coords=%s model=%s",
        len(tools),
        has_coords,
        model_name,
    )

    tool_names = [str((t.get("function") or {}).get("name") or "") for t in tools]
    activity_trace: list[dict[str, Any]] = [
        {
            "phase": "research",
            "kind": "started",
            "model": model_name,
            "tools_available": [n for n in tool_names if n],
            "travel_location_set": loc_set,
            "coordinates_available": has_coords,
        }
    ]

    sys = """You are a research sub-agent for travel healthcare orientation (educational, not clinical diagnosis).

You may call:
- search_nearby_medical_places when coordinates are available: use returned ratings, review snippets, and straight_line_distance_m (meters from origin; NOT driving time). Google price_level is mostly for dining—do not treat it as medical cost.
- web_search for official government / public-health pages first, then other reputable pages. Always treat costs from the web as uncertain and unverified.

Research goals:
1. Understand how healthcare works in the travel location.
2. Collect official/local-system orientation when possible.
3. Collect ballpark cost signals when possible.
4. If coordinates exist and in-person care may be relevant, collect nearby facilities with concrete details.

Mandatory behavior:
- If travel_location is non-empty, run at least one web_search targeting official government/public-health sources.
- If travel_location is non-empty, run at least one web_search targeting typical self-pay/uninsured cost ballparks.
- If coordinates are available, run at least one nearby medical place search.
- Prefer separate place searches that match likely care settings, such as hospital, doctor/clinic, urgent care keywords, or pharmacy, when useful.

When you have enough information, STOP calling tools and return ONLY valid JSON with this schema:

{
  "official_local_system_summary": ["string"],
  "cost_signals": [
    {
      "source_title": "string",
      "source_url": "string|null",
      "note": "string",
      "low_amount": "number|null",
      "high_amount": "number|null",
      "currency": "string|null"
    }
  ],
  "nearby_facilities": [
    {
      "name": "string",
      "type": "string",
      "address": "string|null",
      "distance_m": "number|null",
      "rating": "number|null",
      "review_count": "number|null",
      "open_now": "boolean|null",
      "review_snippets": ["string"],
      "maps_url": "string|null",
      "why_it_may_help": "string"
    }
  ],
  "other_web_findings": [
    {
      "title": "string",
      "url": "string|null",
      "snippet": "string"
    }
  ],
  "research_notes": ["string"]
}

Rules:
- Do not invent facts, URLs, addresses, hours, ratings, reviews, or prices.
- When the user JSON includes destination_local_context, use local_datetime_iso (and timezone_id) as approximate local civil time at the map coordinates to reason qualitatively about access patterns (e.g. late night vs daytime). Still do not invent facility opening hours or schedules unless they appear in tool results.
- nearby_facilities should include up to 5 strong candidates when Places results exist.
- Preserve concrete details from tool results.
- Keep uncertainty explicit.
"""

    user_intro = {
        "traveler_case_excerpt": combined_user_text[:4000],
        "travel_location": travel_location,
        "traveler_home_country": home_country,
        "resolved_coordinates": {"latitude": latitude, "longitude": longitude} if has_coords else None,
        "output_language_for_final_app": language,
        "official_local_government_health_research": loc_set,
        "instructions": (
            "Call tools as needed, then output only valid JSON."
            if loc_set
            else "Use reputable general orientation and output only valid JSON."
        ),
    }
    if destination_local_context:
        user_intro["destination_local_context"] = destination_local_context

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_intro, ensure_ascii=False)},
    ]

    tool_calls_used = 0
    rounds = 0

    collected_places: list[dict[str, Any]] = []
    collected_web_results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        while rounds < max_rounds:
            rounds += 1
            payload: dict[str, Any] = {
                "model": model_name,
                "temperature": openai_chat_temperature_for_model(model_name, 0.2),
                "response_format": {"type": "json_object"},
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json=payload,
            )
            raise_for_openai_response(r, model_id=model_name, context=f"research_tool_loop_round_{rounds}")
            data = r.json()
            msg = data["choices"][0]["message"]
            tcalls = msg.get("tool_calls") or []
            batch: list[dict[str, Any]] = []
            if tcalls:
                for tc in tcalls:
                    fn = tc.get("function") or {}
                    n = str(fn.get("name") or "").strip()
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    batch.append(
                        {
                            "tool_call_id": str(tc.get("id") or "")[:48],
                            "name": n,
                            "arguments": _trace_tool_arguments(n, args),
                        }
                    )
            ex_entry: dict[str, Any] = {
                "phase": "research",
                "round": rounds,
                "kind": "llm_exchange",
                "model": model_name,
                "reasoning_summary": _research_round_reasoning_summary(msg, batch),
                "request_messages": _openai_messages_for_activity_log(messages),
                "response": _openai_assistant_response_for_activity_log(msg),
            }
            if batch:
                ex_entry["calls"] = batch
            activity_trace.append(ex_entry)

            messages.append(msg)

            if not tcalls:
                raw_content = (msg.get("content") or "").strip()
                try:
                    structured = json.loads(raw_content) if raw_content else {}
                except json.JSONDecodeError:
                    structured = {}

                if not isinstance(structured, dict):
                    structured = {}

                existing_facilities = structured.get("nearby_facilities")
                if not isinstance(existing_facilities, list):
                    existing_facilities = []

                merged = _dedupe_and_rank_places(
                    [p for p in existing_facilities if isinstance(p, dict)] + collected_places,
                    limit=5,
                )

                structured["nearby_facilities"] = merged
                structured.setdefault("official_local_system_summary", [])
                structured.setdefault("cost_signals", [])
                structured.setdefault("other_web_findings", [])
                structured.setdefault("research_notes", [])

                structured["digest_text"] = _build_digest_text(structured)
                activity_trace.append(
                    {
                        "phase": "research",
                        "round": rounds,
                        "kind": "model_completed",
                        "model": model_name,
                        "message": (
                            "Research model returned structured JSON (no further tool calls in this round). "
                            "A digest was built for the main traveler-facing assistant."
                        ),
                        "json_response_chars": len(raw_content),
                        "assistant_raw_json": _clip_for_activity_log(raw_content),
                    }
                )
                logger.info(
                    "run_research_tool_loop done rounds=%d tool_calls=%d nearby_n=%d",
                    rounds,
                    tool_calls_used,
                    len(structured.get("nearby_facilities") or []),
                )
                return structured, tool_calls_used, activity_trace

            for tc in tcalls:
                if tool_calls_used >= max_tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps({"error": "tool_call_budget_exceeded"}),
                        }
                    )
                    continue

                tool_calls_used += 1
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}

                logger.debug("research tool_call name=%s round=%d", name, rounds)

                result = await _dispatch(
                    name,
                    args,
                    maps_key=maps_server_key,
                    serper_key=serper_key,
                    serpapi_key=serpapi_key,
                    travel_location=travel_location,
                    default_lat=latitude,
                    default_lng=longitude,
                )

                if name == "search_nearby_medical_places" and isinstance(result, dict):
                    fallback_type = str(args.get("place_type") or "").strip()
                    collected_places.extend(_extract_places_from_result(result, fallback_type=fallback_type))

                if name == "web_search" and isinstance(result, dict):
                    collected_web_results.append(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _clip_for_activity_log(json.dumps(result, ensure_ascii=False)),
                    }
                )

    fallback = {
        "official_local_system_summary": [],
        "cost_signals": [],
        "nearby_facilities": _dedupe_and_rank_places(collected_places, limit=5),
        "other_web_findings": [],
        "research_notes": [
            "Research stopped at the step limit.",
            "Use map pin and official local sources to verify facilities and costs.",
        ],
    }
    fallback["digest_text"] = _build_digest_text(fallback)
    activity_trace.append(
        {
            "phase": "research",
            "kind": "limit_reached",
            "message": (
                f"Research stopped after {max_rounds} model rounds (safety cap). "
                "Partial tool data was merged into the digest when possible."
            ),
            "tool_calls_executed": tool_calls_used,
        }
    )
    logger.warning(
        "run_research_tool_loop hit max_rounds=%s tool_calls=%s",
        max_rounds,
        tool_calls_used,
    )
    return fallback, tool_calls_used, activity_trace