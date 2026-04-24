"""Web search: Serper.dev and/or SerpAPI.com (Google results JSON over HTTP)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Optional SerpAPI Google Search parameters (allowlist only — see https://serpapi.com/search-api).
SERPAPI_OPTIONAL_PARAM_KEYS = frozenset(
    {
        "location",
        "gl",
        "hl",
        "google_domain",
        "lat",
        "lon",
        "radius",
        "start",
        "safe",
        "tbm",
        "device",
        "cr",
        "lr",
        "tbs",
        "nfpr",
        "filter",
        "uule",
        "no_cache",
    }
)


def sanitize_serpapi_options(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Pick and normalize allowlisted SerpAPI query params from tool arguments (ignored for Serper)."""
    if not isinstance(raw, dict) or not raw:
        return {}
    out: dict[str, Any] = {}
    for key in SERPAPI_OPTIONAL_PARAM_KEYS:
        if key not in raw:
            continue
        val = raw[key]
        if val is None or val == "":
            continue
        if key in ("location", "google_domain", "uule", "tbs", "cr", "lr", "tbm", "device", "safe"):
            s = str(val).strip()
            if not s:
                continue
            limits = {
                "location": 240,
                "google_domain": 64,
                "uule": 500,
                "tbs": 200,
                "cr": 120,
                "lr": 120,
                "tbm": 24,
                "device": 16,
                "safe": 16,
            }
            out[key] = s[: limits.get(key, 200)]
        elif key in ("gl", "hl"):
            s = str(val).strip().lower()[:12]
            if s:
                out[key] = s
        elif key in ("lat", "lon"):
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                continue
        elif key == "radius":
            try:
                ri = int(float(val))
                out[key] = max(1, min(ri, 1000))
            except (TypeError, ValueError):
                continue
        elif key == "start":
            try:
                si = int(val)
                if si >= 0:
                    out[key] = min(si, 100)
            except (TypeError, ValueError):
                continue
        elif key in ("nfpr", "filter"):
            out[key] = 1 if str(val).strip().lower() in ("1", "true", "yes", "on") else 0
        elif key == "no_cache":
            out[key] = bool(val)

    lat, lon = out.get("lat"), out.get("lon")
    if (lat is None) ^ (lon is None):
        out.pop("lat", None)
        out.pop("lon", None)
        logger.debug("SerpAPI options: dropped lone lat/lon (both required together)")
    if "uule" in out and ("location" in out or "lat" in out):
        out.pop("uule", None)
        logger.debug("SerpAPI options: dropped uule (incompatible with location/lat per SerpAPI docs)")
    return out


async def serper_search(query: str, api_key: str, num: int = 6) -> dict[str, Any]:
    key = (api_key or "").strip()
    if not (query or "").strip() or not key:
        return {"error": "missing_query_or_api_key", "results": []}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            content=json.dumps({"q": query.strip(), "num": min(max(num, 1), 10)}),
        )
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                j = e.response.json()
                if isinstance(j, dict):
                    detail = str(j.get("message") or j.get("error") or j)[:400]
                else:
                    detail = str(j)[:400]
            except Exception:
                detail = (e.response.text or "")[:400]
            hint = (
                "Serper returned 403/401: key rejected or unauthorized. "
                "Create a key at https://serper.dev/api-key (not SerpAPI, ScaleSerp, etc.), "
                "ensure the dashboard shows free/paid credits, and set SERPER_API_KEY in .env with no quotes/spaces. "
                f"Response: {detail or e.response.status_code}"
            )
            logger.warning("Serper HTTP %s — %s", e.response.status_code, hint[:500])
            raise RuntimeError(hint) from e
        data = r.json()
    organic = data.get("organic") or []
    out: list[dict[str, Any]] = []
    for row in organic[:num]:
        out.append(
            {
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "link": row.get("link"),
            }
        )
    return {"results": out, "query": query.strip(), "provider": "serper"}


async def serpapi_search(
    query: str,
    api_key: str,
    num: int = 6,
    *,
    optional_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Google search via SerpAPI (https://serpapi.com/search.json) — same result shape as Serper for our pipeline."""
    key = (api_key or "").strip()
    if not (query or "").strip() or not key:
        return {"error": "missing_query_or_api_key", "results": []}
    n = min(max(num, 1), 10)
    params: dict[str, Any] = {
        "engine": "google",
        "q": query.strip(),
        "api_key": key,
        "num": n,
    }
    extra = sanitize_serpapi_options(optional_params)
    extra_sent = dict(extra)
    for k, v in extra.items():
        if v is True:
            params[k] = "true"
        elif v is False:
            params[k] = "false"
        else:
            params[k] = v
    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.get("https://serpapi.com/search.json", params=params)
        if r.status_code == 400 and "location" in params:
            try:
                ej = r.json()
                em = str(ej.get("error") or "") if isinstance(ej, dict) else ""
            except Exception:
                em = (r.text or "")[:400]
            if "unsupported" in em.lower() and "location" in em.lower():
                loc_preview = str(params.get("location") or "")[:120]
                logger.warning(
                    "SerpAPI rejected location=%r; retrying without location",
                    loc_preview,
                )
                params.pop("location", None)
                extra_sent.pop("location", None)
                r = await client.get("https://serpapi.com/search.json", params=params)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (e.response.text or "")[:400]
            logger.warning("SerpAPI HTTP %s — %s", e.response.status_code, detail[:200])
            raise RuntimeError(
                f"SerpAPI HTTP {e.response.status_code}. Check SERPAPI_API_KEY at https://serpapi.com/manage-api-key "
                f"and account credits. Response: {detail or e.response.status_code}"
            ) from e
        data = r.json()
    if not isinstance(data, dict):
        return {"error": "invalid_serpapi_response", "results": []}
    err = data.get("error")
    if err:
        msg = str(err)[:500]
        logger.warning("SerpAPI error field: %s", msg)
        return {"error": msg, "results": [], "query": query.strip()}
    organic = data.get("organic_results") or []
    if not isinstance(organic, list):
        organic = []
    out: list[dict[str, Any]] = []
    for row in organic[:n]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "link": row.get("link"),
            }
        )
    result: dict[str, Any] = {"results": out, "query": query.strip(), "provider": "serpapi"}
    if extra_sent:
        result["serpapi_params_used"] = dict(sorted(extra_sent.items()))
    return result


async def run_google_web_search(
    query: str,
    *,
    serper_key: str = "",
    serpapi_key: str = "",
    serpapi_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Try Serper when ``SERPER_API_KEY`` is set; otherwise ``SERPAPI_API_KEY`` (SerpAPI).

    If Serper returns 401/403 (wrong or non-Serper key), we retry the same value against SerpAPI when
    ``SERPAPI_API_KEY`` is unset — common misconfiguration is a SerpAPI key stored under ``SERPER_API_KEY``.
    If both env vars are set, SerpAPI is used only after Serper fails, or immediately when only ``SERPAPI_API_KEY`` is set.

    ``serpapi_options`` is merged into SerpAPI requests only (allowlisted keys); Serper ignores it.
    """
    sk = (serper_key or "").strip()
    ak = (serpapi_key or "").strip()
    if sk:
        try:
            return await serper_search(query, sk)
        except RuntimeError as e:
            if ak:
                logger.info("Serper failed; using SERPAPI_API_KEY for web search")
                return await serpapi_search(query, ak, optional_params=serpapi_options)
            msg = str(e)
            if "403" in msg or "401" in msg:
                logger.warning(
                    "Serper rejected the key; retrying as SerpAPI with the same value "
                    "(move SerpAPI keys to SERPAPI_API_KEY and unset SERPER_API_KEY when possible)"
                )
                try:
                    return await serpapi_search(query, sk, optional_params=serpapi_options)
                except Exception as e2:
                    raise RuntimeError(f"{msg} — SerpAPI retry failed: {e2}") from e2
            raise
    if ak:
        return await serpapi_search(query, ak, optional_params=serpapi_options)
    return {"error": "no_web_search_api_configured", "results": []}
