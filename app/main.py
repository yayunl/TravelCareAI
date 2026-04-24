from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from starlette.requests import Request

QueryStrategy = Literal[
    "single_turn",
    "single_turn_tools",
    "multi_turn",
    "multi_turn_with_tools",
    "unsolvable",
]

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, RedirectResponse

from app import llm, places_client, rag, research_agent
from app.severity_resolution import merge_effective_severity
from app.logging_config import configure_logging
from app.triage import rule_triage

logger = logging.getLogger(__name__)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


def _combined_text_for_signals(message: str, chat_history: list[ChatTurn]) -> str:
    """Triage + RAG over initial story plus every follow-up user line (order preserved)."""
    parts: list[str] = [message.strip()]
    for turn in chat_history:
        if turn.role == "user":
            parts.append(turn.content.strip())
    return "\n\n".join(parts)[:12000]


def _trim_prior_treatment_plans(raw: list[Any]) -> list[dict[str, str]]:
    """Keep ids/titles small for the LLM payload when refining a selected plan."""
    out: list[dict[str, str]] = []
    for p in raw[:8]:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "id": str(p.get("id", "")).strip()[:64],
                "title": str(p.get("title", "")).strip()[:220],
            }
        )
    return [x for x in out if x["id"]]


def _enriched_rag_query(combined: str, travel_location: str, home_country: str) -> str:
    """Bias keyword retrieval toward trip/home when the corpus has matching tokens."""
    parts = [combined.strip()]
    tl = (travel_location or "").strip()
    hc = (home_country or "").strip()
    if tl:
        parts.append(f"Traveler trip location context: {tl}")
    if hc:
        parts.append(f"Traveler home healthcare context: {hc}")
    return "\n\n".join(parts)[:12000]


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
configure_logging()

ACCESS_COOKIE_NAME = "tc_app_access"
ACCESS_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days
_access_code_raw = os.getenv("APP_ACCESS_CODE", "").strip()


def _access_gate_enabled() -> bool:
    return bool(_access_code_raw)


def _access_signing_key() -> bytes:
    secret = (os.getenv("APP_ACCESS_SIGNING_SECRET", "") or "").strip().encode("utf-8")
    if secret:
        return secret[:128]
    if not _access_code_raw:
        return b""
    return hashlib.sha256(f"tc_app_access|{_access_code_raw}".encode("utf-8")).digest()


def _mint_access_cookie_value() -> str:
    exp = int(time.time()) + ACCESS_COOKIE_MAX_AGE
    payload = str(exp).encode("utf-8")
    sig = hmac.new(_access_signing_key(), payload, hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _access_cookie_valid(value: str | None) -> bool:
    if not value or "." not in value:
        return False
    exp_s, _, sig = value.partition(".")
    if not sig:
        return False
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if exp < int(time.time()):
        return False
    payload = str(exp).encode("utf-8")
    expected = hmac.new(_access_signing_key(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def _submitted_code_matches(user_code: str) -> bool:
    if not _access_code_raw:
        return True
    uh = hashlib.sha256(user_code.encode("utf-8")).digest()
    sh = hashlib.sha256(_access_code_raw.encode("utf-8")).digest()
    return hmac.compare_digest(uh, sh)


def _access_cookie_secure_flag() -> bool:
    return os.getenv("APP_ACCESS_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes")


async def require_app_access(request: Request) -> None:
    if not _access_gate_enabled():
        return
    if not _access_cookie_valid(request.cookies.get(ACCESS_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="Access code required")


app = FastAPI(title="Travel Care AI", version="0.1.0")


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request failed method=%s path=%s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class AccessVerifyRequest(BaseModel):
    code: str = Field(default="", max_length=256)


class AssistRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="en", max_length=32)
    location: str = Field(
        default="",
        max_length=512,
        description="City, country, or free-text place the traveler is in (optional).",
    )
    home_country: str = Field(
        default="",
        max_length=256,
        description="Traveler's home country or region for contrasting healthcare systems (optional).",
    )
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=40,
        description="Optional follow-up turns after the initial message (user/assistant).",
    )
    query_strategy: QueryStrategy = Field(
        default="single_turn",
        description=(
            "Interaction pattern: single_turn | single_turn_tools | multi_turn | multi_turn_with_tools | unsolvable. "
            "single_turn and multi_turn use local corpus RAG + OpenAI only (no Google geocoding, Places, or web research). "
            "multi_turn_with_tools matches prior multi-turn server behavior (external research when enabled)."
        ),
    )
    map_latitude: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Optional map pin latitude for server-side Places nearby search.",
    )
    map_longitude: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Optional map pin longitude for server-side Places nearby search.",
    )
    research_tools: bool = Field(
        default=True,
        description="If true, run an OpenAI tool loop (Places + Serper) before the final traveler JSON when keys exist.",
    )
    selected_treatment_plan_id: str = Field(
        default="",
        max_length=64,
        description="When set to an id from a prior response treatment_plan_options, the LLM refines cost_estimate_table for that plan.",
    )
    prior_treatment_plan_options: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=8,
        description="Echo of last assist treatment_plan_options (id+title) so refinements keep context.",
    )


@app.get("/", response_model=None)
async def index_or_gate(request: Request) -> FileResponse | RedirectResponse:
    if _access_gate_enabled():
        if _access_cookie_valid(request.cookies.get(ACCESS_COOKIE_NAME)):
            return RedirectResponse(url="/app", status_code=302)
        return FileResponse(ROOT / "static" / "gate.html")
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/app", response_model=None)
async def app_shell(request: Request) -> FileResponse | RedirectResponse:
    if _access_gate_enabled() and not _access_cookie_valid(request.cookies.get(ACCESS_COOKIE_NAME)):
        return RedirectResponse(url="/", status_code=302)
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/access/verify")
async def access_verify(body: AccessVerifyRequest) -> JSONResponse:
    if not _access_gate_enabled():
        return JSONResponse({"ok": True, "gateDisabled": True})
    submitted = (body.code or "").strip()
    if not submitted or not _submitted_code_matches(submitted):
        raise HTTPException(status_code=401, detail="Invalid access code")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        ACCESS_COOKIE_NAME,
        _mint_access_cookie_value(),
        max_age=ACCESS_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_access_cookie_secure_flag(),
        path="/",
    )
    return resp


@app.get("/api/public-config")
async def public_config(_: None = Depends(require_app_access)) -> dict[str, str | bool]:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    maps_server = os.getenv("GOOGLE_MAPS_SERVER_KEY", "").strip()
    serper = os.getenv("SERPER_API_KEY", "").strip()
    serpapi = os.getenv("SERPAPI_API_KEY", "").strip()
    disable_research = os.getenv("DISABLE_RESEARCH_TOOLS", "").strip().lower() in ("1", "true", "yes")
    judge_model = os.getenv("OPENAI_JUDGE_MODEL", "").strip()
    judge_off = os.getenv("DISABLE_OPENAI_JUDGE", "").strip().lower() in ("1", "true", "yes")
    oa_key = os.getenv("OPENAI_API_KEY", "").strip()
    j_key = (os.getenv("OPENAI_JUDGE_API_KEY", "") or "").strip() or oa_key
    return {
        "googleMapsApiKey": key,
        "mapsEnabled": bool(key),
        "openAiConfigured": bool(oa_key),
        "openAiModel": (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip(),
        "openAiJudgeModel": judge_model,
        "openAiJudgeConfigured": bool(j_key and judge_model and not judge_off),
        "placesServerConfigured": bool(maps_server),
        "webSearchConfigured": bool(serper or serpapi),
        "serperConfigured": bool(serper),
        "serpapiConfigured": bool(serpapi),
        "researchToolsAllowedByEnv": not disable_research,
        "defaultTravelLocation": os.getenv("DEFAULT_TRAVEL_LOCATION", "").strip(),
        "defaultHomeCountry": os.getenv("DEFAULT_HOME_COUNTRY", "").strip(),
    }


@app.post("/api/assist")
async def assist(body: AssistRequest, _: None = Depends(require_app_access)) -> dict:
    logger.info(
        "assist start strategy=%s chat_turns=%d message_chars=%d research_tools=%s",
        body.query_strategy,
        len(body.chat_history),
        len(body.message or ""),
        body.research_tools,
    )
    combined = _combined_text_for_signals(body.message, body.chat_history)
    rules = rule_triage(combined)
    strategy = body.query_strategy
    no_server_external_tools = strategy in ("single_turn", "multi_turn")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    language = (body.language or "").strip() or "en"
    travel_location = (body.location or "").strip() or (
        os.getenv("DEFAULT_TRAVEL_LOCATION", "") or ""
    ).strip()
    home_country = (body.home_country or "").strip() or (
        os.getenv("DEFAULT_HOME_COUNTRY", "") or ""
    ).strip()
    rag_query = _enriched_rag_query(combined, travel_location, home_country)

    assist_activity_log: list[dict[str, Any]] = []
    planned_subqueries: list[str] = []
    retrieval_queries_used: list[str] | None = None
    corpus_planner_llm: dict[str, Any] | None = None
    if strategy == "single_turn_tools" and openai_key:
        try:
            planned_subqueries, corpus_planner_llm = await llm.plan_retrieval_subqueries(user_text=combined)
        except Exception:  # noqa: BLE001
            planned_subqueries = []
            corpus_planner_llm = None
        queries = [rag_query, combined] + [q for q in planned_subqueries if q and q.strip()]
        chunks = rag.retrieve_merged(queries, k_per_query=3, max_chunks=12)
        retrieval_queries_used = queries[:8]
        corpus_entry: dict[str, Any] = {
            "phase": "corpus_retrieval",
            "kind": "internal_tools",
            "title": "Extra corpus keyword searches (single_turn_tools)",
            "detail": (
                "A small OpenAI call proposed English keyword phrases; the server merged retrieval "
                "from those phrases plus your case text and trip/home context."
            ),
            "llm_planned_queries": [str(q)[:220] for q in planned_subqueries[:8]],
            "queries_run": [str(q)[:220] for q in (retrieval_queries_used or [])[:10]],
            "chunks_retrieved": len(chunks),
        }
        if corpus_planner_llm:
            corpus_entry["llm_planner_exchange"] = corpus_planner_llm
        assist_activity_log.append(corpus_entry)
    else:
        chunks = rag.retrieve(rag_query, k=4)

    citations = [{"id": c.id, "title": c.title, "excerpt": c.text[:400]} for c in chunks]
    logger.info("rag citations=%d merged=%s", len(citations), retrieval_queries_used is not None)

    maps_server_key = os.getenv("GOOGLE_MAPS_SERVER_KEY", "").strip()
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
    lat: float | None = body.map_latitude
    lng: float | None = body.map_longitude
    geocoded_travel_location = False
    if (
        not no_server_external_tools
        and maps_server_key
        and travel_location
        and (lat is None or lng is None)
    ):
        geo = await places_client.geocode_address(travel_location, maps_server_key)
        if geo:
            lat, lng = geo
            geocoded_travel_location = True
            logger.debug("geocode succeeded for assist")
        else:
            logger.info("geocode returned no coordinates for assist")

    destination_local_context: dict[str, Any] | None = None
    if not no_server_external_tools and maps_server_key and lat is not None and lng is not None:
        try:
            destination_local_context = await places_client.fetch_destination_local_context(
                lat, lng, maps_server_key
            )
        except Exception:  # noqa: BLE001
            logger.warning("destination local time lookup failed", exc_info=True)

    research_digest = ""
    research_tool_calls = 0
    research_structured: dict[str, Any] | None = None
    research_activity_trace: list[dict[str, Any]] = []
    research_error: str | None = None
    disable_research = os.getenv("DISABLE_RESEARCH_TOOLS", "").strip().lower() in ("1", "true", "yes")
    run_research = (
        not no_server_external_tools
        and bool(openai_key)
        and body.research_tools
        and not disable_research
        and (bool(maps_server_key) or bool(serper_key) or bool(serpapi_key))
    )
    if run_research:
        try:
            research_structured, research_tool_calls, research_activity_trace = await research_agent.run_research_tool_loop(
                combined_user_text=combined,
                travel_location=travel_location or None,
                home_country=home_country or None,
                latitude=lat,
                longitude=lng,
                language=language,
                openai_key=openai_key,
                maps_server_key=maps_server_key,
                serper_key=serper_key,
                serpapi_key=serpapi_key,
                destination_local_context=destination_local_context,
            )
            assist_activity_log.extend(research_activity_trace)
            research_digest = str(research_structured.get("digest_text") or "").strip()
            logger.info(
                "research complete tool_calls=%d digest_chars=%d coords=%s",
                research_tool_calls,
                len(research_digest),
                lat is not None and lng is not None,
            )
        except Exception as e:  # noqa: BLE001
            research_error = str(e)
            logger.warning("research failed: %s", research_error[:500])
            assist_activity_log.append(
                {
                    "phase": "research",
                    "kind": "error",
                    "message": research_error[:1200],
                }
            )
    if not run_research:
        logger.info(
            "research not run: strategy=%s no_server_external_tools=%s openai=%s research_tools=%s "
            "disable_env=%s maps_key=%s serper=%s serpapi=%s",
            strategy,
            no_server_external_tools,
            bool(openai_key),
            body.research_tools,
            disable_research,
            bool(maps_server_key),
            bool(serper_key),
            bool(serpapi_key),
        )

    base: dict = {
        "query_strategy": strategy,
        "travel_location": travel_location or None,
        "home_country": home_country or None,
        "locale": travel_location or "Unspecified",
        "care_level": rules.care_level,
        "emergency": rules.emergency,
        "matched_rules": rules.matched_rules,
        "rule_rationale": rules.rationale,
        "citations": citations,
        "llm": None,
        "map_coordinates_used": (
            {"latitude": lat, "longitude": lng} if lat is not None and lng is not None else None
        ),
        "geocoded_travel_location": geocoded_travel_location,
        **(
            {"destination_local_context": destination_local_context}
            if destination_local_context is not None
            else {}
        ),
        "disclaimer": (
            "Educational decision support only — not a diagnosis. "
            "If you may be having a medical emergency, contact local emergency services or go to the nearest "
            "appropriate emergency department, using the official numbers for your jurisdiction."
        ),
        "severity_source": "rules",
        "rule_triage": {
            "care_level": rules.care_level,
            "emergency": rules.emergency,
            "matched_rules": list(rules.matched_rules),
            "rationale": list(rules.rationale),
        },
    }
    if retrieval_queries_used is not None:
        base["retrieval_queries_used"] = retrieval_queries_used
    if research_digest.strip():
        base["research_from_tools_digest"] = research_digest.strip()
    if research_structured is not None:
        base["research_from_tools_structured"] = {
            k: v for k, v in research_structured.items() if k != "digest_text"
        }
    if research_tool_calls:
        base["research_tool_calls_executed"] = research_tool_calls
    if research_error:
        base["research_from_tools_error"] = research_error

    if not openai_key:
        base["llm_skip_reason"] = "missing_openai_api_key"
        logger.info("llm skipped: missing_openai_api_key")
    else:
        try:
            llm_out, llm_prompt_meta = await llm.augment_with_openai(
                user_message=body.message,
                language=language,
                travel_location=travel_location,
                home_country=home_country,
                care_level=rules.care_level,
                emergency=rules.emergency,
                rule_rationale=rules.rationale,
                citations=citations,
                chat_history=[t.model_dump() for t in body.chat_history],
                query_strategy=strategy,
                research_from_tools_digest=research_digest.strip() or None,
                research_from_tools_structured=research_structured,
                selected_treatment_plan_id=(body.selected_treatment_plan_id or "").strip() or None,
                prior_treatment_plan_options=_trim_prior_treatment_plans(body.prior_treatment_plan_options),
                destination_local_context=destination_local_context,
            )
            base["llm"] = llm_out
            if isinstance(llm_out, dict) and no_server_external_tools:
                # The model may still hallucinate named facilities with ratings from world knowledge even though
                # research_from_tools_structured.nearby_facilities was never fetched — strip so UI matches server behavior.
                llm_out["nearby_care_options"] = []
                llm_out["nearby_care_caveats"] = []
            if isinstance(llm_out, dict):
                merged = merge_effective_severity(rules, llm_out)
                base["care_level"] = merged["care_level"]
                base["emergency"] = merged["emergency"]
                base["severity_source"] = merged["severity_source"]
                base["rule_triage"] = merged["rule_triage"]
                if merged.get("llm_severity_override_rejected"):
                    base["llm_severity_override_rejected"] = True
                    base["llm_severity_override_reject_reason"] = merged["llm_severity_override_reject_reason"]
                else:
                    base.pop("llm_severity_override_rejected", None)
                    base.pop("llm_severity_override_reject_reason", None)
            if llm_prompt_meta:
                base["llm_api_prompt"] = llm_prompt_meta
            if llm_out is not None and llm_prompt_meta:
                assist_activity_log.append(
                    {
                        "phase": "traveler_reply",
                        "kind": "main_llm_augment",
                        "model": llm_prompt_meta.get("model"),
                        "temperature": llm_prompt_meta.get("temperature"),
                        "message": (
                            "OpenAI produced the structured traveler JSON (summary, next steps, "
                            "healthcare contrast, treatment options, costs, pharmacy fields when applicable)."
                        ),
                        "request_messages": [
                            {"role": "system", "content": llm_prompt_meta.get("system")},
                            {"role": "user", "content": llm_prompt_meta.get("user")},
                        ],
                        "response_message": llm_prompt_meta.get("assistant_message"),
                    }
                )

            judge_model = os.getenv("OPENAI_JUDGE_MODEL", "").strip()
            judge_off = os.getenv("DISABLE_OPENAI_JUDGE", "").strip().lower() in ("1", "true", "yes")
            if (
                judge_model
                and not judge_off
                and isinstance(llm_out, dict)
                and llm_prompt_meta
            ):
                try:
                    j_out, j_meta = await llm.evaluate_recommendation_with_judge(
                        openai_key=openai_key,
                        judge_model=judge_model,
                        traveler_case_text=combined,
                        chat_history=[t.model_dump() for t in body.chat_history],
                        travel_location=travel_location,
                        home_country=home_country,
                        server_care_level=str(base.get("care_level", rules.care_level)),
                        server_emergency=bool(base.get("emergency", rules.emergency)),
                        matched_rules=list(rules.matched_rules),
                        rule_rationale=list(rules.rationale),
                        primary_model_name=str(
                            llm_prompt_meta.get("model")
                            or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                        ).strip(),
                        recommendation_llm=llm_out,
                    )
                    base["llm_judge"] = j_out
                    if j_meta:
                        base["llm_judge_api_prompt"] = j_meta
                    assist_activity_log.append(
                        {
                            "phase": "judge_review",
                            "kind": "llm_as_judge",
                            "model": judge_model,
                            "message": "Independent evaluator (LLM-as-judge) reviewed the primary traveler JSON.",
                            "request_messages": [
                                {"role": "system", "content": j_meta.get("system")},
                                {"role": "user", "content": j_meta.get("user")},
                            ],
                            "response_message": j_meta.get("assistant_message"),
                        }
                    )
                except Exception as je:  # noqa: BLE001
                    err_j = str(je)
                    logger.warning("llm judge failed: %s", err_j[:500])
                    base["llm_judge_error"] = err_j[:1500]
                    assist_activity_log.append(
                        {
                            "phase": "judge_review",
                            "kind": "llm_judge_error",
                            "message": err_j[:1200],
                        }
                    )
        except Exception as e:  # noqa: BLE001 — surface to client for class debugging
            base["llm_error"] = str(e)
            logger.exception("llm augment failed: %s", str(e)[:300])
            assist_activity_log.append(
                {
                    "phase": "traveler_reply",
                    "kind": "main_llm_error",
                    "message": str(e)[:1200],
                }
            )

    if assist_activity_log:
        base["assist_activity_log"] = assist_activity_log

    return base


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
