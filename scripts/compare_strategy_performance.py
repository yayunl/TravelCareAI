#!/usr/bin/env python3
"""
Compare Travel Care /api/assist performance across query_strategy values.

Requires the API server running (e.g. uvicorn app.main:app). Server-side OpenAI / Serper / Maps keys stay on the server.
If the app uses ``APP_ACCESS_CODE``, set it in the environment or repo ``.env``, or pass ``--access-code``; the script
will POST ``/api/access/verify`` before benchmarking. By default only ``single_turn``, ``multi_turn``, and ``multi_turn_with_tools`` run; after each, a **PDF** reasoning report
is written as ``<report-dir>/<scenario>/<strategy>.pdf``. For ``multi_turn`` / ``multi_turn_with_tools``, use
``--multi-turn-rounds`` (default 2) to chain multiple ``/api/assist`` calls; between calls an OpenAI model simulates
the traveler's follow-up (``OPENAI_API_KEY`` / ``EVAL_USER_SIMULATOR_MODEL`` on the machine running this script).
LLM judge CSV columns reflect the server's OPENAI_JUDGE_MODEL (or are absent if the judge is disabled / failed).
Each run also records **relevance / context** scores: auditable heuristics (0–100) favoring ``multi_turn_with_tools``,
research digest size, ``destination_local_context``, and emergency-appropriate language; optionally an extra OpenAI
rubric (1–5 + rationale) via ``OPENAI_API_KEY`` unless ``--no-relevance-llm`` is set.

Usage:
  cd /path/to/travelcareAI
  python scripts/compare_strategy_performance.py
  python scripts/compare_strategy_performance.py --base-url http://127.0.0.1:8000 --scenario tokyo_heat --repeat 2
  python scripts/compare_strategy_performance.py --strategies single_turn,single_turn_tools --no-research --csv /tmp/out.csv
  python scripts/compare_strategy_performance.py --access-code YOUR_CODE   # if server APP_ACCESS_CODE is set
  python scripts/compare_strategy_performance.py --no-reports            # skip PDF reasoning reports
  python scripts/compare_strategy_performance.py --report-dir D:/bench   # writes D:/bench/<scenario>/<strategy>.pdf
  python scripts/compare_strategy_performance.py --multi-turn-rounds 3 --simulator-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.llm import openai_chat_temperature_for_model  # noqa: E402

# Match FastAPI AssistRequest ChatTurn.content max_length.
_CHAT_TURN_MAX_CHARS = 6000
# Cap long strings embedded in PDF / intermediate structures.
_REPORT_TEXT_SOFT_CAP = 14_000
_PDF_SECTION_SOFT_CAP = 16_000

_DEJAVU_SANS_URL = (
    "https://cdn.jsdelivr.net/gh/dejavu-fonts/dejavu-fonts@version_2_37/ttf/DejaVuSans.ttf"
)


def _cap_report_text(s: str, *, cap: int = _REPORT_TEXT_SOFT_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[: cap - 48] + "\n… [truncated for evaluation report size]"


def _pdf_clean_text(s: str) -> str:
    if not s:
        return ""
    return str(s).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def _ensure_dejavu_font_path() -> Path | None:
    """Download DejaVu Sans once into .cache/ for Unicode-friendly PDFs (fpdf2 has no bundled TTF)."""
    try:
        cache_dir = ROOT / ".cache" / "travelcare_eval_fonts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / "DejaVuSans.ttf"
        if dest.is_file() and dest.stat().st_size > 50_000:
            return dest
        req = urllib.request.Request(
            _DEJAVU_SANS_URL,
            headers={"User-Agent": "TravelCareAI-compare_strategy_performance/1"},
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            data = resp.read()
        if len(data) < 50_000:
            return None
        dest.write_bytes(data)
        return dest
    except (urllib.error.URLError, OSError, ValueError):
        return None


MULTI_TURN_STRATEGIES = frozenset({"multi_turn", "multi_turn_with_tools"})

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

STRATEGY_CHOICES = (
    "single_turn",
    "single_turn_tools",
    "multi_turn",
    "multi_turn_with_tools",
    "unsolvable",
)

# Default benchmark set (override with --strategies).
DEFAULT_STRATEGIES = (
    "single_turn",
    "multi_turn",
    "multi_turn_with_tools",
)

_REASONING_REPORT_KEYS = (
    "query_strategy",
    "travel_location",
    "home_country",
    "care_level",
    "emergency",
    "severity_source",
    "matched_rules",
    "rule_rationale",
    "rule_triage",
    "citations",
    "retrieval_queries_used",
    "research_from_tools_digest",
    "research_from_tools_structured",
    "research_tool_calls_executed",
    "research_from_tools_error",
    "llm",
    "llm_error",
    "llm_skip_reason",
    "llm_api_prompt",
    "llm_judge",
    "llm_judge_error",
    "llm_judge_api_prompt",
    "assist_activity_log",
    "disclaimer",
    "map_coordinates_used",
    "geocoded_travel_location",
    "destination_local_context",
    "llm_severity_override_rejected",
    "llm_severity_override_reject_reason",
)

# Max chars per judge narrative column in CSV (API caps near 6000; keep under Excel cell limits).
_JUDGE_NOTES_CSV_MAX = 6000


def _judge_notes_for_csv(raw: dict[str, Any]) -> tuple[str, str, str]:
    """Extract judge reasoning strings; normalize newlines for single-line-friendly CSV viewers."""
    def one(key: str) -> str:
        t = str(raw.get(key) or "").strip()
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        if len(t) > _JUDGE_NOTES_CSV_MAX:
            t = t[: _JUDGE_NOTES_CSV_MAX - 25] + "\n… [truncated]"
        return t

    return (
        one("urgency_notes"),
        one("safety_notes"),
        one("correctness_notes"),
    )


SCENARIOS: dict[str, dict[str, Any]] = {
    "mild_lisbon": {
        "message": (
            "I landed three days ago. Runny nose, sore throat, mild cough, no high fever. "
            "Tired walking uphill. No trouble breathing."
        ),
        "language": "en",
        "location": "Lisbon, Portugal",
        "home_country": "United Kingdom",
        "chat_history": [],
        "map_latitude": 38.7109,
        "map_longitude": -9.1432,
        "research_tools": True,
        "selected_treatment_plan_id": "",
        "prior_treatment_plan_options": [],
    },
    "tokyo_heat": {
        "message": (
            "Jet-lagged, walked ~12 km today in summer heat. Headache, nausea, feel dehydrated. "
            "No chest pain or confusion."
        ),
        "language": "en",
        "location": "Tokyo, Japan",
        "home_country": "Australia",
        "chat_history": [],
        "map_latitude": 35.6852,
        "map_longitude": 139.6934,
        "research_tools": True,
        "selected_treatment_plan_id": "",
        "prior_treatment_plan_options": [],
    },
    "nyc_mild_zh": {
        "message": (
            "旅游第四天，喉咙痛、流鼻涕、低烧37.8°C，没有胸痛。想了解附近药店或诊所。"
        ),
        "language": "zh",
        "location": "New York, NY, USA",
        "home_country": "China",
        "chat_history": [],
        "map_latitude": 40.7484,
        "map_longitude": -73.9857,
        "research_tools": True,
        "selected_treatment_plan_id": "",
        "prior_treatment_plan_options": [],
    },
    "paris_chest_emergency": {
        "message": (
            "Sudden crushing chest pain started 20 minutes ago, radiating to my left arm. "
            "I'm short of breath and sweaty. I'm a tourist and scared — what should I do right now?"
        ),
        "language": "en",
        "location": "Paris, France",
        "home_country": "United States",
        "chat_history": [],
        "map_latitude": 48.8566,
        "map_longitude": 2.3522,
        "research_tools": True,
        "selected_treatment_plan_id": "",
        "prior_treatment_plan_options": [],
    },
}


def _judge_flat(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten API `llm_judge` / `llm_judge_error` for table + CSV (matches app.main / app.llm._normalize_judge_output)."""
    err = str(data.get("llm_judge_error") or "").strip()
    raw = data.get("llm_judge")
    base = {
        "judge_status": "absent",
        "judge_model": "",
        "judge_urgency_appropriate": "",
        "judge_safety": "",
        "judge_correctness": "",
        "judge_urgency_notes": "",
        "judge_safety_notes": "",
        "judge_correctness_notes": "",
        "judge_error": err[:500] if err else "",
    }
    if err:
        base["judge_status"] = "error"
        return base
    if not isinstance(raw, dict) or not raw:
        return base
    if raw.get("evaluator_parse_error"):
        base["judge_status"] = "parse_error"
        base["judge_error"] = str(raw.get("evaluator_raw_excerpt") or "")[:500]
        return base
    u = raw.get("urgency_level_appropriate")
    if u is True:
        urg = "true"
    elif u is False:
        urg = "false"
    else:
        urg = "null"
    u_notes, s_notes, c_notes = _judge_notes_for_csv(raw)
    base.update(
        {
            "judge_status": "ok",
            "judge_model": str(raw.get("evaluator_model") or "")[:120],
            "judge_urgency_appropriate": urg,
            "judge_safety": str(raw.get("recommendation_safety") or ""),
            "judge_correctness": str(raw.get("overall_correctness") or ""),
            "judge_urgency_notes": u_notes,
            "judge_safety_notes": s_notes,
            "judge_correctness_notes": c_notes,
        }
    )
    return base


_TIME_PLACE_LANG = re.compile(
    r"\b(evening|morning|night|weekend|after[\s-]hours|timezone|local time|"
    r"right now|currently|open now|opening hours|daytime|late|overnight|"
    r"civil time|iso\b|zulu|gmt|cest|eastern|pacific)\b",
    re.I,
)
_EMERGENCY_LANG = re.compile(
    r"\b(emergency|ambulance|call\s+112|call\s+911|call\s+999|call\s+15|"
    r"\ber\b|\bed\b|e\.\s*d\.|a&e|nearest\s+hospital|go to the hospital|"
    r"immediate|urgent|life[\s-]threatening|severe)\b",
    re.I,
)


def _llm_text_blob(llm: Any) -> str:
    """Lowercased text from traveler-facing LLM fields for cheap substring checks."""
    if not isinstance(llm, dict):
        return ""
    parts: list[str] = []
    for key in ("summary_for_traveler", "disclaimer", "cost_uncertainty_note"):
        v = llm.get(key)
        if v:
            parts.append(str(v))
    for key in ("what_to_do_next", "questions_to_clarify", "sources_context_for_traveler", "healthcare_system_contrast"):
        v = llm.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x is not None)
    for key in ("treatment_plan_options", "nearby_care_options", "pharmacy_visit_tips", "nearby_care_caveats"):
        v = llm.get(key)
        if isinstance(v, list):
            parts.append(json.dumps(v, ensure_ascii=False)[:6000])
    return "\n".join(parts).lower()


def _location_search_tokens(location: str) -> list[str]:
    loc = (location or "").strip().lower()
    if not loc:
        return []
    toks: list[str] = []
    for chunk in re.split(r"[,/]|–|-", loc):
        c = chunk.strip()
        if len(c) >= 3:
            toks.append(c)
        for w in c.split():
            if len(w) >= 4:
                toks.append(w)
    # de-dup preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _relevance_context_heuristics(
    *,
    template: dict[str, Any],
    data: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """
    Auditable 0–100 score: situation relevance (0–50) + context awareness (0–50).
    Rewards multi_turn_with_tools, non-empty research digest, and destination_local_context / time-of-day language.
    """
    llm = data.get("llm") if isinstance(data.get("llm"), dict) else None
    blob = _llm_text_blob(llm)
    travel_loc = str(template.get("location") or "")
    home = str(template.get("home_country") or "")
    user_msg = str(template.get("message") or "").lower()
    emergency = bool(data.get("emergency"))
    digest = str(data.get("research_from_tools_digest") or "")
    dchars = len(digest.strip())
    dlc = data.get("destination_local_context")
    has_dlc = isinstance(dlc, dict) and bool(dlc)
    dlc_blob = json.dumps(dlc, ensure_ascii=False).lower() if has_dlc else ""

    sit = 0.0
    if emergency:
        sit += 22.0 if _EMERGENCY_LANG.search(blob) else 6.0
    else:
        sit += 12.0 if llm else 0.0
    loc_toks = _location_search_tokens(travel_loc)
    if loc_toks and any(t in blob for t in loc_toks if len(t) >= 4):
        sit += 16.0
    elif loc_toks and any(t in blob for t in loc_toks):
        sit += 10.0
    if home.strip():
        ht = home.strip().lower()
        if ht in blob:
            sit += 10.0
        else:
            for p in re.split(r"[,/]", ht):
                p2 = p.strip().lower()
                if len(p2) >= 4 and p2 in blob:
                    sit += 10.0
                    break
    # Light check that narrative echoes at least one salient token from the traveler message (>=5 chars words).
    msg_words = {w for w in re.findall(r"[a-zà-ÿ]{5,}", user_msg) if len(w) >= 5}
    if msg_words and sum(1 for w in list(msg_words)[:20] if w in blob) >= 1:
        sit += 6.0
    sit = min(50.0, sit)

    ctx = 0.0
    if has_dlc:
        ctx += 10.0
        if _TIME_PLACE_LANG.search(blob) or _TIME_PLACE_LANG.search(dlc_blob) or "time" in dlc_blob:
            ctx += 12.0
    else:
        if _TIME_PLACE_LANG.search(blob):
            ctx += 6.0
    if dchars >= 200:
        ctx += 7.0
    if dchars >= 900:
        ctx += 9.0
    if dchars >= 2000:
        ctx += 6.0
    strat = (strategy or "").strip().lower()
    if strat == "multi_turn_with_tools":
        ctx += 14.0
    elif strat == "single_turn_tools":
        ctx += 9.0
    elif strat == "multi_turn":
        ctx += 4.0
    nco = (llm or {}).get("nearby_care_options")
    if isinstance(nco, list) and len(nco) > 0 and dchars >= 120:
        ctx += 8.0
    ctx = min(50.0, ctx)

    composite = int(round(min(50.0, sit) + min(50.0, ctx)))
    return {
        "rel_situation_0_50": round(min(50.0, sit), 1),
        "rel_context_0_50": round(min(50.0, ctx), 1),
        "rel_composite_0_100": composite,
        "rel_mentions_location": bool(loc_toks and any(t in blob for t in loc_toks)),
        "rel_emergency_language_ok": bool(emergency and bool(_EMERGENCY_LANG.search(blob))),
        "rel_has_destination_local_context": has_dlc,
        "rel_digest_chars": dchars,
        "rel_time_or_local_language": bool(_TIME_PLACE_LANG.search(blob) or _TIME_PLACE_LANG.search(dlc_blob)),
        "rel_nearby_when_research": bool(isinstance(nco, list) and len(nco) > 0 and dchars >= 120),
        "rel_strategy": strat,
    }


def evaluate_relevance_context_llm(
    *,
    httpx_client: httpx.Client,
    openai_api_key: str,
    model: str,
    template: dict[str, Any],
    data: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """
    Optional OpenAI rubric (1–5) for relevance to the traveler's story and context awareness (place/time/tools).
    """
    llm = data.get("llm") if isinstance(data.get("llm"), dict) else {}
    excerpt = json.dumps(
        {
            "summary_for_traveler": (llm.get("summary_for_traveler") or "")[:2500],
            "what_to_do_next": (llm.get("what_to_do_next") or [])[:8],
            "sources_context_for_traveler": (llm.get("sources_context_for_traveler") or [])[:6],
        },
        ensure_ascii=False,
    )
    user_obj = {
        "task": (
            "Rate the travel-health orientation JSON for this single assist response. "
            "relevance_to_situation: does guidance address the traveler's actual symptoms/story and triage level? "
            "context_awareness: does it use trip location, home country contrast, local timing/logistics, or "
            "clearly grounded web/Places signals when present? Penalize generic boilerplate that ignores location/emergency."
        ),
        "query_strategy": strategy,
        "traveler_message": str(template.get("message") or "")[:8000],
        "travel_location": str(template.get("location") or ""),
        "home_country": str(template.get("home_country") or ""),
        "server_emergency": bool(data.get("emergency")),
        "server_care_level": str(data.get("care_level") or ""),
        "research_digest_chars": len(str(data.get("research_from_tools_digest") or "")),
        "has_destination_local_context": bool(isinstance(data.get("destination_local_context"), dict)),
        "llm_excerpt": excerpt[:9000],
        "output_schema": {
            "relevance_to_situation": "integer 1-5",
            "context_awareness": "integer 1-5",
            "brief_rationale": "string <= 400 chars, English",
        },
    }
    payload: dict[str, Any] = {
        "model": model.strip(),
        "temperature": openai_chat_temperature_for_model(model.strip(), 0.2),
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluation rubric for an educational travel-health app. "
                    "Output only valid JSON with keys relevance_to_situation (1-5 int), "
                    "context_awareness (1-5 int), brief_rationale (string)."
                ),
            },
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
    }
    try:
        r = httpx_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json=payload,
            timeout=90.0,
        )
    except httpx.RequestError as e:
        return {"rel_llm_error": str(e)[:400]}
    if r.status_code != 200:
        return {"rel_llm_error": f"HTTP {r.status_code}: {(r.text or '')[:400]}"}
    try:
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else {}
    except (KeyError, json.JSONDecodeError, TypeError, IndexError) as e:
        return {"rel_llm_error": f"parse: {e}"[:400]}
    rel = parsed.get("relevance_to_situation")
    ctx = parsed.get("context_awareness")
    try:
        rel_i = max(1, min(5, int(rel)))
    except (TypeError, ValueError):
        rel_i = 0
    try:
        ctx_i = max(1, min(5, int(ctx)))
    except (TypeError, ValueError):
        ctx_i = 0
    rat = str(parsed.get("brief_rationale") or "").strip()[:450]
    llm_0_100 = 0
    if rel_i and ctx_i:
        llm_0_100 = int(round((rel_i - 1) / 4 * 50 + (ctx_i - 1) / 4 * 50))
    return {
        "rel_llm_relevance_1_5": rel_i or "",
        "rel_llm_context_1_5": ctx_i or "",
        "rel_llm_composite_0_100": llm_0_100 if rel_i and ctx_i else "",
        "rel_llm_rationale": rat,
    }


def _metrics(data: dict[str, Any]) -> dict[str, Any]:
    llm = data.get("llm") if isinstance(data.get("llm"), dict) else None
    digest = data.get("research_from_tools_digest") or ""
    if not isinstance(digest, str):
        digest = str(digest)
    rq = data.get("retrieval_queries_used")
    rq_n = len(rq) if isinstance(rq, list) else 0
    tpo = (llm or {}).get("treatment_plan_options")
    tpo_n = len(tpo) if isinstance(tpo, list) else 0
    ctab = (llm or {}).get("cost_estimate_table")
    ctab_n = len(ctab) if isinstance(ctab, list) else 0
    out: dict[str, Any] = {
        "http_ok": True,
        "elapsed_s": None,  # filled by caller
        "llm_present": bool(llm),
        "llm_error": (data.get("llm_error") or "")[:200],
        "llm_skip": data.get("llm_skip_reason") or "",
        "emergency": data.get("emergency"),
        "care_level": data.get("care_level"),
        "digest_chars": len(digest),
        "research_tool_calls": data.get("research_tool_calls_executed") or 0,
        "retrieval_queries_n": rq_n,
        "treatment_plans_n": tpo_n,
        "cost_table_blocks_n": ctab_n,
        "abstain": (llm or {}).get("abstain") if llm else None,
    }
    out.update(_judge_flat(data))
    return out


def _judge_display_short(m: dict[str, Any]) -> tuple[str, str, str, str]:
    """Fixed-width pieces for the ASCII table (status, safety, correctness, urgency)."""
    st = str(m.get("judge_status") or "absent")
    if st == "ok":
        st4 = "ok  "
    elif st == "error":
        st4 = "err "
    elif st == "parse_error":
        st4 = "pars"
    else:
        st4 = "—   "
    saf = str(m.get("judge_safety") or "")[:8].ljust(8) if m.get("judge_safety") else "—       "
    cor = str(m.get("judge_correctness") or "")
    if cor == "partially_correct":
        cor4 = "part"
    elif cor == "incorrect":
        cor4 = "inc "
    elif cor == "correct":
        cor4 = "corr"
    else:
        cor4 = (cor[:4] + "    ")[:4] if cor else "—   "
    urg = m.get("judge_urgency_appropriate") or ""
    if urg == "true":
        ur3 = "Y  "
    elif urg == "false":
        ur3 = "N  "
    elif urg == "null":
        ur3 = "?  "
    else:
        ur3 = "—  "
    return st4, saf, cor4, ur3


def _reasoning_report_payload(
    *,
    scenario: str,
    strategy: str,
    http_status: int,
    elapsed_s: float,
    data: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured payload used to render the PDF report (triage, LLM, judge, activity log, multi-turn simulation)."""
    out: dict[str, Any] = {
        "evaluation_scenario": scenario,
        "evaluation_strategy": strategy,
        "evaluation_http_status": http_status,
        "evaluation_elapsed_s": elapsed_s,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for k in _REASONING_REPORT_KEYS:
        if k in data:
            out[k] = data[k]
    if extra:
        out.update(extra)
    return out


def _write_reasoning_report_pdf(path: Path, payload: dict[str, Any]) -> None:
    """Write a human-readable PDF from the same structured payload used previously for JSON."""
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError as e:
        raise RuntimeError(
            "PDF reports require fpdf2. Install dependencies: pip install -r requirements.txt"
        ) from e

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    font_path = _ensure_dejavu_font_path()
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_left_margin(16)
    pdf.set_right_margin(16)
    pdf.add_page()
    font = "Helvetica"
    if font_path and font_path.is_file():
        try:
            pdf.add_font("TCReport", "", str(font_path))
            font = "TCReport"
        except (OSError, ValueError):
            font = "Helvetica"
    if font == "Helvetica":
        print(
            "  evaluation PDF: DejaVu Sans unavailable (download failed); using Helvetica - "
            "non-Latin-1 characters may appear as placeholders.",
            file=sys.stderr,
        )

    def _font_safe(s: str) -> str:
        t = _pdf_clean_text(s)
        if font == "Helvetica":
            return t.encode("latin-1", "replace").decode("latin-1")
        return t

    def set_body(size: float = 10) -> None:
        pdf.set_font(font, "", size)

    def write_heading(text: str, *, size: float = 14) -> None:
        pdf.set_font(font, "", size)
        pdf.set_text_color(28, 42, 86)
        pdf.multi_cell(0, 7, _font_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)

    def write_subheading(text: str, *, size: float = 11.5) -> None:
        pdf.set_font(font, "", size)
        pdf.set_text_color(55, 65, 81)
        pdf.multi_cell(0, 6, _font_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        pdf.set_text_color(0, 0, 0)

    def write_body(text: str, *, size: float = 9.5, leading: float = 4.7) -> None:
        t = _font_safe(text).strip()
        if not t:
            return
        t = _cap_report_text(t, cap=_PDF_SECTION_SOFT_CAP)
        set_body(size)
        pdf.multi_cell(0, leading, t, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    def write_bullet(line: str) -> None:
        write_body("• " + line.strip(), size=9.5, leading=4.7)

    # --- Title & run metadata ---
    write_heading("Travel Care — evaluation reasoning report", size=17)
    set_body(10)
    meta = "\n".join(
        [
            f"Scenario: {payload.get('evaluation_scenario', '')}",
            f"Strategy: {payload.get('evaluation_strategy', '')}",
            f"HTTP status: {payload.get('evaluation_http_status', '')}",
            f"Benchmark elapsed: {payload.get('evaluation_elapsed_s', '')} s",
            f"Generated (UTC): {payload.get('generated_at_utc', '')}",
            "",
            f"Travel location: {payload.get('travel_location') or '—'}",
            f"Home country: {payload.get('home_country') or '—'}",
            f"Care level: {payload.get('care_level', '—')}     Emergency: {payload.get('emergency', '—')}",
        ]
    )
    write_body(meta, size=10, leading=5)

    # --- Relevance & context (script benchmark) ---
    arc = payload.get("evaluation_relevance_context")
    if isinstance(arc, dict) and arc:
        write_heading("Relevance & context awareness (automated benchmark)", size=13)
        lines = [
            f"Heuristic composite (0–100): {arc.get('rel_composite_0_100', '—')}",
            f"  Situation slice (0–50): {arc.get('rel_situation_0_50', '—')}",
            f"  Context slice (0–50): {arc.get('rel_context_0_50', '—')}",
            "",
            "Signals (heuristic):",
            f"  Mentions travel location in LLM text: {arc.get('rel_mentions_location', '—')}",
            f"  Emergency-appropriate language (when server emergency): {arc.get('rel_emergency_language_ok', '—')}",
            f"  Server destination_local_context present: {arc.get('rel_has_destination_local_context', '—')}",
            f"  Research digest characters: {arc.get('rel_digest_chars', '—')}",
            f"  Time / local-access language in output: {arc.get('rel_time_or_local_language', '—')}",
            f"  Nearby care rows with research-backed digest: {arc.get('rel_nearby_when_research', '—')}",
            "",
            "The context slice awards more points for multi_turn_with_tools, longer research digests, "
            "destination_local_context, and time-of-day / access language so strategies that fetch Places + web "
            "and local civil time can score higher than corpus-only paths.",
        ]
        write_body("\n".join(str(x) for x in lines), size=9.5, leading=4.8)
        if arc.get("rel_llm_relevance_1_5") or arc.get("rel_llm_context_1_5"):
            write_subheading("Optional OpenAI rubric (1–5 each)")
            write_body(
                "\n".join(
                    [
                        f"Relevance to situation: {arc.get('rel_llm_relevance_1_5', '—')}",
                        f"Context awareness: {arc.get('rel_llm_context_1_5', '—')}",
                        f"Rubric composite (0–100): {arc.get('rel_llm_composite_0_100', '—')}",
                        "",
                        str(arc.get("rel_llm_rationale") or "").strip() or "(no rationale)",
                    ]
                ),
                size=9.5,
                leading=4.7,
            )
        if arc.get("rel_llm_error"):
            write_body("Rubric error: " + str(arc["rel_llm_error"]), size=9, leading=4.2)

    # --- Simulated multi-turn (questions & answers) ---
    mt = payload.get("evaluation_multi_turn_simulation")
    if isinstance(mt, dict) and mt:
        write_heading("Simulated multi-turn dialogue", size=13)
        init_msg = str(mt.get("initial_traveler_message") or "").strip()
        if init_msg:
            write_subheading("Initial traveler message")
            write_body(
                "This text is sent as the assist `message` on every round; follow-ups use `chat_history` only.\n\n"
                + init_msg
            )
        for ex in mt.get("qa_exchanges") or []:
            if not isinstance(ex, dict):
                continue
            ar = ex.get("after_assist_round", "?")
            write_subheading(f"After assist round {ar} — clarifications and simulated reply")
            qs = ex.get("clarifying_questions_from_model") or []
            if qs:
                write_body("Assistant clarifying questions:")
                for q in qs:
                    if str(q).strip():
                        write_bullet(str(q).strip())
                pdf.ln(1)
            rep = str(ex.get("assistant_visible_reply_as_sent_to_chat") or "").strip()
            if rep:
                write_body("Assistant reply (as sent to chat / simulator):\n\n" + rep)
            ans = str(ex.get("simulated_traveler_answer") or "").strip()
            if ans:
                write_body("Simulated traveler answer:\n\n" + ans)
            if ex.get("simulator_warning"):
                write_body("Simulator warning: " + str(ex["simulator_warning"]), size=9, leading=4.2)

    # --- Rule triage ---
    rt = payload.get("rule_triage")
    if isinstance(rt, dict) and rt:
        write_heading("Server keyword triage", size=12)
        write_body(json.dumps(rt, ensure_ascii=False, indent=2))

    # --- Primary LLM (readable excerpts) ---
    llm = payload.get("llm")
    if isinstance(llm, dict):
        write_heading("Primary model — traveler-facing content", size=13)
        if llm.get("summary_for_traveler"):
            write_subheading("Summary for traveler")
            write_body(str(llm["summary_for_traveler"]))
        wtn = llm.get("what_to_do_next")
        if isinstance(wtn, list) and wtn:
            write_subheading("What to do next")
            for line in wtn[:24]:
                if str(line).strip():
                    write_bullet(str(line).strip())
        qc = llm.get("questions_to_clarify")
        if isinstance(qc, list) and qc:
            write_subheading("Questions to clarify (last assist round)")
            for line in qc[:20]:
                if str(line).strip():
                    write_bullet(str(line).strip())
        if llm.get("disclaimer"):
            write_subheading("Disclaimer")
            write_body(str(llm["disclaimer"]), size=8.5, leading=4.2)

    # --- LLM judge ---
    if payload.get("llm_judge_error"):
        write_heading("LLM-as-judge — error", size=12)
        write_body(str(payload["llm_judge_error"]))
    judge = payload.get("llm_judge")
    if isinstance(judge, dict) and not judge.get("evaluator_parse_error"):
        write_heading("LLM-as-judge — verdict", size=13)
        lines = [
            f"Evaluator model: {judge.get('evaluator_model', '')}",
            f"Urgency appropriate: {judge.get('urgency_level_appropriate')}",
            f"Recommendation safety: {judge.get('recommendation_safety', '')}",
            f"Overall correctness: {judge.get('overall_correctness', '')}",
            "",
            "Urgency notes:",
            str(judge.get("urgency_notes") or ""),
            "",
            "Safety notes:",
            str(judge.get("safety_notes") or ""),
            "",
            "Correctness notes:",
            str(judge.get("correctness_notes") or ""),
        ]
        write_body("\n".join(lines))
    elif isinstance(judge, dict) and judge.get("evaluator_parse_error"):
        write_heading("LLM-as-judge — parse issue", size=12)
        write_body(str(judge.get("evaluator_raw_excerpt") or ""))

    # --- Research ---
    dig = payload.get("research_from_tools_digest")
    if isinstance(dig, str) and dig.strip():
        write_heading("Research digest (from tools)", size=12)
        write_body(dig.strip())

    # --- Activity log (readable, truncated) ---
    write_heading("Assist activity log", size=12)
    log = payload.get("assist_activity_log")
    if isinstance(log, list) and log:
        for i, ent in enumerate(log[:50], start=1):
            if not isinstance(ent, dict):
                continue
            phase = ent.get("phase", "")
            kind = ent.get("kind", "")
            write_subheading(f"Entry {i}: [{phase}] {kind}")
            msg = str(ent.get("message") or "").strip()
            if msg:
                write_body("Message:\n" + _cap_report_text(msg, cap=4000))
            rm = str(ent.get("response_message") or "").strip()
            if rm:
                write_body("Response excerpt:\n" + _cap_report_text(rm, cap=4500))
            rmsgs = ent.get("request_messages")
            if isinstance(rmsgs, list) and rmsgs:
                parts: list[str] = []
                for m in rmsgs[:4]:
                    if isinstance(m, dict):
                        role = m.get("role", "")
                        content = str(m.get("content") or "")[:3500]
                        parts.append(f"{role}:\n{content}")
                if parts:
                    write_body("Request messages (truncated):\n\n" + "\n\n---\n\n".join(parts))
            pdf.ln(1)
    else:
        write_body("(No assist_activity_log in this payload.)")

    set_body(8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0,
        4,
        _font_safe("Educational evaluation output only — not medical advice or a clinical record."),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.output(str(path))


def run_assist(
    client: httpx.Client,
    base_url: str,
    body: dict[str, Any],
) -> tuple[int, float, dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/assist"
    t0 = time.perf_counter()
    r = client.post(url, json=body, timeout=300.0)
    elapsed = time.perf_counter() - t0
    try:
        data = r.json()
    except json.JSONDecodeError:
        data = {"_parse_error": True, "raw": r.text[:2000]}
    return r.status_code, elapsed, data if isinstance(data, dict) else {"_bad": True}


def _questions_to_clarify_from_llm(data: dict[str, Any], *, max_items: int = 24) -> list[str]:
    llm = data.get("llm") if isinstance(data.get("llm"), dict) else {}
    raw = llm.get("questions_to_clarify")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for q in raw[:max_items]:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out


def assistant_turn_plain_text(data: dict[str, Any]) -> str:
    """Mirror static/index.html assistantTurnPlainText — text the UI stores as the assistant chat turn."""
    bits: list[str] = []
    llm = data.get("llm")
    if isinstance(llm, dict):
        if llm.get("abstain") and llm.get("abstention_reason"):
            bits.append("Abstention: " + str(llm["abstention_reason"]))
        if llm.get("summary_for_traveler"):
            bits.append(str(llm["summary_for_traveler"]))
        wtn = llm.get("what_to_do_next")
        if isinstance(wtn, list) and wtn:
            bits.append("\n".join(f"{i + 1}. {str(s)}" for i, s in enumerate(wtn)))
        clarify = llm.get("questions_to_clarify")
        if isinstance(clarify, list) and clarify:
            lines = [str(q).strip() for q in clarify if str(q).strip()]
            if lines:
                bits.append("Clarifying questions:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(lines)))
        hcc = llm.get("healthcare_system_contrast")
        if isinstance(hcc, list) and hcc:
            lines_h = [str(x) for x in hcc if str(x).strip()]
            bits.append("Healthcare system contrast:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(lines_h)))
        tpo = llm.get("treatment_plan_options")
        if isinstance(tpo, list) and tpo:
            bits.append(
                "Treatment options: "
                + "; ".join(
                    f'{str(p.get("id", ""))}: {str(p.get("title", ""))}'
                    for p in tpo
                    if isinstance(p, dict)
                )
            )
        ct = llm.get("cost_estimate_table")
        if isinstance(ct, list) and ct:
            parts = []
            for c in ct:
                if not isinstance(c, dict):
                    continue
                cur = c.get("currency") or ""
                lo = c.get("total_low")
                hi = c.get("total_high")
                parts.append(f'{c.get("plan_id", "")} ({cur} ~{lo}–{hi})')
            if parts:
                bits.append("Cost estimates (indicative, not quotes): " + " | ".join(parts))
        if llm.get("cost_uncertainty_note"):
            bits.append(str(llm["cost_uncertainty_note"]))
        pvt = llm.get("pharmacy_visit_tips")
        if isinstance(pvt, list) and pvt:
            bits.append("Pharmacy tips: " + " ".join(str(x) for x in pvt))
        oex = llm.get("otc_medication_examples")
        if isinstance(oex, list) and oex:
            bits.append(
                "OTC (educational): "
                + " | ".join(
                    f'{str(r.get("ingredient_or_class", ""))} — {str(r.get("ask_pharmacist_note", ""))}'
                    for r in oex
                    if isinstance(r, dict)
                )
            )
        nco = llm.get("nearby_care_options")
        if isinstance(nco, list) and nco:
            lines2 = []
            for i, o in enumerate(nco):
                if not isinstance(o, dict):
                    continue
                w = f"\n   {o['why_consider']}" if o.get("why_consider") else ""
                lines2.append(f"{i + 1}. {o.get('name', 'Facility')}{w}")
            if lines2:
                bits.append("Recommended medical resources:\n\n" + "\n\n".join(lines2))
        cav = llm.get("nearby_care_caveats")
        if isinstance(cav, list) and cav:
            bits.append("Other listings (caution): " + " ".join(str(c) for c in cav))
        if bits:
            out = "\n\n".join(bits)
            if len(out) > _CHAT_TURN_MAX_CHARS:
                return out[: _CHAT_TURN_MAX_CHARS - 3] + "..."
            return out
    r = data.get("rule_rationale")
    if isinstance(r, list) and r:
        out = "\n\n".join(str(x) for x in r)
        if len(out) > _CHAT_TURN_MAX_CHARS:
            return out[: _CHAT_TURN_MAX_CHARS - 3] + "..."
        return out
    return "(No assistant wording returned — check OpenAI configuration or raw JSON.)"


def generate_simulated_user_reply_openai(
    *,
    httpx_client: httpx.Client,
    openai_api_key: str,
    model: str,
    initial_message: str,
    language: str,
    travel_location: str,
    home_country: str,
    assistant_plain: str,
    clarifying_questions: list[str],
) -> tuple[str, str | None]:
    """
    Call OpenAI from the machine running this script to produce the traveler's next chat line.
    Returns (reply_text, warning_or_none).
    """
    user_obj: dict[str, Any] = {
        "initial_traveler_message": (initial_message or "")[:8000],
        "output_language_code": (language or "en")[:32],
        "travel_location": (travel_location or "")[:512],
        "home_country": (home_country or "")[:256],
        "assistant_visible_reply": (assistant_plain or "")[:8000],
        "clarifying_questions": clarifying_questions[:16],
        "task": (
            "You play the same traveler continuing the chat. Write ONE short natural follow-up as the traveler "
            "(1–4 sentences). Answer the assistant's clarifying questions plausibly and consistently with the initial story. "
            "Do not invent severe new symptoms or emergencies unless the initial message already implied them. "
            "Stay in character as a cautious traveler. If questions ask for missing details, supply reasonable, "
            "low-drama specifics (timing, duration, prior self-care) that do not contradict the initial message."
        ),
    }
    payload: dict[str, Any] = {
        "model": model.strip(),
        "temperature": openai_chat_temperature_for_model(model.strip(), 0.35),
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Output only valid JSON with a single key traveler_reply (string). "
                    "The string is what the traveler types next in the chat (plain text). "
                    "Keep traveler_reply at most 800 characters."
                ),
            },
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
    }
    try:
        r = httpx_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json=payload,
            timeout=90.0,
        )
    except httpx.RequestError as e:
        return (
            "Thanks — no major change since my first message; happy to follow the conservative advice you outlined.",
            f"OpenAI user-simulator transport error: {e}",
        )
    if r.status_code != 200:
        return (
            "Thanks — no major change since my first message; happy to follow the conservative advice you outlined.",
            f"OpenAI user-simulator HTTP {r.status_code}: {(r.text or '')[:500]}",
        )
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else {}
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as e:
        return (
            "Thanks — no major change since my first message; happy to follow the conservative advice you outlined.",
            f"OpenAI user-simulator parse error: {e}",
        )
    reply = str((parsed or {}).get("traveler_reply") or "").strip()
    if not reply:
        return (
            "Thanks — no major change since my first message; happy to follow the conservative advice you outlined.",
            "OpenAI user-simulator returned empty traveler_reply.",
        )
    if len(reply) > _CHAT_TURN_MAX_CHARS:
        reply = reply[: _CHAT_TURN_MAX_CHARS]
    return reply, None


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark /api/assist across query_strategy values.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Assist API origin")
    p.add_argument(
        "--scenario",
        default="mild_lisbon",
        choices=sorted(SCENARIOS.keys()),
        help="Which fixed scenario body to send",
    )
    p.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategies to test (default: single_turn, multi_turn, multi_turn_with_tools)",
    )
    p.add_argument("--repeat", type=int, default=1, help="Runs per strategy (median time if >1)")
    p.add_argument("--warmup", type=int, default=0, help="Extra warmup POSTs (discarded) using first strategy")
    p.add_argument("--no-research", action="store_true", help="Set research_tools false on the scenario body")
    p.add_argument("--csv", type=Path, default=None, help="Append result rows to this CSV path")
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Base directory for per-strategy PDF reports (default: <repo>/evaluation_reports). Files: <scenario>/<strategy>.pdf",
    )
    p.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not write reasoning PDF reports (<strategy>.pdf under the report directory).",
    )
    p.add_argument(
        "--access-code",
        default="",
        help="App access code when APP_ACCESS_CODE is enabled (otherwise uses APP_ACCESS_CODE from env / .env).",
    )
    p.add_argument(
        "--multi-turn-rounds",
        type=int,
        default=2,
        metavar="N",
        help="Number of /api/assist calls to chain for multi_turn and multi_turn_with_tools (default 2). Between calls, "
        "this script calls OpenAI to simulate the traveler's follow-up. Ignored for other strategies.",
    )
    p.add_argument(
        "--simulator-model",
        default=os.getenv("EVAL_USER_SIMULATOR_MODEL", "gpt-4o-mini"),
        help="OpenAI model (on this machine) that writes simulated traveler replies between assist rounds.",
    )
    p.add_argument(
        "--simulator-openai-key",
        default="",
        help="API key for the user simulator only (defaults to OPENAI_API_KEY from env / .env).",
    )
    p.add_argument(
        "--no-relevance-llm",
        action="store_true",
        help="Skip the optional OpenAI 1–5 rubric for relevance/context (heuristic columns are still written).",
    )
    p.add_argument(
        "--relevance-llm-model",
        default="",
        help="OpenAI model for the relevance rubric (defaults to EVAL_RELEVANCE_MODEL env or --simulator-model).",
    )
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in strategies:
        if s not in STRATEGY_CHOICES:
            print(f"Unknown strategy: {s!r} (allowed: {STRATEGY_CHOICES})", file=sys.stderr)
            return 2

    base = args.base_url.rstrip("/")
    template = dict(SCENARIOS[args.scenario])
    if args.no_research:
        template["research_tools"] = False

    # Health check
    with httpx.Client() as client:
        try:
            hr = client.get(base + "/api/health", timeout=10.0)
            if hr.status_code != 200:
                print(f"Health check failed: {hr.status_code}", file=sys.stderr)
                return 1
        except httpx.RequestError as e:
            print(f"Cannot reach server at {base!r}: {e}", file=sys.stderr)
            print("Start the server: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000", file=sys.stderr)
            return 1

        access_code = (args.access_code or os.getenv("APP_ACCESS_CODE", "") or "").strip()
        if access_code:
            av = client.post(
                base + "/api/access/verify",
                json={"code": access_code},
                timeout=30.0,
            )
            if av.status_code != 200:
                print(
                    f"Access verify failed ({av.status_code}): {(av.text or '')[:300]}",
                    file=sys.stderr,
                )
                print("Set --access-code or APP_ACCESS_CODE to match the server.", file=sys.stderr)
                return 1

        raw_mtr = int(args.multi_turn_rounds)
        multi_turn_rounds = max(1, min(12, raw_mtr))
        if raw_mtr != multi_turn_rounds:
            print(
                f"Note: --multi-turn-rounds {raw_mtr} clamped to 1..12 (using {multi_turn_rounds}).",
                file=sys.stderr,
            )
        sim_model = (args.simulator_model or "gpt-4o-mini").strip()
        sim_key = (args.simulator_openai_key or os.getenv("OPENAI_API_KEY", "") or "").strip()
        needs_simulator = multi_turn_rounds > 1 and any(s in MULTI_TURN_STRATEGIES for s in strategies)
        if needs_simulator and not sim_key:
            print(
                "ERROR: --multi-turn-rounds > 1 requires an OpenAI API key on this machine to simulate traveler replies. "
                "Set OPENAI_API_KEY (or pass --simulator-openai-key).",
                file=sys.stderr,
            )
            return 2
        if multi_turn_rounds > 1 and not any(s in MULTI_TURN_STRATEGIES for s in strategies):
            print(
                "Note: --multi-turn-rounds > 1 only applies to multi_turn and multi_turn_with_tools.",
                file=sys.stderr,
            )

        first = strategies[0]
        for _ in range(max(0, args.warmup)):
            body = {**template, "query_strategy": first}
            run_assist(client, base, body)

        rows_out: list[dict[str, Any]] = []
        if args.no_reports:
            reasoning_report_dir: Path | None = None
        else:
            report_root = args.report_dir if args.report_dir is not None else ROOT / "evaluation_reports"
            reasoning_report_dir = Path(report_root) / args.scenario

        print(f"Base URL: {base}")
        print(f"Scenario: {args.scenario}")
        print(
            f"Strategies: {strategies}  repeat={args.repeat}  research_tools={template.get('research_tools')}  "
            f"multi_turn_rounds={multi_turn_rounds}  simulator_model={sim_model!r}  "
            f"relevance_llm={'off' if args.no_relevance_llm else 'on'}"
        )
        if reasoning_report_dir is not None:
            try:
                rel = reasoning_report_dir.relative_to(ROOT)
                print(f"Reasoning reports: {rel}/<strategy>.pdf")
            except ValueError:
                print(f"Reasoning reports: {reasoning_report_dir}/<strategy>.pdf")
        print()

        header = (
            f"{'strategy':<22} {'status':>5} {'t_s':>8} {'digest':>7} {'rtc':>4} {'rq':>4} "
            f"{'plans':>5} {'cost':>5} {'llm':>5} {'emerg':>5} {'R':>3} {'jdg':>4} {'safety':>8} {'cor':>4} {'urg':>3} {'notes'}"
        )
        print(header)
        print("-" * len(header))

        for strat in strategies:
            times: list[float] = []
            last_data: dict[str, Any] = {}
            last_status = 0
            last_mt_meta: dict[str, Any] | None = None
            num_assist_rounds = multi_turn_rounds if strat in MULTI_TURN_STRATEGIES else 1
            for _ in range(args.repeat):
                chat_history: list[dict[str, str]] = []
                total_elapsed = 0.0
                step_log: list[dict[str, Any]] = []
                qa_exchanges: list[dict[str, Any]] = []
                assist_calls_completed = 0
                for ridx in range(num_assist_rounds):
                    prior_user = None
                    if chat_history and chat_history[-1].get("role") == "user":
                        prior_user = str(chat_history[-1].get("content") or "")
                    body = {**template, "query_strategy": strat, "chat_history": chat_history}
                    status, elapsed, data = run_assist(client, base, body)
                    total_elapsed += elapsed
                    last_status = status
                    last_data = data if isinstance(data, dict) else {}
                    asst_plain_for_log = ""
                    q_for_log: list[str] = []
                    if status == 200:
                        asst_plain_for_log = assistant_turn_plain_text(last_data)
                        q_for_log = _questions_to_clarify_from_llm(last_data, max_items=24)
                    assist_entry: dict[str, Any] = {
                        "step": "assist",
                        "round_index": ridx + 1,
                        "http_status": status,
                        "elapsed_s": round(elapsed, 3),
                        "prior_traveler_message_in_chat_history": (
                            _cap_report_text(prior_user) if prior_user else None
                        ),
                        "questions_to_clarify": q_for_log,
                        "assistant_visible_reply": (
                            _cap_report_text(asst_plain_for_log) if status == 200 and asst_plain_for_log else None
                        ),
                        "assistant_visible_reply_chars": len(asst_plain_for_log) if status == 200 else None,
                    }
                    step_log.append(assist_entry)
                    if status == 200:
                        assist_calls_completed += 1
                    if status != 200:
                        break
                    if ridx < num_assist_rounds - 1:
                        asst_plain = asst_plain_for_log
                        if len(asst_plain) > _CHAT_TURN_MAX_CHARS:
                            asst_plain = asst_plain[: _CHAT_TURN_MAX_CHARS - 3] + "..."
                        chat_history = [*chat_history, {"role": "assistant", "content": asst_plain}]
                        q_lines = q_for_log[:12]
                        sim_reply, sim_warn = generate_simulated_user_reply_openai(
                            httpx_client=client,
                            openai_api_key=sim_key,
                            model=sim_model,
                            initial_message=str(template.get("message") or ""),
                            language=str(template.get("language") or "en"),
                            travel_location=str(template.get("location") or ""),
                            home_country=str(template.get("home_country") or ""),
                            assistant_plain=asst_plain,
                            clarifying_questions=q_lines,
                        )
                        qa_exchanges.append(
                            {
                                "after_assist_round": ridx + 1,
                                "clarifying_questions_from_model": list(q_lines),
                                "assistant_visible_reply_as_sent_to_chat": _cap_report_text(asst_plain),
                                "assistant_visible_reply_chars_sent": len(asst_plain),
                                "simulated_traveler_answer": _cap_report_text(sim_reply),
                                "simulated_traveler_answer_chars": len(sim_reply),
                                "simulator_warning": sim_warn,
                            }
                        )
                        step_log.append(
                            {
                                "step": "simulated_user",
                                "after_assist_round": ridx + 1,
                                "clarifying_questions_sent_to_simulator": list(q_lines),
                                "traveler_reply": _cap_report_text(sim_reply),
                                "traveler_reply_chars": len(sim_reply),
                                "simulator_warning": sim_warn,
                            }
                        )
                        if sim_warn:
                            print(f"  simulator warning: {sim_warn[:220]}", file=sys.stderr)
                        chat_history = [*chat_history, {"role": "user", "content": sim_reply}]
                times.append(total_elapsed)
                if strat in MULTI_TURN_STRATEGIES and num_assist_rounds > 1:
                    last_mt_meta = {
                        "requested_rounds": num_assist_rounds,
                        "completed_assist_calls_200": assist_calls_completed,
                        "simulator_model": sim_model,
                        "initial_traveler_message": str(template.get("message") or ""),
                        "qa_exchanges": qa_exchanges,
                        "steps": step_log,
                    }
            t_med = statistics.median(times) if times else 0.0
            m = _metrics(last_data)
            m["elapsed_s"] = round(t_med, 3)
            rel_h = _relevance_context_heuristics(template=template, data=last_data, strategy=strat)
            rel_llm_out: dict[str, Any] = {}
            if (
                not args.no_relevance_llm
                and sim_key
                and last_status == 200
                and isinstance(last_data.get("llm"), dict)
            ):
                rmod = (args.relevance_llm_model or os.getenv("EVAL_RELEVANCE_MODEL") or sim_model).strip()
                rel_llm_out = evaluate_relevance_context_llm(
                    httpx_client=client,
                    openai_api_key=sim_key,
                    model=rmod,
                    template=template,
                    data=last_data,
                    strategy=strat,
                )
            rel_report: dict[str, Any] = {**rel_h, **rel_llm_out}
            note_parts: list[str] = []
            if last_status != 200:
                note_parts.append(f"HTTP {last_status}")
            if m.get("llm_error"):
                note_parts.append("llm_error")
            if m.get("llm_skip"):
                note_parts.append(m["llm_skip"])
            if m.get("judge_status") in ("error", "parse_error") and m.get("judge_error"):
                note_parts.append("judge_err")
            if strat in MULTI_TURN_STRATEGIES and num_assist_rounds > 1:
                note_parts.append(f"assist×{assist_calls_completed}")
            notes = "; ".join(note_parts)[:52]

            st4, saf, cor4, ur3 = _judge_display_short(m)
            row = {
                "scenario": args.scenario,
                "strategy": strat,
                "status": last_status,
                "elapsed_s": m["elapsed_s"],
                "digest_chars": m["digest_chars"],
                "research_tool_calls": m["research_tool_calls"],
                "retrieval_queries_n": m["retrieval_queries_n"],
                "treatment_plans_n": m["treatment_plans_n"],
                "cost_table_blocks_n": m["cost_table_blocks_n"],
                "llm_present": m["llm_present"],
                "emergency": m["emergency"],
                "care_level": m.get("care_level"),
                "multi_turn_rounds_requested": num_assist_rounds if strat in MULTI_TURN_STRATEGIES else 1,
                "multi_turn_assist_calls_200": assist_calls_completed,
                "simulator_model": sim_model if strat in MULTI_TURN_STRATEGIES and num_assist_rounds > 1 else "",
                "judge_status": m.get("judge_status"),
                "judge_model": m.get("judge_model"),
                "judge_urgency_appropriate": m.get("judge_urgency_appropriate"),
                "judge_safety": m.get("judge_safety"),
                "judge_correctness": m.get("judge_correctness"),
                "judge_urgency_notes": m.get("judge_urgency_notes"),
                "judge_safety_notes": m.get("judge_safety_notes"),
                "judge_correctness_notes": m.get("judge_correctness_notes"),
                "judge_error": m.get("judge_error"),
                "rel_composite_0_100": rel_h.get("rel_composite_0_100"),
                "rel_situation_0_50": rel_h.get("rel_situation_0_50"),
                "rel_context_0_50": rel_h.get("rel_context_0_50"),
                "rel_mentions_location": rel_h.get("rel_mentions_location"),
                "rel_emergency_language_ok": rel_h.get("rel_emergency_language_ok"),
                "rel_has_destination_local_context": rel_h.get("rel_has_destination_local_context"),
                "rel_digest_chars": rel_h.get("rel_digest_chars"),
                "rel_time_or_local_language": rel_h.get("rel_time_or_local_language"),
                "rel_nearby_when_research": rel_h.get("rel_nearby_when_research"),
                "rel_llm_relevance_1_5": rel_llm_out.get("rel_llm_relevance_1_5", ""),
                "rel_llm_context_1_5": rel_llm_out.get("rel_llm_context_1_5", ""),
                "rel_llm_composite_0_100": rel_llm_out.get("rel_llm_composite_0_100", ""),
                "rel_llm_rationale": (rel_llm_out.get("rel_llm_rationale") or "")[:500],
                "rel_llm_error": (rel_llm_out.get("rel_llm_error") or "")[:300],
                "notes": notes,
            }
            rows_out.append(row)

            print(
                f"{strat:<22} {last_status:>5} {m['elapsed_s']:>8.3f} {m['digest_chars']:>7} "
                f"{m['research_tool_calls']:>4} {m['retrieval_queries_n']:>4} {m['treatment_plans_n']:>5} "
                f"{m['cost_table_blocks_n']:>5} {str(m['llm_present']):>5} {str(m['emergency']):>5} "
                f"{rel_h.get('rel_composite_0_100', 0):>3} {st4} {saf} {cor4} {ur3} {notes}"
            )

            if reasoning_report_dir is not None:
                report_path = reasoning_report_dir / f"{strat}.pdf"
                extra_pdf: dict[str, Any] = {"evaluation_relevance_context": rel_report}
                if last_mt_meta:
                    extra_pdf["evaluation_multi_turn_simulation"] = last_mt_meta
                payload = _reasoning_report_payload(
                    scenario=args.scenario,
                    strategy=strat,
                    http_status=last_status,
                    elapsed_s=float(m["elapsed_s"]),
                    data=last_data,
                    extra=extra_pdf,
                )
                _write_reasoning_report_pdf(report_path, payload)
                try:
                    shown = report_path.relative_to(ROOT)
                except ValueError:
                    shown = report_path
                print(f"  saved reasoning report -> {shown}")

        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            write_header = not args.csv.exists()
            with args.csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                if write_header:
                    w.writeheader()
                w.writerows(rows_out)
            print()
            print(f"Appended {len(rows_out)} row(s) to {args.csv}")

    print()
    print("Legend: t_s = median wall time; digest = research digest chars; rtc = research tool calls;")
    print("        rq = internal retrieval query count (single_turn_tools); plans/cost = LLM table sizes.")
    print(
        "        jdg/safety/cor/urg = LLM-as-judge (OPENAI_JUDGE_MODEL on server): status, recommendation_safety, "
        "overall_correctness (corr/part/inc), urgency_level_appropriate (Y/N/?). absent = judge disabled or no payload."
    )
    if args.csv:
        print(
            f"        CSV includes judge narrative columns (up to {_JUDGE_NOTES_CSV_MAX} chars each): "
            "judge_urgency_notes, judge_safety_notes, judge_correctness_notes."
        )
    if not args.no_reports:
        print(
            "        Each strategy also writes evaluation_reports/<scenario>/<strategy>.pdf (override with "
            "--report-dir, disable with --no-reports): readable sections for triage, multi-turn Q&A, LLM excerpts, judge, log."
        )
    print(
        "        multi_turn / multi_turn_with_tools: --multi-turn-rounds (default 2) chains assist calls; "
        "OPENAI_API_KEY on this machine drives the traveler-reply simulator (EVAL_USER_SIMULATOR_MODEL, --simulator-model)."
    )
    print(
        "        R = heuristic relevance+context score 0–100 (higher with tools+digest+local time context); "
        "CSV also has rel_* columns and optional rel_llm_* rubric (--no-relevance-llm to skip OpenAI rubric)."
    )
    return 0 if all(r["status"] == 200 for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
