#!/usr/bin/env python3
"""
Compare Travel Care /api/assist performance across query_strategy values.

Requires the API server running (e.g. uvicorn app.main:app). Loads no secrets itself;
OpenAI / Serper / Maps keys must be configured on the server. LLM judge columns reflect
the server's OPENAI_JUDGE_MODEL (or are absent if the judge is disabled / failed).

Usage:
  cd /path/to/travelcareAI
  python scripts/compare_strategy_performance.py
  python scripts/compare_strategy_performance.py --base-url http://127.0.0.1:8000 --scenario tokyo_heat --repeat 2
  python scripts/compare_strategy_performance.py --strategies single_turn,single_turn_tools --no-research --csv /tmp/out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent

STRATEGY_CHOICES = (
    "single_turn",
    "single_turn_tools",
    "multi_turn",
    "multi_turn_with_tools",
    "unsolvable",
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
    base.update(
        {
            "judge_status": "ok",
            "judge_model": str(raw.get("evaluator_model") or "")[:120],
            "judge_urgency_appropriate": urg,
            "judge_safety": str(raw.get("recommendation_safety") or ""),
            "judge_correctness": str(raw.get("overall_correctness") or ""),
        }
    )
    return base


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
        default=",".join(STRATEGY_CHOICES),
        help="Comma-separated strategies to test",
    )
    p.add_argument("--repeat", type=int, default=1, help="Runs per strategy (median time if >1)")
    p.add_argument("--warmup", type=int, default=0, help="Extra warmup POSTs (discarded) using first strategy")
    p.add_argument("--no-research", action="store_true", help="Set research_tools false on the scenario body")
    p.add_argument("--csv", type=Path, default=None, help="Append result rows to this CSV path")
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

        first = strategies[0]
        for _ in range(max(0, args.warmup)):
            body = {**template, "query_strategy": first}
            run_assist(client, base, body)

        rows_out: list[dict[str, Any]] = []
        print(f"Base URL: {base}")
        print(f"Scenario: {args.scenario}")
        print(f"Strategies: {strategies}  repeat={args.repeat}  research_tools={template.get('research_tools')}")
        print()

        header = (
            f"{'strategy':<22} {'status':>5} {'t_s':>8} {'digest':>7} {'rtc':>4} {'rq':>4} "
            f"{'plans':>5} {'cost':>5} {'llm':>5} {'emerg':>5} {'jdg':>4} {'safety':>8} {'cor':>4} {'urg':>3} {'notes'}"
        )
        print(header)
        print("-" * len(header))

        for strat in strategies:
            times: list[float] = []
            last_data: dict[str, Any] = {}
            last_status = 0
            for _ in range(args.repeat):
                body = {**template, "query_strategy": strat}
                status, elapsed, data = run_assist(client, base, body)
                last_status = status
                last_data = data
                times.append(elapsed)
            t_med = statistics.median(times) if times else 0.0
            m = _metrics(last_data)
            m["elapsed_s"] = round(t_med, 3)
            note_parts: list[str] = []
            if last_status != 200:
                note_parts.append(f"HTTP {last_status}")
            if m.get("llm_error"):
                note_parts.append("llm_error")
            if m.get("llm_skip"):
                note_parts.append(m["llm_skip"])
            if m.get("judge_status") in ("error", "parse_error") and m.get("judge_error"):
                note_parts.append("judge_err")
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
                "judge_status": m.get("judge_status"),
                "judge_model": m.get("judge_model"),
                "judge_urgency_appropriate": m.get("judge_urgency_appropriate"),
                "judge_safety": m.get("judge_safety"),
                "judge_correctness": m.get("judge_correctness"),
                "judge_error": m.get("judge_error"),
                "notes": notes,
            }
            rows_out.append(row)

            print(
                f"{strat:<22} {last_status:>5} {m['elapsed_s']:>8.3f} {m['digest_chars']:>7} "
                f"{m['research_tool_calls']:>4} {m['retrieval_queries_n']:>4} {m['treatment_plans_n']:>5} "
                f"{m['cost_table_blocks_n']:>5} {str(m['llm_present']):>5} {str(m['emergency']):>5} "
                f"{st4} {saf} {cor4} {ur3} {notes}"
            )

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
    return 0 if all(r["status"] == 200 for r in rows_out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
