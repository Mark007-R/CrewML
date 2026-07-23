"""Day 14 — per-agent ablations. What do the Planner and the Feature Engineer earn?

    python scripts/run_agent_ablation.py [--datasets credit-g,diabetes]
                                         [--max-iterations 3]
                                         [--no-search] [--no-llm] [--table-only]

Runs, per the design in :mod:`crewml.agent_ablation`, three paired arms per dataset —
the full crew (a fresh same-session reference), ``no_planner`` (profile-blind naive
plan) and ``no_feature_engineer`` (identity transform) — and attributes each agent's
contribution as ``full − ablated`` on the LOCKED holdout, scored outside the graph via
:mod:`crewml.holdout_eval` with the seal re-verified per run (EVAL_PROTOCOL.md §3).

Outputs (committed): ``results/day14_agent_ablation.{json,md}`` and
``results/charts/day14_agent_ablation.png``. ``--table-only`` re-renders the board +
chart from the committed JSON without running the crew.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The board renders real typography (−, —, R²) a cp1252 Windows console cannot encode;
# degrade those characters rather than kill a completed run on a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from crewml import charts
from crewml.agent_ablation import (
    AGENT_ABLATION_RESULT_PATH,
    assemble_report,
    render_markdown,
    run_agent_ablation,
    write_report,
)
from crewml.config import MAX_ITERATIONS, is_mock_mode
from crewml.datasets import REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 14 — per-agent ablations (Planner / Feature Engineer).")
    ap.add_argument("--datasets", default="", help="subset to run (default: all 5)")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--no-search", action="store_true", help="skip grid search (faster)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic core only, no narratives")
    ap.add_argument("--table-only", action="store_true",
                    help="re-render board + chart from committed JSON; no crew runs")
    args = ap.parse_args()

    if args.table_only:
        report = json.loads(AGENT_ABLATION_RESULT_PATH.read_text())
        write_report(report)
        chart = charts.plot_agent_ablation(report)
        print(render_markdown(report))
        print(f"[day14] chart -> {chart}")
        return 0

    if args.no_search:
        os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"
    if args.no_llm:
        for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
            os.environ[var] = "0"

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()] or list(REGISTRY)
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    mode = "MOCK (no LLM key)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[day14] per-agent ablations on {len(keys)} dataset(s) x 3 arms; "
          f"max_iterations={args.max_iterations}; narratives: {mode}")
    if is_mock_mode():
        print("[day14] WARNING: mock mode — these are NOT real LLM numbers.", flush=True)

    study = run_agent_ablation(keys, max_iterations=args.max_iterations,
                               progress=lambda m: print(m, flush=True))
    report = assemble_report(study)
    write_report(report)
    chart = charts.plot_agent_ablation(report)

    print("\n" + render_markdown(report))
    print(f"[day14] wrote agent-ablation report -> {AGENT_ABLATION_RESULT_PATH}")
    print(f"[day14] chart -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
