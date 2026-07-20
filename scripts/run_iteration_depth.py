"""Day 15 — the iteration-depth study: score vs Critic-loop budget, and the cost of each loop.

    python scripts/run_iteration_depth.py [--datasets credit-g,diabetes]
                                          [--probe-datasets kin8nm,cpu_small]
                                          [--natural-depths 1,3] [--probe-depths 1,2,3,4]
                                          [--no-search] [--no-llm] [--table-only]

Runs, per the design in :mod:`crewml.iteration_depth`, two arms — the natural sweep
(real datasets; the prediction is a flat curve, measured not asserted) and the
deficiency sweep (Day-13 handicap; the loop must recover, so depth becomes observable) —
every point holdout-scored outside the graph with the seal re-verified per run
(EVAL_PROTOCOL.md §3).

Outputs (committed): ``results/day15_iteration_depth.{json,md}`` and
``results/charts/day15_iteration_depth.png``. ``--table-only`` re-renders the board +
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
from crewml.config import is_mock_mode
from crewml.datasets import REGISTRY
from crewml.iteration_depth import (
    ITERATION_DEPTH_RESULT_PATH,
    NATURAL_DEPTHS,
    PROBE_DEPTHS,
    PROBE_KEYS,
    assemble_report,
    render_markdown,
    run_depth_study,
    write_report,
)


def _depths(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not raw.strip():
        return default
    return tuple(int(x) for x in raw.split(",") if x.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 15 — iteration-depth study (score vs Critic-loop budget).")
    ap.add_argument("--datasets", default="", help="natural-arm subset (default: all 5)")
    ap.add_argument("--probe-datasets", default=",".join(PROBE_KEYS),
                    help="deficiency-arm subset (default: Day 13's probe scope)")
    ap.add_argument("--natural-depths", default="", help=f"budgets for the natural arm (default: {NATURAL_DEPTHS})")
    ap.add_argument("--probe-depths", default="", help=f"budgets for the deficiency arm (default: {PROBE_DEPTHS})")
    ap.add_argument("--no-search", action="store_true", help="skip grid search (faster)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic core only, no narratives")
    ap.add_argument("--table-only", action="store_true",
                    help="re-render board + chart from committed JSON; no crew runs")
    args = ap.parse_args()

    if args.table_only:
        report = json.loads(ITERATION_DEPTH_RESULT_PATH.read_text())
        write_report(report)
        chart = charts.plot_iteration_depth(report)
        print(render_markdown(report))
        print(f"[day15] chart -> {chart}")
        return 0

    if args.no_search:
        os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"
    if args.no_llm:
        for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
            os.environ[var] = "0"

    natural_keys = [k.strip() for k in args.datasets.split(",") if k.strip()] or list(REGISTRY)
    probe_keys = [k.strip() for k in args.probe_datasets.split(",") if k.strip()]
    for k in natural_keys + probe_keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    natural_depths = _depths(args.natural_depths, NATURAL_DEPTHS)
    probe_depths = _depths(args.probe_depths, PROBE_DEPTHS)

    mode = "MOCK (no LLM key)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    n_runs = len(natural_keys) * len(natural_depths) + len(probe_keys) * len(probe_depths)
    print(f"[day15] iteration-depth study: natural {natural_depths} x {len(natural_keys)} dataset(s), "
          f"deficiency {probe_depths} x {len(probe_keys)} dataset(s) = {n_runs} crew runs; narratives: {mode}")
    if is_mock_mode():
        print("[day15] WARNING: mock mode — these are NOT real LLM numbers.", flush=True)

    study = run_depth_study(
        natural_keys, probe_keys,
        natural_depths=natural_depths,
        probe_depths=probe_depths,
        progress=lambda m: print(m, flush=True),
    )
    report = assemble_report(study)
    write_report(report)
    chart = charts.plot_iteration_depth(report)

    print("\n" + render_markdown(report))
    print(f"[day15] wrote iteration-depth report -> {ITERATION_DEPTH_RESULT_PATH}")
    print(f"[day15] chart -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
