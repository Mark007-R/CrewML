"""Day 13 — the Critic-loop ablation. Does the feedback loop earn its keep?

    python scripts/run_critic_ablation.py [--datasets credit-g,diabetes]
                                          [--probe-datasets kin8nm,cpu_small]
                                          [--no-probe] [--max-iterations 3]
                                          [--no-search] [--no-llm] [--table-only]

Runs, per the design in :mod:`crewml.ablation`, two studies and writes one board:

1. **Natural ablation** — the full crew (Critic loop) vs the ``no_critic`` variant on the
   real datasets. The loop is a conditional safeguard, so on healthy data it correctly
   finalises on pass 1 and the drop is ~0. That is the honest finding, not a bug.
2. **Forced-deficiency probe** — the same pair with a deliberately crippled first pass so
   the Critic's underfit finding fires; here the loop recovers the score and the ablated
   variant ships the stump, so the gap is the loop's real contribution.

Both studies score every model on the LOCKED holdout via :mod:`crewml.holdout_eval`, outside
the graph, with the seal re-verified per run (EVAL_PROTOCOL.md §3).

Outputs (committed): ``results/day13_critic_ablation.{json,md}`` and
``results/charts/day13_critic_ablation.png``. ``--table-only`` re-renders the board + chart
from the committed JSON without running the crew.
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
from crewml.ablation import (
    ABLATION_RESULT_PATH,
    assemble_report,
    render_markdown,
    run_deficiency_probe,
    run_natural_ablation,
    write_report,
)
from crewml.config import MAX_ITERATIONS, is_mock_mode
from crewml.datasets import REGISTRY

# The regression sets trip the Critic's underfit floor (R² ≤ 0.10) most cleanly under the
# handicap, so they are the default probe subset — the demonstration is unambiguous there.
DEFAULT_PROBE = ["kin8nm", "cpu_small"]


def _validate(keys: list[str], what: str) -> list[str]:
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown {what} dataset {k!r}; choose from {list(REGISTRY)}")
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 13 — Critic-loop ablation.")
    ap.add_argument("--datasets", default="", help="natural-ablation subset (default: all 5)")
    ap.add_argument("--probe-datasets", default="",
                    help=f"forced-deficiency subset (default: {','.join(DEFAULT_PROBE)})")
    ap.add_argument("--no-probe", action="store_true", help="skip the forced-deficiency study")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--no-search", action="store_true", help="skip grid search (faster)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic core only, no narratives")
    ap.add_argument("--table-only", action="store_true",
                    help="re-render board + chart from committed JSON; no crew runs")
    args = ap.parse_args()

    if args.table_only:
        report = json.loads(ABLATION_RESULT_PATH.read_text())
        write_report(report)
        chart = charts.plot_critic_ablation(report)
        print(render_markdown(report))
        print(f"[day13] chart -> {chart}")
        return 0

    if args.no_search:
        os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"
    if args.no_llm:
        for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
            os.environ[var] = "0"

    keys = _validate([k.strip() for k in args.datasets.split(",") if k.strip()] or list(REGISTRY), "dataset")
    probe_keys = (
        []
        if args.no_probe
        else _validate(
            [k.strip() for k in args.probe_datasets.split(",") if k.strip()] or DEFAULT_PROBE,
            "probe",
        )
    )

    mode = "MOCK (no LLM key)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[day13] Critic-loop ablation — natural on {len(keys)} dataset(s), "
          f"probe on {len(probe_keys)}; max_iterations={args.max_iterations}; narratives: {mode}")
    if is_mock_mode():
        print("[day13] WARNING: mock mode — these are NOT real LLM numbers.", flush=True)

    natural = run_natural_ablation(keys, max_iterations=args.max_iterations, progress=lambda m: print(m, flush=True))
    probe = (
        run_deficiency_probe(probe_keys, max_iterations=args.max_iterations, progress=lambda m: print(m, flush=True))
        if probe_keys
        else {}
    )

    report = assemble_report(natural, probe)
    write_report(report)
    chart = charts.plot_critic_ablation(report)

    print("\n" + render_markdown(report))
    print(f"[day13] wrote ablation report -> {ABLATION_RESULT_PATH}")
    print(f"[day13] chart -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
