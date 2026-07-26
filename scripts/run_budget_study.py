"""Day 21 — run the budget study: three crew runs under three budget regimes.

    python scripts/run_budget_study.py [--max-iterations 3]

Probes the provider (a real 8-token call — a configured key can be dead and look
live), then runs the full crew on credit-g under config-default caps, a tight
token cap, and a tight wall-clock cap, recording what the run-budget ledger
(Day 21, ``crewml.budget``) actually did: tokens charged per agent, calls
refused, and whether the Critic stopped the loop on budget grounds.

Grid search is disabled for all scenarios so wall-clock differences come from
LLM calls and CV, not the search grid (recorded in the study notes).

Outputs (committed):
* ``results/day21_budget_study.json`` — the full record.
* ``results/day21_budget_study.md`` — the table (registered in the artifact
  registry, so it can never silently drift from the JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Disabled BEFORE crewml imports read the env: same setting for every scenario.
os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"

from crewml import budget_study
from crewml.config import MAX_ITERATIONS


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-21 budget study (three crew runs).")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = ap.parse_args()

    print("[budget] Day 21 — probing the provider, then 3 crew runs on "
          f"{budget_study.DATASET}: reference / tight_tokens / tight_time", flush=True)

    data = budget_study.run_study(max_iterations=args.max_iterations)

    probe = data["probe"]
    print(f"[budget] provider live={probe.get('live')} "
          f"({probe.get('provider')}/{probe.get('model')})", flush=True)
    for s in data["scenarios"]:
        print(f"  {s['scenario']:<13} spent={s.get('tokens_spent') or 0:>6} tok "
              f"calls={s.get('n_calls') or 0}/{s.get('n_refused') or 0} refused "
              f"wall={s.get('crew_seconds')}s passes={s.get('iterations_run')} "
              f"decision={s.get('final_decision')} "
              f"loop_budget_stop={s.get('budget_stopped_loop')}", flush=True)

    budget_study.BUDGET_STUDY_PATH.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    budget_study.BUDGET_STUDY_MD_PATH.write_text(
        budget_study.render_markdown(data), encoding="utf-8")
    print(f"[budget] wrote {budget_study.BUDGET_STUDY_PATH}")
    print(f"[budget] wrote {budget_study.BUDGET_STUDY_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
