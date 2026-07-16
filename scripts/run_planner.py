"""Day 8 — run the Planner agent on the benchmark suite and dump ModelingPlans.

    python scripts/run_planner.py [--dataset diabetes] [--no-llm]

The Planner is the crew's second REAL node. For each dataset it first computes the
train-only DataProfile (via the Profiler), then reasons over that profile — never
the data — to produce a ModelingPlan: column drops, dtype-aware preprocessing,
candidate model families with seed grids, the CV scheme, and the imbalance strategy.
Unless ``--no-llm`` or mock mode, it layers a short advisory LLM refinement note for
the Feature Engineer + Trainer on top.

Outputs:
* ``results/day08_plans.json`` — the DETERMINISTIC plans only (LLM narrative
  stripped), committed as reproducible evidence.
* ``artifacts/crew/<key>/plan.json`` — the full plan incl. the LLM narrative
  (git-ignored; narratives are advisory and provider-specific).

Never touches the locked held-out split — the Planner reads a profile dict and the
Profiler under it only ever loads ``train``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import ARTIFACTS_DIR, RESULTS_DIR, is_mock_mode
from crewml.crew.planner import run_planner
from crewml.crew.profiler import run_profiler
from crewml.datasets import REGISTRY

COMMITTED_PATH = RESULTS_DIR / "day08_plans.json"


def _summarise(plan: dict) -> str:
    """A one-line human summary of a plan for the console."""
    pre = plan["preprocessing"]
    cv = plan["cv"]
    imb = "imb" if plan["imbalance_strategy"]["recommended"] else "-"
    models = ",".join(m["name"] for m in plan["candidate_models"])
    return (
        f"{plan['dataset_key']:<10} drop={len(plan['drop_columns']):<2} "
        f"num={len(pre['numeric']['columns']):<3} cat={len(pre['categorical']['columns']):<3} "
        f"cv={cv['scheme']}({cv['n_splits']})/{cv['scoring']:<8} {imb:<4} models=[{models}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-8 Planner on the benchmark suite.")
    ap.add_argument("--dataset", default=None, help="single dataset key (default: all)")
    ap.add_argument("--no-llm", action="store_true", help="skip the advisory LLM narrative")
    args = ap.parse_args()

    keys = [args.dataset] if args.dataset else list(REGISTRY)
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    with_llm = False if args.no_llm else None  # None => env/mock-aware default
    mode = "mock (no LLM)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[planner] Day 8 — planning {len(keys)} dataset(s), narrative: {mode}")

    committed: dict[str, dict] = {}
    for k in keys:
        # The Planner reasons over the profile; the profile's own advisory narrative is
        # irrelevant here, so build it without one (deterministic facts only).
        profile = run_profiler(k, with_llm=False)
        plan = run_planner(profile, with_llm=with_llm)
        print("  " + _summarise(plan))

        # Full plan (with narrative) -> git-ignored artifacts.
        art_dir = ARTIFACTS_DIR / "crew" / k
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "plan.json").write_text(json.dumps(plan, indent=2, default=str))

        # Deterministic-only copy -> committed results (reproducible).
        deterministic = {kk: vv for kk, vv in plan.items() if kk != "llm_narrative"}
        committed[k] = deterministic

    COMMITTED_PATH.write_text(json.dumps({"datasets": committed}, indent=2, default=str))
    print(f"[planner] wrote deterministic plans -> {COMMITTED_PATH}")
    print(f"[planner] full plans (with narrative) -> {ARTIFACTS_DIR / 'crew' / '<key>' / 'plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
