"""Day 5 — drive the Phase-2 crew SKELETON end-to-end on one dataset.

    python scripts/run_crew.py [--dataset credit-g] [--max-iterations 3]

This compiles the LangGraph crew and invokes it with the stub nodes in place. It
does NOT model anything yet — no LLM call, no data read, no held-out scoring — so
there is nothing here that could be mistaken for a real result. Its purpose is to
prove the wiring: the linear front half runs, the Critic loop iterates up to the
budget, the ``max_iterations`` guard finalises, and the run terminates at the
Reporter. The stub Critic always asks to iterate, so a skeleton run always spends
its full iteration budget before finalising.

Output: the node-visit trace + terminal state summary to stdout, and a
JSON artifact under ``artifacts/crew/<dataset>/skeleton_run.json`` (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import ARTIFACTS_DIR, MAX_ITERATIONS
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY


def run_skeleton(dataset: str, max_iterations: int) -> dict:
    """Invoke the compiled crew skeleton on one dataset and return the final state."""
    if dataset not in REGISTRY:
        raise SystemExit(f"unknown dataset {dataset!r}; choose from {list(REGISTRY)}")
    spec = REGISTRY[dataset]
    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    # recursion_limit comfortably clears the worst case
    # (profiler + max_iterations*(planner,fe,trainer,critic) + ensembler + reporter).
    limit = 3 + max_iterations * 4 + 10
    final = app.invoke(state, config={"recursion_limit": limit})
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the CrewML Phase-2 skeleton on one dataset.")
    ap.add_argument("--dataset", default="credit-g", help="dataset key from the registry")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                    help="Critic-loop budget (default: config.MAX_ITERATIONS)")
    args = ap.parse_args()

    print(f"[crew] SKELETON run — dataset={args.dataset} max_iterations={args.max_iterations}")
    print("[crew] stub nodes only: no LLM, no data, no scoring (Day 5 wiring proof).", flush=True)

    final = run_skeleton(args.dataset, args.max_iterations)

    trace = final.get("trace", [])
    print(f"\n[crew] node trace ({len(trace)} steps):")
    print("       " + " -> ".join(trace))
    print(f"[crew] Critic passes run: {final.get('iteration')} / {final.get('max_iterations')} (budget)")
    print(f"[crew] critiques recorded: {len(final.get('critiques') or [])}")
    print(f"[crew] terminal report present: {final.get('report') is not None}")

    out_dir = ARTIFACTS_DIR / "crew" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skeleton_run.json"
    # State is JSON-friendly by construction; dump the whole thing for inspection.
    out_path.write_text(json.dumps(final, indent=2, default=str))
    print(f"[crew] wrote skeleton run -> {out_path}")

    ok = final.get("report") is not None and trace and trace[-1] == "reporter"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
