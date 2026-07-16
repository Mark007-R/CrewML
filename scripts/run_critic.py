"""Day 10 — drive the FULL crew loop and record what the Critic decided.

    python scripts/run_critic.py [--dataset credit-g] [--all] [--no-search] [--no-llm]
                                 [--max-iterations 3]

This is the first script that runs the crew as a *feedback system*, not a straight
line. It compiles the LangGraph crew (Profiler → Planner → Feature Engineer → Trainer
→ **Critic** → …) and invokes it end-to-end: the Critic diagnoses each training pass,
decides *iterate* vs *finalize*, and — when it iterates — hands the Planner a specific
directive it acts on before the next pass. The loop is bounded by the Critic's own
convergence logic and, as a hard backstop, the ``max_iterations`` guard.

Every number here is a **cross-validated estimate on train**, never a held-out score
(``cv_score_is_holdout: false``) — the locked test split is untouched (final held-out
scoring is a later, Phase-3 step). If a live provider is unavailable the advisory
narratives degrade to ``unavailable`` and the deterministic diagnosis stands.

Outputs:
* ``results/day10_critiques.json`` — committed, reproducible: per dataset the node
  trace, iterations run vs. budget, and every Critic pass (decision, reason, findings,
  CV score + delta). No large streams.
* ``artifacts/crew/<dataset>/loop_run.json`` — git-ignored full final state for inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import ARTIFACTS_DIR, MAX_ITERATIONS, RESULTS_DIR, is_mock_mode
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY

COMMITTED_PATH = RESULTS_DIR / "day10_critiques.json"


def _run_one(key: str, *, max_iterations: int) -> tuple[dict, dict]:
    """Invoke the full compiled crew on one dataset; return (committable record, final state)."""
    spec = REGISTRY[key]
    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    final = app.invoke(state, config={"recursion_limit": limit})

    critiques = final.get("critiques") or []
    passes = [
        {
            "iteration": c.get("iteration"),
            "decision": c.get("decision"),
            "reason": c.get("reason"),
            "cv_score": c.get("cv_score"),
            "score_delta_vs_prev": c.get("score_delta_vs_prev"),
            "best_model": c.get("best_model"),
            "finding_codes": c.get("finding_codes"),
            "findings": c.get("findings"),
        }
        for c in critiques
    ]
    record = {
        "dataset_key": key,
        "task": spec.task,
        "subtype": spec.subtype,
        "metric": spec.metric,
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "trace": final.get("trace"),
        "final_cv_score": critiques[-1].get("cv_score") if critiques else None,
        "critic_passes": passes,
        "cv_score_is_holdout": False,
    }
    return record, final


def _summarise(rec: dict) -> str:
    n_planner = (rec["trace"] or []).count("planner")
    codes = sorted({c for p in rec["critic_passes"] for c in (p["finding_codes"] or [])})
    codes_str = ",".join(codes) if codes else "none"
    score = rec["final_cv_score"]
    score_str = f"{score:.4f}" if isinstance(score, float) else "-"
    return (
        f"{rec['dataset_key']:<10} passes={rec['iterations_run']}/{rec['max_iterations']} "
        f"planner_runs={n_planner} final={rec['final_decision']:<8} "
        f"cv {rec['metric']}={score_str} findings=[{codes_str}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-10 full crew loop (with the real Critic).")
    ap.add_argument("--dataset", default="credit-g", help="single dataset key (default: credit-g)")
    ap.add_argument("--all", action="store_true", help="run the whole benchmark suite")
    ap.add_argument("--no-search", action="store_true", help="skip grid search (CV at default params — faster)")
    ap.add_argument("--no-llm", action="store_true", help="disable all advisory LLM narratives (deterministic only)")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                    help="Critic-loop budget / hard backstop (default: config.MAX_ITERATIONS)")
    args = ap.parse_args()

    keys = list(REGISTRY) if args.all else [args.dataset]
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    if args.no_search:
        os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"
    if args.no_llm:
        for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
            os.environ[var] = "0"

    mode = "mock (no LLM)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[crew] Day 10 — full loop on {len(keys)} dataset(s), "
          f"max_iterations={args.max_iterations}, grid-search={not args.no_search}, narratives: {mode}")
    print("[crew] the Critic diagnoses each pass and decides iterate/finalize; scores are CV on train.", flush=True)

    records: dict[str, dict] = {}
    for k in keys:
        rec, final = _run_one(k, max_iterations=args.max_iterations)
        print("  " + _summarise(rec))
        records[k] = rec

        out_dir = ARTIFACTS_DIR / "crew" / k
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loop_run.json").write_text(json.dumps(final, indent=2, default=str))

    COMMITTED_PATH.write_text(json.dumps({"datasets": records}, indent=2, default=str))
    print(f"[crew] wrote loop results -> {COMMITTED_PATH}")
    print("[crew] all scores are CROSS-VALIDATED estimates on train (holdout untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
