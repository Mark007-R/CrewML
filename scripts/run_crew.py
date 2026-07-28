"""Day 11 — drive the FULL crew end-to-end, every node real, raw dataset to model card.

    python scripts/run_crew.py [--dataset credit-g] [--all] [--no-search] [--no-llm]
                               [--max-iterations 3]

This is the Phase-2 finale: the first run where all seven nodes are real specialists —
Profiler → Planner → Feature Engineer → Trainer → Critic → **Ensembler** → **Reporter**.
The crew takes a raw tabular dataset and produces a trained, cross-validated model plus
a written report and a ``MODEL_CARD.md`` on its own. The Ensembler combines the top
candidates and keeps the ensemble only when it beats the single best on CV; the Reporter
synthesises the whole run into a model card.

Every number here is a **cross-validated estimate on train**, never a held-out score
(``cv_score_is_holdout: false``) — the locked test split is untouched (final held-out
scoring is a later, Phase-3 step). If a live provider is unavailable the advisory
narratives degrade to ``unavailable`` and the deterministic core stands.

Outputs:
* ``results/day11_crew_run.json`` — committed, reproducible: per dataset the final model
  (ensemble vs. single + both CV scores), the node trace, the Critic passes, and the
  honesty warnings. No large streams.
* ``results/sample_model_card.md`` — committed: the Reporter's model card for the
  primary dataset, so the deliverable is inspectable in the repo.
* ``artifacts/crew/<dataset>/{final_run.json, MODEL_CARD.md, report.json}`` — git-ignored
  full state + per-run card for inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml import budget as budget_mod
from crewml import manifest as manifest_mod
from crewml.config import ARTIFACTS_DIR, MAX_ITERATIONS, RESULTS_DIR, is_mock_mode
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY, verify_holdout_untouched

COMMITTED_PATH = RESULTS_DIR / "day11_crew_run.json"
SAMPLE_CARD_PATH = RESULTS_DIR / "sample_model_card.md"


def _run_one(key: str, *, max_iterations: int,
             token_budget: int | None = None,
             time_budget_s: float | None = None) -> tuple[dict, dict]:
    """Invoke the full compiled crew on one dataset; return (committable record, final state).

    Each dataset runs under its own fresh run budget (Day 21) — token/wall-clock
    caps default to config (CREWML_RUN_TOKEN_BUDGET / CREWML_RUN_TIME_BUDGET_S).
    """
    spec = REGISTRY[key]
    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    with budget_mod.run_budget(token_budget, time_budget_s):
        final = app.invoke(state, config={"recursion_limit": limit})

    report = final.get("report") or {}
    ensemble = final.get("ensemble") or {}
    final_model = report.get("final_model") or {}
    critiques = final.get("critiques") or []

    record = {
        "dataset_key": key,
        "task": spec.task,
        "subtype": spec.subtype,
        "metric": spec.metric,
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "trace": final.get("trace"),
        "final_model": {
            "kind": final_model.get("kind"),
            "chosen": final_model.get("chosen"),
            "members": final_model.get("members"),
            "single_best_model": final_model.get("single_best_model"),
            "final_cv_score": final_model.get("cv_score"),
            "ensemble_cv_score": final_model.get("ensemble_cv_score"),
            "single_best_cv_score": final_model.get("single_best_cv_score"),
            "improvement_over_single": final_model.get("improvement_over_single"),
        },
        "ensemble_attempted": ensemble.get("attempted"),
        "holdout_untouched": verify_holdout_untouched(key),
        "warnings": report.get("warnings"),
        "run_budget": report.get("run_budget"),
        "cv_score_is_holdout": False,
    }
    return record, final


def _summarise(rec: dict) -> str:
    fm = rec["final_model"]
    score = fm.get("final_cv_score")
    score_str = f"{score:.4f}" if isinstance(score, float) else "-"
    kind = fm.get("kind") or "-"
    if kind == "ensemble":
        gain = fm.get("improvement_over_single")
        extra = f"ensemble(+{gain:.4f} vs {fm.get('single_best_model')})" if isinstance(gain, float) else "ensemble"
    else:
        extra = f"single({fm.get('single_best_model')})"
    return (
        f"{rec['dataset_key']:<10} passes={rec['iterations_run']}/{rec['max_iterations']} "
        f"final={kind:<8} cv {rec['metric']}={score_str} [{extra}] "
        f"holdout_sealed={rec['holdout_untouched']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-11 full crew end-to-end (every node real).")
    ap.add_argument("--dataset", default="credit-g", help="single dataset key (default: credit-g)")
    ap.add_argument("--all", action="store_true", help="run the whole benchmark suite")
    ap.add_argument("--no-search", action="store_true", help="skip grid search (CV at default params — faster)")
    ap.add_argument("--no-llm", action="store_true", help="disable all advisory LLM narratives (deterministic only)")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                    help="Critic-loop budget / hard backstop (default: config.MAX_ITERATIONS)")
    ap.add_argument("--token-budget", type=int, default=None,
                    help="per-run LLM token cap (default: CREWML_RUN_TOKEN_BUDGET; <=0 uncapped)")
    ap.add_argument("--time-budget-s", type=float, default=None,
                    help="per-run wall-clock cap in seconds (default: CREWML_RUN_TIME_BUDGET_S; <=0 uncapped)")
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
    print(f"[crew] Day 11 — FULL crew end-to-end on {len(keys)} dataset(s), every node real; "
          f"max_iterations={args.max_iterations}, grid-search={not args.no_search}, narratives: {mode}")
    print("[crew] Profiler->Planner->FE->Trainer->Critic->Ensembler->Reporter; scores are CV on train.", flush=True)

    records: dict[str, dict] = {}
    for k in keys:
        rec, final = _run_one(k, max_iterations=args.max_iterations,
                              token_budget=args.token_budget,
                              time_budget_s=args.time_budget_s)
        print("  " + _summarise(rec))
        records[k] = rec

        out_dir = ARTIFACTS_DIR / "crew" / k
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_run.json").write_text(json.dumps(final, indent=2, default=str))
        # Day 23: pin + fingerprint the run so a re-run is checkable, not anecdotal.
        m = manifest_mod.write_run_manifest(final, out_dir / "run_manifest.json")
        rec["result_fingerprint"] = m["result_fingerprint"]

        # Commit the primary dataset's model card as an inspectable sample deliverable.
        if k == keys[0]:
            card = (final.get("report") or {}).get("model_card_markdown")
            if card:
                SAMPLE_CARD_PATH.write_text(card, encoding="utf-8")

    COMMITTED_PATH.write_text(json.dumps({"datasets": records}, indent=2, default=str))
    print(f"[crew] wrote crew results -> {COMMITTED_PATH}")
    print(f"[crew] wrote sample model card -> {SAMPLE_CARD_PATH}")
    print("[crew] all scores are CROSS-VALIDATED estimates on train (holdout untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
