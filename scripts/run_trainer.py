"""Day 9 — run the Feature Engineer + Trainer end-to-end and dump CV results.

    python scripts/run_trainer.py [--dataset credit-g] [--all] [--no-search] [--no-llm]

For each dataset this walks the front half of the crew for real: Profiler ->
Planner -> Feature Engineer -> Trainer. The Feature Engineer generates (and
sandbox-validates) ``add_features`` code; the Trainer assembles a training script,
runs it in the Day-6 sandboxed executor over the **train split only**, cross-validates
the planned candidates, refits the best, and saves the model.

Every number here is a **cross-validated estimate on train**, never a held-out score
(``cv_score_is_holdout: false``) — the locked test split is untouched (final held-out
scoring is a later, Phase-3 step).

Outputs:
* ``results/day09_training.json`` — committed, reproducible CV results (best model +
  per-candidate CV means, feature counts, label mapping). No large streams, no artifacts.
* ``artifacts/executor/<run_id>/artifacts/`` — git-ignored ``model.joblib`` +
  ``fe_source.py`` the run produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import RESULTS_DIR, is_mock_mode
from crewml.crew.feature_engineer import run_feature_engineer
from crewml.crew.planner import run_planner
from crewml.crew.profiler import run_profiler
from crewml.crew.trainer import run_trainer
from crewml.datasets import REGISTRY

COMMITTED_PATH = RESULTS_DIR / "day09_training.json"


def _run_one(key: str, *, param_search: bool, with_llm: bool | None) -> dict:
    """Profiler -> Planner -> FE -> Trainer for one dataset; return a committable record."""
    profile = run_profiler(key, with_llm=False)
    plan = run_planner(profile, with_llm=with_llm)
    fe = run_feature_engineer(plan, key, with_llm=with_llm)
    training = run_trainer(plan, fe["code"], key, param_search=param_search)

    m = training.get("metrics", {})
    fe_meta = fe["meta"]
    record = {
        "dataset_key": key,
        "task": plan["task"],
        "subtype": plan["subtype"],
        "metric": plan["metric"],
        "ok": training["ok"],
        "param_search": training["param_search"],
        "fe_source": fe_meta.get("source"),
        "fe_new_columns": (fe_meta.get("validation") or {}).get("new_columns"),
        "best_model": m.get("best_model"),
        "best_cv_score": m.get("best_cv_score"),
        "best_cv_std": m.get("best_cv_std"),
        "best_params": m.get("best_params"),
        "per_model": m.get("per_model"),
        "n_features_original": m.get("n_features_original"),
        "n_features_after_fe": m.get("n_features_after_fe"),
        "engineered_columns": m.get("engineered_columns"),
        "label_mapping": m.get("label_mapping"),
        "cv_score_is_holdout": False,
        "error": training.get("error"),
    }
    return record


def _summarise(rec: dict) -> str:
    fe = rec["fe_source"] or "-"
    if not rec["ok"]:
        return f"{rec['dataset_key']:<10} FAILED — {rec['error']}"
    return (
        f"{rec['dataset_key']:<10} best={rec['best_model']:<22} "
        f"cv {rec['metric']}={rec['best_cv_score']:.4f} "
        f"(+/-{rec['best_cv_std']:.4f}) fe={fe:<8} "
        f"feats {rec['n_features_original']}->{rec['n_features_after_fe']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-9 Feature Engineer + Trainer.")
    ap.add_argument("--dataset", default="credit-g", help="single dataset key (default: credit-g)")
    ap.add_argument("--all", action="store_true", help="run the whole benchmark suite")
    ap.add_argument("--no-search", action="store_true", help="skip grid search (CV at default params — faster)")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM Feature Engineer (deterministic default only)")
    args = ap.parse_args()

    keys = list(REGISTRY) if args.all else [args.dataset]
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    with_llm = False if args.no_llm else None
    param_search = not args.no_search
    mode = "mock (no LLM)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[trainer] Day 9 — {len(keys)} dataset(s), grid-search={param_search}, FE narrative: {mode}")

    records: dict[str, dict] = {}
    for k in keys:
        rec = _run_one(k, param_search=param_search, with_llm=with_llm)
        print("  " + _summarise(rec))
        records[k] = rec

    COMMITTED_PATH.write_text(json.dumps({"datasets": records}, indent=2, default=str))
    print(f"[trainer] wrote CV results -> {COMMITTED_PATH}")
    print("[trainer] all scores are CROSS-VALIDATED estimates on train (holdout untouched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
