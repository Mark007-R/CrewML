"""Day 4 — Baseline 2: the classical-AutoML ceiling (FLAML), per dataset.

Run after ``prepare_datasets.py``:

    python scripts/run_automl.py

For each dataset it fits FLAML on ``train`` (its own CV inside that split, under a
fixed per-dataset time budget) and scores it once on the LOCKED ``holdout`` via
:mod:`crewml.scoring` → ``results/automl_metrics.json``. The held-out SHA-256 seal
is re-verified after every dataset, proving the ceiling never peeked at or mutated
the holdout. No LLM is involved, so there is no mock mode here — the numbers are
always real (FLAML is a deterministic-per-seed classical system, time-budgeted).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn

from crewml.automl_baseline import AUTOML_SYSTEM, run_automl
from crewml.config import AUTOML_TIME_BUDGET_S, RESULTS_DIR, SEED
from crewml.datasets import (
    REGISTRY,
    holdout_path,
    load_holdout,
    load_manifest,
    load_train,
    train_path,
    verify_holdout_untouched,
)

AUTOML_METRICS_PATH = RESULTS_DIR / "automl_metrics.json"


def _positive_class(manifest: dict, key: str) -> str | None:
    return manifest["datasets"][key]["target"].get("positive_class")


def main() -> int:
    # FLAML/sklearn emit a lot of benign convergence + deprecation chatter.
    warnings.filterwarnings("ignore")

    manifest = load_manifest()
    budget = AUTOML_TIME_BUDGET_S
    print(f"[automl] FLAML ceiling — {budget}s/dataset time budget, seed={SEED}.", flush=True)

    datasets: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for key, spec in REGISTRY.items():
        if not (train_path(key).exists() and holdout_path(key).exists()):
            failures[key] = "not materialised — run scripts/prepare_datasets.py"
            print(f"[automl] {key}: SKIPPED — {failures[key]}", flush=True)
            continue

        print(f"[automl] {key} ({spec.metric}) — searching {budget}s ...", flush=True)
        try:
            train, holdout = load_train(key), load_holdout(key)
            res = run_automl(
                spec, train, holdout, _positive_class(manifest, key), budget
            )
            # Honesty check: the search+scoring must not have altered the seal.
            if not verify_holdout_untouched(key):
                raise RuntimeError("holdout seal broken after AutoML scoring")
            datasets[key] = res
            print(
                f"         -> {spec.metric}={res['value']:.4f} {res['secondary']} "
                f"(best={res['best_estimator']})",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — record and continue, never drop silently
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"         FAILED — {failures[key]}", flush=True)

    report = {
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "system": AUTOML_SYSTEM,
        "time_budget_s": budget,
        "note": (
            "Baseline 2: FLAML classical AutoML — the strong non-agent ceiling. "
            "Fit strictly on train (its own CV), scored once on the LOCKED holdout "
            "through the shared scorer. Time-budgeted, so reproducible in "
            "distribution rather than bit-for-bit; each entry records its "
            "best_estimator and budget. Higher is better for every metric."
        ),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "failures": failures,
    }
    AUTOML_METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[automl] wrote -> {AUTOML_METRICS_PATH}")
    print(f"[automl] {len(datasets)}/{len(REGISTRY)} datasets scored.")
    if failures:
        print(f"[automl] failures: {list(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
