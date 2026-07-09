"""Day 2 — compute Baseline 0 (Dummy + default RandomForest) for every dataset.

Run after ``prepare_datasets.py`` (idempotent):

    python scripts/run_baselines.py

For each dataset it fits both baseline systems on ``train`` and scores them once
on the LOCKED ``holdout`` through :mod:`crewml.scoring`, then writes
``results/baseline_metrics.json`` — the floor (Dummy) and the default-model
anchor (RandomForest) that the solo agent (Day 3), AutoML (Day 4) and the crew
(Phase 2) must beat.

The held-out SHA-256 seal is verified *after* scoring for every dataset, proving
the baselines neither peeked at nor mutated the holdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn

from crewml.baselines import BASELINE_SYSTEMS, fit_score_baseline
from crewml.config import RESULTS_DIR, SEED
from crewml.datasets import (
    REGISTRY,
    holdout_path,
    load_holdout,
    load_manifest,
    load_train,
    train_path,
    verify_holdout_untouched,
)

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"


def _positive_class(manifest: dict, key: str) -> str | None:
    """The rarer/positive class for binary AUC, or None for other tasks."""
    return manifest["datasets"][key]["target"].get("positive_class")


def main() -> int:
    manifest = load_manifest()
    datasets: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for key, spec in REGISTRY.items():
        if not (train_path(key).exists() and holdout_path(key).exists()):
            failures[key] = "not materialised — run scripts/prepare_datasets.py"
            print(f"[baseline] {key}: SKIPPED — {failures[key]}", flush=True)
            continue

        print(f"[baseline] {key} ({spec.metric}) ...", flush=True)
        try:
            train, holdout = load_train(key), load_holdout(key)
            pos = _positive_class(manifest, key)
            entry: dict = {"metric": spec.metric, "task": spec.task}
            for system in BASELINE_SYSTEMS:
                res = fit_score_baseline(system, spec, train, holdout, pos)
                entry[system] = res
                print(
                    f"           {system:>10}: {spec.metric}={res['value']:.4f} "
                    f"{res['secondary']}",
                    flush=True,
                )
            # Honesty check: scoring must not have altered the sealed holdout.
            if not verify_holdout_untouched(key):
                raise RuntimeError("holdout seal broken after scoring")
            datasets[key] = entry
        except Exception as e:  # noqa: BLE001 — report and continue, never drop silently
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"           FAILED — {failures[key]}", flush=True)

    report = {
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "systems": list(BASELINE_SYSTEMS),
        "note": (
            "Baseline 0: Dummy is the honest floor (ignores features); default_rf "
            "is an untuned RandomForest with minimal impute+one-hot preprocessing. "
            "All scores are on the LOCKED holdout. Higher is better for every metric."
        ),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "failures": failures,
    }
    BASELINE_METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[baseline] wrote -> {BASELINE_METRICS_PATH}")
    print(f"[baseline] {len(datasets)}/{len(REGISTRY)} datasets scored.")
    if failures:
        print(f"[baseline] failures: {list(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
