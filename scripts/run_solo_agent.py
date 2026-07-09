"""Day 3 — Baseline 1: the solo agent (one LLM, one shot) per dataset.

Run after ``prepare_datasets.py``:

    python scripts/run_solo_agent.py

For each dataset it hands a single agent the TRAIN profile + task + metric, has it
emit one sklearn ``solve(train_df)`` module, executes that module in a subprocess
(fitting on train, predicting on held-out FEATURES only), and scores the result
once on the LOCKED holdout through :mod:`crewml.scoring`. Output:
``results/solo_agent_metrics.json`` — the direct target the Phase-2 crew must beat.

Without an LLM key the run is in MOCK mode: a fixed competent single-shot script
is used and every result is stamped ``"mock": true`` (EVAL_PROTOCOL.md §5). Mock
numbers are never the real headline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn

from crewml import config
from crewml.config import RESULTS_DIR, SEED
from crewml.datasets import REGISTRY, holdout_path, load_manifest, train_path
from crewml.solo_agent import run_solo_agent

SOLO_METRICS_PATH = RESULTS_DIR / "solo_agent_metrics.json"


def _positive_class(manifest: dict, key: str) -> str | None:
    return manifest["datasets"][key]["target"].get("positive_class")


def main() -> int:
    mock = config.is_mock_mode()
    if mock:
        print("[solo] MOCK MODE — no LLM key; scores are labelled mock, not headline.", flush=True)
    else:
        print(f"[solo] provider={config.LLM_PROVIDER} model live.", flush=True)

    manifest = load_manifest()
    datasets: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for key, spec in REGISTRY.items():
        if not (train_path(key).exists() and holdout_path(key).exists()):
            failures[key] = "not materialised — run scripts/prepare_datasets.py"
            print(f"[solo] {key}: SKIPPED — {failures[key]}", flush=True)
            continue

        print(f"[solo] {key} ({spec.metric}) — generating + executing ...", flush=True)
        try:
            res = run_solo_agent(spec, _positive_class(manifest, key))
            datasets[key] = res
            if res["ok"]:
                print(
                    f"       -> {spec.metric}={res['value']:.4f} {res['secondary']} "
                    f"(mock={res['mock']})",
                    flush=True,
                )
            else:
                failures[key] = f"solo script failed: {res['error']}"
                print(f"       FAILED — {res['error']}", flush=True)
        except Exception as e:  # noqa: BLE001 — record and continue, never drop silently
            failures[key] = f"{type(e).__name__}: {e}"
            print(f"       FAILED — {failures[key]}", flush=True)

    n_ok = sum(1 for v in datasets.values() if v.get("ok"))
    report = {
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "system": "solo_agent",
        "mock": mock,
        "provider": (config.LLM_PROVIDER if not mock else "mock"),
        "note": (
            "Baseline 1: a single agent, one shot, writes one sklearn solve(train_df) "
            "module; executed in a subprocess fitting on train only and scored once on "
            "the LOCKED holdout. This is the number the Phase-2 crew must beat. "
            + ("MOCK run — scores are NOT the real headline (EVAL_PROTOCOL.md §5)."
               if mock else "Live-LLM run.")
        ),
        "n_datasets_ok": n_ok,
        "datasets": datasets,
        "failures": failures,
    }
    SOLO_METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[solo] wrote -> {SOLO_METRICS_PATH}")
    print(f"[solo] {n_ok}/{len(REGISTRY)} datasets scored (mock={mock}).")
    if failures:
        print(f"[solo] failures: {list(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
