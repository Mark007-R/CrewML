"""Day 2 guards: the scorer computes the protocol metrics, baselines anchor sanely.

Two layers:
  * pure unit tests on :mod:`crewml.scoring` with hand-built inputs (no datasets
    needed) — these pin the exact metric semantics from EVAL_PROTOCOL.md;
  * integration checks on ``results/baseline_metrics.json`` — present, complete,
    and Dummy is beaten by the default RandomForest on every dataset.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from crewml.baselines import BASELINE_SYSTEMS, build_preprocessor, split_xy
from crewml.config import RESULTS_DIR
from crewml.datasets import (
    BENCHMARK_KEYS, REGISTRY, TARGET_COLUMN, holdout_path, load_train, train_path,
)
from crewml.scoring import score_predictions

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
KEYS = list(BENCHMARK_KEYS)  # benchmark-scoped, immune to restored uploads


# --- scoring semantics (unit, dataset-free) ---------------------------------

def test_binary_auc_uses_positive_class_probability():
    spec = REGISTRY["credit-g"]  # binary, roc_auc, positive="bad"
    y_true = ["good", "bad", "good", "bad"]
    class_labels = ["bad", "good"]
    # Perfect ranking: higher P(bad) exactly for the "bad" rows.
    y_proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])
    res = score_predictions(
        spec, y_true, y_proba=y_proba, class_labels=class_labels,
        positive_class="bad", y_pred=["good", "bad", "good", "bad"],
    )
    assert res["metric"] == "roc_auc"
    assert res["value"] == pytest.approx(1.0)


def test_binary_auc_constant_probability_is_half():
    spec = REGISTRY["diabetes"]
    y_true = ["tested_positive", "tested_negative", "tested_positive", "tested_negative"]
    class_labels = ["tested_negative", "tested_positive"]
    y_proba = np.tile([0.5, 0.5], (4, 1))
    res = score_predictions(
        spec, y_true, y_proba=y_proba, class_labels=class_labels,
        positive_class="tested_positive",
    )
    assert res["value"] == pytest.approx(0.5)


def test_multiclass_uses_macro_f1():
    spec = REGISTRY["vehicle"]
    y_true = ["bus", "saab", "opel", "van"]
    res = score_predictions(spec, y_true, y_pred=list(y_true))  # perfect
    assert res["metric"] == "f1_macro"
    assert res["value"] == pytest.approx(1.0)
    assert res["secondary"]["accuracy"] == pytest.approx(1.0)


def test_regression_r2_and_rmse():
    spec = REGISTRY["kin8nm"]
    y_true = [1.0, 2.0, 3.0, 4.0]
    res = score_predictions(spec, y_true, y_pred=list(y_true))  # perfect
    assert res["metric"] == "r2"
    assert res["value"] == pytest.approx(1.0)
    assert res["secondary"]["rmse"] == pytest.approx(0.0)


def test_binary_auc_requires_proba():
    spec = REGISTRY["credit-g"]
    with pytest.raises(ValueError):
        score_predictions(spec, ["good", "bad"], y_pred=["good", "bad"])


# --- preprocessor safety ----------------------------------------------------

def _require_prepared(key: str) -> None:
    if not (train_path(key).exists() and holdout_path(key).exists()):
        pytest.skip(f"{key} not materialised — run scripts/prepare_datasets.py")


def test_preprocessor_removes_nans():
    """credit-g carries categoricals; the preprocessor must yield finite numerics."""
    _require_prepared("credit-g")
    X, _ = split_xy(load_train("credit-g"))
    Xt = build_preprocessor(X).fit_transform(X)
    dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    assert np.isfinite(dense).all()


# --- baseline_metrics.json integration --------------------------------------

def _load_baselines() -> dict:
    if not BASELINE_METRICS_PATH.exists():
        pytest.skip("baseline_metrics.json missing — run scripts/run_baselines.py")
    return json.loads(BASELINE_METRICS_PATH.read_text())


def test_baseline_metrics_complete():
    report = _load_baselines()
    assert report["failures"] == {}
    assert set(report["datasets"]) == set(BENCHMARK_KEYS)
    assert list(report["systems"]) == list(BASELINE_SYSTEMS)


@pytest.mark.parametrize("key", KEYS)
def test_default_rf_beats_dummy(key):
    """The whole point of a floor: the default forest must clear it everywhere."""
    report = _load_baselines()
    entry = report["datasets"][key]
    assert entry["default_rf"]["value"] > entry["dummy"]["value"]


def test_dummy_binary_auc_is_half():
    """A feature-blind classifier can only achieve chance AUC."""
    report = _load_baselines()
    for key in BENCHMARK_KEYS:
        spec = REGISTRY[key]
        if spec.subtype == "binary":
            assert report["datasets"][key]["dummy"]["value"] == pytest.approx(0.5, abs=1e-6)
