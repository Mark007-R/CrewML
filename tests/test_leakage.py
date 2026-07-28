"""Day 22 guards: the single-feature leakage screen and the FE-gate leakage checks.

The regression this file pins is the Day-17 measured miss: an injected column
agreeing with the target on 95% of rows passed the Profiler's purity screen and
the Critic's CV ceiling, and the model trained on it. With the calibrated
single-feature screen these tests assert that exact probe is now caught — and,
just as important, that the clean suite's strongest legitimate features still
pass (the thin cpu_small margin is pinned deliberately: `freeswap` at 0.817 R²
sits 0.033 under the 0.85 ceiling, and a careless threshold change should fail
a test, not a benchmark run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crewml.crew.feature_engineer import DEFAULT_FE_SOURCE, _validate_fe, _verdict_error
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.datasets import REGISTRY, holdout_path, load_train, train_path
from crewml.failure_taxonomy import LEAK_COLUMN, make_leak_frame, probe_dataset
from crewml.leakage import (
    SCREEN_MEASURE,
    SINGLE_FEATURE_CEILING,
    screen_features,
    single_feature_cv_score,
)


def _require_prepared(key: str) -> None:
    if not (train_path(key).exists() and holdout_path(key).exists()):
        pytest.skip(f"{key} not materialised — run scripts/prepare_datasets.py")


# --- The screen itself (synthetic, no dataset dependency) --------------------

def test_ceilings_cover_every_primary_metric():
    assert set(SINGLE_FEATURE_CEILING) == {"roc_auc", "f1_macro", "r2"}
    assert all(0.0 < v < 1.0 for v in SINGLE_FEATURE_CEILING.values())


def test_perfect_synthetic_leak_is_screened():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400).astype(str))
    X = pd.DataFrame({
        "noise": rng.standard_normal(400),
        "leak": y.astype("category").cat.codes.astype(float),
    })
    suspects = screen_features(X, y, "classification", "roc_auc")
    assert [s["column"] for s in suspects] == ["leak"]
    assert suspects[0]["measure"] == SCREEN_MEASURE
    assert suspects[0]["signal"] > 0.99


def test_random_features_are_not_screened():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 400).astype(str))
    X = pd.DataFrame({f"f{i}": rng.standard_normal(400) for i in range(5)})
    assert screen_features(X, y, "classification", "roc_auc") == []


def test_unscoreable_target_returns_none_not_a_flag():
    """A single-class target can't be CV-scored — that is None, not a suspect."""
    y = pd.Series(["a"] * 100)
    x = pd.Series(np.arange(100.0))
    assert single_feature_cv_score(x, y, "classification", "roc_auc") is None


def test_all_nan_feature_scores_at_chance_and_is_not_screened():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 100).astype(str))
    X = pd.DataFrame({"empty": [np.nan] * 100})
    assert screen_features(X, y, "classification", "roc_auc") == []


def test_unknown_metric_screens_nothing():
    y = pd.Series([1.0, 2.0, 3.0, 4.0] * 25)
    X = pd.DataFrame({"copy": y})
    assert screen_features(X, y, "regression", "not_a_metric") == []


# --- The Day-17 regression, pinned (real data) -------------------------------

def test_day17_subtle_classification_leak_is_now_flagged():
    """THE regression test: the 95%-agreement leak that Day 17 measured passing
    every surface must now be flagged by the Profiler — via the new screen."""
    _require_prepared("credit-g")
    spec = REGISTRY["credit-g"]
    frame, truth = make_leak_frame(
        load_train("credit-g"), task=spec.task, kind="subtle",
        rng=np.random.default_rng(42),
    )
    assert truth["measured_signal"] == pytest.approx(0.95)  # the Day-17 window
    profile = build_profile(spec, frame)
    suspects = profile["leakage_checks"]["target_correlated_features"]
    entry = next((d for d in suspects if d["column"] == LEAK_COLUMN), None)
    assert entry is not None, "the Day-17 subtle leak went undetected again"
    assert entry["measure"] == SCREEN_MEASURE
    assert "target_leakage_suspected" in profile["assessment"]["flags"]


def test_day17_subtle_regression_leak_is_now_flagged():
    _require_prepared("cpu_small")
    spec = REGISTRY["cpu_small"]
    frame, truth = make_leak_frame(
        load_train("cpu_small"), task=spec.task, kind="subtle",
        rng=np.random.default_rng(42),
    )
    assert truth["measured_signal"] < 0.98  # under the old Pearson screen
    profile = build_profile(spec, frame)
    suspects = profile["leakage_checks"]["target_correlated_features"]
    assert any(
        d["column"] == LEAK_COLUMN and d["measure"] == SCREEN_MEASURE for d in suspects
    )


def test_planner_drops_the_screened_column():
    """The screen needs no new wiring: a flagged suspect flows into the plan's drops."""
    _require_prepared("credit-g")
    spec = REGISTRY["credit-g"]
    frame, _ = make_leak_frame(
        load_train("credit-g"), task=spec.task, kind="subtle",
        rng=np.random.default_rng(42),
    )
    plan = build_plan(build_profile(spec, frame))
    assert LEAK_COLUMN in plan["drop_columns"]


def test_clean_suite_thin_margins_stay_clean():
    """Zero false positives on the two datasets with the least headroom under
    their ceilings (diabetes `plas` 0.785/0.87 AUC, cpu_small `freeswap` 0.817/0.85 R²)."""
    for key in ("diabetes", "cpu_small"):
        _require_prepared(key)
        spec = REGISTRY[key]
        profile = build_profile(spec, load_train(key))
        assert profile["leakage_checks"]["target_correlated_features"] == [], key


# --- The FE validation gate (Day-22 checks) ----------------------------------

def test_default_fe_passes_the_new_gate():
    _require_prepared("credit-g")
    verdict = _validate_fe(DEFAULT_FE_SOURCE, "credit-g")
    assert verdict["ok"] is True
    assert verdict["row_wise_ok"] is True
    assert verdict["no_leakage"] is True
    assert verdict["leakage_ceiling"] == SINGLE_FEATURE_CEILING["roc_auc"]


def test_cross_row_fe_is_rejected_as_not_row_wise():
    _require_prepared("credit-g")
    source = (
        "import pandas as pd\n\n\n"
        "def add_features(df):\n"
        "    out = df.copy()\n"
        "    out['age_centered'] = df['age'] - df['age'].mean()\n"
        "    return out\n"
    )
    verdict = _validate_fe(source, "credit-g")
    assert verdict["ok"] is False
    assert verdict["row_wise_ok"] is False
    error = _verdict_error(verdict)
    assert "row-wise" in error and "across rows" in error


def test_leak_derived_fe_is_rejected_by_the_engineered_column_screen():
    _require_prepared("credit-g")
    spec = REGISTRY["credit-g"]
    frame, _ = make_leak_frame(
        load_train("credit-g"), task=spec.task, kind="subtle",
        rng=np.random.default_rng(42),
    )
    source = (
        "import pandas as pd\n\n\n"
        "def add_features(df):\n"
        "    out = df.copy()\n"
        "    out['risk_signal'] = df['leak_probe'] * 2.0 + 1.0\n"
        "    return out\n"
    )
    with probe_dataset("credit-g", "probe_test_fe_leak", frame) as pspec:
        verdict = _validate_fe(source, pspec.key)
    assert verdict["ok"] is False
    assert verdict["no_leakage"] is False
    assert verdict["leaky_columns"] == ["risk_signal"]
    error = _verdict_error(verdict)
    assert "risk_signal" in error and "leakage" in error
