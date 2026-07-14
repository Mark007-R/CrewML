"""Day 9 guards: the Trainer produces honest cross-validated numbers + artifacts.

The Trainer is the crew's first modeling node, so these assert *content*: it runs
the plan's candidates through the sandboxed executor on the train split, returns a
real CV score (a float, in range, per candidate), saves a loadable model + the exact
FE source, and — critically — labels every number as a cross-validated estimate on
train, never a held-out score. The holdout seal is still intact after a run, the
results are deterministic (seeded), a broken generation is reported rather than
raised, and the module can never load the held-out split.

Runs are real subprocesses, so tests use ``param_search=False`` (CV at default
params) to stay fast; the grid-search path is the same code with larger grids.
"""
from __future__ import annotations

import inspect

import joblib
import pytest

from crewml.crew import trainer as tr
from crewml.crew.feature_engineer import DEFAULT_FE_SOURCE, run_feature_engineer
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.crew.trainer import run_trainer
from crewml.datasets import REGISTRY, load_train, verify_holdout_untouched


def _plan(key: str) -> dict:
    return build_plan(build_profile(REGISTRY[key], load_train(key)))


@pytest.fixture(scope="module")
def credit_run():
    """One shared credit-g training run (binary, imbalanced) — reused across tests."""
    key = "credit-g"
    plan = _plan(key)
    fe = run_feature_engineer(plan, key, with_llm=False)  # deterministic default FE
    return run_trainer(plan, fe["code"], key, param_search=False)


# --- Real CV metrics + artifacts --------------------------------------------

def test_trainer_is_real_not_a_stub(credit_run):
    assert credit_run["stub"] is False
    assert credit_run["node"] == "trainer"
    assert credit_run["ok"] is True


def test_cv_score_is_a_real_number_in_range(credit_run):
    score = credit_run["cv_score"]
    assert isinstance(score, float)
    assert 0.5 <= score <= 1.0        # roc_auc for a competent model on credit-g
    assert credit_run["best_model"] in {
        "hist_gradient_boosting", "random_forest", "logistic_regression"
    }


def test_every_candidate_was_cross_validated(credit_run):
    per_model = credit_run["metrics"]["per_model"]
    assert {r["name"] for r in per_model} == {
        "hist_gradient_boosting", "random_forest", "logistic_regression"
    }
    assert all(isinstance(r["cv_mean"], float) for r in per_model)
    # The reported best matches the max-CV candidate.
    best = max(per_model, key=lambda r: r["cv_mean"])
    assert best["name"] == credit_run["best_model"]


def test_feature_engineering_was_applied(credit_run):
    m = credit_run["metrics"]
    assert m["fe_applied"] is True
    assert m["n_features_after_fe"] == m["n_features_original"] + 1   # default adds one
    assert m["engineered_columns"] == ["row_nan_count"]


def test_saved_model_is_loadable_and_predicts(credit_run):
    assert "model.joblib" in credit_run["artifacts"]
    assert "fe_source.py" in credit_run["artifacts"]
    # run_trainer keeps the workdir; the model lives under the executor artifacts dir.
    from crewml.executor import EXECUTOR_DIR

    art = EXECUTOR_DIR / credit_run["run_id"] / "artifacts" / "model.joblib"
    assert art.is_file()
    model = joblib.load(art)
    X = load_train("credit-g").drop(columns=["target"])
    X = X.assign(row_nan_count=0)   # the FE-added column the fitted pipeline expects
    preds = model.predict(X.head(5))
    assert len(preds) == 5


def test_binary_label_mapping_recorded(credit_run):
    # credit-g positive (rarer) class is 'bad' -> encoded as 1 for roc_auc.
    mapping = credit_run["metrics"]["label_mapping"]
    assert mapping == {"1": "bad", "0": "good"}


# --- Honesty: CV on train, never the holdout --------------------------------

def test_scores_are_labelled_cross_validated_not_holdout(credit_run):
    assert credit_run["cv_score_is_holdout"] is False
    assert credit_run["metrics"]["cv_score_is_holdout"] is False


def test_holdout_seal_intact_after_training(credit_run):
    # The Trainer only ever loaded train; the locked holdout must be untouched.
    assert verify_holdout_untouched("credit-g") is True


def test_training_is_deterministic(credit_run):
    key = "credit-g"
    plan = _plan(key)
    again = run_trainer(plan, DEFAULT_FE_SOURCE, key, param_search=False)
    assert again["cv_score"] == credit_run["cv_score"]
    assert again["best_model"] == credit_run["best_model"]


# --- Regression path --------------------------------------------------------

def test_regression_path_produces_r2():
    key = "kin8nm"
    plan = _plan(key)
    result = run_trainer(plan, DEFAULT_FE_SOURCE, key, param_search=False)
    assert result["ok"] is True
    assert result["metrics"]["scoring"] == "r2"
    assert result["metrics"]["label_mapping"] is None       # no class encoding for regression
    assert isinstance(result["cv_score"], float)


# --- Failure is reported, never raised --------------------------------------

def test_broken_fe_code_is_reported_not_raised():
    key = "credit-g"
    plan = _plan(key)
    broken = "def add_features(df):\n    this is not valid python\n"
    result = run_trainer(plan, broken, key, param_search=False)   # must not raise
    assert result["ok"] is False
    assert result["cv_score"] is None
    assert result["error"]


# --- No-peeking: the module never loads the held-out split ------------------

def test_trainer_source_never_loads_the_holdout():
    src = inspect.getsource(tr)
    assert "load_holdout" not in src
    assert "holdout_path" not in src
