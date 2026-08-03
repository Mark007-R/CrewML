"""Day 11 guards: the Ensembler combines honestly and never ships a worse model.

The Ensembler is the crew's sixth real node. These assert:

  * **Real combination** — it builds a soft-voting / averaging ensemble of the top
    candidates and cross-validates it against the single best on the *same* seeded folds,
    inside the sandbox, on the train split.
  * **Never worse** — the crew's final model is ``max(ensemble, single)`` on CV; a tie or
    a loss keeps the simpler single model. The self-consistent single re-score matches the
    Trainer's number (same params, same seeded CV).
  * **Honest degradation** — a failed Trainer run yields a "not attempted" record (falls
    back to the single model), never a crash; a too-thin candidate set does the same.
  * **No peeking** — the module never loads the held-out split (source-inspected); every
    number is labelled a CV estimate on train.

Runs are real subprocesses, so tests use ``param_search=False`` (CV at default params) to
stay fast; the grid-search path is the same code with the Trainer's tuned params.
"""
from __future__ import annotations

import inspect

import joblib
import pytest

from crewml.crew import ensembler as en
from crewml.crew.ensembler import run_ensembler
from crewml.crew.feature_engineer import DEFAULT_FE_SOURCE, run_feature_engineer
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.crew.trainer import run_trainer
from crewml.datasets import REGISTRY, load_train, verify_holdout_untouched

# Full-crew / model-fit module: minute-scale by nature (Day 28 speed lanes).
pytestmark = pytest.mark.slow


def _plan(key: str) -> dict:
    return build_plan(build_profile(REGISTRY[key], load_train(key)))


@pytest.fixture(scope="module")
def credit_ensemble():
    """One shared credit-g run: train (no grid search) then ensemble — reused across tests."""
    key = "credit-g"
    plan = _plan(key)
    fe = run_feature_engineer(plan, key, with_llm=False)
    training = run_trainer(plan, fe["code"], key, param_search=False)
    ensemble = run_ensembler(plan, training, fe["code"], key)
    return {"plan": plan, "fe": fe, "training": training, "ensemble": ensemble}


# --- Real combination + honest comparison -----------------------------------

def test_ensembler_is_real_not_a_stub(credit_ensemble):
    ens = credit_ensemble["ensemble"]
    assert ens["stub"] is False
    assert ens["node"] == "ensembler"
    assert ens["attempted"] is True and ens["ok"] is True


def test_ensemble_combines_the_top_candidates(credit_ensemble):
    ens = credit_ensemble["ensemble"]
    # Soft voting for a classification task, over at least two members.
    assert ens["voting"] == "soft"
    assert len(ens["members"]) >= 2
    assert set(ens["members"]).issubset({
        "hist_gradient_boosting", "random_forest", "logistic_regression"
    })


def test_both_sides_scored_and_final_is_never_worse_than_single(credit_ensemble):
    ens = credit_ensemble["ensemble"]
    assert isinstance(ens["ensemble_cv_score"], float)
    assert isinstance(ens["single_best_cv_score"], float)
    # The crew ships max(ensemble, single): the final CV is >= the single best.
    assert ens["final_cv_score"] >= ens["single_best_cv_score"] - 1e-9
    # And the chosen model is consistent with the scores.
    if ens["chosen"] == "ensemble":
        assert ens["ensemble_cv_score"] > ens["single_best_cv_score"]
    else:
        assert ens["ensemble_cv_score"] <= ens["single_best_cv_score"] + en.ENSEMBLE_MIN_GAIN


def test_single_rescore_matches_the_trainer_number(credit_ensemble):
    # The self-consistent single re-score uses the same seeded CV + params as the Trainer,
    # so it must reproduce the Trainer's CV score for the winning model.
    ens = credit_ensemble["ensemble"]
    training = credit_ensemble["training"]
    assert ens["single_best_model"] == training["best_model"]
    assert ens["single_best_cv_score"] == pytest.approx(training["cv_score"], abs=1e-6)


def test_final_model_is_persisted_and_loadable(credit_ensemble):
    ens = credit_ensemble["ensemble"]
    assert "final_model.joblib" in ens["artifacts"]
    from crewml.executor import EXECUTOR_DIR

    art = EXECUTOR_DIR / ens["run_id"] / "artifacts" / "final_model.joblib"
    assert art.is_file()
    model = joblib.load(art)
    X = load_train("credit-g").drop(columns=["target"]).head(5)
    X = X.assign(row_nan_count=0)   # the FE-added column the fitted pipeline expects
    assert len(model.predict(X)) == 5


# --- Regression path: averaging, not voting ---------------------------------

def test_regression_ensembles_by_averaging():
    key = "kin8nm"
    plan = _plan(key)
    training = run_trainer(plan, DEFAULT_FE_SOURCE, key, param_search=False)
    ens = run_ensembler(plan, training, DEFAULT_FE_SOURCE, key)
    assert ens["ok"] is True
    assert ens["voting"] == "average"
    assert ens["metrics"]["scoring"] == "r2"
    assert ens["final_cv_score"] >= ens["single_best_cv_score"] - 1e-9


# --- Honesty: CV on train, deterministic, holdout sealed --------------------

def test_scores_are_labelled_cross_validated_not_holdout(credit_ensemble):
    assert credit_ensemble["ensemble"]["cv_score_is_holdout"] is False
    assert credit_ensemble["ensemble"]["metrics"]["cv_score_is_holdout"] is False


def test_holdout_seal_intact_after_ensembling(credit_ensemble):
    assert verify_holdout_untouched("credit-g") is True


def test_ensembling_is_deterministic(credit_ensemble):
    key = "credit-g"
    again = run_ensembler(
        credit_ensemble["plan"], credit_ensemble["training"], credit_ensemble["fe"]["code"], key
    )
    assert again["ensemble_cv_score"] == credit_ensemble["ensemble"]["ensemble_cv_score"]
    assert again["chosen"] == credit_ensemble["ensemble"]["chosen"]


# --- Failure is reported, never raised --------------------------------------

def test_failed_training_yields_not_attempted_not_a_crash():
    key = "credit-g"
    plan = _plan(key)
    failed = {"ok": False, "error": "boom", "best_model": None, "cv_score": None, "metrics": {}}
    ens = run_ensembler(plan, failed, DEFAULT_FE_SOURCE, key)   # must not raise
    assert ens["attempted"] is False
    assert ens["ensemble_cv_score"] is None
    assert "failed" in ens["reason"]


def test_too_few_candidates_falls_back_to_single():
    # A plan/training with a single scored candidate can't form an ensemble.
    key = "credit-g"
    plan = _plan(key)
    plan["candidate_models"] = [plan["candidate_models"][0]]
    training = {
        "ok": True, "best_model": plan["candidate_models"][0]["name"], "cv_score": 0.80,
        "metrics": {"per_model": [{"name": plan["candidate_models"][0]["name"],
                                   "cv_mean": 0.80, "cv_std": 0.02, "best_params": {}}]},
    }
    ens = run_ensembler(plan, training, DEFAULT_FE_SOURCE, key)
    assert ens["attempted"] is False
    assert ens["chosen"] == "single"
    assert ens["final_cv_score"] == 0.80


# --- No-peeking: the module never loads the held-out split ------------------

def test_ensembler_source_never_loads_the_holdout():
    src = inspect.getsource(en)
    assert "load_holdout" not in src
    assert "holdout_path" not in src
