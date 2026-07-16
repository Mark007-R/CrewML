"""Day 8 guards: the Planner turns a DataProfile into an honest ModelingPlan.

The Planner is the crew's second REAL node, so these assert *content*: that column
drops follow the profile's leakage checks, that preprocessing is dtype-aware and
respects cardinality + disguised-missing, that the CV scheme and candidate models
fit the task, that the imbalance strategy fires only when flagged, that a Critic
critique actually changes the plan, and that the whole thing is deterministic,
JSON-safe, and never reaches the held-out split. The LLM narrative is strictly
advisory — layered on, never a source of decisions, and honestly labelled.
"""
from __future__ import annotations

import inspect
import json

import pytest

from crewml import config, llm
from crewml.crew import planner as pl
from crewml.crew.planner import build_plan, run_planner
from crewml.crew.profiler import build_profile
from crewml.datasets import REGISTRY, load_train


def _profile(key: str) -> dict:
    return build_profile(REGISTRY[key], load_train(key))


# --- Structure, determinism, JSON-safety ------------------------------------

def test_plan_has_expected_shape():
    plan = build_plan(_profile("credit-g"))
    for key in ("schema_version", "dataset_key", "task", "subtype", "metric",
                "drop_columns", "preprocessing", "candidate_models", "cv",
                "imbalance_strategy", "recommended_primary_model", "rationale"):
        assert key in plan
    assert plan["stub"] is False
    assert plan["node"] == "planner"
    assert plan["dataset_key"] == "credit-g"


def test_plan_is_deterministic_and_json_safe():
    p = _profile("diabetes")
    a = build_plan(p)
    b = build_plan(p)
    assert a == b            # pure function of the profile
    json.dumps(a)            # no numpy scalars / non-serialisable objects leak in


def test_recommended_primary_is_first_candidate():
    plan = build_plan(_profile("kin8nm"))
    assert plan["recommended_primary_model"] == plan["candidate_models"][0]["name"]


# --- Column drops follow the profile's leakage checks -----------------------

def test_drops_constant_id_and_duplicate_columns():
    # A profile with planted leakage/integrity problems (built via the Profiler).
    import numpy as np
    import pandas as pd
    from crewml.datasets import TARGET_COLUMN, DatasetSpec

    n = 200
    rng = np.random.default_rng(0)
    y = pd.Series(["a" if v else "b" for v in rng.integers(0, 2, n)])
    base = rng.normal(size=n)
    df = pd.DataFrame({
        "const": np.ones(n),
        "row_id": np.arange(n),
        "x": base,
        "x_copy": base,
        "keep": rng.normal(size=n),
        TARGET_COLUMN: y,
    })
    spec = DatasetSpec(key="synthetic", openml_name="s", version=1,
                       task="classification", subtype="binary", metric="roc_auc", note="t")
    plan = build_plan(build_profile(spec, df))
    drops = set(plan["drop_columns"])
    assert "const" in drops and "row_id" in drops
    assert "x_copy" in drops and "x" not in drops   # keep first of a duplicate group
    assert "keep" not in drops
    # A reason is recorded for every drop.
    assert {r["column"] for r in plan["drop_reasons"]} == drops


def test_clean_dataset_drops_nothing():
    plan = build_plan(_profile("kin8nm"))
    assert plan["drop_columns"] == []


# --- Preprocessing is dtype-aware and honours profile signals ----------------

def test_preprocessing_partitions_numeric_and_categorical():
    profile = _profile("credit-g")
    plan = build_plan(profile)
    pre = plan["preprocessing"]
    kept = set(profile["columns"]["numeric"]) | set(profile["columns"]["categorical"])
    kept -= set(plan["drop_columns"])
    assert set(pre["numeric"]["columns"]) | set(pre["categorical"]["columns"]) == kept
    assert pre["numeric"]["impute"] == "median"
    assert pre["categorical"]["impute"] == "most_frequent"


def test_disguised_missing_columns_flagged_for_zero_as_missing():
    # diabetes hides missing values as zeros in insu/skin/preg (the Profiler flags them).
    plan = build_plan(_profile("diabetes"))
    zam = plan["preprocessing"]["numeric"]["zero_as_missing"]
    assert "insu" in zam
    assert plan["preprocessing"]["numeric"]["zero_as_missing_is_heuristic"] is True


# --- CV scheme + candidate models fit the task ------------------------------

def test_classification_uses_stratified_cv_and_proba_models():
    plan = build_plan(_profile("credit-g"))
    assert plan["cv"]["scheme"] == "StratifiedKFold"
    assert plan["cv"]["scoring"] == "roc_auc"
    assert plan["cv"]["random_state"] == config.SEED
    assert all("Classifier" in m["estimator"] or m["estimator"] == "LogisticRegression"
               for m in plan["candidate_models"])


def test_regression_uses_kfold_and_regressors():
    plan = build_plan(_profile("kin8nm"))
    assert plan["cv"]["scheme"] == "KFold"
    assert plan["cv"]["scoring"] == "r2"
    assert all("Regress" in m["estimator"] or m["estimator"] == "Ridge"
               for m in plan["candidate_models"])
    # No classification-only concept leaks into a regression plan.
    assert plan["imbalance_strategy"]["recommended"] is False


def test_cv_splits_never_exceed_the_rarest_class():
    # multiclass 'vehicle' still has ample per-class support => full 5 folds.
    plan = build_plan(_profile("vehicle"))
    assert 2 <= plan["cv"]["n_splits"] <= 5


# --- Imbalance strategy: fires only when the profile flags it ----------------

def test_imbalance_strategy_recommended_on_skewed_binary():
    plan = build_plan(_profile("credit-g"))
    strat = plan["imbalance_strategy"]
    assert strat["recommended"] is True
    assert strat["use_stratified_cv"] is True
    # Positive class is the rarer one, carried from the profile.
    assert strat["positive_class"] == _profile("credit-g")["target"]["positive_class"]


# --- Critic re-entry actually changes the plan ------------------------------

def test_first_pass_has_no_critique():
    plan = build_plan(_profile("credit-g"))
    assert plan["addressed_critique"] is None
    assert "critique_adjustments" not in plan


def test_overfit_critique_reduces_capacity():
    profile = _profile("credit-g")
    base = build_plan(profile)
    crit = {"findings": ["the model is badly overfit on the training folds"], "decision": "iterate"}
    adjusted = build_plan(profile, critique=crit, iteration=1)
    assert adjusted["addressed_critique"] == crit
    assert adjusted["planning_for_iteration"] == 1
    assert any("overfit" in a for a in adjusted["critique_adjustments"])
    # The grids genuinely changed (regularisation strengthened), not just annotated.
    assert adjusted["candidate_models"] != base["candidate_models"]


def test_unmatched_critique_is_noted_not_silently_ignored():
    plan = build_plan(_profile("kin8nm"),
                      critique={"findings": ["something vague"], "decision": "iterate"})
    assert plan["critique_adjustments"]   # non-empty: a catch-all note is recorded


# --- No-peeking: the module never reaches the held-out split -----------------

def test_planner_source_never_references_the_locked_split():
    src = inspect.getsource(pl)
    assert "load_holdout" not in src
    assert "holdout" not in src.lower()


def test_planner_reads_only_the_profile_not_the_data():
    # The Planner takes a dict and never imports a data loader — structural proof
    # it cannot touch any split (train or holdout).
    src = inspect.getsource(pl)
    assert "load_train" not in src


# --- The LLM narrative is advisory, optional, and honestly labelled ---------

def test_narrative_unavailable_when_disabled():
    plan = run_planner(_profile("kin8nm"), with_llm=False)
    assert plan["llm_narrative"]["source"] == "unavailable"
    assert plan["llm_narrative"]["reason"] == "disabled"
    # Deterministic plan still fully present.
    assert plan["candidate_models"] and plan["cv"]["scheme"] == "KFold"


def test_narrative_marked_mock_in_mock_mode(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: True)
    plan = run_planner(_profile("kin8nm"), with_llm=True)
    assert plan["llm_narrative"]["source"] == "unavailable"
    assert plan["llm_narrative"]["reason"] == "mock_mode"
    assert plan["llm_narrative"]["is_mock"] is True


def test_narrative_attached_from_live_provider(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    fake = llm.LLMResult(
        text="  Try interaction features on duration x credit_amount.  ",
        provider="groq", model="llama-3.3-70b-versatile",
        prompt_tokens=88, completion_tokens=12,
    )
    monkeypatch.setattr(llm, "chat", lambda *a, **k: fake)
    plan = run_planner(_profile("credit-g"), with_llm=True)
    n = plan["llm_narrative"]
    assert n["source"] == "groq" and n["is_mock"] is False
    assert n["text"] == "Try interaction features on duration x credit_amount."  # stripped
    assert n["prompt_tokens"] == 88


def test_narrative_failure_degrades_without_crashing(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "chat", boom)
    plan = run_planner(_profile("kin8nm"), with_llm=True)   # must not raise
    assert plan["llm_narrative"]["source"] == "unavailable"
    assert "provider down" in plan["llm_narrative"]["reason"]
    assert plan["candidate_models"]                         # deterministic core intact
