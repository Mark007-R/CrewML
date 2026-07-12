"""Day 7 guards: the Profiler produces an honest, deterministic DataProfile.

The Profiler is the first REAL crew node, so unlike the Day-5 skeleton tests these
assert *content*: that the computed facts are right (imbalance, disguised missing,
leakage), that clean data stays clean (the checks don't cry wolf), that the whole
thing is reproducible and JSON-safe, and that the LLM narrative is strictly
advisory — layered on, never a source of facts, and honestly labelled.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from crewml import config, llm
from crewml.crew import profiler as prof
from crewml.crew.profiler import build_profile, run_profiler
from crewml.datasets import TARGET_COLUMN, DatasetSpec, REGISTRY, load_train


def _spec(task: str, subtype: str, metric: str) -> DatasetSpec:
    return DatasetSpec(
        key="synthetic", openml_name="synthetic", version=1,
        task=task, subtype=subtype, metric=metric, note="synthetic test frame",
    )


# --- Structure, determinism, JSON-safety ------------------------------------

def test_profile_has_expected_shape():
    spec = REGISTRY["credit-g"]
    p = build_profile(spec, load_train("credit-g"))
    for key in ("schema_version", "dataset_key", "task", "n_rows", "n_features",
                "columns", "features", "target", "missingness", "leakage_checks",
                "assessment"):
        assert key in p
    assert p["stub"] is False
    # A per-column fact for every feature (target excluded).
    assert set(p["features"]) == set(load_train("credit-g").columns) - {TARGET_COLUMN}
    assert p["n_features"] == len(p["features"])


def test_profile_is_deterministic_and_json_safe():
    spec = REGISTRY["diabetes"]
    df = load_train("diabetes")
    a = build_profile(spec, df)
    b = build_profile(spec, df)
    assert a == b                       # pure function, no randomness
    json.dumps(a)                       # no numpy scalars leaked in


# --- Target distribution + imbalance ----------------------------------------

def test_binary_imbalance_and_positive_is_rarer_class():
    p = build_profile(REGISTRY["credit-g"], load_train("credit-g"))
    t = p["target"]
    assert t["n_classes"] == 2
    # positive/scored class must be the rarer one (matches crewml.scoring).
    assert t["classes"][t["positive_class"]] == min(t["classes"].values())
    assert t["imbalance_ratio"] > 1.0
    assert "class_imbalance" in p["assessment"]["flags"]


def test_regression_target_summary():
    p = build_profile(REGISTRY["kin8nm"], load_train("kin8nm"))
    t = p["target"]
    assert {"min", "max", "mean", "std", "skew"} <= set(t)
    assert "classes" not in t


# --- Disguised missing (the diabetes signal the crew "must not miss") --------

def test_diabetes_flags_suspected_disguised_missing():
    p = build_profile(REGISTRY["diabetes"], load_train("diabetes"))
    suspected = {d["column"] for d in p["leakage_checks"]["suspected_disguised_missing"]}
    # The insulin column is ~47% zeros — the textbook disguised-missing case.
    assert "insu" in suspected
    assert "disguised_missing_suspected" in p["assessment"]["flags"]


# --- Leakage checks: catch the planted, stay quiet on the clean --------------

def test_clean_dataset_raises_no_hard_leakage_flags():
    p = build_profile(REGISTRY["kin8nm"], load_train("kin8nm"))
    lk = p["leakage_checks"]
    assert lk["constant_columns"] == []
    assert lk["id_like_columns"] == []           # continuous floats are NOT ids
    assert lk["duplicate_feature_columns"] == []
    assert lk["target_correlated_features"] == []


def test_target_leakage_detected_classification():
    rng = np.random.default_rng(0)
    y = pd.Series(["pos" if v else "neg" for v in rng.integers(0, 2, 300)])
    df = pd.DataFrame({
        "noise": rng.normal(size=300),
        "leak": y.map({"pos": 1, "neg": 0}),   # a perfect copy of the target
        TARGET_COLUMN: y,
    })
    p = build_profile(_spec("classification", "binary", "roc_auc"), df)
    leaked = {d["column"] for d in p["leakage_checks"]["target_correlated_features"]}
    assert "leak" in leaked
    assert "noise" not in leaked
    assert "target_leakage_suspected" in p["assessment"]["flags"]


def test_target_leakage_detected_regression():
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(size=300))
    df = pd.DataFrame({
        "noise": rng.normal(size=300),
        "leak": y * 2.0 + 0.001,                # near-perfectly correlated with target
        TARGET_COLUMN: y,
    })
    p = build_profile(_spec("regression", "regression", "r2"), df)
    leaked = {d["column"] for d in p["leakage_checks"]["target_correlated_features"]}
    assert "leak" in leaked
    assert "noise" not in leaked


def test_constant_and_id_like_and_duplicate_columns():
    n = 200
    rng = np.random.default_rng(2)
    y = pd.Series(["a" if v else "b" for v in rng.integers(0, 2, n)])
    base = rng.normal(size=n)
    df = pd.DataFrame({
        "const": np.ones(n),                    # zero-variance
        "row_id": np.arange(n),                 # integer identifier (near-unique)
        "x": base,
        "x_copy": base,                         # exact duplicate of x
        TARGET_COLUMN: y,
    })
    lk = build_profile(_spec("classification", "binary", "roc_auc"), df)["leakage_checks"]
    assert "const" in lk["constant_columns"]
    assert "row_id" in lk["id_like_columns"]
    assert any({"x", "x_copy"} <= set(g) for g in lk["duplicate_feature_columns"])


# --- No-peeking: the module never reaches the held-out split -----------------

def test_profiler_source_never_references_the_locked_split():
    src = inspect.getsource(prof)
    assert "load_holdout" not in src
    assert "holdout" not in src.lower()


# --- The LLM narrative is advisory, optional, and honestly labelled ---------

def test_narrative_unavailable_when_disabled():
    p = run_profiler("kin8nm", with_llm=False)
    assert p["llm_narrative"]["source"] == "unavailable"
    assert p["llm_narrative"]["reason"] == "disabled"
    # Deterministic facts still fully present.
    assert p["assessment"]["source"] == "deterministic"


def test_narrative_marked_mock_in_mock_mode(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: True)
    p = run_profiler("kin8nm", with_llm=True)
    assert p["llm_narrative"]["source"] == "unavailable"
    assert p["llm_narrative"]["reason"] == "mock_mode"
    assert p["llm_narrative"]["is_mock"] is True


def test_narrative_attached_from_live_provider(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    fake = llm.LLMResult(
        text="  Watch the class imbalance.  ", provider="groq",
        model="llama-3.3-70b-versatile", prompt_tokens=42, completion_tokens=7,
    )
    monkeypatch.setattr(llm, "chat", lambda *a, **k: fake)
    p = run_profiler("credit-g", with_llm=True)
    n = p["llm_narrative"]
    assert n["source"] == "groq" and n["is_mock"] is False
    assert n["text"] == "Watch the class imbalance."   # stripped
    assert n["prompt_tokens"] == 42


def test_narrative_failure_degrades_without_crashing(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "chat", boom)
    p = run_profiler("kin8nm", with_llm=True)          # must not raise
    assert p["llm_narrative"]["source"] == "unavailable"
    assert "provider down" in p["llm_narrative"]["reason"]
    assert p["target"]["min"] is not None              # deterministic core intact
