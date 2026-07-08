"""Day 3 guards: the solo-agent baseline is honest, executable, and scored.

Three layers:
  * unit tests on the LLM plumbing (:mod:`crewml.llm`) — code extraction and the
    mock-mode contract — no network needed;
  * unit tests on :mod:`crewml.solo_agent` — the profile is train-only and the
    mock ``solve`` module compiles and honours its estimator contract;
  * integration checks on ``results/solo_agent_metrics.json`` — present, complete,
    finite, and (in mock mode) correctly labelled; plus the holdout seal survives.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from crewml import config, llm
from crewml.config import RESULTS_DIR
from crewml.datasets import (
    REGISTRY,
    TARGET_COLUMN,
    holdout_path,
    load_train,
    train_path,
    verify_holdout_untouched,
)
from crewml.solo_agent import build_profile_summary, mock_solo_script

SOLO_METRICS_PATH = RESULTS_DIR / "solo_agent_metrics.json"
KEYS = sorted(REGISTRY)


# --- llm plumbing (unit, network-free) --------------------------------------

def test_extract_python_prefers_fenced_block():
    reply = "Here you go:\n```python\ndef solve(df):\n    return 1\n```\nDone."
    assert llm.extract_python(reply) == "def solve(df):\n    return 1"


def test_extract_python_picks_largest_block():
    reply = "```python\nx=1\n```\nthen\n```python\ndef solve(df):\n    return df\n```"
    assert "def solve" in llm.extract_python(reply)


def test_extract_python_falls_back_to_raw():
    assert llm.extract_python("def solve(df):\n    return df") == "def solve(df):\n    return df"


def test_chat_raises_in_mock_mode():
    """No key configured -> chat must refuse rather than fabricate a completion."""
    if not config.is_mock_mode():
        pytest.skip("live LLM key configured; mock-mode contract not exercised")
    with pytest.raises(llm.MockModeError):
        llm.chat("sys", "user")


# --- profile summary is train-only ------------------------------------------

@pytest.mark.parametrize("key", KEYS)
def test_profile_summary_is_train_only_and_names_metric(key):
    if not train_path(key).exists():
        pytest.skip(f"{key} not materialised")
    spec = REGISTRY[key]
    summary = build_profile_summary(spec, load_train(key))
    assert spec.metric in summary
    assert spec.task in summary
    # It must describe the training rows, never the holdout.
    assert f"Rows (train): {len(load_train(key))}" in summary
    assert "holdout" not in summary.lower()


# --- mock solve() module compiles and honours its contract ------------------

def _run_solve(script: str, train: pd.DataFrame):
    ns: dict = {}
    exec(compile(script, "<mock_solo>", "exec"), ns)  # noqa: S102 — trusted mock template
    assert "solve" in ns, "generated module must define solve()"
    return ns["solve"](train)


def test_mock_classifier_solve_supports_proba_and_classes():
    spec = REGISTRY["credit-g"]  # binary classification
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "num": rng.normal(size=40),
            "cat": rng.choice(["a", "b", "c"], size=40),
            TARGET_COLUMN: rng.choice(["good", "bad"], size=40),
        }
    )
    model = _run_solve(mock_solo_script(spec), train)
    X = train.drop(columns=[TARGET_COLUMN])
    assert len(model.predict(X)) == len(train)
    proba = model.predict_proba(X)
    assert proba.shape == (len(train), 2)
    assert set(str(c) for c in model.classes_) == {"good", "bad"}


def test_mock_regressor_solve_predicts_numeric():
    spec = REGISTRY["cpu_small"]  # regression
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        {"a": rng.normal(size=30), "b": rng.normal(size=30), TARGET_COLUMN: rng.normal(size=30)}
    )
    model = _run_solve(mock_solo_script(spec), train)
    preds = model.predict(train.drop(columns=[TARGET_COLUMN]))
    assert len(preds) == 30
    assert np.isfinite(np.asarray(preds, dtype=float)).all()


# --- solo_agent_metrics.json integration ------------------------------------

def _load_solo() -> dict:
    if not SOLO_METRICS_PATH.exists():
        pytest.skip("solo_agent_metrics.json missing — run scripts/run_solo_agent.py")
    return json.loads(SOLO_METRICS_PATH.read_text())


def test_solo_metrics_complete_and_no_failures():
    report = _load_solo()
    assert report["failures"] == {}
    assert set(report["datasets"]) == set(REGISTRY)
    for entry in report["datasets"].values():
        assert entry["ok"] is True
        assert np.isfinite(entry["value"])


def test_solo_mock_flag_is_consistent():
    """Every per-dataset mock flag must agree with the run-level flag (§5 honesty)."""
    report = _load_solo()
    run_mock = report["mock"]
    for entry in report["datasets"].values():
        assert entry["mock"] == run_mock


@pytest.mark.parametrize("key", KEYS)
def test_solo_beats_dummy_floor(key):
    """A working solo agent must clear the feature-blind floor on every dataset."""
    report = _load_solo()
    if not holdout_path(key).exists():
        pytest.skip(f"{key} not materialised")
    baselines = json.loads((RESULTS_DIR / "baseline_metrics.json").read_text())
    solo_v = report["datasets"][key]["value"]
    dummy_v = baselines["datasets"][key]["dummy"]["value"]
    assert solo_v > dummy_v


@pytest.mark.parametrize("key", KEYS)
def test_holdout_seal_intact_after_solo_run(key):
    """The honesty proof: scoring the solo agent left the holdout untouched."""
    if not holdout_path(key).exists():
        pytest.skip(f"{key} not materialised")
    assert verify_holdout_untouched(key) is True
