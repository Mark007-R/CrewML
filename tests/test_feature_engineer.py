"""Day 9 guards: the Feature Engineer generates *validated*, leakage-free FE code.

The Feature Engineer is the crew's first generated-code node, so these assert the
trust discipline rather than model quality: the deterministic default always runs
and passes the sandbox contract; a live provider's code is used ONLY after it
passes the same sandbox validation; anything that violates the contract (touches a
row count, emits a non-numeric column, or crashes) is rejected and the agent falls
back to the default; a provider failure degrades without raising; and the module
can never reach the held-out split. The LLM path is exercised with monkeypatched
fakes so the suite stays offline.
"""
from __future__ import annotations

import inspect
import json
import textwrap

import pytest

from crewml import config, llm
from crewml.crew import feature_engineer as fe
from crewml.crew.feature_engineer import (
    DEFAULT_FE_FEATURE,
    DEFAULT_FE_SOURCE,
    run_feature_engineer,
)
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.datasets import REGISTRY, load_train

KEY = "credit-g"


def _plan(key: str = KEY) -> dict:
    return build_plan(build_profile(REGISTRY[key], load_train(key)))


def _fake_llm(code: str):
    """A monkeypatch replacement for llm.chat that returns fixed FE code."""
    def _chat(*_a, **_k):
        return llm.LLMResult(
            text="```python\n" + code + "\n```",
            provider="groq", model="llama-3.3-70b-versatile",
            prompt_tokens=50, completion_tokens=20,
        )
    return _chat


# --- The deterministic default honours the contract -------------------------

def test_default_fe_passes_sandbox_validation():
    verdict = fe._validate_fe(DEFAULT_FE_SOURCE, KEY)
    assert verdict["ok"] is True
    assert DEFAULT_FE_FEATURE in verdict["new_columns"]
    assert verdict["rows_preserved"] is True and verdict["all_numeric"] is True
    assert verdict["n_rows_out"] == verdict["n_rows_in"]


def test_default_used_in_mock_mode(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: True)
    out = run_feature_engineer(_plan(), KEY)
    assert out["code"] == DEFAULT_FE_SOURCE
    assert out["meta"]["source"] == "default"
    assert out["meta"]["reason"] == "mock_mode"
    assert out["meta"]["validation"]["ok"] is True


def test_default_used_when_llm_disabled():
    out = run_feature_engineer(_plan(), KEY, with_llm=False)
    assert out["code"] == DEFAULT_FE_SOURCE
    assert out["meta"]["source"] == "default"


# --- A live provider's code is used only after it validates ------------------

def test_valid_llm_code_is_used(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    good = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd

        def add_features(df):
            out = df.copy()
            num = df.select_dtypes(include=["number"])
            out["num_col_sum"] = num.sum(axis=1)
            return out
        """
    )
    monkeypatch.setattr(llm, "chat", _fake_llm(good))
    out = run_feature_engineer(_plan(), KEY, with_llm=True)
    assert out["meta"]["source"] == "llm"
    assert out["code"].strip() == good.strip()   # extract_python strips surrounding whitespace
    assert out["meta"]["validation"]["ok"] is True
    assert "num_col_sum" in out["meta"]["validation"]["new_columns"]
    assert out["meta"]["prompt_tokens"] == 50


def test_llm_code_that_breaks_the_contract_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    # Drops rows -> violates the row-count contract; must be rejected.
    bad = textwrap.dedent(
        """\
        import pandas as pd

        def add_features(df):
            return df.iloc[:-5].copy()
        """
    )
    monkeypatch.setattr(llm, "chat", _fake_llm(bad))
    out = run_feature_engineer(_plan(), KEY, with_llm=True)
    assert out["meta"]["source"] == "fallback"
    assert out["code"] == DEFAULT_FE_SOURCE          # fell back to the safe default
    assert out["meta"]["llm_validation"]["ok"] is False   # the rejection is recorded
    assert out["meta"]["validation"]["ok"] is True        # the default we used passed


def test_llm_code_with_nonnumeric_column_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    bad = textwrap.dedent(
        """\
        import pandas as pd

        def add_features(df):
            out = df.copy()
            out["a_string_feature"] = "not numeric"
            return out
        """
    )
    monkeypatch.setattr(llm, "chat", _fake_llm(bad))
    out = run_feature_engineer(_plan(), KEY, with_llm=True)
    assert out["meta"]["source"] == "fallback"
    assert out["code"] == DEFAULT_FE_SOURCE


def test_provider_failure_degrades_without_crashing(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)

    def boom(*_a, **_k):
        raise RuntimeError("provider restricted")

    monkeypatch.setattr(llm, "chat", boom)
    out = run_feature_engineer(_plan(), KEY, with_llm=True)   # must not raise
    assert out["meta"]["source"] == "fallback"
    assert "provider restricted" in out["meta"]["fallback_reason"]
    assert out["code"] == DEFAULT_FE_SOURCE


# --- Structure / honesty ----------------------------------------------------

def test_meta_is_json_safe():
    out = run_feature_engineer(_plan(), KEY, with_llm=False)
    json.dumps(out["meta"])   # no numpy scalars / non-serialisable objects leak in


def test_source_never_loads_the_holdout():
    src = inspect.getsource(fe)
    assert "load_holdout" not in src
    assert "holdout_path" not in src


# --- Finiteness gate (added Day 20 — closes the one real fatal live failure) --
#
# A live Phase-3 run lost a whole dataset because generated FE code built an
# unguarded ratio that produced +/-inf: SimpleImputer runs with
# force_all_finite="allow-nan", so it replaced NaN but passed inf through to
# sklearn's finite assertion, which killed the fit. The old check here was a nan
# check misleadingly named `all_finite_ok`, so infinity was never screened.

_INF_FE = """\
import pandas as pd


def add_features(df):
    out = df.copy()
    num = df.select_dtypes(include="number")
    col = num.columns[0]
    out["unguarded_ratio"] = df[col] / (df[col] - df[col])
    return out
"""

_NAN_FE = """\
import pandas as pd
import numpy as np


def add_features(df):
    out = df.copy()
    num = df.select_dtypes(include="number")
    col = num.columns[0]
    keep = df[col].copy().astype("float64")
    keep.iloc[0] = np.nan          # NaN must stay acceptable — imputers handle it
    out["mostly_present"] = keep
    return out
"""


def test_infinite_engineered_column_fails_validation():
    verdict = fe._validate_fe(_INF_FE, KEY)
    assert verdict["executed_ok"] is True      # the code RUNS; that was the trap
    assert verdict["no_infinities"] is False
    assert verdict["ok"] is False              # ...but must not be trusted
    assert "unguarded_ratio" in verdict["infinite_columns"]


def test_nan_engineered_column_still_passes_validation():
    verdict = fe._validate_fe(_NAN_FE, KEY)
    assert verdict["no_infinities"] is True
    assert verdict["ok"] is True


def test_infinite_generated_code_is_rejected_and_falls_back(monkeypatch):
    monkeypatch.setattr(config, "is_mock_mode", lambda: False)
    monkeypatch.setattr(llm, "chat", _fake_llm(_INF_FE))
    out = run_feature_engineer(_plan(), KEY, with_llm=True, self_repair=False)
    assert out["meta"]["source"] == "fallback"
    assert out["meta"]["llm_validation"]["no_infinities"] is False
    assert out["code"] == DEFAULT_FE_SOURCE


def test_repair_prompt_names_the_offending_infinite_columns():
    """The self-repair loop can only fix what the error text tells it."""
    verdict = fe._validate_fe(_INF_FE, KEY)
    message = fe._verdict_error(verdict)
    assert "no_infinities=False" in message
    assert "unguarded_ratio" in message
    assert "safe denominator" in message
