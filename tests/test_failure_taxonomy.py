"""Day 17 — failure-taxonomy tests.

Everything here is synthetic-record and pure-function level: the classifier, the solo
adapter, the reporter-record adapter, the leak-frame builder, the probe dataset scoping,
the aggregation and the board rendering. No test re-runs the crew or reads the
git-ignored ``artifacts/`` archive, so the suite passes on a fresh clone.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from crewml.config import DATA_DIR
from crewml.datasets import REGISTRY, TARGET_COLUMN
from crewml.failure_taxonomy import (
    CATEGORIES,
    LEAK_COLUMN,
    OUTCOMES,
    SUBTLE_LEAK_AGREEMENT,
    SUBTLE_LEAK_CORR,
    _event,
    assemble_report,
    classify_run,
    classify_solo,
    make_leak_frame,
    mine_archive,
    probe_dataset,
    render_markdown,
    run_record_probes,
    summarise,
    _reporter_record_to_state,
)


# --- Synthetic record builders ----------------------------------------------

def _narr(reason=None):
    if reason is None:
        return {"source": "groq", "is_mock": False, "text": "fine"}
    return {"source": "unavailable", "is_mock": False, "reason": reason, "text": None}


def _critique(codes=(), decision="finalize", reason="no actionable failure modes found", cv=0.8):
    return {
        "decision": decision,
        "reason": reason,
        "cv_score": cv,
        "finding_codes": list(codes),
        "diagnoses": [{"code": c, "detail": f"detail for {c}"} for c in codes],
        "llm_narrative": _narr(),
    }


def _record(**over):
    rec = {
        "dataset_key": "credit-g",
        "profile": {"llm_narrative": _narr()},
        "plan": {"llm_narrative": _narr()},
        "fe_meta": {"source": "llm"},
        "training": {"ok": True, "timed_out": False, "error": None,
                     "metrics": {"best_cv_score": 0.8, "best_cv_std": 0.01}},
        "ensemble": {"attempted": True, "ok": True, "chosen": "single",
                     "improvement_over_single": 0.001,
                     "ensemble_cv_score": 0.801, "single_best_cv_score": 0.8},
        "critiques": [_critique()],
        "iteration": 1,
    }
    rec.update(over)
    return rec


# --- Taxonomy integrity ------------------------------------------------------

def test_taxonomy_is_closed_and_complete():
    for code, meta in CATEGORIES.items():
        assert set(meta) == {"group", "stage", "surface", "description"}, code
    with pytest.raises(ValueError):
        _event("not_a_category", "fatal", dataset="d", run="r", system="crew", evidence="e")
    with pytest.raises(ValueError):
        _event("exec_error", "not_an_outcome", dataset="d", run="r", system="crew", evidence="e")


def test_clean_run_yields_no_events():
    assert classify_run(_record(), run="t") == []


# --- Provider outage ---------------------------------------------------------

def test_provider_failure_is_one_handled_event():
    rec = _record(
        profile={"llm_narrative": _narr("BadRequestError: organization_restricted")},
        fe_meta={"source": "fallback", "fallback_reason": "BadRequestError: organization_restricted"},
    )
    events = classify_run(rec, run="t")
    assert [e["category"] for e in events] == ["provider_outage"]
    assert events[0]["outcome"] == "handled"
    assert "2 LLM surface(s)" in events[0]["evidence"]


def test_mock_mode_and_disabled_are_not_outages():
    rec = _record(profile={"llm_narrative": _narr("mock_mode")},
                  plan={"llm_narrative": _narr("disabled")})
    assert classify_run(rec, run="t") == []


# --- Execution failures ------------------------------------------------------

def test_training_timeout_is_fatal_exec_timeout():
    rec = _record(training={"ok": False, "timed_out": True,
                            "error": "execution exceeded timeout of 5s", "metrics": {}},
                  critiques=[_critique(codes=("execution_error",))])
    cats = {e["category"]: e["outcome"] for e in classify_run(rec, run="t")}
    assert cats["exec_timeout"] == "fatal"
    assert "exec_error" not in cats


def test_training_crash_is_fatal_exec_error():
    rec = _record(training={"ok": False, "timed_out": False,
                            "error": "ValueError: boom", "metrics": {}})
    events = classify_run(rec, run="t")
    assert [(e["category"], e["outcome"]) for e in events] == [("exec_error", "fatal")]
    assert "boom" in events[0]["evidence"]


def test_failed_ensemble_is_handled_not_fatal():
    rec = _record(ensemble={"attempted": True, "ok": False, "timed_out": False,
                            "error": "crashed", "chosen": None})
    events = classify_run(rec, run="t")
    assert [(e["category"], e["outcome"]) for e in events] == [("exec_error", "handled")]


# --- Critic findings: recovered vs persisted vs budget-cut -------------------

def test_recovered_finding_is_handled():
    rec = _record(critiques=[
        _critique(codes=("underfit",), decision="iterate", reason="actionable issue(s)", cv=0.05),
        _critique(codes=(), cv=0.82),
    ])
    events = classify_run(rec, run="t")
    assert [(e["category"], e["outcome"]) for e in events] == [("plan_underfit", "handled")]


def test_budget_cut_finding_is_degraded_plus_budget_cutoff():
    rec = _record(critiques=[
        _critique(codes=("underfit",), decision="finalize",
                  reason="iteration budget reached (1/1) — finalising", cv=0.02),
    ])
    cats = {e["category"]: e["outcome"] for e in classify_run(rec, run="t")}
    assert cats["plan_underfit"] == "degraded"
    assert cats["budget_cutoff"] == "degraded"


def test_single_pass_finding_is_detected():
    rec = _record(critiques=[_critique(codes=("overfit",), cv=0.7)])
    events = classify_run(rec, run="t")
    assert [(e["category"], e["outcome"]) for e in events] == [("plan_overfit_variance", "detected")]


def test_loop_without_actuator_is_detected():
    rec = _record(critiques=[
        _critique(codes=("underfit",), decision="iterate", reason="actionable issue(s)", cv=0.05),
        _critique(codes=("underfit",), decision="finalize",
                  reason="diminishing returns — CV moved +0.0000", cv=0.05),
    ])
    cats = {e["category"] for e in classify_run(rec, run="t")}
    assert "loop_no_actuator" in cats
    assert "plan_underfit" in cats


def test_ensemble_regression_is_handled():
    rec = _record(ensemble={"attempted": True, "ok": True, "chosen": "single",
                            "improvement_over_single": -0.005,
                            "ensemble_cv_score": 0.795, "single_best_cv_score": 0.8})
    events = classify_run(rec, run="t")
    assert [(e["category"], e["outcome"]) for e in events] == [("ensemble_regression", "handled")]


def test_census_can_never_emit_missed():
    # ``missed`` requires ground truth only a probe has; even a leaky-looking record
    # classifies as flagged/detected, never missed.
    rec = _record(critiques=[_critique(codes=("leakage",), cv=0.999)])
    assert all(e["outcome"] != "missed" for e in classify_run(rec, run="t"))


# --- Solo adapter ------------------------------------------------------------

def test_classify_solo_maps_and_dedupes():
    solo = {
        "datasets": {
            "ok_ds": {"ok": True, "value": 0.8},
            "kin8nm": {"ok": False, "error": "ValueError: Invalid parameter 'alpha' for estimator"},
        },
        "failures": {
            "vehicle": "TimeoutExpired: Command timed out after 120 seconds",
            "kin8nm": "solo script failed: Invalid parameter 'alpha'",
            "other": "RuntimeError: exploded",
        },
    }
    events = classify_solo(solo)
    by_ds = {e["dataset"]: e for e in events}
    assert len(events) == 3  # kin8nm deduped
    assert by_ds["kin8nm"]["category"] == "plan_search_invalid"
    assert by_ds["vehicle"]["category"] == "exec_timeout"
    assert by_ds["other"]["category"] == "exec_error"
    assert all(e["outcome"] == "fatal" and e["system"] == "solo" for e in events)


# --- Reporter-record adapter -------------------------------------------------

def test_reporter_record_adapter_classifies_like_a_state():
    report = {
        "dataset_key": "vehicle",
        "critic_passes": [{"decision": "finalize", "reason": "clean", "cv_score": 0.9,
                           "finding_codes": []}],
        "training": {"ok": True, "cv_score": 0.9},
        "ensemble": {"attempted": True, "ok": True, "chosen": "ensemble",
                     "improvement_over_single": 0.004},
        "llm_usage": {"narratives": [
            {"node": "profiler", "source": "unavailable", "reason": "mock_mode"},
        ]},
    }
    assert classify_run(_reporter_record_to_state(report), run="t") == []

    report["training"] = {"ok": False, "error": "boom"}
    events = classify_run(_reporter_record_to_state(report), run="t")
    assert ("exec_error", "fatal") in [(e["category"], e["outcome"]) for e in events]


# --- Leak-frame builder ------------------------------------------------------

def _regression_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n),
                         TARGET_COLUMN: rng.standard_normal(n) * 10 + 50})


def _classification_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.standard_normal(n),
                         TARGET_COLUMN: rng.choice(["good", "bad"], size=n, p=[0.7, 0.3])})


def test_blatant_regression_leak_is_a_perfect_copy():
    frame, facts = make_leak_frame(_regression_df(), task="regression", kind="blatant",
                                   rng=np.random.default_rng(1))
    assert facts["measured_signal"] == pytest.approx(1.0)
    assert (frame[LEAK_COLUMN] == frame[TARGET_COLUMN]).all()


def test_subtle_regression_leak_sits_below_the_profiler_screen():
    frame, facts = make_leak_frame(_regression_df(), task="regression", kind="subtle",
                                   rng=np.random.default_rng(1))
    # Inside the window: strong signal, but under the 0.98 Pearson screen.
    assert 0.80 <= facts["measured_signal"] < 0.98
    assert abs(facts["measured_signal"] - SUBTLE_LEAK_CORR) < 0.05


def test_subtle_classification_leak_agreement_and_no_zeros():
    frame, facts = make_leak_frame(_classification_df(), task="classification", kind="subtle",
                                   rng=np.random.default_rng(1))
    assert facts["measured_signal"] == pytest.approx(SUBTLE_LEAK_AGREEMENT, abs=0.01)
    # Codes are offset to 1..k so the disguised-missing (zero-inflation) heuristic
    # cannot fire on the probe column and corrupt the experiment.
    assert (frame[LEAK_COLUMN] > 0).all()


def test_blatant_classification_leak_is_exact():
    frame, facts = make_leak_frame(_classification_df(), task="classification", kind="blatant",
                                   rng=np.random.default_rng(1))
    assert facts["measured_signal"] == pytest.approx(1.0)


# --- Probe dataset scoping ---------------------------------------------------

def test_probe_dataset_registers_and_cleans_up():
    frame = _regression_df()
    key = "probe_unittest_scope"
    try:
        with probe_dataset("cpu_small", key, frame) as spec:
            assert key in REGISTRY and spec.metric == "r2" and spec.version == 0
            assert (DATA_DIR / key / "train.parquet").exists()
            assert not (DATA_DIR / key / "holdout.parquet").exists()  # probes HAVE no holdout
        assert key not in REGISTRY
    finally:
        shutil.rmtree(DATA_DIR / key, ignore_errors=True)


def test_probe_dataset_rejects_unprefixed_keys():
    with pytest.raises(ValueError):
        with probe_dataset("cpu_small", "not_a_probe", _regression_df()):
            pass  # pragma: no cover


# --- Record-level probes (the Critic's real detector) ------------------------

def test_record_probes_fire_the_real_detector():
    record = {
        "profile": {"metric": "r2", "assessment": {"flags": []},
                    "leakage_checks": {"target_correlated_features": []}},
        "plan": {"metric": "r2", "cv": {"scoring": "r2"}, "drop_columns": [],
                 "candidate_models": [], "imbalance_strategy": {}},
        "training": {"ok": True, "dataset_key": "x", "best_model": "rf",
                     "metrics": {"best_cv_score": 0.8, "best_cv_std": 0.01}},
    }
    probes = run_record_probes(record)
    assert {p["probe"] for p in probes} == {"wrong_metric", "exec_error"}
    assert all(p["detected"] and p["record_level"] for p in probes)
    # The original record is never mutated in place.
    assert record["plan"]["cv"]["scoring"] == "r2" and record["training"]["ok"] is True


# --- Aggregation + rendering -------------------------------------------------

def test_summarise_counts_by_category_outcome_and_system():
    events = [
        _event("exec_error", "fatal", dataset="a", run="r1", system="solo", evidence="e"),
        _event("exec_error", "fatal", dataset="b", run="r2", system="solo", evidence="e"),
        _event("provider_outage", "handled", dataset="a", run="r3", system="crew", evidence="e"),
    ]
    s = summarise(events)
    assert s["n_events"] == 3
    assert s["by_category"]["exec_error"]["total"] == 2
    assert s["by_category"]["exec_error"]["by_system"] == {"solo": 2}
    assert s["by_outcome"]["fatal"] == 2 and s["by_outcome"]["handled"] == 1
    assert s["fatal_by_system"] == {"solo": 2}


def test_mine_archive_is_graceful_without_artifacts():
    # On any machine this at least returns the census shape; a fresh clone has zero runs.
    archive = mine_archive()
    assert set(archive) == {"n_crew_runs", "n_solo_runs", "sources", "events"}
    assert all(e["category"] in CATEGORIES and e["outcome"] in OUTCOMES
               for e in archive["events"])


def test_render_markdown_carries_the_honesty_framing():
    archive = {"n_crew_runs": 1, "n_solo_runs": 1,
               "sources": [], "events": [
                   _event("exec_timeout", "fatal", dataset="vehicle", run="day03_solo",
                          system="solo", evidence="timed out")]}
    leak = [{
        "probe": "leak_subtle", "base_dataset": "credit-g",
        "ground_truth": {"measure": "agreement_with_target", "measured_signal": 0.95},
        "expectation": "missed (engineered inside the detection window)",
        "profiler_flagged": False, "plan_dropped": False, "critic_leakage_finding": False,
        "detected": False, "model_saw_leak": True, "cv_score": 0.97, "cv_metric": "roc_auc",
        "iterations_run": 1, "mock": True, "seconds": 1.0,
        "events": [_event("leakage_missed", "missed", dataset="probe_leak_subtle",
                          run="day17", system="crew", evidence="nothing fired")],
    }]
    timeout = {"probe": "exec_timeout", "base_dataset": "cpu_small", "ok": True, "cap_s": 5,
               "expectation": "trainer killed at the cap", "trainer_timed_out": True,
               "critic_filed_blocker": True, "detected": True, "mock": True, "seconds": 30.0,
               "events": []}
    report = assemble_report(archive, leak, timeout, run_record_probes({
        "profile": {"metric": "r2", "assessment": {"flags": []},
                    "leakage_checks": {"target_correlated_features": []}},
        "plan": {"metric": "r2", "cv": {"scoring": "r2"}, "drop_columns": [],
                 "candidate_models": [], "imbalance_strategy": {}},
        "training": {"ok": True, "dataset_key": "x", "best_model": "rf",
                     "metrics": {"best_cv_score": 0.8, "best_cv_std": 0.01}},
    }))
    md = render_markdown(report)
    assert "## The taxonomy" in md
    assert "## Archive census" in md
    assert "## Injection probes" in md
    # The subtle-leak miss must surface as the honest finding, not be buried.
    assert "The measured detection window" in md
    assert "`missed` is only ever assigned" in md
