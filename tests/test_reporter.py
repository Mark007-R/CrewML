"""Day 11 guards: the Reporter synthesises an honest report + model card.

The Reporter is the crew's terminal node and it is deliberately deterministic — it
renders the final state, it doesn't reason. These assert:

  * **Faithful synthesis** — the report's final model, scores, per-candidate table, and
    Critic passes come straight from the state the specialists produced.
  * **Ensemble vs. single** — the shipped model reflects the Ensembler's decision, and
    falls back to the Trainer's single model when no ensemble was attempted.
  * **Honesty surfaced, not buried** — the model card states scores are CV-on-train (not
    held-out), flags a mock/degraded LLM run, and reports a training failure honestly.
  * **No peeking** — the module never loads the held-out split (source-inspected).

These are pure over crafted state dicts (no subprocess), so they are fast; the real
end-to-end render is exercised by ``tests/test_graph.py``.
"""
from __future__ import annotations

import inspect

from crewml.crew import reporter as rp
from crewml.crew.reporter import build_report, render_model_card


# --- Crafted terminal states -------------------------------------------------

def _state(*, ensemble=None, training=None, extra=None):
    st = {
        "dataset_key": "credit-g",
        "task": "classification",
        "subtype": "binary",
        "metric": "roc_auc",
        "iteration": 1,
        "max_iterations": 3,
        "profile": {
            "n_rows": 800, "n_features": 20,
            "assessment": {"flags": ["class_imbalance"]},
        },
        "plan": {
            "task": "classification", "subtype": "binary", "metric": "roc_auc",
            "drop_columns": [],
            "candidate_models": [{"name": "random_forest"}, {"name": "logistic_regression"},
                                 {"name": "hist_gradient_boosting"}],
            "cv": {"scheme": "StratifiedKFold", "n_splits": 5, "scoring": "roc_auc"},
            "imbalance_strategy": {"recommended": True},
            "llm_narrative": {"source": "unavailable", "text": None, "reason": "disabled"},
        },
        "fe_meta": {"source": "default", "is_mock": False,
                    "validation": {"new_columns": ["row_nan_count"]}},
        "training": training if training is not None else {
            "ok": True, "best_model": "random_forest", "cv_score": 0.7940,
            "error": None,
            "metrics": {"per_model": [
                {"name": "random_forest", "cv_mean": 0.7940, "cv_std": 0.0725, "best_params": {}},
                {"name": "logistic_regression", "cv_mean": 0.7872, "cv_std": 0.0677, "best_params": {}},
            ]},
        },
        "ensemble": ensemble if ensemble is not None else {
            "attempted": True, "ok": True, "chosen": "ensemble", "final_model_kind": "ensemble",
            "members": ["random_forest", "logistic_regression", "hist_gradient_boosting"],
            "single_best_model": "random_forest",
            "ensemble_cv_score": 0.7972, "single_best_cv_score": 0.7940,
            "improvement_over_single": 0.0032, "final_cv_score": 0.7972,
        },
        "critiques": [{
            "iteration": 1, "decision": "finalize", "reason": "clean run",
            "cv_score": 0.7940, "finding_codes": [],
            "llm_narrative": {"source": "unavailable", "text": None},
        }],
    }
    if extra:
        st.update(extra)
    return st


# --- Faithful synthesis ------------------------------------------------------

def test_report_is_real_and_names_the_reporter():
    report = build_report(_state())
    assert report["stub"] is False
    assert report["node"] == "reporter"
    assert report["dataset_key"] == "credit-g"
    assert report["cv_score_is_holdout"] is False


def test_final_model_reflects_the_ensemble_decision():
    report = build_report(_state())
    fm = report["final_model"]
    assert fm["kind"] == "ensemble"
    assert fm["cv_score"] == 0.7972
    assert fm["improvement_over_single"] == 0.0032
    assert set(fm["members"]) == {"random_forest", "logistic_regression", "hist_gradient_boosting"}


def test_falls_back_to_single_when_no_ensemble_attempted():
    ens = {"attempted": False, "ok": True, "chosen": "single", "final_model_kind": "single",
           "single_best_model": "random_forest", "single_best_cv_score": 0.7940,
           "final_cv_score": 0.7940, "reason": "only 1 candidate available"}
    report = build_report(_state(ensemble=ens))
    fm = report["final_model"]
    assert fm["kind"] == "single"
    assert fm["single_best_model"] == "random_forest"
    assert fm["cv_score"] == 0.7940


def test_per_candidate_and_critic_tables_come_from_state():
    report = build_report(_state())
    assert [r["name"] for r in report["training"]["per_model"]] == [
        "random_forest", "logistic_regression"
    ]
    assert report["critic_passes"][0]["decision"] == "finalize"
    assert report["final_decision"] == "finalize"


# --- The model card renders the honest story --------------------------------

def test_model_card_states_scores_are_cv_not_holdout():
    card = build_report(_state())["model_card_markdown"]
    assert "Model Card" in card
    assert "cross-validated estimate" in card
    assert "cv_score_is_holdout: false" in card
    assert "0.7972" in card               # the ensemble headline
    assert "ROC AUC" in card


def test_model_card_names_the_ensemble_members_and_gain():
    card = build_report(_state())["model_card_markdown"]
    assert "ensemble" in card.lower()
    assert "+0.0032" in card


def test_warnings_flag_no_live_llm_and_holdout_caveat():
    report = build_report(_state())
    joined = " ".join(report["warnings"])
    assert "held-out" in joined
    assert "deterministic core" in joined
    assert report["llm_usage"]["any_live"] is False
    assert report["llm_usage"]["n_unavailable"] >= 1


def test_mock_mode_is_flagged_in_the_report():
    st = _state()
    st["fe_meta"]["is_mock"] = True
    report = build_report(st)
    assert any("MOCK" in w for w in report["warnings"])


def test_training_failure_is_reported_honestly():
    failed = {"ok": False, "best_model": None, "cv_score": None, "error": "boom", "metrics": {}}
    ens = {"attempted": False, "ok": False, "chosen": None, "final_model_kind": None,
           "single_best_model": None, "final_cv_score": None}
    report = build_report(_state(training=failed, ensemble=ens))
    assert report["final_model"]["kind"] == "none"
    assert any("failed" in w.lower() for w in report["warnings"])
    card = report["model_card_markdown"]
    assert "No model" in card


def test_render_model_card_is_pure_and_stringy():
    report = build_report(_state())
    # render again from the same report -> identical string (deterministic, no I/O).
    assert render_model_card(report) == render_model_card(report)
    assert isinstance(render_model_card(report), str)


# --- LLM usage aggregation ---------------------------------------------------

def test_llm_usage_counts_live_narratives_and_tokens():
    st = _state()
    # A live planner narrative with token accounting.
    st["plan"]["llm_narrative"] = {"source": "groq", "model": "llama-3.3-70b-versatile",
                                   "prompt_tokens": 120, "completion_tokens": 80}
    usage = build_report(st)["llm_usage"]
    assert usage["any_live"] is True
    assert usage["n_live"] >= 1
    assert usage["prompt_tokens"] == 120 and usage["completion_tokens"] == 80


# --- No-peeking: the module never loads the held-out split ------------------

def test_reporter_source_never_loads_the_holdout():
    src = inspect.getsource(rp)
    assert "load_holdout" not in src
    assert "holdout_path" not in src
