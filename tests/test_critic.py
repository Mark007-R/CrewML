"""Day 10 guards: the Critic diagnoses honestly, decides soundly, and closes the loop.

The Critic is the node that turns the crew from a straight line into a feedback
system, so these assert three things:

  * **Diagnosis** — each failure mode (overfit / underfit / leakage / imbalance /
    wrong-metric) is detected from the CV-visible evidence, and each finding embeds the
    exact keyword the Planner's ``_apply_critique`` acts on (an integration test proves
    the hand-off actually lands as a plan change).
  * **Decision** — iterate only when there is an actionable issue under budget and the
    loop is still making progress; finalize on a clean run, an execution failure, a spent
    budget, or diminishing returns. A monkeypatched full-graph run proves the loop opens
    *and closes itself* by convergence, before the ``max_iterations`` guard is reached.
  * **Honesty** — the Critic reasons over CV-on-train only, never loads the held-out
    split (source-inspected), and its advisory narrative is ``unavailable`` offline.

The unit-level diagnosis/decision tests are pure and fast; the one end-to-end loop test
runs real (offline, no-grid-search) training passes.
"""
from __future__ import annotations

import inspect

import pytest

from crewml.crew import critic as cr
from crewml.crew.critic import build_critique, decide, diagnose, run_critic
from crewml.crew.planner import build_plan


# --- Fixtures: minimal profile / plan / training dicts ----------------------

def _profile(*, metric="roc_auc", flags=None, leak_cols=None):
    return {
        "metric": metric,
        "assessment": {"flags": list(flags or [])},
        "leakage_checks": {"target_correlated_features": [{"column": c} for c in (leak_cols or [])]},
    }


def _plan(*, metric="roc_auc", drop=None, scoring=None, imb_recommended=True, winner_cw=True):
    return {
        "metric": metric,
        "drop_columns": list(drop or []),
        "cv": {"scoring": scoring or metric},
        "imbalance_strategy": {"recommended": imb_recommended},
        "candidate_models": [{"name": "random_forest", "supports_class_weight": winner_cw}],
    }


def _training(*, ok=True, mean=0.80, std=0.02, best="random_forest", error=None):
    return {
        "ok": ok,
        "dataset_key": "x",
        "best_model": best if ok else None,
        "error": error,
        "cv_score": mean if ok else None,
        "cv_score_is_holdout": False,
        "metrics": {"best_cv_score": mean, "best_cv_std": std} if ok else {},
    }


def _codes(findings):
    return [f["code"] for f in findings]


# --- Diagnosis: a clean run flags nothing -----------------------------------

def test_clean_run_has_no_findings():
    assert diagnose(_profile(), _plan(), _training(mean=0.80, std=0.02)) == []


# --- Diagnosis: each failure mode, and its Planner keyword ------------------

def test_overfit_from_high_cv_variance():
    f = diagnose(_profile(), _plan(), _training(mean=0.80, std=0.20))  # std/mean = 0.25
    assert _codes(f) == ["overfit"]
    assert f[0]["keyword"] == "overfit"


def test_underfit_from_low_absolute_score():
    f = diagnose(_profile(), _plan(), _training(mean=0.55, std=0.01))
    assert _codes(f) == ["underfit"]
    assert f[0]["keyword"] == "underfit"


def test_leakage_from_residual_undropped_suspect_column():
    # A target-leakage suspect the plan did NOT drop -> leakage finding.
    f = diagnose(_profile(leak_cols=["ssn"]), _plan(drop=[]), _training(mean=0.82, std=0.02))
    assert "leakage" in _codes(f)
    assert next(d for d in f if d["code"] == "leakage")["keyword"] == "leak"


def test_leakage_suspect_dropped_is_not_reflagged():
    # Same suspect, but the plan already dropped it -> no residual leakage finding.
    f = diagnose(_profile(leak_cols=["ssn"]), _plan(drop=["ssn"]), _training(mean=0.82, std=0.02))
    assert "leakage" not in _codes(f)


def test_leakage_from_implausibly_high_score():
    # Too-good-to-be-true CV score is the runtime fingerprint of leakage.
    f = diagnose(_profile(), _plan(), _training(mean=0.999, std=0.001))
    assert "leakage" in _codes(f)
    # ...and the ceiling case suppresses the overfit/underfit signals (evidence dominated).
    assert "overfit" not in _codes(f) and "underfit" not in _codes(f)


def test_imbalance_when_plan_did_not_enable_weighting():
    f = diagnose(_profile(flags=["class_imbalance"]), _plan(imb_recommended=False), _training())
    assert "imbalance" in _codes(f)
    assert next(d for d in f if d["code"] == "imbalance")["keyword"] == "imbalance"


def test_imbalance_when_winner_cannot_take_class_weight():
    # Balancing was recommended but the winning model can't use it -> still unhandled.
    plan = _plan(imb_recommended=True, winner_cw=False)
    f = diagnose(_profile(flags=["class_imbalance"]), plan, _training(best="random_forest"))
    assert "imbalance" in _codes(f)


def test_imbalance_handled_is_not_flagged():
    plan = _plan(imb_recommended=True, winner_cw=True)
    f = diagnose(_profile(flags=["class_imbalance"]), plan, _training())
    assert "imbalance" not in _codes(f)


def test_wrong_metric_when_scorer_mismatches():
    f = diagnose(_profile(metric="roc_auc"), _plan(metric="roc_auc", scoring="accuracy"), _training())
    assert "wrong_metric" in _codes(f)
    assert next(d for d in f if d["code"] == "wrong_metric")["keyword"] == "metric"


def test_execution_failure_short_circuits_diagnosis():
    f = diagnose(_profile(flags=["class_imbalance"]), _plan(), _training(ok=False, error="boom"))
    assert _codes(f) == ["execution_error"]
    assert f[0]["keyword"] is None       # a crash isn't a plan-level fix (Day-20 self-repair)


# --- The hand-off actually lands: findings drive a Planner plan change -------

def test_findings_embed_the_planner_keyword_and_planner_acts_on_them():
    # Build a real critique with an overfit finding, feed it to the Planner, and assert
    # the plan changes the way _apply_critique promises (capacity down / regularisation up).
    crit = build_critique(
        _profile(), _plan(), _training(mean=0.80, std=0.20),
        critiques_so_far=[], iteration=1, max_iterations=3,
    )
    assert crit["findings"] and all(
        any(k in f.lower() for k in ("overfit", "underfit", "leak", "imbalance", "metric"))
        for f in crit["findings"]
    )
    # A real profile-derived plan, then the same plan rebuilt WITH the critique.
    from crewml.crew.profiler import build_profile
    from crewml.datasets import REGISTRY, load_train

    prof = build_profile(REGISTRY["credit-g"], load_train("credit-g"))
    adjusted = build_plan(prof, critique=crit, iteration=1)
    assert adjusted["addressed_critique"] is crit
    assert "critique_adjustments" in adjusted
    assert any("overfit" in a.lower() or "regularis" in a.lower() for a in adjusted["critique_adjustments"])
    # The overfit response tightens the RF grid (shallower / larger leaves).
    rf = next(m for m in adjusted["candidate_models"] if m["name"] == "random_forest")
    assert rf["param_grid"]["model__min_samples_leaf"] == [4, 8]


# --- Decision logic ---------------------------------------------------------

def test_decide_finalizes_on_clean_run():
    decision, reason, delta = decide([], _training(), [], iteration=1, max_iterations=3)
    assert decision == "finalize" and "clean" in reason


def test_decide_finalizes_on_execution_failure():
    f = diagnose(_profile(), _plan(), _training(ok=False, error="boom"))
    decision, _, _ = decide(f, _training(ok=False), [], iteration=1, max_iterations=3)
    assert decision == "finalize"


def test_decide_iterates_on_actionable_under_budget():
    f = diagnose(_profile(), _plan(), _training(mean=0.80, std=0.20))  # overfit
    decision, reason, _ = decide(f, _training(mean=0.80, std=0.20), [], iteration=1, max_iterations=3)
    assert decision == "iterate" and "overfit" in reason


def test_decide_finalizes_when_budget_reached_even_with_findings():
    f = diagnose(_profile(), _plan(), _training(mean=0.80, std=0.20))
    decision, reason, _ = decide(f, _training(mean=0.80, std=0.20), [], iteration=3, max_iterations=3)
    assert decision == "finalize" and "budget" in reason


def test_decide_finalizes_on_diminishing_returns():
    # Previous pass logged the same score + same codes; no improvement, no new issue.
    prev = {"cv_score": 0.80, "finding_codes": ["overfit"]}
    f = diagnose(_profile(), _plan(), _training(mean=0.80, std=0.20))
    decision, reason, delta = decide(
        f, _training(mean=0.80, std=0.20), [prev], iteration=2, max_iterations=5,
    )
    assert decision == "finalize" and "diminishing" in reason
    assert delta == pytest.approx(0.0, abs=1e-9)


def test_decide_iterates_when_score_still_improving():
    prev = {"cv_score": 0.70, "finding_codes": ["overfit"]}
    f = diagnose(_profile(), _plan(), _training(mean=0.80, std=0.20))
    decision, _, delta = decide(
        f, _training(mean=0.80, std=0.20), [prev], iteration=2, max_iterations=5,
    )
    assert decision == "iterate"
    assert delta == pytest.approx(0.10, abs=1e-9)


def test_decide_iterates_on_a_newly_found_issue_even_without_score_gain():
    # Score flat, but a NEW failure mode surfaced this pass -> worth one more look.
    prev = {"cv_score": 0.80, "finding_codes": ["overfit"]}
    f = diagnose(_profile(flags=["class_imbalance"]), _plan(imb_recommended=False),
                 _training(mean=0.80, std=0.20))
    assert "imbalance" in _codes(f)
    decision, _, _ = decide(f, _training(mean=0.80, std=0.20), [prev], iteration=2, max_iterations=5)
    assert decision == "iterate"


# --- build_critique records what the next pass needs ------------------------

def test_build_critique_records_score_and_codes_for_progress_tracking():
    crit = build_critique(
        _profile(), _plan(), _training(mean=0.80, std=0.20),
        critiques_so_far=[], iteration=1, max_iterations=3,
    )
    assert crit["stub"] is False
    assert crit["cv_score"] == 0.80
    assert crit["finding_codes"] == ["overfit"]
    assert crit["decision"] == "iterate"
    assert crit["iteration"] == 1


# --- Honesty: CV-on-train only; no held-out; offline narrative --------------

def test_run_critic_narrative_unavailable_when_disabled():
    crit = run_critic(_profile(), _plan(), _training(), iteration=1, with_llm=False)
    assert crit["llm_narrative"]["source"] == "unavailable"
    assert crit["llm_narrative"]["text"] is None


def test_critic_reasons_over_cv_not_holdout():
    crit = run_critic(_profile(), _plan(), _training(mean=0.80), iteration=1, with_llm=False)
    # The score the Critic reasons over is the Trainer's CV estimate (labelled not-holdout
    # upstream); the Critic itself never introduces a held-out number.
    assert crit["cv_score"] == 0.80


def test_critic_source_never_loads_the_holdout():
    src = inspect.getsource(cr)
    assert "load_holdout" not in src
    assert "holdout" not in src.lower()


# --- End-to-end: the loop opens on a finding and closes itself by convergence -

def test_full_loop_iterates_then_self_finalizes_before_the_guard(monkeypatch):
    """A real (offline) crew run: the Critic finds an issue on pass 1 and iterates, the
    Planner responds, pass 2 is clean so the Critic finalises — all under a generous
    budget, proving the loop closes by its *own* logic, not the max_iterations backstop.
    """
    monkeypatch.setenv("CREWML_PROFILER_LLM", "0")
    monkeypatch.setenv("CREWML_PLANNER_LLM", "0")
    monkeypatch.setenv("CREWML_FE_LLM", "0")
    monkeypatch.setenv("CREWML_CRITIC_LLM", "0")
    monkeypatch.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")

    # Inject one actionable finding on the first Critic pass only; clean thereafter.
    real_diagnose = cr.diagnose
    calls = {"n": 0}

    def fake_diagnose(profile, plan, training):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{
                "code": "overfit", "keyword": "overfit", "severity": "medium",
                "detail": "injected overfit signal for the loop test.",
                "directive": "reduce capacity and strengthen regularisation.",
            }]
        return real_diagnose(profile, plan, training)  # pass 2 is a clean credit-g run

    monkeypatch.setattr(cr, "diagnose", fake_diagnose)

    from crewml.crew import build_crew, initial_state
    from crewml.datasets import REGISTRY

    app = build_crew()
    st = initial_state(REGISTRY["credit-g"], max_iterations=5)  # generous budget
    final = app.invoke(st, config={"recursion_limit": 60})

    # Two passes: iterate (pass 1) then finalize (pass 2) — well under the budget of 5.
    assert final["iteration"] == 2
    assert final["max_iterations"] == 5
    assert [c["decision"] for c in final["critiques"]] == ["iterate", "finalize"]
    # The loop actually looped: the Planner ran twice, and the crew still terminated.
    assert final["trace"].count("planner") == 2
    assert final["trace"][-1] == "reporter"
    # Pass 1's iterate carried a real directive into the Planner's second plan.
    assert final["plan"].get("critique_adjustments")
