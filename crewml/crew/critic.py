"""The Critic agent — the crew's reviewer, and the node that closes the loop (Day 10).

Up to Day 9 the crew was a straight line: Profiler → Planner → Feature Engineer →
Trainer. The Critic is the node that makes it a *feedback system*. It reads the
Trainer's cross-validated result together with the DataProfile and the ModelingPlan,
**diagnoses** the failure modes a competent ML reviewer would look for, decides
whether another pass is worth it (**iterate**) or the run is done (**finalize**), and
— when it iterates — hands the Planner a **specific directive** it already knows how
to act on. The Planner's response side was wired on Day 8 (``_apply_critique``); today
supplies the other half, so the loop runs end to end.

Same honesty discipline as the rest of the crew:

* **Deterministic core.** :func:`diagnose` derives every finding with plain rules over
  what is already in the state — the profile's flags, the plan's dispositions, and the
  Trainer's per-candidate CV metrics. No LLM decides whether to iterate, so the loop is
  reproducible and cannot be talked into spinning forever by a chatty model.
* **CV-visible signals only, honestly labelled.** The Critic never loads the held-out
  set (that is Phase 3), so it diagnoses from *cross-validation on train* — the only
  honest evidence it has at this stage. "Overfit" here means the CV-visible symptom
  (large fold-to-fold variance), not a train-vs-held-out gap; findings say so.
* **Findings speak the Planner's language.** Each finding string embeds the keyword the
  Planner's :func:`crewml.crew.planner._apply_critique` matches on
  (``overfit`` / ``underfit`` / ``leak`` / ``imbalance`` / ``metric``), so a diagnosis
  turns into a concrete plan adjustment without a brittle schema contract between the
  two nodes.
* **Convergence is a property, not a hope.** The Critic finalises when the run is clean,
  when a pass stopped making progress (the score didn't improve and the same issues
  recur — diminishing returns), or when it cannot productively iterate (an execution
  failure, which self-repair addresses on Day 20). The router's ``max_iterations`` guard
  (Day 5) is the hard backstop on top of this.

* **Optional LLM narrative.** When a live provider is configured, :func:`run_critic`
  layers a short review note *on top of* — never in place of — the deterministic
  verdict, tagged with provider/model/token cost. In mock mode (or on any error) the
  narrative is ``unavailable`` and the decision stands on its deterministic core. The
  narrative never changes the decision or the findings.

**Train only, structurally.** The Critic reads three dicts (profile, plan, training)
and nothing else — no data loader, no held-out split. A source-inspection test asserts
the module never names the locked test split; the no-peeking invariant is a property of
the code, not a rule a node has to remember.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from crewml import config, llm

CRITIQUE_SCHEMA_VERSION = 1

# --- Diagnostic thresholds (conservative, so a healthy run finalises cleanly) ---
# Fold-to-fold instability: winner cv_std / |cv_mean| at/above this reads as an
# overfitting / high-variance risk the Planner can answer by pulling capacity down.
OVERFIT_CV_VARIATION = 0.15
# Absolute-score floors below which the models are failing to capture signal
# (underfit) -> the Planner answers by adding capacity. Per primary metric.
UNDERFIT_FLOOR = {
    "roc_auc": 0.60,     # barely above the 0.5 coin-flip
    "f1_macro": 0.50,    # weak for the multiclass sets we use
    "r2": 0.10,          # explains almost none of the variance
}
# Implausibly-good CV score => the runtime fingerprint of leakage the crew introduced
# or the Profiler missed. Per primary metric.
LEAKAGE_CEILING = {
    "roc_auc": 0.995,
    "f1_macro": 0.995,
    "r2": 0.999,
}
# Minimum CV improvement over the previous pass to count as "still making progress".
SCORE_IMPROVE_EPS = 0.002

# The sklearn scorer each primary metric must map to (mirror of the Planner's map) —
# used only for the wrong-metric sanity guard.
_EXPECTED_SCORER = {"roc_auc": "roc_auc", "f1_macro": "f1_macro", "r2": "r2"}


# --- Deterministic diagnosis -------------------------------------------------

def _winner_cv(training: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """The winning candidate's (cv_mean, cv_std) from the Trainer metrics, if present."""
    metrics = training.get("metrics") or {}
    mean = metrics.get("best_cv_score")
    std = metrics.get("best_cv_std")
    return (
        float(mean) if isinstance(mean, (int, float)) else None,
        float(std) if isinstance(std, (int, float)) else None,
    )


def diagnose(
    profile: dict[str, Any],
    plan: dict[str, Any],
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    """Diagnose the run's failure modes deterministically -> a list of findings.

    Each finding is a dict with a stable ``code``, a ``keyword`` the Planner acts on, a
    ``severity``, human-readable ``detail`` (the evidence) and ``directive`` (the specific
    instruction the Planner/FE should act on). Pure over the three input dicts — no I/O,
    no LLM, no data. An empty list means the basic checks found nothing to fix.
    """
    findings: list[dict[str, Any]] = []
    metric = plan.get("metric") or profile.get("metric")
    cv_mean, cv_std = _winner_cv(training)
    flags = set((profile.get("assessment") or {}).get("flags") or [])
    dropped = set(plan.get("drop_columns") or [])

    # -- Execution failure: the self-repair loop already had its turn -----------
    # Since Day 20 the Trainer feeds the traceback back to the provider before the
    # Critic ever sees the record, so reaching here means repair was unavailable,
    # disabled, out of attempts, or the failure was a timeout/OOM (deliberately
    # non-repairable). There is still no plan-level keyword for a crash.
    if not training.get("ok"):
        repair = training.get("repair") or {}
        if repair.get("attempted"):
            spent = len(repair.get("attempts") or [])
            outcome = f"self-repair ran {spent} attempt(s) and did not recover"
        else:
            outcome = (
                "self-repair did not run "
                f"({repair.get('reason_not_attempted') or 'unknown reason'})"
            )
        findings.append({
            "code": "execution_error",
            "keyword": None,   # no Planner keyword: a crash isn't a plan-level fix
            "severity": "blocker",
            "detail": f"Trainer run failed: {training.get('error') or 'unknown error'}.",
            "directive": f"Execution crashed and {outcome}. "
                         "Finalising this pass without a model.",
        })
        return findings

    # -- Leakage: profile-flagged residual, or the runtime too-good-to-be-true score
    leak_cols = [d["column"] for d in (profile.get("leakage_checks") or {}).get("target_correlated_features", [])]
    residual_leak = [c for c in leak_cols if c not in dropped]
    ceiling = LEAKAGE_CEILING.get(metric)
    runtime_leak = cv_mean is not None and ceiling is not None and cv_mean >= ceiling
    if residual_leak or runtime_leak:
        bits = []
        if residual_leak:
            bits.append(f"target-leakage suspect column(s) {residual_leak} were not dropped by the plan")
        if runtime_leak:
            bits.append(f"CV {metric}={cv_mean:.4f} is implausibly high (>= {ceiling}) — a leakage fingerprint")
        findings.append({
            "code": "leakage",
            "keyword": "leak",   # Planner: re-audit drops + target-derived engineered features
            "severity": "high",
            "detail": "Possible leakage: " + "; ".join(bits) + ".",
            "directive": "Re-audit dropped columns and any engineered feature derived from the "
                         "target; drop the suspect column(s) before re-training.",
        })

    # -- Imbalance: flagged in the profile but not effectively handled for the winner
    if "class_imbalance" in flags:
        imb = plan.get("imbalance_strategy") or {}
        winner = training.get("best_model")
        winner_supports_cw = next(
            (m.get("supports_class_weight") for m in (plan.get("candidate_models") or []) if m.get("name") == winner),
            None,
        )
        # Unhandled if the plan didn't recommend balancing, or the winning model can't
        # take class weights (so the recommendation didn't actually reach it).
        if not imb.get("recommended") or winner_supports_cw is False:
            why = ("the plan did not enable class weighting" if not imb.get("recommended")
                   else f"the winning model '{winner}' does not support class_weight, so balancing never reached it")
            findings.append({
                "code": "imbalance",
                "keyword": "imbalance",   # Planner: force class_weight='balanced' + stratified CV
                "severity": "medium",
                "detail": f"Class imbalance is present and {why}.",
                "directive": "Force class_weight='balanced' and prefer a class_weight-capable "
                             "candidate (RandomForest / LogisticRegression); keep stratified CV.",
            })

    # -- Underfit: absolute score is at the floor => models can't capture signal
    floor = UNDERFIT_FLOOR.get(metric)
    if cv_mean is not None and floor is not None and cv_mean <= floor and not runtime_leak:
        findings.append({
            "code": "underfit",
            "keyword": "underfit",   # Planner: increase model capacity in the grids
            "severity": "high",
            "detail": f"CV {metric}={cv_mean:.4f} is at/below the underfit floor ({floor}) — "
                      f"the candidates are barely beating chance.",
            "directive": "Increase model capacity (larger grids / more iterations) and ask the "
                         "Feature Engineer for richer features; the current setup underfits.",
        })

    # -- Overfit / variance: large fold-to-fold spread (CV-visible symptom)
    # Skip when we've already flagged underfit (a near-chance score's variance isn't
    # an overfit signal) or leakage (the ceiling case dominates).
    if (cv_mean is not None and cv_std is not None and abs(cv_mean) > 1e-9
            and not runtime_leak and not (floor is not None and cv_mean <= floor)):
        variation = cv_std / abs(cv_mean)
        if variation >= OVERFIT_CV_VARIATION:
            findings.append({
                "code": "overfit",
                "keyword": "overfit",   # Planner: reduce capacity + strengthen regularisation
                "severity": "medium",
                "detail": f"High cross-validation variance: cv_std/|cv_mean| = {variation:.2f} "
                          f"(>= {OVERFIT_CV_VARIATION}) — fold instability, an overfitting/variance risk.",
                "directive": "Reduce model capacity and strengthen regularisation in the seed grids "
                             "(shallower trees, larger min_samples_leaf, smaller C / larger alpha).",
            })

    # -- Wrong metric: the CV scorer doesn't match the primary metric (guard) -----
    scoring = (plan.get("cv") or {}).get("scoring")
    expected = _EXPECTED_SCORER.get(metric, metric)
    if scoring is not None and scoring != expected:
        findings.append({
            "code": "wrong_metric",
            "keyword": "metric",   # Planner: confirm CV scoring matches the primary metric
            "severity": "high",
            "detail": f"CV scoring '{scoring}' does not match the primary metric '{metric}' "
                      f"(expected scorer '{expected}').",
            "directive": f"Set the CV scoring to '{expected}' so the crew optimises the metric it "
                         f"is graded on.",
        })

    return findings


# --- Decision: iterate or finalize ------------------------------------------

def _prev_score(critiques_so_far: list[dict[str, Any]]) -> Optional[float]:
    """The winning CV score recorded by the most recent previous Critic pass, if any."""
    for c in reversed(critiques_so_far or []):
        s = c.get("cv_score")
        if isinstance(s, (int, float)):
            return float(s)
    return None


def decide(
    findings: list[dict[str, Any]],
    training: dict[str, Any],
    critiques_so_far: list[dict[str, Any]],
    *,
    iteration: int,
    max_iterations: int,
) -> tuple[str, str, Optional[float]]:
    """Decide iterate vs finalize, with a reason and the score delta vs the last pass.

    The rules, in order (the router's ``max_iterations`` guard is the hard backstop on
    top of all of them):

    1. **Execution failure** -> finalize (a crash isn't productively iterable pre-Day-20).
    2. **Clean run** (no actionable findings) -> finalize; the crew is done.
    3. **Budget spent** (this pass reached ``max_iterations``) -> finalize; the router
       would force it anyway, but the Critic states it honestly.
    4. **Diminishing returns** — there was a previous pass, the score did not improve by
       ``SCORE_IMPROVE_EPS``, and no *new* issue was found this pass -> finalize.
    5. Otherwise -> iterate; hand the Planner the findings' directives.
    """
    cv_mean, _ = _winner_cv(training)
    prev = _prev_score(critiques_so_far)
    delta = (cv_mean - prev) if (cv_mean is not None and prev is not None) else None

    # Only findings the Planner can actually act on count toward "should we iterate".
    actionable = [f for f in findings if f.get("keyword")]

    if not training.get("ok"):
        return "finalize", "training run failed and self-repair did not recover it — nothing to iterate on", delta
    if not actionable:
        return "finalize", "no actionable failure modes found — the run is clean, finalising", delta
    if int(iteration) >= int(max_iterations):
        return "finalize", f"iteration budget reached ({iteration}/{max_iterations}) — finalising", delta

    if prev is not None:
        prev_codes = set((critiques_so_far[-1].get("finding_codes") or []) if critiques_so_far else [])
        new_codes = {f["code"] for f in actionable} - prev_codes
        improved = delta is not None and delta >= SCORE_IMPROVE_EPS
        if not improved and not new_codes:
            return (
                "finalize",
                f"diminishing returns — CV moved {delta:+.4f} (< {SCORE_IMPROVE_EPS}) and no new "
                f"issue since the last pass; finalising rather than spinning",
                delta,
            )

    codes = ", ".join(sorted({f["code"] for f in actionable}))
    return "iterate", f"actionable issue(s) [{codes}] under budget — asking the Planner for another pass", delta


# --- Assemble the critique ---------------------------------------------------

def build_critique(
    profile: dict[str, Any],
    plan: dict[str, Any],
    training: dict[str, Any],
    *,
    critiques_so_far: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> dict[str, Any]:
    """Compute the full deterministic critique for one Critic pass.

    ``iteration`` is the number of this Critic pass (1-based: the pass now completing).
    The returned dict is JSON-serialisable and carries both a Planner-facing ``findings``
    list (prose strings embedding the action keyword) and the structured ``diagnoses`` the
    reports render. ``cv_score`` + ``finding_codes`` are recorded so the *next* pass can
    measure progress and detect diminishing returns.
    """
    diagnoses = diagnose(profile, plan, training)
    decision, reason, delta = decide(
        diagnoses, training, critiques_so_far,
        iteration=iteration, max_iterations=max_iterations,
    )
    cv_mean, cv_std = _winner_cv(training)

    # Planner-facing findings: prose that embeds the keyword its matcher looks for.
    findings = [f"{d['detail']} {d['directive']}" for d in diagnoses]

    return {
        "schema_version": CRITIQUE_SCHEMA_VERSION,
        "stub": False,
        "node": "critic",
        "dataset_key": training.get("dataset_key") or plan.get("dataset_key"),
        "iteration": int(iteration),
        "decision": decision,
        "reason": reason,
        "metric": plan.get("metric") or profile.get("metric"),
        "cv_score": cv_mean,
        "cv_std": cv_std,
        "best_model": training.get("best_model"),
        "score_delta_vs_prev": round(delta, 6) if delta is not None else None,
        "findings": findings,               # what the Planner reads
        "diagnoses": diagnoses,             # structured, for reports
        "finding_codes": [d["code"] for d in diagnoses],
        "n_findings": len(diagnoses),
        "training_ok": bool(training.get("ok")),
    }


# --- Optional LLM narrative (advisory, never a source of the decision) -------

_NARRATIVE_SYSTEM = (
    "You are the Critic agent in a multi-agent ML crew. You receive a DETERMINISTIC "
    "review of one training pass: the primary metric, the winning model's cross-validated "
    "score (an estimate on TRAIN, never a held-out score), the structured diagnoses already "
    "found (overfit / underfit / leakage / imbalance / wrong-metric), and the iterate/finalize "
    "decision that was made. Do NOT restate the numbers or overturn the decision. In <=140 "
    "words, give the Planner and Feature Engineer 2-4 CONCRETE, specific-to-this-run refinements "
    "for the next pass, or — if finalising — the single biggest caveat a reader should know about "
    "this model. Plain prose, no code."
)


def _narrative_payload(critique: dict[str, Any]) -> dict[str, Any]:
    """A compact, token-light view of the critique for the LLM prompt."""
    return {
        "dataset_key": critique.get("dataset_key"),
        "metric": critique.get("metric"),
        "cv_score": critique.get("cv_score"),
        "cv_std": critique.get("cv_std"),
        "best_model": critique.get("best_model"),
        "decision": critique.get("decision"),
        "reason": critique.get("reason"),
        "diagnoses": [
            {"code": d["code"], "severity": d["severity"], "detail": d["detail"]}
            for d in critique.get("diagnoses", [])
        ],
    }


def _llm_narrative(critique: dict[str, Any]) -> dict[str, Any]:
    """Ask the live provider for a short review note. Never raises."""
    try:
        result = llm.chat(
            _NARRATIVE_SYSTEM,
            "Review of this training pass (JSON):\n" + json.dumps(_narrative_payload(critique), default=str),
            temperature=0.0,
            max_tokens=400,
        )
        return {
            "source": result.provider,
            "model": result.model,
            "is_mock": False,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "text": result.text.strip(),
        }
    except Exception as exc:  # network / provider / parse — degrade, never crash the node
        return {"source": "unavailable", "is_mock": False, "reason": f"{type(exc).__name__}: {exc}", "text": None}


def _llm_enabled(with_llm: Optional[bool]) -> bool:
    """Explicit flag wins; else the ``CREWML_CRITIC_LLM`` env toggle (default on)."""
    if with_llm is not None:
        return with_llm
    return os.getenv("CREWML_CRITIC_LLM", "1") != "0"


def run_critic(
    profile: dict[str, Any],
    plan: dict[str, Any],
    training: dict[str, Any],
    *,
    critiques_so_far: Optional[list[dict[str, Any]]] = None,
    iteration: int = 1,
    max_iterations: int = config.MAX_ITERATIONS,
    with_llm: Optional[bool] = None,
) -> dict[str, Any]:
    """Review one training pass and return the critique (decision + findings + narrative).

    The deterministic verdict is always computed. An LLM review note is attached only
    when enabled *and* a live provider is configured; otherwise the critique records the
    narrative as ``unavailable`` and stands on its deterministic core.
    """
    critique = build_critique(
        profile, plan, training,
        critiques_so_far=critiques_so_far or [],
        iteration=iteration,
        max_iterations=max_iterations,
    )

    if _llm_enabled(with_llm) and not config.is_mock_mode():
        critique["llm_narrative"] = _llm_narrative(critique)
    else:
        reason = "mock_mode" if config.is_mock_mode() else "disabled"
        critique["llm_narrative"] = {
            "source": "unavailable", "is_mock": config.is_mock_mode(),
            "reason": reason, "text": None,
        }
    return critique
