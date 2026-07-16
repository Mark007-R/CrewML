"""The Reporter agent — the crew's terminal node: final report + model card (Day 11).

The Reporter writes nothing new to the model; it *synthesises* what the crew already
produced into two artifacts a human reads: a structured **report** (stored on the
state and dumped as JSON) and a **MODEL_CARD.md** — the model's "nutrition label".
It is deliberately deterministic: it reads the final :class:`CrewState`
(profile, plan, feature-engineering provenance, training, critiques, ensemble) and
renders it. No LLM is consulted — there is nothing to reason about that the specialist
nodes did not already decide, and a report that could hallucinate would undermine the
whole honesty story.

The honesty disciplines the rest of the crew enforces are *surfaced* here so a reader
can't miss them:

* **Every score is labelled a cross-validated estimate on train, not a held-out
  number** (``cv_score_is_holdout: false``). The model card states this in plain words
  and points at Phase 3 for the sealed-split result.
* **Degraded/mock LLM narratives are flagged**, never quietly presented as real
  (EVAL_PROTOCOL §5). The report aggregates which advisory narratives ran live vs.
  came back ``unavailable`` and totals any token cost.
* **Training failure is reported honestly** — if the Trainer crashed, the card says so
  and there is no headline metric to overclaim.

**Train only, structurally.** The Reporter reads state dicts and nothing else — no data
loader, no held-out split. A source-inspection test asserts the module never names the
locked test split; the no-peeking invariant is a property of the code.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from crewml.config import ARTIFACTS_DIR

REPORT_SCHEMA_VERSION = 1


# --- Small helpers to read the state safely ---------------------------------

def _fmt_score(value: Optional[float], metric: str) -> str:
    """Render a metric value for the card, or an em dash when absent."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.4f}"


def _metric_label(metric: str) -> str:
    return {"roc_auc": "ROC AUC", "f1_macro": "macro-F1", "r2": "R²"}.get(metric, metric)


def _final_model(training: dict[str, Any], ensemble: dict[str, Any]) -> dict[str, Any]:
    """Resolve the crew's shipped model from the ensemble decision, falling back sanely."""
    if ensemble.get("attempted") and ensemble.get("ok") and ensemble.get("chosen"):
        return {
            "kind": ensemble.get("final_model_kind"),
            "chosen": ensemble.get("chosen"),
            "members": ensemble.get("members"),
            "single_best_model": ensemble.get("single_best_model"),
            "cv_score": ensemble.get("final_cv_score"),
            "ensemble_cv_score": ensemble.get("ensemble_cv_score"),
            "single_best_cv_score": ensemble.get("single_best_cv_score"),
            "improvement_over_single": ensemble.get("improvement_over_single"),
            "artifact": "final_model.joblib",
        }
    if training.get("ok"):
        return {
            "kind": "single",
            "chosen": "single",
            "members": None,
            "single_best_model": training.get("best_model"),
            "cv_score": training.get("cv_score"),
            "ensemble_cv_score": None,
            "single_best_cv_score": training.get("cv_score"),
            "improvement_over_single": None,
            "artifact": "model.joblib",
            "note": ensemble.get("reason") or "ensemble not attempted",
        }
    return {
        "kind": "none",
        "chosen": None,
        "cv_score": None,
        "note": "training failed — no model produced (self-repair is Day 20)",
    }


def _collect_llm_usage(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate advisory-LLM provenance across the nodes that request a narrative.

    Reports how many narratives ran live vs. came back ``unavailable`` (mock mode or a
    provider error), and totals any prompt/completion tokens — so a reader sees exactly
    how much of the run was model-assisted, and that no mock number was dressed up as real.
    """
    narratives: list[dict[str, Any]] = []

    def _add(node: str, narr: Optional[dict[str, Any]]) -> None:
        if not narr:
            return
        narratives.append({
            "node": node,
            "source": narr.get("source"),
            "model": narr.get("model"),
            "live": narr.get("source") not in (None, "unavailable"),
            "prompt_tokens": narr.get("prompt_tokens"),
            "completion_tokens": narr.get("completion_tokens"),
            "reason": narr.get("reason"),
        })

    profile = state.get("profile") or {}
    plan = state.get("plan") or {}
    _add("profiler", profile.get("llm_narrative") or (profile.get("assessment") or {}).get("llm_narrative"))
    _add("planner", plan.get("llm_narrative"))
    for c in state.get("critiques") or []:
        _add("critic", c.get("llm_narrative"))

    # The Feature Engineer records tokens on its meta, not a narrative dict.
    fe_meta = state.get("fe_meta") or {}
    if fe_meta.get("source") == "llm":
        narratives.append({
            "node": "feature_engineer", "source": fe_meta.get("provider"),
            "model": fe_meta.get("model"), "live": True,
            "prompt_tokens": fe_meta.get("prompt_tokens"),
            "completion_tokens": fe_meta.get("completion_tokens"), "reason": None,
        })

    live = sum(1 for n in narratives if n["live"])
    prompt_tokens = sum(n["prompt_tokens"] or 0 for n in narratives if n["live"])
    completion_tokens = sum(n["completion_tokens"] or 0 for n in narratives if n["live"])
    return {
        "narratives": narratives,
        "n_requested": len(narratives),
        "n_live": live,
        "n_unavailable": len(narratives) - live,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "any_live": live > 0,
    }


# --- Build the structured report --------------------------------------------

def build_report(state: dict[str, Any]) -> dict[str, Any]:
    """Synthesise the final report from the crew's terminal state (pure, no I/O).

    Returns a JSON-serialisable dict summarising the run end-to-end: the data profile,
    the plan, the feature engineering, the training + Critic loop, the ensemble decision,
    and the honesty caveats a reader must see. Renders the model-card markdown too.
    """
    dataset_key = state["dataset_key"]
    metric = state.get("metric") or (state.get("plan") or {}).get("metric")
    profile = state.get("profile") or {}
    plan = state.get("plan") or {}
    fe_meta = state.get("fe_meta") or {}
    training = state.get("training") or {}
    ensemble = state.get("ensemble") or {}
    critiques = state.get("critiques") or []

    final = _final_model(training, ensemble)
    llm_usage = _collect_llm_usage(state)

    critic_passes = [
        {
            "iteration": c.get("iteration"),
            "decision": c.get("decision"),
            "reason": c.get("reason"),
            "cv_score": c.get("cv_score"),
            "finding_codes": c.get("finding_codes"),
        }
        for c in critiques
    ]

    warnings: list[str] = [
        "All scores are CROSS-VALIDATED estimates on the train split "
        "(cv_score_is_holdout: false); the sealed held-out score is Phase 3.",
    ]
    if not training.get("ok"):
        warnings.append("Training failed — no model was produced; the report records the failure honestly.")
    if fe_meta.get("is_mock"):
        warnings.append("Run was in MOCK LLM mode — advisory narratives are unavailable, not real model output.")
    if not llm_usage["any_live"]:
        warnings.append("No advisory LLM narrative ran live this run; every decision stands on the deterministic core.")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stub": False,
        "node": "reporter",
        "dataset_key": dataset_key,
        "task": state.get("task") or plan.get("task"),
        "subtype": state.get("subtype") or plan.get("subtype"),
        "metric": metric,
        "cv_score_is_holdout": False,
        "final_model": final,
        "iterations_run": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations"),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "profile_summary": {
            "n_rows": profile.get("n_rows"),
            "n_features": profile.get("n_features") or len((profile.get("features") or {})),
            "flags": (profile.get("assessment") or {}).get("flags"),
        },
        "plan_summary": {
            "drop_columns": plan.get("drop_columns"),
            "candidate_models": [m["name"] for m in plan.get("candidate_models", [])],
            "cv": plan.get("cv"),
            "imbalance_recommended": (plan.get("imbalance_strategy") or {}).get("recommended"),
        },
        "feature_engineering": {
            "source": fe_meta.get("source"),
            "new_columns": (fe_meta.get("validation") or {}).get("new_columns"),
        },
        "training": {
            "ok": training.get("ok"),
            "best_model": training.get("best_model"),
            "cv_score": training.get("cv_score"),
            "per_model": (training.get("metrics") or {}).get("per_model"),
            "error": training.get("error"),
        },
        "ensemble": {
            "attempted": ensemble.get("attempted"),
            "ok": ensemble.get("ok"),
            "chosen": ensemble.get("chosen"),
            "members": ensemble.get("members"),
            "ensemble_cv_score": ensemble.get("ensemble_cv_score"),
            "single_best_cv_score": ensemble.get("single_best_cv_score"),
            "improvement_over_single": ensemble.get("improvement_over_single"),
        },
        "critic_passes": critic_passes,
        "llm_usage": llm_usage,
        "warnings": warnings,
    }
    report["model_card_markdown"] = render_model_card(report)
    return report


# --- Render the model card ---------------------------------------------------

def render_model_card(report: dict[str, Any]) -> str:
    """Render a MODEL_CARD.md string from a report dict (pure).

    Follows a conventional model-card shape (details / data / evaluation / metrics /
    limitations) so the artifact reads like the ML-community standard, with the crew's
    honesty caveats front and centre.
    """
    key = report["dataset_key"]
    metric = report.get("metric") or ""
    metric_lbl = _metric_label(metric)
    final = report.get("final_model") or {}
    kind = final.get("kind")
    score = _fmt_score(final.get("cv_score"), metric)

    if kind == "ensemble":
        model_desc = (
            f"Soft/averaging **ensemble** of {final.get('members')} "
            f"(beat the single best '{final.get('single_best_model')}' by "
            f"{final.get('improvement_over_single'):+.4f} {metric} on CV)."
        )
    elif kind == "single":
        model_desc = f"Single model: **{final.get('single_best_model')}** (ensemble did not improve on it)."
    else:
        model_desc = "**No model** — the training run failed (see limitations)."

    plan_s = report.get("plan_summary") or {}
    prof_s = report.get("profile_summary") or {}
    fe = report.get("feature_engineering") or {}
    cv = plan_s.get("cv") or {}
    llm = report.get("llm_usage") or {}

    per_model = (report.get("training") or {}).get("per_model") or []
    per_model_rows = "\n".join(
        f"| {r['name']} | {_fmt_score(r.get('cv_mean'), metric)} | ±{_fmt_score(r.get('cv_std'), metric).lstrip('—') or '—'} |"
        for r in per_model
    ) or "| — | — | — |"

    critic_rows = "\n".join(
        f"| {p['iteration']} | {p['decision']} | {', '.join(p['finding_codes']) if p['finding_codes'] else 'none'} "
        f"| {_fmt_score(p['cv_score'], metric)} |"
        for p in (report.get("critic_passes") or [])
    ) or "| — | — | — | — |"

    warnings = "\n".join(f"- {w}" for w in (report.get("warnings") or []))

    llm_line = (
        f"{llm.get('n_live', 0)} live / {llm.get('n_requested', 0)} requested "
        f"({llm.get('prompt_tokens', 0)}+{llm.get('completion_tokens', 0)} tokens)"
        if llm.get("any_live") else
        f"none live ({llm.get('n_requested', 0)} requested, all unavailable/mock)"
    )

    return f"""# Model Card — CrewML on `{key}`

*Generated by the CrewML crew's Reporter node. Every number below is a
**cross-validated estimate on the training split**, not a held-out score
(`cv_score_is_holdout: false`). The sealed held-out evaluation is Phase 3.*

## Model details

- **Dataset:** `{key}` — {report.get('task')} ({report.get('subtype')}), primary metric **{metric_lbl}**.
- **Final model:** {model_desc}
- **Headline CV score:** **{score} {metric_lbl}** (mean over {cv.get('n_splits', '—')}-fold {cv.get('scheme', 'CV')}).
- **Built by:** the full seven-node crew (Profiler → Planner → Feature Engineer →
  Trainer → Critic → Ensembler → Reporter) over {report.get('iterations_run', 0)} Critic pass(es).

## Training data

- **Rows (train split):** {prof_s.get('n_rows', '—')} · **features profiled:** {prof_s.get('n_features', '—')}.
- **Data flags:** {', '.join(prof_s.get('flags') or []) or 'none'}.
- **Columns dropped by the plan:** {plan_s.get('drop_columns') or 'none'} (leakage / integrity grounds).
- **Feature engineering:** `{fe.get('source')}` — added {fe.get('new_columns') or 'no'} column(s).
- The **held-out split was never loaded** during modeling — it is reserved for final scoring.

## Evaluation

- **Protocol:** {cv.get('scheme', 'CV')}({cv.get('n_splits', '—')}), scorer `{cv.get('scoring', metric)}`, seeded for reproducibility.
- **Class imbalance handling:** {'enabled (class_weight=balanced)' if plan_s.get('imbalance_recommended') else 'not needed'}.

### Per-candidate CV ({metric_lbl})

| model | CV mean | CV std |
|-------|:-------:|:------:|
{per_model_rows}

### Critic loop

| pass | decision | findings | CV score |
|:----:|----------|----------|:--------:|
{critic_rows}

## LLM assistance

- Advisory narratives (never decision-making): **{llm_line}**.
- The crew's every *decision* is deterministic; the LLM only ever adds an optional
  written second opinion, so a degraded provider never changes what the crew does.

## Limitations & honest caveats

{warnings}

- The executor is a **process-isolation** sandbox, not yet a security sandbox
  (hardening is Phase 4 / Day 19).
- "Overfit" signals from the Critic are **cross-validation fold-instability**, not a
  train-vs-held-out gap — the crew cannot see the held-out split by construction.
"""


# --- Public entry point (writes the artifacts) ------------------------------

def run_reporter(state: dict[str, Any]) -> dict[str, Any]:
    """Build the report and write ``MODEL_CARD.md`` + ``report.json`` for the run.

    Returns the report dict (stored on the state). The markdown card and a
    narrative-free JSON copy are written under ``artifacts/crew/<dataset>/`` — git-ignored
    per-run artifacts, so the crew genuinely "writes the report" without polluting the repo.
    """
    report = build_report(state)

    out_dir = ARTIFACTS_DIR / "crew" / state["dataset_key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL_CARD.md").write_text(report["model_card_markdown"], encoding="utf-8")
    json_copy = {k: v for k, v in report.items() if k != "model_card_markdown"}
    (out_dir / "report.json").write_text(json.dumps(json_copy, indent=2, default=str), encoding="utf-8")
    return report
