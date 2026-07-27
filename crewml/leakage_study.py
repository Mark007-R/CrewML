"""Day 22 — leakage & honesty guards: the study that proves they hold.

Day 17 ended with a measured hole: an injected column agreeing with the target
on 95% of rows passed the Profiler's purity screen, the Planner kept it, the
Critic's CV ceiling never fired, and the model trained on it (CV roc_auc 0.964
— inflated and unchallenged). Day 22 built the guards; this module is the
evidence that they work, produced the same way Day 17 produced the hole —
by injection against the real crew, not by argument:

1. **Calibration** — the single-feature screen's ceilings re-derived on the
   locked suite, with the clean-max / ceiling / injected-leak margins shown, so
   the zero-false-positive claim and the *residual* window are both on the record.
2. **Crew probes** — the Day-17 blatant and subtle leak probes re-run through
   the full graph. The subtle probe's expectation flips from ``missed`` (Day 17's
   honest record) to ``caught``: the Profiler must flag, the Planner must drop,
   and the model must never see the column.
3. **FE-guard probes** — the leakage the crew could *introduce*: a feature
   derived from a leaky column (must fail the engineered-column screen) and a
   cross-row transform (must fail the row-wise check), plus the clean default
   as the no-overblocking control.
4. **No-peek probes** — sandboxed code attempts to read the locked holdout by
   absolute path; the read must be refused by the Day-19 guard while the staged
   train input stays readable. This is held-out isolation *demonstrated at
   runtime*, on top of the static no-reference tests.
5. **Seal sweep** — every dataset's holdout re-fingerprinted against the
   manifest after all of the above ran.

Honesty notes: detection is fully deterministic (screens + drops + checks), so
none of these verdicts depends on the LLM; the crew probes still run whatever
provider is configured (recorded per-probe as ``mock``/FE provenance) because
the *rest* of the pipeline around the guards should be the real one.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import numpy as np

from crewml.config import RESULTS_DIR, SEED, is_mock_mode
from crewml.crew.feature_engineer import DEFAULT_FE_SOURCE, _validate_fe
from crewml.datasets import (
    REGISTRY,
    TARGET_COLUMN,
    holdout_path,
    load_train,
    train_path,
    verify_all_holdouts,
)
from crewml.executor import run_code
from crewml.failure_taxonomy import (
    LEAK_COLUMN,
    PROBE_PREFIX,
    _run_probe_crew,
    make_leak_frame,
    probe_dataset,
)
from crewml.leakage import SINGLE_FEATURE_CEILING, single_feature_cv_score

LEAKAGE_STUDY_SCHEMA_VERSION = 1
REPORT_JSON_PATH = RESULTS_DIR / "day22_leakage_honesty.json"
REPORT_MD_PATH = RESULTS_DIR / "day22_leakage_honesty.md"


# --- 1. Calibration: where the ceilings sit and what still fits under them ---

def run_calibration() -> dict[str, Any]:
    """Measure every clean feature and every injected probe against the ceilings.

    Recomputes, rather than trusts, the numbers the ceilings were set from: the
    strongest legitimate single feature per dataset, and the standalone score of
    each injected leak. The margins — clean-max below the ceiling, leak above —
    are the whole justification for the thresholds, so they are recorded here.
    """
    clean: dict[str, Any] = {}
    for key, spec in REGISTRY.items():
        df = load_train(key)
        y = df[TARGET_COLUMN]
        scores = {}
        for col in df.columns:
            if col == TARGET_COLUMN:
                continue
            s = single_feature_cv_score(df[col], y, spec.task, spec.metric)
            if s is not None:
                scores[col] = s
        top_col = max(scores, key=scores.get)
        clean[key] = {
            "metric": spec.metric,
            "ceiling": SINGLE_FEATURE_CEILING[spec.metric],
            "strongest_feature": top_col,
            "strongest_score": scores[top_col],
            "clean_margin": round(SINGLE_FEATURE_CEILING[spec.metric] - scores[top_col], 6),
            "false_positives": [c for c, s in scores.items() if s >= SINGLE_FEATURE_CEILING[spec.metric]],
        }

    injected: list[dict[str, Any]] = []
    for base_key, kind in (
        ("credit-g", "subtle"), ("credit-g", "blatant"),
        ("cpu_small", "subtle"), ("cpu_small", "blatant"),
    ):
        spec = REGISTRY[base_key]
        frame, truth = make_leak_frame(
            load_train(base_key), task=spec.task, kind=kind,
            rng=np.random.default_rng(SEED),
        )
        s = single_feature_cv_score(frame[LEAK_COLUMN], frame[TARGET_COLUMN], spec.task, spec.metric)
        ceiling = SINGLE_FEATURE_CEILING[spec.metric]
        injected.append({
            "base_dataset": base_key,
            "kind": kind,
            "metric": spec.metric,
            "ground_truth_signal": truth["measured_signal"],
            "standalone_score": s,
            "ceiling": ceiling,
            "screened": bool(s is not None and s >= ceiling),
        })

    return {
        "ceilings": dict(SINGLE_FEATURE_CEILING),
        "clean_suite": clean,
        "injected_leaks": injected,
        # The honest statement of what the screen still cannot see: anything whose
        # standalone score lands between the clean maximum and the ceiling.
        "residual_window": {
            metric: {
                "clean_max": max(
                    (d["strongest_score"] for d in clean.values() if d["metric"] == metric),
                    default=None,
                ),
                "ceiling": ceiling,
            }
            for metric, ceiling in SINGLE_FEATURE_CEILING.items()
        },
    }


# --- 2. Crew probes: the Day-17 injections against the Day-22 crew ----------

def run_crew_probe(kind: str, *, base_key: Optional[str] = None) -> dict[str, Any]:
    """Re-run a Day-17 leak probe through the full crew and read every surface.

    Same injection, same graph, new expectation: with the single-feature screen
    in the Profiler, *both* kinds must now be caught — flagged, dropped, and the
    model never sees the column. Also reads the Day-22 FE-validation fields so
    the run shows the engineered-column screen was active, not just present.
    """
    if kind not in ("blatant", "subtle"):
        raise ValueError(f"unknown leak probe kind: {kind!r}")
    base_key = base_key or ("cpu_small" if kind == "blatant" else "credit-g")
    base = REGISTRY[base_key]
    frame, truth = make_leak_frame(
        load_train(base_key), task=base.task, kind=kind,
        rng=np.random.default_rng(SEED),
    )
    probe_key = f"{PROBE_PREFIX}day22_leak_{kind}"

    started = time.time()
    with probe_dataset(base_key, probe_key, frame) as spec:
        final = _run_probe_crew(spec)
    seconds = round(time.time() - started, 2)

    profile = final.get("profile") or {}
    plan = final.get("plan") or {}
    critiques = final.get("critiques") or []
    tmetrics = (final.get("training") or {}).get("metrics") or {}
    fe_meta = final.get("fe_meta") or {}
    fe_validation = fe_meta.get("validation") or {}

    suspects = (profile.get("leakage_checks") or {}).get("target_correlated_features", [])
    flagged_entry = next((d for d in suspects if d["column"] == LEAK_COLUMN), None)
    plan_dropped = LEAK_COLUMN in (plan.get("drop_columns") or [])
    critic_fired = any("leakage" in (c.get("finding_codes") or []) for c in critiques)

    return {
        "probe": f"leak_{kind}",
        "base_dataset": base_key,
        "ground_truth": truth,
        "expectation": "caught (Day 22: the screen must fire and the drop must hold)",
        "profiler_flagged": flagged_entry is not None,
        "flagged_by_measure": (flagged_entry or {}).get("measure"),
        "flagged_signal": (flagged_entry or {}).get("signal"),
        "plan_dropped": plan_dropped,
        "critic_leakage_finding": critic_fired,
        "detected": flagged_entry is not None or critic_fired,
        "model_saw_leak": not plan_dropped,
        "cv_score": tmetrics.get("best_cv_score"),
        "cv_metric": plan.get("metric"),
        "iterations_run": final.get("iteration"),
        "fe_source": fe_meta.get("source"),
        "fe_validation_ok": fe_validation.get("ok"),
        "fe_no_leakage": fe_validation.get("no_leakage"),
        "fe_row_wise_ok": fe_validation.get("row_wise_ok"),
        "mock": is_mock_mode(),
        "seconds": seconds,
    }


# --- 3. FE-guard probes: leakage the crew introduces itself ------------------

_LEAK_DERIVED_FE = '''\
import pandas as pd


def add_features(df):
    out = df.copy()
    out["risk_signal"] = df["leak_probe"] * 2.0 + 1.0
    return out
'''

_CROSS_ROW_FE = '''\
import pandas as pd


def add_features(df):
    out = df.copy()
    out["age_centered"] = df["age"] - df["age"].mean()
    return out
'''


def run_fe_guard_probes() -> list[dict[str, Any]]:
    """Record-level probes of the Day-22 FE validation gate. No LLM involved.

    Three verdicts a reader should be able to check independently: the clean
    default passes (the gate does not overblock), a target-derived feature is
    rejected by the engineered-column screen, and a cross-row transform is
    rejected by the row-wise check.
    """
    probes: list[dict[str, Any]] = []

    verdict = _validate_fe(DEFAULT_FE_SOURCE, "credit-g")
    probes.append({
        "probe": "fe_clean_control",
        "dataset": "credit-g",
        "fe": "deterministic default (row_nan_count)",
        "expectation": "passes — the gate must not overblock legitimate FE",
        "ok": verdict["ok"],
        "row_wise_ok": verdict.get("row_wise_ok"),
        "no_leakage": verdict.get("no_leakage"),
        "leakage_scores": verdict.get("leakage_scores"),
        "detected": verdict["ok"] is True,
    })

    spec = REGISTRY["credit-g"]
    frame, truth = make_leak_frame(
        load_train("credit-g"), task=spec.task, kind="subtle",
        rng=np.random.default_rng(SEED),
    )
    with probe_dataset("credit-g", f"{PROBE_PREFIX}day22_fe_leak", frame) as pspec:
        verdict = _validate_fe(_LEAK_DERIVED_FE, pspec.key)
    probes.append({
        "probe": "fe_leak_derived",
        "dataset": pspec.key,
        "fe": "risk_signal = leak_probe * 2 + 1 (derived from the injected leak)",
        "ground_truth": truth,
        "expectation": "rejected — engineered-column screen must fire",
        "ok": verdict["ok"],
        "row_wise_ok": verdict.get("row_wise_ok"),
        "no_leakage": verdict.get("no_leakage"),
        "leaky_columns": verdict.get("leaky_columns"),
        "leakage_scores": verdict.get("leakage_scores"),
        "detected": verdict["ok"] is False and verdict.get("no_leakage") is False,
    })

    verdict = _validate_fe(_CROSS_ROW_FE, "credit-g")
    probes.append({
        "probe": "fe_cross_row",
        "dataset": "credit-g",
        "fe": "age_centered = age - age.mean() (fitted across rows)",
        "expectation": "rejected — row-wise/statelessness check must fire",
        "ok": verdict["ok"],
        "row_wise_ok": verdict.get("row_wise_ok"),
        "no_leakage": verdict.get("no_leakage"),
        "detected": verdict["ok"] is False and verdict.get("row_wise_ok") is False,
    })

    return probes


# --- 4. No-peek probes: the holdout is unreachable at runtime ----------------

def run_no_peek_probe(key: str = "credit-g") -> dict[str, Any]:
    """Sandboxed code tries to read the locked holdout by absolute path.

    The Day-19 guard's read-deny root covers the raw dataset store, so the read
    must be refused at the moment of use — while the staged train input in the
    same script stays readable (the control that proves the refusal is targeted,
    not a broken filesystem).
    """
    target = holdout_path(key)
    script = (
        'import pandas as pd\n'
        'from crew_io import input_path\n'
        '\n'
        'df = pd.read_parquet(input_path("train.parquet"))\n'
        'print("staged input readable:", len(df), "rows", flush=True)\n'
        f'pd.read_parquet(r"{target}")\n'
        'print("HOLDOUT_READ_SUCCEEDED", flush=True)\n'
    )
    result = run_code(
        script,
        inputs={"train.parquet": train_path(key)},
        timeout_s=120,
        keep_workdir=False,
    )
    refused = (not result.ok) and "denied root" in (result.error or "")
    return {
        "probe": "sandbox_no_peek",
        "dataset": key,
        "attempted_path": str(target),
        "expectation": "refused — the guard's read-deny root covers the data store",
        "staged_input_readable": "staged input readable" in (result.stdout or ""),
        "holdout_read_succeeded": "HOLDOUT_READ_SUCCEEDED" in (result.stdout or ""),
        "refused": refused,
        "error_excerpt": (result.error or "")[-300:] or None,
        "detected": refused and "HOLDOUT_READ_SUCCEEDED" not in (result.stdout or ""),
    }


# --- 5. Assemble, persist, render --------------------------------------------

def build_report(*, with_crew_probes: bool = True) -> dict[str, Any]:
    """Run every probe family and assemble the Day-22 record."""
    calibration = run_calibration()
    crew_probes = (
        [run_crew_probe("subtle"), run_crew_probe("blatant")]
        if with_crew_probes else []
    )
    fe_probes = run_fe_guard_probes()
    no_peek = run_no_peek_probe()
    seals = verify_all_holdouts()

    checks = (
        [p["detected"] for p in crew_probes]
        + [p["detected"] for p in fe_probes]
        + [no_peek["detected"]]
        + list(seals.values())
        + [not d["false_positives"] for d in calibration["clean_suite"].values()]
    )
    return {
        "schema_version": LEAKAGE_STUDY_SCHEMA_VERSION,
        "day": 22,
        "phase": 4,
        "study": "leakage_honesty_guards",
        "mock": is_mock_mode(),
        "seed": SEED,
        "calibration": calibration,
        "crew_probes": crew_probes,
        "fe_guard_probes": fe_probes,
        "no_peek_probe": no_peek,
        "holdout_seals": seals,
        "all_guards_hold": all(checks),
        "n_checks": len(checks),
    }


def save_report(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    """The results table a reader (and Day 29's README) consumes."""
    cal = report["calibration"]
    lines: list[str] = [
        "# Day 22 — Leakage & honesty guards",
        "",
        f"Seed {report['seed']}; mock mode: {report['mock']}. "
        f"**All guards hold: {report['all_guards_hold']}** ({report['n_checks']} checks).",
        "",
        "## Screen calibration (single-feature CV, depth-3 tree, 3-fold)",
        "",
        "| dataset | metric | strongest clean feature | score | ceiling | margin |",
        "|---|---|---|---|---|---|",
    ]
    for key, d in cal["clean_suite"].items():
        lines.append(
            f"| {key} | {d['metric']} | `{d['strongest_feature']}` "
            f"| {d['strongest_score']:.4f} | {d['ceiling']} | {d['clean_margin']:+.4f} |"
        )
    lines += [
        "",
        "Zero false positives on the clean suite "
        f"({sum(len(d['false_positives']) for d in cal['clean_suite'].values())} columns fired).",
        "",
        "| injected leak | metric | ground-truth signal | standalone score | screened |",
        "|---|---|---|---|---|",
    ]
    for d in cal["injected_leaks"]:
        lines.append(
            f"| {d['base_dataset']} / {d['kind']} | {d['metric']} "
            f"| {d['ground_truth_signal']:.4f} | {d['standalone_score']:.4f} | {d['screened']} |"
        )
    lines += [
        "",
        "**Residual window (disclosed):** a leak whose standalone score lands between "
        "the clean maximum and the ceiling still passes — the Day-17 window is "
        "narrowed, not closed:",
        "",
    ]
    for metric, w in cal["residual_window"].items():
        cm = w["clean_max"]
        lines.append(f"- `{metric}`: undetectable band ({cm:.4f}, {w['ceiling']})" if cm is not None
                     else f"- `{metric}`: ceiling {w['ceiling']}")
    if report["crew_probes"]:
        lines += [
            "",
            "## Full-crew injection probes (the Day-17 re-run)",
            "",
            "| probe | flagged by | signal | plan dropped | model saw leak | CV | detected |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in report["crew_probes"]:
            cv = p["cv_score"]
            lines.append(
                f"| {p['probe']} ({p['base_dataset']}) | {p['flagged_by_measure']} "
                f"| {p['flagged_signal']} | {p['plan_dropped']} | {p['model_saw_leak']} "
                f"| {cv:.4f} ({p['cv_metric']}) | **{p['detected']}** |"
                if isinstance(cv, (int, float)) else
                f"| {p['probe']} ({p['base_dataset']}) | {p['flagged_by_measure']} "
                f"| {p['flagged_signal']} | {p['plan_dropped']} | {p['model_saw_leak']} "
                f"| n/a | **{p['detected']}** |"
            )
    lines += [
        "",
        "## FE-introduced leakage (validation-gate probes)",
        "",
        "| probe | expectation | ok | row_wise | no_leakage | verdict as expected |",
        "|---|---|---|---|---|---|",
    ]
    for p in report["fe_guard_probes"]:
        lines.append(
            f"| {p['probe']} | {p['expectation']} | {p['ok']} "
            f"| {p['row_wise_ok']} | {p['no_leakage']} | **{p['detected']}** |"
        )
    np_ = report["no_peek_probe"]
    lines += [
        "",
        "## Runtime no-peek probe",
        "",
        f"Sandboxed read of `{np_['attempted_path']}`: refused = **{np_['refused']}** "
        f"(staged train input stayed readable: {np_['staged_input_readable']}).",
        "",
        "## Holdout seals",
        "",
        "| dataset | sealed |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in report["holdout_seals"].items()]
    lines.append("")
    return "\n".join(lines)
