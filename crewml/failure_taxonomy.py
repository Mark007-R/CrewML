"""Day 17 — the failure taxonomy: where does the crew actually fail, and what catches it?

Phase 3 has so far measured *how well* the crew scores (Day 12), what each part
contributes (Days 13-14), what the loop costs (Day 15) and what the provider changes
(Day 16). This module asks the complementary question an honest engineering report
cannot skip: **when things go wrong, where do they go wrong, who notices, and what does
it cost?** The answer is a taxonomy — a fixed vocabulary of failure categories, each
tied to the *stage* that owns it and the *detection surface* that is supposed to catch
it — applied two ways:

1. **Archive census** (:func:`mine_archive`). Every archived run record the project has
   produced (the full final states under ``artifacts/ablation/``, the Reporter records
   under ``artifacts/crew/``, and the solo-agent baseline with its two real crashes) is
   classified by :func:`classify_run` into taxonomy events. Nothing is re-run; the
   census is a *reading* of evidence that already exists, so it cannot flatter.

2. **Injection probes** (:func:`run_leak_probe`, :func:`run_timeout_probe`,
   :func:`run_record_probes`). A census only shows the failures that happened to occur.
   To prove a category is *detectable* — or honestly show it is not — we inject known
   faults and watch the detection surfaces respond: a blatant leaked column (the
   Profiler's Pearson screen must catch it), a subtle leaked column engineered to sit
   inside the detection window (below the Profiler's threshold AND below the Critic's
   too-good-to-be-true ceiling — the crew SHOULD miss it, and the probe documents that
   window as measured fact), and a starved executor timeout (the sandbox must kill the
   run and the Critic must file it, not hang). Live probes run the real crew graph on a
   real injected dataset; record-level probes mutate a real archived record and run the
   Critic's actual ``diagnose`` over it.

Every event carries an **outcome**, because "failure" is not one thing:

* ``fatal``    — no scored model shipped (the solo agent's mode: 2/5 datasets).
* ``degraded`` — a model shipped but quality was knowably impacted (e.g. the budget
  cut the loop off mid-repair).
* ``handled``  — a guard absorbed the fault with no quality impact (FE fell back to the
  validated default; the Ensembler's chooser kept the single model; the loop recovered
  a deficient pass).
* ``detected`` — the signal was recorded but the impact is indeterminate from the record.
* ``missed``   — ground truth says a fault was present and NO surface fired. Only ever
  assignable when ground truth is known — i.e. by an injection probe, never by the
  census (a census cannot see what nothing detected; pretending it could would be the
  exact dishonesty this project exists to avoid).

Honesty notes: probes never touch the locked holdout (they run the graph and read the
*detection* facts, not scores — no holdout scoring, no seal risk; probe datasets are
throwaway copies of the train split only). Probe REGISTRY entries are scoped and
removed afterward. Runs without a live provider are labelled, never passed off as live.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from crewml.config import ARTIFACTS_DIR, DATA_DIR, RESULTS_DIR, ROOT, SEED, is_mock_mode
from crewml.crew import build_crew, initial_state
from crewml.crew.critic import diagnose
from crewml.datasets import REGISTRY, TARGET_COLUMN, DatasetSpec

TAXONOMY_SCHEMA_VERSION = 1
TAXONOMY_RESULT_PATH = RESULTS_DIR / "day17_failure_taxonomy.json"
TAXONOMY_TABLE_MD_PATH = RESULTS_DIR / "day17_failure_taxonomy.md"

SOLO_METRICS_PATH = RESULTS_DIR / "solo_agent_metrics.json"

# CV movement below this between first and last pass counts as "the loop moved nothing".
LOOP_PROGRESS_EPS = 0.002

# --- The taxonomy ------------------------------------------------------------
# code -> {group, stage, detection surface, description}. ``group`` maps each fine-
# grained code onto the four families the study brief names (bad plan / exec error /
# wrong metric / missed leakage) plus the operational families the archive actually
# exhibits. The taxonomy is closed: classify_run may only emit these codes, so counts
# are comparable across studies and days.
CATEGORIES: dict[str, dict[str, str]] = {
    "exec_error": {
        "group": "exec_error", "stage": "executor/trainer",
        "surface": "sandbox exit status -> Critic `execution_error` blocker",
        "description": "Generated or assembled code crashed in the sandbox (non-zero exit).",
    },
    "exec_timeout": {
        "group": "exec_error", "stage": "executor",
        "surface": "sandbox wall-clock cap -> `timed_out` -> Critic blocker",
        "description": "Code exceeded the executor's wall-clock cap and was killed.",
    },
    "plan_underfit": {
        "group": "bad_plan", "stage": "planner",
        "surface": "Critic underfit floor (per-metric absolute score)",
        "description": "The plan's capacity was insufficient — winner scored at/below the underfit floor.",
    },
    "plan_overfit_variance": {
        "group": "bad_plan", "stage": "planner",
        "surface": "Critic fold-variance threshold (cv_std/|cv_mean|)",
        "description": "High fold-to-fold variance — an overfitting/instability risk in the plan's grids.",
    },
    "plan_search_invalid": {
        "group": "bad_plan", "stage": "planner",
        "surface": "sklearn parameter validation at fit time",
        "description": "A hyper-parameter grid named parameters the estimator does not have.",
    },
    "wrong_metric": {
        "group": "wrong_metric", "stage": "planner",
        "surface": "Critic scorer-vs-primary-metric guard",
        "description": "The CV scorer does not match the metric the run is graded on.",
    },
    "leakage_flagged": {
        "group": "missed_leakage", "stage": "profiler/planner",
        "surface": "Profiler leakage screen + Critic residual/ceiling checks",
        "description": "A leakage signal fired (flagged column or implausibly high score).",
    },
    "leakage_missed": {
        "group": "missed_leakage", "stage": "profiler/planner/critic",
        "surface": "NONE fired (ground truth from an injection probe)",
        "description": "A leaked feature was present and no detection surface caught it.",
    },
    "imbalance_unhandled": {
        "group": "bad_plan", "stage": "planner",
        "surface": "Critic imbalance check (flag present, strategy absent/ineffective)",
        "description": "Class imbalance was flagged but the winning model never received balancing.",
    },
    "provider_outage": {
        "group": "provider", "stage": "llm",
        "surface": "per-call fallback (`unavailable` narrative / FE deterministic fallback)",
        "description": "The LLM provider failed; deterministic fallbacks carried the run.",
    },
    "budget_cutoff": {
        "group": "budget", "stage": "critic/router",
        "surface": "Critic budget-reached finalise with actionable findings remaining",
        "description": "The iteration budget stopped the loop while it still had repairs to make.",
    },
    "loop_no_actuator": {
        "group": "budget", "stage": "critic",
        "surface": "cross-pass score delta (loop fired, nothing moved)",
        "description": "The loop iterated but recovered nothing — no actuator for its directives.",
    },
    "ensemble_regression": {
        "group": "ensemble", "stage": "ensembler",
        "surface": "Ensembler same-fold CV comparison + chooser",
        "description": "The ensemble scored below the single best model.",
    },
}

OUTCOMES = ("fatal", "degraded", "handled", "detected", "missed")

# Critic finding code -> taxonomy category.
_FINDING_CATEGORY = {
    "underfit": "plan_underfit",
    "overfit": "plan_overfit_variance",
    "leakage": "leakage_flagged",
    "imbalance": "imbalance_unhandled",
    "wrong_metric": "wrong_metric",
}

# LLM-narrative fallback reasons that are NOT a provider failure.
_BENIGN_NARRATIVE_REASONS = {"mock_mode", "disabled"}


def _event(
    category: str,
    outcome: str,
    *,
    dataset: Optional[str],
    run: str,
    system: str,
    evidence: str,
) -> dict[str, Any]:
    """One taxonomy event. ``category`` and ``outcome`` must come from the closed sets."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown taxonomy category: {category!r}")
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome: {outcome!r}")
    meta = CATEGORIES[category]
    return {
        "category": category,
        "group": meta["group"],
        "stage": meta["stage"],
        "outcome": outcome,
        "dataset": dataset,
        "run": run,
        "system": system,
        "evidence": evidence[:400],
    }


# --- Classifier over one run record ------------------------------------------

def _outage_reasons(record: dict[str, Any]) -> list[str]:
    """Provider-failure reasons across every LLM surface of a full final state."""
    reasons: list[str] = []
    for node_key in ("profile", "plan"):
        narr = (record.get(node_key) or {}).get("llm_narrative") or {}
        if narr.get("source") == "unavailable" and narr.get("reason") not in _BENIGN_NARRATIVE_REASONS:
            if narr.get("reason"):
                reasons.append(str(narr["reason"]))
    fe_meta = record.get("fe_meta") or {}
    if fe_meta.get("source") == "fallback" and fe_meta.get("fallback_reason"):
        reasons.append(str(fe_meta["fallback_reason"]))
    for c in record.get("critiques") or []:
        narr = c.get("llm_narrative") or {}
        if narr.get("source") == "unavailable" and narr.get("reason") not in _BENIGN_NARRATIVE_REASONS:
            if narr.get("reason"):
                reasons.append(str(narr["reason"]))
    return reasons


def _budget_bound(last_critique: dict[str, Any]) -> bool:
    return bool(
        last_critique.get("decision") == "finalize"
        and "budget reached" in (last_critique.get("reason") or "")
    )


def _finding_detail(critiques: list[dict], code: str) -> str:
    """The recorded evidence detail for a finding code, from the latest pass that has it."""
    for c in reversed(critiques):
        for d in c.get("diagnoses") or []:
            if d.get("code") == code:
                return str(d.get("detail") or "")
    return f"finding code '{code}' recorded by the Critic"


def classify_run(record: dict[str, Any], *, run: str, system: str = "crew") -> list[dict[str, Any]]:
    """Classify one full crew final-state record into taxonomy events (pure).

    Reads only what the record already contains — training/ensemble exec status, the
    Critic's per-pass finding codes, the budget flag, the LLM fallback trail — and maps
    each signal to a category + outcome. A clean run yields no events. This function
    can never emit ``leakage_missed``: "missed" requires ground truth the record does
    not carry (see the module docstring), and only probes supply it.
    """
    events: list[dict[str, Any]] = []
    dataset = record.get("dataset_key")
    critiques = record.get("critiques") or []
    last = critiques[-1] if critiques else {}
    training = record.get("training") or {}
    ensemble = record.get("ensemble") or {}

    # Provider outage — one event per run, however many surfaces fell back.
    reasons = _outage_reasons(record)
    if reasons:
        events.append(_event(
            "provider_outage", "handled", dataset=dataset, run=run, system=system,
            evidence=f"{len(reasons)} LLM surface(s) fell back; first: {reasons[0]}",
        ))

    # Execution failures.
    if training and not training.get("ok", True):
        cat = "exec_timeout" if training.get("timed_out") else "exec_error"
        events.append(_event(
            cat, "fatal", dataset=dataset, run=run, system=system,
            evidence=str(training.get("error") or "training run failed"),
        ))
    if ensemble.get("attempted") and ensemble.get("ok") is False:
        cat = "exec_timeout" if ensemble.get("timed_out") else "exec_error"
        events.append(_event(
            cat, "handled", dataset=dataset, run=run, system=system,
            evidence="Ensembler pass failed; the chooser kept the Trainer's model — "
                     + str(ensemble.get("error") or "no error text"),
        ))

    # Critic findings across passes: recovered on a later pass -> handled;
    # persisted to the final pass -> degraded if the budget cut repair off, else detected.
    budget_bound = _budget_bound(last)
    seen_codes = {code for c in critiques for code in (c.get("finding_codes") or [])}
    last_codes = set(last.get("finding_codes") or [])
    for code, category in _FINDING_CATEGORY.items():
        if code not in seen_codes:
            continue
        persisted = code in last_codes
        if not persisted and len(critiques) > 1:
            outcome = "handled"
            note = "recovered by a later pass"
        elif persisted and budget_bound:
            outcome = "degraded"
            note = "still present when the budget cut the loop off"
        else:
            outcome = "detected"
            note = "present on the final pass" if persisted else "recorded on the only pass"
        events.append(_event(
            category, outcome, dataset=dataset, run=run, system=system,
            evidence=f"{_finding_detail(critiques, code)} [{note}]",
        ))

    # Budget cut the loop off mid-repair.
    if budget_bound:
        remaining = ", ".join(sorted(last_codes)) or "none recorded"
        events.append(_event(
            "budget_cutoff", "degraded", dataset=dataset, run=run, system=system,
            evidence=f"{last.get('reason')}; findings remaining: [{remaining}]",
        ))

    # Loop fired but moved nothing and issues remain (the Day-14 no-actuator pattern).
    if len(critiques) >= 2 and last_codes:
        first_cv, last_cv = critiques[0].get("cv_score"), last.get("cv_score")
        if isinstance(first_cv, (int, float)) and isinstance(last_cv, (int, float)):
            delta = float(last_cv) - float(first_cv)
            if delta < LOOP_PROGRESS_EPS:
                events.append(_event(
                    "loop_no_actuator", "detected", dataset=dataset, run=run, system=system,
                    evidence=f"loop ran {len(critiques)} passes, CV moved {delta:+.6f} "
                             f"(< {LOOP_PROGRESS_EPS}) with findings still open",
                ))

    # Ensemble scored below the single model — the chooser absorbed it.
    imp = ensemble.get("improvement_over_single")
    if ensemble.get("attempted") and isinstance(imp, (int, float)) and imp < 0 \
            and ensemble.get("chosen") == "single":
        events.append(_event(
            "ensemble_regression", "handled", dataset=dataset, run=run, system=system,
            evidence=f"ensemble CV {ensemble.get('ensemble_cv_score')} vs single "
                     f"{ensemble.get('single_best_cv_score')} ({imp:+.6f}); chooser kept single",
        ))

    return events


def _reporter_record_to_state(report: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Reporter record (``artifacts/crew/*/report.json``) to the final-state shape.

    The Reporter's summaries carry the same signals under different keys; this widens
    them just enough for :func:`classify_run`, so both archive shapes get one classifier.
    """
    critiques = [
        {
            "decision": p.get("decision"),
            "reason": p.get("reason"),
            "cv_score": p.get("cv_score"),
            "finding_codes": p.get("finding_codes") or [],
            "diagnoses": [],
        }
        for p in report.get("critic_passes") or []
    ]
    training = dict(report.get("training") or {})
    training.setdefault("ok", True)
    ens = report.get("ensemble") or {}
    narratives = (report.get("llm_usage") or {}).get("narratives") or []
    outage = [
        n for n in narratives
        if n.get("source") == "unavailable" and n.get("reason") not in _BENIGN_NARRATIVE_REASONS
    ]
    pseudo: dict[str, Any] = {
        "dataset_key": report.get("dataset_key"),
        "critiques": critiques,
        "training": training,
        "ensemble": {**ens, "chosen": ens.get("chosen"),
                     "improvement_over_single": ens.get("improvement_over_single")},
    }
    if outage:
        pseudo["profile"] = {"llm_narrative": {
            "source": "unavailable",
            "reason": f"{len(outage)} narrative(s) unavailable; first: {outage[0].get('reason')}",
        }}
    return pseudo


def classify_solo(solo_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify the solo-agent baseline's recorded failures (Day 3 archive).

    The solo agent has no Critic, no fallback and no chooser — its failures are always
    ``fatal`` (no scored model). A timeout maps to ``exec_timeout``; an invalid
    hyper-parameter grid is a *plan* fault surfaced at fit time (``plan_search_invalid``);
    anything else is ``exec_error``.
    """
    events: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _classify(key: str, error: str) -> None:
        if key in seen:
            return
        seen.add(key)
        if re.search(r"timed out|TimeoutExpired", error, re.IGNORECASE):
            cat = "exec_timeout"
        elif "Invalid parameter" in error:
            cat = "plan_search_invalid"
        else:
            cat = "exec_error"
        events.append(_event(
            cat, "fatal", dataset=key, run="day03_solo", system="solo",
            evidence=error.strip().splitlines()[-1][:400],
        ))

    for key, rec in (solo_metrics.get("datasets") or {}).items():
        if rec.get("ok") is False:
            _classify(key, str(rec.get("error") or "solo run failed"))
    for key, error in (solo_metrics.get("failures") or {}).items():
        _classify(key, str(error))
    return events


# --- Archive census ----------------------------------------------------------

def mine_archive() -> dict[str, Any]:
    """Classify every archived run record on disk. Returns events + census facts.

    ``artifacts/`` is git-ignored, so a fresh clone may have none of it — the census
    then reports zero runs honestly rather than failing. Nothing is re-run.
    """
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    state_paths = sorted((ARTIFACTS_DIR / "ablation").glob("*/*.json"))
    for p in state_paths:
        record = json.loads(p.read_text())
        run = f"{p.parent.name}/{p.stem}"
        evs = classify_run(record, run=run, system="crew")
        events.extend(evs)
        sources.append({"run": run, "shape": "final_state", "n_events": len(evs)})

    report_paths = sorted((ARTIFACTS_DIR / "crew").glob("*/report.json"))
    for p in report_paths:
        report = json.loads(p.read_text())
        run = f"{p.parent.name}/day11_report"
        evs = classify_run(_reporter_record_to_state(report), run=run, system="crew")
        events.extend(evs)
        sources.append({"run": run, "shape": "reporter", "n_events": len(evs)})

    n_solo = 0
    if SOLO_METRICS_PATH.exists():
        solo = json.loads(SOLO_METRICS_PATH.read_text())
        solo_events = classify_solo(solo)
        events.extend(solo_events)
        n_solo = len((solo.get("datasets") or {})) + sum(
            1 for k in (solo.get("failures") or {}) if k not in (solo.get("datasets") or {})
        )

    return {
        "n_crew_runs": len(sources),
        "n_solo_runs": n_solo,
        "sources": sources,
        "events": events,
    }


# --- Injection probes --------------------------------------------------------
# The probes run the REAL crew graph on a throwaway copy of a train split with a known
# fault injected, then read the detection surfaces. They never load or score the
# holdout, so the seal is structurally out of reach.

PROBE_PREFIX = "probe_"
# The subtle leak's target correlation: comfortably below the Profiler's regression
# screen (LEAKAGE_PEARSON = 0.98) so a miss is the *expected* behaviour under test.
SUBTLE_LEAK_CORR = 0.90
# The subtle classification leak's per-row agreement with the target: below the
# Profiler's purity screen (LEAKAGE_PURITY = 0.995) yet strong enough to inflate CV.
SUBTLE_LEAK_AGREEMENT = 0.95
LEAK_COLUMN = "leak_probe"


def make_leak_frame(
    df: pd.DataFrame,
    *,
    task: str,
    kind: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy ``df`` and inject a leaked feature; return (frame, ground-truth facts).

    ``blatant`` is an exact transform of the target (regression: the target itself;
    classification: its integer codes) — the case every screen must catch. ``subtle``
    is engineered inside the detection window: regression gets a noisy copy at
    ~``SUBTLE_LEAK_CORR`` Pearson; classification gets the codes with a fraction of
    rows flipped to a different class so per-group purity sits near
    ``SUBTLE_LEAK_AGREEMENT``. Pure over (df, rng) — the caller owns file placement.
    """
    out = df.copy()
    y = out[TARGET_COLUMN]
    if task == "regression":
        yv = y.to_numpy(dtype=float)
        if kind == "blatant":
            leak = yv.copy()
        else:
            z = (yv - yv.mean()) / (yv.std() or 1.0)
            noise = rng.standard_normal(len(yv))
            leak = SUBTLE_LEAK_CORR * z + float(np.sqrt(1.0 - SUBTLE_LEAK_CORR ** 2)) * noise
        measured = float(abs(np.corrcoef(leak, yv)[0, 1]))
    else:
        codes, uniques = pd.factorize(y)
        # Codes are offset to 1..k: a 0-heavy numeric column would trip the Profiler's
        # disguised-missing heuristic and get zero-imputed, corrupting the injected
        # signal — the probe must test the LEAKAGE screens, not collide with that one.
        leak = codes.astype(float) + 1.0
        if kind == "subtle":
            n_flip = int(round((1.0 - SUBTLE_LEAK_AGREEMENT) * len(leak)))
            flip_idx = rng.choice(len(leak), size=n_flip, replace=False)
            for i in flip_idx:
                others = [c for c in range(len(uniques)) if c != codes[i]]
                leak[i] = float(rng.choice(others)) + 1.0
        measured = float((leak == codes.astype(float) + 1.0).mean())
    out[LEAK_COLUMN] = leak
    facts = {
        "kind": kind,
        "task": task,
        "column": LEAK_COLUMN,
        "measure": "pearson_corr_with_target" if task == "regression" else "agreement_with_target",
        "measured_signal": round(measured, 6),
    }
    return out, facts


@contextmanager
def probe_dataset(base_key: str, probe_key: str, frame: pd.DataFrame) -> Iterator[DatasetSpec]:
    """Materialise ``frame`` as a temporary registered dataset; deregister on exit.

    Only ``train.parquet`` is written — a probe dataset HAS no holdout, so no code path
    can score one even by accident. The REGISTRY entry is scoped to the ``with`` block.
    """
    if not probe_key.startswith(PROBE_PREFIX):
        raise ValueError(f"probe keys must start with {PROBE_PREFIX!r}: {probe_key!r}")
    base = REGISTRY[base_key]
    spec = DatasetSpec(
        key=probe_key, openml_name=f"{base.openml_name}+injected", version=0,
        task=base.task, subtype=base.subtype, metric=base.metric,
        note=f"Day 17 failure-taxonomy probe derived from '{base_key}' — train split only, throwaway.",
    )
    ddir = DATA_DIR / probe_key
    ddir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ddir / "train.parquet", index=False)
    REGISTRY[probe_key] = spec
    try:
        yield spec
    finally:
        REGISTRY.pop(probe_key, None)


def _run_probe_crew(spec: DatasetSpec, *, max_iterations: int = 2) -> dict[str, Any]:
    """Run the full crew graph on a probe spec and return the final state (no scoring)."""
    app = build_crew(variant="full")
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    return app.invoke(state, config={"recursion_limit": limit})


def run_leak_probe(kind: str, *, base_key: Optional[str] = None) -> dict[str, Any]:
    """Inject a leaked column, run the real crew, and report which surfaces fired.

    ``blatant`` (regression base): the screen must catch it — expected outcome
    ``handled`` (flagged by the Profiler, dropped by the Planner). ``subtle``
    (classification base): engineered inside the detection window — the expected
    result is a MISS, and the probe assigns the ``leakage_missed`` event from ground
    truth, with the CV inflation vs the flagged/dropped path as the damage evidence.
    """
    if kind not in ("blatant", "subtle"):
        raise ValueError(f"unknown leak probe kind: {kind!r}")
    base_key = base_key or ("cpu_small" if kind == "blatant" else "credit-g")
    base = REGISTRY[base_key]
    rng = np.random.default_rng(SEED)
    frame, truth = make_leak_frame(
        pd.read_parquet(DATA_DIR / base_key / "train.parquet"),
        task=base.task, kind=kind, rng=rng,
    )
    probe_key = f"{PROBE_PREFIX}leak_{kind}"

    started = time.time()
    with probe_dataset(base_key, probe_key, frame) as spec:
        final = _run_probe_crew(spec)
    seconds = round(time.time() - started, 2)

    profile = final.get("profile") or {}
    plan = final.get("plan") or {}
    critiques = final.get("critiques") or []
    tmetrics = (final.get("training") or {}).get("metrics") or {}

    flagged = [
        d["column"] for d in (profile.get("leakage_checks") or {}).get("target_correlated_features", [])
    ]
    profiler_flagged = LEAK_COLUMN in flagged
    plan_dropped = LEAK_COLUMN in (plan.get("drop_columns") or [])
    critic_fired = any("leakage" in (c.get("finding_codes") or []) for c in critiques)
    detected = profiler_flagged or critic_fired
    model_saw_leak = not plan_dropped

    events = classify_run(final, run=f"day17_{probe_key}", system="crew")
    if not detected:
        events.append(_event(
            "leakage_missed", "missed", dataset=probe_key, run=f"day17_{probe_key}", system="crew",
            evidence=f"injected '{LEAK_COLUMN}' ({truth['measure']}={truth['measured_signal']}) "
                     f"passed every screen; the model trained on it "
                     f"(CV {plan.get('metric')}={tmetrics.get('best_cv_score')})",
        ))

    return {
        "probe": f"leak_{kind}",
        "base_dataset": base_key,
        "ground_truth": truth,
        "expectation": "caught (screen must fire)" if kind == "blatant"
                       else "missed (engineered inside the detection window)",
        "profiler_flagged": profiler_flagged,
        "plan_dropped": plan_dropped,
        "critic_leakage_finding": critic_fired,
        "detected": detected,
        "model_saw_leak": model_saw_leak,
        "cv_score": tmetrics.get("best_cv_score"),
        "cv_metric": plan.get("metric"),
        "iterations_run": final.get("iteration"),
        "mock": is_mock_mode(),
        "seconds": seconds,
        "events": events,
    }


# The timeout probe must re-exec: the executor reads its cap from the environment at
# import time, so an in-process env change cannot reach it. The child runs the real
# crew under a starved cap and dumps its final state for the parent to classify.
_TIMEOUT_PROBE_CHILD = """\
import json, sys
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY
spec = REGISTRY[sys.argv[2]]
app = build_crew(variant="full")
final = app.invoke(initial_state(spec, max_iterations=1), config={"recursion_limit": 17})
with open(sys.argv[1], "w") as fh:
    json.dump(final, fh, default=str)
"""

# Starved cap for the child: generous enough for the Feature Engineer's sub-second
# validation pass, far below full-capacity training — so the fault lands in the Trainer.
TIMEOUT_PROBE_CAP_S = 5


def run_timeout_probe(base_key: str = "cpu_small", *, child_timeout_s: int = 900) -> dict[str, Any]:
    """Starve the executor's wall-clock cap and verify the timeout path end to end.

    Expected chain: the sandbox kills the training subprocess at the cap ->
    ``timed_out=True`` -> the Critic files an ``execution_error`` blocker and finalises
    -> classify_run emits ``exec_timeout``/``fatal``. The probe fails honestly if the
    chain breaks anywhere (including the run unexpectedly finishing under the cap).
    """
    out_path = ARTIFACTS_DIR / "taxonomy" / f"day17_timeout_probe_{base_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "CREWML_EXECUTOR_TIMEOUT_S": str(TIMEOUT_PROBE_CAP_S)}
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", _TIMEOUT_PROBE_CHILD, str(out_path), base_key],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=child_timeout_s,
    )
    seconds = round(time.time() - started, 2)

    if proc.returncode != 0 or not out_path.exists():
        return {
            "probe": "exec_timeout", "base_dataset": base_key, "ok": False,
            "seconds": seconds,
            "error": (proc.stderr or "child produced no final state").strip()[-600:],
            "events": [],
        }

    final = json.loads(out_path.read_text())
    training = final.get("training") or {}
    critiques = final.get("critiques") or []
    events = classify_run(final, run=f"day17_probe_timeout_{base_key}", system="crew")
    return {
        "probe": "exec_timeout",
        "base_dataset": base_key,
        "ok": True,
        "cap_s": TIMEOUT_PROBE_CAP_S,
        "expectation": "trainer killed at the cap; Critic files execution_error and finalises",
        "trainer_timed_out": bool(training.get("timed_out")),
        "critic_filed_blocker": any(
            "execution_error" in (c.get("finding_codes") or []) for c in critiques
        ),
        "detected": bool(training.get("timed_out")),
        "mock": is_mock_mode(),
        "seconds": seconds,
        "events": events,
    }


def run_record_probes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Record-level probes: mutate a real archived record, run the Critic's real detector.

    Covers the categories a live run cannot cheaply produce (the deterministic Planner
    never emits a wrong scorer, and forcing a live crash would mean sabotaging code):
    each probe mutates one field of a *real* final state and asserts the Critic's
    ``diagnose`` — the production detector, not a test double — fires the right code.
    Labelled ``record_level`` so they are never mistaken for live runs.
    """
    probes = []

    mutated = deepcopy(record)
    mutated["plan"]["cv"]["scoring"] = "accuracy"
    findings = diagnose(mutated["profile"], mutated["plan"], mutated["training"])
    codes = [f["code"] for f in findings]
    probes.append({
        "probe": "wrong_metric", "record_level": True,
        "mutation": "plan.cv.scoring set to 'accuracy' (primary metric unchanged)",
        "expected_code": "wrong_metric",
        "detected": "wrong_metric" in codes,
        "detector": "crewml.crew.critic.diagnose",
        "finding_codes": codes,
    })

    mutated = deepcopy(record)
    mutated["training"]["ok"] = False
    mutated["training"]["error"] = "ValueError: Invalid parameter 'alpha' for estimator (record-level probe)"
    findings = diagnose(mutated["profile"], mutated["plan"], mutated["training"])
    codes = [f["code"] for f in findings]
    probes.append({
        "probe": "exec_error", "record_level": True,
        "mutation": "training.ok forced False with a real-shaped sklearn error",
        "expected_code": "execution_error",
        "detected": "execution_error" in codes,
        "detector": "crewml.crew.critic.diagnose",
        "finding_codes": codes,
    })

    return probes


# --- Aggregation + report ----------------------------------------------------

def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Census counts: per category (split by outcome and system) + per outcome totals."""
    by_category: dict[str, dict[str, Any]] = {}
    for e in events:
        c = by_category.setdefault(e["category"], {
            "group": e["group"], "stage": e["stage"], "total": 0,
            "by_outcome": {}, "by_system": {},
        })
        c["total"] += 1
        c["by_outcome"][e["outcome"]] = c["by_outcome"].get(e["outcome"], 0) + 1
        c["by_system"][e["system"]] = c["by_system"].get(e["system"], 0) + 1
    by_outcome = {o: sum(1 for e in events if e["outcome"] == o) for o in OUTCOMES}
    fatal_by_system: dict[str, int] = {}
    for e in events:
        if e["outcome"] == "fatal":
            fatal_by_system[e["system"]] = fatal_by_system.get(e["system"], 0) + 1
    return {
        "n_events": len(events),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1]["total"])),
        "by_outcome": by_outcome,
        "fatal_by_system": fatal_by_system,
    }


def assemble_report(
    archive: dict[str, Any],
    leak_probes: list[dict[str, Any]],
    timeout_probe: dict[str, Any],
    record_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bundle census + probes into the committed Day-17 report object."""
    probe_events = [e for p in leak_probes for e in p["events"]] + timeout_probe.get("events", [])
    return {
        "schema_version": TAXONOMY_SCHEMA_VERSION,
        "day": 17,
        "phase": 3,
        "study": "failure_taxonomy",
        "mock": is_mock_mode(),
        "taxonomy": CATEGORIES,
        "outcomes": list(OUTCOMES),
        "archive_census": {
            "n_crew_runs": archive["n_crew_runs"],
            "n_solo_runs": archive["n_solo_runs"],
            "summary": summarise(archive["events"]),
            "events": archive["events"],
        },
        "probes": {
            "live_leak": leak_probes,
            "live_timeout": timeout_probe,
            "record_level": record_probes,
            "summary": summarise(probe_events),
        },
    }


# --- Rendering ---------------------------------------------------------------

def _yn(v: bool) -> str:
    return "yes" if v else "no"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the committed Day-17 board."""
    census = report["archive_census"]
    summary = census["summary"]
    lines = [
        "# Day 17 — Failure taxonomy",
        "",
        "*A closed vocabulary of failure categories — each owned by a stage and tied to the "
        "detection surface that should catch it — applied to every archived run record "
        "(census: nothing re-run, nothing flattered) and to live injection probes that "
        "plant known faults and watch what fires. `missed` is only ever assigned when an "
        "injection probe supplies ground truth; a census structurally cannot see what "
        "nothing detected.*",
        "",
        "## The taxonomy",
        "",
        "| Code | Group | Stage | Detection surface |",
        "|---|---|---|---|",
    ]
    for code, meta in CATEGORIES.items():
        lines.append(f"| `{code}` | {meta['group']} | {meta['stage']} | {meta['surface']} |")

    lines += [
        "",
        f"## Archive census — {census['n_crew_runs']} crew runs + "
        f"{census['n_solo_runs']} solo-agent runs",
        "",
        "| Category | Total | fatal | degraded | handled | detected | crew | solo |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for code, c in summary["by_category"].items():
        o, s = c["by_outcome"], c["by_system"]
        lines.append(
            f"| `{code}` | {c['total']} | {o.get('fatal', 0)} | {o.get('degraded', 0)} | "
            f"{o.get('handled', 0)} | {o.get('detected', 0)} | "
            f"{s.get('crew', 0)} | {s.get('solo', 0)} |"
        )
    fatal = summary["fatal_by_system"]
    n_crew_fatal = fatal.get("crew", 0)
    if n_crew_fatal:
        crew_fatal_note = (
            f"The {n_crew_fatal} crew-side fatal(s) were each *caught and filed* — the Critic "
            "recorded the `execution_error` blocker and finalised honestly without a model "
            "rather than shipping garbage; automated self-repair (feed the traceback back and "
            "retry) is Day 20's feature and these are its motivating cases. Every other "
            "crew-side event was absorbed by a guard (`handled`), disclosed as "
            "quality-impacting (`degraded`), or recorded (`detected`)."
        )
    else:
        crew_fatal_note = (
            "Every crew-side event above was either absorbed by a guard (`handled`), "
            "disclosed as quality-impacting (`degraded` — the deliberate budget-starvation "
            "runs of Day 15), or recorded with indeterminate impact (`detected`)."
        )
    lines += [
        "",
        f"Fatal failures (no scored model): **crew {n_crew_fatal}** vs "
        f"**solo {fatal.get('solo', 0)}** across the whole archive. " + crew_fatal_note
        + " The solo agent's failures are all fatal: it has "
        "no Critic to file the fault, no fallback to absorb it, and no chooser to contain it.",
        "",
        "## Injection probes — is each surface actually live?",
        "",
        "| Probe | Fault injected | Expectation | Detected | Where it fired |",
        "|---|---|---|---|---|",
    ]
    for p in report["probes"]["live_leak"]:
        gt = p["ground_truth"]
        fired = []
        if p["profiler_flagged"]:
            fired.append("Profiler screen")
        if p["plan_dropped"]:
            fired.append("Planner drop")
        if p["critic_leakage_finding"]:
            fired.append("Critic finding")
        lines.append(
            f"| `{p['probe']}` (live, on {p['base_dataset']}) | leaked column, "
            f"{gt['measure']}={gt['measured_signal']} | {p['expectation']} | "
            f"{_yn(p['detected'])} | {', '.join(fired) or '— nothing fired'} |"
        )
    tp = report["probes"]["live_timeout"]
    if tp.get("ok"):
        fired = []
        if tp["trainer_timed_out"]:
            fired.append("sandbox kill + `timed_out`")
        if tp["critic_filed_blocker"]:
            fired.append("Critic blocker")
        lines.append(
            f"| `exec_timeout` (live, on {tp['base_dataset']}) | executor cap starved to "
            f"{tp['cap_s']}s | {tp['expectation']} | {_yn(tp['detected'])} | "
            f"{', '.join(fired) or '— nothing fired'} |"
        )
    else:
        lines.append(
            f"| `exec_timeout` (live) | executor cap starved | probe run | FAILED | {tp.get('error', '')[:120]} |"
        )
    for p in report["probes"]["record_level"]:
        lines.append(
            f"| `{p['probe']}` (record-level) | {p['mutation']} | Critic `diagnose` fires "
            f"`{p['expected_code']}` | {_yn(p['detected'])} | {p['detector']} |"
        )

    subtle = next((p for p in report["probes"]["live_leak"] if p["probe"] == "leak_subtle"), None)
    if subtle is not None and not subtle["detected"]:
        lines += [
            "",
            "### The measured detection window (the honest finding)",
            "",
            f"The subtle probe planted a leaked column agreeing with the target on "
            f"{subtle['ground_truth']['measured_signal']:.1%} of rows — below the Profiler's "
            "purity screen (0.995) — and the resulting CV score "
            f"({subtle['cv_metric']}={subtle['cv_score']}) stayed under the Critic's "
            "too-good-to-be-true ceiling (0.995). **No surface fired and the model trained on "
            "the leak.** That window — leak strong enough to inflate the score, weak enough to "
            "pass both screens — is a real, now-measured gap. Logged as Day 22 (leakage & "
            "honesty guards) input, not patched today: the taxonomy's job is to find gaps, "
            "and papering one over inside the study that found it would defeat the study.",
        ]
    # Label the LLM situation honestly: mock mode, or a configured-but-failing provider
    # (the probes' own runs record the outage) — either way the numbers above are the
    # deterministic core's, and are never presented as live-LLM results.
    probe_outage = any(
        e["category"] == "provider_outage"
        for p in report["probes"]["live_leak"] for e in p["events"]
    )
    if report["mock"]:
        lines += ["", "*(mock)* — no live LLM key; deterministic core only (EVAL_PROTOCOL.md §5)."]
    elif probe_outage:
        lines += [
            "",
            "*Provider status:* a key is configured but every LLM call failed during the live "
            "probes (see the runs' `provider_outage` events — the Day-15 Groq restriction is "
            "still in effect), so all probe runs executed on the deterministic core. Detection "
            "surfaces are deterministic by design, so the probe verdicts are unaffected — but "
            "no number here is a live-LLM result.",
        ]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    TAXONOMY_RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    TAXONOMY_TABLE_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report
