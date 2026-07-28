"""Day 20 — measuring the self-repair loop: what fraction of crashes come back?

:mod:`crewml.repair` gives a crashed generation a bounded second chance. This
module answers the question that makes the feature a *result* rather than a
feature: **when generated code breaks, how often does the loop actually recover
a scored model — and does the recovered model score what the unbroken one
would have?**

Natural crashes are too rare to measure (Phase 3's census found two, both in the
solo baseline), so the study uses **fault injection**, the same discipline as
Day 13's forced-deficiency probe and Day 17's detection probes: take the real
Trainer pipeline on real datasets, splice a *known, realistic bug* into the FE
module the training script embeds, run with self-repair ON, and watch. Because
each fault is injected downstream of FE validation (the study hands the faulty
source straight to :func:`~crewml.crew.trainer.run_trainer`), every fault
actually detonates inside the sandboxed training run — this measures the
Trainer's repair net, the last one before the Critic.

The fault suite spans the crash families the Day-17 taxonomy filed under
``exec_error`` — name/key/type/attribute/index errors, a syntax error, a
division by zero, and a sandbox-refused import — one bug each, every one of
them a single planted line in an otherwise-working module.

Per (dataset x fault) the study records: recovered?, attempts used, wall-clock,
token cost, and — the honesty metric — **score fidelity**: the repaired run's
CV score against the same dataset's clean-run score. A repair that "recovers"
to a materially different number didn't restore the pipeline, it replaced it;
fidelity is how we would notice. A clean control run per dataset doubles as the
no-false-positive check (repair must NOT fire on working code).

Honesty rules, as ever: the locked holdout is never touched (everything here is
CV-on-train through the standard no-peeking Trainer path); every number in the
emitted JSON carries provider/model provenance.

**Two modes, and the reason the second exists.** ``repairer="live"`` is the real
study. It refuses to run in mock mode *and* refuses when a provider is
configured-but-dead — a distinction learned on 2026-07-25, when a restricted Groq
account failed every call and the study was minutes from publishing "recovery
rate 0%", a figure describing a billing state rather than a repair loop.
``repairer="scripted"`` swaps in a deterministic, fault-blind stand-in
(:func:`scripted_repairer`) so the *mechanism* can still be measured when no
provider is reachable: fault detonation inside the real Trainer, loop firing,
sandboxed re-run adoption, score fidelity, artifact consistency, and the
no-false-positive property on clean runs. Reports from that mode are stamped
``is_measurement_of_llm_capability: false`` and every renderer reproduces the
label, because a mechanism check is not a capability result.
"""
from __future__ import annotations

import json
import textwrap
import time
from typing import Any, Optional

from crewml.config import (
    ARTIFACTS_DIR,
    GROQ_MODEL,
    LLM_PROVIDER,
    RESULTS_DIR,
    SELF_REPAIR_MAX_ATTEMPTS,
    is_mock_mode,
)
from crewml import config as config_module
from crewml import llm
from crewml.crew.feature_engineer import DEFAULT_FE_SOURCE
from crewml.crew.planner import build_plan
from crewml.crew.profiler import build_profile
from crewml.crew.trainer import run_trainer
from crewml.datasets import REGISTRY, load_train, verify_holdout_untouched

SELF_REPAIR_SCHEMA_VERSION = 1
SELF_REPAIR_RESULT_PATH = RESULTS_DIR / "day20_self_repair.json"
SELF_REPAIR_TABLE_MD_PATH = RESULTS_DIR / "day20_self_repair.md"

# One classification + one regression set keeps the live bill honest while
# covering both task families; the loop under test is dataset-agnostic.
STUDY_DATASETS: tuple[str, ...] = ("credit-g", "cpu_small")


def _fe_module(body: str, *, prelude: str = "") -> str:
    """A complete FE module with ``body`` as the add_features implementation."""
    return (
        '"""Injected-fault FE module — Day 20 self-repair study. NOT crew output."""\n'
        "import pandas as pd\n"
        + prelude
        + "\n\ndef add_features(df):\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
    )


# The fault suite: every entry is a realistic single-bug corruption of a
# working FE module. ``taxonomy`` names the Day-17 category the crash files
# under; every one of these lands as ``exec_error`` (that is the point — the
# repair loop is the new second net under exactly that category).
FAULTS: tuple[dict[str, str], ...] = (
    {
        "key": "name_error",
        "taxonomy": "exec_error",
        "description": "references an undefined variable",
        "source": _fe_module(
            """\
            out = df.copy()
            out["row_nan_count"] = df.isna().sum(axis=1) + undefined_offset
            return out
            """
        ),
    },
    {
        "key": "key_error",
        "taxonomy": "exec_error",
        "description": "indexes a column that does not exist",
        "source": _fe_module(
            """\
            out = df.copy()
            out["ratio"] = out["definitely_not_a_real_column"] / 2.0
            return out
            """
        ),
    },
    {
        "key": "type_error",
        "taxonomy": "exec_error",
        "description": "adds a string to a numeric series",
        "source": _fe_module(
            """\
            out = df.copy()
            out["row_nan_count"] = df.isna().sum(axis=1) + "1"
            return out
            """
        ),
    },
    {
        "key": "syntax_error",
        "taxonomy": "exec_error",
        "description": "unbalanced parenthesis — the module cannot even parse",
        "source": _fe_module(
            """\
            out = df.copy()
            out["row_nan_count"] = df.isna().sum(axis=1
            return out
            """
        ),
    },
    {
        "key": "zero_division",
        "taxonomy": "exec_error",
        "description": "unconditional integer division by zero",
        "source": _fe_module(
            """\
            scale = 1 // 0
            out = df.copy()
            out["scaled"] = df.isna().sum(axis=1) * scale
            return out
            """
        ),
    },
    {
        "key": "import_error",
        "taxonomy": "exec_error",
        "description": "imports a package the Day-19 sandbox refuses",
        "source": _fe_module(
            """\
            out = df.copy()
            out["row_nan_count"] = df.isna().sum(axis=1)
            return out
            """,
            prelude="import requests\n",
        ),
    },
    {
        "key": "attribute_error",
        "taxonomy": "exec_error",
        "description": "calls a DataFrame method that does not exist",
        "source": _fe_module(
            """\
            out = df.copy()
            out["row_nan_count"] = df.isna().applymapp(int).sum(axis=1)
            return out
            """
        ),
    },
    {
        "key": "index_error",
        "taxonomy": "exec_error",
        "description": "positional index far past the end of the frame",
        "source": _fe_module(
            """\
            anchor = df.iloc[10**9]
            out = df.copy()
            out["row_nan_count"] = df.isna().sum(axis=1)
            return out
            """
        ),
    },
    # The only fault in this suite that is NOT invented. On a live Phase-3 run
    # (diabetes) the Feature Engineer's generated code built an unguarded ratio
    # that produced +/-inf; FE validation passed it (its nan check does not test
    # finiteness), and training died at sklearn's finite assertion. That is the
    # crew's one real fatal failure to date and the motivating case for this
    # whole day, so the measurement would be hollow without it. Verified to
    # reproduce the same ValueError("Input X contains infinity ...").
    {
        "key": "non_finite",
        "taxonomy": "exec_error",
        "description": "unguarded ratio emits +/-inf — the crew's one REAL live crash",
        "source": _fe_module(
            """\
            out = df.copy()
            num = df.select_dtypes(include="number")
            col = num.columns[0]
            out["unguarded_ratio"] = df[col] / (df[col] - df[col])
            return out
            """
        ),
    },
)


# --- The scripted repairer: mechanism validation without a provider ----------
#
# When no live provider is reachable the recovery RATE is unmeasurable (see the
# preflight in run_self_repair_study). But the *mechanism* — does an injected
# fault detonate inside the real Trainer, does the loop fire, does the sandboxed
# re-run get adopted, does the adopted run reproduce the clean score, does the
# persisted fe_source.py artifact stay consistent — is measurable offline, and
# worth measuring, provided the label is unambiguous.
#
# The stand-in below is NOT a model and is NOT an oracle over the fault list. It
# applies one fixed, generic repair policy to whatever source it is shown:
#
#   1. strip any import of a module the Day-19 sandbox allowlist refuses;
#   2. replace the body of ``add_features`` with the contract-minimal known-good
#      implementation, and update the FE_SOURCE_TEXT artifact constant to match.
#
# That policy is blind to which fault it is repairing — it is the deterministic
# "revert the broken part to something that honours the contract" a maintainer
# would reach for. What it measures: the plumbing. What it does NOT measure:
# whether an LLM can diagnose a traceback. Any report using this mode must say
# so in those words.

_SANDBOX_ALLOWED_TOPS = frozenset(
    {"numpy", "pandas", "scipy", "sklearn", "joblib", "matplotlib", "json",
     "math", "os", "sys", "re", "crew_io", "textwrap", "warnings"}
)

_CANONICAL_FE = (
    'import pandas as pd\n\n\n'
    'def add_features(df):\n'
    '    out = df.copy()\n'
    '    out["row_nan_count"] = df.isna().sum(axis=1).astype("int64")\n'
    '    return out\n'
)


def _strip_refused_imports(lines: list[str]) -> list[str]:
    kept = []
    for line in lines:
        stripped = line.strip()
        top = None
        if stripped.startswith("import "):
            top = stripped[len("import "):].split(".")[0].split(" as ")[0].strip()
        elif stripped.startswith("from ") and " import " in stripped:
            top = stripped[len("from "):].split(" import ")[0].split(".")[0].strip()
        if top and top not in _SANDBOX_ALLOWED_TOPS:
            continue
        kept.append(line)
    return kept


def _replace_add_features(source: str) -> str:
    """Swap the module's ``add_features`` for the canonical minimal version."""
    lines = _strip_refused_imports(source.splitlines())
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("def add_features")), None
    )
    if start is None:
        return "\n".join(lines) + "\n"
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end][:1].isspace()
    ):
        end += 1
    out = lines[:start] + _CANONICAL_FE.splitlines()[2:] + lines[end:]
    # Keep the persisted FE artifact consistent with the repaired function.
    out = [
        f"FE_SOURCE_TEXT = {_CANONICAL_FE!r}"
        if ln.startswith("FE_SOURCE_TEXT = ")
        else ln
        for ln in out
    ]
    return "\n".join(out) + "\n"


def scripted_repairer():
    """A deterministic stand-in for ``llm.chat`` implementing the policy above."""
    calls: list[str] = []

    def chat(system: str, user: str, **kwargs: Any):
        calls.append(user)
        blocks = llm._CODE_FENCE.findall(user)
        source = max(blocks, key=len) if blocks else ""
        return llm.LLMResult(
            text=f"```python\n{_replace_add_features(source)}```",
            provider="scripted_stand_in",
            model="deterministic-repair-policy",
            prompt_tokens=0,
            completion_tokens=0,
        )

    chat.calls = calls
    return chat


# --- Which faults have a RESTORABLE intent (fidelity's scope) ----------------
#
# Score fidelity compares a repaired run against the clean control, whose FE is
# DEFAULT_FE_SOURCE. Most faults here are "the default FE plus one planted bug",
# so the only sensible fix converges back on the default FE and Δ == 0 is very
# nearly forced — which makes Δ == 0 a useful *regression* check (the repair did
# not wander off) but NOT independent evidence that the loop "restored" anything.
#
# Two faults have no restorable intent at all:
#   * key_error   — indexes a column that does not exist, so there is no original
#                   feature to rebuild; any fix must invent something else.
#   * non_finite  — an unguarded ratio; correctly guarding it yields a genuinely
#                   different (and legitimate) feature, not the default's.
# For those, Δ != 0 is the EXPECTED outcome and must never be read as misbehaviour.
# Reporting one mean over both groups would blur exactly that distinction, so the
# summary scopes fidelity to the restorable faults and lists the others.
NON_RESTORABLE_FAULTS: frozenset[str] = frozenset({"key_error", "non_finite"})


def _plan_for(key: str) -> dict[str, Any]:
    """The deterministic plan core — no LLM tokens spent on planning here."""
    return build_plan(build_profile(REGISTRY[key], load_train(key)))


def _fe_artifact_consistent(training: dict[str, Any]) -> Optional[bool]:
    """Is the run's persisted ``fe_source.py`` WELL-FORMED — parses and defines
    ``add_features``?

    Honesty note: despite the name this is a well-formedness check, **not** proof
    that the artifact matches the feature engineering the shipped model was fitted
    with. It catches the realistic corruption (a repair that ships a stale or
    unparseable artifact and thereby sabotages later holdout scoring); it cannot
    catch an artifact that is valid Python but describes different features.
    Reported as "FE-artifact malformed" for that reason.
    """
    if not training.get("ok") or not training.get("run_id"):
        return None
    art = ARTIFACTS_DIR / "executor" / training["run_id"] / "artifacts" / "fe_source.py"
    try:
        source = art.read_text(encoding="utf-8")
        compile(source, str(art), "exec")
        return "def add_features" in source
    except (OSError, SyntaxError):
        return False


def _run_record(
    dataset_key: str,
    fault: Optional[dict[str, str]],
    training: dict[str, Any],
    wall_s: float,
    clean_score: Optional[float],
) -> dict[str, Any]:
    repair = training.get("repair") or {}
    score = training.get("cv_score")
    fidelity = None
    if score is not None and clean_score is not None:
        fidelity = round(score - clean_score, 6)
    return {
        "dataset": dataset_key,
        "fault": fault["key"] if fault else "none_control",
        "taxonomy": fault["taxonomy"] if fault else "control",
        "description": fault["description"] if fault else "clean FE — repair must not fire",
        "ok": bool(training.get("ok")),
        "repair_attempted": bool(repair.get("attempted")),
        "recovered": bool(repair.get("recovered")),
        "recovered_on_attempt": repair.get("recovered_on_attempt"),
        "attempts_used": len(repair.get("attempts") or []),
        "attempt_errors": [
            {
                "attempt": a.get("attempt"),
                "stage": a.get("stage"),
                "ok": a.get("ok"),
                # Keep the reason, truncated. Without it a provider failure is
                # indistinguishable from a model failure in the committed record —
                # which is how a rate limit gets published as a 0% recovery rate.
                "error": (a.get("error") or "")[:240] or None,
            }
            for a in (repair.get("attempts") or [])
        ],
        # True when repair was attempted, did not recover, and EVERY attempt died
        # at the llm stage — i.e. the model was never actually consulted. Such a
        # run is UNMEASURED, not a failure to repair.
        "llm_unavailable": bool(
            repair.get("attempted")
            and not repair.get("recovered")
            and (repair.get("attempts") or [])
            and all(a.get("stage") == "llm" for a in repair["attempts"])
        ),
        "cv_score": score,
        "score_fidelity_vs_clean": fidelity,
        # False => a correct fix CANNOT reproduce the control's FE, so a non-zero
        # fidelity is expected here and is not evidence of a bad repair.
        "restorable": bool(fault is None or fault["key"] not in NON_RESTORABLE_FAULTS),
        "fe_artifact_consistent": _fe_artifact_consistent(training),
        "prompt_tokens": int(repair.get("total_prompt_tokens") or 0),
        "completion_tokens": int(repair.get("total_completion_tokens") or 0),
        "wall_s": round(wall_s, 1),
    }


def run_self_repair_study(
    datasets: tuple[str, ...] = STUDY_DATASETS,
    faults: tuple[dict[str, str], ...] = FAULTS,
    *,
    progress: bool = True,
    repairer: str = "live",
) -> dict[str, Any]:
    """Inject every fault on every study dataset; measure the recovery rate.

    ``repairer="live"`` (default) is the real measurement: the configured
    provider does the repairing. It refuses to run in mock mode — a recovery
    rate without a live provider is definitionally zero and must not be
    published as a measurement — and refuses just as hard when a provider is
    *configured but dead* (revoked key, restricted org, outage), since
    ``is_mock_mode`` only checks key presence. Without that preflight the study
    would run 18 trainer passes against a provider failing every call and
    publish "recovery rate 0%" — a number measuring the provider's billing
    state, not the loop. Discovered the hard way on 2026-07-25, when Groq
    returned ``organization_restricted`` for every call.

    ``repairer="scripted"`` substitutes the deterministic stand-in
    (:func:`scripted_repairer`) for the provider. This measures the **mechanism**
    — fault detonates, loop fires, sandboxed re-run is adopted, score is
    reproduced, artifact stays consistent — and explicitly **not** whether a
    model can diagnose a traceback. The emitted report is stamped
    ``is_measurement_of_llm_capability: false`` and every consumer must carry
    that label.
    """
    if repairer not in ("live", "scripted"):
        raise ValueError(f"unknown repairer {repairer!r}; use 'live' or 'scripted'")
    # Refuse an empty matrix up front. Otherwise the pass "succeeds" with nothing
    # injected, overwrites the committed results JSON with a contentless report,
    # and only then crashes formatting a None rate — losing a real measurement to
    # a typo in --faults.
    if not datasets:
        raise ValueError("no datasets selected — nothing to measure")
    if not faults:
        raise ValueError(
            "no faults selected — check --faults against "
            f"{sorted(f['key'] for f in FAULTS)}"
        )

    if repairer == "live":
        if is_mock_mode():
            raise RuntimeError(
                "self-repair study requires a live LLM provider; a mock-mode "
                "'recovery rate' would be fiction (EVAL_PROTOCOL §5)"
            )
        try:
            llm.chat(
                "You are a liveness probe for a measurement harness.",
                "Reply with the single word OK.",
                temperature=0.0,
                max_tokens=8,
            )
        except Exception as exc:
            raise RuntimeError(
                "live provider preflight failed — refusing to measure a recovery "
                f"rate against a dead provider ({type(exc).__name__}: {exc}). A 0% "
                "rate from failed API calls would be a false measurement, not a "
                "result (EVAL_PROTOCOL §5)."
            ) from exc
        return _execute_study(datasets, faults, progress=progress, repairer=repairer)

    # Scripted mode: the stand-in replaces the provider for the whole study, and
    # is_mock_mode is forced False so the repair loop engages (it correctly
    # refuses to call a provider in mock mode).
    import unittest.mock as mock

    with mock.patch.object(config_module, "is_mock_mode", lambda: False), \
         mock.patch.object(llm, "chat", scripted_repairer()):
        return _execute_study(datasets, faults, progress=progress, repairer=repairer)


def _execute_study(
    datasets: tuple[str, ...],
    faults: tuple[dict[str, str], ...],
    *,
    progress: bool,
    repairer: str,
) -> dict[str, Any]:
    """The study body, run under whichever repairer the caller established."""

    runs: list[dict[str, Any]] = []
    clean_scores: dict[str, Optional[float]] = {}

    for key in datasets:
        plan = _plan_for(key)

        # Clean control: repair must not fire on working code.
        t0 = time.monotonic()
        control = run_trainer(
            plan, DEFAULT_FE_SOURCE, key, param_search=False, self_repair=True
        )
        clean_scores[key] = control.get("cv_score")
        runs.append(_run_record(key, None, control, time.monotonic() - t0, None))
        if progress:
            print(
                f"[study] {key} control: ok={control['ok']} "
                f"cv={control.get('cv_score')}", flush=True,
            )

        for fault in faults:
            t0 = time.monotonic()
            training = run_trainer(
                plan, fault["source"], key, param_search=False, self_repair=True
            )
            rec = _run_record(key, fault, training, time.monotonic() - t0, clean_scores[key])
            runs.append(rec)
            if progress:
                print(
                    f"[study] {key} {fault['key']}: recovered={rec['recovered']} "
                    f"attempts={rec['attempts_used']} cv={rec['cv_score']} "
                    f"fidelity={rec['score_fidelity_vs_clean']}", flush=True,
                )

    injected = [r for r in runs if r["fault"] != "none_control"]
    controls = [r for r in runs if r["fault"] == "none_control"]
    recovered = [r for r in injected if r["recovered"]]
    # A run whose repair never reached the model measures the provider's quota,
    # not the loop. Exclude those from the denominator instead of scoring them as
    # failures — on 2026-07-25 a mid-run Groq daily-token exhaustion turned an
    # entire 18-run pass into a spurious "recovery rate 0%".
    unmeasured = [r for r in injected if r["llm_unavailable"]]
    measurable = [r for r in injected if not r["llm_unavailable"]]
    measurable_recovered = [r for r in measurable if r["recovered"]]
    # Scope fidelity to faults whose intent a correct fix CAN restore (see
    # NON_RESTORABLE_FAULTS). Averaging the two groups together would hide the
    # only distinction that makes the metric meaningful.
    fidelity_vals = [
        abs(r["score_fidelity_vs_clean"])
        for r in recovered
        if r["score_fidelity_vs_clean"] is not None and r["restorable"]
    ]
    non_restorable_observed = [
        {"dataset": r["dataset"], "fault": r["fault"],
         "delta": r["score_fidelity_vs_clean"]}
        for r in recovered
        if not r["restorable"] and r["score_fidelity_vs_clean"] is not None
    ]

    scripted = repairer == "scripted"
    report = {
        "schema_version": SELF_REPAIR_SCHEMA_VERSION,
        "study": "day20_self_repair",
        "repairer": repairer,
        # The honesty stamp. False => this run measures the harness/adoption
        # path, NOT whether a model can diagnose a traceback. Every table, chart
        # and report rendered from this JSON must reproduce the label.
        "is_measurement_of_llm_capability": not scripted,
        "repairer_note": (
            "DETERMINISTIC STAND-IN, NOT AN LLM: a fixed fault-blind policy "
            "(strip sandbox-refused imports; restore add_features to the "
            "contract-minimal implementation). Measures the mechanism — fault "
            "detonation, loop firing, sandboxed re-run adoption, score fidelity, "
            "artifact consistency. Says nothing about model repair skill."
            if scripted
            else "Live provider performed every repair."
        ),
        "provider": "scripted_stand_in" if scripted else LLM_PROVIDER,
        "model": (
            "deterministic-repair-policy"
            if scripted
            else (GROQ_MODEL if LLM_PROVIDER == "groq" else None)
        ),
        # EVAL_PROTOCOL §5: a result produced without a live LLM IS mock and must
        # be labelled so. Scripted mode contacts no provider at all, so hard-coding
        # False here would have mislabelled it for every generic consumer.
        "is_mock": bool(scripted or is_mock_mode()),
        "max_attempts": SELF_REPAIR_MAX_ATTEMPTS,
        "datasets": list(datasets),
        "n_faults": len(faults),
        "n_injected_runs": len(injected),
        # Rate over MEASURABLE runs only, and None when nothing was measurable —
        # never a number manufactured from provider failures.
        "recovery_rate": (
            round(len(measurable_recovered) / len(measurable), 4) if measurable else None
        ),
        "recovered_runs": len(measurable_recovered),
        "measurable_runs": len(measurable),
        "unmeasured_runs": len(unmeasured),
        "unmeasured_faults": sorted({r["fault"] for r in unmeasured}),
        # Loud, machine-checkable verdict for every downstream consumer.
        "measurement_valid": len(measurable) > 0,
        "measurement_caveat": (
            None
            if not unmeasured
            else (
                f"{len(unmeasured)} of {len(injected)} injected runs never reached "
                "the provider (all repair attempts failed at the llm stage — e.g. a "
                "rate limit or outage). Those runs are UNMEASURED, excluded from the "
                "recovery-rate denominator, and must not be read as failures to repair."
            )
        ),
        "first_attempt_recoveries": sum(
            1 for r in measurable_recovered if r["recovered_on_attempt"] == 1
        ),
        "mean_abs_score_fidelity": (
            round(sum(fidelity_vals) / len(fidelity_vals), 6) if fidelity_vals else None
        ),
        "max_abs_score_fidelity": round(max(fidelity_vals), 6) if fidelity_vals else None,
        "fidelity_scope": (
            "restorable faults only — for these the only sensible fix converges on "
            "the control's FE, so |delta| ~ 0 is a REGRESSION check (the repair did "
            "not wander), not independent proof of restoration. Faults with no "
            "restorable intent are listed under non_restorable_deltas, where a "
            "non-zero delta is the expected outcome."
        ),
        "n_fidelity_scored": len(fidelity_vals),
        "non_restorable_faults": sorted(NON_RESTORABLE_FAULTS),
        "non_restorable_deltas": non_restorable_observed,
        "false_positive_repairs_on_clean": sum(
            1 for r in controls if r["repair_attempted"]
        ),
        "fe_artifact_inconsistencies": sum(
            1 for r in recovered if r["fe_artifact_consistent"] is False
        ),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in runs),
        "total_completion_tokens": sum(r["completion_tokens"] for r in runs),
        "total_wall_s": round(sum(r["wall_s"] for r in runs), 1),
        "clean_scores": clean_scores,
        "runs": runs,
        "holdout_seal_intact": all(verify_holdout_untouched(k) for k in datasets),
    }
    return report


def render_table_md(report: dict[str, Any]) -> str:
    """The per-run table the day report embeds, as Markdown."""
    scripted = not report.get("is_measurement_of_llm_capability", True)
    rate_label = "mechanism recovery" if scripted else "recovery rate"
    lines = ["# Day 20 — Self-repair recovery study", ""]
    if scripted:
        lines += [
            "> **NOT AN LLM MEASUREMENT.** The repairer here is a deterministic, "
            "fault-blind stand-in, run because no live provider was reachable "
            "(Groq returned `organization_restricted` for every call on "
            "2026-07-25). These numbers measure the harness — does an injected "
            "fault detonate in the real Trainer, does the loop fire, is the "
            "sandboxed re-run adopted, is the clean score reproduced, does the "
            "persisted FE artifact stay consistent. They say **nothing** about "
            "whether a model can diagnose a traceback; that measurement is "
            "deferred until a provider is live.",
            "",
        ]
    if report.get("measurement_caveat"):
        lines += [f"> **PARTIALLY UNMEASURED.** {report['measurement_caveat']}", ""]
    rate = (
        f"**{report['recovered_runs']}/{report.get('measurable_runs', report['n_injected_runs'])}"
        f" = {report['recovery_rate']:.0%}**"
        if report.get("recovery_rate") is not None
        else "**not measurable** (no injected run reached the provider)"
    )
    lines += [
        f"Repairer: **{report['provider']}** ({report['model']}) · "
        f"attempt budget: {report['max_attempts']} · "
        f"{rate_label}: {rate} · "
        f"false-positive repairs on clean runs: "
        f"{report['false_positive_repairs_on_clean']} · "
        # Deliberately "malformed", not "inconsistent": the check parses the
        # artifact and looks for add_features. It does NOT prove the artifact
        # matches the FE the model was fitted with, and must not imply it.
        f"FE-artifact malformed: {report['fe_artifact_inconsistencies']} · "
        f"holdout seal intact: {report['holdout_seal_intact']}",
        "",
        "| Dataset | Fault | Recovered | Attempt | CV after | Δ vs clean | Tokens | Wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["runs"]:
        tokens = r["prompt_tokens"] + r["completion_tokens"]
        cv = "—" if r["cv_score"] is None else f"{r['cv_score']:.4f}"
        fid = (
            "—"
            if r["score_fidelity_vs_clean"] is None
            else f"{r['score_fidelity_vs_clean']:+.4f}"
        )
        if r["fault"] == "none_control":
            rec = "n/a (control)"
        elif r["recovered"]:
            rec = "yes"
        elif r.get("llm_unavailable"):
            rec = "unmeasured (provider)"
        else:
            rec = "NO"
        if fid != "—" and not r.get("restorable", True):
            fid += " ¹"      # expected to differ; see the footnote
        lines.append(
            f"| {r['dataset']} | {r['fault']} | {rec} | "
            f"{r['recovered_on_attempt'] or '—'} | {cv} | {fid} | {tokens} | {r['wall_s']} |"
        )
    lines.append("")
    if any(not r.get("restorable", True) for r in report["runs"]):
        lines += [
            "¹ This fault has **no restorable intent** (it references a column that "
            "does not exist, or a ratio that must be guarded differently), so a "
            "correct fix cannot reproduce the control's feature set. A non-zero Δ "
            "here is EXPECTED and is not evidence of a bad repair. Fidelity "
            "statistics are scoped to the restorable faults.",
            "",
        ]
    return "\n".join(lines)


def save_report(report: dict[str, Any]) -> None:
    SELF_REPAIR_RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    SELF_REPAIR_TABLE_MD_PATH.write_text(render_table_md(report), encoding="utf-8")
