"""The Feature Engineer agent — the crew's first *generated-code* node (Day 9).

The Feature Engineer is the first agent that writes code the crew then **runs**.
It reads the Planner's **ModelingPlan** (never the data directly) and produces a
small, self-contained Python source string defining::

    def add_features(df: pd.DataFrame) -> pd.DataFrame

that returns the feature frame with extra engineered columns appended. That source
is handed to the Day-6 sandboxed executor for a **smoke validation** on the train
split before it is trusted, and later to the Trainer (Day 9) which applies it ahead
of cross-validation.

Two honesty/robustness disciplines carry over from the Profiler and Planner:

* **Deterministic default, always.** :data:`DEFAULT_FE_SOURCE` is a competent,
  leakage-free, row-wise transform that always runs. It is the floor the agent
  falls back to, so the crew never stalls on a bad generation. When no live
  provider is configured (mock mode) the default *is* the feature engineering.
* **Generate-then-validate.** When a live provider is configured the agent asks
  it for dataset-specific ``add_features`` code, then **executes that code in the
  sandbox** on the train split. Only if it runs cleanly, preserves the row count
  and index, keeps every original column, and adds *numeric* columns is it
  trusted; otherwise the agent records the reason and — since Day 20 — hands the
  failed verdict to the :mod:`crewml.repair` loop for a bounded second chance
  before falling back to the default. An LLM never contributes an *unvalidated*
  line of code to a real run; repaired code passes the same gate as any other.

**The row-wise contract (why applying FE before CV is leakage-free).** Engineered
features must be a function of a single row's own values — no fitting, no target,
no cross-row aggregation. Under that contract, computing features once on the whole
train frame is identical to computing them fold-by-fold, so the Trainer can apply
FE up front without leaking across CV folds (EVAL_PROTOCOL §3). The validation
rejects any source that references the target column; the contract is stated in the
prompt and enforced structurally where it can be.

**Train only, structurally.** This module reasons over the plan dict and validates
against the *train* split alone (via the executor, which is handed only
``train.parquet``). It never names the held-out loader — a source-inspection test
asserts it — so the no-peeking invariant is a property of the code.
"""
from __future__ import annotations

import os
import textwrap
from typing import Any, Optional

from crewml import config, llm
from crewml.datasets import train_path
from crewml.executor import run_code
from crewml.repair import repair_enabled_for_fe, repair_loop

FE_SCHEMA_VERSION = 2  # v2 (Day 20): + "llm_repaired" source and "repair" provenance

# The engineered-column name the deterministic default contributes. Kept as a
# constant so the report and tests can name it without magic strings.
DEFAULT_FE_FEATURE = "row_nan_count"


# --- The deterministic default: safe, row-wise, always runs -----------------

DEFAULT_FE_SOURCE = textwrap.dedent(
    f'''\
    """Deterministic default feature engineering (no LLM).

    Row-wise and leakage-free: adds a single count of missing values per row.
    This is a genuine, generic tabular feature (rows with more missing fields
    often behave differently) that never touches the target, never fits, and
    never aggregates across rows — so applying it before cross-validation cannot
    leak information across folds.
    """
    import pandas as pd


    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["{DEFAULT_FE_FEATURE}"] = df.isna().sum(axis=1).astype("int64")
        return out
    '''
)


# --- Ablation stand-in (Day 14) — the FE a crew with NO Feature Engineer runs

IDENTITY_FE_SOURCE = textwrap.dedent(
    '''\
    """Identity feature engineering — the Day-14 ``no_feature_engineer`` ablation.

    Adds nothing: the model trains on the raw feature frame exactly as loaded.
    This is the honest meaning of "remove the Feature Engineer" — the Trainer's
    contract still requires an ``add_features`` callable, so removal means the
    identity transform, not a competing feature set.
    """
    import pandas as pd


    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        return df
    '''
)


def run_identity_fe(dataset_key: str) -> dict[str, Any]:
    """The ``no_feature_engineer`` variant's FE step: identity code, no LLM (Day 14).

    Mirrors :func:`run_feature_engineer`'s return shape (``{"code", "meta"}``) so the
    Trainer and Reporter consume it unchanged. The identity source is still put
    through the same sandbox validation as any other FE code — not because it could
    fail the contract, but so every arm of the ablation carries the same evidence
    trail (``meta.validation``) and no arm is trusted on argument alone.
    """
    return {
        "code": IDENTITY_FE_SOURCE,
        "meta": {
            "schema_version": FE_SCHEMA_VERSION,
            "node": "feature_engineer_identity",
            "ablated": "feature_engineer",
            "dataset_key": dataset_key,
            "is_mock": config.is_mock_mode(),
            "source": "ablated",
            "reason": "no_feature_engineer variant — identity transform, no generation",
            "validation": _validate_fe(IDENTITY_FE_SOURCE, dataset_key),
        },
    }


# --- LLM prompt for dataset-specific feature engineering --------------------

_FE_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are the Feature Engineer in a multi-agent ML crew. You are given a
    deterministic modeling plan (columns, their dtypes, the task and metric) and
    must write ONE self-contained Python module defining exactly:

        def add_features(df):

    Hard requirements — the code is executed and validated automatically:
    - `df` is a pandas DataFrame of FEATURES ONLY (the target column is NOT
      present). Return a NEW DataFrame that keeps every original column and its
      row order/index, with extra engineered columns appended.
    - Engineered columns MUST be ROW-WISE and STATELESS: functions of a single
      row's own values only. No fitting, no statistics computed across rows
      (no means/std/target-encoding/group aggregates), no randomness, no I/O,
      no network. This keeps the features leakage-free under cross-validation.
    - There is NO target column to read; never reference a column named 'target'.
    - Every new column must be NUMERIC. Guard against divide-by-zero and missing
      values so the transform never raises (use safe denominators, fillna where
      needed). Keep it to a handful of well-motivated features.
    - Use only pandas and numpy. Output ONLY the Python module in one ```python
      code block. No prose.
    """
)


# Which Critic findings are the Feature Engineer's business. `leak` is the one that
# most needs FE's attention — the Planner can only re-audit column drops, while an
# engineered feature derived from the target can only be removed by whoever writes
# the FE code. `overfit` matters because gratuitous features widen fold variance.
_FE_RELEVANT_CODES: frozenset[str] = frozenset({"leakage", "overfit", "wrong_metric"})


def _critique_directives(critique: Optional[dict[str, Any]]) -> list[str]:
    """The FE-addressed directives from a critique, newest pass only.

    Reads ``diagnoses`` — the Critic's *structured* findings. Note that
    ``critique["findings"]`` is a list of pre-rendered **strings** for the Planner,
    not dicts; reading that key instead raises ``AttributeError`` on a real looped
    run. Tolerates either shape so a caller passing the structured list directly
    (as tests do) also works.
    """
    if not critique:
        return []
    entries = critique.get("diagnoses") or critique.get("findings") or []
    return [
        d["directive"]
        for d in entries
        if isinstance(d, dict)
        and d.get("code") in _FE_RELEVANT_CODES
        and d.get("directive")
    ]


def _fe_user_prompt(plan: dict[str, Any], critique: Optional[dict[str, Any]] = None) -> str:
    pre = plan["preprocessing"]
    base = textwrap.dedent(
        f"""\
        Dataset: {plan['dataset_key']} — task {plan['task']} ({plan['subtype']}),
        optimising {plan['metric']}.
        Numeric feature columns available: {pre['numeric']['columns']}
        Categorical feature columns available: {pre['categorical']['columns']}
        Columns the plan will DROP (do not rely on them): {plan['drop_columns']}

        Write `add_features(df)` now, adding a few leakage-free, row-wise numeric
        features you judge useful for maximising {plan['metric']} on unseen data of
        this shape. Return only the code.
        """
    )
    # Day 10 promised the Critic's instructions feed back to "Planner/FE", but only
    # the Planner was ever wired up: on a looped pass the FE regenerated from the
    # plan alone and could re-introduce the very feature the Critic objected to.
    directives = _critique_directives(critique)
    if not directives:
        return base
    joined = "\n".join(f"- {d}" for d in directives)
    return base + textwrap.dedent(
        f"""
        A previous pass was CRITIQUED. Address these findings in the features you
        write this time — do not re-introduce what they object to:
        {joined}
        """
    )


# --- Sandbox smoke validation (the trust gate for generated code) -----------

_VALIDATION_HARNESS = textwrap.dedent(
    '''\
    """Trusted validation harness — NOT agent code. Applies the candidate
    add_features() to the TRAIN feature frame and reports whether it honours the
    contract. Never scores anything; never touches the held-out split.
    """
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    from crew_io import emit_metrics, input_path

    TARGET = "target"

    df = pd.read_parquet(input_path("train.parquet"))
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    original = list(df.columns)
    n_in = len(df)

    # --- candidate code is spliced in below by the agent (defines add_features) ---
    #<<FE_SOURCE>>

    out = add_features(df.copy())

    if not isinstance(out, pd.DataFrame):
        raise TypeError("add_features must return a pandas DataFrame")

    new_cols = [c for c in out.columns if c not in original]
    original_preserved = all(c in out.columns for c in original)
    rows_preserved = (len(out) == n_in) and out.index.equals(df.index)
    all_numeric = all(is_numeric_dtype(out[c]) for c in new_cols)
    all_new_have_values = all(out[c].notna().any() or len(out) == 0 for c in new_cols)


    def _has_inf(col):
        """True if the column holds +/-inf. NaN is fine; infinity is not."""
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype="float64")
        return bool(np.isinf(values).any())

    # Infinity is the one non-finite value that survives the whole preprocessing
    # chain and then kills the fit: SimpleImputer runs with
    # force_all_finite="allow-nan", so it replaces NaN but passes inf straight
    # through to sklearn's finite assertion. A live run lost a whole dataset to
    # exactly this (an unguarded ratio in generated FE code), because the old
    # check here was a nan check misleadingly named `all_finite_ok`. Rejecting inf
    # at the validation gate means such code never reaches the Trainer at all.
    no_infinities = not any(_has_inf(c) for c in new_cols)

    emit_metrics(
        ok=bool(original_preserved and rows_preserved and all_numeric and no_infinities),
        n_rows_in=int(n_in),
        n_rows_out=int(len(out)),
        new_columns=list(new_cols),
        n_new=len(new_cols),
        original_preserved=bool(original_preserved),
        rows_preserved=bool(rows_preserved),
        all_numeric=bool(all_numeric),
        all_new_have_values=bool(all_new_have_values),
        no_infinities=bool(no_infinities),
        infinite_columns=[c for c in new_cols if _has_inf(c)],
    )
    print("FE_VALIDATION_OK", flush=True)
    '''
)


def _validate_fe(fe_source: str, dataset_key: str, *, timeout_s: int = 60) -> dict[str, Any]:
    """Execute ``fe_source`` on the train split in the sandbox and report the verdict.

    Returns a JSON-friendly dict: ``ok`` plus the observed row/column facts (or the
    executor error on a crash). Never raises for *code* failures — a broken
    ``add_features`` is reported as ``ok: False``, not an exception.
    """
    # The marker sits at module level (0 indent) after dedent — splice as-is.
    harness = _VALIDATION_HARNESS.replace("#<<FE_SOURCE>>", fe_source)
    result = run_code(
        harness,
        inputs={"train.parquet": train_path(dataset_key)},
        timeout_s=timeout_s,
        keep_workdir=False,
    )
    metrics = result.metrics or {}
    verdict: dict[str, Any] = {
        "ok": bool(result.ok and metrics.get("ok") is True),
        "executed_ok": bool(result.ok),
        "duration_s": result.duration_s,
    }
    if result.ok:
        verdict.update(
            {k: metrics.get(k) for k in (
                "n_rows_in", "n_rows_out", "new_columns", "n_new",
                "original_preserved", "rows_preserved", "all_numeric",
                "all_new_have_values", "no_infinities", "infinite_columns",
            )}
        )
    else:
        verdict["error"] = result.error or "execution failed"
    return verdict


# --- Self-repair support (Day 20) -------------------------------------------

def _verdict_error(verdict: dict[str, Any]) -> str:
    """Render a failed validation verdict as the 'traceback' the repair loop shows."""
    if not verdict.get("executed_ok", False):
        return verdict.get("error") or "add_features crashed during sandbox validation"
    broken = [
        check
        for check in (
            "original_preserved", "rows_preserved", "all_numeric",
            "all_new_have_values", "no_infinities",
        )
        if verdict.get(check) is False
    ]
    detail = ""
    if verdict.get("no_infinities") is False:
        bad = verdict.get("infinite_columns") or []
        detail = (
            f" The column(s) {bad} contain +/-inf — guard every division and log "
            "with a safe denominator; NaN is acceptable but infinity is not."
        )
    return (
        "add_features ran but broke the contract: "
        + (", ".join(f"{c}=False" for c in broken) or "validation reported not-ok")
        + ". Every original column and the row order/index must be preserved and "
        "every engineered column must be numeric, finite, and have at least one "
        "non-null value." + detail
    )


_FE_REPAIR_CONTEXT = (
    "The module defines add_features(df) for a multi-agent ML crew's Feature "
    "Engineer. Contract: df is a pandas DataFrame of features only (no 'target' "
    "column exists); return a NEW DataFrame keeping every original column and "
    "the row order/index, with extra engineered columns appended. Engineered "
    "columns must be numeric, row-wise and stateless (no fitting, no cross-row "
    "statistics, no randomness, no I/O), and must never raise on missing values "
    "or zero denominators."
)


def _fe_repair_run_fn(dataset_key: str):
    """Adapt sandbox validation to the repair loop's (ok, error, payload) shape."""

    def run(source: str):
        verdict = _validate_fe(source, dataset_key)
        return verdict["ok"], (None if verdict["ok"] else _verdict_error(verdict)), verdict

    return run


# --- LLM generation (optional, always validated before trust) ---------------

def _llm_enabled(with_llm: Optional[bool]) -> bool:
    """Explicit flag wins; else the ``CREWML_FE_LLM`` env toggle (default on)."""
    if with_llm is not None:
        return with_llm
    return os.getenv("CREWML_FE_LLM", "1") != "0"


def _generate_llm_fe(
    plan: dict[str, Any], critique: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Ask the live provider for dataset-specific ``add_features`` code. Never raises."""
    try:
        result = llm.chat(
            _FE_SYSTEM_PROMPT,
            _fe_user_prompt(plan, critique),
            temperature=0.0,
            max_tokens=1024,
            agent="feature_engineer",
        )
        return {
            "ok": True,
            "code": llm.extract_python(result.text),
            "provider": result.provider,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }
    except Exception as exc:  # provider down / restricted / parse — degrade
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


# --- Public entry point -----------------------------------------------------

def run_feature_engineer(
    plan: dict[str, Any],
    dataset_key: str,
    *,
    with_llm: Optional[bool] = None,
    self_repair: Optional[bool] = None,
    critique: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Produce validated feature-engineering code for ``dataset_key`` from the plan.

    Returns ``{"code": <source>, "meta": <provenance>}``. ``code`` is always a
    runnable ``add_features`` module — the LLM's if it was generated *and* passed
    sandbox validation, otherwise the deterministic default. ``meta.source`` is one
    of ``"llm"``, ``"llm_repaired"`` (Day 20 — the generation failed validation but
    the self-repair loop produced a passing fix, trail under ``meta.repair``),
    ``"default"`` (mock mode / LLM disabled), or ``"fallback"`` (LLM tried, and any
    repair attempts failed too), and carries the validation verdict and token
    accounting so a report can see exactly what happened. Repaired code earns
    exactly the same trust as first-try code — a full sandbox validation pass —
    and the fallback ladder beneath it is unchanged.
    """
    meta: dict[str, Any] = {
        "schema_version": FE_SCHEMA_VERSION,
        "node": "feature_engineer",
        "dataset_key": dataset_key,
        "is_mock": config.is_mock_mode(),
        # Day 10 promised critiques feed back to the Planner AND the FE; only the
        # Planner was wired. Record whether this pass actually saw one.
        "critique_directives": _critique_directives(critique),
    }

    use_llm = _llm_enabled(with_llm) and not config.is_mock_mode()

    if use_llm:
        gen = _generate_llm_fe(plan, critique)
        if gen["ok"]:
            verdict = _validate_fe(gen["code"], dataset_key)
            if verdict["ok"]:
                meta.update(
                    source="llm", provider=gen["provider"], model=gen["model"],
                    prompt_tokens=gen["prompt_tokens"],
                    completion_tokens=gen["completion_tokens"],
                    validation=verdict,
                )
                return {"code": gen["code"], "meta": meta}
            # Generated but failed the contract — before falling back, give the
            # self-repair loop (Day 20) a shot: the failed verdict becomes the
            # "traceback", and any fix must pass the *same* sandbox validation.
            if repair_enabled_for_fe(self_repair):
                repair = repair_loop(
                    gen["code"],
                    _verdict_error(verdict),
                    run_fn=_fe_repair_run_fn(dataset_key),
                    context=_FE_REPAIR_CONTEXT,
                )
                repair_trail = {
                    k: v for k, v in repair.items() if k not in ("code", "payload")
                }
                if repair["recovered"]:
                    meta.update(
                        source="llm_repaired", provider=gen["provider"], model=gen["model"],
                        prompt_tokens=gen["prompt_tokens"],
                        completion_tokens=gen["completion_tokens"],
                        validation=repair["payload"],
                        llm_validation=verdict,
                        repair=repair_trail,
                    )
                    return {"code": repair["code"], "meta": meta}
                meta["repair"] = repair_trail
            # Record why and fall back. Keep the failed verdict under its own key
            # so the default's verdict (below) can still populate `validation`
            # for the code actually used.
            meta.update(
                source="fallback", provider=gen.get("provider"), model=gen.get("model"),
                fallback_reason="generated code failed sandbox validation",
                llm_validation=verdict,
            )
        else:
            meta.update(source="fallback", fallback_reason=gen["reason"])
    else:
        meta["source"] = "default"
        meta["reason"] = "mock_mode" if config.is_mock_mode() else "disabled"

    # Deterministic default path (also validated, so a report can trust it ran).
    meta["validation"] = _validate_fe(DEFAULT_FE_SOURCE, dataset_key)
    meta["default_feature"] = DEFAULT_FE_FEATURE
    return {"code": DEFAULT_FE_SOURCE, "meta": meta}
