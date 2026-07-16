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
  trusted; otherwise the agent records the reason and falls back to the default.
  An LLM never contributes an *unvalidated* line of code to a real run.

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

FE_SCHEMA_VERSION = 1

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


def _fe_user_prompt(plan: dict[str, Any]) -> str:
    pre = plan["preprocessing"]
    return textwrap.dedent(
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


# --- Sandbox smoke validation (the trust gate for generated code) -----------

_VALIDATION_HARNESS = textwrap.dedent(
    '''\
    """Trusted validation harness — NOT agent code. Applies the candidate
    add_features() to the TRAIN feature frame and reports whether it honours the
    contract. Never scores anything; never touches the held-out split.
    """
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
    all_finite_ok = all(out[c].notna().any() or len(out) == 0 for c in new_cols)

    emit_metrics(
        ok=bool(original_preserved and rows_preserved and all_numeric),
        n_rows_in=int(n_in),
        n_rows_out=int(len(out)),
        new_columns=list(new_cols),
        n_new=len(new_cols),
        original_preserved=bool(original_preserved),
        rows_preserved=bool(rows_preserved),
        all_numeric=bool(all_numeric),
        all_new_have_values=bool(all_finite_ok),
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
                "all_new_have_values",
            )}
        )
    else:
        verdict["error"] = result.error or "execution failed"
    return verdict


# --- LLM generation (optional, always validated before trust) ---------------

def _llm_enabled(with_llm: Optional[bool]) -> bool:
    """Explicit flag wins; else the ``CREWML_FE_LLM`` env toggle (default on)."""
    if with_llm is not None:
        return with_llm
    return os.getenv("CREWML_FE_LLM", "1") != "0"


def _generate_llm_fe(plan: dict[str, Any]) -> dict[str, Any]:
    """Ask the live provider for dataset-specific ``add_features`` code. Never raises."""
    try:
        result = llm.chat(
            _FE_SYSTEM_PROMPT,
            _fe_user_prompt(plan),
            temperature=0.0,
            max_tokens=1024,
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
) -> dict[str, Any]:
    """Produce validated feature-engineering code for ``dataset_key`` from the plan.

    Returns ``{"code": <source>, "meta": <provenance>}``. ``code`` is always a
    runnable ``add_features`` module — the LLM's if it was generated *and* passed
    sandbox validation, otherwise the deterministic default. ``meta.source`` is one
    of ``"llm"``, ``"default"`` (mock mode / LLM disabled), or ``"fallback"`` (LLM
    tried but its code failed validation), and carries the validation verdict and
    any token accounting so a report can see exactly what happened.
    """
    meta: dict[str, Any] = {
        "schema_version": FE_SCHEMA_VERSION,
        "node": "feature_engineer",
        "dataset_key": dataset_key,
        "is_mock": config.is_mock_mode(),
    }

    use_llm = _llm_enabled(with_llm) and not config.is_mock_mode()

    if use_llm:
        gen = _generate_llm_fe(plan)
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
            # Generated but failed the contract — record why and fall back. Keep the
            # failed verdict under its own key so the default's verdict (below) can
            # still populate `validation` for the code actually used.
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
