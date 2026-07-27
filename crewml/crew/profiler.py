"""The Profiler agent — the crew's first *real* node (Day 7).

The Profiler turns the raw ``train`` split into a structured, JSON-friendly
**DataProfile**: schema and dtypes, missingness (including *disguised* missing —
implausible zeros the diabetes set hides), the target's distribution and class
imbalance, and a set of **basic leakage checks**. Everything the Planner (Day 8)
reasons over starts here, so the facts must be *computed*, never guessed:

* **Deterministic core.** :func:`build_profile` derives every number with plain
  pandas/numpy on the training frame alone. No LLM touches a statistic, so the
  profile is reproducible and cannot hallucinate (EVAL_PROTOCOL §5 honesty).
* **Rule-based assessment.** A small deterministic synthesis turns the raw facts
  into flags + notes (``class_imbalance``, ``disguised_missing_suspected``,
  ``target_leakage_suspected`` …) — the agent's *read* of the data, still fully
  reproducible.
* **Optional LLM narrative.** When a live provider is configured, :func:`run_profiler`
  layers a short briefing for the Planner *on top of* — never in place of — the
  deterministic facts, tagged with its provider/model/token cost. In mock mode (or
  on any error) the narrative is marked ``unavailable`` and the profile stands on
  its deterministic core alone. The narrative is advisory; it never overwrites a
  computed value.

**Train only, structurally.** The Profiler loads exactly one split —
:func:`crewml.datasets.load_train` — and nothing here can reach the locked test
split (a source-inspection test asserts the module never names it). The no-peeking
invariant is a property of the code, not a rule a node has to remember.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from crewml import config, llm
from crewml.datasets import REGISTRY, TARGET_COLUMN, DatasetSpec, load_train
from crewml.leakage import screen_features

PROFILE_SCHEMA_VERSION = 1

# --- Heuristic thresholds (deliberately conservative so clean data stays clean) ---
IMBALANCE_WARN_RATIO = 1.5     # majority/minority count ratio worth flagging
HIGH_CARDINALITY_FRAC = 0.5    # categorical n_unique / n_rows above this is "high"
ID_LIKE_UNIQUE_FRAC = 0.99     # near-per-row-unique int/categorical => identifier-like
DISGUISED_MISSING_ZERO_FRAC = 0.05  # numeric zero share worth flagging as maybe-missing
LEAKAGE_PEARSON = 0.98         # |corr(feature, target)| above this => leakage suspect (regression)
LEAKAGE_PURITY = 0.995         # per-group target purity above this => leakage suspect (classification)
LEAKAGE_PURITY_LIFT = 0.30     # ...and it must beat the majority-class rate by this much
NUMERIC_BIN_UNIQUE = 20        # numeric features with more uniques get binned for the purity test
N_PURITY_BINS = 10


# --- Small JSON-safety helpers (numpy scalars are not JSON-serialisable) ----

def _f(x: Any) -> Optional[float]:
    """Round to a float, or None if not finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if np.isfinite(v) else None


def _i(x: Any) -> int:
    return int(x)


def _numeric_cols(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["number"]).columns.tolist()


# --- Per-feature facts ------------------------------------------------------

def _feature_facts(X: pd.DataFrame, numeric: set[str]) -> dict[str, dict[str, Any]]:
    """Compact per-column facts: dtype, missingness, cardinality, and stats."""
    n = len(X)
    out: dict[str, dict[str, Any]] = {}
    for col in X.columns:
        s = X[col]
        n_missing = _i(s.isna().sum())
        n_unique = _i(s.nunique(dropna=True))
        fact: dict[str, Any] = {
            "dtype": str(s.dtype),
            "kind": "numeric" if col in numeric else "categorical",
            "n_missing": n_missing,
            "missing_frac": _f(n_missing / n) if n else 0.0,
            "n_unique": n_unique,
            "unique_frac": _f(n_unique / n) if n else 0.0,
        }
        if col in numeric:
            nonnull = s.dropna()
            fact.update(
                min=_f(nonnull.min()) if len(nonnull) else None,
                max=_f(nonnull.max()) if len(nonnull) else None,
                mean=_f(nonnull.mean()) if len(nonnull) else None,
                std=_f(nonnull.std()) if len(nonnull) else None,
                zero_frac=_f((s == 0).sum() / n) if n else 0.0,
            )
        else:
            vc = s.value_counts(dropna=True)
            if len(vc):
                fact["top"] = str(vc.index[0])
                fact["top_frac"] = _f(vc.iloc[0] / n)
        out[col] = fact
    return out


# --- Target facts -----------------------------------------------------------

def _target_facts(spec: DatasetSpec, y: pd.Series) -> dict[str, Any]:
    """Distribution of the target: class counts + imbalance, or numeric summary."""
    facts: dict[str, Any] = {"name": TARGET_COLUMN, "dtype": str(y.dtype)}
    if spec.task == "classification":
        counts = y.value_counts()
        counts = counts[counts > 0]
        classes = {str(k): _i(v) for k, v in counts.items()}
        majority = max(classes, key=classes.get)
        minority = min(classes, key=classes.get)
        facts.update(
            classes=classes,
            n_classes=len(classes),
            majority_class=majority,
            minority_class=minority,
            # The rarer class is the positive/scored class for binary AUC — computed
            # train-only here, matching crewml.scoring's convention exactly.
            positive_class=minority,
            imbalance_ratio=_f(classes[majority] / classes[minority]),
        )
    else:
        facts.update(
            min=_f(y.min()), max=_f(y.max()), mean=_f(y.mean()),
            std=_f(y.std()), skew=_f(y.skew()),
        )
    return facts


# --- Leakage / integrity checks --------------------------------------------

def _duplicate_feature_columns(X: pd.DataFrame) -> list[list[str]]:
    """Groups of columns that are byte-identical (NaN-aware) — redundant or leaky."""
    sig_to_cols: dict[tuple, list[str]] = {}
    for col in X.columns:
        s = X[col]
        sig = tuple(s.where(s.notna(), "__NaN__").astype(str).tolist())
        sig_to_cols.setdefault(sig, []).append(col)
    return [cols for cols in sig_to_cols.values() if len(cols) > 1]


def _classification_purity(feature: pd.Series, y: pd.Series) -> Optional[float]:
    """Weighted mean target purity after grouping by the feature (binned if wide).

    Purity ~1.0 means the feature alone almost perfectly determines the target —
    the classic fingerprint of a leaked column. Returns None if it can't be
    computed (e.g. all-missing feature).
    """
    s = feature
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > NUMERIC_BIN_UNIQUE:
        # Rank-then-qcut gives deterministic, tie-stable deciles.
        s = pd.qcut(s.rank(method="first"), q=N_PURITY_BINS, labels=False, duplicates="drop")
    df = pd.DataFrame({"g": s, "y": y}).dropna(subset=["g"])
    if df.empty:
        return None
    correct = 0
    for _, grp in df.groupby("g", observed=True):
        vc = grp["y"].value_counts()
        if len(vc):
            correct += int(vc.iloc[0])
    return correct / len(df)


def _leakage_checks(
    spec: DatasetSpec,
    X: pd.DataFrame,
    y: pd.Series,
    numeric: set[str],
    facts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Basic, deterministic leakage & integrity signals — advisory, high-precision."""
    constant = [c for c in X.columns if facts[c]["n_unique"] <= 1]

    # Identifier-like: near-unique, but only for int/categorical columns — continuous
    # floats are expected to be unique and are NOT identifiers (would false-positive).
    id_like = [
        c for c in X.columns
        if facts[c]["unique_frac"] is not None
        and facts[c]["unique_frac"] >= ID_LIKE_UNIQUE_FRAC
        and not pd.api.types.is_float_dtype(X[c])
    ]

    dup_cols = _duplicate_feature_columns(X)
    dup_rows = _i(X.duplicated().sum())

    # Numeric zero-inflation => *suspected* disguised missing (zeros may be legit;
    # this is a candidate list for the Planner/LLM to adjudicate, not a verdict).
    suspected_missing = [
        {"column": c, "zero_frac": facts[c]["zero_frac"]}
        for c in numeric
        if facts[c].get("zero_frac") and facts[c]["zero_frac"] >= DISGUISED_MISSING_ZERO_FRAC
        and c not in constant
    ]

    # Near-perfect single-feature predictors of the target => leakage suspects.
    # Two layers (Day 22): the fast Phase-2 screens (Pearson / group purity)
    # catch the blatant copies, then the calibrated single-feature CV screen
    # (crewml.leakage) catches the "implausibly strong but not perfect" band
    # Day 17 measured slipping through — a 95%-agreement leak scores ~0.95
    # standalone AUC, far above any legitimate feature in the locked suite.
    target_leaks: list[dict[str, Any]] = []
    if spec.task == "regression":
        for c in numeric:
            xi = X[c]
            mask = xi.notna() & y.notna()
            if mask.sum() < 3 or xi[mask].std() == 0:
                continue
            r = np.corrcoef(xi[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))[0, 1]
            if np.isfinite(r) and abs(r) >= LEAKAGE_PEARSON:
                target_leaks.append({"column": c, "measure": "pearson", "signal": _f(abs(r))})
    else:
        counts = y.value_counts()
        majority_rate = counts.iloc[0] / len(y) if len(counts) else 1.0
        for c in X.columns:
            if c in id_like or c in constant:
                continue  # a unique id trivially "predicts" everything — not informative
            purity = _classification_purity(X[c], y)
            if purity is not None and purity >= LEAKAGE_PURITY and (purity - majority_rate) >= LEAKAGE_PURITY_LIFT:
                target_leaks.append({"column": c, "measure": "target_purity", "signal": _f(purity)})

    # The Day-22 layer: calibrated standalone-CV screen over everything the fast
    # screens didn't already flag (ids/constants excluded — a unique id scores at
    # chance out-of-fold anyway, and a constant can't split).
    already = {d["column"] for d in target_leaks} | set(id_like) | set(constant)
    target_leaks.extend(
        screen_features(X, y, spec.task, spec.metric, skip=frozenset(already))
    )

    return {
        "constant_columns": constant,
        "id_like_columns": id_like,
        "duplicate_feature_columns": dup_cols,
        "duplicate_rows": dup_rows,
        "suspected_disguised_missing": suspected_missing,
        "target_correlated_features": target_leaks,
    }


# --- Deterministic assessment (the agent's read, no LLM) --------------------

def _assessment(spec: DatasetSpec, profile: dict[str, Any]) -> dict[str, Any]:
    """Turn the raw facts into flags + human-readable notes — fully reproducible."""
    flags: list[str] = []
    notes: list[str] = []
    tgt = profile["target"]
    miss = profile["missingness"]
    leak = profile["leakage_checks"]
    feats = profile["features"]

    if spec.task == "classification":
        ratio = tgt.get("imbalance_ratio")
        if ratio and ratio >= IMBALANCE_WARN_RATIO:
            flags.append("class_imbalance")
            notes.append(
                f"Target is imbalanced ({tgt['majority_class']}:{tgt['minority_class']} "
                f"~ {ratio:.2f}:1). Prefer stratified CV and a threshold-independent "
                f"metric; the rarer class '{tgt['minority_class']}' is the positive class."
            )

    if miss["any_missing"]:
        flags.append("explicit_missing")
        notes.append(
            f"{miss['n_columns_with_missing']} column(s) carry explicit NaNs "
            f"({miss['total_missing_cells']} cells) — impute inside the pipeline."
        )
    if leak["suspected_disguised_missing"]:
        flags.append("disguised_missing_suspected")
        cols = ", ".join(d["column"] for d in leak["suspected_disguised_missing"])
        notes.append(
            f"Zero-inflated numeric column(s) [{cols}] may encode missing values as 0 "
            f"(heuristic — zeros can be legitimate; verify before imputing)."
        )

    hi_card = [
        c for c, f in feats.items()
        if f["kind"] == "categorical" and (f["unique_frac"] or 0) >= HIGH_CARDINALITY_FRAC
    ]
    if hi_card:
        flags.append("high_cardinality_categoricals")
        notes.append(f"High-cardinality categoricals {hi_card} — one-hot may explode; consider target/ordinal encoding.")

    numeric = set(profile["columns"]["numeric"])
    if numeric and set(profile["columns"]["categorical"]):
        flags.append("mixed_dtypes")
        notes.append("Mixed numeric + categorical features — use a ColumnTransformer so preprocessing is dtype-aware.")

    for key, flag, msg in (
        ("constant_columns", "constant_columns", "constant (zero-variance) column(s)"),
        ("id_like_columns", "id_like_columns", "identifier-like near-unique column(s)"),
        ("duplicate_feature_columns", "duplicate_columns", "duplicate feature column group(s)"),
        ("target_correlated_features", "target_leakage_suspected", "feature(s) that near-perfectly predict the target"),
    ):
        if leak[key]:
            flags.append(flag)
            notes.append(f"Leakage check: {leak[key]!r} — {msg}; drop or investigate before training.")
    if leak["duplicate_rows"]:
        flags.append("duplicate_rows")
        notes.append(f"{leak['duplicate_rows']} duplicate feature-row(s) present.")

    if not any(f in flags for f in ("constant_columns", "id_like_columns", "duplicate_columns", "target_leakage_suspected")):
        notes.append("No hard leakage signals in the basic checks (clean on constant/id/duplicate/target-purity).")

    return {"source": "deterministic", "flags": flags, "notes": notes}


# --- Public: build the deterministic profile --------------------------------

def build_profile(spec: DatasetSpec, train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute the full deterministic DataProfile from the training frame alone.

    Pure and reproducible: no I/O, no network, no LLM. The returned dict is
    JSON-serialisable so it can live in :class:`~crewml.crew.state.CrewState`.
    """
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]
    numeric = set(_numeric_cols(X))
    categorical = [c for c in X.columns if c not in numeric]

    facts = _feature_facts(X, numeric)
    target = _target_facts(spec, y)

    cols_with_missing = {c: facts[c]["missing_frac"] for c in X.columns if facts[c]["n_missing"] > 0}
    missingness = {
        "any_missing": bool(cols_with_missing),
        "n_columns_with_missing": len(cols_with_missing),
        "total_missing_cells": _i(X.isna().sum().sum()),
        "columns_with_missing": cols_with_missing,
    }

    profile: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "stub": False,
        "node": "profiler",
        "dataset_key": spec.key,
        "task": spec.task,
        "subtype": spec.subtype,
        "metric": spec.metric,
        "n_rows": _i(len(train_df)),
        "n_features": _i(X.shape[1]),
        "columns": {"numeric": sorted(numeric), "categorical": sorted(categorical)},
        "features": facts,
        "target": target,
        "missingness": missingness,
    }
    profile["leakage_checks"] = _leakage_checks(spec, X, y, numeric, facts)
    profile["assessment"] = _assessment(spec, profile)
    return profile


# --- Optional LLM narrative (advisory, never a source of facts) -------------

_NARRATIVE_SYSTEM = (
    "You are the Profiler agent in a multi-agent ML crew. You receive a "
    "DETERMINISTIC data profile computed from the TRAINING split only. Do NOT "
    "invent or restate raw numbers as new facts — interpret them. In <=180 words, "
    "brief the Planner agent on the 3-5 most important modeling implications: how "
    "to handle any class imbalance, explicit or disguised-missing values, leakage "
    "risks to avoid, and metric-specific concerns for the stated metric. Be "
    "concrete and specific to THIS dataset. Plain prose, no code."
)


def _narrative_payload(profile: dict[str, Any]) -> dict[str, Any]:
    """A compact, token-light view of the profile for the LLM prompt."""
    return {
        "dataset_key": profile["dataset_key"],
        "task": profile["task"],
        "subtype": profile["subtype"],
        "metric": profile["metric"],
        "n_rows": profile["n_rows"],
        "n_features": profile["n_features"],
        "columns": {k: len(v) for k, v in profile["columns"].items()},
        "target": profile["target"],
        "missingness": {k: v for k, v in profile["missingness"].items() if k != "columns_with_missing"} | {
            "columns_with_missing": list(profile["missingness"]["columns_with_missing"].keys())
        },
        "leakage_checks": profile["leakage_checks"],
        "assessment_flags": profile["assessment"]["flags"],
    }


def _llm_narrative(profile: dict[str, Any]) -> dict[str, Any]:
    """Ask the live provider for a short Planner briefing. Never raises."""
    import json

    try:
        result = llm.chat(
            _NARRATIVE_SYSTEM,
            "Data profile (JSON):\n" + json.dumps(_narrative_payload(profile), default=str),
            temperature=0.0,
            max_tokens=512,
            agent="profiler",
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
    """Explicit flag wins; else the ``CREWML_PROFILER_LLM`` env toggle (default on)."""
    if with_llm is not None:
        return with_llm
    return os.getenv("CREWML_PROFILER_LLM", "1") != "0"


def run_profiler(dataset_key: str, *, with_llm: Optional[bool] = None) -> dict[str, Any]:
    """Load the training split for ``dataset_key`` and produce its DataProfile.

    The deterministic profile is always computed. An LLM narrative is attached
    only when enabled *and* a live provider is configured; otherwise the profile
    records the narrative as ``unavailable`` and stands on its deterministic core.
    """
    spec = REGISTRY[dataset_key]
    profile = build_profile(spec, load_train(dataset_key))

    if _llm_enabled(with_llm) and not config.is_mock_mode():
        profile["llm_narrative"] = _llm_narrative(profile)
    else:
        reason = "mock_mode" if config.is_mock_mode() else "disabled"
        profile["llm_narrative"] = {
            "source": "unavailable", "is_mock": config.is_mock_mode(),
            "reason": reason, "text": None,
        }
    return profile
