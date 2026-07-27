"""The single-feature predictive-power leakage screen (Day 22).

Day 17 measured the detection window the Phase-2 screens leave open: an injected
column agreeing with the target on 95% of rows sailed under the Profiler's purity
screen (``LEAKAGE_PURITY = 0.995``) *and* under the Critic's CV ceiling
(``0.995``), and the model trained on it (CV roc_auc 0.964 — inflated, plausible,
wrong). The blind spot is structural: purity asks "does this column *alone,
perfectly* determine the target?", while a real leak only has to be *implausibly
strong*, not perfect.

This module closes most of that window by asking the model's own question: **how
well does each feature predict the target on rows it hasn't seen?** Each column
gets a depth-3 decision tree, cross-validated 3-fold on the train split, scored
with the dataset's primary metric. A column whose *standalone* out-of-fold score
rivals a whole tuned model is a leak fingerprint no purity heuristic can miss.

The ceilings are **calibrated, not guessed** — measured on the locked 5-dataset
suite (Day 22, seed 42):

* strongest *legitimate* single feature per metric: roc_auc 0.785 (diabetes
  ``plas``), f1_macro 0.489 (vehicle), r2 0.817 (cpu_small ``freeswap``);
* the Day-17 subtle leak scores 0.946 standalone AUC; its regression twin
  (Pearson 0.90) scores 0.881 R².

Each ceiling sits between those bands, so on the clean suite the screen fires
zero false positives while catching every injected probe. The **residual
window is disclosed, not hidden**: a leak whose standalone score lands between
the clean maximum and the ceiling (e.g. roc_auc in (0.785, 0.87)) still passes —
the window is narrowed from "agreement < 0.995" to roughly "standalone score <
ceiling", not closed. ``results/day22_leakage_honesty.json`` records the
calibration this claim rests on.

Two callers, one screen:

* the **Profiler** runs it over the raw train columns (Day-22 upgrade to its
  ``leakage_checks``), so the Planner's existing drop machinery and the Critic's
  residual check pick the suspects up with no new wiring;
* the **Feature Engineer's validation harness** runs the same logic (inlined —
  the sandbox cannot import ``crewml``) over *engineered* columns, catching
  leakage the crew introduces itself.

Train only, structurally: this module takes in-memory frames, never loads any
split, and a no-peeking test asserts it does not name the holdout loader.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from crewml.config import SEED

# Calibrated per-metric ceilings for a single feature's standalone CV score.
# Clean-suite maxima (measured): roc_auc 0.785 / f1_macro 0.489 / r2 0.817.
# Injected leaks (measured):     roc_auc 0.946 (subtle) / r2 0.881 (subtle).
SINGLE_FEATURE_CEILING: dict[str, float] = {
    "roc_auc": 0.87,
    "f1_macro": 0.75,
    "r2": 0.85,
}
SCREEN_MEASURE = "single_feature_cv"  # the `measure` tag screened suspects carry
_CV_FOLDS = 3
_TREE_DEPTH = 3
_MISSING_SENTINEL = -999.0  # trees split around it; keeps NaN rows scoreable


def _encode_feature(x: pd.Series) -> np.ndarray:
    """One feature as a (n, 1) float matrix a decision tree can split on."""
    if not pd.api.types.is_numeric_dtype(x):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        values = enc.fit_transform(x.astype(str).to_frame()).ravel()
        x = pd.Series(values, index=x.index)
    X = x.to_frame().to_numpy(dtype=float)
    return np.nan_to_num(X, nan=_MISSING_SENTINEL, posinf=_MISSING_SENTINEL, neginf=_MISSING_SENTINEL)


def single_feature_cv_score(
    x: pd.Series,
    y: pd.Series,
    task: str,
    metric: str,
    *,
    seed: int = SEED,
) -> Optional[float]:
    """Cross-validated standalone score of one feature against the target.

    A depth-``_TREE_DEPTH`` decision tree, ``_CV_FOLDS``-fold CV, scored with the
    dataset's primary metric — deterministic under ``seed``. Returns ``None``
    when the score cannot be computed (degenerate column, single-class fold);
    an unscoreable feature is not evidence of leakage.
    """
    X = _encode_feature(x)
    if task == "classification":
        model = DecisionTreeClassifier(max_depth=_TREE_DEPTH, random_state=seed)
        cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=seed)
        target = pd.factorize(y)[0]
        scoring = "roc_auc" if metric == "roc_auc" else "f1_macro"
    else:
        model = DecisionTreeRegressor(max_depth=_TREE_DEPTH, random_state=seed)
        cv = KFold(n_splits=_CV_FOLDS, shuffle=True, random_state=seed)
        target = y.to_numpy(dtype=float)
        scoring = "r2"
    try:
        scores = cross_val_score(model, X, target, cv=cv, scoring=scoring)
        value = float(np.mean(scores))
    except Exception:
        return None
    return round(value, 6) if np.isfinite(value) else None


def screen_features(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    metric: str,
    *,
    columns: Optional[list[str]] = None,
    skip: frozenset[str] | set[str] = frozenset(),
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Screen columns for implausible standalone predictive power.

    Returns suspects as ``{"column", "measure": "single_feature_cv", "signal"}``
    dicts — the same shape the Profiler's other leakage suspects use, so they
    flow through the Planner's drops and the Critic's residual check unchanged.
    ``skip`` names columns other screens already flagged (no double-reporting).
    """
    ceiling = SINGLE_FEATURE_CEILING.get(metric)
    if ceiling is None:
        return []
    suspects: list[dict[str, Any]] = []
    for col in (columns if columns is not None else list(X.columns)):
        if col in skip:
            continue
        score = single_feature_cv_score(X[col], y, task, metric, seed=seed)
        if score is not None and score >= ceiling:
            suspects.append({"column": col, "measure": SCREEN_MEASURE, "signal": score})
    return suspects
