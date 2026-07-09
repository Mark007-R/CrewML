"""Baseline 2 — the classical-AutoML ceiling (FLAML).

This is the strong *non-agent* competitor. Where Baseline 0 (Dummy / default RF)
marks the floor and Baseline 1 (the solo agent) is one LLM's single shot, FLAML
is a mature, heavily-optimised AutoML system that searches over gradient-boosted
trees, random forests, linear models and more under a fixed time budget. The
headline claim — *"the crew beats a solo agent AND classical AutoML"* — needs a
credible ceiling to clear, and this module builds it.

Fairness + honesty (EVAL_PROTOCOL.md §3–4):

* FLAML fits **strictly on ``train``**, doing all model selection via its own
  cross-validation *inside* that split. It never sees ``holdout``.
* It is scored **once** on the LOCKED ``holdout`` through :mod:`crewml.scoring`,
  the same canonical scorer every other system uses — same data, same metric.
* The held-out SHA-256 seal is re-verified after scoring (in the driver), proving
  the ceiling neither peeked at nor mutated the holdout.
* FLAML is *trusted* library code (not agent-generated), so — like the Day 2
  baselines — it runs in-process; no subprocess sandbox is needed.

A note on reproducibility: FLAML's search is **time-budgeted**, so the exact model
it lands on depends on how many trials the budget buys on a given machine. Unlike
the seed-locked Day 2 baselines, the AutoML number is therefore reproducible *in
distribution*, not bit-for-bit. Every run records its ``time_budget_s``, the
chosen ``best_estimator``, the FLAML version and the seed so a result can always
be situated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crewml.baselines import split_xy
from crewml.config import SEED
from crewml.datasets import DatasetSpec
from crewml.scoring import score_predictions

# The AutoML system id used in reports/results.
AUTOML_SYSTEM = "automl_flaml"

# CrewML primary metric -> FLAML optimisation metric. FLAML maximises AUC/R^2 and
# minimises error internally; we hand it the metric that matches our scorer so the
# search optimises the same objective the holdout is graded on.
FLAML_METRIC = {
    "roc_auc": "roc_auc",   # binary — FLAML uses P(positive) AUC, as does our scorer
    "f1_macro": "macro_f1",  # multiclass — class-balanced, matches EVAL_PROTOCOL §2
    "r2": "r2",             # regression
}


def run_automl(
    spec: DatasetSpec,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    positive_class: str | None,
    time_budget_s: int,
) -> dict:
    """Fit FLAML on ``train`` and score it once on the LOCKED ``holdout``.

    Returns the scoring dict augmented with the AutoML run metadata
    (``best_estimator``, ``time_budget_s``, ``flaml_version`` …) for the report.
    Raises on a null model so a budget-starved run surfaces as a driver failure
    rather than a silent empty result (no silent drops, EVAL_PROTOCOL §5).
    """
    # Lazy import so the package (and test collection) does not hard-require FLAML.
    from flaml import AutoML
    import flaml

    X_tr, y_tr = split_xy(train)
    X_ho, y_ho = split_xy(holdout)

    is_clf = spec.task == "classification"
    task = "classification" if is_clf else "regression"
    metric = FLAML_METRIC[spec.metric]

    automl = AutoML()
    automl.fit(
        X_train=X_tr,
        y_train=y_tr,
        task=task,
        metric=metric,
        time_budget=time_budget_s,
        eval_method="cv",
        n_splits=5,
        seed=SEED,
        verbose=0,
        log_file_name="",   # keep the run log out of the repo
        early_stop=True,
    )

    # FLAML returns a null model if the budget bought no completed trial — treat
    # that as a hard failure (the ceiling must actually exist to be a ceiling).
    if getattr(automl, "model", None) is None or automl.best_estimator is None:
        raise RuntimeError(
            f"FLAML produced no model within {time_budget_s}s — raise the budget."
        )

    needs_proba = is_clf and spec.subtype == "binary"
    y_pred = np.asarray(automl.predict(X_ho))
    y_proba = np.asarray(automl.predict_proba(X_ho)) if needs_proba else None
    class_labels = [str(c) for c in list(automl.classes_)] if is_clf else None

    scored = score_predictions(
        spec,
        [str(v) for v in y_ho] if is_clf else list(y_ho),
        y_pred=[str(v) for v in y_pred] if is_clf else list(y_pred),
        y_proba=y_proba,
        class_labels=class_labels,
        positive_class=positive_class,
    )

    return {
        "metric": spec.metric,
        "task": spec.task,
        "system": AUTOML_SYSTEM,
        "best_estimator": str(automl.best_estimator),
        "best_config": {k: _jsonable(v) for k, v in (automl.best_config or {}).items()},
        "time_budget_s": int(time_budget_s),
        # FLAML's time to train the single best config (not the total search time,
        # which is bounded by time_budget_s above).
        "best_model_train_time_s": float(
            getattr(automl, "best_config_train_time", 0.0) or 0.0
        ),
        "flaml_version": flaml.__version__,
        "ok": True,
        "value": scored["value"],
        "secondary": scored["secondary"],
        "n_holdout": int(len(holdout)),
    }


def _jsonable(v):
    """Coerce a FLAML config value into something ``json.dumps`` accepts."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v
