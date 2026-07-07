"""Canonical metric computation for the CrewML evaluation protocol.

Every competing system — the Day 2 baselines, the Day 3 solo agent, the Day 4
AutoML ceiling, and the Phase 2 crew — scores through this one module, so that a
number in any report means exactly the same thing. The primary metric per
dataset is fixed in ``docs/EVAL_PROTOCOL.md`` and the registry:

- binary classification  -> ROC AUC on the predicted probability of the *rarer*
  (positive) class recorded in the manifest;
- multiclass             -> macro-F1 (accuracy reported as a secondary number);
- regression             -> R^2 (RMSE reported alongside).

All three primaries are "higher is better", which keeps cross-system comparisons
in a single direction.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from crewml.datasets import DatasetSpec

# Every primary metric here is maximised — no per-metric sign juggling downstream.
HIGHER_IS_BETTER = True


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_predictions(
    spec: DatasetSpec,
    y_true: Sequence,
    *,
    y_pred: Sequence | None = None,
    y_proba: np.ndarray | None = None,
    class_labels: Sequence | None = None,
    positive_class: str | None = None,
) -> dict:
    """Score one system's predictions on one dataset per the eval protocol.

    Parameters
    ----------
    spec:
        The dataset's registry spec (carries task / subtype / primary metric).
    y_true:
        Ground-truth targets from the held-out split.
    y_pred:
        Predicted labels (classification) or values (regression). Required for
        multiclass-F1, regression-R^2, and the classification accuracy secondary.
    y_proba:
        Predicted class-probability matrix (``n_samples`` x ``n_classes``),
        required for binary ROC AUC. Columns must align with ``class_labels``.
    class_labels:
        The estimator's ``classes_`` — used to locate the positive column in
        ``y_proba`` for binary AUC.
    positive_class:
        The rarer / positive class from the manifest, scored for binary AUC.

    Returns a JSON-friendly dict: ``{"metric", "value", "secondary"}``.
    """
    if spec.task == "classification":
        if spec.subtype == "binary":
            if y_proba is None or class_labels is None or positive_class is None:
                raise ValueError(
                    "binary ROC AUC needs y_proba, class_labels and positive_class"
                )
            labels = list(class_labels)
            # Manifest stores classes as strings; compare on string form to match.
            str_labels = [str(c) for c in labels]
            if str(positive_class) not in str_labels:
                raise ValueError(
                    f"positive_class {positive_class!r} not in class_labels {labels!r}"
                )
            pos_idx = str_labels.index(str(positive_class))
            y_score = np.asarray(y_proba)[:, pos_idx]
            y_bin = (np.asarray([str(v) for v in y_true]) == str(positive_class)).astype(int)
            value = float(roc_auc_score(y_bin, y_score))
            secondary = {}
            if y_pred is not None:
                secondary["accuracy"] = float(accuracy_score(y_true, y_pred))
        else:  # multiclass
            if y_pred is None:
                raise ValueError("multiclass macro-F1 needs y_pred")
            value = float(f1_score(y_true, y_pred, average="macro"))
            secondary = {"accuracy": float(accuracy_score(y_true, y_pred))}
    else:  # regression
        if y_pred is None:
            raise ValueError("regression R^2 needs y_pred")
        value = float(r2_score(y_true, y_pred))
        secondary = {"rmse": _rmse(y_true, y_pred)}

    return {"metric": spec.metric, "value": value, "secondary": secondary}
