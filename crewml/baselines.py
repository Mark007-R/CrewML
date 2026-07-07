"""Baseline 0 — the honest floor and a default-model anchor.

Two non-agent systems per dataset:

- **dummy** — ``DummyClassifier(strategy="prior")`` / ``DummyRegressor(strategy=
  "mean")``. It ignores the features entirely, so it marks the score any real
  system must clear. A crew that fails to beat Dummy is reported as broken.
- **default_rf** — an out-of-the-box ``RandomForest`` wrapped in the *minimum*
  preprocessing needed to run at all (median impute for numerics, most-frequent
  impute + one-hot for categoricals). No tuning. This is the "just throw a forest
  at it" reference the crew and solo agent are trying to improve on.

Both fit strictly on ``train`` and are scored once on the LOCKED ``holdout`` via
:mod:`crewml.scoring`, exactly like every other competing system. The preprocessor
is fit only on training data, so nothing leaks from the holdout (eval protocol
sec. 3.4).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from crewml.config import SEED
from crewml.datasets import TARGET_COLUMN, DatasetSpec
from crewml.scoring import score_predictions

# The two baseline systems, in report order.
BASELINE_SYSTEMS = ("dummy", "default_rf")


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a materialised frame into features and the standardised target."""
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Minimal, leakage-safe preprocessing: impute, and one-hot the categoricals.

    Numeric columns are median-imputed; everything else (object / category /
    bool) is most-frequent-imputed then one-hot encoded with unknown categories
    ignored so the holdout can carry values unseen in train.
    """
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def make_estimator(system: str, spec: DatasetSpec, X: pd.DataFrame) -> Pipeline:
    """Build the fitted-on-train pipeline for a baseline system on a dataset."""
    is_clf = spec.task == "classification"
    if system == "dummy":
        model = (
            DummyClassifier(strategy="prior")
            if is_clf
            else DummyRegressor(strategy="mean")
        )
    elif system == "default_rf":
        model = (
            RandomForestClassifier(random_state=SEED, n_jobs=-1)
            if is_clf
            else RandomForestRegressor(random_state=SEED, n_jobs=-1)
        )
    else:  # pragma: no cover - guarded by BASELINE_SYSTEMS
        raise ValueError(f"unknown baseline system {system!r}")
    return Pipeline([("prep", build_preprocessor(X)), ("model", model)])


def fit_score_baseline(
    system: str,
    spec: DatasetSpec,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    positive_class: str | None,
) -> dict:
    """Fit ``system`` on ``train`` and score it once on ``holdout``.

    Returns the scoring dict augmented with ``n_holdout`` for the report.
    """
    X_tr, y_tr = split_xy(train)
    X_ho, y_ho = split_xy(holdout)

    pipe = make_estimator(system, spec, X_tr)
    pipe.fit(X_tr, y_tr)

    needs_proba = spec.task == "classification" and spec.subtype == "binary"
    y_pred = pipe.predict(X_ho)
    y_proba = pipe.predict_proba(X_ho) if needs_proba else None
    class_labels = list(pipe.classes_) if spec.task == "classification" else None

    result = score_predictions(
        spec,
        y_ho,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        positive_class=positive_class,
    )
    result["n_holdout"] = int(len(holdout))
    return result
