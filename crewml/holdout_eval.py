"""Final held-out scoring for the crew — the honesty proof, run once, at the end.

Everything the crew produced up to Day 11 is a *cross-validated estimate on train*
(``cv_score_is_holdout: false``). Those numbers cannot be compared to the Day 2-4
baselines, which are all scored on the LOCKED holdout. This module closes that gap
and is what makes Day 12's crew-vs-solo-vs-AutoML table an apples-to-apples board.

The design keeps the hard invariant structural rather than a rule someone has to
remember (EVAL_PROTOCOL.md §3):

* **The crew is already finished.** This module runs *after* the graph has returned.
  It reads the fitted model the Ensembler/Trainer persisted and never re-enters a
  node, so no holdout signal can flow back into modeling.
* **The model never refits.** :func:`score_on_holdout` only calls ``predict`` /
  ``predict_proba``. There is no ``fit`` in the scorer at all.
* **The sandbox never receives labels.** Only holdout *features* are copied into the
  workdir. The target column is dropped in the parent, and the parent alone holds
  ``y_true``, mirroring the Day-3 solo-agent runner exactly.
* **Scoring is not re-implemented.** The sandbox hands back raw predictions; the
  value is computed by :func:`crewml.scoring.score_predictions` — the same call the
  baselines, the solo agent and AutoML go through — so a crew number means exactly
  what a baseline number means.
* **The seal is checked after.** :func:`crewml.datasets.verify_holdout_untouched`
  re-fingerprints the split post-scoring; a broken seal raises rather than returns.

Binary tasks carry one wrinkle: the Trainer/Ensembler fit on a 0/1-mapped target
(1 = the rarer positive class), so the fitted estimator predicts 0/1 while the
holdout labels are the original strings. The recorded ``label_mapping`` is inverted
here to put predictions back in the dataset's own vocabulary *before* scoring, which
keeps ROC AUC pointed at the protocol's positive class and the accuracy secondary
meaningful.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Optional

import numpy as np

from crewml.baselines import split_xy
from crewml.config import ARTIFACTS_DIR, EXECUTOR_TIMEOUT_S
from crewml.datasets import (
    DatasetSpec,
    load_holdout,
    verify_holdout_untouched,
)
from crewml.executor import EXECUTOR_DIR, run_code
from crewml.scoring import score_predictions

HOLDOUT_EVAL_SCHEMA_VERSION = 1

# Where holdout FEATURES are staged for the sandbox (git-ignored, never labels).
HOLDOUT_EVAL_DIR = ARTIFACTS_DIR / "holdout_eval"


# --- The prediction script (runs in the sandbox; predicts, never fits) -------

_PREDICT_TEMPLATE = textwrap.dedent(
    '''\
    """Generated final-scoring script — CrewML holdout evaluator.

    Loads the model the crew already fitted on train, applies the crew's own
    feature engineering to holdout FEATURES (labels are not present in this
    workdir at all), and writes raw predictions for the parent to score.

    There is deliberately no ``fit`` call here: the crew's modeling is over.
    """
    import json

    import numpy as np
    import pandas as pd
    import joblib

    from crew_io import artifact_path, emit_metrics, input_path

    CONFIG = json.loads(r"""#<<CONFIG_JSON>>""")

    # --- The crew's validated add_features(df) is spliced in here -------------
    #<<FE_SOURCE>>

    X = pd.read_parquet(input_path("holdout_features.parquet"))
    n_rows_in = int(len(X))
    X = add_features(X.copy())

    model = joblib.load(input_path("final_model.joblib"))
    y_pred = np.asarray(model.predict(X))

    out = {
        "y_pred": [str(v) for v in y_pred] if CONFIG["classification"] else [float(v) for v in y_pred],
    }
    if CONFIG["needs_proba"]:
        proba = np.asarray(model.predict_proba(X))
        out["y_proba"] = proba.tolist()
        out["class_labels"] = [str(c) for c in list(model.classes_)]

    artifact_path("predictions.json").write_text(json.dumps(out))
    emit_metrics(
        n_rows_in=n_rows_in,
        n_rows_scored=int(len(X)),
        n_features_after_fe=int(X.shape[1]),
        refit_on_holdout=False,
        predictions_artifact="predictions.json",
    )
    print(f"[holdout_eval] predicted {len(y_pred)} rows (no fit performed)", flush=True)
    '''
)


def _build_script(fe_code: str, config_payload: dict[str, Any]) -> str:
    """Assemble the runnable prediction script from the crew's FE source + config."""
    script = _PREDICT_TEMPLATE.replace("#<<CONFIG_JSON>>", json.dumps(config_payload))
    return script.replace("#<<FE_SOURCE>>", fe_code)


# --- Locating what the crew actually shipped --------------------------------

def final_model_ref(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Resolve which fitted model the crew shipped for a run, and where it lives.

    Prefers the Ensembler's ``final_model.joblib`` (it is the node that made the
    ship/no-ship call between the ensemble and the single best). Falls back to the
    Trainer's ``model.joblib`` when the Ensembler never attempted or crashed — in
    which case the Trainer's best single model *is* what the crew shipped.

    Returns ``None`` when the run produced no usable model at all, so a failed
    dataset is reported honestly rather than silently scored as something else.
    """
    ensemble = state.get("ensemble") or {}
    training = state.get("training") or {}

    if ensemble.get("attempted") and ensemble.get("ok") and ensemble.get("run_id"):
        metrics = ensemble.get("metrics") or {}
        return {
            "source": "ensembler",
            "run_id": ensemble["run_id"],
            "model_artifact": metrics.get("final_model_artifact", "final_model.joblib"),
            "model_kind": ensemble.get("final_model_kind"),
            "cv_score": ensemble.get("final_cv_score"),
            "positive_class": metrics.get("positive_class"),
            "label_mapping": metrics.get("label_mapping"),
        }

    if training.get("ok") and training.get("run_id"):
        metrics = training.get("metrics") or {}
        return {
            "source": "trainer",
            "run_id": training["run_id"],
            "model_artifact": metrics.get("model_artifact", "model.joblib"),
            "model_kind": "single",
            "cv_score": training.get("cv_score"),
            "positive_class": metrics.get("positive_class"),
            "label_mapping": metrics.get("label_mapping"),
        }

    return None


def _artifact_path(run_id: str, name: str) -> Path:
    return EXECUTOR_DIR / run_id / "artifacts" / name


def _decode_predictions(
    values: list, label_mapping: Optional[dict[str, str]]
) -> list:
    """Map 0/1 predictions back to the dataset's own class vocabulary.

    The Trainer fits binary targets as 0/1 with 1 = the positive class; the holdout
    labels are the originals. Without this inversion the scorer would compare "1"
    against "bad" and score a perfectly good model at chance.
    """
    if not label_mapping:
        return [str(v) for v in values]
    return [str(label_mapping.get(str(v), v)) for v in values]


# --- The public entry point --------------------------------------------------

def score_on_holdout(
    spec: DatasetSpec,
    state: dict[str, Any],
    *,
    positive_class: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> dict[str, Any]:
    """Score one finished crew run on the LOCKED holdout. Never refits, never raises
    for modeling failures.

    Parameters
    ----------
    spec:
        The dataset's registry spec (task / subtype / primary metric).
    state:
        The final LangGraph state returned by the crew for this dataset.
    positive_class:
        The manifest's rarer/positive class, scored for binary ROC AUC.
    timeout_s:
        Sandbox wall-clock cap; defaults to :data:`config.EXECUTOR_TIMEOUT_S`.

    Returns a JSON-friendly record carrying the held-out value, the CV estimate it
    is paired with (so CV-vs-holdout optimism is visible), and the seal check.
    Raises only if the holdout fingerprint diverges — that is not a result, it is a
    broken invariant.
    """
    key = spec.key
    ref = final_model_ref(state)
    if ref is None:
        return {
            "schema_version": HOLDOUT_EVAL_SCHEMA_VERSION,
            "dataset_key": key,
            "metric": spec.metric,
            "ok": False,
            "error": "crew produced no usable fitted model — nothing to score",
            "holdout_score_is_holdout": True,
        }

    model_path = _artifact_path(ref["run_id"], ref["model_artifact"])
    fe_path = _artifact_path(ref["run_id"], "fe_source.py")
    if not model_path.exists():
        return {
            "schema_version": HOLDOUT_EVAL_SCHEMA_VERSION,
            "dataset_key": key,
            "metric": spec.metric,
            "ok": False,
            "error": f"fitted model artifact missing: {model_path}",
            "final_model_source": ref["source"],
            "holdout_score_is_holdout": True,
        }

    # The crew's FE source rides with the model; fall back to the live state copy.
    fe_code = fe_path.read_text(encoding="utf-8") if fe_path.exists() else (state.get("fe_code") or "")
    if "def add_features" not in fe_code:
        fe_code = "def add_features(df):\n    return df\n"

    # Stage holdout FEATURES ONLY — the target column never enters the sandbox.
    holdout = load_holdout(key)
    X_ho, y_ho = split_xy(holdout)
    stage_dir = HOLDOUT_EVAL_DIR / key
    stage_dir.mkdir(parents=True, exist_ok=True)
    features_path = stage_dir / "holdout_features.parquet"
    X_ho.to_parquet(features_path, index=False)

    is_clf = spec.task == "classification"
    needs_proba = is_clf and spec.subtype == "binary"
    script = _build_script(
        fe_code, {"classification": is_clf, "needs_proba": needs_proba}
    )

    result = run_code(
        script,
        inputs={
            "holdout_features.parquet": features_path,
            "final_model.joblib": model_path,
        },
        timeout_s=int(timeout_s if timeout_s is not None else EXECUTOR_TIMEOUT_S),
        keep_workdir=True,
    )

    base = {
        "schema_version": HOLDOUT_EVAL_SCHEMA_VERSION,
        "dataset_key": key,
        "task": spec.task,
        "subtype": spec.subtype,
        "metric": spec.metric,
        "final_model_source": ref["source"],
        "final_model_kind": ref.get("model_kind"),
        "cv_score": ref.get("cv_score"),
        "run_id": result.run_id,
        "duration_s": round(result.duration_s, 3),
        "refit_on_holdout": False,
        "holdout_score_is_holdout": True,
    }

    if not result.ok:
        return {
            **base,
            "ok": False,
            "error": result.error or "prediction script failed",
            "timed_out": result.timed_out,
        }

    pred_path = _artifact_path(result.run_id, "predictions.json")
    if not pred_path.exists():
        return {**base, "ok": False, "error": "prediction script wrote no predictions"}

    preds = json.loads(pred_path.read_text())
    label_mapping = ref.get("label_mapping")
    pos = positive_class if positive_class is not None else ref.get("positive_class")

    if is_clf:
        y_pred = _decode_predictions(preds["y_pred"], label_mapping)
        y_true = [str(v) for v in y_ho]
    else:
        y_pred = [float(v) for v in preds["y_pred"]]
        y_true = list(y_ho)

    y_proba = np.asarray(preds["y_proba"]) if "y_proba" in preds else None
    class_labels = preds.get("class_labels")
    if class_labels is not None and label_mapping:
        class_labels = _decode_predictions(class_labels, label_mapping)

    scored = score_predictions(
        spec,
        y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        positive_class=pos,
    )

    # The seal is the whole point of the project — a break is fatal, not a datum.
    if not verify_holdout_untouched(key):
        raise RuntimeError(f"{key}: holdout seal broken after crew holdout scoring")

    cv = ref.get("cv_score")
    value = scored["value"]
    return {
        **base,
        "ok": True,
        "value": value,
        "secondary": scored["secondary"],
        "n_holdout": int(len(holdout)),
        # Positive => the CV estimate was optimistic relative to reality.
        "cv_minus_holdout": round(cv - value, 6) if isinstance(cv, (int, float)) else None,
        "holdout_untouched": verify_holdout_untouched(key),
    }
