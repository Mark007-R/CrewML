"""Baseline 1 — the solo agent: one LLM, one shot, one sklearn script.

This is the number the *crew* must beat. A single agent is handed exactly what
the crew will get (EVAL_PROTOCOL.md §4): the ``train`` split's profile, the task
type, and the primary metric — and nothing else. In one shot it must emit a
self-contained Python module defining::

    def solve(train_df: pd.DataFrame) -> estimator

where ``estimator`` is a *fitted* scikit-learn model (typically a ``Pipeline``)
that accepts the raw feature frame (target column dropped) and supports
``predict`` — plus ``predict_proba`` for binary classification.

Honesty boundary (EVAL_PROTOCOL.md §3):

* The agent — its prompt and its generated ``solve`` — only ever sees ``train``.
  The profile summary is computed from ``train`` alone.
* The generated code is executed in a **subprocess** by a *trusted* runner we
  write (:data:`RUNNER_TEMPLATE`), not by the agent. The runner fits ``solve`` on
  ``train`` and calls ``predict`` on held-out **features only** (no labels). It
  never fits on the holdout.
* Scoring against the held-out labels happens back in this trusted parent process
  through :mod:`crewml.scoring`, and the holdout SHA-256 seal is re-verified after.

When no LLM key is configured the run is in **mock mode**: instead of calling a
model we emit a fixed, competent single-shot script (:func:`mock_solo_script`)
and every result is stamped ``"mock": true`` so it can never masquerade as the
real headline (EVAL_PROTOCOL.md §5).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from crewml import config, llm
from crewml.baselines import split_xy
from crewml.config import ARTIFACTS_DIR, EXECUTOR_TIMEOUT_S, SEED
from crewml.datasets import (
    TARGET_COLUMN,
    DatasetSpec,
    load_holdout,
    load_train,
    verify_holdout_untouched,
)
from crewml.scoring import score_predictions

SOLO_DIR = ARTIFACTS_DIR / "solo"


# --- Data profile handed to the agent (train only) --------------------------

def build_profile_summary(spec: DatasetSpec, train: pd.DataFrame) -> str:
    """A compact, train-only description of the modeling problem for the agent.

    Deliberately mirrors what the Phase-2 Profiler will surface, so the solo
    agent and the crew start from the same knowledge. Never touches the holdout.
    """
    X, y = split_xy(train)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    lines = [
        f"Task: {spec.task} ({spec.subtype}).",
        f"Primary metric to maximise: {spec.metric}.",
        f"Rows (train): {len(train)}. Features: {X.shape[1]} "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical).",
        f"Target column: '{TARGET_COLUMN}'.",
    ]

    if spec.task == "classification":
        counts = y.value_counts()
        dist = ", ".join(f"{cls}={n}" for cls, n in counts.items())
        lines.append(f"Class distribution: {dist}.")
    else:
        lines.append(
            f"Target range: [{y.min():.4g}, {y.max():.4g}], mean={y.mean():.4g}."
        )

    miss = X.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        top = ", ".join(f"{c}={int(n)}" for c, n in miss.sort_values(ascending=False).head(8).items())
        lines.append(f"Missing values (top columns): {top}.")
    else:
        lines.append("Missing values: none flagged as NaN (watch for disguised missing, e.g. 0s).")

    if cat_cols:
        card = {c: int(X[c].nunique()) for c in cat_cols[:8]}
        lines.append("Categorical cardinalities: " + ", ".join(f"{c}={v}" for c, v in card.items()) + ".")

    return "\n".join(lines)


# --- Prompts ----------------------------------------------------------------

SOLO_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a senior ML engineer working ALONE with a single attempt. You are
    given a profile of a TRAINING dataset and must write one self-contained
    Python module that trains the best model you can for the stated metric.

    Hard requirements:
    - Define exactly one top-level function: `def solve(train_df):`.
    - `train_df` is a pandas DataFrame whose target is the column named 'target'.
    - Return a FITTED scikit-learn estimator (prefer a Pipeline) that accepts the
      raw feature frame (target dropped) and supports `.predict`. For BINARY
      classification it must also support `.predict_proba` and expose `.classes_`.
    - Build ALL preprocessing (imputation, encoding, scaling) inside the returned
      estimator so it applies identically at prediction time. Fit only on
      `train_df`. Do NOT read any file, download anything, or access the network.
    - Use only pandas, numpy and scikit-learn. Set random_state=42 for
      reproducibility ONLY on objects that accept it (models, CV splitters,
      randomized/grid search estimators). Do NOT pass random_state to transformers
      such as SimpleImputer, StandardScaler or OneHotEncoder, nor to GridSearchCV —
      they do not accept it and it will raise a TypeError.
    - The module must run first time: import only symbols that actually exist in
      scikit-learn (e.g. there is NO `f1_macro_score` — use
      `f1_score(..., average="macro")` or `make_scorer(f1_score, average="macro")`),
      and pass a constructor only the keyword arguments that class truly accepts.
      You get ONE attempt with no chance to fix a crash, so prefer APIs you are
      certain of over clever ones you are unsure about.
    - Output ONLY the Python module in a single ```python code block. No prose.

    You never see the held-out test set; it is scored separately by predicting
    with your returned estimator. Optimise honestly via cross-validation on
    `train_df` only.
    """
)


def build_user_prompt(spec: DatasetSpec, profile: str) -> str:
    return textwrap.dedent(
        f"""\
        Dataset profile (training split only):
        {profile}

        Write the `solve(train_df)` module now. Choose the model family and
        preprocessing you judge best for maximising {spec.metric} on unseen data
        of this shape. Return only the code.
        """
    )


# --- Mock single-shot script (offline path) ---------------------------------

def mock_solo_script(spec: DatasetSpec) -> str:
    """A fixed, competent one-shot solution used when no LLM key is present.

    Represents a plausible single-agent attempt: leakage-safe preprocessing plus
    an untuned histogram gradient-boosting model. Deterministic (seed=42). Its
    scores are MOCK and are never reported as the real headline.
    """
    is_clf = spec.task == "classification"
    estimator = "HistGradientBoostingClassifier" if is_clf else "HistGradientBoostingRegressor"
    return textwrap.dedent(
        f'''\
        """MOCK solo-agent solution (no LLM). Deterministic single-shot baseline."""
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import {estimator}
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        TARGET = "target"
        SEED = {SEED}


        def solve(train_df):
            X = train_df.drop(columns=[TARGET])
            y = train_df[TARGET]
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = [c for c in X.columns if c not in num_cols]
            pre = ColumnTransformer(
                [
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    ("cat", Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]), cat_cols),
                ],
                remainder="drop",
            )
            model = {estimator}(random_state=SEED, learning_rate=0.1, max_iter=300)
            pipe = Pipeline([("prep", pre), ("model", model)])
            pipe.fit(X, y)
            return pipe
        '''
    )


# --- Trusted subprocess runner (fits solve on train, predicts on holdout) ---

RUNNER_TEMPLATE = textwrap.dedent(
    '''\
    """Trusted runner — NOT agent code. Fits solve() on train, predicts on holdout
    features (labels never present here), and writes predictions for the parent to
    score. The parent holds the holdout labels; this process never fits on holdout.
    """
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    WORK = Path(__file__).resolve().parent
    cfg = json.loads((WORK / "run_config.json").read_text())

    sys.path.insert(0, str(WORK))
    import solution  # the (possibly LLM-generated) module under test

    train = pd.read_parquet(WORK / "train.parquet")
    X_holdout = pd.read_parquet(WORK / "holdout_features.parquet")

    model = solution.solve(train)
    y_pred = np.asarray(model.predict(X_holdout))

    out = {"y_pred": [str(v) for v in y_pred] if cfg["classification"] else [float(v) for v in y_pred]}
    if cfg["needs_proba"]:
        proba = np.asarray(model.predict_proba(X_holdout))
        out["y_proba"] = proba.tolist()
        out["class_labels"] = [str(c) for c in list(model.classes_)]

    (WORK / "predictions.json").write_text(json.dumps(out))
    print("RUNNER_OK", flush=True)
    '''
)


# --- Orchestration: generate -> execute -> score ----------------------------

def generate_script(spec: DatasetSpec, train: pd.DataFrame) -> tuple[str, dict]:
    """Produce the solo agent's script + generation metadata (mock or live LLM)."""
    profile = build_profile_summary(spec, train)
    if config.is_mock_mode():
        return mock_solo_script(spec), {
            "mock": True,
            "provider": "mock",
            "model": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    result = llm.chat(
        SOLO_SYSTEM_PROMPT,
        build_user_prompt(spec, profile),
        temperature=0.0,
        max_tokens=4096,
        agent="solo_agent",
    )
    return llm.extract_python(result.text), {
        "mock": False,
        "provider": result.provider,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def _workdir(key: str) -> Path:
    d = SOLO_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_solo_agent(spec: DatasetSpec, positive_class: str | None) -> dict:
    """Generate, execute and score the solo agent on one dataset.

    Returns a JSON-friendly dict with the scoring result, the generation metadata
    and (on failure) the captured traceback — never raising, so one broken dataset
    is reported as a failure rather than dropping the whole run.
    """
    key = spec.key
    train = load_train(key)
    work = _workdir(key)

    script, meta = generate_script(spec, train)
    (work / "solution.py").write_text(script, encoding="utf-8")
    (work / "runner.py").write_text(RUNNER_TEMPLATE, encoding="utf-8")

    is_clf = spec.task == "classification"
    needs_proba = is_clf and spec.subtype == "binary"
    (work / "run_config.json").write_text(
        json.dumps({"classification": is_clf, "needs_proba": needs_proba})
    )

    # Materialise train + holdout FEATURES (no labels) into the sandboxed workdir.
    train.to_parquet(work / "train.parquet", index=False)
    holdout = load_holdout(key)
    X_ho, y_ho = split_xy(holdout)
    X_ho.to_parquet(work / "holdout_features.parquet", index=False)

    proc = subprocess.run(
        [sys.executable, str(work / "runner.py")],
        capture_output=True,
        text=True,
        timeout=EXECUTOR_TIMEOUT_S,
        cwd=str(work),
    )
    (work / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (work / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    pred_path = work / "predictions.json"
    if proc.returncode != 0 or not pred_path.exists():
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        return {
            "metric": spec.metric,
            "task": spec.task,
            **meta,
            "ok": False,
            "error": "\n".join(tail) or f"runner exited {proc.returncode}",
        }

    preds = json.loads(pred_path.read_text())
    y_pred = preds["y_pred"]
    y_proba = np.asarray(preds["y_proba"]) if "y_proba" in preds else None
    class_labels = preds.get("class_labels")

    scored = score_predictions(
        spec,
        [str(v) for v in y_ho] if is_clf else list(y_ho),
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
        positive_class=positive_class,
    )

    if not verify_holdout_untouched(key):
        raise RuntimeError(f"{key}: holdout seal broken after solo-agent scoring")

    return {
        "metric": spec.metric,
        "task": spec.task,
        **meta,
        "ok": True,
        "value": scored["value"],
        "secondary": scored["secondary"],
        "n_holdout": int(len(holdout)),
    }
