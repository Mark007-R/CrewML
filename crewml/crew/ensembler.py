"""The Ensembler agent — combine the run's best models, keep only if it helps (Day 11).

By the time the Ensembler runs, the Trainer (Day 9) has cross-validated every
candidate in the plan, refit the single best, and the Critic (Day 10) has decided
the run is done. The Ensembler asks one more question a good ML engineer always
asks before shipping: **would combining the strongest candidates beat the best one
alone?** It builds a soft-voting (classification) / averaging (regression) ensemble
over the top-``k`` candidates — each with the exact hyper-parameters the Trainer
found for it — cross-validates that ensemble under the **identical CV scheme and
scorer** the Trainer used, and compares it, number to number, against the single
best model re-scored on the same folds.

**Ensembling never hurts, by construction.** The Ensembler keeps the ensemble only
when it beats the single best by more than a small epsilon; on a tie or a loss it
keeps the single model (simpler is better — Occam, and less to serve). So the crew's
final model is *max(ensemble, single)* on CV, never worse than what the Trainer
already had.

Same honesty discipline as the Trainer:

* **CV on train, never the holdout.** Every number is a cross-validated estimate on
  the train split (``cv_score_is_holdout: false``). The comparison that decides
  ensemble-vs-single is self-consistent: both are scored with the *same* seeded CV
  object and scorer inside one sandbox run, so the choice can't be an artifact of
  two different evaluations. Final held-out scoring is a separate, later step (Phase 3).
* **Same preprocessing as the Trainer.** The ensemble config is derived from
  :func:`crewml.crew.trainer._training_config`, so every member's ColumnTransformer,
  imputation, encoding and scaling is byte-for-byte what the Trainer built — the
  ensemble is a genuine combination of the *same* pipelines, not a re-derivation that
  could quietly diverge.
* **Generate-and-run, structurally no-peeking.** The Ensembler assembles a script and
  runs it through :func:`crewml.executor.run_code`, which is handed only
  ``train.parquet``. This module never names the held-out loader — a source-inspection
  test asserts it — so the no-peeking invariant is a property of the code.
* **Failure is reported, never raised.** If the Trainer run failed (no models to
  combine) the Ensembler produces an honest "not attempted" record; if the ensemble
  script itself crashes, that is captured as ``ok: False`` rather than propagated.
"""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Optional

from crewml import config
from crewml.crew import trainer
from crewml.datasets import train_path
from crewml.executor import run_code

ENSEMBLE_SCHEMA_VERSION = 1

# How many of the top CV-ranked candidates go into the vote. The plan ships three
# candidates, so 3 combines all of them; the constant keeps the choice explicit.
ENSEMBLE_TOP_K = 3
# The ensemble must clear the single best by more than this on CV to be worth the
# extra complexity; within epsilon we keep the simpler single model.
ENSEMBLE_MIN_GAIN = 1e-4


# --- Ensemble config (extends the Trainer's, so preprocessing is identical) --

def _ensemble_config(
    plan: dict[str, Any], training: dict[str, Any], top_k: int
) -> tuple[dict[str, Any], list[str]]:
    """Build the ensemble script config + the list of member names.

    Reuses the Trainer's config so the ColumnTransformer/preprocessing is identical,
    then attaches the ``members`` (top-``k`` candidates by the Trainer's CV score, each
    carrying the best hyper-parameters the Trainer found) and the ``single_best`` model
    to re-score for a self-consistent comparison.
    """
    base = trainer._training_config(plan)
    per_model = list((training.get("metrics") or {}).get("per_model") or [])
    by_name = {m["name"]: m for m in plan.get("candidate_models", [])}

    ranked = sorted(per_model, key=lambda r: r.get("cv_mean", float("-inf")), reverse=True)

    def _member(row: dict[str, Any]) -> Optional[dict[str, Any]]:
        cand = by_name.get(row["name"])
        if cand is None:
            return None
        return {
            "name": row["name"],
            "best_params": row.get("best_params") or {},
            "needs_scaling": bool(cand["needs_scaling"]),
            "supports_class_weight": bool(cand["supports_class_weight"]),
        }

    members = [m for m in (_member(r) for r in ranked[:top_k]) if m is not None]

    best_name = training.get("best_model") or (ranked[0]["name"] if ranked else None)
    best_row = next((r for r in per_model if r["name"] == best_name), None)
    single_best = _member(best_row) if best_row is not None else (members[0] if members else None)

    base["members"] = members
    base["single_best"] = single_best
    base["ensemble_min_gain"] = ENSEMBLE_MIN_GAIN
    return base, [m["name"] for m in members]


# --- The ensemble script (assembled from the ensemble config + FE source) ----

_ENSEMBLE_TEMPLATE = textwrap.dedent(
    '''\
    """Generated ensemble script — assembled by the CrewML Ensembler agent.

    Runs inside the sandboxed executor on the TRAIN split only. Cross-validates a
    soft-voting / averaging ensemble of the top candidates against the single best
    model on the SAME seeded folds, keeps whichever wins, refits it, and persists it.
    Never sees the holdout.
    """
    import json

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
        VotingClassifier, VotingRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    from crew_io import SEED, artifact_path, emit_metrics, input_path

    CONFIG = json.loads(r"""#<<CONFIG_JSON>>""")
    TARGET = "target"
    IS_CLF = CONFIG["task"] == "classification"
    IS_BINARY = CONFIG["subtype"] == "binary"

    # --- Feature Engineer's validated add_features(df) is spliced in here ------
    #<<FE_SOURCE>>

    # --- Load train, apply FE ---------------------------------------------------
    train = pd.read_parquet(input_path("train.parquet"))
    y_raw = train[TARGET]
    X = train.drop(columns=[TARGET])
    X = add_features(X.copy())
    original_cols = set(train.columns) - {TARGET}
    engineered = [c for c in X.columns if c not in original_cols]

    # Binary target -> 0/1 with 1 = the plan's positive (rarer) class, so the roc_auc
    # scorer measures exactly the eval-protocol quantity (identical to the Trainer).
    label_mapping = None
    if IS_CLF and IS_BINARY and CONFIG["positive_class"] is not None:
        pos = str(CONFIG["positive_class"])
        y = (y_raw.astype(str) == pos).astype(int)
        neg = sorted(set(y_raw.astype(str)) - {pos})
        label_mapping = {"1": pos, "0": neg[0] if neg else "other"}
    else:
        y = y_raw

    # --- Column groups (engineered columns ride the plain-numeric branch) ------
    num_plain = [c for c in CONFIG["num_plain"] if c in X.columns] + engineered
    num_zero = [c for c in CONFIG["num_zero"] if c in X.columns]
    cat_ohe = [c for c in CONFIG["cat_ohe"] if c in X.columns]
    cat_ord = [c for c in CONFIG["cat_ord"] if c in X.columns]


    def build_preprocessor(needs_scaling):
        transformers = []
        if num_plain:
            steps = [("impute", SimpleImputer(strategy="median"))]
            if needs_scaling:
                steps.append(("scale", StandardScaler()))
            transformers.append(("num", Pipeline(steps), num_plain))
        if num_zero:
            steps = [
                ("zero", SimpleImputer(missing_values=0, strategy="median")),
                ("nan", SimpleImputer(strategy="median")),
            ]
            if needs_scaling:
                steps.append(("scale", StandardScaler()))
            transformers.append(("num_zero", Pipeline(steps), num_zero))
        if cat_ohe:
            transformers.append(("cat_ohe", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_ohe))
        if cat_ord:
            transformers.append(("cat_ord", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_ord))
        return ColumnTransformer(transformers, remainder="drop")


    def make_model(name, supports_cw):
        cw = "balanced" if (CONFIG["use_class_weight"] and supports_cw) else None
        if name == "hist_gradient_boosting":
            return (HistGradientBoostingClassifier(random_state=SEED) if IS_CLF
                    else HistGradientBoostingRegressor(random_state=SEED))
        if name == "random_forest":
            # n_jobs=1: the CV layer parallelises across folds, so each forest stays
            # single-process to avoid nested oversubscription on Windows.
            return (RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=cw) if IS_CLF
                    else RandomForestRegressor(random_state=SEED, n_jobs=1))
        if name == "logistic_regression":
            return LogisticRegression(random_state=SEED, max_iter=1000, class_weight=cw)
        if name == "ridge":
            return Ridge(random_state=SEED)
        raise ValueError("unknown estimator " + name)


    def make_pipeline(member):
        pipe = Pipeline([("prep", build_preprocessor(member["needs_scaling"])),
                         ("model", make_model(member["name"], member["supports_class_weight"]))])
        if member.get("best_params"):
            # Trainer grid keys are pipeline-scoped (model__*), so they apply cleanly.
            pipe.set_params(**member["best_params"])
        return pipe


    if CONFIG["cv_scheme"] == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=SEED)
    else:
        cv = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=SEED)
    scoring = CONFIG["scoring"]

    members = CONFIG["members"]
    single = CONFIG["single_best"]

    # --- Single best, re-scored on THESE folds (self-consistent baseline) -------
    single_pipe = make_pipeline(single)
    single_scores = cross_val_score(single_pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    single_cv = float(np.mean(single_scores))
    single_std = float(np.std(single_scores))
    print(f"[ensembler] single best '{single['name']}': cv {scoring}={single_cv:.4f}", flush=True)

    # --- The ensemble over the top-k members, same folds ------------------------
    # n_jobs=1 on the voter: fold-level parallelism happens at cross_val_score, and
    # the members already run single-process, so this avoids nested oversubscription.
    estimators = [(m["name"], make_pipeline(m)) for m in members]
    if IS_CLF:
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    else:
        ensemble = VotingRegressor(estimators=estimators, n_jobs=1)
    ens_scores = cross_val_score(ensemble, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    ens_cv = float(np.mean(ens_scores))
    ens_std = float(np.std(ens_scores))
    print(f"[ensembler] {'soft-vote' if IS_CLF else 'average'} ensemble "
          f"{[m['name'] for m in members]}: cv {scoring}={ens_cv:.4f}", flush=True)

    # --- Keep the ensemble only if it clears the single best by the margin ------
    gain = ens_cv - single_cv
    if gain > CONFIG["ensemble_min_gain"]:
        chosen, final_model, final_cv, final_std = "ensemble", ensemble, ens_cv, ens_std
    else:
        chosen, final_model, final_cv, final_std = "single", single_pipe, single_cv, single_std

    final_model.fit(X, y)
    model_path = artifact_path("final_model.joblib")
    joblib.dump(final_model, model_path)
    artifact_path("fe_source.py").write_text(FE_SOURCE_TEXT, encoding="utf-8")

    emit_metrics(
        chosen=chosen,
        final_model_kind=chosen,
        members=[m["name"] for m in members],
        n_members=len(members),
        voting=("soft" if IS_CLF else "average"),
        ensemble_cv_score=round(ens_cv, 6),
        ensemble_cv_std=round(ens_std, 6),
        single_best_model=single["name"],
        single_best_cv_score=round(single_cv, 6),
        single_best_cv_std=round(single_std, 6),
        final_cv_score=round(final_cv, 6),
        final_cv_std=round(final_std, 6),
        improvement_over_single=round(gain, 6),
        scoring=scoring,
        cv_scheme=CONFIG["cv_scheme"],
        n_splits=CONFIG["n_splits"],
        task=CONFIG["task"],
        subtype=CONFIG["subtype"],
        metric=CONFIG["metric"],
        positive_class=CONFIG["positive_class"],
        label_mapping=label_mapping,
        engineered_columns=engineered,
        cv_score_is_holdout=False,
        final_model_artifact="final_model.joblib",
        fe_artifact="fe_source.py",
    )
    print(f"[ensembler] chose {chosen} (gain {gain:+.4f}) cv={final_cv:.4f} -> {model_path.name}", flush=True)
    '''
)


def _build_script(fe_code: str, config_payload: dict[str, Any]) -> str:
    """Assemble the runnable ensemble script from the FE source and ensemble config."""
    fe_block = fe_code + "\n\nFE_SOURCE_TEXT = " + repr(fe_code) + "\n"
    script = _ENSEMBLE_TEMPLATE.replace("#<<CONFIG_JSON>>", json.dumps(config_payload))
    script = script.replace("#<<FE_SOURCE>>", fe_block)
    return script


def _ensemble_timeout(timeout_s: Optional[int]) -> int:
    """Ensembling re-CVs several fitted models, so default to a roomier cap than one train."""
    if timeout_s is not None:
        return int(timeout_s)
    env = os.getenv("CREWML_ENSEMBLER_TIMEOUT_S")
    if env:
        return int(env)
    return 2 * config.EXECUTOR_TIMEOUT_S


def _not_attempted(
    dataset_key: str, training: dict[str, Any], reason: str, *, iteration: int
) -> dict[str, Any]:
    """An honest record for when there is nothing to ensemble (falls back to the Trainer's model)."""
    return {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "stub": False,
        "node": "ensembler",
        "dataset_key": dataset_key,
        "iteration": int(iteration),
        "attempted": False,
        "ok": bool(training.get("ok")),
        "reason": reason,
        "chosen": "single" if training.get("ok") else None,
        "final_model_kind": "single" if training.get("ok") else None,
        "members": None,
        "single_best_model": training.get("best_model"),
        "single_best_cv_score": training.get("cv_score"),
        "ensemble_cv_score": None,
        "improvement_over_single": None,
        "final_cv_score": training.get("cv_score"),
        "cv_score_is_holdout": False,
        "metrics": {},
    }


def run_ensembler(
    plan: dict[str, Any],
    training: dict[str, Any],
    fe_code: str,
    dataset_key: str,
    *,
    iteration: int = 0,
    top_k: int = ENSEMBLE_TOP_K,
    timeout_s: Optional[int] = None,
) -> dict[str, Any]:
    """Combine the top candidates and keep the ensemble only if it beats the single best.

    Builds a soft-voting/averaging ensemble over the top-``top_k`` CV-ranked candidates
    (each with the Trainer's best params), cross-validates it against the single best on
    the same seeded folds inside the sandbox, and returns which model the crew ships plus
    both scores. Never raises for modeling failures — a Trainer that failed, too few
    candidates to combine, or a crash in the generated script all yield an honest record.
    """
    # Nothing to combine: the Trainer failed, or fewer than two candidates scored.
    if not training.get("ok"):
        return _not_attempted(dataset_key, training, "training run failed — no models to combine", iteration=iteration)

    cfg, member_names = _ensemble_config(plan, training, top_k)
    if len(member_names) < 2 or cfg.get("single_best") is None:
        return _not_attempted(
            dataset_key, training,
            f"only {len(member_names)} candidate(s) available — an ensemble needs at least two",
            iteration=iteration,
        )

    script = _build_script(fe_code, cfg)
    result = run_code(
        script,
        inputs={"train.parquet": train_path(dataset_key)},
        timeout_s=_ensemble_timeout(timeout_s),
        keep_workdir=True,
    )
    metrics = result.metrics or {}

    ensemble: dict[str, Any] = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "stub": False,
        "node": "ensembler",
        "dataset_key": dataset_key,
        "iteration": int(iteration),
        "attempted": True,
        "ok": bool(result.ok),
        "run_id": result.run_id,
        "duration_s": round(result.duration_s, 3),
        "timed_out": result.timed_out,
        "error": result.error,
        "artifacts": list(result.artifacts),
        "members": member_names,
        "voting": metrics.get("voting"),
        "chosen": metrics.get("chosen") if result.ok else None,
        "final_model_kind": metrics.get("final_model_kind") if result.ok else None,
        "single_best_model": metrics.get("single_best_model") or training.get("best_model"),
        "single_best_cv_score": metrics.get("single_best_cv_score"),
        "ensemble_cv_score": metrics.get("ensemble_cv_score"),
        "improvement_over_single": metrics.get("improvement_over_single"),
        # Convenience surface for the Reporter — always a CV estimate on train.
        "final_cv_score": metrics.get("final_cv_score") if result.ok else training.get("cv_score"),
        "cv_score_is_holdout": False,
        "metrics": metrics,
    }
    # If the ensemble script crashed, fall back to the Trainer's single model honestly.
    if not result.ok:
        ensemble["chosen"] = "single"
        ensemble["final_model_kind"] = "single"
        ensemble["reason"] = "ensemble script failed — keeping the Trainer's single best model"
        ensemble["final_cv_score"] = training.get("cv_score")
    return ensemble
