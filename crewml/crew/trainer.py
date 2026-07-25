"""The Trainer agent — the crew's first real *modeling* node (Day 9).

The Trainer turns the Planner's **ModelingPlan** and the Feature Engineer's
validated ``add_features`` code into a **training script**, hands it to the Day-6
sandboxed executor, and reads back **cross-validated metrics + a saved model
artifact**. It is the node that produces the crew's first honest numbers.

What the generated script does, all inside the sandbox on the **train split only**:

1. Load ``train.parquet``, split off the ``target`` column.
2. Apply the Feature Engineer's row-wise ``add_features`` to the feature frame
   (leakage-free by the FE contract, so it may run once up front).
3. Build a dtype-aware preprocessing ``ColumnTransformer`` straight from the plan —
   median imputation for numerics (a separate *treat-zero-as-missing* branch for the
   Profiler's disguised-missing columns), most-frequent + one-hot for low-cardinality
   categoricals and ordinal encoding for high-cardinality ones, optional scaling for
   scale-sensitive models. Engineered columns join the numeric branch.
4. For each candidate model in the plan, cross-validate under the plan's CV scheme
   and primary-metric scorer — optionally searching the plan's seed grid — and keep
   the mean/std. Class weights are applied when the plan's imbalance strategy asks
   for them and the model supports it.
5. Select the best candidate by mean CV score, refit it on the full train split,
   and persist it (``model.joblib``) alongside the exact ``fe_source.py`` used.
6. Emit a structured ``metrics.json`` the crew reads back.

**Honesty — the number is a CV estimate, not a held-out score.** Every metric the
Trainer returns is a *cross-validated estimate on train* (``cv_score_is_holdout:
false``). The locked held-out set is never loaded here; final held-out scoring is a
separate, later step (Phase 3). For binary tasks the target is mapped to 0/1 with
``1`` = the plan's positive (rarer) class, so the sklearn ``roc_auc`` scorer measures
exactly the protocol's quantity; the mapping is recorded for later inversion.

**Generate-and-run, structurally no-peeking.** The Trainer writes code and executes
it through :func:`crewml.executor.run_code`, which is handed only ``train.parquet``.
This module never names the held-out loader — a source-inspection test asserts it —
so the no-peeking invariant is a property of the code, not a rule to remember.

**Self-repairing since Day 20.** A script that crashes in the sandbox no longer
just files an honest failure: the Trainer hands the traceback and the full source
to the :mod:`crewml.repair` loop and adopts a repaired run when one succeeds —
with the whole attempt trail recorded under ``training["repair"]``.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Optional

from crewml import config
from crewml.datasets import train_path
from crewml.executor import ARTIFACTS_SUBDIR, run_code
from crewml.repair import is_repairable, repair_enabled_for_trainer, repair_loop

TRAINER_SCHEMA_VERSION = 2  # v2 (Day 20): + "repair"/"repaired" self-repair provenance


# --- Config the generated script needs (all derived from the plan) ----------

def _training_config(plan: dict[str, Any]) -> dict[str, Any]:
    """Distil the plan into the compact, JSON-safe config the script consumes."""
    pre = plan["preprocessing"]
    imb = plan["imbalance_strategy"]
    numeric = list(pre["numeric"]["columns"])
    zero_as_missing = list(pre["numeric"]["zero_as_missing"])
    num_plain = [c for c in numeric if c not in set(zero_as_missing)]

    return {
        "task": plan["task"],
        "subtype": plan["subtype"],
        "metric": plan["metric"],
        "positive_class": imb.get("positive_class"),
        "num_plain": num_plain,
        "num_zero": zero_as_missing,
        "cat_ohe": list(pre["categorical"]["onehot_columns"]),
        "cat_ord": list(pre["categorical"]["ordinal_columns"]),
        "drop_columns": list(pre["drop_columns"]),
        "cv_scheme": plan["cv"]["scheme"],
        "n_splits": int(plan["cv"]["n_splits"]),
        "scoring": plan["cv"]["scoring"],
        "use_class_weight": bool(imb.get("recommended")),
        "candidates": [
            {
                "name": m["name"],
                "estimator": m["estimator"],
                "needs_scaling": bool(m["needs_scaling"]),
                "supports_class_weight": bool(m["supports_class_weight"]),
                "param_grid": m["param_grid"],
            }
            for m in plan["candidate_models"]
        ],
    }


# --- The training script (assembled from plan config + FE source) -----------

_TRAINER_TEMPLATE = textwrap.dedent(
    '''\
    """Generated training script — assembled by the CrewML Trainer agent.

    Runs inside the sandboxed executor on the TRAIN split only. Cross-validates the
    planned candidates, refits the best, and persists it. Never sees the holdout.
    """
    import json

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import (
        GridSearchCV, KFold, StratifiedKFold, cross_val_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    from crew_io import SEED, artifact_path, emit_metrics, input_path

    CONFIG = json.loads(r"""#<<CONFIG_JSON>>""")
    PARAM_SEARCH = #<<PARAM_SEARCH>>
    TARGET = "target"
    IS_CLF = CONFIG["task"] == "classification"
    IS_BINARY = CONFIG["subtype"] == "binary"

    # --- Feature Engineer's validated add_features(df) is spliced in here ------
    #<<FE_SOURCE>>

    # --- Load train, apply FE ---------------------------------------------------
    train = pd.read_parquet(input_path("train.parquet"))
    y_raw = train[TARGET]
    X = train.drop(columns=[TARGET])
    n_original = X.shape[1]
    X = add_features(X.copy())
    original_cols = set(train.columns) - {TARGET}
    engineered = [c for c in X.columns if c not in original_cols]

    # Binary target -> 0/1 with 1 = the plan's positive (rarer) class, so the
    # roc_auc scorer measures exactly the eval-protocol quantity.
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


    def make_model(cand):
        name = cand["name"]
        cw = "balanced" if (CONFIG["use_class_weight"] and cand["supports_class_weight"]) else None
        if name == "hist_gradient_boosting":
            return (HistGradientBoostingClassifier(random_state=SEED) if IS_CLF
                    else HistGradientBoostingRegressor(random_state=SEED))
        if name == "random_forest":
            # n_jobs=1 here: the CV/grid-search layer parallelises (n_jobs=-1), so the
            # forest stays single-process to avoid nested oversubscription on Windows.
            return (RandomForestClassifier(random_state=SEED, n_jobs=1, class_weight=cw) if IS_CLF
                    else RandomForestRegressor(random_state=SEED, n_jobs=1))
        if name == "logistic_regression":
            return LogisticRegression(random_state=SEED, max_iter=1000, class_weight=cw)
        if name == "ridge":
            return Ridge(random_state=SEED)
        raise ValueError("unknown estimator " + name)


    if CONFIG["cv_scheme"] == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=SEED)
    else:
        cv = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=SEED)

    scoring = CONFIG["scoring"]

    # --- Cross-validate every candidate ----------------------------------------
    per_model = []
    fitted_by_name = {}
    unfit_by_name = {}
    for cand in CONFIG["candidates"]:
        pipe = Pipeline([("prep", build_preprocessor(cand["needs_scaling"])),
                         ("model", make_model(cand))])
        grid = cand["param_grid"] if PARAM_SEARCH else None
        if grid:
            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, refit=True)
            gs.fit(X, y)
            cv_mean = float(gs.best_score_)
            cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
            best_params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                           for k, v in gs.best_params_.items()}
            fitted_by_name[cand["name"]] = gs.best_estimator_
        else:
            scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))
            best_params = {}
            unfit_by_name[cand["name"]] = pipe
        per_model.append({"name": cand["name"], "cv_mean": round(cv_mean, 6),
                          "cv_std": round(cv_std, 6), "best_params": best_params})
        print(f"[trainer] {cand['name']}: cv {scoring}={cv_mean:.4f} (+/-{cv_std:.4f})", flush=True)

    # --- Pick the winner, refit on full train, persist -------------------------
    best = max(per_model, key=lambda r: r["cv_mean"])
    winner = best["name"]
    model = fitted_by_name.get(winner)
    if model is None:
        model = unfit_by_name[winner].fit(X, y)

    model_path = artifact_path("model.joblib")
    joblib.dump(model, model_path)
    artifact_path("fe_source.py").write_text(FE_SOURCE_TEXT, encoding="utf-8")

    emit_metrics(
        best_model=winner,
        best_cv_score=best["cv_mean"],
        best_cv_std=best["cv_std"],
        best_params=best["best_params"],
        per_model=per_model,
        scoring=scoring,
        cv_scheme=CONFIG["cv_scheme"],
        n_splits=CONFIG["n_splits"],
        param_search=bool(PARAM_SEARCH),
        task=CONFIG["task"],
        subtype=CONFIG["subtype"],
        metric=CONFIG["metric"],
        positive_class=CONFIG["positive_class"],
        label_mapping=label_mapping,
        n_features_original=int(n_original),
        n_features_after_fe=int(X.shape[1]),
        engineered_columns=engineered,
        n_engineered=len(engineered),
        fe_applied=True,
        cv_score_is_holdout=False,
        model_artifact="model.joblib",
        fe_artifact="fe_source.py",
    )
    print(f"[trainer] best={winner} cv={best['cv_mean']:.4f} -> {model_path.name}", flush=True)
    '''
)


def _build_script(fe_code: str, config_payload: dict[str, Any], param_search: bool) -> str:
    """Assemble the runnable training script from the FE source and plan config."""
    # Expose the FE source as a string constant too, so the winner's exact FE is
    # persisted next to the model for later (Phase-3) held-out scoring.
    fe_block = (
        fe_code
        + "\n\nFE_SOURCE_TEXT = "
        + repr(fe_code)
        + "\n"
    )
    script = _TRAINER_TEMPLATE.replace("#<<CONFIG_JSON>>", json.dumps(config_payload))
    script = script.replace("#<<PARAM_SEARCH>>", "True" if param_search else "False")
    script = script.replace("#<<FE_SOURCE>>", fe_block)
    return script


def _param_search_enabled(param_search: Optional[bool]) -> bool:
    """Explicit flag wins; else the ``CREWML_TRAINER_PARAM_SEARCH`` toggle (default on)."""
    if param_search is not None:
        return param_search
    return os.getenv("CREWML_TRAINER_PARAM_SEARCH", "1") != "0"


def _read_persisted_fe(result: Any) -> Optional[str]:
    """The ``fe_source.py`` a successful run wrote, or None if unreadable.

    This is the FE the winning model was actually fitted with, so it is what the
    rest of the crew (Ensembler, later holdout scoring) must use after a repair.
    Returns None rather than raising: a missing artifact is a reportable
    condition, not a crash, and the acceptance gate already rejects runs that
    fail to persist it.
    """
    workdir = getattr(result, "workdir", "") or ""
    if not workdir:
        return None
    path = Path(workdir) / ARTIFACTS_SUBDIR / "fe_source.py"
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "def add_features" not in source:
        return None
    return source


def _repair_context(dataset_key: str, cfg: dict[str, Any]) -> str:
    """What the Self-Repair specialist is told the failing script is."""
    return (
        f"The module is a training script for dataset {dataset_key!r} "
        f"(task: {cfg['task']}/{cfg['subtype']}, CV metric: {cfg['scoring']}). "
        "It loads train.parquet, applies add_features, cross-validates the "
        "candidate models, refits the best, saves model.joblib + fe_source.py, "
        "and emits metrics via crew_io.emit_metrics. All of that must still "
        "happen after your fix. The module also defines FE_SOURCE_TEXT, a string "
        "constant persisted as the fe_source.py artifact for later scoring — if "
        "your fix changes the add_features code, update FE_SOURCE_TEXT to the "
        "matching corrected source so the artifact stays consistent."
    )


def run_trainer(
    plan: dict[str, Any],
    fe_code: str,
    dataset_key: str,
    *,
    iteration: int = 0,
    param_search: Optional[bool] = None,
    timeout_s: Optional[int] = None,
    self_repair: Optional[bool] = None,
) -> dict[str, Any]:
    """Train the plan's candidates under CV and return metrics + artifact paths.

    Generates a training script from the plan and the Feature Engineer's validated
    ``fe_code``, runs it in the sandboxed executor over the train split, and reads
    back the cross-validated results. Never raises for *modeling* failures — a crash
    in the generated code is reported as ``ok: False`` with the captured error, so a
    single bad run is observable rather than fatal.

    **Self-repair (Day 20).** When the script *crashes* (non-zero exit — never a
    timeout or memory kill) and a live provider is configured, the Trainer shows
    the traceback and the full script to :func:`crewml.repair.repair_loop` and
    adopts a repaired run if one succeeds. A successful run is one the executor
    accepts *and* that emitted a ``best_cv_score`` — a "fix" that runs but stops
    reporting its number is still a failure. The returned record always carries a
    ``repair`` block (``attempted: False, reason "not_needed"`` on a clean first
    run) and ``repaired: True`` when the shipped result came from a repaired
    script, so the Critic and Reporter see the stumble, not a silent save.
    """
    param_search = _param_search_enabled(param_search)
    cfg = _training_config(plan)
    script = _build_script(fe_code, cfg, param_search)
    effective_timeout = timeout_s if timeout_s is not None else config.EXECUTOR_TIMEOUT_S

    def _run(source: str):
        return run_code(
            source,
            inputs={"train.parquet": train_path(dataset_key)},
            timeout_s=effective_timeout,
            keep_workdir=True,
        )

    result = _run(script)

    repair: dict[str, Any] = {"attempted": False, "reason_not_attempted": "not_needed"}
    repaired = False
    fe_code_used: Optional[str] = None
    if not result.ok and not repair_enabled_for_trainer(self_repair):
        repair = {"attempted": False, "reason_not_attempted": "disabled"}
    elif not result.ok:
        if is_repairable(result):

            def _run_fn(candidate: str):
                res = _run(candidate)
                # Acceptance is deliberately stricter than "exit 0". A repaired
                # script must deliver everything the downstream crew depends on:
                # the CV score, the fitted model, and the fe_source.py artifact
                # that later holdout scoring re-applies. A "fix" that drops any
                # of them is a silent corruption, not a recovery.
                metrics_ok = (res.metrics or {}).get("best_cv_score") is not None
                arts = set(res.artifacts or ())
                missing = [
                    name for name in ("model.joblib", "fe_source.py") if name not in arts
                ]
                ok = bool(res.ok and metrics_ok and not missing)
                err = res.error
                if res.ok and not metrics_ok:
                    err = "script ran but emitted no best_cv_score in metrics.json"
                elif res.ok and missing:
                    err = (
                        "script ran but did not persist required artifact(s): "
                        + ", ".join(missing)
                        + ". Keep the joblib.dump of the fitted model and the "
                        "fe_source.py write intact."
                    )
                return ok, err, res

            repair = repair_loop(
                script,
                result.error or "",
                run_fn=_run_fn,
                context=_repair_context(dataset_key, cfg),
                # Enforce the no-repairing-resource-exhaustion rule on the
                # loop's OWN runs too, not just the caller's first failure.
                not_repairable_fn=lambda res: res is not None and not is_repairable(res),
            )
            if repair["recovered"]:
                result = repair["payload"]
                repaired = True
                # The repaired script may carry a DIFFERENT add_features than the
                # one handed in — that is usually the whole point of the fix. The
                # crew must therefore adopt it: the Ensembler is called with
                # ``state["fe_code"]`` and would otherwise re-run the very code
                # that just crashed. Read back the fe_source.py the winning run
                # persisted, which is by construction the FE the model was fitted
                # with (the acceptance gate above requires the artifact to exist).
                fe_code_used = _read_persisted_fe(result)
            # The (large) source and ExecResult stay out of the state record —
            # provenance keeps the attempt trail, the workdir keeps the rest.
            repair = {k: v for k, v in repair.items() if k not in ("code", "payload")}
        else:
            repair = {
                "attempted": False,
                "reason_not_attempted": "timeout_or_oom_not_repairable",
            }

    metrics = result.metrics or {}
    training: dict[str, Any] = {
        "schema_version": TRAINER_SCHEMA_VERSION,
        "stub": False,
        "node": "trainer",
        "dataset_key": dataset_key,
        "iteration": int(iteration),
        "param_search": bool(param_search),
        "ok": bool(result.ok),
        "run_id": result.run_id,
        "duration_s": round(result.duration_s, 3),
        "timed_out": result.timed_out,
        "error": result.error,
        "artifacts": list(result.artifacts),
        "metrics": metrics,
        "repaired": repaired,
        "repair": repair,
        # Set only when a repair changed the code that produced the shipped model.
        # ``crewml.crew.nodes.trainer`` writes it back over ``state["fe_code"]`` so
        # every later consumer uses the FE the model was actually fitted with.
        "fe_code_used": fe_code_used,
        # Convenience surface for the Critic (Day 10) and reports — always a CV
        # estimate on train, never a held-out number.
        "cv_score": metrics.get("best_cv_score") if result.ok else None,
        "best_model": metrics.get("best_model") if result.ok else None,
        "cv_score_is_holdout": False,
    }
    return training
