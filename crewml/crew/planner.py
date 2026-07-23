"""The Planner agent — turns a DataProfile into a ModelingPlan (Day 8).

The Planner is the crew's second real node. It never touches the data: it reasons
purely over the **DataProfile** the Profiler (Day 7) computed from the ``train``
split, and emits a structured, JSON-friendly **ModelingPlan** that the Feature
Engineer (Day 9) and Trainer (Day 9) execute. Same honesty discipline as the
Profiler:

* **Deterministic core.** :func:`build_plan` derives every decision — which
  columns to drop, how to preprocess numeric vs. categorical features, which model
  families to try, the cross-validation scheme, and the imbalance strategy — with
  plain rules over the profile. No LLM chooses a hyperparameter, so the plan is
  reproducible and cannot hallucinate a model that doesn't exist.
* **Critic-aware.** On a Critic-triggered re-entry (Day 10) the Planner consumes
  the latest critique and *adjusts* the plan — regularise on overfit, drop more on
  leakage, force class weights on imbalance. On the first pass there is no critique
  and the plan is built from the profile alone.
* **Optional LLM narrative.** When a live provider is configured, :func:`run_planner`
  layers a short advisory refinement note *on top of* — never in place of — the
  deterministic plan, tagged with provider/model/token cost. In mock mode (or on any
  error) the narrative is ``unavailable`` and the plan stands on its deterministic
  core. The narrative never overwrites a chosen value.

**Train only, structurally.** The Planner reads a dict (the profile) and nothing
else — no data loader, no held-out split. A source-inspection test asserts the
module never names the locked test split; the no-peeking invariant is a property of
the code, not a rule a node has to remember.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from crewml import config, llm

PLAN_SCHEMA_VERSION = 1

# --- Deterministic heuristics (kept conservative; mirror the Profiler's) -----
DEFAULT_CV_SPLITS = 5          # folds when the data comfortably supports them
MIN_CV_SPLITS = 2             # never fewer than this
HIGH_CARDINALITY_FRAC = 0.5   # categorical n_unique / n_rows above this => encode compactly

# sklearn scorer string per primary metric (Trainer passes these to cross_val_score).
_SCORER_FOR_METRIC = {
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "r2": "r2",
}


# --- Column disposition (what to drop, and why) -----------------------------

def _drop_columns(profile: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Decide which columns to drop before modeling, with a reason for each.

    Draws only on the Profiler's deterministic leakage checks: constant columns,
    identifier-like near-unique columns, near-perfect single-feature predictors of
    the target (leakage suspects), and all-but-one of each duplicate-column group.
    """
    leak = profile["leakage_checks"]
    reasons: dict[str, str] = {}

    for c in leak.get("constant_columns", []):
        reasons.setdefault(c, "constant (zero-variance) — carries no signal")
    for c in leak.get("id_like_columns", []):
        reasons.setdefault(c, "identifier-like (near-unique) — memorises rows, no generalisation")
    for d in leak.get("target_correlated_features", []):
        reasons.setdefault(d["column"], f"target-leakage suspect ({d['measure']}={d['signal']}) — near-perfectly predicts target")
    # Duplicate groups: keep the first column, drop the rest as redundant.
    for group in leak.get("duplicate_feature_columns", []):
        for c in group[1:]:
            reasons.setdefault(c, f"exact duplicate of '{group[0]}' — redundant")

    drops = sorted(reasons)
    return drops, [{"column": c, "reason": reasons[c]} for c in drops]


# --- Preprocessing plan ------------------------------------------------------

def _preprocessing(profile: dict[str, Any], drop: set[str]) -> dict[str, Any]:
    """Build a dtype-aware preprocessing spec the Trainer realises as a ColumnTransformer."""
    feats = profile["features"]
    numeric = [c for c in profile["columns"]["numeric"] if c not in drop]
    categorical = [c for c in profile["columns"]["categorical"] if c not in drop]

    # Zero-inflated numeric columns the Profiler flagged as *maybe* disguised-missing.
    # A plan is advice, not a verdict: recommend treating 0 as NaN before imputing,
    # but carry the heuristic caveat forward (some zeros are legitimate).
    zero_as_missing = [
        d["column"] for d in profile["leakage_checks"].get("suspected_disguised_missing", [])
        if d["column"] in numeric
    ]

    # Split categoricals by cardinality: one-hot the low-cardinality ones (dense,
    # interpretable); encode the high-cardinality ones compactly (ordinal) so the
    # feature space doesn't explode.
    high_card = [
        c for c in categorical
        if (feats[c].get("unique_frac") or 0) >= HIGH_CARDINALITY_FRAC
    ]
    low_card = [c for c in categorical if c not in high_card]

    return {
        "numeric": {
            "columns": numeric,
            "impute": "median",
            "scale": "standard",  # available; Trainer applies it only for scale-sensitive models
            "zero_as_missing": zero_as_missing,
            "zero_as_missing_is_heuristic": bool(zero_as_missing),
        },
        "categorical": {
            "columns": categorical,
            "impute": "most_frequent",
            "onehot_columns": low_card,
            "onehot_params": {"handle_unknown": "ignore"},
            "ordinal_columns": high_card,  # high-cardinality => ordinal to avoid OHE blow-up
        },
        "drop_columns": sorted(drop),
    }


# --- Candidate models --------------------------------------------------------

def _candidate_models(task: str, subtype: str) -> list[dict[str, Any]]:
    """Task-appropriate model families to try, ordered strong-first, with seed grids.

    Grids are deliberately small — the Trainer (Day 9) searches them under a CV
    budget. Prefix ``model__`` matches a Pipeline whose estimator step is ``model``.
    """
    if task == "classification":
        return [
            {
                "name": "hist_gradient_boosting",
                "estimator": "HistGradientBoostingClassifier",
                "family": "tree", "needs_scaling": False,
                "supports_proba": True, "supports_class_weight": False,
                "param_grid": {
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_iter": [200, 400],
                    "model__max_leaf_nodes": [31, 63],
                },
                "rationale": "Strong default on tabular data; handles mixed scales and interactions without tuning.",
            },
            {
                "name": "random_forest",
                "estimator": "RandomForestClassifier",
                "family": "tree", "needs_scaling": False,
                "supports_proba": True, "supports_class_weight": True,
                "param_grid": {
                    "model__n_estimators": [300, 600],
                    "model__max_depth": [None, 20],
                    "model__min_samples_leaf": [1, 4],
                },
                "rationale": "Low-variance bagged trees; supports class_weight='balanced' for skew.",
            },
            {
                "name": "logistic_regression",
                "estimator": "LogisticRegression",
                "family": "linear", "needs_scaling": True,
                "supports_proba": True, "supports_class_weight": True,
                "param_grid": {"model__C": [0.1, 1.0, 10.0]},
                "rationale": "Calibrated linear baseline; fast, interpretable, class_weight-capable.",
            },
        ]
    # regression
    return [
        {
            "name": "hist_gradient_boosting",
            "estimator": "HistGradientBoostingRegressor",
            "family": "tree", "needs_scaling": False,
            "supports_proba": False, "supports_class_weight": False,
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__max_leaf_nodes": [31, 63],
            },
            "rationale": "Strong default for non-linear tabular regression; robust to feature scale.",
        },
        {
            "name": "random_forest",
            "estimator": "RandomForestRegressor",
            "family": "tree", "needs_scaling": False,
            "supports_proba": False, "supports_class_weight": False,
            "param_grid": {
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 4],
            },
            "rationale": "Low-variance ensemble; captures interactions the linear model misses.",
        },
        {
            "name": "ridge",
            "estimator": "Ridge",
            "family": "linear", "needs_scaling": True,
            "supports_proba": False, "supports_class_weight": False,
            "param_grid": {"model__alpha": [0.1, 1.0, 10.0]},
            "rationale": "Regularised linear baseline; cheap reference that exposes non-linearity gains.",
        },
    ]


# --- Cross-validation scheme -------------------------------------------------

def _cv_scheme(profile: dict[str, Any]) -> dict[str, Any]:
    """Choose the CV splitter + scorer. Stratified for classification, plain KFold for regression."""
    task = profile["task"]
    metric = profile["metric"]
    scoring = _SCORER_FOR_METRIC.get(metric, metric)

    n_splits = DEFAULT_CV_SPLITS
    if task == "classification":
        # Never ask for more folds than the rarest class has members.
        classes = profile["target"].get("classes", {})
        min_class = min(classes.values()) if classes else DEFAULT_CV_SPLITS
        n_splits = max(MIN_CV_SPLITS, min(DEFAULT_CV_SPLITS, int(min_class)))
        splitter = "StratifiedKFold"
    else:
        splitter = "KFold"

    return {
        "scheme": splitter,
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": config.SEED,
        "scoring": scoring,
    }


# --- Imbalance strategy ------------------------------------------------------

def _imbalance_strategy(profile: dict[str, Any]) -> dict[str, Any]:
    """Recommend how to handle class skew — only for classification, only when flagged."""
    if profile["task"] != "classification":
        return {"recommended": False, "reason": "regression — no class balance to manage"}
    if "class_imbalance" not in profile["assessment"]["flags"]:
        return {"recommended": False, "reason": "classes are roughly balanced"}
    tgt = profile["target"]
    return {
        "recommended": True,
        "method": "class_weight='balanced'",
        "apply_to": "models that support class_weight (RandomForest, LogisticRegression)",
        "use_stratified_cv": True,
        "positive_class": tgt.get("positive_class"),
        "imbalance_ratio": tgt.get("imbalance_ratio"),
        "note": (
            f"Rarer class '{tgt.get('minority_class')}' is the positive class; the primary metric "
            f"'{profile['metric']}' is threshold-independent, so optimise ranking rather than a 0.5 cutoff."
        ),
    }


# --- Critic-triggered adjustments (the loop's response) ---------------------

def _apply_critique(plan: dict[str, Any], critique: Optional[dict[str, Any]]) -> None:
    """Mutate ``plan`` in place to address the latest critique.

    Day 10 ships the real Critic; this is the Planner's *response* side of the loop,
    wired now so the loop is functional the moment the Critic produces findings. It
    keys off substrings in the critique's ``findings`` so a structured or a prose
    critique both land. On the first pass ``critique`` is ``None`` and nothing here
    runs.
    """
    plan["addressed_critique"] = critique
    if not critique:
        return
    findings = " ".join(str(f) for f in (critique.get("findings") or [])).lower()
    adjustments: list[str] = []

    if "overfit" in findings:
        # Pull complexity down and regularisation up across the seed grids.
        for m in plan["candidate_models"]:
            g = m["param_grid"]
            if "model__max_iter" in g:
                g["model__max_iter"] = [150, 250]
            if "model__min_samples_leaf" in g:
                g["model__min_samples_leaf"] = [4, 8]
            if "model__max_depth" in g:
                g["model__max_depth"] = [10, 20]
            if "model__C" in g:
                g["model__C"] = [0.01, 0.1, 1.0]      # stronger L2
            if "model__alpha" in g:
                g["model__alpha"] = [1.0, 10.0, 100.0]  # stronger L2
        adjustments.append("overfit flagged — reduced model capacity and strengthened regularisation in the grids")

    if "underfit" in findings:
        for m in plan["candidate_models"]:
            g = m["param_grid"]
            if "model__max_iter" in g:
                g["model__max_iter"] = [400, 800]
            if "model__max_leaf_nodes" in g:
                g["model__max_leaf_nodes"] = [63, 127]
        adjustments.append("underfit flagged — increased model capacity in the grids")

    if "leak" in findings:
        adjustments.append("leakage flagged — re-audit dropped columns and any engineered feature derived from the target")

    if "imbalance" in findings:
        plan["imbalance_strategy"]["recommended"] = True
        adjustments.append("imbalance flagged — force class_weight='balanced' and stratified CV")

    if "metric" in findings:
        adjustments.append("wrong-metric flagged — confirm CV scoring matches the primary metric")

    if not adjustments:
        adjustments.append("critique noted — no structured directive matched; carrying the profile-driven plan forward")

    plan["critique_adjustments"] = adjustments


# --- Ablation instrumentation (Day 13) — OFF unless explicitly enabled -------

def _apply_ablation_handicap(plan: dict[str, Any], iteration: int) -> None:
    """Cripple the FIRST pass's model capacity — ablation instrumentation only.

    Gated behind ``CREWML_ABLATION_HANDICAP`` (default ``"0"`` — a normal run never
    touches this). It exists for one purpose: the Critic loop is a *conditional*
    safeguard, so on healthy datasets the Critic correctly finalises on pass 1 and the
    loop never fires — which makes its contribution invisible to measure. This hook
    collapses every capacity knob on the first pass to a near-stump so the winning CV
    score falls at/below the Critic's underfit floor. Then the loop's value becomes
    observable: *with* the Critic, an ``underfit`` finding fires and the Planner restores
    capacity on the next pass (:func:`_apply_critique`); *without* it (the ``no_critic``
    variant), the crew ships the crippled model. The gap between the two is the loop's
    contribution, cleanly attributable.

    Only the first pass (``iteration == 0``) is handicapped — recovery on later passes is
    exactly the loop's job, not something this hook grants for free.
    """
    if os.getenv("CREWML_ABLATION_HANDICAP", "0") == "0":
        return
    if int(iteration) != 0:
        return
    for m in plan["candidate_models"]:
        g = m["param_grid"]
        if "model__max_iter" in g:
            g["model__max_iter"] = [1]
        if "model__max_leaf_nodes" in g:
            g["model__max_leaf_nodes"] = [2]
        if "model__learning_rate" in g:
            g["model__learning_rate"] = [0.01]
        if "model__max_depth" in g:
            g["model__max_depth"] = [1]
        if "model__n_estimators" in g:
            g["model__n_estimators"] = [1]
        if "model__min_samples_leaf" in g:
            # Larger than any split can honour -> the tree cannot branch, so even a
            # depth-1 forest collapses to a near-constant predictor (a lone stump can
            # otherwise clear the underfit floor on its own on easy regression sets).
            g["model__min_samples_leaf"] = [10_000_000]
        if "model__C" in g:
            g["model__C"] = [1e-4]     # near-zero-capacity linear
        if "model__alpha" in g:
            g["model__alpha"] = [1e6]  # extreme shrinkage -> underfit
    plan["ablation_handicap"] = (
        "first-pass capacity capped to a near-stump (CREWML_ABLATION_HANDICAP=1); "
        "ablation instrumentation, not a production setting"
    )


# --- Public: build the deterministic plan -----------------------------------

def build_plan(
    profile: dict[str, Any],
    *,
    critique: Optional[dict[str, Any]] = None,
    iteration: int = 0,
) -> dict[str, Any]:
    """Compute the full deterministic ModelingPlan from a DataProfile.

    Pure and reproducible: reads only the ``profile`` dict, calls no LLM and no
    data loader. The returned dict is JSON-serialisable so it can live in
    :class:`~crewml.crew.state.CrewState`.
    """
    task = profile["task"]
    subtype = profile["subtype"]

    drop_list, drop_reasons = _drop_columns(profile)
    drop_set = set(drop_list)

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "stub": False,
        "node": "planner",
        "dataset_key": profile["dataset_key"],
        "task": task,
        "subtype": subtype,
        "metric": profile["metric"],
        "planning_for_iteration": int(iteration),
        "drop_columns": drop_list,
        "drop_reasons": drop_reasons,
        "preprocessing": _preprocessing(profile, drop_set),
        "candidate_models": _candidate_models(task, subtype),
        "cv": _cv_scheme(profile),
        "imbalance_strategy": _imbalance_strategy(profile),
    }
    plan["recommended_primary_model"] = plan["candidate_models"][0]["name"]
    plan["rationale"] = _rationale(profile, plan)
    _apply_ablation_handicap(plan, iteration)  # no-op unless CREWML_ABLATION_HANDICAP=1
    _apply_critique(plan, critique)  # no-op on the first pass
    return plan


def _rationale(profile: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """A short, human-readable justification of the plan's key choices."""
    pre = plan["preprocessing"]
    notes = [
        f"Task {profile['task']} ({profile['subtype']}); optimise {profile['metric']} "
        f"via {plan['cv']['scheme']}({plan['cv']['n_splits']}).",
        f"{len(pre['numeric']['columns'])} numeric + {len(pre['categorical']['columns'])} categorical feature(s) "
        f"after dropping {len(plan['drop_columns'])}.",
    ]
    if plan["drop_columns"]:
        notes.append(f"Dropping {plan['drop_columns']} on leakage/integrity grounds.")
    if pre["numeric"]["zero_as_missing"]:
        notes.append(
            f"Treating zeros as missing (heuristic) in {pre['numeric']['zero_as_missing']} before median imputation."
        )
    if pre["categorical"]["ordinal_columns"]:
        notes.append(f"High-cardinality {pre['categorical']['ordinal_columns']} ordinal-encoded to avoid one-hot blow-up.")
    if plan["imbalance_strategy"]["recommended"]:
        notes.append("Class imbalance present — class_weight='balanced' + stratified CV recommended.")
    notes.append(
        "Candidate models: " + ", ".join(m["name"] for m in plan["candidate_models"])
        + f" (start with {plan['recommended_primary_model']})."
    )
    return notes


# --- Ablation stand-in (Day 14) — the plan a crew with NO Planner would run --

def build_naive_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """The profile-blind plan the ``no_planner`` ablation variant runs (Day 14).

    The Planner cannot be deleted the way the Critic could: the Trainer needs *a*
    plan to execute at all, so removing the specialist means replacing it with the
    naive floor — what a crew with no planning expertise would do. This plan reads
    only the bare schema facts no pipeline can run without (which columns are
    numeric vs. categorical, the task/metric, and the protocol's positive class —
    a property of the eval protocol, not a planning decision) and ignores
    everything the real Planner reasons over:

    * **No leakage screen.** Nothing is dropped — id-like columns, constants,
      duplicates and target-leakage suspects all ride into the model.
    * **No cardinality awareness.** Every categorical is one-hot encoded, however
      wide that makes the feature space.
    * **No disguised-missing handling.** Zeros stay zeros.
    * **No imbalance strategy.** Class skew is never assessed, so no class weights.
    * **One default model, no search.** A single RandomForest with library
      defaults (the empty grid disables the Trainer's search) instead of the
      Planner's ordered families with seed grids.
    * **Critique-deaf.** The critique parameter is deliberately absent: on a
      Critic-triggered re-entry this function rebuilds the identical plan, so the
      loop has no actuator — measuring exactly that is part of the ablation.

    Keeps :data:`PLAN_SCHEMA_VERSION`'s shape so the Trainer, Critic, Ensembler
    and Reporter consume it unchanged — the variant differs in plan *content* only.
    """
    task = profile["task"]
    numeric = list(profile["columns"]["numeric"])
    categorical = list(profile["columns"]["categorical"])

    estimator = "RandomForestClassifier" if task == "classification" else "RandomForestRegressor"
    candidate = {
        "name": "random_forest",
        "estimator": estimator,
        "family": "tree", "needs_scaling": False,
        "supports_proba": task == "classification",
        "supports_class_weight": task == "classification",
        "param_grid": {},  # empty grid => the Trainer skips its search entirely
        "rationale": "naive floor: one library-default model, no profile-driven choice, no search",
    }

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "stub": False,
        "node": "planner_naive",
        "ablated": "planner",
        "dataset_key": profile["dataset_key"],
        "task": task,
        "subtype": profile["subtype"],
        "metric": profile["metric"],
        "planning_for_iteration": 0,
        "drop_columns": [],
        "drop_reasons": [],
        "preprocessing": {
            "numeric": {
                "columns": numeric,
                "impute": "median",
                "scale": "standard",
                "zero_as_missing": [],
                "zero_as_missing_is_heuristic": False,
            },
            "categorical": {
                "columns": categorical,
                "impute": "most_frequent",
                "onehot_columns": categorical,  # cardinality-blind: one-hot everything
                "onehot_params": {"handle_unknown": "ignore"},
                "ordinal_columns": [],
            },
            "drop_columns": [],
        },
        "candidate_models": [candidate],
        "cv": {
            "scheme": "StratifiedKFold" if task == "classification" else "KFold",
            "n_splits": DEFAULT_CV_SPLITS,  # fixed; never clamped to the rarest class
            "shuffle": True,
            "random_state": config.SEED,
            "scoring": _SCORER_FOR_METRIC.get(profile["metric"], profile["metric"]),
        },
        "imbalance_strategy": {
            "recommended": False,
            "reason": "no planner — class balance never assessed",
            # The protocol's positive class is part of the task definition (it keeps
            # the 0/1 label mapping — and therefore held-out ROC AUC — identical
            # across arms); carrying it is NOT a planning decision.
            "positive_class": (profile.get("target") or {}).get("positive_class"),
        },
        "recommended_primary_model": "random_forest",
        "rationale": [
            "ABLATION (no_planner): profile-blind naive plan — no leakage drops, "
            "one-hot for every categorical, no imbalance handling, a single "
            "library-default RandomForest with no search, critique-deaf on re-entry.",
        ],
    }
    plan["llm_narrative"] = {
        "source": "unavailable", "is_mock": config.is_mock_mode(),
        "reason": "ablated — the no_planner variant has no planning specialist", "text": None,
    }
    return plan


# --- Optional LLM narrative (advisory, never a source of decisions) ----------

_NARRATIVE_SYSTEM = (
    "You are the Planner agent in a multi-agent ML crew. You receive a DETERMINISTIC "
    "modeling plan already derived from a train-only data profile: which columns to "
    "drop, the preprocessing, the candidate models with seed grids, the CV scheme, "
    "and the imbalance strategy. Do NOT restate the plan. In <=160 words, give the "
    "Feature Engineer and Trainer 3-5 CONCRETE refinements specific to THIS dataset: "
    "feature ideas worth trying, a preprocessing risk to watch, or which candidate "
    "you'd prioritise and why. Do not invent columns not in the plan. Plain prose, no code."
)


def _narrative_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """A compact, token-light view of the plan for the LLM prompt."""
    pre = plan["preprocessing"]
    return {
        "dataset_key": plan["dataset_key"],
        "task": plan["task"],
        "subtype": plan["subtype"],
        "metric": plan["metric"],
        "n_numeric": len(pre["numeric"]["columns"]),
        "n_categorical": len(pre["categorical"]["columns"]),
        "drop_columns": plan["drop_columns"],
        "zero_as_missing": pre["numeric"]["zero_as_missing"],
        "ordinal_columns": pre["categorical"]["ordinal_columns"],
        "candidate_models": [m["name"] for m in plan["candidate_models"]],
        "cv": plan["cv"],
        "imbalance_strategy": plan["imbalance_strategy"],
    }


def _llm_narrative(plan: dict[str, Any]) -> dict[str, Any]:
    """Ask the live provider for a short FE/Trainer briefing. Never raises."""
    try:
        result = llm.chat(
            _NARRATIVE_SYSTEM,
            "Modeling plan (JSON):\n" + json.dumps(_narrative_payload(plan), default=str),
            temperature=0.0,
            max_tokens=512,
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
    """Explicit flag wins; else the ``CREWML_PLANNER_LLM`` env toggle (default on)."""
    if with_llm is not None:
        return with_llm
    return os.getenv("CREWML_PLANNER_LLM", "1") != "0"


def run_planner(
    profile: dict[str, Any],
    *,
    critique: Optional[dict[str, Any]] = None,
    iteration: int = 0,
    with_llm: Optional[bool] = None,
) -> dict[str, Any]:
    """Build the ModelingPlan for a profile and attach an optional advisory narrative.

    The deterministic plan is always computed. An LLM refinement note is attached
    only when enabled *and* a live provider is configured; otherwise the plan records
    the narrative as ``unavailable`` and stands on its deterministic core.
    """
    plan = build_plan(profile, critique=critique, iteration=iteration)

    if _llm_enabled(with_llm) and not config.is_mock_mode():
        plan["llm_narrative"] = _llm_narrative(plan)
    else:
        reason = "mock_mode" if config.is_mock_mode() else "disabled"
        plan["llm_narrative"] = {
            "source": "unavailable", "is_mock": config.is_mock_mode(),
            "reason": reason, "text": None,
        }
    return plan
