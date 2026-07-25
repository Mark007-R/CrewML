"""Crew node stubs — Phase 2 skeleton (Day 5).

Each function here is a LangGraph node: it takes the current :class:`CrewState`
and returns a *partial* update dict that LangGraph merges into the state (list
channels are appended via their reducers, scalars overwritten). On Day 5 the
bodies are deliberately hollow — they emit clearly-labelled placeholders
(``"stub": True``) and touch neither an LLM nor the dataset. Their job today is
to prove the graph is wired: the control flow, the Critic loop, and the
``max_iterations`` guard all exercise end-to-end with these stubs in place.

Days 7-11 replaced each body with a real agent, one at a time, without changing
the graph topology or the state schema:

    Profiler (Day 7) · Planner (Day 8) · Feature Engineer + Trainer (Day 9)
    Critic (Day 10) · Ensembler + Reporter (Day 11)

As of Day 11 every node is real — no stubs remain. The one piece of *real* control
logic that shipped on Day 5 is :func:`route_after_critic` — the conditional edge out
of the Critic — because the loop budget is a safety property.
"""
from __future__ import annotations

from typing import Any

from crewml.config import MAX_ITERATIONS
from crewml.crew.critic import run_critic
from crewml.crew.ensembler import run_ensembler
from crewml.crew.feature_engineer import run_feature_engineer, run_identity_fe
from crewml.crew.planner import build_naive_plan as run_naive_planner
from crewml.crew.planner import run_planner
from crewml.crew.profiler import run_profiler
from crewml.crew.reporter import run_reporter
from crewml.crew.state import CrewState
from crewml.crew.trainer import run_trainer


# --- Linear front half: Profiler -> Planner -> Feature Engineer -> Trainer ---

def profiler(state: CrewState) -> dict[str, Any]:
    """Profiler (REAL — Day 7). Train-only EDA -> structured DataProfile.

    The first stub retired: computes schema, dtypes, missingness (incl. suspected
    disguised-missing zeros), the target distribution + class imbalance, and basic
    leakage checks — all deterministically, with an optional advisory LLM briefing
    layered on top (see :mod:`crewml.crew.profiler`). Reads only the ``train``
    split; the profile it returns is what the Planner (Day 8) reasons over.
    """
    profile = run_profiler(state["dataset_key"])
    return {"profile": profile, "trace": ["profiler"]}


def planner(state: CrewState) -> dict[str, Any]:
    """Planner (REAL — Day 8). Read the DataProfile -> structured ModelingPlan.

    The second stub retired: reasons purely over the profile the Profiler produced
    (never the data) to decide column drops, dtype-aware preprocessing, candidate
    model families with seed grids, the CV scheme, and the imbalance strategy — all
    deterministically, with an optional advisory LLM briefing layered on top (see
    :mod:`crewml.crew.planner`). On a Critic-triggered re-entry it consumes the latest
    critique and adjusts the plan; on the first pass ``critiques`` is empty and the
    plan is built from the profile alone. Feeds the Feature Engineer + Trainer (Day 9).
    """
    critiques = state.get("critiques") or []
    plan = run_planner(
        state["profile"],
        critique=critiques[-1] if critiques else None,
        iteration=state.get("iteration", 0),
    )
    return {"plan": plan, "trace": ["planner"]}


def feature_engineer(state: CrewState) -> dict[str, Any]:
    """Feature Engineer (REAL — Day 9). Generate + sandbox-validate ``add_features`` code.

    The third stub retired: reads the Planner's ModelingPlan and produces a validated
    row-wise, leakage-free ``add_features(df)`` module (see
    :mod:`crewml.crew.feature_engineer`). When a live provider is configured it asks
    for dataset-specific code and trusts it *only* after the sandbox confirms it honours
    the contract; otherwise (mock mode, LLM disabled, or a failed generation) it falls
    back to the deterministic default. The chosen source feeds the Trainer.
    """
    result = run_feature_engineer(state["plan"], state["dataset_key"])
    return {"fe_code": result["code"], "fe_meta": result["meta"], "trace": ["feature_engineer"]}


def trainer(state: CrewState) -> dict[str, Any]:
    """Trainer (REAL — Day 9). Execute FE + train the candidates under CV; return metrics.

    The fourth stub retired: assembles a training script from the plan and the Feature
    Engineer's validated code, runs it in the Day-6 sandboxed executor over the train
    split, and returns cross-validated metrics + a saved model artifact (see
    :mod:`crewml.crew.trainer`). The number it surfaces is a **CV estimate on train**,
    never a held-out score — held-out scoring is a separate later step.
    """
    training = run_trainer(
        state["plan"],
        state["fe_code"],
        state["dataset_key"],
        iteration=state.get("iteration", 0),
    )
    update: dict[str, Any] = {"training": training, "trace": ["trainer"]}
    # A self-repair (Day 20) may have rewritten add_features to get the run to
    # complete. The Ensembler is called with state["fe_code"], so without this
    # write-back it would re-run the exact code that just crashed — and any later
    # holdout scoring would re-apply an FE the shipped model was never fitted
    # with. Adopt the FE the winning run actually persisted.
    if training.get("fe_code_used"):
        update["fe_code"] = training["fe_code_used"]
    return update


# --- Ablation stand-ins (Day 14) — naive floors, not smarter alternatives ---

def naive_planner(state: CrewState) -> dict[str, Any]:
    """Planner stand-in for the ``no_planner`` ablation variant (Day 14).

    Emits :func:`crewml.crew.planner.build_naive_plan` — the profile-blind naive
    floor — under the same state key and trace name as the real Planner, so the
    topology (including the Critic's loop edge back to this node) is untouched.
    Deliberately ignores ``critiques``: without a planning specialist the loop has
    no actuator, and on a re-entry this node rebuilds the identical plan.
    """
    plan = run_naive_planner(state["profile"])
    return {"plan": plan, "trace": ["planner"]}


def identity_feature_engineer(state: CrewState) -> dict[str, Any]:
    """Feature Engineer stand-in for the ``no_feature_engineer`` variant (Day 14).

    Ships the identity ``add_features`` (raw features only, no LLM, no default
    engineered column) via :func:`crewml.crew.feature_engineer.run_identity_fe`,
    under the same state keys and trace name as the real node.
    """
    result = run_identity_fe(state["dataset_key"])
    return {"fe_code": result["code"], "fe_meta": result["meta"], "trace": ["feature_engineer"]}


# --- Critic + its conditional edge -----------------------------------------

def critic(state: CrewState) -> dict[str, Any]:
    """Critic (REAL — Day 10). Diagnose the training pass, decide iterate vs finalize.

    The fifth stub retired, and the node that closes the loop: reads the Trainer's
    cross-validated result together with the profile and the plan, diagnoses the
    failure modes a competent reviewer looks for (overfit / underfit / leakage /
    imbalance / wrong-metric), and decides whether another pass earns its keep (see
    :mod:`crewml.crew.critic`). Its findings embed the keywords the Planner's
    ``_apply_critique`` acts on, so an ``iterate`` decision turns into a concrete plan
    change on re-entry. ``iteration`` counts completed Critic passes;
    :func:`route_after_critic` reads it and applies the hard ``max_iterations`` guard.
    """
    it = state.get("iteration", 0) + 1
    critique = run_critic(
        state["profile"],
        state["plan"],
        state["training"],
        critiques_so_far=state.get("critiques") or [],
        iteration=it,
        max_iterations=int(state.get("max_iterations", MAX_ITERATIONS)),
    )
    return {
        "iteration": it,
        "decision": critique["decision"],
        "critiques": [critique],
        "trace": ["critic"],
    }


def route_after_critic(state: CrewState) -> str:
    """Conditional edge out of the Critic: loop back to Planner, or finalise.

    Real, shipping logic (not a stub) — the loop budget is a safety property:

    1. **Guard first.** Once ``iteration`` reaches ``max_iterations`` we finalise
       no matter what the Critic wants, so the loop can never run away.
    2. Otherwise honour the Critic's ``decision`` — ``"iterate"`` returns to the
       Planner with the new critique in hand; anything else finalises.
    """
    max_iters = int(state.get("max_iterations", MAX_ITERATIONS))
    if int(state.get("iteration", 0)) >= max_iters:
        return "finalize"
    if state.get("decision") == "iterate":
        return "iterate"
    return "finalize"


# --- Finalise: Ensembler -> Reporter ---------------------------------------

def ensembler(state: CrewState) -> dict[str, Any]:
    """Ensembler (REAL — Day 11). Combine the top candidates; keep only if it beats the best.

    The sixth stub retired: builds a soft-voting (classification) / averaging (regression)
    ensemble over the top CV-ranked candidates — each with the Trainer's best params — and
    cross-validates it against the single best model on the *same* seeded folds inside the
    sandbox (see :mod:`crewml.crew.ensembler`). The crew ships whichever scores higher, so
    the final model is never worse than the Trainer already had. Every number is a CV
    estimate on train; the held-out split is untouched. On a failed/too-thin run it records an
    honest "not attempted" and keeps the Trainer's model.
    """
    ensemble = run_ensembler(
        state["plan"],
        state["training"],
        state["fe_code"],
        state["dataset_key"],
        iteration=state.get("iteration", 0),
    )
    return {"ensemble": ensemble, "trace": ["ensembler"]}


def reporter(state: CrewState) -> dict[str, Any]:
    """Reporter (REAL — Day 11). Synthesise the final report + write MODEL_CARD.md.

    The last stub retired, and the crew's terminal node: reads the full final state
    (profile, plan, FE, training, critiques, ensemble) and renders a structured report
    plus a model card — deterministically, no LLM (see :mod:`crewml.crew.reporter`). It
    surfaces the honesty caveats (scores are CV-on-train not held-out, any degraded/mock
    narratives, a training failure) so a reader can't miss them, and writes the card +
    a JSON copy to the run's git-ignored artifact dir.
    """
    report = run_reporter(state)
    return {"report": report, "trace": ["reporter"]}
