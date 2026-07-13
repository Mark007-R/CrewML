"""Crew node stubs — Phase 2 skeleton (Day 5).

Each function here is a LangGraph node: it takes the current :class:`CrewState`
and returns a *partial* update dict that LangGraph merges into the state (list
channels are appended via their reducers, scalars overwritten). On Day 5 the
bodies are deliberately hollow — they emit clearly-labelled placeholders
(``"stub": True``) and touch neither an LLM nor the dataset. Their job today is
to prove the graph is wired: the control flow, the Critic loop, and the
``max_iterations`` guard all exercise end-to-end with these stubs in place.

Days 7-11 replace each body with a real agent, one at a time, without changing
the graph topology or the state schema:

    Profiler (Day 7) · Planner (Day 8) · Feature Engineer + Trainer (Day 9)
    Critic (Day 10) · Ensembler + Reporter (Day 11)

The one piece of *real* control logic that ships today is :func:`route_after_critic`
— the conditional edge out of the Critic — because the loop budget is a safety
property, not a stub.
"""
from __future__ import annotations

from typing import Any

from crewml.config import MAX_ITERATIONS
from crewml.crew.planner import run_planner
from crewml.crew.profiler import run_profiler
from crewml.crew.state import CrewState


def _stub(node: str, **fields: Any) -> dict[str, Any]:
    """A uniform placeholder payload so reports can see which nodes are still stubs."""
    return {"stub": True, "node": node, **fields}


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
    """Feature Engineer (stub). Day 9: generate FE code for the sandboxed executor."""
    fe_code = "# FE code placeholder — generated + executed for real on Day 9\n"
    return {"fe_code": fe_code, "trace": ["feature_engineer"]}


def trainer(state: CrewState) -> dict[str, Any]:
    """Trainer (stub). Day 9: execute FE + train with CV; return metrics + artifact paths.

    ``cv_score`` is ``None`` today on purpose: a stub must never emit a number
    that could be mistaken for a real held-out result (EVAL_PROTOCOL §5).
    """
    training = _stub(
        "trainer",
        note="TrainingResult placeholder — real CV training lands Day 9.",
        cv_score=None,
        iteration=state.get("iteration", 0),
    )
    return {"training": training, "trace": ["trainer"]}


# --- Critic + its conditional edge -----------------------------------------

def critic(state: CrewState) -> dict[str, Any]:
    """Critic (stub). Day 10: diagnose overfit / leakage / imbalance / wrong metric.

    The stub always requests another iteration — it stands in for "there is
    always something a real Critic could flag" — so that a plain skeleton run
    genuinely drives the loop and the ``max_iterations`` guard is what stops it
    (rather than the stub politely finalising on pass one). ``iteration`` counts
    completed Critic passes; :func:`route_after_critic` reads it.
    """
    it = state.get("iteration", 0) + 1
    critique = _stub(
        "critic",
        iteration=it,
        decision="iterate",
        findings=["placeholder diagnosis — real critique lands Day 10"],
    )
    return {
        "iteration": it,
        "decision": "iterate",
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
    """Ensembler (stub). Day 11: combine the best models from the run's trials."""
    ensemble = _stub(
        "ensembler",
        note="Ensemble placeholder — real model combination lands Day 11.",
        n_iterations_run=state.get("iteration", 0),
    )
    return {"ensemble": ensemble, "trace": ["ensembler"]}


def reporter(state: CrewState) -> dict[str, Any]:
    """Reporter (stub). Day 11: write the final report + MODEL_CARD.md.

    Terminal node — emits a summary of the run's control flow so the skeleton has
    an observable end product even before any real modeling exists.
    """
    report = _stub(
        "reporter",
        note="Final report placeholder — real report + model card land Day 11.",
        dataset_key=state["dataset_key"],
        iterations_run=state.get("iteration", 0),
        n_critiques=len(state.get("critiques") or []),
        stub_run=True,
    )
    return {"report": report, "trace": ["reporter"]}
