"""Day 5 guards: the Phase-2 crew skeleton is wired, bounded, and honest.

The node bodies are stubs, so these tests assert *control flow*, not model
quality:

  * the graph compiles and contains exactly the seven crew nodes;
  * the conditional Critic router honours the decision AND the ``max_iterations``
    guard (the guard wins even when the Critic wants to iterate);
  * a full skeleton invocation drives the loop to its budget, finalises, and
    terminates at the Reporter with a growing critique/trace history;
  * nothing a stub emits could pass for a real result (no numeric held-out score,
    everything flagged ``stub``); the held-out set is never referenced.
"""
from __future__ import annotations

import inspect

import pytest

from crewml.config import MAX_ITERATIONS
from crewml.crew import (
    CREW_NODES,
    CrewState,
    build_crew,
    build_graph,
    initial_state,
    route_after_critic,
)
from crewml.crew import nodes as crew_nodes
from crewml.datasets import REGISTRY

SPEC = REGISTRY["credit-g"]


# --- Topology ---------------------------------------------------------------

def test_expected_seven_nodes():
    assert CREW_NODES == (
        "profiler", "planner", "feature_engineer",
        "trainer", "critic", "ensembler", "reporter",
    )


def test_graph_compiles_with_all_nodes():
    app = build_crew()
    graph_nodes = set(app.get_graph().nodes)
    for name in CREW_NODES:
        assert name in graph_nodes
    # START and END are present as the entry/exit sentinels.
    assert "__start__" in graph_nodes and "__end__" in graph_nodes


def test_build_graph_returns_uncompiled_and_compiles():
    g = build_graph()
    # A second, independent compile must also succeed (no shared mutable state).
    assert g.compile() is not None


# --- The conditional Critic router (the one piece of real logic) ------------

def _state(iteration: int, decision: str | None, max_iterations: int = 3) -> CrewState:
    return CrewState(iteration=iteration, decision=decision, max_iterations=max_iterations)


def test_router_iterates_when_asked_and_under_budget():
    assert route_after_critic(_state(1, "iterate", max_iterations=3)) == "iterate"


def test_router_finalizes_when_critic_declines():
    assert route_after_critic(_state(1, "finalize", max_iterations=3)) == "finalize"
    assert route_after_critic(_state(1, None, max_iterations=3)) == "finalize"


def test_guard_forces_finalize_at_budget_even_if_critic_wants_more():
    # Critic still says "iterate", but the budget is spent -> guard wins.
    assert route_after_critic(_state(3, "iterate", max_iterations=3)) == "finalize"
    assert route_after_critic(_state(4, "iterate", max_iterations=3)) == "finalize"


def test_guard_defaults_to_config_when_unset():
    st = CrewState(iteration=MAX_ITERATIONS, decision="iterate")
    assert route_after_critic(st) == "finalize"


# --- Full skeleton run (stubs, no LLM / no data) ----------------------------

@pytest.fixture(scope="module")
def final_state():
    # The Profiler (Day 7) and Planner (Day 8) are real; keep the wiring test
    # offline + free by disabling their advisory LLM narratives (the deterministic
    # facts and plan are still computed in full).
    mp = pytest.MonkeyPatch()
    mp.setenv("CREWML_PROFILER_LLM", "0")
    mp.setenv("CREWML_PLANNER_LLM", "0")
    app = build_crew()
    st = initial_state(SPEC, max_iterations=3)
    final = app.invoke(st, config={"recursion_limit": 50})
    mp.undo()
    return final


def test_run_terminates_at_reporter(final_state):
    assert final_state["trace"][-1] == "reporter"
    assert final_state["report"] is not None


def test_run_spends_full_budget_then_finalizes(final_state):
    # Stub Critic always iterates, so the loop runs exactly max_iterations passes.
    assert final_state["iteration"] == 3
    assert len(final_state["critiques"]) == 3


def test_trace_is_the_expected_looped_sequence(final_state):
    expected = (
        ["profiler"]
        + ["planner", "feature_engineer", "trainer", "critic"] * 3
        + ["ensembler", "reporter"]
    )
    assert final_state["trace"] == expected


def test_all_produced_fields_populated(final_state):
    for field in ("profile", "plan", "fe_code", "training", "ensemble", "report"):
        assert final_state[field] is not None


# --- Honesty: a stub can never masquerade as a real result ------------------

def test_every_stub_payload_is_flagged(final_state):
    # Profiler (Day 7) and Planner (Day 8) are real — the rest are still stubs.
    for field in ("training", "ensemble", "report"):
        assert final_state[field].get("stub") is True
    assert all(c.get("stub") is True for c in final_state["critiques"])


def test_profiler_is_real_not_a_stub(final_state):
    # The first stub retired: a genuine, train-derived DataProfile.
    profile = final_state["profile"]
    assert profile.get("stub") is False
    assert profile["dataset_key"] == SPEC.key
    assert profile["n_rows"] > 0 and profile["features"]
    assert profile["assessment"]["source"] == "deterministic"


def test_planner_is_real_not_a_stub(final_state):
    # The second stub retired: a genuine, profile-derived ModelingPlan.
    plan = final_state["plan"]
    assert plan.get("stub") is False
    assert plan["dataset_key"] == SPEC.key
    assert plan["candidate_models"] and plan["cv"]["scheme"] == "StratifiedKFold"
    assert plan["recommended_primary_model"] == plan["candidate_models"][0]["name"]


def test_trainer_emits_no_numeric_score(final_state):
    # A stub must never surface a number that could read as a held-out result.
    assert final_state["training"]["cv_score"] is None


def test_crew_source_never_references_the_holdout():
    # Structural no-peeking: no crew-package module names the holdout loader.
    for mod in (crew_nodes,
                __import__("crewml.crew.state", fromlist=["x"]),
                __import__("crewml.crew.graph", fromlist=["x"]),
                __import__("crewml.crew.profiler", fromlist=["x"]),
                __import__("crewml.crew.planner", fromlist=["x"])):
        src = inspect.getsource(mod)
        assert "load_holdout" not in src
        assert "holdout" not in src.lower()
