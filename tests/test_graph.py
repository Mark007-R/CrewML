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
    # Profiler/Planner/FE/Trainer/Critic are all real now (Days 7-10). Keep the wiring
    # test offline + fast: disable the advisory LLM narratives and the FE LLM (so the
    # deterministic default FE is used), and skip grid search (CV at default params).
    # credit-g is a clean run, so the real Critic finalises on pass one; with
    # max_iterations=1 the router's guard would finalise anyway. Either way exactly one
    # loop passes through, which is what the wiring test needs; the router's guard is
    # covered exhaustively above and the Critic's own logic in test_critic.py.
    mp = pytest.MonkeyPatch()
    mp.setenv("CREWML_PROFILER_LLM", "0")
    mp.setenv("CREWML_PLANNER_LLM", "0")
    mp.setenv("CREWML_FE_LLM", "0")
    mp.setenv("CREWML_CRITIC_LLM", "0")
    mp.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")
    app = build_crew()
    st = initial_state(SPEC, max_iterations=1)
    final = app.invoke(st, config={"recursion_limit": 50})
    mp.undo()
    return final


def test_run_terminates_at_reporter(final_state):
    assert final_state["trace"][-1] == "reporter"
    assert final_state["report"] is not None


def test_run_makes_exactly_one_pass_then_finalizes(final_state):
    # credit-g is clean, so the real Critic finalises on pass one; and with
    # max_iterations=1 the guard would finalise anyway. Exactly one pass either way.
    assert final_state["iteration"] == 1
    assert len(final_state["critiques"]) == 1


def test_trace_is_the_expected_looped_sequence(final_state):
    expected = (
        ["profiler"]
        + ["planner", "feature_engineer", "trainer", "critic"] * 1
        + ["ensembler", "reporter"]
    )
    assert final_state["trace"] == expected


def test_all_produced_fields_populated(final_state):
    for field in ("profile", "plan", "fe_code", "fe_meta", "training", "ensemble", "report"):
        assert final_state[field] is not None


# --- Honesty: a stub can never masquerade as a real result ------------------

def test_every_stub_payload_is_flagged(final_state):
    # Profiler/Planner/FE/Trainer/Critic (Days 7-10) are real — only Ensembler +
    # Reporter remain stubs.
    for field in ("ensemble", "report"):
        assert final_state[field].get("stub") is True
    assert all(c.get("stub") is False for c in final_state["critiques"])


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


def test_feature_engineer_is_real_not_a_stub(final_state):
    # The third stub retired: a validated add_features module + provenance.
    meta = final_state["fe_meta"]
    assert meta["source"] in {"default", "llm", "fallback"}
    assert meta["validation"]["ok"] is True
    assert "def add_features" in final_state["fe_code"]


def test_trainer_is_real_with_a_cross_validated_score(final_state):
    # The fourth stub retired: a real CV number, honestly labelled as NOT held-out.
    training = final_state["training"]
    assert training.get("stub") is False
    assert isinstance(training["cv_score"], float)
    assert training["cv_score_is_holdout"] is False
    assert training["best_model"] in {
        "hist_gradient_boosting", "random_forest", "logistic_regression"
    }
    assert "model.joblib" in training["artifacts"]


def test_critic_is_real_with_a_decision(final_state):
    # The fifth stub retired: a genuine critique that diagnoses and decides.
    critique = final_state["critiques"][-1]
    assert critique.get("stub") is False
    assert critique["node"] == "critic"
    assert critique["decision"] in {"iterate", "finalize"}
    # credit-g is clean at default params -> nothing actionable -> finalize.
    assert critique["decision"] == "finalize"
    assert critique["finding_codes"] == []
    assert isinstance(critique["cv_score"], float)


def test_crew_source_never_references_the_holdout():
    # Structural no-peeking: no crew-package module names the holdout loader.
    for mod in (crew_nodes,
                __import__("crewml.crew.state", fromlist=["x"]),
                __import__("crewml.crew.graph", fromlist=["x"]),
                __import__("crewml.crew.profiler", fromlist=["x"]),
                __import__("crewml.crew.planner", fromlist=["x"]),
                __import__("crewml.crew.critic", fromlist=["x"])):
        src = inspect.getsource(mod)
        assert "load_holdout" not in src
        assert "holdout" not in src.lower()
