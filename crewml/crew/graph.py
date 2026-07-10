"""The CrewML LangGraph — Phase 2 skeleton (Day 5).

Wires the seven crew nodes into the pipeline with the Critic feedback loop::

    START
      -> profiler
      -> planner  <-------------------.
      -> feature_engineer             |  (iterate: Critic asked for another pass
      -> trainer                      |   AND the iteration budget isn't spent)
      -> critic --[route_after_critic]-`
                   \--(finalize)--> ensembler -> reporter -> END

The only conditional edge is out of the Critic (:func:`route_after_critic` in
:mod:`crewml.crew.nodes`): it either loops back to the Planner for another
iteration or hands off to the Ensembler to finalise. The ``max_iterations`` guard
lives inside that router, so the loop is bounded by construction — a runaway crew
is structurally impossible, not merely unlikely.

Day 5 compiles and *runs* this graph with stub nodes end-to-end; Days 7-11 swap
real agents in behind the same topology.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from crewml.crew import nodes
from crewml.crew.state import CrewState

# The seven nodes, in nominal execution order. Kept as data so tests and the
# future dashboard can enumerate the crew without importing the graph internals.
CREW_NODES: tuple[str, ...] = (
    "profiler",
    "planner",
    "feature_engineer",
    "trainer",
    "critic",
    "ensembler",
    "reporter",
)

_NODE_FUNCS = {
    "profiler": nodes.profiler,
    "planner": nodes.planner,
    "feature_engineer": nodes.feature_engineer,
    "trainer": nodes.trainer,
    "critic": nodes.critic,
    "ensembler": nodes.ensembler,
    "reporter": nodes.reporter,
}


def build_graph() -> StateGraph:
    """Assemble (but do not compile) the crew's :class:`StateGraph`.

    Separated from :func:`build_crew` so tests can inspect the topology and so a
    checkpointer/config can be attached at compile time without rebuilding.
    """
    g = StateGraph(CrewState)
    for name in CREW_NODES:
        g.add_node(name, _NODE_FUNCS[name])

    # Linear front half.
    g.add_edge(START, "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")
    g.add_edge("trainer", "critic")

    # The one conditional edge: loop back to Planner, or move on to finalise.
    g.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"iterate": "planner", "finalize": "ensembler"},
    )

    # Finalise tail.
    g.add_edge("ensembler", "reporter")
    g.add_edge("reporter", END)
    return g


def build_crew(*, checkpointer=None):
    """Compile the crew graph into a runnable app.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer (e.g. for the Phase-5 run store). ``None``
        keeps runs purely in-memory, which is all the skeleton needs.
    """
    return build_graph().compile(checkpointer=checkpointer)
