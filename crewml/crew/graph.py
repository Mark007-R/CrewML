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


# The crew variants the graph builder knows how to assemble. ``full`` is the real
# crew shipped in Phase 2; ``no_critic`` is the Day-13 ablation topology — the same
# specialists in the same order, but with the Critic node and its feedback loop
# removed, so the crew makes exactly one forward pass. Keeping both behind one
# builder means the ablation runs the *identical* nodes as production and differs in
# one structural way only (the loop), which is the whole point of an ablation.
VARIANTS: tuple[str, ...] = ("full", "no_critic")

# The ablation's node set: everything except the Critic.
_NO_CRITIC_NODES: tuple[str, ...] = tuple(n for n in CREW_NODES if n != "critic")


def build_graph(variant: str = "full") -> StateGraph:
    """Assemble (but do not compile) the crew's :class:`StateGraph`.

    Separated from :func:`build_crew` so tests can inspect the topology and so a
    checkpointer/config can be attached at compile time without rebuilding.

    Parameters
    ----------
    variant:
        ``"full"`` (default) — the real crew with the Critic feedback loop.
        ``"no_critic"`` — the Day-13 ablation: the Critic node and its conditional
        loop edge are dropped and the Trainer hands straight to the Ensembler, so
        the crew runs one forward pass. Nothing else about the topology changes, so
        any score difference between the two is attributable to the loop alone.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown crew variant {variant!r}; choose from {VARIANTS}")

    g = StateGraph(CrewState)
    node_names = CREW_NODES if variant == "full" else _NO_CRITIC_NODES
    for name in node_names:
        g.add_node(name, _NODE_FUNCS[name])

    # Linear front half (shared by both variants).
    g.add_edge(START, "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")

    if variant == "full":
        g.add_edge("trainer", "critic")
        # The one conditional edge: loop back to Planner, or move on to finalise.
        g.add_conditional_edges(
            "critic",
            nodes.route_after_critic,
            {"iterate": "planner", "finalize": "ensembler"},
        )
    else:
        # Ablation: no Critic, no loop — a single forward pass to the finalise tail.
        g.add_edge("trainer", "ensembler")

    # Finalise tail (shared by both variants).
    g.add_edge("ensembler", "reporter")
    g.add_edge("reporter", END)
    return g


def build_crew(*, variant: str = "full", checkpointer=None):
    """Compile the crew graph into a runnable app.

    Parameters
    ----------
    variant:
        Which topology to build — ``"full"`` (default) or the ``"no_critic"``
        ablation (see :func:`build_graph`).
    checkpointer:
        Optional LangGraph checkpointer (e.g. for the Phase-5 run store). ``None``
        keeps runs purely in-memory, which is all the skeleton needs.
    """
    return build_graph(variant).compile(checkpointer=checkpointer)
