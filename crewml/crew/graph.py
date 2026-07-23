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
# crew shipped in Phase 2; the rest are ablations, each removing exactly one
# specialist so its contribution is measurable in isolation:
#
# * ``no_critic`` (Day 13) — a *structural* removal: the Critic node and its loop
#   edge are dropped, so the crew makes one forward pass.
# * ``no_planner`` / ``no_feature_engineer`` (Day 14) — *substitutional* removals:
#   the graph cannot run without a plan / an ``add_features``, so the specialist is
#   replaced by its naive floor (:func:`crewml.crew.nodes.naive_planner` — a
#   profile-blind generic plan, critique-deaf; :func:`~crewml.crew.nodes.identity_feature_engineer`
#   — the identity transform). Topology, node names and the Critic loop are all
#   untouched; only the one node's body changes.
#
# Keeping every variant behind one builder means each ablation runs the *identical*
# crew except for the one removal — which is the whole point of an ablation.
VARIANTS: tuple[str, ...] = ("full", "no_critic", "no_planner", "no_feature_engineer")

# The Day-13 ablation's node set: everything except the Critic.
_NO_CRITIC_NODES: tuple[str, ...] = tuple(n for n in CREW_NODES if n != "critic")

# The Day-14 ablations' node-body swaps (same node name, naive-floor implementation).
_NODE_OVERRIDES: dict[str, dict[str, object]] = {
    "no_planner": {"planner": nodes.naive_planner},
    "no_feature_engineer": {"feature_engineer": nodes.identity_feature_engineer},
}


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
        ``"no_planner"`` / ``"no_feature_engineer"`` — the Day-14 ablations: the
        identical topology (Critic loop included), with that one specialist's node
        body swapped for its naive floor (see ``_NODE_OVERRIDES``), so any score
        difference is attributable to that specialist alone.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown crew variant {variant!r}; choose from {VARIANTS}")

    g = StateGraph(CrewState)
    node_names = _NO_CRITIC_NODES if variant == "no_critic" else CREW_NODES
    node_funcs = {**_NODE_FUNCS, **_NODE_OVERRIDES.get(variant, {})}
    for name in node_names:
        g.add_node(name, node_funcs[name])

    # Linear front half (shared by both variants).
    g.add_edge(START, "profiler")
    g.add_edge("profiler", "planner")
    g.add_edge("planner", "feature_engineer")
    g.add_edge("feature_engineer", "trainer")

    if variant == "no_critic":
        # Day-13 ablation: no Critic, no loop — a single forward pass to the finalise tail.
        g.add_edge("trainer", "ensembler")
    else:
        g.add_edge("trainer", "critic")
        # The one conditional edge: loop back to Planner, or move on to finalise.
        # (In the no_planner variant "planner" is the naive stand-in — the loop
        # edge survives, but re-entry rebuilds the identical plan by design.)
        g.add_conditional_edges(
            "critic",
            nodes.route_after_critic,
            {"iterate": "planner", "finalize": "ensembler"},
        )

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
