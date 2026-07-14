"""CrewML's multi-agent crew (LangGraph).

Phase 2 builds this out day by day. Day 5 ships the skeleton: the shared
:class:`CrewState`, seven node stubs, and the wired graph with the Critic loop
and its ``max_iterations`` guard. Real agents replace the stubs on Days 7-11.
"""
from crewml.crew.feature_engineer import run_feature_engineer
from crewml.crew.graph import CREW_NODES, build_crew, build_graph
from crewml.crew.nodes import route_after_critic
from crewml.crew.planner import build_plan, run_planner
from crewml.crew.profiler import build_profile, run_profiler
from crewml.crew.state import CrewState, initial_state
from crewml.crew.trainer import run_trainer

__all__ = [
    "CrewState",
    "initial_state",
    "build_graph",
    "build_crew",
    "CREW_NODES",
    "route_after_critic",
    "build_profile",
    "run_profiler",
    "build_plan",
    "run_planner",
    "run_feature_engineer",
    "run_trainer",
]
