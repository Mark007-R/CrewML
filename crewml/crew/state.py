"""LangGraph shared state for the CrewML crew — Phase 2 skeleton (Day 5).

This module defines the single mutable :class:`CrewState` that flows through
every node of the crew graph. Day 5 locks the *schema and the control flow*; the
node bodies (see :mod:`crewml.crew.nodes`) are honest stubs — marked
``"stub": True`` — that Days 7-11 replace with real agents. Nothing here calls an
LLM or reads a dataset yet.

Design rules (so LangGraph checkpointing and the Day-26 dashboard both work):

* State values stay JSON / msgpack-friendly — dicts, lists, primitives, string
  keys. No live objects (a fitted estimator lives on disk under ``artifacts/`` and
  is referenced by path, never parked in the state).
* List-typed channels use an append reducer (:func:`operator.add`) so the Critic
  loop *grows* history — each pass appends a critique and a trace entry rather
  than clobbering the last.
* The held-out set is NEVER named here. The crew knows only ``dataset_key`` and
  loads ``train`` downstream via :func:`crewml.datasets.load_train`; the honesty
  invariant (EVAL_PROTOCOL §3 — no peeking) is therefore structural, not a rule a
  node has to remember.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from crewml.datasets import DatasetSpec


class CrewState(TypedDict, total=False):
    """Everything the crew shares as it moves a dataset from raw to reported.

    ``total=False`` so partial node updates are valid; :func:`initial_state`
    seeds every field once at invocation. Fields are grouped by lifecycle.
    """

    # --- Immutable run inputs (set once, at invocation) ---
    dataset_key: str
    task: str            # "classification" | "regression"
    subtype: str         # "binary" | "multiclass" | "regression"
    metric: str          # primary metric to maximise (see EVAL_PROTOCOL.md)
    max_iterations: int  # Critic-loop budget (config.MAX_ITERATIONS)

    # --- Produced by nodes as the crew works (None until produced) ---
    profile: Optional[dict[str, Any]]   # Profiler        (Day 7)  -> DataProfile
    plan: Optional[dict[str, Any]]      # Planner         (Day 8)  -> ModelingPlan
    fe_code: Optional[str]              # Feature Engineer (Day 9) -> generated code
    training: Optional[dict[str, Any]]  # Trainer         (Day 9)  -> CV metrics + artifact paths
    decision: Optional[str]             # Critic          (Day 10) -> "iterate" | "finalize"
    ensemble: Optional[dict[str, Any]]  # Ensembler       (Day 11) -> combined model info
    report: Optional[dict[str, Any]]    # Reporter        (Day 11) -> final report + model card

    # --- Loop bookkeeping ---
    iteration: int  # number of COMPLETED Critic passes so far

    # --- Append-only history channels (reducer = list concat) ---
    critiques: Annotated[list[dict[str, Any]], operator.add]  # one entry per Critic pass
    trace: Annotated[list[str], operator.add]                 # ordered node-visit log


def initial_state(spec: DatasetSpec, *, max_iterations: int) -> CrewState:
    """Build the seed state for one crew run from a dataset registry spec.

    Only the run inputs and the empty history channels are set; every produced
    field starts ``None`` so a node populating it is always an observable event.
    """
    return CrewState(
        dataset_key=spec.key,
        task=spec.task,
        subtype=spec.subtype,
        metric=spec.metric,
        max_iterations=int(max_iterations),
        profile=None,
        plan=None,
        fe_code=None,
        training=None,
        decision=None,
        ensemble=None,
        report=None,
        iteration=0,
        critiques=[],
        trace=[],
    )
