"""Execute one crew run for the API — the same invocation the CLI drivers make.

:func:`execute_crew_run` is the job the async runner performs: build the crew,
seed the state from the requested ``dataset_key``, run it under a fresh Day-21
budget, then package the outcome as (summary record, Day-23 run manifest,
Reporter model card). It deliberately mirrors ``scripts/run_crew.py`` — the API
must not grow a second, subtly different way of running the crew.

Request options map onto the same knobs the CLI exposes (``param_search``,
``llm``, budgets). The two boolean toggles are plumbed through the existing
``CREWML_*`` environment switches the node bodies read; the worker executes one
run at a time, and each variable is restored to its prior value afterwards, so
the toggles cannot leak between runs.

Honesty invariant: the request carries a ``dataset_key`` and nothing else about
data location. ``initial_state`` never learns a holdout path (structural, per
``CrewState``), and the returned record carries ``verify_holdout_untouched`` +
``cv_score_is_holdout: False`` so no API consumer can mistake a CV estimate for
a held-out score.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from crewml import budget as budget_mod
from crewml import manifest as manifest_mod
from crewml import telemetry as telemetry_mod
from crewml.config import MAX_ITERATIONS
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY, verify_holdout_untouched

# request-option name -> env switch consumed by the crew nodes
_PARAM_SEARCH_ENV = "CREWML_TRAINER_PARAM_SEARCH"
_LLM_ENVS = ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM",
             "CREWML_FE_LLM", "CREWML_CRITIC_LLM")


@contextmanager
def _env_overrides(overrides: dict[str, str]):
    prior = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_run_record(spec, final: dict[str, Any]) -> dict[str, Any]:
    """Summarise a final crew state — same shape as the Day-11 driver's record."""
    report = final.get("report") or {}
    ensemble = final.get("ensemble") or {}
    final_model = report.get("final_model") or {}
    critiques = final.get("critiques") or []
    return {
        "dataset_key": spec.key,
        "task": spec.task,
        "subtype": spec.subtype,
        "metric": spec.metric,
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "trace": final.get("trace"),
        "final_model": {
            "kind": final_model.get("kind"),
            "chosen": final_model.get("chosen"),
            "members": final_model.get("members"),
            "single_best_model": final_model.get("single_best_model"),
            "final_cv_score": final_model.get("cv_score"),
            "ensemble_cv_score": final_model.get("ensemble_cv_score"),
            "single_best_cv_score": final_model.get("single_best_cv_score"),
            "improvement_over_single": final_model.get("improvement_over_single"),
        },
        "ensemble_attempted": ensemble.get("attempted"),
        "holdout_untouched": verify_holdout_untouched(spec.key),
        "warnings": report.get("warnings"),
        "run_budget": report.get("run_budget"),
        "cv_score_is_holdout": False,
    }


def _progress_view(state: dict[str, Any]) -> dict[str, Any]:
    """The small live snapshot the dashboard polls while a run is executing.

    Derived entirely from CrewState channels the graph already maintains — the
    node-visit trace, the iteration counter, Critic decisions. Nothing here can
    name a holdout: the state itself can't (structural, Day 5).
    """
    critiques = state.get("critiques") or []
    trace = state.get("trace") or []
    return {
        "trace": trace,
        "nodes_visited": len(trace),
        "current_node": trace[-1] if trace else None,
        "iteration": state.get("iteration"),
        "max_iterations": state.get("max_iterations"),
        "decisions": [c.get("decision") for c in critiques],
    }


def execute_crew_run(params: dict[str, Any],
                     on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
                     ) -> dict[str, Any]:
    """Run the full crew per an API run request; return record/manifest/model card.

    ``params``: dataset_key (required, must be in REGISTRY), max_iterations,
    param_search (bool), llm (bool), token_budget, time_budget_s.

    ``on_progress`` (Day 26): called with a :func:`_progress_view` snapshot
    after every graph step — the dashboard's live agent trace. The run is
    driven through ``app.stream(..., stream_mode="values")``, whose final
    yielded state is exactly what ``invoke`` returns; a progress callback that
    raises is swallowed so a flaky observer can never kill a crew run.
    """
    key = params["dataset_key"]
    spec = REGISTRY[key]
    max_iterations = int(params.get("max_iterations") or MAX_ITERATIONS)
    token_budget: Optional[int] = params.get("token_budget")
    time_budget_s: Optional[float] = params.get("time_budget_s")

    overrides: dict[str, str] = {}
    if params.get("param_search") is False:
        overrides[_PARAM_SEARCH_ENV] = "0"
    if params.get("llm") is False:
        overrides.update({var: "0" for var in _LLM_ENVS})

    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    started = time.monotonic()
    with _env_overrides(overrides):
        with budget_mod.run_budget(token_budget, time_budget_s) as ledger:
            final: dict[str, Any] = state
            for step_state in app.stream(
                state, config={"recursion_limit": limit}, stream_mode="values",
            ):
                final = step_state
                if on_progress is not None:
                    try:
                        on_progress(_progress_view(step_state))
                    except Exception:
                        pass  # observers never kill the run
            budget_snapshot = ledger.snapshot()
    duration_s = time.monotonic() - started

    report = final.get("report") or {}
    record = build_run_record(spec, final)
    return {
        "record": record,
        "manifest": manifest_mod.build_run_manifest(final),
        "model_card": report.get("model_card_markdown") or "",
        # Day 25: measurement, not result — stored beside the run, aggregated
        # by GET /metrics, and never part of the result fingerprint.
        "telemetry": telemetry_mod.build_run_telemetry(
            duration_s=duration_s,
            budget_snapshot=budget_snapshot,
            cache_events=final.get("cache_events"),
            record=record,
        ),
    }
