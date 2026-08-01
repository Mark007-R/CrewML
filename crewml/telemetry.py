"""Per-run telemetry — what one crew run cost and what it bought (Day 25).

The Day-21 budget ledger already itemises a run's LLM spend per agent; the
Day-23 manifest pins what was held fixed. What was missing is the operational
record the API can aggregate: how long the run took end-to-end, how many tokens
and provider seconds it burned, what the node cache saved, and what score came
out. :func:`build_run_telemetry` assembles exactly that — one JSON-safe dict
persisted next to the run in the store and served by ``GET /metrics``.

Telemetry is measurement, never a result: nothing in here feeds the Day-23
result fingerprint (wall-clock and token counts legitimately differ between two
honest runs of the same pins — see ``crewml.manifest.canonical_result``), and
every surfaced score keeps its ``cv_score_is_holdout: False`` label.
"""
from __future__ import annotations

from typing import Any, Optional

from crewml import config

TELEMETRY_SCHEMA_VERSION = 1


def summarize_cache_events(events: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate the crew's per-node cache events into counters + a hit rate.

    ``hit_rate`` is over *attempted* lookups only — a Critic-loop bypass is the
    cache correctly standing aside, not a miss to punish it for.
    """
    events = list(events or [])
    hits = sum(1 for e in events if e.get("hit"))
    bypassed = sum(1 for e in events if e.get("bypass"))
    misses = len(events) - hits - bypassed
    attempted = hits + misses
    return {
        "n_events": len(events),
        "n_hits": hits,
        "n_misses": misses,
        "n_bypassed": bypassed,
        "n_stored": sum(1 for e in events if e.get("stored")),
        "hit_rate": round(hits / attempted, 4) if attempted else None,
        "events": events,
    }


def _llm_block(snapshot: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The run's LLM cost from a budget-ledger snapshot (zeros when absent)."""
    snap = snapshot or {}
    return {
        "provider": config.LLM_PROVIDER,
        "model": (config.GROQ_MODEL if config.LLM_PROVIDER == "groq"
                  else config.ANTHROPIC_MODEL),
        "mock_mode": config.is_mock_mode(),
        "n_calls": snap.get("n_calls", 0),
        "n_refused": snap.get("n_refused", 0),
        "prompt_tokens": snap.get("prompt_tokens", 0),
        "completion_tokens": snap.get("completion_tokens", 0),
        "tokens_spent": snap.get("tokens_spent", 0),
        "llm_time_s": snap.get("llm_time_s", 0.0),
        "token_budget": snap.get("token_budget"),
        "time_budget_s": snap.get("time_budget_s"),
        "exhausted": snap.get("exhausted", False),
        "stop_reason": snap.get("stop_reason"),
        "per_agent": snap.get("per_agent", {}),
    }


def _outcome_block(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The headline outcome — CV-on-train, labelled as such, never a holdout score."""
    if not record:
        return None
    final_model = record.get("final_model") or {}
    return {
        "dataset_key": record.get("dataset_key"),
        "metric": record.get("metric"),
        "final_cv_score": final_model.get("final_cv_score"),
        "final_model_kind": final_model.get("kind"),
        "iterations_run": record.get("iterations_run"),
        "holdout_untouched": record.get("holdout_untouched"),
        "cv_score_is_holdout": False,
    }


def build_run_telemetry(
    *,
    duration_s: float,
    budget_snapshot: Optional[dict[str, Any]] = None,
    cache_events: Optional[list[dict[str, Any]]] = None,
    record: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble one run's full telemetry record (JSON-safe, store-ready)."""
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "duration_s": round(float(duration_s), 3),
        "llm": _llm_block(budget_snapshot),
        "cache": summarize_cache_events(cache_events),
        "outcome": _outcome_block(record),
    }
