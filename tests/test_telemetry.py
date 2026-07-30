"""Day 25 guards: telemetry measures the run without touching the result.

Pins the telemetry record's shape and zero-safety, the cache-event arithmetic
(bypasses are not misses), the latency accounting the budget ledger grew, the
store's schema migration + persistence, and the /metrics aggregation maths —
including that every surfaced score stays labelled CV-on-train.
"""
from __future__ import annotations

import json
import sqlite3

from crewml.api.store import RunStore, _quantile
from crewml.budget import RunBudget
from crewml.telemetry import build_run_telemetry, summarize_cache_events


# --- The telemetry record ----------------------------------------------------

def test_build_run_telemetry_is_zero_safe_and_json_safe():
    tel = build_run_telemetry(duration_s=1.23456)
    assert tel["duration_s"] == 1.235
    assert tel["llm"]["n_calls"] == 0
    assert tel["llm"]["tokens_spent"] == 0
    assert tel["cache"]["n_events"] == 0
    assert tel["cache"]["hit_rate"] is None
    assert tel["outcome"] is None
    json.dumps(tel)


def test_build_run_telemetry_carries_ledger_cache_and_outcome():
    ledger = RunBudget(token_budget=1000, clock=lambda: 0.0)
    ledger.charge(agent="profiler", prompt_tokens=100, completion_tokens=50,
                  latency_s=0.75)
    events = [{"node": "profiler", "kind": "profile", "hit": True}]
    record = {"dataset_key": "credit-g", "metric": "roc_auc",
              "iterations_run": 1, "holdout_untouched": True,
              "final_model": {"kind": "single", "final_cv_score": 0.79}}

    tel = build_run_telemetry(duration_s=10.0, budget_snapshot=ledger.snapshot(),
                              cache_events=events, record=record)
    assert tel["llm"]["tokens_spent"] == 150
    assert tel["llm"]["llm_time_s"] == 0.75
    assert tel["llm"]["per_agent"]["profiler"]["llm_time_s"] == 0.75
    assert tel["cache"]["n_hits"] == 1
    assert tel["outcome"]["final_cv_score"] == 0.79
    assert tel["outcome"]["cv_score_is_holdout"] is False  # the honesty label rides along


def test_cache_event_summary_arithmetic():
    events = [
        {"hit": True}, {"hit": True},
        {"hit": False, "stored": True},
        {"bypass": "critique_reentry"},
    ]
    s = summarize_cache_events(events)
    assert (s["n_hits"], s["n_misses"], s["n_bypassed"], s["n_stored"]) == (2, 1, 1, 1)
    # A bypass is the cache correctly standing aside — hit rate is 2/3, not 2/4.
    assert s["hit_rate"] == round(2 / 3, 4)


# --- Store: migration + persistence -----------------------------------------

def test_store_migrates_a_pre_day25_database(tmp_path):
    """A runs.sqlite created before the telemetry column existed must upgrade."""
    path = tmp_path / "runs.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, dataset_key TEXT NOT NULL,"
            " task TEXT, subtype TEXT, metric TEXT, status TEXT NOT NULL,"
            " created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,"
            " params TEXT NOT NULL, record TEXT, manifest TEXT, model_card TEXT,"
            " error TEXT)"
        )
        conn.execute(
            "INSERT INTO runs (run_id, dataset_key, status, created_at, params)"
            " VALUES ('old1', 'credit-g', 'succeeded', '2026-07-29T00:00:00+00:00', '{}')"
        )

    store = RunStore(path)  # must ALTER TABLE, not fail and not drop rows
    row = store.get("old1")
    assert row["status"] == "succeeded"
    assert row["telemetry"] is None

    store.finish_success("old1", record={}, manifest={}, model_card="",
                         telemetry={"duration_s": 1.0})
    assert store.get("old1")["telemetry"]["duration_s"] == 1.0


def test_finish_failure_records_telemetry(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    store.create("credit-g", run_id="r1")
    store.finish_failure("r1", "boom", telemetry={"duration_s": 2.5})
    row = store.get("r1")
    assert row["status"] == "failed"
    assert row["telemetry"]["duration_s"] == 2.5


# --- Store: /metrics aggregation --------------------------------------------

def _seed_runs(store: RunStore) -> None:
    tel = lambda dur, tokens, hits, misses: {  # noqa: E731 — local table builder
        "duration_s": dur,
        "llm": {"n_calls": 2, "n_refused": 0, "tokens_spent": tokens,
                "llm_time_s": 1.0},
        "cache": {"n_hits": hits, "n_misses": misses, "n_bypassed": 0,
                  "n_stored": misses},
    }
    record = lambda score: {"final_model": {"final_cv_score": score}}  # noqa: E731

    store.create("credit-g", metric="roc_auc", run_id="a")
    store.finish_success("a", record=record(0.78), manifest={}, model_card="",
                         telemetry=tel(10.0, 1000, 0, 2))
    store.create("credit-g", metric="roc_auc", run_id="b")
    store.finish_success("b", record=record(0.80), manifest={}, model_card="",
                         telemetry=tel(6.0, 0, 2, 0))
    store.create("diabetes", metric="roc_auc", run_id="c")
    store.finish_failure("c", "boom", telemetry={"duration_s": 2.0})
    store.create("kin8nm", metric="r2", run_id="d")  # still queued — no telemetry


def test_metrics_aggregates_lifecycle_cost_cache_and_scores(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    _seed_runs(store)
    m = store.metrics()

    assert m["runs"]["total"] == 4
    assert m["runs"]["by_status"] == {"queued": 1, "running": 0,
                                      "succeeded": 2, "failed": 1}
    assert m["runs"]["success_rate"] == round(2 / 3, 4)  # over finished runs only

    assert m["latency"]["n_measured"] == 3
    assert m["latency"]["mean_s"] == 6.0
    assert m["latency"]["p50_s"] == 6.0

    assert m["llm"]["tokens_spent"] == 1000
    assert m["llm"]["n_calls"] == 4

    assert m["cache"]["n_hits"] == 2 and m["cache"]["n_misses"] == 2
    assert m["cache"]["hit_rate"] == 0.5

    ds = m["datasets"]["credit-g"]
    assert (ds["n_runs"], ds["n_succeeded"]) == (2, 2)
    assert ds["mean_cv_score"] == 0.79
    assert ds["best_cv_score"] == 0.80
    assert m["datasets"]["kin8nm"]["mean_cv_score"] is None

    assert m["cv_score_is_holdout"] is False
    json.dumps(m)


def test_metrics_on_an_empty_store(tmp_path):
    m = RunStore(tmp_path / "runs.sqlite").metrics()
    assert m["runs"]["total"] == 0
    assert m["runs"]["success_rate"] is None
    assert m["latency"]["mean_s"] is None
    assert m["cache"]["hit_rate"] is None
    assert m["datasets"] == {}


def test_quantile_nearest_rank():
    assert _quantile([], 0.5) is None
    assert _quantile([3.0], 0.95) == 3.0
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0  # round(0.5*3)=2 -> ordered[2]
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
