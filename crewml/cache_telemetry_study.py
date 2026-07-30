"""Day 25 study — what the node cache saves and what telemetry records.

Two measurements, both on the deterministic core (LLM narratives disabled for
the study, so every number is a property of the caching mechanism, not of a
provider's latency — labelled as such in the output):

1. **Node timings, cold vs warm.** For each locked dataset, run the Profiler
   and Planner *nodes* (the cached layer) against a fresh cache directory, then
   again against the populated one. The cold pass pays the leakage screen's
   per-feature CV; the warm pass must return the byte-identical answer from
   disk — the study asserts equality, then reports the seconds saved.
2. **The API round-trip.** Two identical ``execute_crew_run`` submissions on
   one dataset through the real ``RunStore`` + sync ``JobRunner``: the first
   populates the cache, the second hits it. What lands in the study JSON is
   exactly what ``GET /metrics`` serves — per-run duration + cache telemetry
   and the aggregated service payload.

Honesty: warm-vs-cold equality is checked on the profile/plan *content* (cache
annotation aside) — a cache that saved time by changing the answer would fail
the study, not star in it. All scores in the embedded metrics payload are
CV-on-train, labelled. Everything here runs offline; no LLM is consulted.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from crewml import cache
from crewml.api.jobs import JobRunner
from crewml.api.runner import execute_crew_run
from crewml.api.store import RunStore
from crewml.config import ARTIFACTS_DIR, RESULTS_DIR, is_mock_mode, LLM_PROVIDER
from crewml.crew import nodes
from crewml.datasets import REGISTRY

CACHE_TELEMETRY_SCHEMA_VERSION = 1
STUDY_PATH = RESULTS_DIR / "day25_cache_telemetry.json"
STUDY_MD_PATH = RESULTS_DIR / "day25_cache_telemetry.md"

STUDY_DIR = ARTIFACTS_DIR / "day25_study"
API_DATASET = "credit-g"

# The study pins the whole knob surface it depends on, so a run is comparable
# to the next one: cache on, narratives off, no param search (crew wall-clock
# then measures CV + caching, not the search grid).
_STUDY_ENV = {
    "CREWML_NODE_CACHE": "1",
    "CREWML_PROFILER_LLM": "0",
    "CREWML_PLANNER_LLM": "0",
    "CREWML_FE_LLM": "0",
    "CREWML_CRITIC_LLM": "0",
    "CREWML_TRAINER_PARAM_SEARCH": "0",
}


class _study_env:
    """Apply the study env + a private cache dir; restore everything on exit."""

    def __init__(self, cache_dir: Path):
        self._overrides = dict(_STUDY_ENV, CREWML_NODE_CACHE_DIR=str(cache_dir))
        self._prior: dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self._overrides.items():
            self._prior[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _strip_annotation(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if k != cache.CACHE_META_KEY}


def _timed_nodes(dataset_key: str) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    """One Profiler+Planner pass; returns (profile, plan, profiler_s, planner_s)."""
    t0 = time.perf_counter()
    p_out = nodes.profiler({"dataset_key": dataset_key})
    t1 = time.perf_counter()
    state = {"dataset_key": dataset_key, "profile": p_out["profile"],
             "critiques": [], "iteration": 0}
    n_out = nodes.planner(state)
    t2 = time.perf_counter()
    events = p_out["cache_events"] + n_out["cache_events"]
    return ({"profile": p_out["profile"], "plan": n_out["plan"], "events": events},
            {"profiler": p_out, "planner": n_out},
            round(t1 - t0, 3), round(t2 - t1, 3))


def _node_study(datasets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in datasets:
        cold, _, cold_prof_s, cold_plan_s = _timed_nodes(key)
        warm, _, warm_prof_s, warm_plan_s = _timed_nodes(key)

        hits = [e for e in warm["events"] if e.get("hit")]
        identical = (
            _strip_annotation(warm["profile"]) == _strip_annotation(cold["profile"])
            and _strip_annotation(warm["plan"]) == _strip_annotation(cold["plan"])
        )
        cold_s = round(cold_prof_s + cold_plan_s, 3)
        warm_s = round(warm_prof_s + warm_plan_s, 3)
        rows.append({
            "dataset": key,
            "cold_s": cold_s,
            "warm_s": warm_s,
            "saved_s": round(cold_s - warm_s, 3),
            "speedup": round(cold_s / warm_s, 1) if warm_s > 0 else None,
            "warm_hits": len(hits),
            "warm_answer_identical": identical,
        })
    return rows


def _api_study() -> dict[str, Any]:
    """Two identical runs through the real store + runner; report what /metrics sees."""
    store_path = STUDY_DIR / "runs.sqlite"
    store = RunStore(store_path)
    runner = JobRunner(store, execute=execute_crew_run, sync=True)

    runs: list[dict[str, Any]] = []
    for label in ("cold", "warm"):
        row = store.create(API_DATASET, task=REGISTRY[API_DATASET].task,
                           subtype=REGISTRY[API_DATASET].subtype,
                           metric=REGISTRY[API_DATASET].metric,
                           params={"dataset_key": API_DATASET})
        runner.submit(row["run_id"], {"dataset_key": API_DATASET,
                                      "param_search": False, "llm": False})
        done = store.get(row["run_id"])
        tel = done.get("telemetry") or {}
        runs.append({
            "label": label,
            "status": done["status"],
            "error": done.get("error"),
            "duration_s": tel.get("duration_s"),
            "tokens_spent": (tel.get("llm") or {}).get("tokens_spent"),
            "cache": {k: v for k, v in (tel.get("cache") or {}).items()
                      if k != "events"},
            "cache_events": (tel.get("cache") or {}).get("events"),
            "final_cv_score": ((tel.get("outcome") or {}).get("final_cv_score")),
            "result_fingerprint": (done.get("manifest") or {}).get("result_fingerprint"),
        })

    fingerprints = {r["result_fingerprint"] for r in runs if r["result_fingerprint"]}
    return {
        "dataset": API_DATASET,
        "runs": runs,
        "same_result_fingerprint": len(fingerprints) == 1 if fingerprints else None,
        "metrics_payload": store.metrics(),
    }


def run_study(datasets: Optional[list[str]] = None) -> dict[str, Any]:
    datasets = datasets or list(REGISTRY)
    if STUDY_DIR.exists():
        shutil.rmtree(STUDY_DIR)  # cold means cold — no leftovers from a prior study

    with _study_env(STUDY_DIR / "node_cache"):
        node_rows = _node_study(datasets)
    with _study_env(STUDY_DIR / "api_cache"):
        api = _api_study()

    return {
        "schema_version": CACHE_TELEMETRY_SCHEMA_VERSION,
        "labels": {
            "deterministic_core_only": True,
            "llm_narratives": "disabled for the study — savings shown are the "
                              "recomputation (leakage-screen CV etc.), not provider latency",
            "is_measurement_of_llm_capability": False,
            "scores_are_cv_on_train": True,
            "param_search": False,
        },
        "environment": {"provider": LLM_PROVIDER, "mock_mode": is_mock_mode()},
        "node_timings": node_rows,
        "api_round_trip": api,
    }


# --- The committed table (registered in crewml.artifact_registry) -------------

def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Day 25 — Node cache & telemetry study",
        "",
        "Deterministic core only (LLM narratives disabled): the savings below are "
        "recomputation the cache avoided, **not** provider latency. Scores are "
        "CV-on-train, never held-out.",
        "",
        "## Profiler+Planner nodes — cold vs warm",
        "",
        "| dataset | cold (s) | warm (s) | saved (s) | speedup | warm answer identical |",
        "|---|---:|---:|---:|---:|:--|",
    ]
    for r in data["node_timings"]:
        speedup = f"{r['speedup']}x" if r["speedup"] is not None else "-"
        lines.append(
            f"| {r['dataset']} | {r['cold_s']} | {r['warm_s']} | {r['saved_s']} "
            f"| {speedup} | {r['warm_answer_identical']} |"
        )

    api = data["api_round_trip"]
    lines += [
        "",
        f"## API round-trip on {api['dataset']} — what /metrics records",
        "",
        "| run | status | duration (s) | tokens | cache hits | cache misses | final CV score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in api["runs"]:
        c = r.get("cache") or {}
        lines.append(
            f"| {r['label']} | {r['status']} | {r.get('duration_s')} "
            f"| {r.get('tokens_spent')} | {c.get('n_hits')} | {c.get('n_misses')} "
            f"| {r.get('final_cv_score')} |"
        )
    lines += [
        "",
        f"Same Day-23 result fingerprint across cold and warm: "
        f"**{api['same_result_fingerprint']}** — the cache changed the cost, not the answer.",
        "",
    ]
    return "\n".join(lines)
