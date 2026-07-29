"""FastAPI service — Day 24: /run, /status, /report over the crew.

The API is a thin, honest shell: it validates a run request, persists it,
queues it, and reads the store back out. All ML happens in the crew; all
persistence in :mod:`crewml.api.store`; all execution in
:mod:`crewml.api.jobs`. ``create_app`` is a factory taking an injectable store
and runner so tests exercise the real routes against a fake executor.

Endpoints:

* ``GET  /healthz``       — liveness + provider/mock mode + known datasets.
* ``POST /run``           — submit a run (202 + run_id); body names a registered
                            ``dataset_key`` — CSV upload lands Day 26 with an
                            API-side sealed split, per the Phase-5 plan.
* ``GET  /status/{id}``   — lifecycle + headline outcome, small payload.
* ``GET  /report/{id}``   — full record + Day-23 manifest + model card
                            (409 until the run finishes).
* ``GET  /runs``          — recent runs, newest first.

Scores surfaced here are CV-on-train estimates (``cv_score_is_holdout:
false``); the manifest's seals prove the holdout stayed untouched.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crewml.config import MAX_ITERATIONS, is_mock_mode, LLM_PROVIDER
from crewml.api.jobs import JobRunner
from crewml.api.store import RunStore
from crewml.datasets import REGISTRY

API_VERSION = "0.1.0"  # Day 24


class RunRequest(BaseModel):
    """Everything a caller may say about a run — note: no data paths, ever."""

    dataset_key: str
    max_iterations: int = Field(default=MAX_ITERATIONS, ge=1, le=10)
    param_search: bool = True
    llm: bool = True
    token_budget: Optional[int] = Field(default=None, ge=1)
    time_budget_s: Optional[float] = Field(default=None, gt=0)


def _status_view(row: dict[str, Any]) -> dict[str, Any]:
    """The small /status payload: lifecycle + headline, no bulk fields."""
    view = {
        "run_id": row["run_id"],
        "dataset_key": row["dataset_key"],
        "task": row.get("task"),
        "subtype": row.get("subtype"),
        "metric": row.get("metric"),
        "status": row["status"],
        "created_at": row.get("created_at"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "error": row.get("error"),
    }
    record = row.get("record") or {}
    if record:
        fm = record.get("final_model") or {}
        view["headline"] = {
            "final_model_kind": fm.get("kind"),
            "final_cv_score": fm.get("final_cv_score"),
            "cv_score_is_holdout": record.get("cv_score_is_holdout", False),
            "iterations_run": record.get("iterations_run"),
            "holdout_untouched": record.get("holdout_untouched"),
        }
    return view


def create_app(store: Optional[RunStore] = None,
               runner: Optional[JobRunner] = None) -> FastAPI:
    store = store or RunStore()
    runner = runner or JobRunner(store)

    app = FastAPI(title="CrewML API", version=API_VERSION,
                  description="Submit tabular ML tasks to the CrewML multi-agent crew.")
    # visible to tests and scripts that need the wired instances
    app.state.store = store
    app.state.runner = runner

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": API_VERSION,
            "provider": LLM_PROVIDER,
            "mock_mode": is_mock_mode(),
            "datasets": list(REGISTRY),
        }

    @app.post("/run", status_code=202)
    def submit_run(req: RunRequest) -> dict[str, Any]:
        if req.dataset_key not in REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"unknown dataset {req.dataset_key!r}; known: {list(REGISTRY)}",
            )
        spec = REGISTRY[req.dataset_key]
        params = req.model_dump()
        row = store.create(
            req.dataset_key,
            task=spec.task, subtype=spec.subtype, metric=spec.metric,
            params=params,
        )
        runner.submit(row["run_id"], params)
        return {"run_id": row["run_id"], "status": "queued"}

    @app.get("/status/{run_id}")
    def run_status(run_id: str) -> dict[str, Any]:
        row = store.get(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return _status_view(row)

    @app.get("/report/{run_id}")
    def run_report(run_id: str) -> dict[str, Any]:
        row = store.get(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        if row["status"] in ("queued", "running"):
            raise HTTPException(
                status_code=409,
                detail={"status": row["status"],
                        "message": "run not finished; poll /status"},
            )
        if row["status"] == "failed":
            raise HTTPException(
                status_code=409,
                detail={"status": "failed", "error": row.get("error")},
            )
        return {
            "run_id": row["run_id"],
            "dataset_key": row["dataset_key"],
            "status": row["status"],
            "record": row.get("record"),
            "manifest": row.get("manifest"),
            "model_card": row.get("model_card"),
        }

    @app.get("/runs")
    def list_runs(limit: int = 50) -> dict[str, Any]:
        return {"runs": [_status_view(r) for r in store.list(limit=limit)]}

    return app


# uvicorn entry point: `uvicorn crewml.api.app:app`
app = create_app()
