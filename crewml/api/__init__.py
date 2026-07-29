"""CrewML production API (Phase 5).

Day 24 wraps the crew in a service boundary: a FastAPI app (:mod:`crewml.api.app`)
over an async job runner (:mod:`crewml.api.jobs`) and a SQLite run-store
(:mod:`crewml.api.store`). The crew itself is untouched — the API submits the same
``build_crew()`` invocation the CLI drivers use, records the Day-23 run manifest
alongside the outcome, and preserves the structural honesty invariant: a run request
names a ``dataset_key`` and nothing else; no holdout path ever crosses the API
boundary or enters the store.
"""
from crewml.api.store import RunStore
from crewml.api.jobs import JobRunner
from crewml.api.app import create_app

__all__ = ["RunStore", "JobRunner", "create_app"]
