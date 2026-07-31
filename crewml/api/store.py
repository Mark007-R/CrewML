"""SQLite run-store — the durable record behind the Day-24 API.

One row per submitted crew run, carrying the full lifecycle
(``queued -> running -> succeeded | failed``) plus, on success, the summary
record, the Day-23 run manifest (pins + result fingerprint slot straight into
the ``manifest`` column), and the Reporter's model card. SQLite because the API
runs single-host with one worker consuming a queue — a server database would be
ceremony; a JSON file would lose atomicity.

Concurrency model: the FastAPI request threads and the job-runner worker each
open a short-lived connection per operation (no shared connection object), and
WAL mode lets readers proceed while the worker writes. That keeps the store
safe under the app's actual concurrency without any locking of our own.

Honesty invariant (structural, mirrored from ``CrewState``): the store schema
has nowhere to put a holdout path. A run is pinned to a ``dataset_key``; the
seals inside the stored manifest prove the holdout was untouched, they never
say where it lives.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from crewml.config import ARTIFACTS_DIR

# git-ignored by default (artifacts/); override for tests or deployment.
DEFAULT_STORE_PATH = Path(
    os.getenv("CREWML_RUN_STORE", str(ARTIFACTS_DIR / "api" / "runs.sqlite"))
)

STATUSES = ("queued", "running", "succeeded", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    dataset_key TEXT NOT NULL,
    task        TEXT,
    subtype     TEXT,
    metric      TEXT,
    status      TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    params      TEXT NOT NULL,
    record      TEXT,
    manifest    TEXT,
    model_card  TEXT,
    error       TEXT,
    telemetry   TEXT,
    progress    TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunStore:
    """Thread-safe run persistence via per-operation SQLite connections."""

    def __init__(self, path: Path | str = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Day-25 migration: a store created before the telemetry column
            # existed keeps its rows; CREATE IF NOT EXISTS won't add the column.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
            if "telemetry" not in cols:
                conn.execute("ALTER TABLE runs ADD COLUMN telemetry TEXT")
            # Day-26 migration: live progress for the dashboard's agent trace.
            if "progress" not in cols:
                conn.execute("ALTER TABLE runs ADD COLUMN progress TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # --- lifecycle -----------------------------------------------------------

    def create(self, dataset_key: str, *, task: Optional[str] = None,
               subtype: Optional[str] = None, metric: Optional[str] = None,
               params: Optional[dict[str, Any]] = None,
               run_id: Optional[str] = None) -> dict[str, Any]:
        rid = run_id or new_run_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, dataset_key, task, subtype, metric,"
                " status, created_at, params) VALUES (?,?,?,?,?,?,?,?)",
                (rid, dataset_key, task, subtype, metric,
                 "queued", _utcnow(), json.dumps(params or {})),
            )
        row = self.get(rid)
        assert row is not None
        return row

    def mark_running(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='running', started_at=? WHERE run_id=?",
                (_utcnow(), run_id),
            )

    def update_progress(self, run_id: str, progress: dict[str, Any]) -> None:
        """Day 26: overwrite the live-progress snapshot for a running crew run.

        Written once per node visit by the streaming runner (~15 small writes
        per run), read by the dashboard's /status polling. Progress is a
        courtesy view, never part of the record — a lost write costs nothing.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET progress=? WHERE run_id=?",
                (json.dumps(progress, default=str), run_id),
            )

    def finish_success(self, run_id: str, *, record: dict[str, Any],
                       manifest: dict[str, Any], model_card: str,
                       telemetry: Optional[dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='succeeded', finished_at=?,"
                " record=?, manifest=?, model_card=?, telemetry=? WHERE run_id=?",
                (_utcnow(), json.dumps(record, default=str),
                 json.dumps(manifest, default=str), model_card,
                 json.dumps(telemetry, default=str) if telemetry else None,
                 run_id),
            )

    def finish_failure(self, run_id: str, error: str, *,
                       telemetry: Optional[dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='failed', finished_at=?, error=?,"
                " telemetry=? WHERE run_id=?",
                (_utcnow(), error,
                 json.dumps(telemetry, default=str) if telemetry else None,
                 run_id),
            )

    # --- reads ---------------------------------------------------------------

    def get(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for field in ("params", "record", "manifest", "telemetry", "progress"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d

    # --- Day 25: service-level aggregates for GET /metrics -------------------

    def metrics(self) -> dict[str, Any]:
        """Aggregate every run's lifecycle + telemetry into one JSON payload.

        Computed in Python over a full scan: the store is single-host and
        small by construction (one row per crew run, runs take minutes), so a
        readable aggregation beats a page of SQL. All scores are CV-on-train —
        the payload says so — and higher-is-better for every protocol metric,
        so ``best`` is a plain max.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT dataset_key, metric, status, record, telemetry FROM runs"
            ).fetchall()

        by_status = {s: 0 for s in STATUSES}
        durations: list[float] = []
        llm = {"n_calls": 0, "n_refused": 0, "tokens_spent": 0, "llm_time_s": 0.0}
        cache = {"n_hits": 0, "n_misses": 0, "n_bypassed": 0, "n_stored": 0}
        datasets: dict[str, dict[str, Any]] = {}

        for raw in rows:
            row = self._to_dict(raw)
            by_status[row["status"]] = by_status.get(row["status"], 0) + 1

            ds = datasets.setdefault(row["dataset_key"], {
                "metric": row.get("metric"), "n_runs": 0, "n_succeeded": 0,
                "scores": [],
            })
            ds["n_runs"] += 1

            tel = row.get("telemetry") or {}
            if tel.get("duration_s") is not None:
                durations.append(float(tel["duration_s"]))
            tl = tel.get("llm") or {}
            llm["n_calls"] += tl.get("n_calls", 0)
            llm["n_refused"] += tl.get("n_refused", 0)
            llm["tokens_spent"] += tl.get("tokens_spent", 0)
            llm["llm_time_s"] = round(llm["llm_time_s"] + tl.get("llm_time_s", 0.0), 3)
            tc = tel.get("cache") or {}
            for k in cache:
                cache[k] += tc.get(k, 0)

            if row["status"] == "succeeded":
                ds["n_succeeded"] += 1
                score = ((row.get("record") or {}).get("final_model") or {}
                         ).get("final_cv_score")
                if score is not None:
                    ds["scores"].append(float(score))

        for ds in datasets.values():
            scores = ds.pop("scores")
            ds["mean_cv_score"] = round(sum(scores) / len(scores), 6) if scores else None
            ds["best_cv_score"] = round(max(scores), 6) if scores else None

        finished = by_status["succeeded"] + by_status["failed"]
        attempted = cache["n_hits"] + cache["n_misses"]
        return {
            "runs": {
                "total": len(rows),
                "by_status": by_status,
                "success_rate": (round(by_status["succeeded"] / finished, 4)
                                 if finished else None),
            },
            "latency": {
                "n_measured": len(durations),
                "mean_s": round(sum(durations) / len(durations), 3) if durations else None,
                "p50_s": _quantile(durations, 0.50),
                "p95_s": _quantile(durations, 0.95),
            },
            "llm": llm,
            "cache": {**cache,
                      "hit_rate": (round(cache["n_hits"] / attempted, 4)
                                   if attempted else None)},
            "datasets": datasets,
            "cv_score_is_holdout": False,
        }


def _quantile(values: list[float], q: float) -> Optional[float]:
    """Nearest-rank quantile — deterministic, no interpolation surprises."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 3)
