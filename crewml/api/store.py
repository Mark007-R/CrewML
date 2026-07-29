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
    error       TEXT
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

    def finish_success(self, run_id: str, *, record: dict[str, Any],
                       manifest: dict[str, Any], model_card: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='succeeded', finished_at=?,"
                " record=?, manifest=?, model_card=? WHERE run_id=?",
                (_utcnow(), json.dumps(record, default=str),
                 json.dumps(manifest, default=str), model_card, run_id),
            )

    def finish_failure(self, run_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status='failed', finished_at=?, error=?"
                " WHERE run_id=?",
                (_utcnow(), error, run_id),
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
        for field in ("params", "record", "manifest"):
            if d.get(field):
                d[field] = json.loads(d[field])
        return d
