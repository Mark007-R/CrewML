"""Day 24 — run-store guarantees: lifecycle, persistence, and no-holdout schema."""
from __future__ import annotations

import json
import sqlite3

import pytest

from crewml.api.store import RunStore, new_run_id


@pytest.fixture()
def store(tmp_path):
    return RunStore(tmp_path / "runs.sqlite")


def test_create_starts_queued_with_params(store):
    row = store.create("credit-g", task="classification", subtype="binary",
                       metric="roc_auc", params={"llm": False})
    assert row["status"] == "queued"
    assert row["dataset_key"] == "credit-g"
    assert row["params"] == {"llm": False}
    assert row["created_at"] and row["started_at"] is None


def test_full_success_lifecycle(store):
    rid = store.create("diabetes", metric="roc_auc")["run_id"]
    store.mark_running(rid)
    assert store.get(rid)["status"] == "running"
    assert store.get(rid)["started_at"] is not None

    store.finish_success(rid, record={"final_model": {"kind": "single"}},
                         manifest={"result_fingerprint": "abc"},
                         model_card="# Model Card")
    row = store.get(rid)
    assert row["status"] == "succeeded"
    assert row["finished_at"] is not None
    assert row["record"]["final_model"]["kind"] == "single"
    assert row["manifest"]["result_fingerprint"] == "abc"
    assert row["model_card"].startswith("# Model Card")
    assert row["error"] is None


def test_failure_records_error(store):
    rid = store.create("vehicle")["run_id"]
    store.mark_running(rid)
    store.finish_failure(rid, "ValueError: boom")
    row = store.get(rid)
    assert row["status"] == "failed"
    assert "boom" in row["error"]
    assert row["record"] is None


def test_get_unknown_returns_none(store):
    assert store.get("nope") is None


def test_list_newest_first_and_limit(store):
    ids = [store.create(f"credit-g")["run_id"] for _ in range(3)]
    rows = store.list(limit=2)
    assert len(rows) == 2
    # same-second timestamps are broken by rowid, so newest insert leads
    assert rows[0]["run_id"] == ids[-1]


def test_invalid_status_rejected_by_schema(store):
    with pytest.raises(sqlite3.IntegrityError):
        with store._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, dataset_key, status, created_at, params)"
                " VALUES (?,?,?,?,?)",
                (new_run_id(), "credit-g", "done", "2026-07-29", "{}"),
            )


def test_schema_has_no_holdout_column(store):
    """Structural honesty: the store cannot record a holdout location."""
    with store._connect() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
    assert not any("holdout" in c.lower() or "test" in c.lower() for c in cols)


def test_run_ids_unique():
    assert len({new_run_id() for _ in range(200)}) == 200
