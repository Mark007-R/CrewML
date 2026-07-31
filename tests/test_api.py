"""Day 24 — API guarantees: real routes over an injected executor.

These tests exercise the actual FastAPI routes, the real store, and the real
JobRunner lifecycle; only ``execute`` is faked, so a test run is milliseconds
instead of a full crew invocation. One test drives the true async worker thread
end-to-end. The honesty tests pin the API-boundary invariants: no holdout
reference ever crosses into params or the store, and every surfaced score is
labelled CV-on-train.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from crewml.api.app import create_app
from crewml.api.jobs import JobRunner
from crewml.api.store import RunStore
from crewml.datasets import REGISTRY

FAKE_RESULT = {
    "record": {
        "dataset_key": "credit-g",
        "metric": "roc_auc",
        "iterations_run": 1,
        "final_model": {"kind": "single", "final_cv_score": 0.79},
        "holdout_untouched": True,
        "cv_score_is_holdout": False,
    },
    "manifest": {"schema_version": 1, "result_fingerprint": "f" * 64},
    "model_card": "# Model Card\n\nCV estimate; holdout untouched.",
    "telemetry": {
        "schema_version": 1,
        "duration_s": 12.5,
        "llm": {"n_calls": 3, "n_refused": 0, "tokens_spent": 2400,
                "llm_time_s": 2.1},
        "cache": {"n_hits": 1, "n_misses": 1, "n_bypassed": 0, "n_stored": 1,
                  "hit_rate": 0.5},
        "outcome": {"final_cv_score": 0.79, "cv_score_is_holdout": False},
    },
}


@pytest.fixture()
def harness(tmp_path):
    """(client, store, seen) with a sync fake executor recording its params."""
    seen: list[dict] = []

    def fake_execute(params):
        seen.append(params)
        return FAKE_RESULT

    store = RunStore(tmp_path / "runs.sqlite")
    runner = JobRunner(store, execute=fake_execute, sync=True)
    client = TestClient(create_app(store, runner))
    return client, store, seen


def test_healthz_lists_datasets(harness):
    client, _, _ = harness
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["datasets"] == list(REGISTRY)
    assert "mock_mode" in body


def test_submit_status_report_roundtrip(harness):
    client, _, _ = harness
    resp = client.post("/run", json={"dataset_key": "credit-g"})
    assert resp.status_code == 202
    rid = resp.json()["run_id"]

    status = client.get(f"/status/{rid}").json()
    assert status["status"] == "succeeded"
    assert status["metric"] == "roc_auc"
    assert status["headline"]["final_cv_score"] == 0.79
    assert status["headline"]["cv_score_is_holdout"] is False

    report = client.get(f"/report/{rid}").json()
    assert report["record"]["final_model"]["kind"] == "single"
    assert report["manifest"]["result_fingerprint"] == "f" * 64
    assert report["model_card"].startswith("# Model Card")


def test_unknown_dataset_404(harness):
    client, _, _ = harness
    resp = client.post("/run", json={"dataset_key": "not-a-dataset"})
    assert resp.status_code == 404
    assert "known" in resp.json()["detail"]


def test_bad_options_422(harness):
    client, _, _ = harness
    assert client.post("/run", json={}).status_code == 422
    assert client.post(
        "/run", json={"dataset_key": "credit-g", "max_iterations": 0}
    ).status_code == 422


def test_status_and_report_404_on_unknown_run(harness):
    client, _, _ = harness
    assert client.get("/status/deadbeef").status_code == 404
    assert client.get("/report/deadbeef").status_code == 404


def test_report_conflict_while_pending(tmp_path):
    """A queued (never-executed) run answers 409 on /report, not a partial body."""
    store = RunStore(tmp_path / "runs.sqlite")

    class NeverRuns(JobRunner):
        def submit(self, run_id, params):  # queue without a worker
            pass

    client = TestClient(create_app(store, NeverRuns(store)))
    rid = client.post("/run", json={"dataset_key": "diabetes"}).json()["run_id"]
    assert client.get(f"/status/{rid}").json()["status"] == "queued"
    resp = client.get(f"/report/{rid}")
    assert resp.status_code == 409
    assert resp.json()["detail"]["status"] == "queued"


def test_failed_run_recorded_and_reported(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")

    def boom(params):
        raise ValueError("executor exploded")

    runner = JobRunner(store, execute=boom, sync=True)
    client = TestClient(create_app(store, runner))
    rid = client.post("/run", json={"dataset_key": "vehicle"}).json()["run_id"]

    status = client.get(f"/status/{rid}").json()
    assert status["status"] == "failed"
    assert "executor exploded" in status["error"]

    resp = client.get(f"/report/{rid}")
    assert resp.status_code == 409
    assert resp.json()["detail"]["status"] == "failed"


def test_async_worker_thread_completes(tmp_path):
    """The real queue + daemon worker drives a run to succeeded."""
    store = RunStore(tmp_path / "runs.sqlite")
    runner = JobRunner(store, execute=lambda p: FAKE_RESULT, sync=False)
    client = TestClient(create_app(store, runner))
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]

    assert runner.wait_idle(timeout=10)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if store.get(rid)["status"] == "succeeded":
            break
        time.sleep(0.05)
    assert store.get(rid)["status"] == "succeeded"


def test_worker_survives_a_crashing_run(tmp_path):
    """One failing run must not kill the worker for the next submission."""
    store = RunStore(tmp_path / "runs.sqlite")
    calls = {"n": 0}

    def flaky(params):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first run dies")
        return FAKE_RESULT

    runner = JobRunner(store, execute=flaky, sync=False)
    client = TestClient(create_app(store, runner))
    rid1 = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]
    rid2 = client.post("/run", json={"dataset_key": "diabetes"}).json()["run_id"]
    assert runner.wait_idle(timeout=10)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (store.get(rid1)["status"], store.get(rid2)["status"]) == ("failed", "succeeded"):
            break
        time.sleep(0.05)
    assert store.get(rid1)["status"] == "failed"
    assert store.get(rid2)["status"] == "succeeded"


# --- honesty at the API boundary --------------------------------------------

def test_no_holdout_reference_reaches_executor_or_store(harness):
    """The run request pins a dataset_key only — never a data path.

    EVAL_PROTOCOL's no-peeking rule is structural at the API boundary too: what
    the executor receives and what the store persists must not name the holdout
    or any filesystem location for it.
    """
    client, store, seen = harness
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]

    assert len(seen) == 1
    blob = json.dumps(seen[0]).lower()
    assert "holdout" not in blob and "test.parquet" not in blob

    row = store.get(rid)
    params_blob = json.dumps(row["params"]).lower()
    assert "holdout" not in params_blob and "path" not in params_blob


def test_run_request_schema_has_no_path_fields():
    from crewml.api.app import RunRequest
    fields = set(RunRequest.model_fields)
    assert fields == {"dataset_key", "max_iterations", "param_search", "llm",
                      "token_budget", "time_budget_s"}


# --- Day 25: telemetry surfaces + /metrics -----------------------------------

def test_status_carries_the_compact_cost_line(harness):
    client, _, _ = harness
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]
    tel = client.get(f"/status/{rid}").json()["telemetry"]
    assert tel == {"duration_s": 12.5, "tokens_spent": 2400, "llm_calls": 3,
                   "cache_hits": 1, "cache_hit_rate": 0.5}


def test_report_carries_full_telemetry(harness):
    client, _, _ = harness
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]
    tel = client.get(f"/report/{rid}").json()["telemetry"]
    assert tel["llm"]["tokens_spent"] == 2400
    assert tel["cache"]["n_hits"] == 1
    assert tel["outcome"]["cv_score_is_holdout"] is False


def test_metrics_aggregates_over_runs(harness):
    client, _, _ = harness
    for _ in range(2):
        client.post("/run", json={"dataset_key": "credit-g"})

    m = client.get("/metrics").json()
    assert m["service"]["version"]
    assert m["runs"]["total"] == 2
    assert m["runs"]["by_status"]["succeeded"] == 2
    assert m["runs"]["success_rate"] == 1.0
    assert m["latency"]["mean_s"] == 12.5
    assert m["llm"]["tokens_spent"] == 4800
    assert m["cache"]["hit_rate"] == 0.5
    assert m["datasets"]["credit-g"]["best_cv_score"] == 0.79
    assert m["cv_score_is_holdout"] is False


def test_metrics_on_a_fresh_service_is_empty_not_broken(harness):
    client, _, _ = harness
    m = client.get("/metrics").json()
    assert m["runs"]["total"] == 0
    assert m["runs"]["success_rate"] is None
    assert m["datasets"] == {}


def test_failed_run_still_gets_duration_telemetry(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")

    def boom(params):
        raise ValueError("executor exploded")

    client = TestClient(create_app(store, JobRunner(store, execute=boom, sync=True)))
    rid = client.post("/run", json={"dataset_key": "vehicle"}).json()["run_id"]
    tel = store.get(rid)["telemetry"]
    assert tel["duration_s"] >= 0.0
    assert tel["llm"]["n_calls"] == 0  # spend died with the run's ledger — unmeasured, not invented


# --- Day 26: CSV upload + sealed split + live progress -----------------------

def _churn_csv(n: int = 120) -> bytes:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "income": rng.normal(50_000, 12_000, n).round(2),
        "churned": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    })
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture()
def upload_harness(tmp_path, monkeypatch):
    """API harness whose upload storage lands in tmp, registry restored after."""
    import crewml.datasets as datasets_mod
    import crewml.uploads as uploads_mod

    monkeypatch.setattr(datasets_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(uploads_mod, "DATA_DIR", tmp_path / "data")
    seen: list[dict] = []

    def fake_execute(params):
        seen.append(params)
        return FAKE_RESULT

    store = RunStore(tmp_path / "runs.sqlite")
    runner = JobRunner(store, execute=fake_execute, sync=True)
    client = TestClient(create_app(store, runner))
    before = set(REGISTRY)
    yield client, store, seen
    for key in set(REGISTRY) - before:
        REGISTRY.pop(key, None)


def test_upload_then_run_roundtrip(upload_harness):
    """A CSV with a CHOSEN target ingests, seals, registers, and runs."""
    client, _, seen = upload_harness
    resp = client.post(
        "/upload",
        files={"file": ("churn.csv", _churn_csv(), "text/csv")},
        data={"target_column": "churned"},
    )
    assert resp.status_code == 201
    body = resp.json()
    key = body["dataset_key"]
    man = body["manifest"]
    assert key.startswith("upload-")
    # the caller sees exactly what was derived and sealed before running
    assert man["derivation"]["metric"] == "roc_auc"
    assert man["derivation"]["rule"]
    assert len(man["holdout_sha256"]) == 64
    assert man["n_train"] + man["n_holdout"] == 120

    run = client.post("/run", json={"dataset_key": key})
    assert run.status_code == 202
    rid = run.json()["run_id"]
    assert client.get(f"/status/{rid}").json()["status"] == "succeeded"
    # the executor still receives a key, never a path (same rule as benchmarks)
    blob = json.dumps(seen[0]).lower()
    assert "holdout" not in blob and "parquet" not in blob


def test_upload_appears_in_datasets_route(upload_harness):
    client, _, _ = upload_harness
    key = client.post(
        "/upload",
        files={"file": ("churn.csv", _churn_csv(), "text/csv")},
        data={"target_column": "churned"},
    ).json()["dataset_key"]
    listed = client.get("/datasets").json()["datasets"]
    assert key in listed and listed[key]["metric"] == "roc_auc"


def test_upload_bad_target_column_is_400_with_columns(upload_harness):
    client, _, _ = upload_harness
    resp = client.post(
        "/upload",
        files={"file": ("churn.csv", _churn_csv(), "text/csv")},
        data={"target_column": "not_a_column"},
    )
    assert resp.status_code == 400
    assert "churned" in resp.json()["detail"]  # available columns are named


def test_upload_without_target_choice_is_422(upload_harness):
    """No target, no ingestion — the form field is required, never defaulted."""
    client, _, _ = upload_harness
    resp = client.post(
        "/upload", files={"file": ("churn.csv", _churn_csv(), "text/csv")},
    )
    assert resp.status_code == 422


def test_upload_unparseable_csv_is_400(upload_harness):
    client, _, _ = upload_harness
    resp = client.post(
        "/upload",
        files={"file": ("junk.bin", b"\x00\x01\xff\xfe", "text/csv")},
        data={"target_column": "y"},
    )
    assert resp.status_code == 400


def test_status_surfaces_live_progress_while_running(upload_harness):
    """The dashboard's trace: progress shows mid-run and disappears after."""
    client, store, _ = upload_harness
    row = store.create("credit-g", task="classification", subtype="binary",
                       metric="roc_auc", params={})
    rid = row["run_id"]
    store.mark_running(rid)
    store.update_progress(rid, {"trace": ["profiler", "planner"],
                                "nodes_visited": 2, "current_node": "planner",
                                "iteration": 0, "decisions": []})
    status = client.get(f"/status/{rid}").json()
    assert status["progress"]["current_node"] == "planner"
    assert status["progress"]["trace"] == ["profiler", "planner"]

    store.finish_success(rid, record=FAKE_RESULT["record"],
                         manifest=FAKE_RESULT["manifest"], model_card="x")
    assert "progress" not in client.get(f"/status/{rid}").json()


def test_jobrunner_feeds_progress_to_the_store(tmp_path):
    """An executor that accepts on_progress gets a live line to the store."""
    store = RunStore(tmp_path / "runs.sqlite")

    def stepping_execute(params, on_progress=None):
        for i, node in enumerate(["profiler", "planner", "trainer"], 1):
            on_progress({"trace": ["profiler", "planner", "trainer"][:i],
                         "nodes_visited": i, "current_node": node})
        return FAKE_RESULT

    runner = JobRunner(store, execute=stepping_execute, sync=True)
    client = TestClient(create_app(store, runner))
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]
    row = store.get(rid)
    assert row["status"] == "succeeded"
    assert row["progress"]["nodes_visited"] == 3  # last snapshot persisted


def test_jobrunner_still_accepts_plain_executors(tmp_path):
    """Feature detection: a single-arg fake (the Day-24 contract) still works."""
    store = RunStore(tmp_path / "runs.sqlite")
    runner = JobRunner(store, execute=lambda params: FAKE_RESULT, sync=True)
    client = TestClient(create_app(store, runner))
    rid = client.post("/run", json={"dataset_key": "credit-g"}).json()["run_id"]
    assert store.get(rid)["status"] == "succeeded"
