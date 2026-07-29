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
