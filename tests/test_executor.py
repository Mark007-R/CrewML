"""Day 6 guards: the sandboxed executor honours its contract.

The executor is the crux tool every real agent runs code through, so these tests
pin the whole contract — not just the happy path:

  * captured stdout + a clean success;
  * the metrics protocol (crew_io.emit_metrics ⇒ parsed metrics dict) and artifact
    collection;
  * failures are **reported, never raised** — a crash, a non-zero exit, malformed
    metrics.json, and a timed-out infinite loop each come back as a structured
    ExecResult with ok=False;
  * isolation — each run gets its own workdir; inputs are staged in; a missing
    input source is the one thing that legitimately raises;
  * honesty — the executor never references the holdout loader (structural
    no-peeking), matching the crew package's guarantee.
"""
from __future__ import annotations

import inspect
import textwrap

import pytest

from crewml import executor
from crewml.config import EXECUTOR_TIMEOUT_S


# --- Happy path: capture + clean exit ---------------------------------------

def test_stdout_captured_and_ok_on_clean_exit():
    res = executor.run_code("print('hello from sandbox')\n")
    assert res.ok is True
    assert res.returncode == 0
    assert res.timed_out is False
    assert "hello from sandbox" in res.stdout
    assert res.error is None


def test_default_timeout_is_config_value(monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        raise AssertionError("stop after capturing timeout")

    monkeypatch.setattr(executor.subprocess, "run", fake_run)
    with pytest.raises(AssertionError):
        executor.run_code("print('x')\n")
    assert captured["timeout"] == EXECUTOR_TIMEOUT_S


# --- The metrics + artifact protocol ----------------------------------------

def test_emit_metrics_round_trips_via_crew_io():
    code = textwrap.dedent(
        """\
        from crew_io import emit_metrics
        emit_metrics(cv_score=0.83, model="hgb")
        emit_metrics(extra=1)  # merge, not clobber
        """
    )
    res = executor.run_code(code)
    assert res.ok
    assert res.metrics == {"cv_score": 0.83, "model": "hgb", "extra": 1}


def test_artifacts_are_collected():
    code = textwrap.dedent(
        """\
        from crew_io import artifact_path
        artifact_path("model.bin").write_bytes(b"weights")
        (artifact_path("sub/nested.txt")).parent.mkdir(parents=True, exist_ok=True)
        artifact_path("sub/nested.txt").write_text("hi")
        """
    )
    res = executor.run_code(code)
    assert res.ok
    assert "model.bin" in res.artifacts
    assert "sub/nested.txt" in res.artifacts  # nested, forward-slashed


def test_seed_is_exposed_to_child():
    code = "from crew_io import SEED\nprint('SEED=%d' % SEED)\n"
    res = executor.run_code(code)
    assert res.ok
    assert "SEED=42" in res.stdout


# --- Failures are reported, never raised ------------------------------------

def test_crash_is_reported_not_raised():
    res = executor.run_code("raise ValueError('boom')\n")
    assert res.ok is False
    assert res.returncode not in (0, None)
    assert "ValueError" in res.error
    assert "boom" in res.error


def test_nonzero_exit_is_a_failure():
    res = executor.run_code("import sys\nsys.exit(3)\n")
    assert res.ok is False
    assert res.returncode == 3


def test_timeout_kills_infinite_loop():
    res = executor.run_code("while True:\n    pass\n", timeout_s=2)
    assert res.ok is False
    assert res.timed_out is True
    assert res.returncode is None
    assert "timeout" in res.error.lower()


def test_malformed_metrics_is_a_warning_not_a_crash():
    code = textwrap.dedent(
        """\
        import os
        with open(os.environ["CREWML_METRICS_PATH"], "w") as f:
            f.write("{not valid json")
        print("done")
        """
    )
    res = executor.run_code(code)
    assert res.ok  # the run itself succeeded
    assert res.metrics == {}
    assert any("metrics.json" in w for w in res.warnings)


def test_non_object_metrics_is_ignored_with_warning():
    code = textwrap.dedent(
        """\
        import json, os
        with open(os.environ["CREWML_METRICS_PATH"], "w") as f:
            json.dump([1, 2, 3], f)
        """
    )
    res = executor.run_code(code)
    assert res.ok
    assert res.metrics == {}
    assert res.warnings


# --- Isolation + inputs ------------------------------------------------------

def test_each_run_gets_its_own_workdir():
    a = executor.run_code("print(1)\n")
    b = executor.run_code("print(2)\n")
    assert a.workdir and b.workdir and a.workdir != b.workdir
    assert a.run_id != b.run_id


def test_inputs_are_staged_into_workdir(tmp_path):
    src = tmp_path / "payload.txt"
    src.write_text("the-answer-42")
    code = textwrap.dedent(
        """\
        from crew_io import input_path
        print("READBACK:" + input_path("payload.txt").read_text())
        """
    )
    res = executor.run_code(code, inputs={"payload.txt": src})
    assert res.ok
    assert "READBACK:the-answer-42" in res.stdout


def test_missing_input_source_raises():
    with pytest.raises(FileNotFoundError):
        executor.run_code("print('x')\n", inputs={"nope.parquet": "does/not/exist.parquet"})


def test_keep_workdir_false_removes_dir():
    import os

    res = executor.run_code("print('ephemeral')\n", keep_workdir=False)
    assert res.ok
    assert res.workdir == ""


def test_stable_run_id_reuses_and_cleans_workdir():
    first = executor.run_code(
        "from crew_io import artifact_path\nartifact_path('a.txt').write_text('1')\n",
        run_id="fixed-demo-run",
    )
    assert "a.txt" in first.artifacts
    # Re-running the same id starts clean — the stale artifact must not linger.
    second = executor.run_code("print('clean')\n", run_id="fixed-demo-run")
    assert second.run_id == "fixed-demo-run"
    assert second.artifacts == []


# --- Result shape ------------------------------------------------------------

def test_as_dict_is_json_friendly_and_omits_streams():
    import json

    res = executor.run_code("from crew_io import emit_metrics\nemit_metrics(x=1)\n")
    d = res.as_dict()
    json.dumps(d)  # must not raise
    assert "stdout" not in d and "stderr" not in d
    assert d["metrics"] == {"x": 1}
    assert d["ok"] is True


# --- Honesty: structural no-peeking -----------------------------------------

def test_executor_never_references_the_holdout():
    src = inspect.getsource(executor)
    assert "load_holdout" not in src
    assert "holdout" not in src.lower()
