"""Day 19 adversarial suite: the hardened executor refuses what it must.

Every test here runs *hostile-ish* code through the real executor and asserts
the refusal (or the allowance — the DS stack and loopback IPC must keep
working, or the sandbox would just be an elaborate off switch). The suite pins
the exact boundary documented in crewml/sandbox.py:

  * network egress refused (DNS and raw connect), loopback allowed;
  * writes/renames outside the workdir jail refused, writes inside allowed;
  * reads of parent-named deny roots refused — including the raw dataset store
    under the default policy (staged-inputs-only, structurally);
  * non-allowlisted imports refused (pip, and crewml itself — the child cannot
    even import the loaders), the DS stack importable;
  * shell-outs refused, same-Python children allowed (loky must survive);
  * memory cap kills a hog; timeout still works under the guard;
  * stream flood truncated; guard failures fail closed (exit 97 path).

Honesty: these tests prove defence-in-depth against careless generated code.
They deliberately do NOT claim ctypes/_winapi escapes are impossible — see the
sandbox module docstring.
"""
from __future__ import annotations

import os
import sys
import textwrap

import pytest

from crewml import executor
from crewml.config import DATA_DIR
from crewml.sandbox import DEFAULT_ALLOWED_IMPORTS, SandboxPolicy, default_policy

# A policy with no memory cap — most tests don't want watchdog noise.
POLICY = SandboxPolicy(active=True, mem_mb=0)


def run(code: str, **kw):
    kw.setdefault("sandbox", POLICY)
    return executor.run_code(textwrap.dedent(code), **kw)


# --- Policy plumbing ----------------------------------------------------------


def test_default_policy_is_active_and_denies_the_dataset_store():
    pol = default_policy()
    assert pol.active is True
    assert str(DATA_DIR) in pol.read_deny
    assert pol.mem_mb > 0


def test_result_reports_sandboxed_flag():
    res = run("print('ok')\n")
    assert res.ok and res.sandboxed is True
    d = res.as_dict()
    assert d["sandboxed"] is True and d["oom"] is False


def test_sandbox_can_be_switched_off_explicitly():
    res = executor.run_code("import pip\nprint('unsandboxed')\n",
                            sandbox=SandboxPolicy(active=False))
    assert res.ok and res.sandboxed is False


# --- Network ------------------------------------------------------------------


def test_dns_egress_refused():
    res = run(
        """\
        import urllib.request
        urllib.request.urlopen("http://example.com", timeout=10)
        """
    )
    assert res.ok is False
    assert "sandbox" in res.stderr.lower()


def test_raw_socket_connect_refused_without_dns():
    res = run(
        """\
        import socket
        s = socket.socket()
        s.settimeout(5)
        s.connect(("93.184.216.34", 80))
        """
    )
    assert res.ok is False
    assert "network egress refused" in res.stderr


def test_loopback_stays_available_for_library_ipc():
    res = run(
        """\
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        print("LOOPBACK_OK", s.getsockname()[1] > 0)
        s.close()
        """
    )
    assert res.ok, res.stderr
    assert "LOOPBACK_OK True" in res.stdout


# --- Filesystem jail ------------------------------------------------------------


def test_write_outside_jail_refused_and_nothing_created():
    escape = executor.EXECUTOR_DIR / "escape.txt"
    res = run(
        """\
        with open("../escape.txt", "w") as f:
            f.write("out")
        """
    )
    assert res.ok is False
    assert "write outside workdir refused" in res.stderr
    assert not escape.exists()


def test_os_open_write_flags_outside_jail_refused():
    res = run(
        """\
        import os
        os.open(os.path.join("..", "escape2.bin"), os.O_CREAT | os.O_WRONLY)
        """
    )
    assert res.ok is False
    assert "write outside workdir refused" in res.stderr
    assert not (executor.EXECUTOR_DIR / "escape2.bin").exists()


def test_rename_outside_jail_refused(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    res = run(
        f"""\
        import os
        os.rename({str(victim)!r}, {str(victim) + ".gone"!r})
        """
    )
    assert res.ok is False
    assert victim.exists()  # untouched


def test_writes_inside_jail_and_devnull_allowed():
    res = run(
        """\
        import os
        from crew_io import artifact_path, emit_metrics
        artifact_path("model.bin").write_bytes(b"w")
        emit_metrics(score=1.0)
        with open(os.devnull, "w") as f:
            f.write("gone")
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as tf:
            tf.write(b"scratch")  # temp dir is redirected into the jail
        print("WRITES_OK")
        """
    )
    assert res.ok, res.stderr
    assert "model.bin" in res.artifacts and res.metrics == {"score": 1.0}


def test_read_deny_root_refused_but_staged_copy_readable(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("y\n1\n")
    pol = SandboxPolicy(active=True, read_deny=(str(tmp_path),), mem_mb=0)
    # Direct read of the denied root: refused.
    res = executor.run_code(
        f"open({str(secret)!r}).read()\n", sandbox=pol
    )
    assert res.ok is False
    assert "read of denied root refused" in res.stderr
    # The staged copy inside the workdir: fine — staged-inputs-only, enforced.
    res2 = executor.run_code(
        "from crew_io import input_path\n"
        "print('STAGED:' + input_path('secret.csv').read_text().strip().split()[0])\n",
        sandbox=pol,
        inputs={"secret.csv": secret},
    )
    assert res2.ok, res2.stderr
    assert "STAGED:y" in res2.stdout


def test_dataset_store_read_refused_under_default_policy():
    probe = DATA_DIR / "_sandbox_probe.txt"
    probe.write_text("locked")
    try:
        res = executor.run_code(
            f"open({str(probe)!r}).read()\n", sandbox=default_policy()
        )
        assert res.ok is False
        assert "read of denied root refused" in res.stderr
    finally:
        probe.unlink(missing_ok=True)


# --- Import allowlist -----------------------------------------------------------


def test_non_allowlisted_import_refused():
    res = run("import pip\n")
    assert res.ok is False
    assert "import of 'pip' is not allowed" in res.stderr


def test_child_cannot_import_crewml_itself():
    # Upgrades Day 6's "executor never references the loaders" to
    # "the child structurally cannot import them".
    res = run("import crewml\n")
    assert res.ok is False
    assert "not allowed" in res.stderr


def test_ds_stack_imports_allowed():
    res = run(
        """\
        import numpy, pandas, scipy, sklearn, joblib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("STACK_OK")
        """,
        timeout_s=180,
    )
    assert res.ok, res.stderr
    assert "STACK_OK" in res.stdout


def test_allowlist_is_policy_configurable():
    assert "pip" not in DEFAULT_ALLOWED_IMPORTS
    pol = SandboxPolicy(active=True, imports=DEFAULT_ALLOWED_IMPORTS | {"pip"}, mem_mb=0)
    res = executor.run_code("import pip\nprint('PIP_OK')\n", sandbox=pol, timeout_s=180)
    assert res.ok, res.stderr


# --- Processes -------------------------------------------------------------------


def test_shell_out_refused():
    res = run("import os\nos.system('echo pwned')\n")
    assert res.ok is False
    assert "os.system refused" in res.stderr


def test_spawning_other_binaries_refused():
    shell, flag = ("cmd.exe", "/c") if os.name == "nt" else ("/bin/sh", "-c")
    res = run(
        f"""\
        import subprocess
        subprocess.run([{shell!r}, {flag!r}, "echo hi"])
        """
    )
    assert res.ok is False
    assert "refused" in res.stderr


def test_same_python_child_allowed():  # loky/GridSearchCV(n_jobs=-1) must survive
    res = run(
        """\
        import subprocess, sys
        out = subprocess.run([sys.executable, "-c", "print('SUB_OK')"],
                             capture_output=True, text=True)
        print(out.stdout.strip())
        """
    )
    assert res.ok, res.stderr
    assert "SUB_OK" in res.stdout


def test_parallel_cv_survives_the_sandbox():
    res = run(
        """\
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 5)); y = (X[:, 0] > 0).astype(int)
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=10, random_state=0),
            X, y, cv=3, n_jobs=-1)
        print("CV_OK", len(scores) == 3)
        """,
        timeout_s=300,
    )
    assert res.ok, res.stderr
    assert "CV_OK True" in res.stdout


# --- Resource caps ----------------------------------------------------------------


def test_memory_cap_kills_a_hog():
    pol = SandboxPolicy(active=True, mem_mb=256)
    res = executor.run_code(
        # os.urandom chunks so every page is actually touched — a zeroed
        # bytearray can stay lazily committed and fool a working-set watchdog.
        "import os, time\n"
        "chunks = [bytearray(os.urandom(10 * 1024 * 1024)) for _ in range(80)]\n"
        "time.sleep(10)\n",
        sandbox=pol,
        timeout_s=60,
    )
    assert res.ok is False
    assert res.oom is True
    assert "memory cap" in res.error


def test_timeout_still_enforced_under_sandbox():
    res = run("while True:\n    pass\n", timeout_s=3)
    assert res.ok is False and res.timed_out is True


def test_stream_flood_is_truncated():
    res = run("print('x' * 3_000_000)\n")
    assert res.ok, res.stderr
    assert len(res.stdout) <= executor.MAX_STREAM_CHARS + 100
    assert any("truncated" in w for w in res.warnings)


# --- Fail-closed boot --------------------------------------------------------------


def test_guard_boot_fails_closed_when_unconfigured():
    # Simulate a run where the sandbox was requested but the config vanished:
    # guard_boot must exit 97, never fall through to main.py.
    res = run("print('should never run')\n", env={"CREWML_SB": "0"})
    assert res.ok is False
    assert res.returncode == 97
    assert "should never run" not in res.stdout


def test_guard_files_are_not_reported_as_artifacts():
    res = run("print('clean')\n")
    assert res.ok and res.artifacts == []
