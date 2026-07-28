"""The sandboxed Python executor — the crew's central shared tool (Day 6).

Every *real* agent that writes code (Feature Engineer, Trainer, and the Day-20
self-repair loop) hands that code to **one** place: :func:`run_code`. The executor
runs it in a **fresh subprocess** with a **wall-clock timeout**, in an **isolated
temp workdir**, and gives the agent back a single structured :class:`ExecResult`
— stdout/stderr, exit status, a parsed ``metrics.json``, and a list of the
artifacts the code produced. Nothing else in the crew shells out to Python; this
module is the crux the whole pipeline runs on.

What Day 6 delivers (the *tool* and its contract):

* **Subprocess isolation** — generated code runs in its own interpreter process
  (``sys.executable``), never ``exec``'d into the crew process, so a crash,
  ``sys.exit``, or runaway allocation takes down only the child.
* **Timeout** — a hard wall-clock cap (default :data:`config.EXECUTOR_TIMEOUT_S`);
  on expiry the child is killed and ``timed_out=True`` is reported, never a hang.
* **Isolated workdir** — each call gets a fresh directory under
  ``artifacts/executor/<run_id>/``; inputs are copied *in*, outputs stay *there*,
  and ``cwd`` is set to it so relative paths can't wander.
* **Captured output** — full stdout/stderr are returned *and* written to
  ``stdout.log`` / ``stderr.log`` for the Day-26 dashboard and post-mortems.
* **A metrics + artifacts protocol** — a tiny helper module (:data:`HELPER_SOURCE`,
  written into the workdir as ``crew_io.py``) lets executed code do
  ``from crew_io import emit_metrics, artifact_path`` to hand structured numbers
  and files back across the process boundary. No helper import is required — code
  may also just write ``metrics.json`` itself.

**Hardened since Day 19.** The security layer the Day-6 docstring promised now
exists: every run executes under a :class:`crewml.sandbox.SandboxPolicy` (import
allowlist, network egress refusal, filesystem write jail + read-deny roots,
memory/CPU caps), enforced by a guard installed inside the child interpreter —
see :mod:`crewml.sandbox` for mechanisms and the honesty scope (defence-in-depth
against careless generated code, not a hostile-adversary boundary; that arrives
with the Day-27 Docker wrapper). The Day-6 contract is unchanged: same call, same
:class:`ExecResult`, refusals surface as ordinary reported failures.

**No peeking, structurally.** The executor is data-agnostic: it copies in exactly
the input files the caller names and knows nothing about datasets or splits. It
never imports a held-out-set loader, so it cannot leak the locked test split into
a run — a property a test asserts by source inspection.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from crewml.config import ARTIFACTS_DIR, EXECUTOR_TIMEOUT_S, SEED
from crewml.sandbox import (
    GUARD_BOOT,
    GUARD_BOOT_SOURCE,
    GUARD_MODULE,
    GUARD_SOURCE,
    SITECUSTOMIZE,
    SITECUSTOMIZE_SOURCE,
    SandboxPolicy,
    default_policy,
    policy_env,
)

EXECUTOR_DIR = ARTIFACTS_DIR / "executor"

# Filenames the executor owns inside every workdir (agent code must not rely on
# clobbering these — they are read back by the trusted parent).
MAIN_SCRIPT = "main.py"
HELPER_MODULE = "crew_io.py"
METRICS_FILE = "metrics.json"
ARTIFACTS_SUBDIR = "artifacts"
STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"

# How many trailing stderr lines to surface as the short ``error`` summary.
_ERROR_TAIL_LINES = 8

# Cap on each captured stream (chars). Protects the *parent* from a child that
# floods stdout — the full stream still lands in the workdir logs up to this cap.
MAX_STREAM_CHARS = 1_000_000

# How often the Windows memory watchdog samples the child's working set.
_WATCHDOG_POLL_S = 0.2


# --- The helper injected into every workdir as ``crew_io.py`` ---------------

HELPER_SOURCE = '''\
"""crew_io — injected by the CrewML executor. Import me from generated code to
hand metrics and artifacts back to the crew across the subprocess boundary.

    from crew_io import emit_metrics, artifact_path, input_path, SEED
    emit_metrics(cv_score=0.83, model="hgb")     # merge-writes metrics.json
    joblib.dump(model, artifact_path("model.joblib"))
    train = pd.read_parquet(input_path("train.parquet"))

Nothing here is required — code may write metrics.json / the artifacts dir by
hand — but using it keeps the contract in one place.
"""
import json
import os
from pathlib import Path

WORKDIR = Path(os.environ["CREWML_WORKDIR"])
ARTIFACTS_DIR = Path(os.environ["CREWML_ARTIFACTS_DIR"])
METRICS_PATH = Path(os.environ["CREWML_METRICS_PATH"])
SEED = int(os.environ.get("CREWML_SEED", "42"))


def input_path(name):
    """Path to an input file the caller copied into the workdir."""
    return WORKDIR / name


def artifact_path(name):
    """Path under the artifacts dir to write a file the crew should keep."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / name


def emit_metrics(**kwargs):
    """Merge-write keyword numbers/values into metrics.json (repeatable)."""
    data = {}
    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text())
        except Exception:
            data = {}
    data.update(kwargs)
    METRICS_PATH.write_text(json.dumps(data))
    return data
'''


@dataclass
class ExecResult:
    """The single structured value :func:`run_code` returns.

    JSON-friendly (paths are strings) so it can be checkpointed into
    :class:`~crewml.crew.state.CrewState` and rendered by the Day-26 dashboard.
    """

    ok: bool                         # returncode == 0 and not timed out
    returncode: Optional[int]        # child exit code (None if timed out/killed)
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str
    error: Optional[str]             # short human summary on failure, else None
    metrics: dict[str, Any] = field(default_factory=dict)   # parsed metrics.json
    artifacts: list[str] = field(default_factory=list)      # relative artifact paths
    warnings: list[str] = field(default_factory=list)
    workdir: str = ""
    run_id: str = ""
    oom: bool = False                # killed/failed over the memory cap (Day 19)
    sandboxed: bool = False          # a SandboxPolicy was active for this run

    def as_dict(self) -> dict[str, Any]:
        """A crew-state-friendly summary — omits the (potentially large) streams."""
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "oom": self.oom,
            "sandboxed": self.sandboxed,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
            "metrics": self.metrics,
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "run_id": self.run_id,
            "workdir": self.workdir,
        }


def _new_workdir(run_id: str) -> Path:
    work = EXECUTOR_DIR / run_id
    if work.exists():  # a reused run_id starts clean
        shutil.rmtree(work, ignore_errors=True)
    (work / ARTIFACTS_SUBDIR).mkdir(parents=True, exist_ok=True)
    return work


def _collect_artifacts(artifacts_dir: Path) -> list[str]:
    """Relative paths of every file the code left under the artifacts dir."""
    if not artifacts_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(artifacts_dir)).replace(os.sep, "/")
        for p in artifacts_dir.rglob("*")
        if p.is_file()
    )


def _posix_rlimits(mem_mb: int, cpu_s: int):
    """A preexec_fn setting hard limits between fork and exec (POSIX only)."""

    def set_limits():  # pragma: no cover — POSIX branch, suite runs on Windows too
        import resource

        if mem_mb:
            cap = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))

    return set_limits


def _win_mem_watchdog(proc: subprocess.Popen, cap_bytes: int, hit: dict) -> None:
    """Poll the direct child's working set; kill it when it exceeds the cap.

    Windows has no inheritable rlimit, so the parent watches. Worker
    grandchildren are not summed — a documented limit of this mechanism.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        while proc.poll() is None:
            handle = int(getattr(proc, "_handle"))
            if get_info(handle, ctypes.byref(pmc), pmc.cb) and pmc.WorkingSetSize > cap_bytes:
                hit["oom"] = True
                proc.kill()
                return
            time.sleep(_WATCHDOG_POLL_S)
    except Exception:  # a broken watchdog must never take down the run
        return


def _truncated(stream: str, name: str, warnings: list[str]) -> str:
    if len(stream) <= MAX_STREAM_CHARS:
        return stream
    warnings.append(
        f"{name} truncated by executor at {MAX_STREAM_CHARS} chars "
        f"({len(stream)} produced)"
    )
    return stream[:MAX_STREAM_CHARS] + f"\n...[{name} truncated by executor]"


def run_code(
    code: str,
    *,
    inputs: Optional[Mapping[str, os.PathLike | str]] = None,
    timeout_s: Optional[int] = None,
    run_id: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    keep_workdir: bool = True,
    sandbox: Optional[SandboxPolicy] = None,
) -> ExecResult:
    """Execute agent-generated Python in an isolated subprocess and report back.

    Parameters
    ----------
    code:
        The Python source to run. Written verbatim to ``main.py`` in a fresh
        workdir and executed as ``python main.py`` with the workdir as ``cwd``.
    inputs:
        Optional ``{name: source_path}`` files copied into the workdir before the
        run (e.g. ``{"train.parquet": data/credit-g/train.parquet}``). The code
        reads them via ``crew_io.input_path(name)`` or a plain relative path.
        **Only what the caller names is provided** — the executor never fetches
        data itself, so it cannot smuggle in the locked test split.
    timeout_s:
        Wall-clock cap; defaults to :data:`config.EXECUTOR_TIMEOUT_S`. On expiry
        the child is killed and ``timed_out=True`` is returned (never a hang).
    run_id:
        Optional stable id (⇒ stable workdir), for reproducible/inspectable runs.
        Defaults to a random 12-hex id.
    env:
        Extra environment variables merged over the inherited environment and the
        executor's own ``CREWML_*`` vars.
    keep_workdir:
        Keep the workdir (default) so artifacts and logs remain inspectable; set
        ``False`` to delete it after reading results back.
    sandbox:
        The :class:`~crewml.sandbox.SandboxPolicy` to enforce; defaults to
        :func:`~crewml.sandbox.default_policy` (active unless
        ``CREWML_EXECUTOR_SANDBOX=0``). Pass ``SandboxPolicy(active=False)`` to
        run un-sandboxed deliberately.

    Returns
    -------
    ExecResult
        Never raises for *code* failures — a crash, non-zero exit, or timeout is
        reported in the result. Only genuine executor/host errors (e.g. a missing
        input source file) propagate.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    timeout_s = int(timeout_s if timeout_s is not None else EXECUTOR_TIMEOUT_S)
    policy = sandbox if sandbox is not None else default_policy()

    work = _new_workdir(run_id)
    artifacts_dir = work / ARTIFACTS_SUBDIR
    metrics_path = work / METRICS_FILE

    (work / MAIN_SCRIPT).write_text(code, encoding="utf-8")
    (work / HELPER_MODULE).write_text(HELPER_SOURCE, encoding="utf-8")

    # Stage inputs. A missing source is a caller bug, not agent-code failure — raise.
    for name, src in (inputs or {}).items():
        src_path = Path(src)
        if not src_path.is_file():
            raise FileNotFoundError(f"executor input {name!r}: no such file {src_path}")
        shutil.copy2(src_path, work / name)

    child_env = dict(os.environ)
    child_env.update(
        {
            "CREWML_WORKDIR": str(work),
            "CREWML_ARTIFACTS_DIR": str(artifacts_dir),
            "CREWML_METRICS_PATH": str(metrics_path),
            "CREWML_SEED": str(SEED),
            # Determinism knobs (Day-23 reproducibility leans on these).
            "PYTHONHASHSEED": str(SEED),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    argv = [sys.executable, MAIN_SCRIPT]
    preexec_fn = None
    if policy.active:
        # Guard files: fail-closed boot for the main child, best-effort
        # sitecustomize for worker grandchildren (workdir rides on PYTHONPATH).
        (work / GUARD_MODULE).write_text(GUARD_SOURCE, encoding="utf-8")
        (work / GUARD_BOOT).write_text(GUARD_BOOT_SOURCE, encoding="utf-8")
        (work / SITECUSTOMIZE).write_text(SITECUSTOMIZE_SOURCE, encoding="utf-8")
        argv = [sys.executable, GUARD_BOOT]

        # Library scratch space lands inside the jail.
        tmp_dir = work / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        child_env.update(policy_env(policy, str(work)))
        child_env.update(
            {
                "TMP": str(tmp_dir),
                "TEMP": str(tmp_dir),
                "TMPDIR": str(tmp_dir),
                "MPLCONFIGDIR": str(work / "mpl"),
                "PYTHONPATH": str(work)
                + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else ""),
            }
        )
        if os.name == "posix":
            preexec_fn = _posix_rlimits(policy.mem_mb, timeout_s)
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})

    timed_out = False
    warnings: list[str] = []
    oom_hit: dict[str, bool] = {}
    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=str(work),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        preexec_fn=preexec_fn,
    )
    if policy.active and policy.mem_mb and os.name == "nt":
        threading.Thread(
            target=_win_mem_watchdog,
            args=(proc, policy.mem_mb * 1024 * 1024, oom_hit),
            daemon=True,
        ).start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        proc.kill()
        stdout, stderr = proc.communicate()
    duration_s = time.monotonic() - start
    stdout, stderr = stdout or "", stderr or ""
    oom = bool(oom_hit.get("oom"))
    if not oom and policy.active and policy.mem_mb and "MemoryError" in stderr:
        oom = True  # POSIX RLIMIT_AS surfaces as MemoryError in the child

    stdout = _truncated(stdout, "stdout", warnings)
    stderr = _truncated(stderr, "stderr", warnings)
    (work / STDOUT_LOG).write_text(stdout, encoding="utf-8")
    (work / STDERR_LOG).write_text(stderr, encoding="utf-8")

    # Parse metrics.json if the code produced one; malformed JSON is a warning,
    # not a hard failure (the run may still be useful).
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            loaded = json.loads(metrics_path.read_text())
            if isinstance(loaded, dict):
                metrics = loaded
            else:
                warnings.append("metrics.json is not a JSON object; ignored")
        except json.JSONDecodeError:
            warnings.append("metrics.json is not valid JSON; ignored")

    artifacts = _collect_artifacts(artifacts_dir)

    ok = (not timed_out) and (not oom) and returncode == 0
    error: Optional[str] = None
    if timed_out:
        error = f"execution exceeded timeout of {timeout_s}s"
    elif oom:
        error = f"execution exceeded memory cap of {policy.mem_mb} MiB"
    elif returncode != 0:
        tail = [ln for ln in stderr.strip().splitlines() if ln.strip()][-_ERROR_TAIL_LINES:]
        error = "\n".join(tail) or f"process exited with code {returncode}"

    result = ExecResult(
        ok=ok,
        returncode=returncode,
        timed_out=timed_out,
        duration_s=duration_s,
        stdout=stdout,
        stderr=stderr,
        error=error,
        metrics=metrics,
        artifacts=artifacts,
        warnings=warnings,
        workdir=str(work),
        run_id=run_id,
        oom=oom,
        sandboxed=policy.active,
    )

    if not keep_workdir:
        shutil.rmtree(work, ignore_errors=True)
        result.workdir = ""

    return result
