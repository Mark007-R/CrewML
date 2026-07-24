"""The executor's security layer — sandbox policy + the in-child guard (Day 19).

Day 6 built the *process* sandbox: fresh subprocess, wall-clock timeout, isolated
workdir. Its docstring was explicit that this was not yet a *security* sandbox.
Day 19 delivers that layer without changing the Day-6 contract: the same
:func:`crewml.executor.run_code` call now runs agent code under a
:class:`SandboxPolicy`, enforced by a guard installed **inside the child
interpreter** before ``main.py`` executes.

The guard's three mechanisms (see :data:`GUARD_SOURCE`):

1. **Import allowlist** — a ``sys.meta_path`` gate: the stdlib plus an explicit
   third-party list (the DS stack). Anything else — ``pip``, ``crewml`` itself,
   random installed packages — fails with a clear ``ImportError``. Blocking
   ``crewml`` upgrades the Day-6 "executor never references the loaders" property
   to "the child structurally *cannot import* them".
2. **Behaviour hooks** (``sys.addaudithook``) — operations are refused at the
   moment of *use*, not import: network egress (DNS + connect/bind/sendto; the
   loopback interface stays available for library IPC), spawning anything that is
   not the same Python interpreter (so ``GridSearchCV(n_jobs=-1)``'s loky workers
   run, but ``os.system("curl ...")`` does not), and filesystem writes/renames/
   deletes outside the workdir jail. Reads of parent-named **deny roots** (the raw
   dataset store) are refused, enforcing staged-inputs-only.
3. **Resource caps** — on POSIX, hard rlimits (address space, CPU seconds, file
   size) set between fork and exec; on Windows, a parent-side watchdog thread that
   polls the child's working set and kills it over budget. Temp dirs
   (``TMP``/``TEMP``/``TMPDIR``) and ``MPLCONFIGDIR`` are pointed inside the
   workdir so library scratch space lands in the jail too.

**Honesty scope.** This is defence-in-depth against careless or misguided
*generated* code — the realistic failure mode of an LLM crew — not a hostile-
adversary boundary. An in-process Python sandbox is escapable by construction
(``ctypes`` must stay importable because numpy needs it; ``_winapi`` can spawn
processes below the audit layer). The adversarial tests therefore pin exactly
what IS refused, and the docs claim no more. True OS-level isolation arrives
with the Day 27 Docker wrapper. The Windows memory watchdog observes the direct
child only — worker grandchildren are not summed (POSIX rlimits *are* inherited).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from crewml.config import DATA_DIR, EXECUTOR_MEM_MB, EXECUTOR_SANDBOX

# Filenames the executor writes into the workdir when a policy is active.
GUARD_MODULE = "crew_guard.py"
GUARD_BOOT = "guard_boot.py"
SITECUSTOMIZE = "sitecustomize.py"

# Third-party top-level imports allowed by default: the DS stack the crew's
# generated code legitimately uses, plus the transitive top-levels those
# packages pull in (matplotlib -> PIL/cycler/kiwisolver/..., pandas ->
# dateutil/pytz/...). The stdlib is implied; everything else is refused.
DEFAULT_ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        # the stack itself
        "numpy", "pandas", "scipy", "sklearn", "joblib", "threadpoolctl",
        "xgboost", "lightgbm",
        # plotting
        "matplotlib", "mpl_toolkits", "pylab", "seaborn",
        # transitive top-levels of the above
        "PIL", "cycler", "kiwisolver", "pyparsing", "dateutil", "pytz",
        "tzdata", "packaging", "six", "fontTools", "pyarrow",
        # legacy import shims some wheels still touch at runtime
        "setuptools", "pkg_resources",
    }
)


@dataclass(frozen=True)
class SandboxPolicy:
    """What the guard enforces for one :func:`run_code` call.

    ``net`` is one of ``"deny"`` (no sockets at all), ``"loopback"`` (default —
    local IPC allowed, egress refused) or ``"allow"``. ``imports`` is the
    third-party allowlist (stdlib always implied). ``read_deny`` roots are
    directories the child must never read — the executor's default policy puts
    the raw dataset store here so the child sees staged inputs only.
    ``mem_mb=0`` means no memory cap.
    """

    active: bool = True
    net: str = "loopback"
    imports: frozenset[str] = DEFAULT_ALLOWED_IMPORTS
    read_deny: tuple[str, ...] = ()
    mem_mb: int = 0


def default_policy() -> SandboxPolicy:
    """The policy :func:`run_code` applies when the caller passes none."""
    return SandboxPolicy(
        active=EXECUTOR_SANDBOX,
        read_deny=(str(DATA_DIR),),
        mem_mb=EXECUTOR_MEM_MB,
    )


def policy_env(policy: SandboxPolicy, workdir: str) -> dict[str, str]:
    """The ``CREWML_SB_*`` env vars that configure the guard in the child."""
    return {
        "CREWML_SB": "1",
        "CREWML_SB_JAIL": workdir,
        "CREWML_SB_NET": policy.net,
        "CREWML_SB_READ_DENY": os.pathsep.join(policy.read_deny),
        "CREWML_SB_IMPORTS": ",".join(sorted(policy.imports)),
    }


# --- The guard injected into every sandboxed workdir as ``crew_guard.py`` ----

GUARD_SOURCE = '''\
"""crew_guard — CrewML sandbox guard, injected by the executor (Day 19).

Activated inside the child interpreter before agent code runs (fail-closed via
guard_boot.py for the main process; best-effort via sitecustomize.py for worker
grandchildren, which inherit the workdir on PYTHONPATH). Configuration arrives
in CREWML_SB_* environment variables; see crewml/sandbox.py for the policy side
and the honesty scope (defence-in-depth, not a hostile-adversary boundary).
"""
import os
import sys


class SandboxViolation(PermissionError):
    """Raised from the audit hook; aborts the refused operation.

    Subclasses PermissionError deliberately: a refusal should look exactly like
    an OS-level denial, so library fallbacks built for locked-down hosts keep
    working (e.g. platform._syscmd_ver shells out ``cmd /c ver`` on Windows and
    catches OSError; sklearn imports would otherwise die on our refusal).
    """


_ACTIVE = False
_JAIL = None
_READ_DENY = ()
_NET = "loopback"
_ALLOWED_TOP = frozenset()

_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND
    | getattr(os, "O_CREAT", 0) | getattr(os, "O_TRUNC", 0)
)
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1", ""}
_DEVNULL = {"nul", "/dev/null"}


def _canon(path):
    """Canonical comparable form of a path-ish value; None if not path-ish."""
    try:
        p = os.fspath(path)
        if isinstance(p, bytes):  # scipy et al. pass bytes paths to mkstemp
            p = os.fsdecode(p)
        return os.path.normcase(os.path.realpath(os.path.abspath(p)))
    except (TypeError, ValueError):
        return None


def _inside(path, root):
    return path == root or path.startswith(root + os.sep)


def _write_ok(path):
    if str(path).lower() in _DEVNULL:
        return True
    p = _canon(path)
    if p is None:  # int fd or exotic object — already vetted when opened
        return True
    return _inside(p, _JAIL)


def _read_denied(path):
    p = _canon(path)
    if p is None:
        return False
    return any(_inside(p, root) for root in _READ_DENY)


def _is_this_python(executable):
    if executable is None:
        return False
    exe = _canon(executable)
    return exe is not None and exe in (
        _canon(sys.executable),
        _canon(getattr(sys, "_base_executable", sys.executable)),
    )


def _first_cmdline_token(cmdline):
    """The executable at the front of a Windows-style command-line string."""
    if isinstance(cmdline, bytes):
        cmdline = cmdline.decode("utf-8", "replace")
    s = str(cmdline).strip()
    if s.startswith('"'):
        end = s.find('"', 1)
        return s[1:end] if end > 0 else s
    return s.split(" ", 1)[0]


def _host_of(addr):
    host = addr[0] if isinstance(addr, tuple) and addr else addr
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    return host


def _net_refused(event, detail):
    raise SandboxViolation(
        "CrewML sandbox: network egress refused (%s %r)" % (event, detail)
    )


def _audit(event, args):
    if event == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        if isinstance(mode, str):
            writing = any(c in mode for c in "wax+")
        else:
            writing = bool((flags or 0) & _WRITE_FLAGS)
        if writing and not _write_ok(path):
            raise SandboxViolation(
                "CrewML sandbox: write outside workdir refused: %r" % (path,)
            )
        if _read_denied(path):
            raise SandboxViolation(
                "CrewML sandbox: read of denied root refused: %r" % (path,)
            )
    elif event in ("os.remove", "os.rmdir", "os.rename", "os.link", "os.symlink"):
        for p in args:
            if p is None or isinstance(p, int):
                continue
            cp = _canon(p)
            if cp is not None and not _inside(cp, _JAIL):
                raise SandboxViolation(
                    "CrewML sandbox: %s outside workdir refused: %r" % (event, p)
                )
    elif event in ("socket.connect", "socket.bind", "socket.sendto"):
        if _NET == "allow":
            return
        addr = args[1] if len(args) > 1 else None
        if isinstance(addr, (str, bytes)):  # AF_UNIX — filesystem IPC, not egress
            return
        host = _host_of(addr)
        if _NET == "loopback" and isinstance(host, str) and host in _LOOPBACK:
            return
        _net_refused(event, addr)
    elif event in ("socket.getaddrinfo", "socket.gethostbyname",
                   "socket.gethostbyaddr", "socket.getnameinfo"):
        if _NET == "allow":
            return
        host = args[0]
        if isinstance(host, bytes):
            host = host.decode("utf-8", "replace")
        if isinstance(host, tuple):  # getnameinfo((host, port), flags)
            host = _host_of(host)
        if _NET == "loopback" and (host is None or host in _LOOPBACK):
            return
        _net_refused(event, host)
    elif event == "subprocess.Popen":
        executable, popen_args = args[0], args[1]
        exe = executable
        if exe is None and popen_args:
            if isinstance(popen_args, (list, tuple)):
                exe = popen_args[0]
            else:  # Windows audits the joined command line as one string
                exe = _first_cmdline_token(popen_args)
        if not _is_this_python(exe):
            raise SandboxViolation(
                "CrewML sandbox: spawning %r refused (only this Python)" % (exe,)
            )
    elif event == "os.system":
        raise SandboxViolation("CrewML sandbox: os.system refused")
    elif event in ("os.exec", "os.posix_spawn") or event == "os.spawn":
        path = args[1] if event == "os.spawn" else args[0]
        if not _is_this_python(path):
            raise SandboxViolation(
                "CrewML sandbox: %s of %r refused (only this Python)" % (event, path)
            )


class _ImportGate:
    """sys.meta_path gate: stdlib + allowlisted third-party tops only.

    Underscore-prefixed top-levels (C-extension internals of allowed packages)
    pass — a documented softness; the dangerous *behaviours* they could reach
    are refused by the audit hook instead.
    """

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top in _ALLOWED_TOP or top.startswith("_"):
            return None  # defer to the normal finders
        raise ImportError(
            "CrewML sandbox: import of %r is not allowed" % fullname
        )

    def find_module(self, fullname, path=None):  # pragma: no cover — py<3.12 shim
        self.find_spec(fullname, path)
        return None


def activate():
    """Install the guard once. Returns True when the sandbox is active."""
    global _ACTIVE, _JAIL, _READ_DENY, _NET, _ALLOWED_TOP
    if _ACTIVE:
        return True
    if os.environ.get("CREWML_SB") != "1":
        return False
    _JAIL = _canon(os.environ["CREWML_SB_JAIL"])
    deny = os.environ.get("CREWML_SB_READ_DENY", "")
    _READ_DENY = tuple(
        c for c in (_canon(p) for p in deny.split(os.pathsep) if p) if c
    )
    _NET = os.environ.get("CREWML_SB_NET", "loopback")
    extra = {
        m.strip()
        for m in os.environ.get("CREWML_SB_IMPORTS", "").split(",")
        if m.strip()
    }
    _ALLOWED_TOP = (
        frozenset(sys.stdlib_module_names)
        | extra
        | {"crew_io", "crew_guard", "sitecustomize", "usercustomize", "main"}
    )
    sys.meta_path.insert(0, _ImportGate())
    sys.addaudithook(_audit)
    _ACTIVE = True
    return True
'''


# --- Fail-closed boot for the main child process -----------------------------

GUARD_BOOT_SOURCE = '''\
"""guard_boot — activate the CrewML sandbox guard, then run main.py.

The executor launches ``python guard_boot.py`` instead of ``python main.py``
when a policy is active, so a missing/failed guard is a hard failure (exit 97),
never a silently unsandboxed run.
"""
import sys

import crew_guard

if not crew_guard.activate():
    sys.stderr.write("crew_guard: sandbox requested but not configured\\n")
    sys.exit(97)

import runpy

sys.argv = ["main.py"]
runpy.run_path("main.py", run_name="__main__")
'''

# Best-effort coverage for worker grandchildren (loky/multiprocessing): the
# workdir rides on PYTHONPATH, so their interpreters import this at site init.
SITECUSTOMIZE_SOURCE = '''\
try:
    import crew_guard

    crew_guard.activate()
except Exception:  # never break a worker over guard bootstrap
    pass
'''
