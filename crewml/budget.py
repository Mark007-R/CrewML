"""Per-run cost & latency budgets — token/time caps with graceful early-stop (Day 21).

``CREWML_RUN_TOKEN_BUDGET`` existed since Day 5 but nothing ever *enforced* it: a
crew run could spend unbounded tokens and wall-clock and nobody would say stop.
This module is the enforcement layer. One :class:`RunBudget` ledger is installed
per crew run (:func:`start_run` / :func:`run_budget`); every LLM call routed
through :func:`crewml.llm.chat` is gated on it before the network is touched and
charged to it after, with a per-agent breakdown, so the run's cost is both capped
and itemised.

Enforcement is **cooperative, never a kill.** A budget that runs out does not
abort work in flight — it refuses the *next* LLM call
(:class:`BudgetExhaustedError`) and tells the Critic to finalise. Every caller of
``llm.chat`` already degrades gracefully on an exception (advisory narratives
become ``unavailable``, FE generation falls back to the deterministic default,
self-repair stops attempting), so an exhausted budget produces a *completed,
honestly-labelled run on the deterministic core* — exactly what mock mode
produces — not a crash. The hard per-step caps remain where they were:
the sandbox's ``EXECUTOR_TIMEOUT_S`` / memory watchdog (Days 6/19).

Two caps, both opt-out (``<= 0`` means uncapped):

* **Tokens** — ``CREWML_RUN_TOKEN_BUDGET`` (default 200k). Checked before every
  provider call; totals may overshoot by at most the one call already in flight,
  because the gate is pre-call and a granted call is always charged in full.
* **Wall-clock** — ``CREWML_RUN_TIME_BUDGET_S`` (default 1800s). Same pre-call
  gate, and the Critic reads it at each pass boundary (its natural checkpoint)
  to stop the loop early.

Why a process-global ledger and not a contextvar: LangGraph executes nodes on
worker threads, and a contextvar set in the invoking thread is not reliably
visible there. CrewML runs one crew at a time per process (the benchmark scripts
iterate datasets sequentially), so a single active ledger guarded by a lock is
the honest model; :func:`start_run` replaces any leftover ledger so a crashed
run cannot leak its spend into the next.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from crewml import config


class BudgetExhaustedError(RuntimeError):
    """Raised (pre-call) when the active run budget cannot afford another LLM call.

    Carries the ledger ``status`` snapshot at refusal time so callers can log an
    honest reason without re-reading the ledger.
    """

    def __init__(self, message: str, status: dict[str, Any]):
        super().__init__(message)
        self.status = status


def _cap(value: Optional[float], default: float) -> Optional[float]:
    """Normalise a cap: None -> config default; <= 0 -> uncapped (None)."""
    v = default if value is None else value
    return None if v is None or v <= 0 else v


class RunBudget:
    """Thread-safe token + wall-clock ledger for ONE crew run.

    ``clock`` is injectable (monotonic seconds) so tests can drive time.
    """

    def __init__(
        self,
        token_budget: Optional[int] = None,
        time_budget_s: Optional[float] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.token_budget = _cap(token_budget, config.RUN_TOKEN_BUDGET)
        if self.token_budget is not None:
            self.token_budget = int(self.token_budget)
        self.time_budget_s = _cap(time_budget_s, config.RUN_TIME_BUDGET_S)
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._n_calls = 0
        self._n_refused = 0
        self._llm_time_s = 0.0
        self._per_agent: dict[str, dict[str, Any]] = {}

    # --- Accounting -----------------------------------------------------------

    def _agent_row(self, agent: str) -> dict[str, Any]:
        return self._per_agent.setdefault(
            agent, {"calls": 0, "tokens": 0, "refused": 0, "llm_time_s": 0.0}
        )

    def charge(self, *, agent: str, prompt_tokens: int, completion_tokens: int,
               latency_s: float = 0.0) -> None:
        """Record one completed LLM call. Always succeeds — a granted call is
        charged in full even if it pushes the ledger past its cap (the gate is
        pre-call; see the module docstring on overshoot). ``latency_s`` is the
        provider round-trip (Day 25 telemetry) — recorded, never enforced; the
        wall-clock cap remains the run-level ``time_budget_s``."""
        with self._lock:
            self._prompt_tokens += int(prompt_tokens)
            self._completion_tokens += int(completion_tokens)
            self._n_calls += 1
            self._llm_time_s += float(latency_s)
            row = self._agent_row(agent)
            row["calls"] += 1
            row["tokens"] += int(prompt_tokens) + int(completion_tokens)
            row["llm_time_s"] = round(row["llm_time_s"] + float(latency_s), 3)

    def enforce(self, *, agent: str) -> None:
        """Pre-call gate: raise :class:`BudgetExhaustedError` if a cap is spent.

        The refusal is recorded (globally and per agent) so the run's report
        shows how much work the budget turned away, not just what it allowed.
        """
        if not self.exhausted:
            return
        with self._lock:
            self._n_refused += 1
            self._agent_row(agent)["refused"] += 1
        status = self.snapshot()
        raise BudgetExhaustedError(
            f"run budget exhausted ({status['stop_reason']}): refusing LLM call "
            f"from {agent!r} — {brief(status)}",
            status,
        )

    # --- State ----------------------------------------------------------------

    @property
    def tokens_spent(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def tokens_remaining(self) -> Optional[int]:
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.tokens_spent)

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    @property
    def time_remaining_s(self) -> Optional[float]:
        if self.time_budget_s is None:
            return None
        return max(0.0, self.time_budget_s - self.elapsed_s)

    @property
    def tokens_exhausted(self) -> bool:
        return self.token_budget is not None and self.tokens_spent >= self.token_budget

    @property
    def time_exhausted(self) -> bool:
        return self.time_budget_s is not None and self.elapsed_s >= self.time_budget_s

    @property
    def exhausted(self) -> bool:
        return self.tokens_exhausted or self.time_exhausted

    @property
    def stop_reason(self) -> Optional[str]:
        reasons = [r for r, hit in (("tokens", self.tokens_exhausted),
                                    ("time", self.time_exhausted)) if hit]
        return "+".join(reasons) or None

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serialisable view of the ledger — what reports and critiques embed."""
        with self._lock:
            per_agent = {k: dict(v) for k, v in self._per_agent.items()}
            n_calls, n_refused = self._n_calls, self._n_refused
            prompt, completion = self._prompt_tokens, self._completion_tokens
            llm_time_s = self._llm_time_s
        return {
            "token_budget": self.token_budget,
            "time_budget_s": self.time_budget_s,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "tokens_spent": prompt + completion,
            "tokens_remaining": self.tokens_remaining,
            "elapsed_s": round(self.elapsed_s, 3),
            "time_remaining_s": (round(self.time_remaining_s, 3)
                                 if self.time_remaining_s is not None else None),
            "n_calls": n_calls,
            "n_refused": n_refused,
            "llm_time_s": round(llm_time_s, 3),
            "per_agent": per_agent,
            "tokens_exhausted": self.tokens_exhausted,
            "time_exhausted": self.time_exhausted,
            "exhausted": self.exhausted,
            "stop_reason": self.stop_reason,
        }


def brief(status: dict[str, Any]) -> str:
    """One human-readable line from a snapshot, for decision reasons and logs."""
    tb = status.get("token_budget")
    tokens = (f"tokens {status.get('tokens_spent', 0)}/{tb}" if tb is not None
              else f"tokens {status.get('tokens_spent', 0)} (uncapped)")
    wb = status.get("time_budget_s")
    elapsed = status.get("elapsed_s", 0.0)
    clock = (f"elapsed {elapsed:.0f}s/{wb:.0f}s" if wb is not None
             else f"elapsed {elapsed:.0f}s (uncapped)")
    refused = status.get("n_refused", 0)
    tail = f", {refused} call(s) refused" if refused else ""
    return f"{tokens}, {clock}{tail}"


# --- The per-process active ledger -------------------------------------------

_ACTIVE: Optional[RunBudget] = None
_ACTIVE_LOCK = threading.Lock()


def start_run(
    token_budget: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> RunBudget:
    """Install a fresh ledger for a new crew run (replacing any leftover one)."""
    global _ACTIVE
    b = RunBudget(token_budget, time_budget_s, clock=clock)
    with _ACTIVE_LOCK:
        _ACTIVE = b
    return b


def end_run() -> Optional[RunBudget]:
    """Retire the active ledger (returning it for final snapshotting)."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        b, _ACTIVE = _ACTIVE, None
    return b


def active() -> Optional[RunBudget]:
    """The ledger for the run in progress, or None (enforcement dormant)."""
    return _ACTIVE


@contextmanager
def run_budget(
    token_budget: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    *,
    clock: Callable[[], float] = time.monotonic,
):
    """Scope one crew run under a fresh budget; always retires it on exit."""
    b = start_run(token_budget, time_budget_s, clock=clock)
    try:
        yield b
    finally:
        end_run()
