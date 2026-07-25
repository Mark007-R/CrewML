"""The self-repair loop — generated code that crashes gets to read its own
traceback and try again (Day 20).

Until today the crew's stance on a crashed generation was *observe and degrade*:
the executor reported ``ok: False``, the Trainer surfaced it, the Critic filed an
``execution_error`` blocker, and — for FE code — the agent silently fell back to
the deterministic default. Honest, safe, and wasteful: a large fraction of
generated-code crashes are one-line bugs (a misnamed column, a stray import, a
dtype mix-up) that the very model which wrote the code can fix *when shown the
traceback*. Day 20 adds that second chance as one reusable primitive:

    :func:`repair_loop` — given the failing source, the captured error, and a
    ``run_fn`` that executes candidate source and reports back, ask the live
    provider for a corrected module, re-run it, and iterate — feeding each new
    failure back into the next attempt — up to a hard attempt budget.

Two callers wire it in today (both behind default-on env toggles):

* the **Trainer** (:mod:`crewml.crew.trainer`) — a crashed training script is
  repaired and re-run in the sandbox; a recovered run carries full provenance
  (``training["repair"]``) so the Critic and Reporter can see the stumble.
* the **Feature Engineer** (:mod:`crewml.crew.feature_engineer`) — generated
  ``add_features`` code that fails sandbox validation gets repair attempts
  *before* the agent falls back to the default, with the contract verdict as
  the "traceback".

Safety posture (the loop must not weaken Day 19):

* **Same sandbox, always.** Repaired code runs through the exact ``run_fn`` the
  original ran through — the executor's :class:`~crewml.sandbox.SandboxPolicy`
  applies to attempt N exactly as it did to attempt 0. The repair loop grants no
  new capability; it only spends more LLM tokens.
* **Static no-peek guard.** A repaired source that names any way of *loading*
  the held-out split (``load_holdout``, ``holdout.parquet``, ``test.parquet``)
  is rejected without being run.
  The executor makes peeking structurally impossible anyway (the split is never
  staged), but the guard keeps even the *attempt* out of the record.
* **Compile gate.** Candidates must ``compile()`` before they earn a subprocess;
  a syntactically broken fix is bounced back to the model with the compile error
  as the new traceback, not executed.
* **Bounded.** ``CREWML_SELF_REPAIR_MAX_ATTEMPTS`` (default 2) caps the loop;
  timeouts and memory kills are *not* repairable (rewriting code cannot be
  trusted to fix a resource exhaustion, and retrying one doubles the bill).
* **Honest in mock mode.** Without a live provider there is no repair — the
  loop records ``attempted: False, reason: "mock_mode"`` rather than pretending.

Everything the loop does is recorded — every attempt, every guard rejection,
every token — so the Day-20 study can measure the recovery rate instead of
asserting it.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from crewml import config, llm

REPAIR_SCHEMA_VERSION = 1

# Substrings a repaired source may never contain — anything that names a way to
# LOAD the held-out split is disqualifying even though the executor could not
# stage it. (The bare word "holdout" is legitimately present in honest scripts —
# e.g. the Trainer's ``cv_score_is_holdout=False`` label — so the guard targets
# the loading surfaces, not the vocabulary.)
FORBIDDEN_TOKENS: tuple[str, ...] = ("load_holdout", "holdout.parquet", "test.parquet")

# A repaired module larger than this is a runaway generation, not a fix.
MAX_REPAIRED_CHARS = 64_000

# How many trailing error lines the model is shown (full tracebacks can be huge
# under CV — the tail carries the exception and the frames that matter).
_ERROR_TAIL_LINES = 40

# RunFn: candidate source -> (ok, error_text_or_None, payload). The payload is
# whatever the caller wants back from a successful run (an ExecResult, a
# validation verdict, ...); the loop treats it as opaque.
RunFn = Callable[[str], tuple[bool, Optional[str], Any]]


_REPAIR_SYSTEM_PROMPT = """\
You are the Self-Repair specialist in a multi-agent ML crew. A Python module
written by another agent failed when executed in a locked-down sandbox. You are
shown the module and the captured error. Produce a corrected version.

Hard requirements — the corrected module is executed and checked automatically:
- Fix the cause of the reported error with the SMALLEST change that preserves
  the module's intent. Do not redesign, reorder, or "improve" working parts.
- Keep the module's contract intact: every function it defines, every metric it
  emits, every artifact it writes, and every `crew_io` call must survive.
- The sandbox allows imports of the standard library and the scientific stack
  only (numpy, pandas, scipy, sklearn, joblib, matplotlib, xgboost, lightgbm).
  No network access, no subprocesses, no new files outside the working
  directory. If the error is a refused import, remove or replace the import —
  do not try to work around the sandbox.
- Read only the input files the module already reads. Never reference held-out
  or test data of any kind.
- Output ONLY the complete corrected Python module in one ```python code block.
  No prose, no diff, no commentary.
"""


def _error_tail(error: str) -> str:
    lines = [ln for ln in (error or "").splitlines() if ln.strip()]
    return "\n".join(lines[-_ERROR_TAIL_LINES:])


def _repair_user_prompt(code: str, error: str, context: str) -> str:
    return (
        f"{context.strip()}\n\n"
        f"The module failed with this error:\n"
        f"```\n{_error_tail(error)}\n```\n\n"
        f"The full module source:\n"
        f"```python\n{code}\n```\n\n"
        "Return the complete corrected module now."
    )


def _static_guard(candidate: str) -> Optional[str]:
    """Reject a candidate before it earns a run. Returns the reason, or None."""
    if len(candidate) > MAX_REPAIRED_CHARS:
        return f"candidate exceeds {MAX_REPAIRED_CHARS} chars ({len(candidate)})"
    lowered = candidate.lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            return f"candidate references forbidden token {token!r}"
    try:
        compile(candidate, "<repair-candidate>", "exec")
    except SyntaxError as exc:
        return f"candidate does not compile: {exc}"
    return None


def is_repairable(result: Any) -> bool:
    """Whether an :class:`~crewml.executor.ExecResult`-shaped failure is worth a fix.

    Repairable = the code *crashed* (non-zero exit with a traceback). Timeouts
    and memory kills are resource exhaustion — a rewrite cannot be trusted to fix
    them and each retry re-spends the whole budget — so they stay unrepairable
    and flow to the Critic exactly as before Day 20.
    """
    if getattr(result, "ok", False):
        return False
    if getattr(result, "timed_out", False) or getattr(result, "oom", False):
        return False
    return getattr(result, "returncode", None) is not None


def _enabled(explicit: Optional[bool], env_var: str) -> bool:
    """Explicit flag wins; else the node's env toggle; else the master switch."""
    if explicit is not None:
        return explicit
    node = os.getenv(env_var)
    if node is not None:
        return node.lower() not in ("0", "false", "off")
    return config.SELF_REPAIR


def repair_loop(
    code: str,
    error: str,
    *,
    run_fn: RunFn,
    context: str,
    max_attempts: Optional[int] = None,
    max_tokens: int = 6000,
) -> dict[str, Any]:
    """Ask the live provider to fix ``code`` given ``error``; re-run; iterate.

    Each attempt: prompt the provider with the *current* failing source and the
    *current* error (so attempt 2 learns from attempt 1's failed fix, not just
    the original crash), gate the reply through the static guard, then execute
    it via ``run_fn``. Stops on the first success or when the attempt budget is
    spent. Never raises — provider errors are recorded as failed attempts.

    Returns a JSON-friendly record::

        {"schema_version", "attempted", "reason_not_attempted", "max_attempts",
         "attempts": [{"attempt", "ok", "stage", "error", "provider", "model",
                       "prompt_tokens", "completion_tokens", "duration_note"}...],
         "recovered", "recovered_on_attempt", "code", "payload",
         "total_prompt_tokens", "total_completion_tokens"}

    ``code``/``payload`` are the repaired source and ``run_fn`` payload when
    ``recovered`` — callers adopt them; otherwise both are ``None``.
    """
    budget = int(
        max_attempts
        if max_attempts is not None
        else config.SELF_REPAIR_MAX_ATTEMPTS
    )
    record: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "attempted": False,
        "reason_not_attempted": None,
        "max_attempts": budget,
        "attempts": [],
        "recovered": False,
        "recovered_on_attempt": None,
        "code": None,
        "payload": None,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }

    if config.is_mock_mode():
        record["reason_not_attempted"] = "mock_mode"
        return record
    if budget <= 0:
        record["reason_not_attempted"] = "zero_attempt_budget"
        return record

    record["attempted"] = True
    current_code, current_error = code, error or "process failed with no captured error"

    for attempt in range(1, budget + 1):
        entry: dict[str, Any] = {"attempt": attempt, "ok": False}

        try:
            reply = llm.chat(
                _REPAIR_SYSTEM_PROMPT,
                _repair_user_prompt(current_code, current_error, context),
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # provider down/rate-limited — record, stop
            entry.update(stage="llm", error=f"{type(exc).__name__}: {exc}")
            record["attempts"].append(entry)
            break

        entry.update(
            provider=reply.provider,
            model=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
        )
        record["total_prompt_tokens"] += reply.prompt_tokens
        record["total_completion_tokens"] += reply.completion_tokens

        candidate = llm.extract_python(reply.text)
        guard_reason = _static_guard(candidate)
        if guard_reason is not None:
            # A non-compiling candidate becomes the next attempt's subject — the
            # model is shown its own broken fix. Forbidden-token/size rejections
            # keep the previous subject (adopting such a candidate is unsafe).
            entry.update(stage="guard", error=guard_reason)
            record["attempts"].append(entry)
            if guard_reason.startswith("candidate does not compile"):
                current_code, current_error = candidate, guard_reason
            continue

        ok, run_error, payload = run_fn(candidate)
        if ok:
            entry["ok"] = True
            record["attempts"].append(entry)
            record.update(
                recovered=True,
                recovered_on_attempt=attempt,
                code=candidate,
                payload=payload,
            )
            return record

        entry.update(stage="run", error=_error_tail(run_error or "run failed"))
        record["attempts"].append(entry)
        current_code, current_error = candidate, run_error or "run failed"

    return record


def repair_enabled_for_trainer(explicit: Optional[bool] = None) -> bool:
    """The Trainer's toggle: ``CREWML_TRAINER_SELF_REPAIR``, else the master switch."""
    return _enabled(explicit, "CREWML_TRAINER_SELF_REPAIR")


def repair_enabled_for_fe(explicit: Optional[bool] = None) -> bool:
    """The FE's toggle: ``CREWML_FE_SELF_REPAIR``, else the master switch."""
    return _enabled(explicit, "CREWML_FE_SELF_REPAIR")
