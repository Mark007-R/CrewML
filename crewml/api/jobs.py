"""Async job runner — one worker thread consuming a FIFO of crew runs.

A crew run takes minutes; an HTTP request must not. ``POST /run`` therefore
persists a ``queued`` row and hands the run_id to this runner, which executes
runs strictly one at a time on a daemon worker thread. Single-flight is a
deliberate choice, not a limitation: concurrent crew runs would contend for the
sandbox workdir, the per-run budget scope (``budget.run_budget`` is
process-global), and the Groq daily token budget — serialising them keeps every
Day-21/23 guarantee intact without new locking.

Every submitted run terminates in exactly one of ``succeeded`` / ``failed`` in
the store; a crashing run records its exception and never takes the worker
down. ``sync=True`` executes inline at submit time (tests, scripts); the async
path is identical code minus the queue hop.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable, Optional

from crewml.api.runner import execute_crew_run
from crewml.api.store import RunStore
from crewml.telemetry import build_run_telemetry

ExecuteFn = Callable[[dict[str, Any]], dict[str, Any]]


class JobRunner:
    def __init__(self, store: RunStore, execute: Optional[ExecuteFn] = None,
                 *, sync: bool = False) -> None:
        self.store = store
        self.execute: ExecuteFn = execute or execute_crew_run
        self.sync = sync
        self._queue: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # --- submission ----------------------------------------------------------

    def submit(self, run_id: str, params: dict[str, Any]) -> None:
        if self.sync:
            self._run_one(run_id, params)
            return
        self._ensure_worker()
        self._queue.put((run_id, params))

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._work_loop, name="crewml-job-runner", daemon=True
                )
                self._worker.start()

    # --- execution -----------------------------------------------------------

    def _work_loop(self) -> None:
        while True:
            run_id, params = self._queue.get()
            try:
                self._run_one(run_id, params)
            finally:
                self._queue.task_done()

    def _run_one(self, run_id: str, params: dict[str, Any]) -> None:
        self.store.mark_running(run_id)
        started = time.monotonic()
        try:
            result = self.execute(params)
        except Exception as exc:  # any failure -> recorded, worker survives
            tb_last = traceback.format_exception_only(type(exc), exc)[-1].strip()
            # A crashed run still gets duration telemetry; its LLM spend died
            # with the run's ledger, so the zeros here mean "unmeasured", and
            # /metrics only aggregates what actually got recorded.
            self.store.finish_failure(
                run_id, tb_last,
                telemetry=build_run_telemetry(duration_s=time.monotonic() - started),
            )
            return
        self.store.finish_success(
            run_id,
            record=result.get("record") or {},
            manifest=result.get("manifest") or {},
            model_card=result.get("model_card") or "",
            telemetry=result.get("telemetry"),
        )

    # --- test/shutdown support ----------------------------------------------

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until the queue drains (best-effort; True if drained)."""
        if self.sync:
            return True
        try:
            with self._queue.all_tasks_done:
                if timeout is None:
                    while self._queue.unfinished_tasks:
                        self._queue.all_tasks_done.wait()
                    return True
                remaining = timeout
                import time
                deadline = time.monotonic() + timeout
                while self._queue.unfinished_tasks and remaining > 0:
                    self._queue.all_tasks_done.wait(remaining)
                    remaining = deadline - time.monotonic()
                return not self._queue.unfinished_tasks
        except Exception:
            return False
