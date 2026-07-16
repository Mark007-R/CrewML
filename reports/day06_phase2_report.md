# Day 6 — Phase 2 (MVP Crew) · The sandboxed Python executor

**Date:** 2026-07-11 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** open (Phase 2, mid-phase).

## Goal

Build the **crux tool of the whole system**: the one place every real agent runs
code. On Day 9 the Feature Engineer and Trainer will *generate* Python; something
has to run it safely on the data and hand back a structured result. Day 6 ships
exactly that — `run_code`: a fresh **subprocess**, a hard **timeout**, an isolated
**temp workdir**, **captured** stdout/stderr, and a **metrics + artifacts
protocol** so numbers and files cross the process boundary cleanly. No agent
touches `subprocess` again; they all go through here.

## What shipped

New module **`crewml/executor.py`**:

- **`run_code(code, *, inputs=None, timeout_s=None, run_id=None, env=None,
  keep_workdir=True) -> ExecResult`** — the single entry point.
  - **Subprocess isolation.** Generated code is written to `main.py` and run with
    `sys.executable` in its own interpreter — never `exec`'d into the crew process.
    A crash, `sys.exit`, or runaway allocation takes down only the child.
  - **Timeout.** A hard wall-clock cap (default `config.EXECUTOR_TIMEOUT_S` = 120 s);
    on expiry the child is killed and `timed_out=True` is returned — never a hang.
  - **Isolated workdir.** Each call gets `artifacts/executor/<run_id>/` (git-ignored),
    with `cwd` set to it. Inputs are copied *in*, outputs stay *there*. A reused
    `run_id` starts clean, so a stale artifact can never leak into the next run.
  - **Captured output.** Full stdout/stderr are returned **and** written to
    `stdout.log` / `stderr.log` for post-mortems and the Day-26 dashboard.
- **`ExecResult`** (dataclass) — `ok`, `returncode`, `timed_out`, `duration_s`,
  `stdout`, `stderr`, `error` (short failure summary), `metrics`, `artifacts`,
  `warnings`, `workdir`, `run_id`. `as_dict()` gives a crew-state-friendly summary
  that omits the (large) streams and is JSON-serialisable for checkpointing.
- **The metrics + artifacts protocol** — a tiny helper (`HELPER_SOURCE`) is written
  into every workdir as **`crew_io.py`**, so generated code can:
  ```python
  from crew_io import emit_metrics, artifact_path, input_path, SEED
  emit_metrics(cv_score=0.83, model="hgb")      # merge-writes metrics.json
  joblib.dump(pipe, artifact_path("model.joblib"))
  train = pd.read_parquet(input_path("train.parquet"))
  ```
  The helper is *ergonomic, not mandatory*: code may write `metrics.json` / the
  artifacts dir by hand. The parent reads `metrics.json` back (malformed or
  non-object JSON becomes a **warning**, not a failure) and lists every file under
  the artifacts dir.

New script **`scripts/run_executor_demo.py`** — drives the tool end-to-end on real
(train-only) data: a Trainer-style 5-fold CV fit + a saved model artifact, plus a
crash case and a timeout case, so all three contracts are visible at once.

New tests **`tests/test_executor.py`** (17) — the full contract (below).

## Verification

`python scripts/run_executor_demo.py --dataset credit-g`:

```
[1/3] happy path — a Trainer-style CV fit on the TRAIN split only:
  ok=true  duration≈4.4s  metrics={cv_score: 0.7636, cv_std: 0.0585, scoring: roc_auc, n: 800}
  artifacts=[model.joblib]   stdout: "trained HistGradientBoostingClassifier: roc_auc=0.7636"
[2/3] failure path — a crash is reported, not raised:
  ok=false  returncode=1  error="...ValueError: deliberate boom..."
[3/3] timeout path — an infinite loop is killed at the cap (2s):
  ok=false  timed_out=true  error="execution exceeded timeout of 2s"
all three executor contracts verified [OK]
```

The `cv_score=0.7636` here is a **train-only 5-fold CV** number produced by the
demo to prove the round-trip — it is **not** a held-out result and is not reported
as one (EVAL_PROTOCOL §5).

**Tests: 105 passed, 3 skipped** (88 prior + 17 new). The new tests pin:

- capture + clean success; the default timeout equals `config.EXECUTOR_TIMEOUT_S`;
- the metrics protocol (`emit_metrics` merge-writes ⇒ parsed `metrics` dict) and
  artifact collection (including nested, forward-slashed paths); `SEED` exposed;
- **failures are reported, never raised** — crash, non-zero exit, timed-out
  infinite loop, and malformed / non-object `metrics.json` each come back as a
  structured `ExecResult` (the timeout test really spins an infinite loop and
  confirms the 2 s kill);
- isolation — distinct workdir + `run_id` per run; inputs staged in and read back;
  a reused `run_id` starts clean; `keep_workdir=False` deletes the dir;
- the one thing that *should* raise — a missing input source file (caller bug);
- `as_dict()` is JSON-serialisable and omits the streams;
- **honesty** — the executor source never references a held-out-set loader
  (structural no-peeking, asserted by `inspect.getsource`), matching the crew
  package's guarantee.

## Honesty & scope notes

- **Process isolation, not yet security isolation.** This is a subprocess sandbox
  with a timeout and a workdir — it is trusted with *our own* generated code, not
  hostile input. Import allow-listing, a **network jail**, filesystem confinement,
  and adversarial resource limits are **Day 19** (Phase 4 hardening); they layer on
  top of this same contract without changing it. The module docstring says so
  explicitly so no later report can overclaim the word "sandboxed".
- **Structural no-peeking.** The executor is data-agnostic: it copies in exactly
  the files the caller names and knows nothing about datasets or splits. It cannot
  fetch the locked test split, and a test fails the build if the word ever appears
  in its source — the same discipline the `crewml/crew/` package holds.
- **Determinism groundwork (for Day 23).** The child gets `PYTHONHASHSEED` and
  `CREWML_SEED` from the parent's seed, and `PYTHONDONTWRITEBYTECODE=1`.

## Next

Day 7 — the **Profiler** agent: turn the `train` split into a structured
`DataProfile` (schema, dtypes, missingness, target distribution, basic leakage
checks). First real agent, and the first real consumer of this executor.
