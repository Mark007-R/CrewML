# Update — the solo agent, now LIVE (Groq Llama-3.3-70B)

**Date:** 2026-07-09 (post-Phase-1, out of the daily cadence).
**Supersedes:** the MOCK solo column in the Day 3 / Day 4 reports (those remain as
honest, dated records of the harness before a key was available).

A `GROQ_API_KEY` is now configured, so Baseline 1 (the solo agent) ran for **real**
for the first time — one LLM, one shot, one sklearn script per dataset, executed in
the trusted subprocess sandbox and scored once on the LOCKED holdout. This retires
the mock stand-in for the solo column.

## What changed in the code

- **`crewml/llm.py`** — the Groq call is now **seeded** (`seed=SEED`) so a given
  prompt re-runs to the same code (best-effort on Groq); the project is seed-locked
  everywhere else, and the LLM call was the one unseeded step (EVAL_PROTOCOL §3.1).
- **`crewml/solo_agent.py`** — the system prompt's reproducibility instruction was
  **corrected**: it previously said "set `random_state=42` everywhere", which induced
  the model to pass `random_state` to transformers and `GridSearchCV` that don't
  accept it (a guaranteed `TypeError`). It now scopes `random_state` to objects that
  accept it, and adds a general "write code that imports/constructs real APIs" note —
  the same baseline competence the crew's code-writing agents will get. This is fair
  general guidance, **not** dataset-specific coaching.
- **`tests/test_solo_agent.py`** — the integration tests no longer assume the mock's
  guaranteed 5/5 success. They now enforce the honesty invariants that hold for a
  *real* run: no silent drops (every dataset is scored **or** listed as a failure),
  every scored entry is finite and beats the Dummy floor, and every holdout seal
  survives. A crash is recorded as a failure, never a fabricated score.

## Live held-out results

Solo agent = Groq **llama-3.3-70b-versatile**, one shot, seeded, `temperature=0`.

| Dataset    | Metric   | Dummy  | default RF | **Solo (live)** | AutoML (FLAML) | Best non-crew |
|------------|----------|-------:|-----------:|----------------:|---------------:|---------------|
| credit-g   | ROC AUC  | 0.5000 |     0.7783 |          0.6517 |         0.7352 | default RF 0.7783 |
| diabetes   | ROC AUC  | 0.5000 |     0.8118 |      **0.8147** |         0.8039 | **solo 0.8147** |
| vehicle    | macro-F1 | 0.1028 |     0.7260 |        ✗ crash  |         0.7785 | AutoML 0.7785 |
| cpu_small  | R²       |−0.0029 |     0.9726 |          0.7129 |         0.9759 | AutoML 0.9759 |
| kin8nm     | R²       |−0.0002 |     0.6948 |        ✗ crash  |         0.8421 | AutoML 0.8421 |

**Scored 3/5.** Two honest failures: **vehicle** — a `GridSearchCV` that exceeded the
120 s executor timeout; **kin8nm** — an invalid hyper-parameter grid (`alpha` passed
to an estimator that has no such parameter). An earlier run also hallucinated a
non-existent `f1_macro_score` import. These are recorded as failures, not hidden.

## Reading it — this is the case for a crew, in one table

A single Llama-3.3 shot is unreliable on **two axes at once**:

1. **Correctness.** With no second agent to catch it and no repair loop, it shipped
   code that crashes on 2/5 datasets — hallucinated symbols, bad param grids, and an
   un-budgeted search. A crew's Critic reviews the plan before it runs; the Day-20
   self-repair loop feeds a traceback back for a fix. Neither existed here.
2. **Quality.** Even where it ran, it was wildly inconsistent — **best on the board**
   on diabetes (0.815, edging both the forest and AutoML) yet a poor **0.713** on
   cpu_small where the plain forest scores 0.973. Nobody told it "you left a third of
   the R² on the table." That is precisely the Critic's job.

So the target the Phase-2 crew must clear is **per-dataset, the best non-crew cell**
in each row (right column above), not a single flat line — and on 4 of 5 rows that
bar is currently held by a non-agent system, which is the honest, unflattering
starting point the crew has to overturn.

## Honesty / protocol adherence

- Live run: `mock=False` at run- and per-dataset level; no mock number remains in the
  solo column.
- Real failures are reported (2/5), never silently dropped or back-filled.
- All 5 holdout seals re-verified intact after scoring.
- The solo number is **model-specific** (Llama-3.3-70B) and time-budgeted at the
  executor timeout — not the ceiling of what a solo agent could achieve with a
  stronger model (Day-16) or a repair loop (Day-20).

## Verification

`python -m pytest tests/` → **74 passed, 3 skipped** (the 3 skips: the mock-mode
contract test, now that a live key exists; and the two datasets the solo agent failed
on this run, which have no score to compare).
