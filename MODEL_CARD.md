# Model Card — CrewML

CrewML is not one model: it is a **system that produces models**. This card
documents the system — what it builds, how it is evaluated, and where it fails.
Every individual run additionally emits its own per-run model card (written by
the Reporter agent; an example lives in the local `results/sample_model_card.md`)
carrying that run's dataset, candidates, CV scores, Critic decisions and seals.
This card is the contract those per-run cards inherit.

## System details

- **What it is:** a LangGraph crew of seven agents — Profiler → Planner →
  Feature Engineer → Trainer → Critic (loop) → Ensembler → Reporter — that takes
  a raw tabular dataset plus a task (classification or regression) and produces
  a trained sklearn-family model, a report, and a model card.
- **Model classes it produces:** RandomForest, HistGradientBoosting /
  LightGBM-class boosters, LogisticRegression / linear models, and soft-voting
  ensembles of those — chosen per-dataset by the Planner and Critic, never fixed
  in advance.
- **LLM providers:** Groq Llama-3.3-70B (default), Anthropic Claude (optional),
  or a deterministic mock mode. The deterministic core carries the pipeline when
  no provider is available; mock runs are always labelled and never reported as
  real results.
- **Version:** built over a 30-day public sprint (2026-07-06 → 2026-08-04),
  single author (Mark Rodrigues). API version 0.3.0.

## Intended use

- **Primary:** autonomous first-pass modeling of *tabular* datasets — a
  profiled, critiqued, honestly-evaluated baseline model plus a readable account
  of how it was built and what to distrust.
- **Appropriate users:** practitioners who will read the produced report and
  treat the output as a strong starting point subject to review.
- **Out of scope:** images, text, time-series forecasting with temporal
  structure the row-splitter would break, online learning, and any
  high-stakes decision (credit, medical, hiring) without a human review of the
  produced model and its card. The system optimises a single primary metric per
  task type; it does not perform fairness auditing.

## Training data

- **Benchmark suite:** 5 pinned OpenML datasets — credit-g, diabetes (binary),
  vehicle (multiclass), cpu_small, kin8nm (regression) — split once,
  seed-locked, into `train` and a **LOCKED holdout**; both splits SHA-256-sealed
  in `results/dataset_manifest.json` (local artifact set, regenerated and
  verified by `scripts/prepare_datasets.py`).
- **User data:** uploaded CSVs are ingested under the same rules — the target
  column is **chosen by the user, never guessed**; task, subtype and metric are
  derived from the chosen column per the evaluation protocol; the server splits
  and seals a holdout at ingestion, before any agent can run. Rows with a
  missing target are dropped, never imputed.
- The crew (and every baseline it is compared against) touches only the train
  split during modeling. `CrewState` carries a dataset id, never a holdout path.

## Evaluation protocol

- **Metrics:** binary → ROC AUC · multiclass → macro-F1 · regression → R².
  One scoring authority (`crewml/scoring.py`): an AST-scan test asserts it is
  the only module importing `sklearn.metrics` and that every compared system
  calls it.
- **Comparators, same data, same ruler:** Dummy floor, default RandomForest,
  a one-shot solo LLM agent, and FLAML (classical AutoML).
- **Holdout scores (locked split, higher is better):**

| Dataset | Metric | Dummy | default RF | Solo (live) | FLAML | **Crew** |
|---|---|---|---|---|---|---|
| credit-g | ROC AUC | 0.5000 | 0.7783 | 0.6517 | 0.7352 | **0.7913** |
| diabetes | ROC AUC | 0.5000 | 0.8118 | 0.8147 | 0.8039 | **0.8150** |
| vehicle | macro-F1 | 0.1028 | 0.7260 | ✗ crash | 0.7785 | **0.8326** |
| cpu_small | R² | −0.0029 | 0.9726 | 0.7129 | **0.9759** | 0.9750 |
| kin8nm | R² | −0.0002 | 0.6948 | ✗ crash | **0.8421** | 0.8182 |

Crew vs solo **3/3** (solo crashed on 2), vs default RF **5/5**, vs FLAML
**3/5** — the two losses are reported as losses. Scores inside a run's report
are **CV-on-train estimates** (`cv_score_is_holdout: false`); only the final,
one-time holdout evaluation produces the numbers above.

**Provenance.** The Crew column is the **deterministic core's** score: the
Day-12 archival runs executed during a Groq organization restriction (a key was
configured but every live call failed), while the Solo column is a genuinely
live Groq run — the columns were not produced under equivalent LLM conditions.
The live-LLM crew arm, scored separately on Day 16, differs from the Crew
column by −0.0016 (credit-g), +0.0113 (vehicle), −0.0008 (cpu_small), −0.0222
(kin8nm), and failed outright on diabetes. Two of the wins above are thinner
than that live-vs-core gap: diabetes vs solo (+0.0003) and cpu_small vs RF
(+0.0023).

## Safety & honesty measures

- **Sandboxed execution:** all agent-generated code runs under an import
  allowlist, no network egress, a filesystem jail and resource caps —
  adversarially tested (Day 19).
- **Leakage guards:** a calibrated single-feature screen at profiling, row-wise
  enforcement on generated feature code, and runtime no-peeking probes
  (Day 22; 16/16 checks hold).
- **Self-repair:** crashed generated code is retried with its traceback;
  18/18 injected faults recovered live, no false-positive repairs (Day 20).
- **Budgets:** per-run token/time caps; exhaustion degrades a run to its
  deterministic core rather than crashing it (Day 21).
- **Reproducibility:** every run writes a manifest pinning seed, split seals,
  package versions and provider, plus a result fingerprint; the deterministic
  core reproduces bit-identically across fresh processes (Day 23).

## Limitations & known failure modes

The failure taxonomy (Day 17) is part of this card. Known modes, each tied to
the detection surface that should catch it: executor errors and timeouts
(caught by the sandbox → Critic), under/overfit plans and invalid search spaces
(Critic floors and fit-time validation), wrong-metric choices (scorer guard),
introduced leakage (screen + residual checks — the screen's window is
calibrated, not infinite), unhandled imbalance, provider outages (deterministic
fallbacks), and budget cutoffs (finalise-with-findings, visibly marked).
Residual risks:

- A leak subtler than the calibrated screen's ceiling can pass undetected.
- LLM narratives can be wrong even when the numbers are right; numbers come
  from executed code, never from the LLM.
- Live-LLM runs are not bit-reproducible; divergence is recorded in the
  manifest rather than hidden.
- The benchmark is 5 mid-sized OpenML datasets; claims do not automatically
  extend to very wide, very tall, or heavily categorical data.

## Contact

Mark Rodrigues — https://github.com/Mark007-R/CrewML
