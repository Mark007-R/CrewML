# Day 9 — Phase 2 (MVP Crew) · The Feature Engineer + Trainer

**Date:** 2026-07-14 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** open (Phase 2, mid-phase).

## Goal

Hire the crew's **builders** and retire two stubs at once. The Feature Engineer is
the first agent that *writes code the crew then runs*; the Trainer is the first agent
that produces a **number**. Together they turn the Planner's ModelingPlan into real,
executed modeling: generate feature-engineering code, run it through the Day-6
sandboxed executor, cross-validate the candidate models under the planned CV scheme,
pick and refit the best, and save the artifact. This is where the crew stops
describing the problem and starts solving it — honestly, on the **train split only**.

## What shipped

New module **`crewml/crew/feature_engineer.py`** — the first generated-code node:

- **`DEFAULT_FE_SOURCE`** — a competent, **row-wise, leakage-free** default that always
  runs: an `add_features(df)` adding `row_nan_count` (missing-values-per-row). Being a
  pure function of each row's own values, it can be applied once up front without
  leaking across CV folds.
- **`run_feature_engineer(plan, dataset_key, *, with_llm=None) -> {code, meta}`** —
  **generate-then-validate**. When a live provider is configured it asks for
  dataset-specific `add_features` code, then **executes that code in the sandbox** on
  the train split. The generation is trusted **only** if it runs cleanly, preserves the
  row count and index, keeps every original column, and adds *numeric* columns —
  otherwise the agent records why and falls back to the deterministic default. An LLM
  never contributes an *unvalidated* line to a real run. `meta.source` is one of `llm`
  / `default` / `fallback`, with the validation verdict and token accounting attached.

New module **`crewml/crew/trainer.py`** — the first modeling node:

- **`run_trainer(plan, fe_code, dataset_key, *, iteration=0, param_search=None) -> dict`**
  — assembles a **training script** from the plan + the validated FE code and runs it in
  the sandboxed executor over `train.parquet` (the executor is handed nothing else). The
  generated script:
  - applies the Feature Engineer's `add_features`, then builds a dtype-aware
    **`ColumnTransformer`** straight from the plan — median impute for numerics (a
    separate *treat-zero-as-missing* branch for the Profiler's disguised-missing
    columns via `SimpleImputer(missing_values=0)`), most-frequent + one-hot for
    low-cardinality categoricals, ordinal for high-cardinality, optional scaling for
    scale-sensitive models. Engineered columns join the numeric branch.
  - **cross-validates every candidate** under the plan's CV splitter + primary-metric
    scorer, optionally searching the plan's seed grid (`GridSearchCV`, parallelised),
    applying `class_weight='balanced'` when the plan's imbalance strategy asks and the
    model supports it.
  - selects the best candidate by mean CV score, **refits it on the full train split**,
    and persists **`model.joblib`** alongside the exact **`fe_source.py`** used.
  - for binary tasks, maps the target to 0/1 with `1` = the plan's positive (rarer)
    class so the sklearn `roc_auc` scorer measures **exactly** the eval-protocol
    quantity; the mapping is recorded for later inversion.

**Nodes wired in** — `crewml/crew/nodes.py::feature_engineer` and `::trainer` now call
the real agents; the Day-5 stubs are gone. Graph topology and state schema are unchanged
apart from one added channel, **`fe_meta`** (the FE's provenance + validation verdict),
seeded in `initial_state`.

New script **`scripts/run_trainer.py`** — runs Profiler → Planner → FE → Trainer for a
dataset (or `--all`), writing reproducible CV results to committed
**`results/day09_training.json`**; the model + FE artifacts land git-ignored under
`artifacts/executor/<run_id>/`.

New tests **`tests/test_feature_engineer.py`** (9) + **`tests/test_trainer.py`** (11),
and **`tests/test_graph.py`** updated (FE + Trainer are real now; the wiring fixture runs
one Critic pass offline with the default FE and no grid search).

## Verification

`python scripts/run_trainer.py --dataset credit-g` (live provider requested, grid search on):

```
credit-g   best=random_forest  cv roc_auc=0.7994 (+/-0.0623)  fe=fallback  feats 20->21
```

Per-candidate cross-validated ROC AUC on credit-g (5-fold stratified, seeded):

| candidate               | CV roc_auc | best grid params |
|-------------------------|-----------:|------------------|
| **random_forest**       | **0.7994** | n_estimators=600, max_depth=20, min_samples_leaf=1 |
| logistic_regression     | 0.7907     | C=0.1 |
| hist_gradient_boosting  | 0.7703     | lr=0.05, max_iter=200, max_leaf_nodes=31 |

The Trainer selected `random_forest` (highest mean CV), refit it on all of train, and
saved it with the exact FE source. Feature engineering took the frame from **20 → 21**
columns (the default `row_nan_count`), and the binary target was encoded `bad → 1`
(the rarer/positive class) so the AUC is the protocol's AUC.

**Advisory FE — honest degradation (again).** A live provider was requested for the
Feature Engineer; Groq returned the same `organization_restricted` error seen on Day 8.
The design's trust gate handled it exactly: the generation failed, so `fe.source` is
`fallback`, the deterministic default was used (and itself validated in the sandbox,
`ok: true`), and the reason is recorded. No LLM code — validated or not — was passed off
as real, and the crew produced a complete result without it.

**Where 0.7994 sits — and what it is *not*.** This is a **cross-validated estimate on
train**, not a held-out score (`cv_score_is_holdout: false`), so it is **not** yet
comparable to the Phase-1 held-out numbers (default_rf 0.778, FLAML 0.735, solo-agent
0.652 — all on the locked holdout). The apples-to-apples *crew vs. solo vs. AutoML on the
held-out set* is **Day 12** (Phase 3), which runs the full crew and scores every system
once on the sealed split. Today's deliverable is that the crew now *produces* an honest,
reproducible number and a loadable model — not the head-to-head verdict.

**Tests: 163 passed, 3 skipped** (141 prior + 22 net new). The suite stays **fully
offline** — the FE LLM path is exercised via monkeypatched fakes, never a live call. New
guards pin:

- **the FE trust gate** — the default passes the sandbox contract (row count + index
  preserved, numeric columns); valid LLM code is used; LLM code that drops rows or emits
  a non-numeric column is **rejected and falls back**, with the rejection recorded; a
  provider failure degrades without raising;
- **real CV metrics** — every candidate is cross-validated, the reported best matches the
  max-CV candidate, the score is a float in range, FE was applied (`20 → 21` features),
  and the binary label mapping is `{1: bad, 0: good}`;
- **a loadable artifact** — `model.joblib` reloads and predicts; `fe_source.py` is saved
  beside it;
- **honesty** — every number is labelled a CV estimate, and the **holdout seal is intact
  after training** (`verify_holdout_untouched` still true); results are **deterministic**
  (same seed ⇒ identical CV score); a broken generation is **reported, never raised**;
- **structural no-peeking** — neither module names `load_holdout` / `holdout_path`; they
  are handed only `train.parquet` by the executor.

## Honesty & scope notes

- **The number is a CV estimate, not a grade.** The Trainer never loads the held-out set;
  its score is cross-validation on train. Reports label it so, and a test re-verifies the
  holdout fingerprint after every run. Held-out scoring is a deliberate separate step.
- **Generated code is guilty until validated.** The FE agent runs a live provider's code
  in the sandbox and trusts it only if it honours the contract. This is the honest version
  of "let the LLM do feature engineering": the crew benefits from a good generation and is
  immune to a bad one. Full self-repair (feed the traceback back and retry) is **Day 20**;
  today the response to a bad generation is a clean, recorded fallback.
- **The row-wise FE contract earns the up-front application.** Because engineered features
  must be functions of a single row's own values (no target, no cross-row statistics),
  computing them once is identical to computing them fold-by-fold — so applying FE before
  CV is leakage-free by construction, not by luck. The validation enforces the mechanical
  half of the contract (shape + dtype); the prompt states the rest.
- **Fair compute holds.** The whole per-node training run fits inside the executor's 120 s
  budget (the same budget that anchors the Day-4 AutoML ceiling), so any future "crew beats
  AutoML" can't be an artifact of handing the crew more time. Grid search is parallelised
  across cores to stay inside it.

## Next

Day 10 — the **Critic**: read the Trainer's CV results + the profile and diagnose the real
failure modes (overfit / underfit / leakage / imbalance / wrong metric), then decide
*iterate* vs. *finalize* and hand the Planner a **specific** directive. The Planner's
response side is already wired (Day 8), so Day 10 closes the loop end-to-end — the moment
the crew stops being a straight line and becomes a feedback system.
