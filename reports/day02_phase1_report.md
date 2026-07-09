# Day 2 — Phase 1 (Foundation & Baselines)

**Date:** 2026-07-07 · **Sub-task:** Baseline 0 — Dummy + default RandomForest per
dataset → `results/baseline_metrics.json`. **PR:** Phase 1 opened.

## Goal

Establish the two non-agent reference points every later system is measured
against: an **honest floor** (a feature-blind Dummy) and a **default-model
anchor** (an untuned RandomForest). Both are scored once on the LOCKED holdout
through the single canonical scorer, so the eventual headline — *"the crew beats
solo and AutoML"* — is measured on the same ruler for every competitor.

## What shipped

- **`crewml/scoring.py`** — the one canonical metric module for the whole project.
  Implements the EVAL_PROTOCOL metrics exactly: binary → ROC AUC on the *rarer*
  (positive) class probability; multiclass → macro-F1 (accuracy secondary);
  regression → R² (RMSE secondary). Every future system (solo, AutoML, crew)
  scores through this, so numbers are directly comparable.
- **`crewml/baselines.py`** — reusable baseline logic: a minimal, **leakage-safe**
  preprocessor (median-impute numerics; most-frequent-impute + one-hot
  categoricals, `handle_unknown="ignore"`), the two estimators, and a
  fit-on-train / score-on-holdout runner.
- **`scripts/run_baselines.py`** — driver that scores both systems on all 5
  datasets and writes `results/baseline_metrics.json`. It re-verifies the holdout
  SHA-256 seal *after* scoring each dataset, proving the baselines never peeked at
  or mutated the held-out data.
- **`tests/test_baselines.py`** — 13 new tests: dataset-free unit tests pinning the
  exact scorer semantics (perfect predictions → AUC/F1/R² = 1, constant proba →
  AUC = 0.5), a preprocessor-produces-finite-numerics check, and integration
  checks that the metrics file is complete and default_rf beats Dummy everywhere.

## Results — held-out scores (primary metric, higher is better)

| Dataset   | Metric   | Dummy (floor) | default_rf | RF − Dummy | Secondary (RF) |
|-----------|----------|--------------:|-----------:|-----------:|----------------|
| credit-g  | ROC AUC  |        0.5000 |   0.7783   |   +0.2783  | acc 0.760 |
| diabetes  | ROC AUC  |        0.5000 |   0.8118   |   +0.3118  | acc 0.760 |
| vehicle   | macro-F1 |        0.1028 |   0.7260   |   +0.6232  | acc 0.729 |
| cpu_small | R²       |       −0.0029 |   0.9726   |   +0.9755  | rmse 2.863 |
| kin8nm    | R²       |       −0.0002 |   0.6948   |   +0.6950  | rmse 0.145 |

Seed 42, scikit-learn 1.8.0. All scores on the untouched holdout.

### Reading the floor

- **Dummy binary AUC is exactly 0.500** on both classification datasets — a
  feature-blind model can only rank at chance, precisely as the protocol expects.
- **Dummy R² ≈ 0** (very slightly negative) on both regression sets — the mean
  predictor is, by definition, the R² origin; slightly-below-zero just reflects
  the train-mean applied to the holdout.
- **default_rf clears the floor on every dataset**, so the floor is doing its job
  and the RandomForest is a legitimate anchor. The gaps are the room a smarter
  system has to win: **large** on cpu_small (near-ceiling already) and vehicle,
  and **meaningful** on kin8nm and the two binary sets, where R²=0.69 and
  AUC≈0.78–0.81 leave clear headroom for the solo agent, AutoML, and the crew.

## Honesty / protocol notes

- Baselines fit **only** on `train`; the preprocessor is fit on training data and
  applied to the holdout, so nothing leaks (protocol §3.4).
- The holdout seal is re-checked after scoring each dataset (`verify_holdout_
  untouched`) — all five passed.
- No LLM was involved today, so **mock-mode is not applicable**; these are real,
  deterministic sklearn numbers.
- Full suite: **36 tests pass** (23 Day 1 + 13 Day 2).

## Next

- **Day 3:** Baseline 1 — the solo agent (one LLM, one shot) writes a full sklearn
  script; execute it and capture its held-out score. That is the number the crew
  must beat.
- **Day 4:** Baseline 2 — classical AutoML (FLAML) as the strong non-agent ceiling;
  assemble the full baselines table; Phase 1 wrap-up + merge.
