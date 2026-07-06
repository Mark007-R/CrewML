# Day 1 — Phase 1: Foundation & Baselines

**Date:** 2026-07-06 · **Phase:** 1 (Foundation & Baselines, Days 1–4) · **Branch:** `main`

## Goal

Stand up the repository, lock the benchmark suite, and write the evaluation
protocol that makes the project's central claim falsifiable. Day 1 ships straight
to `main` (no PR); the Phase 1 PR opens on Day 2.

## What was built

### Repository scaffold
- `crewml/` package: `config.py` (paths, env, budgets, mock-mode detection) and
  `datasets.py` (dataset registry, loaders, SHA-256 integrity guards).
- `scripts/prepare_datasets.py` — idempotent download → split → lock → manifest.
- `tests/test_datasets.py` — 23 tests enforcing the split invariants.
- `.gitignore` (excludes `.env`, the SKILL files, `data/`, `models/`, `artifacts/`,
  venvs, caches), `.env.example`, `requirements.txt`, `README.md`.

### The locked benchmark suite (5 OpenML datasets)

| Dataset    | Task           | Subtype    | Rows  | Feats | Metric   | Train / Holdout |
|------------|----------------|------------|-------|-------|----------|-----------------|
| credit-g   | classification | binary     | 1,000 | 20    | ROC AUC  | 800 / 200       |
| diabetes   | classification | binary     | 768   | 8     | ROC AUC  | 614 / 154       |
| vehicle    | classification | multiclass | 846   | 18    | macro-F1 | 676 / 170       |
| cpu_small  | regression     | —          | 8,192 | 12    | R²       | 6,553 / 1,639   |
| kin8nm     | regression     | —          | 8,192 | 8     | R²       | 6,553 / 1,639   |

Mix achieved: **2 binary, 1 multiclass, 2 regression**. Each was chosen to stress a
distinct failure mode — imbalance (credit-g 700/300), disguised-missing zeros
(diabetes), multiclass metric choice (vehicle), and feature-engineering payoff
(kin8nm).

### The held-out lock (honesty core)
- One seed-locked split (`seed=42`, `holdout=0.2`, stratified for classification).
- Each split's contents are **SHA-256 sealed** into
  `results/dataset_manifest.json` (committed). `data/` itself is git-ignored and
  reproduced from the script + manifest.
- `verify_holdout_untouched(key)` recomputes the seal — the anchor for the Day 22
  no-peeking gate. A tampering test proves the guard actually fires.

### Evaluation protocol
`docs/EVAL_PROTOCOL.md` defines: one primary metric per dataset, the no-peeking
rule, fair-comparison rules (crew / solo / AutoML get identical inputs and ≥ equal
compute), and the honesty guards (mock numbers never reported as real, no silent
dataset drops).

## Verification

```
$ python scripts/prepare_datasets.py
[prepare] 5/5 datasets prepared.

$ python -m pytest tests/ -q
23 passed in 2.15s
```

Tests cover: registry mix, manifest completeness, target standardisation, **train⇔
holdout disjointness (leakage)**, split fraction sanity, holdout seal integrity,
and guard self-test.

## Decisions & notes

- **Datasets fetched via `sklearn.datasets.fetch_openml`** (network-verified)
  rather than the `openml` package, which isn't installed in this environment;
  `openml` stays in `requirements.txt` for parity. Versions are pinned for repro.
- **Positive class = rarer class** for binary AUC (credit-g → `bad`, diabetes →
  `tested_positive`), recorded in the manifest.
- **Data not committed** (git-ignored) — reproducibility comes from the pinned
  script + seed + committed manifest, keeping the repo lean and the holdout local.

## Next (Day 2)

Baseline 0 — `DummyClassifier/Regressor` + a single default RandomForest per
dataset → `results/baseline_metrics.json`. **Open the Phase 1 PR.**
