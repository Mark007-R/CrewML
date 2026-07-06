# CrewML Evaluation Protocol

This document is the contract every CrewML result is measured against. Its single
purpose is to make the headline claim — *"the multi-agent crew beats a solo agent
and classical AutoML on data it never saw"* — **honest and reproducible**. Any
number in a report that violates this protocol is invalid.

## 1. The benchmark suite

Five OpenML datasets, pinned by version, chosen to span task types and to stress
different failure modes (imbalance, disguised-missing values, multiclass metric
choice, feature engineering). The registry of record is
[`crewml/datasets.py`](../crewml/datasets.py); the materialised fingerprints live
in [`results/dataset_manifest.json`](../results/dataset_manifest.json).

| Dataset    | Task           | Subtype    | Rows  | Feats | Primary metric | Why it's here |
|------------|----------------|------------|-------|-------|----------------|---------------|
| credit-g   | classification | binary     | 1,000 | 20    | ROC AUC        | Imbalanced 700/300, mixed dtypes |
| diabetes   | classification | binary     | 768   | 8     | ROC AUC        | Disguised-missing zeros |
| vehicle    | classification | multiclass | 846   | 18    | macro-F1       | 4 near-balanced classes |
| cpu_small  | regression     | —          | 8,192 | 12    | R²             | Larger clean numeric target |
| kin8nm     | regression     | —          | 8,192 | 8     | R²             | Smooth non-linear target |

## 2. Metrics

- **Binary classification → ROC AUC**, computed on the predicted probability of
  the **rarer** class (the "positive" class recorded in the manifest). AUC is
  threshold-free and robust to the class imbalance these datasets carry.
- **Multiclass classification → macro-F1**, which weights every class equally so a
  model cannot win by ignoring a minority class. Accuracy is reported as a
  secondary, never-decisive number.
- **Regression → R²**, with RMSE reported alongside for interpretability.

Each dataset has exactly **one** primary metric. Comparisons across systems
(crew / solo / AutoML / default) are always on that primary metric.

## 3. The no-peeking rule (the honesty core)

For every dataset the data is split **once**, seed-locked, into:

- **`train.parquet`** — the *only* split any modeling code may read. Every agent,
  every baseline, all cross-validation, all feature engineering and model
  selection happens strictly inside this split.
- **`holdout.parquet`** — **LOCKED**. It is loaded exactly once per system, by the
  final scorer, to produce the number that goes in the report. No agent, prompt,
  plan, or feature-engineering step may read it, and nothing may fit on it.

Enforcement:

1. **Split provenance** — the split is stratified (classification) at a fixed seed
   (`42`) and fraction (`0.2`); anyone can reproduce the identical split by
   running `python scripts/prepare_datasets.py`.
2. **Cryptographic seal** — a SHA-256 of each split is recorded in the manifest.
   `crewml.datasets.verify_holdout_untouched(key)` recomputes it and fails if the
   held-out set was altered. This runs as a test
   ([`tests/test_datasets.py`](../tests/test_datasets.py)) and will become a
   pre-scoring gate (Day 22).
3. **Access discipline** — modeling code calls `load_train`; only the scorer calls
   `load_holdout`. Introducing a `load_holdout` call anywhere in the crew path is a
   protocol violation.
4. **Leakage within train** — cross-validation folds are built inside `train`
   only; any preprocessing that learns parameters (scaling, encoding, imputation,
   target statistics) is fit on training folds and applied to validation folds,
   never fit on the whole split before CV.

## 4. Fair-comparison rules

- **Same data, same metric, same holdout** for every competing system.
- The **solo agent** (Day 3) and the **crew** (Phase 2) receive identical inputs:
  the `train` split + the task type + the metric name. Neither is handed hints the
  other lacks.
- **Classical AutoML** (Day 4, FLAML) gets the same `train` split and an equal or
  larger compute/time budget than the crew, so beating it is never an artifact of
  giving the crew more compute.
- A **default RandomForest** and a **DummyClassifier/Regressor** (Day 2) anchor the
  floor: any system that fails to beat Dummy is reported as broken.

## 5. Honesty guards on reporting

- **Mock-mode is never real.** When no LLM key is configured the pipeline runs in
  mock mode (`crewml.config.is_mock_mode()`); any report containing mock numbers
  must label them **MOCK** and must not be used for the headline claim.
- **Seeds and versions are logged** with every result so a run can be reproduced.
- **No silent dataset drops.** If a dataset fails to prepare or score, it is listed
  as a failure, not omitted from the table.

## 6. Reproducing the splits

```bash
pip install -r requirements.txt
python scripts/prepare_datasets.py      # writes data/<key>/{train,holdout}.parquet
python -m pytest tests/                  # verifies the seals and disjointness
```

The `data/` directory is git-ignored; the splits are reproduced from OpenML +
this script + the pinned seed, and validated against the committed manifest.
