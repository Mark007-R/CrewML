# CrewML — Progress Log

A one-entry-per-day trail of what shipped. Full write-ups live in `reports/` and
plain-English versions in `explainers/`.

---

## Day 1 — 2026-07-06 · Phase 1 (Foundation & Baselines)

**Shipped:** Repo scaffold + the locked evaluation foundation.

- Initialised the repo (`crewml/` package, `scripts/`, `tests/`, docs, `.gitignore`,
  `.env.example`, `requirements.txt`, `README.md`).
- Locked a **5-dataset OpenML benchmark suite** (2 binary, 1 multiclass, 2
  regression): credit-g, diabetes, vehicle, cpu_small, kin8nm.
- Built `scripts/prepare_datasets.py`: seed-locked stratified split into `train` +
  a **LOCKED `holdout`**, with a **SHA-256 seal** per split written to
  `results/dataset_manifest.json`.
- Wrote **`docs/EVAL_PROTOCOL.md`** — metric per dataset, the no-peeking rule,
  fair-comparison rules, and honesty guards.
- **23 tests pass**, including train⇔holdout disjointness and a holdout-tampering
  self-test. All 5 datasets prepared cleanly.

**Next:** Day 2 — Dummy + default-RandomForest baselines → `baseline_metrics.json`;
open the Phase 1 PR.

---

## Day 2 — 2026-07-07 · Phase 1 (Foundation & Baselines)

**Shipped:** Baseline 0 — the honest floor + a default-model anchor, on a shared scorer.

- Built **`crewml/scoring.py`** — the single canonical metric module the whole
  project scores through (binary→ROC AUC on the rarer class, multiclass→macro-F1,
  regression→R²), so crew/solo/AutoML numbers stay directly comparable.
- Built **`crewml/baselines.py`** + **`scripts/run_baselines.py`**: a leakage-safe
  minimal preprocessor, `DummyClassifier/Regressor` (floor) and an untuned
  `RandomForest` (anchor), fit on `train` and scored once on the LOCKED holdout →
  **`results/baseline_metrics.json`**. Holdout seal re-verified after scoring.
- **Held-out results:** Dummy AUC=0.500 on both binary sets and R²≈0 on both
  regression sets (the floor behaves exactly as designed); default_rf clears it
  everywhere — credit-g 0.778, diabetes 0.812 AUC; vehicle 0.726 macro-F1;
  cpu_small 0.973, kin8nm 0.695 R². These are the numbers the crew must beat.
- **36 tests pass** (23 Day 1 + 13 Day 2). Opened the Phase 1 PR.

**Next:** Day 3 — Baseline 1: solo LLM agent writes+runs one sklearn script → its
held-out score (the crew's direct target).
