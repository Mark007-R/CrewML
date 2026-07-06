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
