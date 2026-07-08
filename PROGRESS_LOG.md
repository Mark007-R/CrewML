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

---

## Day 3 — 2026-07-08 · Phase 1 (Foundation & Baselines)

**Shipped:** Baseline 1 — the solo agent (one LLM, one shot, one sklearn script).

- Built **`crewml/llm.py`** — a thin provider abstraction (`chat()` → `LLMResult`
  with token accounting; Groq default, Anthropic optional). In **mock mode** it
  raises `MockModeError` so callers must take a deterministic offline path — no
  network needed to run. Plus `extract_python()` for fenced-code replies.
- Built **`crewml/solo_agent.py`** + **`scripts/run_solo_agent.py`**: a train-only
  profile summary + prompts asking for one `solve(train_df)` sklearn module;
  executed in a **subprocess** by a *trusted* runner that fits on `train` and
  predicts on held-out **features only** (never fits on holdout); scored once
  through `crewml.scoring` → **`results/solo_agent_metrics.json`**. Holdout seal
  re-verified after every dataset.
- **MOCK run** (no LLM key): a fixed HistGradientBoosting single-shot script,
  every score stamped `mock:true` (EVAL_PROTOCOL §5 — not the headline). Held-out:
  credit-g 0.752, diabetes 0.799 AUC; vehicle 0.776 macro-F1; cpu_small 0.975,
  kin8nm 0.812 R². Clears the Dummy floor everywhere; beats default_rf on
  vehicle/cpu_small/kin8nm, trails slightly on the two small binary sets — the gap
  the crew's Critic must close.
- **59 tests pass** (36 prior + 23 new): LLM extraction + mock-mode refusal,
  profile-is-train-only, the mock `solve` contract, and metrics completeness +
  seal-intact integration checks.

**Next:** Day 4 — Baseline 2: classical AutoML (FLAML) ceiling + full baselines
table; Phase 1 Wrap-Up; merge the Phase 1 PR.
