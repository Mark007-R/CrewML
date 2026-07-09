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

---

## Day 4 — 2026-07-09 · Phase 1 (Foundation & Baselines) · **phase close**

**Shipped:** Baseline 2 — the classical-AutoML ceiling (FLAML) + the full baselines
board. Phase 1 complete.

- Built **`crewml/automl_baseline.py`** + **`scripts/run_automl.py`**: FLAML fit
  strictly on `train` (its own 5-fold CV, 120 s/dataset budget held ≥ the crew's
  executor timeout for fairness), scored once on the LOCKED holdout via the shared
  scorer → **`results/automl_metrics.json`**. Metric mapped to FLAML's objective
  (roc_auc/macro_f1/r2); null-model guard; seal re-verified per dataset. **Real**
  run (no LLM → no mock caveat).
- Built **`crewml/leaderboard.py`** + **`scripts/build_baselines_table.py`**: reshape
  the three metrics files into one board → **`results/baselines_table.{json,md}`**
  (missing system → `—`, mock columns flagged; never re-scores).
- **Held-out results:** FLAML tops 3/5 — kin8nm 0.842 (+0.147 over RF), vehicle
  0.779, cpu_small 0.976. **Honest surprise:** on the two small binary sets AutoML
  does *not* win — credit-g 0.735 (< RF's 0.778), diabetes 0.804 (≈ RF's 0.812).
  Under a fixed budget, aggressive search on ~600–800 rows can generalise slightly
  worse than a plain forest — pinpointing where the crew must earn its keep.
- **77 tests pass** (59 prior + 18 new): metric mapping totality, leaderboard
  assembler/renderer, and AutoML/board integration (complete, beats floor, board
  agrees with sources). All 5 holdout seals intact across every scoring run.
- Updated README (baselines board + Phase 1 ✓). **Phase 1 Wrap-Up** in the Day 4
  report; merged the Phase 1 PR.

**Next:** Phase 2, Day 5 — LangGraph state schema + graph skeleton (node stubs,
wired edges, conditional Critic edge, `max_iterations` guard); open the Phase 2 PR.

---

## Update — 2026-07-09 · Solo agent now LIVE (Groq Llama-3.3-70B)

**Shipped:** retired the MOCK solo column — Baseline 1 ran for real once a
`GROQ_API_KEY` was configured. See [`reports/solo_live_update.md`](reports/solo_live_update.md).

- **`crewml/llm.py`** — seeded the Groq call (`seed=SEED`) for reproducibility (the
  one unseeded step in an otherwise seed-locked project).
- **`crewml/solo_agent.py`** — corrected the prompt's "set `random_state` everywhere"
  instruction (it forced `TypeError`s on transformers/`GridSearchCV`) and added a fair
  general "use real APIs / valid kwargs" note. Not dataset-specific coaching.
- **`tests/test_solo_agent.py`** — relaxed the mock-only "5/5 must succeed" assumption
  to honesty invariants that hold for a real run (no silent drops; scored entries
  finite + beat Dummy; seals intact; crashes reported as failures).
- **Live held-out (llama-3.3-70b, one shot, seeded):** credit-g 0.6517, diabetes
  **0.8147** (best on that row), cpu_small 0.7129; **vehicle & kin8nm CRASHED** (a
  timed-out `GridSearchCV`; an invalid hyper-param grid). **3/5 scored.** A single
  shot is unreliable on both correctness (2/5 crash) and quality (0.71 on cpu_small
  vs the forest's 0.97) — the exact gap the crew's Critic + Day-20 self-repair close.
- **74 tests pass, 3 skipped** (mock-mode contract + the 2 solo failures this run).

**Note:** the solo number is Llama-3.3-70B-specific, not the ceiling of any solo
agent; a stronger model (Day 16) or a repair loop (Day 20) would lift it.
