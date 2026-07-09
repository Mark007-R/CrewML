# Day 4 — Phase 1 (Foundation & Baselines)

**Baseline 2: the classical-AutoML ceiling (FLAML) — and the full baselines board.**

Today closes Phase 1. It adds the last and strongest *non-agent* competitor — a
mature AutoML system — assembles every baseline onto one leaderboard, and wraps up
the phase. This is a **real** run (FLAML is a classical system; no LLM, so no mock
caveat applies to this column).

## What shipped

- **`crewml/automl_baseline.py`** — the FLAML ceiling. `run_automl()` fits FLAML on
  `train` only (its own 5-fold CV inside that split, under a fixed per-dataset time
  budget), then scores once on the LOCKED `holdout` through the shared
  `crewml.scoring`. The CrewML primary metric is mapped to FLAML's optimisation
  objective (`roc_auc` → `roc_auc`, `f1_macro` → `macro_f1`, `r2` → `r2`) so the
  search optimises exactly what the holdout grades. FLAML is *trusted* library code,
  so — like the Day 2 baselines — it runs in-process; no subprocess sandbox needed.
  A null-model guard turns a budget-starved run into a reported failure rather than a
  silent empty result.
- **`crewml/config.py`** — `AUTOML_TIME_BUDGET_S` (default **120 s/dataset**),
  deliberately held ≥ the crew's per-node executor timeout so beating AutoML is never
  an artifact of giving the crew more compute (EVAL_PROTOCOL §4).
- **`scripts/run_automl.py`** — the Day 4 driver → `results/automl_metrics.json`.
  Records seed, budget, per-dataset `best_estimator` / `best_config` / FLAML version,
  and re-verifies every holdout seal after scoring. No mock mode here — the numbers
  are always real.
- **`crewml/leaderboard.py` + `scripts/build_baselines_table.py`** — consolidate the
  three metrics files (Dummy + default RF, solo agent, FLAML) into one board →
  `results/baselines_table.{json,md}`. It only *reshapes* committed results (never
  re-scores), renders a missing system as `—` (never a fabricated number), and flags
  any mock column — so the board can't silently present a mock number as real.
- **`tests/test_automl.py`** — 18 new tests: the metric mapping is total and
  class-balanced where it matters; the leaderboard assembler shapes every
  dataset/system, propagates the mock flag, and em-dashes missing systems; plus
  integration checks that `automl_metrics.json` is complete, well-formed, beats the
  Dummy floor everywhere, and that the board agrees with its source files.

## Held-out results — the full baselines board

Primary metric per dataset; higher is better. Same `train`, same `holdout`, same
scorer for every column.

| Dataset    | Metric   | Dummy (floor) | default RF | Solo (mock) | **AutoML (FLAML)** |
|------------|----------|--------------:|-----------:|------------:|-------------------:|
| credit-g   | ROC AUC  |        0.5000 |     0.7783 |      0.7521 |         **0.7352** |
| diabetes   | ROC AUC  |        0.5000 |     0.8118 |      0.7987 |         **0.8039** |
| vehicle    | macro-F1 |        0.1028 |     0.7260 |      0.7763 |         **0.7785** |
| cpu_small  | R²       |       −0.0029 |     0.9726 |      0.9747 |         **0.9759** |
| kin8nm     | R²       |       −0.0002 |     0.6948 |      0.8120 |         **0.8421** |

FLAML's chosen learners: lgbm (credit-g), extra_tree (diabetes), xgboost (vehicle,
cpu_small), catboost (kin8nm) — a genuinely diverse search, not one model in a
trench coat.

**Reading it.** The AutoML ceiling is the top system on three of five datasets —
clearly so on **kin8nm** (0.842, +0.147 over the default forest), and by a nose on
vehicle and cpu_small. But — and this is the honest, useful surprise — on the two
small binary sets it does **not** win: on **credit-g** FLAML (0.7352) actually
*trails* the untuned RandomForest (0.7783), and on **diabetes** it lands in a dead
heat (0.8039 vs 0.8118). Under a fixed budget, aggressive hyperparameter search on
~600–800 rows can pick a configuration that CV-looks-good but generalises slightly
worse than a plain forest. That's not a bug in the ceiling; it's the texture of the
problem. It says the crew's real opportunity is exactly where automated search
stumbles — small, imbalanced, disguised-missing binary data — and that "beat AutoML"
is not a single flat line but a per-dataset target.

## Honesty / protocol adherence

- FLAML fit **only** on `train` (its own internal CV); the holdout entered only the
  final `predict` + scorer.
- Every one of the 5 holdout seals re-verified intact after scoring.
- This column is **real** (no LLM → no mock caveat). The solo column remains
  MOCK-labelled and is not treated as a headline number.
- Budget, seed, FLAML version and the winning config are logged per dataset for
  reproducibility. FLAML is time-budgeted, so its number is reproducible *in
  distribution*, not bit-for-bit — noted explicitly in the results file.
- All 5 datasets scored; `failures = {}`.

## Verification

`python -m pytest tests/` → **77 passed** (59 prior + 18 new).

---

## Phase 1 Wrap-Up — Foundation & Baselines (Days 1–4)

**Goal of the phase:** stand up an *honest, reproducible* measuring stick before a
single agent is written, and draw the baseline lines the multi-agent crew will be
judged against. Done.

**What Phase 1 delivered**

1. **A locked, sealed benchmark** — 5 pinned OpenML datasets (2 binary, 1
   multiclass, 2 regression), each split once at seed 42 into `train` and a
   **LOCKED `holdout`**, every split SHA-256-sealed in the manifest with a
   tamper-detection self-test. (Day 1)
2. **One canonical scorer** — `crewml/scoring.py`: binary → ROC AUC on the rarer
   class, multiclass → macro-F1, regression → R². Every system in the project scores
   through this one module, so any number means the same thing. (Day 2)
3. **Four baseline systems, floor → ceiling**, all on the same holdout:
   - **Dummy** — the feature-blind floor (AUC 0.500, R² ≈ 0). (Day 2)
   - **default RF** — untuned RandomForest with minimal leakage-safe preprocessing. (Day 2)
   - **Solo agent** — one LLM, one shot, one sklearn script, executed in a trusted
     subprocess sandbox. Currently **MOCK** pending an LLM key. (Day 3)
   - **AutoML (FLAML)** — the strong classical ceiling. (Day 4)
4. **The evaluation contract** — `docs/EVAL_PROTOCOL.md`: the no-peeking rule,
   fair-comparison rules (equal data/metric/holdout; AutoML gets ≥ the crew's
   compute), and the honesty guards (mock is never real; no silent drops).

**The state of the scoreboard the crew inherits** (per-dataset best non-crew system):

| Dataset   | Best baseline to beat | Value  | Held by      |
|-----------|-----------------------|-------:|--------------|
| credit-g  | default RF            | 0.7783 | RandomForest |
| diabetes  | default RF            | 0.8118 | RandomForest |
| vehicle   | AutoML (FLAML)        | 0.7785 | xgboost      |
| cpu_small | AutoML (FLAML)        | 0.9759 | xgboost      |
| kin8nm    | AutoML (FLAML)        | 0.8421 | catboost     |

(The solo-agent column is excluded from "best to beat" because it is currently mock;
it re-enters as a real target once a key is set and Day 3's driver is re-run.)

**Honest caveats carried into Phase 2**

- The solo baseline is **mock** until a Groq key is configured — the headline "crew
  beats solo" claim is not yet quantified with a live number, only wired end-to-end.
- FLAML's numbers are **time-budgeted** (reproducible in distribution, not
  bit-for-bit); the 120 s/dataset budget is the fairness anchor for the crew.

**What's verifiably true after Phase 1:** the harness works end to end, the holdout
has never been touched (5/5 seals intact across every scoring run), and there is a
concrete, per-dataset number the crew must clear. **77 tests pass.**

**Next:** Phase 2, Day 5 — the LangGraph state schema + graph skeleton (nodes as
stubs, edges wired, the conditional Critic edge, a `max_iterations` guard). Open the
Phase 2 PR.
