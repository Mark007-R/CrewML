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

---

## Day 5 — 2026-07-10 · Phase 2 (MVP Crew) · **phase open**

**Shipped:** the LangGraph crew **skeleton** — shared state schema + wired graph with
the Critic loop and its `max_iterations` guard. No LLM, no data, no scoring yet.

- New subpackage **`crewml/crew/`**:
  - **`state.py`** — `CrewState` (`TypedDict`): run inputs set once; produced fields
    (`profile`/`plan`/`fe_code`/`training`/`decision`/`ensemble`/`report`) start `None`;
    two append-only channels (`critiques`, `trace`) via an `operator.add` reducer so the
    loop grows history. All values JSON/msgpack-friendly (checkpointing + Day-26 dashboard).
    The holdout is never named — no-peeking is structural. `initial_state(spec, …)` seeds a run.
  - **`nodes.py`** — seven node stubs (Profiler→Planner→FE→Trainer→Critic→Ensembler→Reporter),
    each flagged `"stub": True`. The one real piece of logic is **`route_after_critic`**: guard
    first (`iteration >= max_iterations` ⇒ finalize regardless), else honour the Critic's decision.
  - **`graph.py`** — `build_graph()`/`build_crew()`; single conditional edge out of the Critic
    (`iterate`→planner, `finalize`→ensembler); `CREW_NODES` exposes topology as data.
- New **`scripts/run_crew.py`** — compiles + invokes the skeleton; trace + terminal summary →
  `artifacts/crew/<key>/skeleton_run.json`. Verified: `profiler → (planner,fe,trainer,critic)×3
  → ensembler → reporter`; the always-iterate stub Critic spends the full budget and the **guard**
  stops the loop (a runaway crew is structurally impossible).
- **88 tests pass, 3 skipped** (74 prior + 14 new): topology (7 nodes compile), router's three
  cases (iterate only when asked *and* under budget; guard overrides "iterate" at the ceiling),
  full-run termination at Reporter with 3 accumulated critiques, and honesty guards (Trainer stub
  emits `cv_score=None`; no crew module references the holdout — asserted by source inspection).
- Installed `langgraph 1.2.9` / `langchain-core 1.4.9` (already pinned in requirements). Opened
  the Phase 2 PR.

**Next:** Day 6 — the sandboxed Python executor tool (subprocess, timeout, captured
stdout/artifacts/metrics, temp workdir) — the shared tool every real agent calls.

---

## Day 6 — 2026-07-11 · Phase 2 (MVP Crew)

**Shipped:** the **sandboxed Python executor** — the crux tool every real agent runs
code through. No agent shells out to Python again; they all go through here.

- New **`crewml/executor.py`** — **`run_code(...) -> ExecResult`**: runs generated
  code in a fresh **subprocess** (`sys.executable`, never `exec`'d into the crew),
  with a hard **timeout** (default `EXECUTOR_TIMEOUT_S`=120 s → kill + `timed_out`,
  never a hang), an **isolated workdir** (`artifacts/executor/<run_id>/`, git-ignored,
  `cwd` set to it; a reused `run_id` starts clean), and **captured** stdout/stderr
  (returned *and* logged). A **metrics + artifacts protocol** via an injected
  `crew_io.py` helper (`emit_metrics`, `artifact_path`, `input_path`, `SEED`) hands
  numbers + files back across the process boundary; malformed `metrics.json` is a
  warning, not a failure. `ExecResult.as_dict()` is JSON-friendly for checkpointing.
- New **`scripts/run_executor_demo.py`** — end-to-end on real (train-only) data: a
  Trainer-style 5-fold CV fit (credit-g `cv_score≈0.7636`, a **train-only** number,
  not held-out) + a saved `model.joblib`, plus a crash case and a timeout case — all
  three contracts visible at once.
- **105 tests pass, 3 skipped** (88 prior + 17 new): capture + clean exit; default
  timeout == config; the metrics protocol + artifact collection (nested paths); every
  failure mode **reported not raised** (crash, non-zero exit, real infinite-loop
  timeout killed at 2 s, malformed/non-object metrics); isolation (distinct workdir
  per run, inputs staged in, reused id cleaned, `keep_workdir=False` deletes); missing
  input source is the one legit raise; `as_dict()` JSON-safe; and **structural
  no-peeking** — the executor source never references a held-out loader (source-inspection).
- **Scope honesty:** this is *process* isolation, not yet *security* isolation — import
  allow-listing, a network jail, and adversarial resource limits are **Day 19**; the
  module docstring says so explicitly so no report overclaims "sandboxed".

**Next:** Day 7 — the **Profiler** agent: `train` split → structured `DataProfile`
(schema, dtypes, missingness, target distribution, basic leakage checks). First real
agent, and the executor's first real consumer.

---

## Day 7 — 2026-07-12 · Phase 2 (MVP Crew)

**Shipped:** the **Profiler** — the crew's first REAL node, retiring the first stub.
Train-only EDA → a structured, JSON-friendly `DataProfile` the Planner reasons over.

- New **`crewml/crew/profiler.py`**:
  - **`build_profile(spec, train_df)`** — the deterministic core (no I/O, no LLM):
    schema/dtypes + per-feature facts (missingness, cardinality, numeric stats +
    **zero-fraction**); the target's class counts / imbalance / `positive_class`
    (rarer class, matching `crewml.scoring`) or regression summary; and **basic
    leakage checks** — constant, id-like (int/categorical near-unique only, so
    continuous floats aren't false-flagged), duplicate columns/rows, suspected
    disguised-missing (zero-inflated numerics), and near-perfect target predictors
    (regression Pearson ≥ 0.98; classification per-group target purity ≥ 0.995 with
    ≥ 0.30 lift). A rule-based **assessment** turns facts → flags + notes.
  - **`run_profiler(key, *, with_llm=None)`** — loads `train`, builds the profile,
    and layers an **optional advisory LLM narrative** for the Planner *on top of* the
    facts (live provider only; `unavailable` in mock mode / on error — degrades, never
    crashes). The narrative never supplies or overwrites a computed value.
- **Node wired in** (`nodes.py::profiler` → `run_profiler`); graph topology + state
  schema unchanged. New **`scripts/run_profiler.py`** → committed deterministic
  profiles in **`results/day07_profiles.json`** (+ full profiles w/ narrative under
  git-ignored `artifacts/`).
- **Real run:** correctly flags credit-g imbalance (2.33:1) + mixed dtypes; catches
  the diabetes **disguised-missing** signal (`insu`≈47% zeros, `skin`, `preg`); and
  raises **zero** leakage flags on clean kin8nm (the checks don't cry wolf). Live
  Groq narrative (real, `is_mock:false`, token-accounted) briefed the Planner on
  imbalance + zero-inflation + the ROC-AUC objective.
- **120 tests pass, 3 skipped** (105 prior + 15 net new), suite **fully offline**
  (LLM path via monkeypatched fakes): profile shape/determinism/JSON-safety;
  imbalance + rarer-positive; diabetes disguised-missing; **leakage planted-vs-clean**
  (synthetic target-copy caught, noise ignored, kin8nm silent) + constant/id/dupe
  detection; narrative advisory/honest across disabled/mock/live/failure; and
  **structural no-peeking** now covering the new module.

**Honesty note:** corrected Day 6's "Profiler = first executor consumer" — the
executor sandboxes *generated* code; the Profiler's EDA is trusted first-party code
with nothing to sandbox. The executor's first real consumer is the **Feature
Engineer (Day 9)**. Flagged, not quietly dropped.

**Next:** Day 8 — the **Planner** agent: read the `DataProfile` → a `ModelingPlan`
(preprocessing, candidate model families, CV scheme); consumes the latest critique on
a Critic-triggered re-entry.

## Day 8 — 2026-07-13 · Phase 2 (MVP Crew)

**Built:** the **Planner** agent — the crew's second real node — retiring the second
stub. `crewml/crew/planner.py`: `build_plan(profile, *, critique, iteration)` reasons
purely over the Profiler's `DataProfile` (never the data) to produce a `ModelingPlan`:
column drops (from the profile's leakage checks, each with a reason), dtype-aware
preprocessing (median/most-frequent imputation, standard scaling for scale-sensitive
models, `zero_as_missing` for suspected disguised-missing, one-hot vs. ordinal by
cardinality), task-appropriate candidate models with seed grids, the CV scheme
(`StratifiedKFold`/`KFold`, folds clamped to the rarest class, seeded, correct scorer),
and an imbalance strategy that fires only when flagged. `run_planner(...)` layers an
optional advisory LLM refinement note on top. Node wired in; graph topology unchanged.

- **Real run** (`scripts/run_planner.py`, all 5 datasets): credit-g → ColumnTransformer
  plan + stratified AUC CV + imbalance **on** (pos=`bad`); diabetes → `zero_as_missing =
  [preg, skin, insu]` flows from the profile's disguised-missing signal; vehicle → macro-F1,
  imbalance **off** (near-balanced); cpu_small/kin8nm → KFold/r2 tree-first + Ridge.
  Deterministic plans committed to `results/day08_plans.json`.
- **Honest degradation:** a live Groq narrative was requested but the provider returned
  `400 organization_restricted` on every call. The design's failure path held — each plan
  recorded `llm_narrative.source: "unavailable"` with the provider error as reason,
  `is_mock:false`, and the **deterministic plan (source of truth) fully intact**. No
  narrative invented; nothing mock reported as real (EVAL_PROTOCOL §5).
- **141 tests pass, 3 skipped** (120 prior + 21 net new), suite **fully offline** (LLM
  path via monkeypatched fakes): plan shape/determinism/JSON-safety; drops-follow-profile
  (planted constant/id/dupe dropped with reasons, clean kin8nm drops nothing); dtype-aware
  preprocessing + diabetes zero-as-missing; stratified-vs-KFold + correct scorer + folds ≤
  rarest class; imbalance on-skew/off-balanced; **the Critic loop responds** (an *overfit*
  critique genuinely strengthens regularisation, unmatched critique noted not dropped); and
  **structural no-peeking** — planner.py names neither `holdout` nor `load_train` (it reads
  a dict).

**Honesty note:** the Planner's *response* to Critic feedback is wired now, ahead of the
Day-10 Critic, so the loop is functional on arrival — a no-op until a critique exists.

**Next:** Day 9 — **Feature Engineer + Trainer**: generate FE code from the plan, run it
through the Day-6 sandboxed executor (its first real *generated*-code consumer), train the
candidates under the planned CV, and return cross-validated metrics + artifacts. The crew's
first real numbers land here.
