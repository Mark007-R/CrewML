# Day 8 — Phase 2 (MVP Crew) · The Planner agent

**Date:** 2026-07-13 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** open (Phase 2, mid-phase).

## Goal

Hire the crew's **second real worker** and retire the second stub. The Planner
reads the Profiler's **DataProfile** — and *only* that, never the data — and turns
it into a structured, JSON-friendly **ModelingPlan**: which columns to drop, how to
preprocess numeric vs. categorical features, which model families to try (with seed
hyperparameter grids), the cross-validation scheme, and the class-imbalance
strategy. The Feature Engineer and Trainer (Day 9) execute this plan, so — like the
Profiler — every decision must be **derived by rule, never guessed by an LLM**.

## What shipped

New module **`crewml/crew/planner.py`**:

- **`build_plan(profile, *, critique=None, iteration=0) -> dict`** — the deterministic
  core. Pure function of the profile dict; no LLM, no data loader, no I/O. Produces
  the ModelingPlan:
  - **column drops** — driven strictly by the Profiler's leakage checks: constant
    (zero-variance) columns, identifier-like near-unique columns, near-perfect
    single-feature target predictors (leakage suspects), and all-but-the-first of
    each duplicate-column group. Every drop carries a recorded `reason`.
  - **preprocessing** — dtype-aware. Numeric: median imputation, standard scaling
    available (applied only for scale-sensitive models), and a `zero_as_missing`
    list for the Profiler's *suspected* disguised-missing columns (carried with its
    heuristic caveat). Categorical: most-frequent imputation, one-hot for
    low-cardinality columns and **ordinal** for high-cardinality ones (so the feature
    space doesn't explode).
  - **candidate_models** — task-appropriate, ordered strong-first, each with a small
    seed grid, `needs_scaling`/`supports_proba`/`supports_class_weight` flags, and a
    rationale. Classification: HistGradientBoosting → RandomForest → LogisticRegression.
    Regression: HistGradientBoosting → RandomForest → Ridge.
  - **cv** — `StratifiedKFold` for classification, `KFold` for regression; 5 folds
    (clamped so it never exceeds the rarest class), shuffled, seeded (`config.SEED`),
    with the sklearn scorer string mapped from the primary metric (`roc_auc` /
    `f1_macro` / `r2`).
  - **imbalance_strategy** — recommends `class_weight='balanced'` + stratified CV
    **only** when the profile flagged `class_imbalance`; otherwise explicitly not
    recommended. Carries the positive (rarer) class from the profile.
  - **critique-aware** — `_apply_critique` mutates the plan to address the latest
    Critic finding (regularise on *overfit*, add capacity on *underfit*, re-audit on
    *leakage*, force class weights on *imbalance*, check scoring on *metric*). Wired
    now so the loop is functional the moment the real Critic (Day 10) produces
    findings; on the first pass `critiques` is empty and this is a no-op.
- **`run_planner(profile, *, critique=None, iteration=0, with_llm=None) -> dict`** —
  builds the plan, then layers an **optional advisory LLM refinement note**
  (`llm_narrative`) for the FE + Trainer *on top of* the deterministic plan. Enabled
  only when a live provider is configured and the `CREWML_PLANNER_LLM` toggle is on;
  in mock mode or on any provider error the narrative is `unavailable` and the plan
  stands on its deterministic core. **The narrative never overwrites a chosen value**,
  and a provider failure degrades gracefully — it can never crash the node.

**Node wired in** — `crewml/crew/nodes.py::planner` now calls `run_planner`, passing
the latest critique (if any) and the current iteration; the Day-5 stub is gone. Graph
topology and state schema are unchanged (Days 9–11 swap in the remaining agents behind
the same wiring).

New script **`scripts/run_planner.py`** — plans the whole suite; writes the
deterministic plans to committed **`results/day08_plans.json`** and the full plans
(with narrative) to git-ignored `artifacts/crew/<key>/plan.json`.

New tests **`tests/test_planner.py`** (24) + updated `tests/test_graph.py` (Planner is
real now, not a stub; the wiring fixture disables the Planner narrative to stay offline).

## Verification

`python scripts/run_planner.py` (live provider requested):

```
credit-g   drop=0  num=7   cat=13  cv=StratifiedKFold(5)/roc_auc   imb   models=[hist_gradient_boosting,random_forest,logistic_regression]
diabetes   drop=0  num=8   cat=0   cv=StratifiedKFold(5)/roc_auc   imb   models=[hist_gradient_boosting,random_forest,logistic_regression]
vehicle    drop=0  num=18  cat=0   cv=StratifiedKFold(5)/f1_macro  -     models=[hist_gradient_boosting,random_forest,logistic_regression]
cpu_small  drop=0  num=12  cat=0   cv=KFold(5)/r2                  -     models=[hist_gradient_boosting,random_forest,ridge]
kin8nm     drop=0  num=8   cat=0   cv=KFold(5)/r2                  -     models=[hist_gradient_boosting,random_forest,ridge]
```

Reading the results — the Planner turns the survey into a strategy:

- **credit-g** — mixed 7-numeric / 13-categorical schema → a ColumnTransformer plan,
  stratified AUC CV, and the imbalance strategy switched **on** (`class_weight='balanced'`,
  positive class `bad`) because the Profiler flagged the 2.33:1 skew.
- **diabetes** — the Profiler's disguised-missing signal flows straight into the plan:
  `zero_as_missing = [preg, skin, insu]` (treat 0 as NaN before median imputation),
  carried with the heuristic caveat. Imbalance strategy on.
- **vehicle** — 4-class → macro-F1 scoring; imbalance **not** recommended (classes are
  near-balanced), showing the strategy fires only when the profile earns it.
- **cpu_small / kin8nm** — regression → `KFold`/`r2`, tree-first candidates with Ridge
  as the regularised linear reference; no classification concept leaks in.

**Advisory LLM narrative — honest degradation.** A live narrative was requested, but the
Groq provider returned `400 organization_restricted` for every call. This is exactly the
failure path the design guards: each plan recorded `llm_narrative.source: "unavailable"`
with the provider error as the reason, `is_mock: false`, and the **deterministic plan —
the source of truth — is fully intact**. No narrative was invented, and nothing mock was
passed off as real (EVAL_PROTOCOL §5). The advisory layer is a bonus, never a crutch; the
monkeypatched tests prove that when a provider *does* answer, the note is attached with
provider/model/token accounting.

**Tests: 141 passed, 3 skipped** (120 prior + 21 net new). The suite stays **fully
offline** — the LLM path is exercised via monkeypatched fakes, never a live call. New
guards pin:

- plan shape, `recommended_primary_model` = first candidate, **determinism** (same
  profile ⇒ identical dict) and **JSON-safety**;
- **drops follow the profile** — a synthetic frame with planted constant / id / duplicate
  columns drops exactly those (keeping the first of a duplicate group) with a reason each,
  while a clean dataset (kin8nm) drops nothing;
- preprocessing partitions numeric vs. categorical over the *kept* columns, and diabetes's
  disguised-missing `insu` lands in `zero_as_missing`;
- classification → stratified CV + proba-capable models + correct scorer; regression →
  KFold + regressors; CV folds never exceed the rarest class;
- imbalance strategy recommended on skewed binary (positive = rarer class) and **off** on
  regression / balanced data;
- **the Critic loop responds** — an *overfit* critique genuinely strengthens regularisation
  in the grids (not just an annotation), and an unmatched critique is noted, never silently
  dropped; the first pass has no critique;
- **structural no-peeking** — `crewml/crew/planner.py` names neither the holdout loader nor
  even `load_train`: it reads a dict, so it cannot touch any split.

## Honesty & scope notes

- **The deterministic core is the source of truth.** Every choice in the plan is a rule
  applied to the profile. The LLM *refines*; it never supplies a decision. That keeps the
  plan reproducible and un-hallucinatable — and it held up today when the provider was
  restricted: the plans are complete and correct with the narrative absent.
- **Disguised-missing stays "suspected."** The plan recommends `zero_as_missing` for the
  Profiler's flagged columns but marks it heuristic (`zero_as_missing_is_heuristic: true`).
  `preg = 0` can be a legitimate value; the Planner points, and the Feature Engineer/Critic
  make the final call. It doesn't overrule the caveat it inherited.
- **The critique response is wired ahead of the Critic.** Day 10 ships the real Critic; the
  Planner's *response* side of the loop is built now (keyed off critique findings) so the
  loop is functional on arrival rather than a second integration later. It is a no-op until
  a critique exists.
- **Structural no-peeking holds.** The Planner is one step further from the data than the
  Profiler — it never even loads `train`, only reasons over the already-computed profile —
  and the build fails if `holdout` appears in its source.

## Next

Day 9 — the **Feature Engineer + Trainer**: generate feature-engineering code from the plan,
run it through the Day-6 sandboxed executor (its first real consumer of *generated* code),
train the candidate models with the planned CV, and return cross-validated metrics + artifact
paths. The first numbers the crew produces land here.
