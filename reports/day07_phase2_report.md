# Day 7 — Phase 2 (MVP Crew) · The Profiler agent

**Date:** 2026-07-12 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** open (Phase 2, mid-phase).

## Goal

Hire the crew's **first real worker** and retire the first stub. The Profiler
opens the `train` split and writes down what is actually in it — schema, dtypes,
missingness (including the *disguised* kind), the target's distribution and class
imbalance, and a set of **basic leakage checks** — as a structured, JSON-friendly
**DataProfile**. Everything the Planner (Day 8) reasons over starts here, so the
facts must be **computed, never guessed**.

## What shipped

New module **`crewml/crew/profiler.py`**:

- **`build_profile(spec, train_df) -> dict`** — the deterministic core. Pure, no
  I/O, no network, no LLM. Produces the DataProfile:
  - **schema / dtypes** — numeric vs. categorical column lists, and per-feature
    facts: dtype, missing count + fraction, cardinality, numeric min/max/mean/std +
    **zero-fraction**, categorical top value + share.
  - **target** — for classification: class counts, `n_classes`, majority/minority,
    `imbalance_ratio`, and `positive_class` = the rarer class (computed train-only,
    matching `crewml.scoring`'s convention exactly). For regression: min/max/mean/
    std/skew.
  - **missingness** — explicit NaN columns + total missing cells.
  - **leakage_checks** — `constant_columns`, `id_like_columns` (near-unique *int/
    categorical* only — continuous floats are expected unique and are **not**
    flagged), `duplicate_feature_columns` (NaN-aware byte-identical columns),
    `duplicate_rows`, `suspected_disguised_missing` (zero-inflated numeric columns),
    and `target_correlated_features` (near-perfect single-feature predictors of the
    target — Pearson ≥ 0.98 for regression; per-group **target purity** ≥ 0.995 with
    ≥ 0.30 lift over the majority rate for classification).
  - **assessment** — a deterministic, rule-based synthesis into `flags` + human
    `notes` (`class_imbalance`, `disguised_missing_suspected`, `mixed_dtypes`,
    `high_cardinality_categoricals`, `target_leakage_suspected`, …). Fully
    reproducible; `source: "deterministic"`.
- **`run_profiler(dataset_key, *, with_llm=None) -> dict`** — loads `train`, builds
  the profile, then layers an **optional advisory LLM narrative** (`llm_narrative`)
  for the Planner *on top of* the deterministic facts. Enabled only when a live
  provider is configured and the `CREWML_PROFILER_LLM` toggle is on; in mock mode or
  on any provider error the narrative is marked `unavailable` and the profile stands
  on its deterministic core. **The narrative never overwrites a computed value**, and
  a network failure degrades gracefully — it can never crash the node.

**Node wired in** — `crewml/crew/nodes.py::profiler` now calls `run_profiler`; the
Day-5 stub is gone. The graph topology and state schema are unchanged (Days 8–11
swap in the remaining agents behind the same wiring).

New script **`scripts/run_profiler.py`** — profiles the whole suite; writes the
deterministic profiles to committed **`results/day07_profiles.json`** and the full
profiles (with narrative) to git-ignored `artifacts/crew/<key>/profile.json`.

New tests **`tests/test_profiler.py`** (17) + updated `tests/test_graph.py`.

## Verification

`python scripts/run_profiler.py` (live Groq narrative on):

```
credit-g   rows=800   feats=20  2-class imb=2.333333 pos=bad              flags=[class_imbalance,mixed_dtypes]
diabetes   rows=614   feats=8   2-class imb=1.869159 pos=tested_positive  flags=[class_imbalance,disguised_missing_suspected]
vehicle    rows=676   feats=18  4-class imb=1.09434  pos=van              flags=[disguised_missing_suspected]
cpu_small  rows=6553  feats=12  reg mean=83.78 skew=-3.37                 flags=[disguised_missing_suspected]
kin8nm     rows=6553  feats=8   reg mean=0.715 skew=0.0835                flags=[none]
```

Reading the results — the Profiler earns its keep:

- **credit-g** — correctly flags the 2.33:1 imbalance and the mixed 7-numeric /
  13-categorical schema (the Planner will need a `ColumnTransformer`). `pos=bad` is
  the rarer, scored class.
- **diabetes** — catches the registry's headline signal: **suspected disguised
  missing** on `insu` (~47% zeros), `skin`, `preg`. This is the "missing-value
  detection the crew must not miss" — surfaced as candidates with a caveat (zeros
  *can* be legitimate; the Planner/LLM adjudicates).
- **kin8nm** — **no flags at all.** The leakage checks stay quiet on clean data:
  the continuous floats are *not* mistaken for identifiers, and no phantom target
  leakage is invented. A check that cries wolf is worse than no check; this one
  doesn't.

**Live LLM narrative** (real, `source: groq`, `is_mock: false`, token-accounted)
on diabetes correctly told the Planner to weigh class-imbalance handling, verify
the zero-inflated `insu`/`skin`/`preg` before imputing, and optimise for ROC AUC —
interpreting the deterministic facts without inventing new numbers.

**Tests: 120 passed, 3 skipped** (105 prior + 15 net new). The suite is **fully
offline** — the LLM path is exercised via monkeypatched fakes, never a live call.
New guards pin:

- profile shape + a per-column fact for every feature; **determinism** (same frame
  ⇒ identical dict) and **JSON-safety** (no numpy scalars leak);
- binary imbalance + `positive_class` = rarer class; regression target summary;
- the diabetes disguised-missing signal fires (`insu` in the suspected set);
- **leakage: planted vs. clean** — a synthetic target-copy column is caught
  (classification purity *and* regression Pearson), a noise column is not, and
  kin8nm raises **zero** hard leakage flags; constant / id-like / duplicate columns
  are each detected on a synthetic frame;
- the narrative is advisory & honest — `unavailable`+`disabled` when off,
  `unavailable`+`mock_mode`+`is_mock:true` in mock mode, attached with provider/
  tokens from a live result, and a **provider crash degrades without raising** while
  the deterministic core stays intact;
- **structural no-peeking** — `crewml/crew/profiler.py` never names the held-out
  loader (asserted by source inspection, now covering the new module too).

## Honesty & scope notes

- **The deterministic core is the source of truth.** Every number in the profile is
  computed with pandas/numpy on `train` only. The LLM *interprets*; it never
  supplies a fact. That keeps the profile reproducible and un-hallucinatable.
- **Disguised-missing is a heuristic, labelled as one.** Zero-inflation flags
  *candidates* (`preg`'s zeros are legitimate pregnancies, for instance). The
  profile says `suspected_disguised_missing` and the assessment note spells out the
  caveat — surfacing the signal without overclaiming a verdict.
- **Correcting Day 6's "first executor consumer" framing.** Day 6 anticipated the
  Profiler as the executor's first real consumer. On reflection that's the wrong
  fit: the executor exists to sandbox *agent-generated, possibly-buggy* code, and
  the Profiler's EDA is **trusted, deterministic first-party code** with nothing to
  sandbox. Running it through a subprocess would add latency and a failure mode for
  no isolation benefit. The executor's first real consumer of *generated* code is
  the **Feature Engineer (Day 9)** — which is exactly what it was built for. Flagging
  the change rather than quietly honouring a stale plan.
- **Structural no-peeking holds.** The Profiler loads exactly one split (`train`);
  the module cannot reach the locked test set, and the build fails if the word ever
  appears in its source — the same discipline the rest of `crewml/crew/` keeps.

## Next

Day 8 — the **Planner** agent: read the DataProfile and produce a `ModelingPlan`
(preprocessing steps, candidate model families, CV scheme). On a Critic-triggered
re-entry it will consume the latest critique; today's profile is its first input.
