# Day 14 — per-agent ablations: Planner and Feature Engineer

*Each ablated arm replaces exactly one specialist with its naive floor — the Planner with a profile-blind default plan (no leakage drops, no cardinality or imbalance awareness, one library-default RandomForest, critique-deaf), the Feature Engineer with the identity transform (raw features only). Topology, seed, LLM settings and holdout scoring are identical across arms; the `full` reference was re-run in the same session so all three are paired. `drop` = full − ablated on the primary metric (higher-is-better): positive means the specialist added held-out score.*

| Dataset | Metric | Full crew | No-Planner | Planner drop | No-FE | FE drop | Full model | Naive model | FE cols (full) |
|---|---|---|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 0.7713 | +0.0200 | 0.7790 | +0.0123 | random_forest | random_forest | 1 |
| diabetes | roc_auc | 0.8150 | 0.8112 | +0.0038 | 0.8150 | +0.0000 | logistic_regression | random_forest | 1 |
| vehicle | f1_macro | 0.8326 | 0.7443 | +0.0882 | 0.8260 | +0.0065 | logistic_regression | random_forest | 1 |
| cpu_small | r2 | 0.9750 | 0.9726 | +0.0024 | 0.9750 | +0.0000 | hist_gradient_boosting | random_forest | 1 |
| kin8nm | r2 | 0.8182 | 0.6937 | +0.1245 | 0.8182 | +0.0000 | hist_gradient_boosting | random_forest | 1 |

**Planner** — compared on 5/5 dataset(s): helped on 5, hurt on 0; mean drop **+0.0478**, range +0.0024 … +0.1245 (largest on `kin8nm`).

**Feature Engineer** — compared on 5/5 dataset(s): helped on 2, hurt on 0; mean drop **+0.0038**, range +0.0000 … +0.0123 (largest on `credit-g`).

A negative drop means the naive floor beat the specialist on that dataset and is reported as-is — the ablation exists to *measure* the architecture, not to flatter it. In the `no_planner` arm the Critic loop still exists but points at a critique-deaf stand-in, so any iterate decision there changes nothing: without the Planner, the Critic's instructions have no actuator.

**Provenance.** The crew scores here come from archival runs executed during the Groq organization restriction (2026-07-20 → 2026-07-22): a key was configured — so `mock: false`, which keys off key *presence* — but every live LLM call failed with `organization_restricted` and the deterministic core produced every score (see results/day16_provider_study.md and results/day17_failure_taxonomy.json). Read these as deterministic-core results, not live-LLM-crew results. “LLM settings are identical across arms” is literally true — and no live LLM call succeeded in any arm, so the drops measure the deterministic Planner/Feature-Engineer implementations, not LLM agents.

