# Day 22 — Leakage & honesty guards

Seed 42; mock mode: False. **All guards hold: True** (16 checks).

## Screen calibration (single-feature CV, depth-3 tree, 3-fold)

| dataset | metric | strongest clean feature | score | ceiling | margin |
|---|---|---|---|---|---|
| credit-g | roc_auc | `checking_status` | 0.7080 | 0.87 | +0.1620 |
| diabetes | roc_auc | `plas` | 0.7847 | 0.87 | +0.0853 |
| vehicle | f1_macro | `SCALED_VARIANCE_MINOR` | 0.4893 | 0.75 | +0.2607 |
| cpu_small | r2 | `freeswap` | 0.8168 | 0.85 | +0.0332 |
| kin8nm | r2 | `theta3` | 0.2670 | 0.85 | +0.5830 |

Zero false positives on the clean suite (0 columns fired).

| injected leak | metric | ground-truth signal | standalone score | screened |
|---|---|---|---|---|
| credit-g / subtle | roc_auc | 0.9500 | 0.9464 | True |
| credit-g / blatant | roc_auc | 1.0000 | 1.0000 | True |
| cpu_small / subtle | r2 | 0.8982 | 0.8812 | True |
| cpu_small / blatant | r2 | 1.0000 | 0.9786 | True |

**Residual window (disclosed):** a leak whose standalone score lands between the clean maximum and the ceiling still passes — the Day-17 window is narrowed, not closed:

- `roc_auc`: undetectable band (0.7847, 0.87)
- `f1_macro`: undetectable band (0.4893, 0.75)
- `r2`: undetectable band (0.8168, 0.85)

## Full-crew injection probes (the Day-17 re-run)

| probe | flagged by | signal | plan dropped | model saw leak | CV | detected |
|---|---|---|---|---|---|---|
| leak_subtle (credit-g) | single_feature_cv | 0.946416 | True | False | 0.7945 (roc_auc) | **True** |
| leak_blatant (cpu_small) | pearson | 1.0 | True | False | 0.9779 (r2) | **True** |

## FE-introduced leakage (validation-gate probes)

| probe | expectation | ok | row_wise | no_leakage | verdict as expected |
|---|---|---|---|---|---|
| fe_clean_control | passes — the gate must not overblock legitimate FE | True | True | True | **True** |
| fe_leak_derived | rejected — engineered-column screen must fire | False | True | False | **True** |
| fe_cross_row | rejected — row-wise/statelessness check must fire | False | False | True | **True** |

## Runtime no-peek probe

Sandboxed read of `C:\Users\antho\Desktop\Mark\ALL\DATA_SCIENTIST\CrewML\data\credit-g\holdout.parquet`: refused = **True** (staged train input stayed readable: True).

## Holdout seals

| dataset | sealed |
|---|---|
| credit-g | True |
| diabetes | True |
| vehicle | True |
| cpu_small | True |
| kin8nm | True |
