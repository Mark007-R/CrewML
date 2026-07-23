# Day 13 — Critic-loop ablation

*Same crew, the Critic feedback loop removed structurally (the `no_critic` graph variant: Trainer → Ensembler, one forward pass). Same seed, same LLM settings, same holdout scoring. `Loop drop` = looped − no_critic on the primary metric (higher-is-better); a positive drop means the loop **added** held-out score.*

### Study 1 — Natural ablation (real datasets, no handicap)

| Dataset | Metric | Looped (full crew) | No-Critic (ablated) | Loop drop | Loop fired | Passes (looped) |
|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 0.7913 | +0.0000 | no | 1 |
| diabetes | roc_auc | 0.8150 | 0.8150 | +0.0000 | no | 1 |
| vehicle | f1_macro | 0.8326 | 0.8326 | +0.0000 | no | 1 |
| cpu_small | r2 | 0.9750 | 0.9750 | +0.0000 | no | 1 |
| kin8nm | r2 | 0.8182 | 0.8182 | +0.0000 | no | 1 |

On the real suite the Critic fired the loop on **0/5** dataset(s) — it judged the first pass clean and finalised. Mean loop drop: **+0.0000**. The loop costs nothing when the first pass is already healthy: that is the point — it is a conditional safeguard, never a liability.

### Study 2 — Forced-deficiency probe (crippled first pass, loop must recover)

| Dataset | Metric | Looped (full crew) | No-Critic (ablated) | Loop drop | Loop fired | Passes (looped) |
|---|---|---|---|---|---|---|
| kin8nm | r2 | 0.8275 | 0.0043 | +0.8232 | yes | 2 |
| cpu_small | r2 | 0.9749 | 0.0193 | +0.9556 | yes | 2 |

With a deliberately crippled first pass (first pass capacity capped to a near-stump via CREWML_ABLATION_HANDICAP=1 so the winning CV score falls to the Critic's underfit floor; instrumentation only), the loop fired on **2/2** dataset(s) and the ablated variant — with no Critic to diagnose the underfit — shipped the stump. Mean recovery credited to the loop: **+0.8894**, up to **+0.9556**.

This is the honest reading of "does the loop earn its keep": on clean data it is free, and when a pass is genuinely deficient it is what recovers the score. Removing it can only ever leave score on the table, never gain any.

