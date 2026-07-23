# Day 15 — iteration-depth study: what does each extra Critic loop buy?

*The iteration budget (`max_iterations` = allowed Critic passes; budget 1 makes looping structurally impossible) swept on the full crew, everything else held constant. Scores are LOCKED-holdout, scored outside the graph, seal re-verified per run. **budget-bound** marks a run whose final Critic pass still had actionable findings — the crew was cut off, not done. Cost ratios price a *point* = 0.01 of the primary metric and are only computed when the marginal lift is real (> 1e-4) — never divided by noise.*

> **Session note (LLM):** Groq organization restricted this session (HTTP 400 on every call): all LLM narratives fell back to 'unavailable'; scores come from the deterministic core, token costs are unmeasurable (0 live narratives).

> **Session note (re-runs):** 9 point(s) re-ran with `CREWML_EXECUTOR_TIMEOUT_S=600` after a host slowdown tripped the default 120s executor cap (cpu_small@1 (natural), kin8nm@1 (natural), kin8nm@3 (natural), kin8nm@2 (deficiency_probe), kin8nm@3 (deficiency_probe), kin8nm@4 (deficiency_probe), cpu_small@2 (deficiency_probe), cpu_small@3 (deficiency_probe), cpu_small@4 (deficiency_probe)). Scores are timeout-independent given completion; treat the seconds columns as indicative only this session.

### Arm 1 — Natural sweep (real datasets, no handicap)

| Dataset | Metric | Budget 1 | Budget 3 | Spread | Passes used |
|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 0.7913 | 0.000000 | 1 |
| diabetes | roc_auc | 0.8150 | 0.8150 | 0.000000 | 1 |
| vehicle | f1_macro | 0.8326 | 0.8326 | 0.000000 | 1 |
| cpu_small | r2 | 0.9750 | 0.9750 | 0.000000 | 1 |
| kin8nm | r2 | 0.8182 | 0.8182 | 0.000000 | 1 |

On healthy data the Critic finalises pass 1 at every budget, so the sweep is **flat**: unused budget changes neither the score nor the work done. The production setting of 3 is free insurance, not a tax.

### Arm 2 — Deficiency sweep (first pass handicapped; the loop must recover)

*Same instrumentation as Day 13 (`CREWML_ABLATION_HANDICAP=1`): pass 1 capacity is capped to a near-stump so the winning CV score falls under the Critic's underfit floor and the loop has real work to do. Scope is Day 13's two regression sets — the handicap cannot push a classifier under the 0.60 ROC-AUC floor, so the loop would never arm there.*

| Dataset | Budget | Score (R²) | Passes | Budget-bound | Marginal lift | Marginal cost | s per point | Tokens per point |
|---|---|---|---|---|---|---|---|---|
| kin8nm | 1 | 0.0043 | 1 | **yes** | — | — | — | — |
| kin8nm | 2 | 0.8275 | 2 | no | +0.8232 | 663s / 0 tok | 8.06 | 0 |
| kin8nm | 3 | 0.8275 | 2 | no | +0.0000 | -27s / 0 tok | — | — |
| kin8nm | 4 | 0.8275 | 2 | no | +0.0000 | -474s / 0 tok | — | — |
| cpu_small | 1 | 0.0193 | 1 | **yes** | — | — | — | — |
| cpu_small | 2 | 0.9749 | 2 | no | +0.9556 | 130s / 0 tok | 1.36 | 0 |
| cpu_small | 3 | 0.9749 | 2 | no | +0.0000 | -13s / 0 tok | — | — |
| cpu_small | 4 | 0.9749 | 2 | no | +0.0000 | 25s / 0 tok | — | — |

**The depth-response is a cliff, not a slope**: the first allowed loop buys +0.8232 and +0.9556 of held-out R² (budget 1 → 2), while every loop after it buys +0.0000 and +0.0000. Saturation depth per dataset: `kin8nm` at budget 2, `cpu_small` at budget 2. Past saturation the Critic finalises on its own — extra budget is unused, not merely unhelpful.

Reading the two arms together: the right budget is *at least 2* (budget 1 ships the stump and reads as budget-bound — starvation is visible, not silent) and anything ≥ the crew's observed need is free. The production `max_iterations = 3` sits on the safe plateau of both curves.

