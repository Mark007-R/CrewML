# Day 17 — Failure taxonomy

*A closed vocabulary of failure categories — each owned by a stage and tied to the detection surface that should catch it — applied to every archived run record (census: nothing re-run, nothing flattered) and to live injection probes that plant known faults and watch what fires. `missed` is only ever assigned when an injection probe supplies ground truth; a census structurally cannot see what nothing detected.*

## The taxonomy

| Code | Group | Stage | Detection surface |
|---|---|---|---|
| `exec_error` | exec_error | executor/trainer | sandbox exit status -> Critic `execution_error` blocker |
| `exec_timeout` | exec_error | executor | sandbox wall-clock cap -> `timed_out` -> Critic blocker |
| `plan_underfit` | bad_plan | planner | Critic underfit floor (per-metric absolute score) |
| `plan_overfit_variance` | bad_plan | planner | Critic fold-variance threshold (cv_std/|cv_mean|) |
| `plan_search_invalid` | bad_plan | planner | sklearn parameter validation at fit time |
| `wrong_metric` | wrong_metric | planner | Critic scorer-vs-primary-metric guard |
| `leakage_flagged` | missed_leakage | profiler/planner | Profiler leakage screen + Critic residual/ceiling checks |
| `leakage_missed` | missed_leakage | profiler/planner/critic | NONE fired (ground truth from an injection probe) |
| `imbalance_unhandled` | bad_plan | planner | Critic imbalance check (flag present, strategy absent/ineffective) |
| `provider_outage` | provider | llm | per-call fallback (`unavailable` narrative / FE deterministic fallback) |
| `budget_cutoff` | budget | critic/router | Critic budget-reached finalise with actionable findings remaining |
| `loop_no_actuator` | budget | critic | cross-pass score delta (loop fired, nothing moved) |
| `ensemble_regression` | ensemble | ensembler | Ensembler same-fold CV comparison + chooser |

## Archive census — 50 crew runs + 5 solo-agent runs

| Category | Total | fatal | degraded | handled | detected | crew | solo |
|---|---|---|---|---|---|---|---|
| `ensemble_regression` | 42 | 0 | 0 | 42 | 0 | 42 | 0 |
| `provider_outage` | 35 | 0 | 0 | 35 | 0 | 35 | 0 |
| `plan_underfit` | 8 | 0 | 2 | 6 | 0 | 8 | 0 |
| `budget_cutoff` | 2 | 0 | 2 | 0 | 0 | 2 | 0 |
| `imbalance_unhandled` | 2 | 0 | 0 | 0 | 2 | 2 | 0 |
| `loop_no_actuator` | 2 | 0 | 0 | 0 | 2 | 2 | 0 |
| `exec_error` | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| `plan_search_invalid` | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| `exec_timeout` | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

Fatal failures (no scored model): **crew 1** vs **solo 2** across the whole archive. The 1 crew-side fatal(s) were each *caught and filed* — the Critic recorded the `execution_error` blocker and finalised honestly without a model rather than shipping garbage; automated self-repair (feed the traceback back and retry) is Day 20's feature and these are its motivating cases. Every other crew-side event was absorbed by a guard (`handled`), disclosed as quality-impacting (`degraded`), or recorded (`detected`). The solo agent's failures are all fatal: it has no Critic to file the fault, no fallback to absorb it, and no chooser to contain it.

## Injection probes — is each surface actually live?

| Probe | Fault injected | Expectation | Detected | Where it fired |
|---|---|---|---|---|
| `leak_blatant` (live, on cpu_small) | leaked column, pearson_corr_with_target=1.0 | caught (screen must fire) | yes | Profiler screen, Planner drop |
| `leak_subtle` (live, on credit-g) | leaked column, agreement_with_target=0.95 | missed (engineered inside the detection window) | no | — nothing fired |
| `exec_timeout` (live, on cpu_small) | executor cap starved to 5s | trainer killed at the cap; Critic files execution_error and finalises | yes | sandbox kill + `timed_out`, Critic blocker |
| `wrong_metric` (record-level) | plan.cv.scoring set to 'accuracy' (primary metric unchanged) | Critic `diagnose` fires `wrong_metric` | yes | crewml.crew.critic.diagnose |
| `exec_error` (record-level) | training.ok forced False with a real-shaped sklearn error | Critic `diagnose` fires `execution_error` | yes | crewml.crew.critic.diagnose |

### The measured detection window (the honest finding)

The subtle probe planted a leaked column agreeing with the target on 95.0% of rows — below the Profiler's purity screen (0.995) — and the resulting CV score (roc_auc=0.964435) stayed under the Critic's too-good-to-be-true ceiling (0.995). **No surface fired and the model trained on the leak.** That window — leak strong enough to inflate the score, weak enough to pass both screens — is a real, now-measured gap. Logged as Day 22 (leakage & honesty guards) input, not patched today: the taxonomy's job is to find gaps, and papering one over inside the study that found it would defeat the study.
