# CrewML — Phase 3 results (consolidated)

*Six studies (Days 12–17), one section. Every number below is reshaped from a committed study file — nothing was re-run for this document, so it cannot disagree with the boards it summarises. Per-study boards with full method notes are linked at the end. All scores are on the LOCKED held-out split the crew never sees while modeling (EVAL_PROTOCOL.md §3); missing numbers render as an em dash, never a zero; runs without a live LLM stay labelled mock (§5).*

## 1. Does the crew win? (Day 12)

**Crew vs solo agent: 3/3 · vs AutoML (FLAML): 3/5 · vs default RF: 5/5** (wins counted only where both systems produced a real score).

| Dataset | Metric | Dummy (floor) | default RF | Solo agent | AutoML (FLAML) | **Crew** | Crew − Solo | Crew − AutoML | Crew − default RF |
|---|---|---|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.5000 | 0.7783 | 0.6517 | 0.7352 | 0.7913 | +0.1396 | +0.0561 | +0.0130 |
| diabetes | roc_auc | 0.5000 | 0.8118 | 0.8147 | 0.8039 | 0.8150 | +0.0003 | +0.0111 | +0.0032 |
| vehicle | f1_macro | 0.1028 | 0.7260 | — | 0.7785 | 0.8326 | — | +0.0541 | +0.1065 |
| cpu_small | r2 | -0.0029 | 0.9726 | 0.7129 | 0.9759 | 0.9750 | +0.2621 | -0.0009 | +0.0023 |
| kin8nm | r2 | -0.0002 | 0.6948 | — | 0.8421 | 0.8182 | — | -0.0239 | +0.1234 |

The two crew losses are to AutoML (`cpu_small` −0.0009, `kin8nm` −0.0239) and are reported as losses; the solo agent produced no scorable model on 2/5 datasets, so those deltas do not exist rather than counting as wins.

## 2. What each agent earns (Days 13–15)

**Critic loop (Day 13).** On the healthy suite the loop fired on 0/5 datasets — cost when idle: +0.0000. Under a deliberately crippled first pass it fired 2/2 and recovered a mean +0.8894 of held-out score (up to +0.9556). The loop is free when clean and is the entire recovery when not.

**Planner (Day 14).** Helped on 5/5 datasets, hurt on 0; mean drop when removed +0.0478, largest +0.1245 on `kin8nm`. **Feature Engineer (Day 14).** Helped on 2/5, hurt on 0; mean +0.0038, largest +0.0123 on `credit-g` — small but never negative.

| Dataset | Planner drop | FE drop | Critic probe recovery |
|---|---|---|---|
| credit-g | +0.0200 | +0.0123 | — |
| diabetes | +0.0038 | +0.0000 | — |
| vehicle | +0.0882 | +0.0065 | — |
| cpu_small | +0.0024 | +0.0000 | +0.9556 |
| kin8nm | +0.1245 | +0.0000 | +0.8232 |

*(Drop = full crew − ablated arm on the locked holdout; positive means the specialist added score. Critic recovery is from the forced-deficiency probe and exists only for the two probe datasets — an em dash is a dataset the probe cannot reach, not a zero.)*

**Iteration depth (Day 15).** The natural sweep is flat — unused budget changes nothing across 5 datasets. Under the deficiency probe the depth-response is a cliff, not a slope:

| Probe dataset | Budget 1 | Budget 2 | First-loop lift | Beyond first loop | Saturation |
|---|---|---|---|---|---|
| kin8nm | 0.0043 | 0.8275 | +0.8232 | +0.0000 | budget 2 |
| cpu_small | 0.0193 | 0.9749 | +0.9556 | +0.0000 | budget 2 |

Budget 1 ships the stump and reads as budget-bound (starvation is visible, not silent); the first allowed loop buys the entire recovery; every further loop buys nothing and goes unused. The production `max_iterations = 3` sits on the safe plateau.

## 3. What it costs (Day 16)

**Live crew runs (Groq — Llama 3.3 70B, prices as of 2026-07-21):** 4/5 datasets scored, total measured cost $0.0067 for the whole suite — under a cent per dataset. Costs are computed from measured tokens only; a run with no live calls has no cost, not a zero cost.

The live arm's 1 failure (`diabetes`: generated FE code produced non-finite features and training died) is on the board as a failure — the Critic filed it and finalised without a model. It is Day 20's (self-repair) motivating case.

**Outage resilience.** Fresh no-provider runs vs archival runs with a failing provider: 5/5 datasets bit-identical (max |Δ| = 0.00e+00) — holdout quality is provably independent of provider availability; an outage costs narrative richness, never score.

Still blocked: **anthropic** (not_configured). The study re-runs and re-prices itself when a key appears.

## 4. Where it fails, and who catches it (Day 17)

Census of 50 archived crew runs + 5 solo runs → 94 classified events. Fatal failures (no scored model): **crew 1** vs **solo 2** — and the crew's fatal was *caught and filed* by the Critic (an honest no-model finalise), while every solo failure is silent-fatal: no Critic to file it, no fallback to absorb it, no chooser to contain it.

| Outcome | fatal | degraded | handled | detected | missed |
|---|---|---|---|---|---|
| Events | 3 | 4 | 83 | 4 | 0 |

**Injection probes:** `leak_blatant` caught · `leak_subtle` **MISSED** · `exec_timeout` caught · `wrong_metric` caught · `exec_error` caught.

The missed probe is the phase's most valuable negative result: a leaked column at 95% target agreement sits *below* the Profiler's purity screen (0.995) and keeps CV *under* the Critic's too-good-to-be-true ceiling (0.995), so nothing fires and the model trains on the leak. That measured detection window is logged as direct input to Day 22 (leakage & honesty guards) — deliberately not patched inside the study that found it.

## 5. Caveats — read before quoting

* Baseline scores (Dummy / default RF / solo / AutoML) and the crew's headline column come from the deterministic core (seed-locked); the live Groq arm reproduces them within noise but is scored separately (Day 16).
* The solo agent failed outright on 2/5 datasets; its column is honest about that, and so are the missing deltas.
* The Critic's recovery numbers come from an instrumented handicap (`CREWML_ABLATION_HANDICAP=1`), not from natural runs — on healthy data the loop never fired. Both facts are the finding.
* The leakage detection window (§4) is open until Day 22. The claim "the crew detects leakage" holds only for leaks outside that window.
* Anthropic provider arm still unpriced-live: no ANTHROPIC_API_KEY this phase.

## 6. Per-study boards (full method notes)

* Day 12 — crew vs solo vs AutoML vs default: `results/comparison_table.md`
* Day 13 — Critic-loop ablation: `results/day13_critic_ablation.md`
* Day 14 — per-agent ablations: `results/day14_agent_ablation.md`
* Day 15 — iteration-depth study: `results/day15_iteration_depth.md`
* Day 16 — provider study: `results/day16_provider_study.md`
* Day 17 — failure taxonomy: `results/day17_failure_taxonomy.md`

*Chart: `results/charts/day18_phase3_summary.png` — the four headline panels (board deltas, agent attribution, depth cliff, failure outcomes).*
