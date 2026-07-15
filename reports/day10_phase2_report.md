# Day 10 — Phase 2 (MVP Crew) · The Critic + closing the loop

**Date:** 2026-07-15 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** open (Phase 2, mid-phase).

## Goal

Hire the crew's **reviewer** and turn the pipeline into a **feedback system**. Through Day 9
the crew was a straight line — Profiler → Planner → Feature Engineer → Trainer. The Critic is
the node that reads the Trainer's cross-validated result, **diagnoses** the failure modes a
competent ML reviewer looks for (overfit / underfit / leakage / imbalance / wrong-metric),
decides whether another pass earns its keep (**iterate**) or the run is done (**finalize**),
and — when it iterates — hands the Planner a **specific directive** it already knows how to
act on. The Planner's response side was wired on Day 8; today supplies the other half, so the
loop runs end to end. This is the day the crew stops being a line and becomes a team that
learns from its own first attempt.

## What shipped

New module **`crewml/crew/critic.py`** — the fifth stub retired, and the one that closes the
loop:

- **`diagnose(profile, plan, training) -> [findings]`** — the deterministic core. Pure over
  the three state dicts (no I/O, no LLM, no data), it detects, from the **CV-visible evidence
  only**:
  - **leakage** — a target-leakage suspect column the plan didn't drop, *or* an
    implausibly-high CV score (`roc_auc ≥ 0.995`, `r2 ≥ 0.999`) — the runtime fingerprint of
    leakage;
  - **imbalance** — the profile flagged class skew but it wasn't effectively handled (the plan
    didn't enable class weighting, or the winning model can't take class weights so the
    balancing never reached it);
  - **underfit** — the absolute CV score is at/below a per-metric floor (`roc_auc ≤ 0.60`,
    `f1_macro ≤ 0.50`, `r2 ≤ 0.10`) — the candidates are barely beating chance;
  - **overfit / variance** — large fold-to-fold spread (`cv_std/|cv_mean| ≥ 0.15`), the
    CV-visible symptom (honestly *not* a train-vs-held-out gap, which the crew can't see yet);
  - **wrong-metric** — the CV scorer doesn't match the primary metric (a cheap guard);
  - **execution failure** — a Trainer crash is surfaced, not diagnosed away (self-repair is Day 20).
  Every finding carries the **keyword the Planner's `_apply_critique` matches on**
  (`overfit`/`underfit`/`leak`/`imbalance`/`metric`) plus a specific `directive`, so a diagnosis
  becomes a concrete plan change without a brittle schema contract between the two nodes.
- **`decide(...) -> (decision, reason, delta)`** — iterate **only** when there's an actionable
  issue, the budget isn't spent, and the loop is still making progress. Finalize on a clean run,
  an execution failure, a spent budget, or **diminishing returns** (the score didn't improve by
  ≥ `0.002` *and* no new issue surfaced since the last pass). Convergence is a property, not a
  hope; the router's `max_iterations` guard (Day 5) is the hard backstop on top of it.
- **`run_critic(...)`** — assembles the critique and layers an **optional advisory LLM review
  note** on top of — never in place of — the deterministic verdict. In mock mode / on any error
  the narrative is `unavailable` and the decision stands on its deterministic core.

**Node wired in** — `crewml/crew/nodes.py::critic` now calls the real agent; the Day-5 stub
(which *always* asked to iterate, so a skeleton run always spent its full budget) is gone. Graph
topology and state schema are unchanged — the critique the Critic appends now carries a real
decision, structured `diagnoses`, and the `cv_score` + `finding_codes` the *next* pass needs to
measure progress.

New script **`scripts/run_critic.py`** — drives the **full compiled crew loop** on a dataset (or
`--all`), recording per dataset the node trace, iterations run vs. budget, and every Critic pass
to committed **`results/day10_critiques.json`**.

New tests **`tests/test_critic.py`** (24) and **`tests/test_graph.py`** updated (the Critic is
real now; a monkeypatched full-graph run proves the loop opens on a finding and closes itself by
convergence, before the guard).

## Verification

**Full crew loop, all 5 datasets** (`python scripts/run_critic.py --all`, grid search on):

| dataset   | task            | passes | planner runs | CV score            | Critic decision | findings |
|-----------|-----------------|:------:|:------------:|---------------------|-----------------|----------|
| credit-g  | classification  |  1/3   |      1       | roc_auc **0.7994**  | finalize        | none     |
| diabetes  | classification  |  1/3   |      1       | roc_auc **0.8372**  | finalize        | none     |
| vehicle   | classification  |  1/3   |      1       | f1_macro **0.7931** | finalize        | none     |
| cpu_small | regression      |  1/3   |      1       | r2 **0.9779**       | finalize        | none     |
| kin8nm    | regression      |  1/3   |      1       | r2 **0.8214**       | finalize        | none     |

**The headline is the decision, not the score.** On all five datasets the Critic found **nothing
actionable** and **finalised on pass one**. That is the loop earning its keep in the *other*
direction: the old stub Critic always asked to iterate, so every skeleton run burned its full
3-iteration budget (planner→FE→trainer ×3); the real Critic runs the Planner **once** on a clean
run and stops. Knowing when *not* to iterate is as much a part of a good feedback loop as knowing
when to. It also says something honest about the crew so far: the Planner's deterministic core
(stratified CV, class weighting when the profile flags skew, dtype-aware preprocessing, sensible
seed grids) is already producing sound plans — there's no low-hanging failure mode for the Critic
to catch on these particular sets.

**The loop demonstrably works — proven where a clean run can't show it.** Because none of the five
real datasets trips a diagnosis, the *iterate* path is proven in the test suite rather than by this
run: a monkeypatched full-graph test injects one actionable finding on pass 1, and asserts the real
compiled crew iterates (the **Planner runs twice**, its second plan carries `critique_adjustments`),
then finalises on pass 2 by convergence — all under a budget of 5, so the loop closes by its *own*
logic, not the backstop. Each diagnosis branch (overfit, underfit, both leakage signals, both
imbalance cases, wrong-metric, exec-failure) is unit-tested against a crafted scenario, and an
integration test proves a Critic finding flows into a real Planner plan change (an overfit finding
tightens the RandomForest grid to `min_samples_leaf=[4, 8]`).

**Advisory narrative — honest degradation (again).** A live provider was requested for the Critic's
review note; Groq returned the same `organization_restricted` error seen on Days 8–9. The design
handled it exactly: the narrative is `unavailable` with the reason recorded, and the decision stands
entirely on its deterministic core. No model influenced whether the crew iterated.

**Scores are unchanged and deterministic.** credit-g's `0.7994` is identical to Day 9 — the Critic
reads the Trainer's number, it doesn't perturb it. Every score here remains a **cross-validated
estimate on train** (`cv_score_is_holdout: false`); the crew-vs-solo-vs-AutoML head-to-head on the
sealed split is still **Day 12**.

**Tests: 188 passed, 3 skipped** (163 prior + 25 net new). The suite stays **fully offline** — the
Critic's LLM path is exercised via the disabled/mock branch, never a live call. New guards pin:

- **every diagnosis** — clean run flags nothing; overfit from high CV variance; underfit from a
  floor score; leakage from an undropped suspect *and* from a too-good score (which correctly
  suppresses the overfit/underfit signals); imbalance when weighting is off *and* when the winner
  can't use it; wrong-metric on a scorer mismatch; an exec failure short-circuits to a blocker;
- **the hand-off lands** — findings embed the Planner keyword, and a real Planner plan built with a
  Critic overfit finding actually tightens the grid;
- **the decision** — finalize on clean / failure / spent-budget / diminishing-returns; iterate on an
  actionable issue under budget, on a still-improving score, or on a newly-found issue even without a
  score gain;
- **end-to-end convergence** — the compiled crew opens the loop on a finding and closes it itself
  before the `max_iterations` backstop;
- **honesty** — the Critic reasons over CV-on-train only and its narrative is `unavailable` offline;
- **structural no-peeking** — the module never names the held-out loader.

## Honesty & scope notes

- **"Overfit" here means fold instability, not a holdout gap.** The Critic diagnoses from
  cross-validation on train — the only honest evidence at this stage — so its overfit signal is high
  fold-to-fold variance, and the finding text says exactly that. A true train-vs-held-out overfit
  check belongs to Phase-3 held-out scoring, which the Critic is structurally barred from touching.
- **A clean finalize is a real result, not a non-event.** The deliverable was a Critic that decides
  *soundly* and a loop that runs *end to end* — both shipped. That the five sound plans give it
  nothing to fix is a property of the data + Planner, not a gap in the Critic; its diagnostic and
  iterate paths are fully exercised by tests.
- **The loop can't run away.** Two independent bounds: the Critic's own convergence (clean /
  diminishing-returns / failure → finalize) and the router's hard `max_iterations` guard. A runaway
  crew is structurally impossible.
- **Self-repair is still Day 20.** Today an execution failure is *reported* as a blocker and the run
  finalises without a model; feeding the traceback back to the Trainer to fix and retry is a later,
  separate step.

## Phase 2 status

Six of the crew's seven nodes are now real specialists — Profiler, Planner, Feature Engineer,
Trainer, Critic — with the loop closed between them. Only the **Ensembler + Reporter** remain stubs.

## Next

Day 11 (Phase 2 finale) — the **Ensembler + Reporter**: combine the best models from the run's
trials and write the final report + `MODEL_CARD.md`, then the **first full end-to-end crew run** on
one dataset with every node real. Phase 2 wrap-up + PR merge.
