# Day 11 — Phase 2 (MVP Crew) · The Ensembler + Reporter — the crew goes end-to-end

**Date:** 2026-07-16 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** Phase 2 — **wrap-up + merge**.

## Goal

Retire the **last two stubs** and run the **whole crew end to end for the first time, every
node real**. The Ensembler asks the question a careful ML engineer asks before shipping —
*would combining the strongest candidates beat the best one alone?* — and keeps the ensemble
**only when it actually helps**. The Reporter is the crew's terminal node: it synthesises the
entire run into a structured report and a **`MODEL_CARD.md`** — the model's "nutrition label" —
with the honesty caveats front and centre. With these two in place the crew takes a raw tabular
dataset and produces a trained, cross-validated model *plus its own write-up*, on its own. This
is the Phase-2 finale.

## What shipped

New module **`crewml/crew/ensembler.py`** — the sixth stub retired:

- **`run_ensembler(plan, training, fe_code, dataset_key) -> ensemble`** — builds a **soft-voting**
  (classification) / **averaging** (regression) ensemble over the top-`k` CV-ranked candidates,
  each carrying the **exact hyper-parameters the Trainer found for it**, and cross-validates that
  ensemble against the single best model **on the same seeded folds, inside the sandbox, on the
  train split**.
- **Ensembling never hurts, by construction.** The crew keeps the ensemble only when it clears the
  single best by more than a small epsilon; on a tie or a loss it keeps the **simpler single
  model** (Occam, and less to serve). So the final model is `max(ensemble, single)` on CV — never
  worse than what the Trainer already had.
- **Same preprocessing as the Trainer, guaranteed.** The ensemble config is derived from
  `trainer._training_config(plan)`, so every member's ColumnTransformer, imputation, encoding and
  scaling is byte-for-byte what the Trainer built — a genuine combination of the *same* pipelines.
  The single-best re-score is self-consistent (same seeded CV + params), so the ensemble-vs-single
  comparison can't be an artifact of two different evaluations.
- **Honest degradation.** A failed Trainer run (no models to combine) or a too-thin candidate set
  yields an "not attempted" record that falls back to the Trainer's model; a crash in the generated
  ensemble script is captured as `ok: False`, never raised.

New module **`crewml/crew/reporter.py`** — the seventh (last) stub retired, and the crew's terminal
node:

- **`build_report(state) -> report`** (pure) + **`render_model_card(report) -> markdown`** (pure) +
  **`run_reporter(state)`** (writes the artifacts). Deliberately **deterministic — no LLM**: it
  *synthesises* what the specialists already decided (there is nothing to reason about that a node
  didn't already settle, and a report that could hallucinate would undermine the whole honesty
  story).
- The model card follows the ML-community shape (details / training data / evaluation / metrics /
  limitations) and **surfaces the honesty caveats a reader must not miss**: every score is a
  **cross-validated estimate on train, not a held-out number** (`cv_score_is_holdout: false`); any
  **degraded/mock LLM narrative** is flagged, never dressed up as real; a **training failure** is
  reported honestly with no headline metric to overclaim; and the executor is still a
  *process-isolation*, not a *security*, sandbox (Day 19).
- Writes `MODEL_CARD.md` + a narrative-free `report.json` to the run's (git-ignored) artifact dir.

**Both nodes wired in** — `crewml/crew/nodes.py::ensembler` and `::reporter` now call the real
agents; the `_stub` helper is gone. **Graph topology and state schema are unchanged** — the seven
nodes and the Critic loop are exactly as wired on Day 5.

New driver **`scripts/run_crew.py`** (rewritten from the Day-5 skeleton driver) — runs the **full
real crew** end-to-end on a dataset (or `--all`), writing committed **`results/day11_crew_run.json`**
(final model, ensemble-vs-single scores, trace, warnings, holdout seal) and a committed
**`results/sample_model_card.md`** so the deliverable is inspectable in the repo.

New tests **`tests/test_ensembler.py`** (14) + **`tests/test_reporter.py`** (12), and
**`tests/test_graph.py`** updated (the last two stubs are real now).

## Verification

**First full end-to-end crew run — all 5 datasets, every node real, grid search on**
(`python scripts/run_crew.py --all`):

| dataset   | metric   | final model | headline CV        | single vs. ensemble | Ensembler kept | holdout sealed |
|-----------|----------|-------------|--------------------|---------------------|----------------|:--------------:|
| credit-g  | ROC AUC  | random_forest            | **0.7994** | 0.7994 vs 0.7955 (−0.0039) | single | ✅ |
| diabetes  | ROC AUC  | logistic_regression      | **0.8372** | 0.8372 vs 0.8224 (−0.0148) | single | ✅ |
| vehicle   | macro-F1 | logistic_regression      | **0.7931** | 0.7931 vs 0.7814 (−0.0118) | single | ✅ |
| cpu_small | R²       | hist_gradient_boosting   | **0.9779** | 0.9779 vs 0.9486 (−0.0294) | single | ✅ |
| kin8nm    | R²       | hist_gradient_boosting   | **0.8214** | 0.8214 vs 0.7138 (−0.1077) | single | ✅ |

**The headline is the Ensembler's *restraint*, and it's an honest result.** On all five datasets,
with the candidates **already tuned by the Trainer's grid search**, the equal-weight ensemble of
three candidates of *unequal* strength scored **below** the single tuned winner — the weaker members
(a plain `ridge`, an untuned-regime `logistic_regression`) drag a soft-vote / average away from the
strong tuned model. The Ensembler saw that on the same seeded folds and **declined the ensemble every
time, shipping the single model** — so the crew's final scores are exactly the Day-10 tuned numbers
(credit-g 0.7994, diabetes 0.8372, vehicle 0.7931, cpu_small 0.9779, kin8nm 0.8214), never a point
worse. A combiner that *always* combines isn't sophisticated — it's a liability; knowing when the
team is better off with its single best player is the point.

**The winning path is proven, not assumed.** Because the tuned runs all decline the ensemble, the
*win* path is demonstrated in the **untuned regime**, where the candidates are weaker and closer in
strength: at default params on credit-g the soft-vote ensemble scores **0.7972 vs the single 0.7940
(+0.0031)** and the Ensembler **keeps it** — a real ensemble genuinely improving the model, its
combined estimator refit and persisted. So both branches of the "keep-only-if-it-helps" rule are
exercised on real data: it takes the ensemble when it wins, and refuses it when it doesn't. The unit
suite pins the invariant directly — the final CV score is always `>= single_best_cv_score`.

**The Reporter's model card is a real deliverable** (committed as `results/sample_model_card.md`): it
names the final model and why it was chosen, the per-candidate CV table, the Critic loop, the LLM
assistance ("none live — 3 requested, all unavailable/mock"), and a Limitations section that states
in plain words that the numbers are CV-on-train not held-out. Nothing in it could be mistaken for a
sealed-split result.

**Honesty invariants, all green.** Every score is a **cross-validated estimate on train**
(`cv_score_is_holdout: false`); the run re-verifies the **holdout seal after every dataset** (all
`holdout_sealed: True`); and the advisory narratives were run with `--no-llm`, so the deterministic
core stands alone (the Groq `organization_restricted` lockout that dogged Days 8–10 is now simply
sidestepped for the committed run — no model got a vote on any decision).

**Tests: 214 passed, 3 skipped** (188 prior + 26 net new), suite **fully offline**. New guards pin:

- **Ensembler** — real soft-vote/averaging combination; the self-consistent single re-score
  **reproduces the Trainer's number**; the final model is **never worse than the single best**; the
  combined model is persisted and loadable; regression averages (not votes); determinism; a failed
  Trainer run yields "not attempted" (no crash); a too-thin candidate set falls back to single;
  structural no-peeking (never calls the held-out loader).
- **Reporter** — faithful synthesis of the final model / tables / Critic passes; the model card
  states scores are CV-not-holdout and names the ensemble decision; warnings flag a mock run, a
  no-live-LLM run, and a training failure; LLM-usage aggregation; render is pure; structural
  no-peeking.
- **Graph** — the full run terminates at a **real** Reporter with **every node non-stub**, the
  Ensembler compares against the single best and never ships worse, and no crew module references
  the held-out loader.

## Honesty & scope notes

- **"Kept the single model" is a win for the design, not a null result.** The deliverable was an
  Ensembler that *decides soundly* — takes the ensemble when it helps, refuses it when it doesn't —
  and both paths are exercised on real data (declined under tuning, accepted at default params). That
  the tuned runs don't need it is a property of strong single models + equal-weight voting, not a gap.
- **Scores are unchanged and CV-on-train.** The final numbers equal Day 10's because the Ensembler
  read the Trainer's tuned models and correctly kept the best one; it never perturbs a number it
  decides not to use. The crew-vs-solo-vs-AutoML head-to-head on the **sealed split** is **Day 12**
  (Phase 3), which is where these CV estimates finally meet the held-out exam.
- **The Reporter has no LLM by design.** It renders decisions the specialists already made; giving it
  a model to "write up" the run would introduce exactly the hallucination risk the rest of the crew
  is built to avoid.
- **Weighted/stacked ensembling is deliberately out of scope here.** Equal-weight voting is the honest
  default (tuning ensemble weights on the same CV it's judged by is its own overfitting trap); a
  smarter combiner is a candidate for the Phase-3 ablation studies, not a Day-11 claim.

## Phase 2 Wrap-Up (Days 5–11)

Phase 2 set out to build the **MVP crew**: a genuine multi-agent LangGraph system with a working
critique loop. It is **done** — all seven nodes are real specialists and the crew runs a dataset from
raw to finished model + model card on its own:

- **Day 5 — skeleton.** `CrewState`, the seven-node graph, the conditional Critic edge, and the
  `max_iterations` guard. The loop was bounded by construction from day one.
- **Day 6 — the executor.** The sandboxed subprocess tool (timeout, isolated workdir, captured
  metrics/artifacts) every code-writing agent runs through — the crux the pipeline stands on.
- **Day 7 — Profiler.** Train-only DataProfile: schema, missingness (incl. disguised-missing zeros),
  imbalance, leakage checks.
- **Day 8 — Planner.** Profile → ModelingPlan: drops, dtype-aware preprocessing, candidate models +
  seed grids, CV scheme, imbalance strategy; the Critic-response side wired in advance.
- **Day 9 — Feature Engineer + Trainer.** The first generated-code node (generate-then-validate FE)
  and the first *number* — cross-validated on train, honestly labelled, with a saved model.
- **Day 10 — Critic.** The reviewer that closes the loop: deterministic diagnosis (overfit / underfit
  / leakage / imbalance / wrong-metric) → iterate-or-finalize, with convergence + the guard.
- **Day 11 — Ensembler + Reporter.** Keep-only-if-it-helps model combination, and the deterministic
  report + model card. **First full end-to-end run, every node real.**

**Cross-cutting invariants held all phase:** the crew never loaded the held-out split (structural,
source-inspected on every module); every score is CV-on-train, never dressed up as a held-out result;
every LLM contribution is advisory and validated (FE) or advisory-only (narratives), never a source of
a decision, and degrades honestly when the provider is out; the loop cannot run away (Critic
convergence + hard guard). **214 tests, fully offline.**

**What Phase 2 is *not* yet (by design):** no held-out scoring (Phase 3 — the honest head-to-head vs.
the solo agent and the AutoML ceiling); the executor is process-isolation, not a security sandbox
(Phase 4); no self-repair on execution failure (Day 20); no API/dashboard (Phase 5). Those are the
next phases, and Phase 2 deliberately drew the line at "a working, honest MVP crew".

## Next

**Day 12 (Phase 3 opens) — Comparison Studies.** Run the full crew across all 5 datasets and score the
final models on the **LOCKED held-out split** for the first time, head-to-head against the solo-agent
baseline (Day 3) and the FLAML AutoML ceiling (Day 4): *does the team actually beat the lone expert and
the off-the-shelf tool?* This is where Phase 2's cross-validated estimates finally meet the sealed exam.
