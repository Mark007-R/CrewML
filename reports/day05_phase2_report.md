# Day 5 — Phase 2 (MVP Crew) · LangGraph state schema + graph skeleton

**Date:** 2026-07-10 · **Phase:** 2 (MVP Crew, Days 5–11) · **PR:** opens today.

## Goal

Lock the crew's **control flow and shared state** before writing a single real
agent. Day 5 ships a compilable, runnable LangGraph skeleton: the seven nodes
exist as honest stubs, the edges are wired, the Critic feedback loop is closed
with a conditional edge, and the `max_iterations` guard bounds it. No LLM call,
no data read, no held-out scoring happens today — this is a wiring proof, not a
modeling result.

## What shipped

New subpackage **`crewml/crew/`**:

- **`state.py`** — the shared **`CrewState`** (`TypedDict`, `total=False`) that
  flows through every node. Run inputs (`dataset_key`, `task`, `subtype`,
  `metric`, `max_iterations`) are set once; produced fields (`profile`, `plan`,
  `fe_code`, `training`, `decision`, `ensemble`, `report`) start `None` so a node
  populating one is always an observable event. Two **append-only channels** use
  an `operator.add` reducer — `critiques` (one per Critic pass) and `trace` (node
  visit order) — so the loop *grows* history instead of overwriting it. All values
  are JSON/msgpack-friendly so LangGraph checkpointing and the Day-26 dashboard
  will work unchanged. `initial_state(spec, max_iterations=…)` seeds a run.
- **`nodes.py`** — the seven node stubs (Profiler → Planner → Feature Engineer →
  Trainer → Critic → Ensembler → Reporter). Each returns a partial state update
  and a `trace` entry; every placeholder payload is flagged `"stub": True`. The
  one piece of **real, shipping logic** is **`route_after_critic`**, the
  conditional edge out of the Critic.
- **`graph.py`** — `build_graph()` assembles the `StateGraph`; `build_crew()`
  compiles it (optional checkpointer arg for the Phase-5 run store). `CREW_NODES`
  exposes the topology as data for tests and the future dashboard.

New script **`scripts/run_crew.py`** — compiles and invokes the skeleton on one
dataset, prints the node trace + terminal summary, and writes
`artifacts/crew/<dataset>/skeleton_run.json` (git-ignored).

New tests **`tests/test_graph.py`** (14) — topology, the router's three cases,
a full skeleton invocation, and the honesty guards.

## The control flow

```
START
  → profiler
  → planner  ←──────────────────────────.
  → feature_engineer                     │  iterate: Critic asked for another pass
  → trainer                              │           AND the budget isn't spent
  → critic ──[route_after_critic]────────┘
              \──(finalize)──→ ensembler → reporter → END
```

The **only** conditional edge is out of the Critic. `route_after_critic` applies
the guard first — once `iteration >= max_iterations` it finalises no matter what
the Critic wants — then honours the Critic's `decision`. The loop is therefore
bounded **by construction**: a runaway crew is structurally impossible, not merely
unlikely.

## Verification

Skeleton run (`python scripts/run_crew.py --dataset credit-g --max-iterations 3`):

```
node trace (15 steps):
profiler → planner → feature_engineer → trainer → critic
         → planner → feature_engineer → trainer → critic
         → planner → feature_engineer → trainer → critic
         → ensembler → reporter
Critic passes run: 3 / 3 (budget)   critiques recorded: 3   report present: True
```

The stub Critic always requests another pass, so the run spends its full budget
and the **guard** is what stops the loop — exactly the property worth proving on
day one of the crew.

**Tests: 88 passed, 3 skipped** (74 prior + 14 new; the 3 skips are the
solo-agent mock-mode contract tests, unchanged). The new tests confirm the graph
compiles with all seven nodes, the router iterates only when asked *and* under
budget while the guard overrides an "iterate" at the ceiling, a full run
terminates at the Reporter with three accumulated critiques, and the honesty
guards hold: the Trainer stub emits `cv_score=None`, every payload is flagged
`stub`, and **no module in `crewml/crew/` references the holdout** (structural
no-peeking, asserted by source inspection).

## Honesty notes

- Nothing today produces a number. The Trainer stub deliberately returns
  `cv_score=None` so no placeholder can be mistaken for a real held-out result
  (EVAL_PROTOCOL §5).
- The held-out isolation invariant is now *structural*: `CrewState` carries only
  `dataset_key`, and a test fails the build if any crew module so much as mentions
  the holdout loader.
- `langgraph>=0.2` / `langchain-core>=0.3` were already pinned in
  `requirements.txt` (Phase-2 section); installed `langgraph 1.2.9`,
  `langchain-core 1.4.9` this run.

## Next

Day 6 — the **sandboxed Python executor tool**: subprocess with a timeout,
captured stdout/artifacts/metrics, temp workdir. The crux of the whole system and
the shared tool every real agent will call.
