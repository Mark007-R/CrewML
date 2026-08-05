---
title: CrewML
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent ML crew that beats a solo agent — honestly.
---

# CrewML — an autonomous multi-agent ML engineering crew

Upload a tabular CSV, **choose the target column** (never guessed), and a
LangGraph crew — Profiler → Planner → Feature Engineer → Trainer → Critic
(loops back with specific instructions) → Ensembler → Reporter — trains and
evaluates a model live, then writes a report and model card.

**The honesty rules apply to your upload too:** the API splits your CSV
itself, SHA-256-seals the holdout into a run manifest, and keeps it
structurally unreachable from the crew — final scoring is the only reader.

Measured on the project's locked 5-dataset benchmark: the crew beats a
one-shot solo agent 3/3 where solo produced a model at all (solo crashed on
the other 2), beats a default RandomForest 5/5, and is competitive with
classical AutoML (3/5). Those crew scores come from the **deterministic core**
(the benchmark runs executed during a provider outage; the live-LLM arm is
scored separately and differs per dataset) — see the repo's README *Provenance*
note for the full accounting.

- **Source, results, and the full 30-day build log:**
  [github.com/Mark007-R/CrewML](https://github.com/Mark007-R/CrewML)
- If no `GROQ_API_KEY` secret is configured the Space runs in **mock mode**
  (offline, clearly labelled) — the pipeline still executes end-to-end, but
  no numbers from it are real LLM output.
- Runs on the Space share one small CPU pod: a live crew run takes a few
  minutes; the run store is ephemeral and resets on restart.
