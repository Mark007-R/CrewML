# CrewML

**An autonomous multi-agent ML engineering crew.** Give it a raw tabular dataset
and a task (classification or regression); a LangGraph crew of specialised agents
profiles the data, plans an approach, engineers features, trains and critiques
models in a loop, ensembles the best, and writes a report — then proves it beat a
single solo agent and classical AutoML on data it never saw.

> **The claim we are out to earn:** a genuine multi-agent crew with a critique loop
> produces better models than one LLM doing it single-shot, and is competitive with
> a strong classical AutoML baseline — measured honestly on a locked held-out set.

## The crew

```
Profiler → Planner → Feature Engineer → Trainer → Critic ─┐
   ↑                                                        │ (iterate with
   └──────────────── specific fix instructions ────────────┘  specific fixes)
                                   │ (finalize)
                                   ▼
                          Ensembler → Reporter
```

Each node is a LangGraph agent. The **Critic** is the differentiator: it diagnoses
over/underfitting, leakage, class imbalance, and wrong-metric choices, then loops
back to the Planner / Feature Engineer with *specific* instructions — bounded by a
`max_iterations` guard. A shared **sandboxed Python executor** lets agents run code
against the training data.

## Honest evaluation

The project lives or dies on its evaluation, spelled out in
[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md):

- 5 pinned OpenML datasets (2 binary, 1 multiclass, 2 regression).
- One split, seed-locked, into `train` (all the crew ever touches) and a
  **LOCKED `holdout`** used once for final scoring.
- Every split is SHA-256 sealed in [`results/dataset_manifest.json`](results/dataset_manifest.json);
  a test proves the holdout was never altered.
- The crew is compared against a solo agent, classical AutoML (FLAML), a default
  RandomForest, and a Dummy floor — same data, same metric, same holdout.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                 # add GROQ_API_KEY, or leave blank for mock mode
python scripts/prepare_datasets.py   # download + split + lock the 5 datasets
python -m pytest tests/              # verify the seals hold
```

The pipeline runs offline in **mock mode** when no LLM key is set — but mock numbers
are never reported as real.

## Baselines (the bar the crew must clear)

Held-out scores per dataset, primary metric, higher is better. The full board and
its provenance live in [`results/baselines_table.md`](results/baselines_table.md).

| Dataset | Metric | Dummy (floor) | default RF | Solo agent | AutoML (FLAML) |
|---|---|---|---|---|---|
| credit-g | ROC AUC | 0.5000 | 0.7783 | 0.7521 *(mock)* | 0.7352 |
| diabetes | ROC AUC | 0.5000 | 0.8118 | 0.7987 *(mock)* | 0.8039 |
| vehicle | macro-F1 | 0.1028 | 0.7260 | 0.7763 *(mock)* | 0.7785 |
| cpu_small | R² | −0.0029 | 0.9726 | 0.9747 *(mock)* | 0.9759 |
| kin8nm | R² | −0.0002 | 0.6948 | 0.8120 *(mock)* | 0.8421 |

*(mock)* — the solo agent ran without an LLM key, so its column is MOCK and not a
headline number (EVAL_PROTOCOL §5). Notably the FLAML ceiling does **not** dominate
everywhere under its budget — the untuned RandomForest still edges it on the two
small binary sets — which is exactly the terrain where the Phase-2 crew has to earn
its seat.

## Status

Built in the open over 30 days. See [`PROGRESS_LOG.md`](PROGRESS_LOG.md) for the
daily trail and `reports/` / `explainers/` for the per-day write-ups.

- **Phase 1 — Foundation & Baselines** (Days 1–4) ✓ complete
- Phase 2 — MVP Crew (Days 5–11) ← next
- Phase 3 — Comparison Studies (Days 12–18)
- Phase 4 — Hardening & Safety (Days 19–23)
- Phase 5 — Production Wrapper (Days 24–27)
- Phase 6 — Ship (Days 28–30)

## License

Single-author project by Mark Rodrigues.
