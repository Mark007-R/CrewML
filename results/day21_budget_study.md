# Day 21 — Run-budget study: cost & latency caps, enforced and measured

The full crew ran on `credit-g` under three budget regimes
(provider: live **groq** (llama-3.3-70b-versatile), probe 1.816s). The run budget (Day 21, `crewml.budget`) gates every
LLM call pre-call, charges it post-call, and lets the Critic finalise on an
exhausted or unaffordable budget — so a spent budget degrades a run to its
deterministic core; it never crashes it.

| scenario | caps (tok/time) | tokens spent | LLM calls | wall-clock | passes | final decision | budget bound? | CV score |
|----------|-----------------|-------------:|-----------|-----------:|:------:|----------------|---------------|:--------:|
| reference | 200,000 tok / 1800s | 2,057 | 4 live / 0 refused | 29.26s | 1/3 | finalize | never bound | 0.7946 |
| tight_tokens | 1,200 tok / 1800s | 1,710 | 3 live / 1 refused | 27.86s | 1/3 | finalize | refusals | 0.7946 |
| tight_time | 200,000 tok / 10s | 1,704 | 3 live / 1 refused | 28.36s | 1/3 | finalize | refusals | 0.7946 |

**Reading the table.** "Budget bound?" reports whether enforcement actually fired:
`refusals` = calls turned away by the gate, `loop-stop` = the Critic finalised on
budget grounds, `never bound` = the run finished inside its caps (the expected
shape for `reference`). CV scores are estimates on train, never holdout.

## Notes

- Scores are CV estimates on train (cv_score_is_holdout: false); the sealed holdout was verified untouched after every scenario.
- A budget-starved run is not a mock run: zero live narratives beside a non-zero n_refused is the enforcement working, and is labelled as such.
- Grid search was disabled (CREWML_TRAINER_PARAM_SEARCH=0) for every scenario so wall-clock differences come from LLM calls and CV, not the search grid.
