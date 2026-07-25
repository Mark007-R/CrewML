# Day 20 — Self-repair recovery study

> **NOT AN LLM MEASUREMENT.** The repairer here is a deterministic, fault-blind stand-in, run because no live provider was reachable (Groq returned `organization_restricted` for every call on 2026-07-25). These numbers measure the harness — does an injected fault detonate in the real Trainer, does the loop fire, is the sandboxed re-run adopted, is the clean score reproduced, does the persisted FE artifact stay consistent. They say **nothing** about whether a model can diagnose a traceback; that measurement is deferred until a provider is live.

Repairer: **scripted_stand_in** (deterministic-repair-policy) · attempt budget: 2 · mechanism recovery: **16/16 = 100%** · false-positive repairs on clean runs: 0 · FE-artifact inconsistencies: 0 · holdout seal intact: True

| Dataset | Fault | Recovered | Attempt | CV after | Δ vs clean | Tokens | Wall s |
|---|---|---|---|---|---|---|---|
| credit-g | none_control | n/a (control) | — | 0.7940 | — | 0 | 11.3 |
| credit-g | name_error | yes | 1 | 0.7940 | +0.0000 | 0 | 15.0 |
| credit-g | key_error | yes | 1 | 0.7940 | +0.0000 | 0 | 14.3 |
| credit-g | type_error | yes | 1 | 0.7940 | +0.0000 | 0 | 14.6 |
| credit-g | syntax_error | yes | 1 | 0.7940 | +0.0000 | 0 | 12.7 |
| credit-g | zero_division | yes | 1 | 0.7940 | +0.0000 | 0 | 13.4 |
| credit-g | import_error | yes | 1 | 0.7940 | +0.0000 | 0 | 13.8 |
| credit-g | attribute_error | yes | 1 | 0.7940 | +0.0000 | 0 | 14.2 |
| credit-g | index_error | yes | 1 | 0.7940 | +0.0000 | 0 | 15.6 |
| cpu_small | none_control | n/a (control) | — | 0.9773 | — | 0 | 16.0 |
| cpu_small | name_error | yes | 1 | 0.9773 | +0.0000 | 0 | 19.4 |
| cpu_small | key_error | yes | 1 | 0.9773 | +0.0000 | 0 | 19.7 |
| cpu_small | type_error | yes | 1 | 0.9773 | +0.0000 | 0 | 18.8 |
| cpu_small | syntax_error | yes | 1 | 0.9773 | +0.0000 | 0 | 16.0 |
| cpu_small | zero_division | yes | 1 | 0.9773 | +0.0000 | 0 | 18.7 |
| cpu_small | import_error | yes | 1 | 0.9773 | +0.0000 | 0 | 18.0 |
| cpu_small | attribute_error | yes | 1 | 0.9773 | +0.0000 | 0 | 17.7 |
| cpu_small | index_error | yes | 1 | 0.9773 | +0.0000 | 0 | 17.2 |
