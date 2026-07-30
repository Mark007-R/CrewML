# Day 25 — Node cache & telemetry study

Deterministic core only (LLM narratives disabled): the savings below are recomputation the cache avoided, **not** provider latency. Scores are CV-on-train, never held-out.

## Profiler+Planner nodes — cold vs warm

| dataset | cold (s) | warm (s) | saved (s) | speedup | warm answer identical |
|---|---:|---:|---:|---:|:--|
| credit-g | 0.362 | 0.002 | 0.36 | 181.0x | True |
| diabetes | 0.142 | 0.003 | 0.139 | 47.3x | True |
| vehicle | 0.264 | 0.003 | 0.261 | 88.0x | True |
| cpu_small | 0.172 | 0.002 | 0.17 | 86.0x | True |
| kin8nm | 0.148 | 0.001 | 0.147 | 148.0x | True |

## API round-trip on credit-g — what /metrics records

| run | status | duration (s) | tokens | cache hits | cache misses | final CV score |
|---|---|---:|---:|---:|---:|---:|
| cold | succeeded | 23.156 | 0 | 0 | 2 | 0.797173 |
| warm | succeeded | 23.25 | 0 | 2 | 0 | 0.797173 |

Same Day-23 result fingerprint across cold and warm: **True** — the cache changed the cost, not the answer.
