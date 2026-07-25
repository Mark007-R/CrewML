# Day 20 — Self-repair recovery study

Repairer: **groq** (llama-3.3-70b-versatile) · attempt budget: 2 · recovery rate: **2/2 = 100%** · false-positive repairs on clean runs: 0 · FE-artifact malformed: 0 · holdout seal intact: True

| Dataset | Fault | Recovered | Attempt | CV after | Δ vs clean | Tokens | Wall s |
|---|---|---|---|---|---|---|---|
| credit-g | none_control | n/a (control) | — | 0.7940 | — | 0 | 12.3 |
| credit-g | non_finite | yes | 1 | 0.7895 | -0.0046 ¹ | 5183 | 21.5 |
| cpu_small | none_control | n/a (control) | — | 0.9773 | — | 0 | 14.0 |
| cpu_small | non_finite | yes | 1 | 0.9772 | -0.0001 ¹ | 5112 | 25.4 |

¹ This fault has **no restorable intent** (it references a column that does not exist, or a ratio that must be guarded differently), so a correct fix cannot reproduce the control's feature set. A non-zero Δ here is EXPECTED and is not evidence of a bad repair. Fidelity statistics are scoped to the restorable faults.
