# Day 20 — Self-repair recovery study

Repairer: **groq** (llama-3.3-70b-versatile) · attempt budget: 2 · recovery rate: **18/18 = 100%** · false-positive repairs on clean runs: 0 · FE-artifact malformed: 0 · holdout seal intact: True

| Dataset | Fault | Recovered | Attempt | CV after | Δ vs clean | Tokens | Wall s |
|---|---|---|---|---|---|---|---|
| credit-g | none_control | n/a (control) | — | 0.7940 | — | 0 | 12.6 |
| credit-g | name_error | yes | 1 | 0.7940 | +0.0000 | 5008 | 21.7 |
| credit-g | key_error | yes | 1 | 0.7872 | -0.0069 ¹ | 5064 | 39.5 |
| credit-g | type_error | yes | 1 | 0.7940 | +0.0000 | 5099 | 27.6 |
| credit-g | syntax_error | yes | 1 | 0.7940 | +0.0000 | 5032 | 29.2 |
| credit-g | zero_division | yes | 1 | 0.7940 | +0.0000 | 5023 | 22.8 |
| credit-g | import_error | yes | 1 | 0.7940 | +0.0000 | 5034 | 25.6 |
| credit-g | attribute_error | yes | 1 | 0.7940 | +0.0000 | 5051 | 25.5 |
| credit-g | index_error | yes | 1 | 0.7940 | +0.0000 | 5103 | 25.5 |
| cpu_small | none_control | n/a (control) | — | 0.9773 | — | 0 | 21.4 |
| cpu_small | name_error | yes | 1 | 0.9773 | +0.0000 | 4922 | 31.3 |
| cpu_small | key_error | yes | 1 | 0.9773 | +0.0000 ¹ | 4978 | 26.0 |
| cpu_small | type_error | yes | 1 | 0.9773 | +0.0000 | 5005 | 33.3 |
| cpu_small | syntax_error | yes | 1 | 0.9773 | +0.0000 | 4946 | 23.0 |
| cpu_small | zero_division | yes | 1 | 0.9773 | +0.0000 | 4925 | 26.7 |
| cpu_small | import_error | yes | 1 | 0.9773 | +0.0000 | 4948 | 34.1 |
| cpu_small | attribute_error | yes | 1 | 0.9773 | +0.0000 | 4965 | 26.2 |
| cpu_small | index_error | yes | 1 | 0.9773 | +0.0000 | 5017 | 48.5 |
| credit-g | non_finite | yes | 1 | 0.7895 | -0.0046 ¹ | 5183 | 21.5 |
| cpu_small | non_finite | yes | 1 | 0.9772 | -0.0001 ¹ | 5112 | 25.4 |

¹ This fault has **no restorable intent** (it references a column that does not exist, or a ratio that must be guarded differently), so a correct fix cannot reproduce the control's feature set. A non-zero Δ here is EXPECTED and is not evidence of a bad repair. Fidelity statistics are scoped to the restorable faults.
