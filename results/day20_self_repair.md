# Day 20 — Self-repair recovery study

> **PARTIALLY UNMEASURED.** 2 of 2 injected runs never reached the provider (all repair attempts failed at the llm stage — e.g. a rate limit or outage). Those runs are UNMEASURED, excluded from the recovery-rate denominator, and must not be read as failures to repair.

Repairer: **groq** (llama-3.3-70b-versatile) · attempt budget: 2 · recovery rate: **not measurable** (no injected run reached the provider) · false-positive repairs on clean runs: 0 · FE-artifact malformed: 0 · holdout seal intact: True

| Dataset | Fault | Recovered | Attempt | CV after | Δ vs clean | Tokens | Wall s |
|---|---|---|---|---|---|---|---|
| credit-g | none_control | n/a (control) | — | 0.7940 | — | 0 | 10.6 |
| credit-g | non_finite | unmeasured (provider) | — | — | — | 0 | 7.3 |
| cpu_small | none_control | n/a (control) | — | 0.9773 | — | 0 | 17.1 |
| cpu_small | non_finite | unmeasured (provider) | — | — | — | 0 | 8.7 |

¹ This fault has **no restorable intent** (it references a column that does not exist, or a ratio that must be guarded differently), so a correct fix cannot reproduce the control's feature set. A non-zero Δ here is EXPECTED and is not evidence of a bad repair. Fidelity statistics are scoped to the restorable faults.
