# Day 23 — Run-level reproducibility study

Claim under test: **same pins ⇒ same result fingerprint** (`crewml.manifest.result_fingerprint` — SHA-256 over scores, chosen model, FE-code hash, trace and Critic decisions; prose/latency/tokens excluded). Every run is a separate fresh process; the parent compares only the run manifests, exactly as a human re-running the pipeline would. Grid search off (CV at default params) for bounded wall-clock; `max_iterations=3` (production default).

## Arm 1 — deterministic core, run twice (fresh process each)

| dataset | run 1 fingerprint | run 2 fingerprint | cv score | final model | identical |
|---|---|---|---|---|---|
| credit-g | `8874e648ad83` | `8874e648ad83` | 0.797173 | ensemble | ✅ bit-identical |
| cpu_small | `be1590f21366` | `be1590f21366` | 0.977309 | single | ✅ bit-identical |

**Verdict: all deterministic-core runs reproduce bit-identically.**

## Arm 2 — the seed must matter

Fingerprints do not embed the seed, so Arm 1 alone cannot distinguish *controlled* from *ignored*. A different `CREWML_SEED` must move the outcome:

| dataset | seed | fingerprint | cv score |
|---|---|---|---|
| credit-g | 42 | `8874e648ad83` | 0.797173 |
| credit-g | 43 | `c1f90b1bb215` | 0.801711 |

**Verdict: the seed reaches the model — fingerprint and score both moved.**

## Arm 3 — live LLM double-run (labelled; not a determinism claim)

Provider: **groq / llama-3.3-70b-versatile**, temperature 0.0, identical pins, two fresh runs on `credit-g`:

| layer | run 1 | run 2 | reproduced |
|---|---|---|---|
| scored result (fingerprint) | `c0b730929550` | `10c259b53208` | ❌ |
| generated FE code (sha256) | `b948352137c8` | `f36c2d2b6f5b` | ❌ |
| advisory narratives (sha256) | `e6d00e849702` | `83dddb0a6636` | ❌ |

CV 0.803088 vs 0.798698; final model ensemble vs ensemble.

**Verdict: the live runs DIVERGED at the scored layer** — the provider returned different FE/plan content across identical prompts. This is the expected failure mode of a live LLM arm, recorded rather than hidden; the run manifests pin exactly what differed.

## Pins of record

- seed 42 · python 3.11.9 · numpy 1.26.4 · pandas 2.1.4 · scikit-learn 1.8.0 · langgraph 1.2.9
- provider groq / llama-3.3-70b-versatile (mock_mode=False), temperature default 0.0
- git `7ae606b76739` (dirty tree — run predates the Day-23 commit)
- holdout untouched throughout: ✅
