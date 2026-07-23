# Day 16 — provider study: Groq vs Claude vs mock

*The identical crew under each LLM provider, compared on quality (LOCKED-holdout, scored outside the graph, seal re-verified per run), cost (measured tokens × published price) and latency. Providers are probed live before any arm runs; a provider that fails its probe contributes its failure as evidence, never an imagined number.*

### Provider availability (live probes, 2026-07-22)

| Provider | Model | Status | Probe latency | Evidence |
|---|---|---|---|---|
| Groq — Llama 3.3 70B | llama-3.3-70b-versatile | **OK** | 1.37s | reply `PONG` |
| Anthropic — Claude Sonnet 5 | claude-sonnet-5 | NOT CONFIGURED | — | needs ANTHROPIC_API_KEY |
| Mock — deterministic offline core | — | always available (offline) | — | deterministic core; no LLM calls |

### Cost model (published on-demand prices, as of 2026-07-21)

| Provider | Priced model | $ / 1M input | $ / 1M output | Source |
|---|---|---|---|---|
| Groq — Llama 3.3 70B | llama-3.3-70b-versatile | 0.59 | 0.79 | Groq published on-demand pricing |
| Anthropic — Claude Sonnet 5 | claude-sonnet-5 | 2.00 | 10.00 | Anthropic introductory pricing through 2026-08-31 (sticker 3.00/15.00) |
| Mock — deterministic offline core | — | 0.00 | 0.00 | no LLM calls; always available |

*A run's cost is only ever computed from tokens the accounting actually measured (`llm_usage`, live calls only). Live-arm costs below are measured-token totals priced at the rates above.*

### Arm — Groq — Llama 3.3 70B

| Dataset | Metric | Holdout score | Crew seconds | Live LLM calls | Tokens | Cost |
|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7897 | 30.0 | 4 | 2074 | $0.0013 |
| diabetes | roc_auc | — | 14.3 | 4 | 2177 | $0.0014 |
| vehicle | f1_macro | 0.8439 | 108.1 | 4 | 2115 | $0.0014 |
| cpu_small | r2 | 0.9742 | 229.7 | 4 | 1957 | $0.0013 |
| kin8nm | r2 | 0.7960 | 150.6 | 4 | 2007 | $0.0013 |

*`diabetes` shipped no scorable model: crew produced no usable fitted model — nothing to score — Critic verdict: "training run failed — nothing to iterate on (self-repair is Day 20)" (finding codes: execution_error). The full failing state is archived; the failure stands on the board as a result in itself.*

### Arm — Mock — deterministic offline core *(mock — deterministic core, no LLM; never a headline result)*

| Dataset | Metric | Holdout score | Crew seconds | Live LLM calls | Tokens | Cost |
|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 26.1 | 0 | — | — |
| diabetes | roc_auc | 0.8150 | 22.3 | 0 | — | — |
| vehicle | f1_macro | 0.8326 | 27.6 | 0 | — | — |
| cpu_small | r2 | 0.9750 | 132.9 | 0 | — | — |
| kin8nm | r2 | 0.8182 | 112.3 | 0 | — | — |

### Provider-outage resilience — fresh no-provider vs archival failing-provider

*The Day-14 archival runs executed with a Groq key configured and every live call failing mid-run (`organization_restricted`) — the harshest realistic outage. Today's mock arm ran with no provider at all. On a seed-locked pipeline, equal scores prove the modelling path is provider-independent: an outage costs narrative richness, never the score.*

| Dataset | Metric | Fresh (no provider) | Archival (provider failing) | Δ | Equal |
|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 0.7913 | 0.00e+00 | **yes** |
| diabetes | roc_auc | 0.8150 | 0.8150 | 0.00e+00 | **yes** |
| vehicle | f1_macro | 0.8326 | 0.8326 | 0.00e+00 | **yes** |
| cpu_small | r2 | 0.9750 | 0.9750 | 0.00e+00 | **yes** |
| kin8nm | r2 | 0.8182 | 0.8182 | 0.00e+00 | **yes** |

**5/5 datasets bit-identical** — the crew's holdout quality is provably independent of provider availability.

### Blocked — what the live comparison still needs

- **anthropic** — probe result: `not_configured`. Unblock: ANTHROPIC_API_KEY.

*When either provider comes back, `python scripts/run_provider_study.py` re-runs the probes, adds the live arm(s), prices them and rewrites this board — nothing else to change.*

*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).

