# Day 16 — provider study: Groq vs Claude vs mock

*The identical crew under each LLM provider, compared on quality (LOCKED-holdout, scored outside the graph, seal re-verified per run), cost (measured tokens × published price) and latency. Providers are probed live before any arm runs; a provider that fails its probe contributes its failure as evidence, never an imagined number.*

### Provider availability (live probes, 2026-07-21)

| Provider | Model | Status | Probe latency | Evidence |
|---|---|---|---|---|
| Groq — Llama 3.3 70B | llama-3.3-70b-versatile | **UNAVAILABLE** | 1.50s | `BadRequestError: Error code: 400 - {'error': {'message': 'Organization has been restricted. Please reach out to support if you believe this was in error.', 'type': 'invalid_request_error', 'code': 'organization_restricted'}}` |
| Anthropic — Claude Sonnet 5 | claude-sonnet-5 | NOT CONFIGURED | — | needs ANTHROPIC_API_KEY |
| Mock — deterministic offline core | — | always available (offline) | — | deterministic core; no LLM calls |

### Cost model (published on-demand prices, as of 2026-07-21)

| Provider | Priced model | $ / 1M input | $ / 1M output | Source |
|---|---|---|---|---|
| Groq — Llama 3.3 70B | llama-3.3-70b-versatile | 0.59 | 0.79 | Groq published on-demand pricing |
| Anthropic — Claude Sonnet 5 | claude-sonnet-5 | 2.00 | 10.00 | Anthropic introductory pricing through 2026-08-31 (sticker 3.00/15.00) |
| Mock — deterministic offline core | — | 0.00 | 0.00 | no LLM calls; always available |

*A run's cost is only ever computed from tokens the accounting actually measured (`llm_usage`, live calls only). No live arm ran this session, so no cost is reported — the model above is wired into the runner and prices any future live arm with no code change.*

### Arm — Mock — deterministic offline core *(mock — deterministic core, no LLM; never a headline result)*

| Dataset | Metric | Holdout score | Crew seconds | Live LLM calls | Tokens | Cost |
|---|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 25.5 | 0 | — | — |
| diabetes | roc_auc | 0.8150 | 21.2 | 0 | — | — |
| vehicle | f1_macro | 0.8326 | 28.3 | 0 | — | — |
| cpu_small | r2 | 0.9750 | 105.7 | 0 | — | — |
| kin8nm | r2 | 0.8182 | 119.0 | 0 | — | — |

### Provider-outage resilience — fresh no-provider vs archival failing-provider

*The Day-14 archival runs executed with a Groq key configured and every live call failing mid-run (`organization_restricted`) — the harshest realistic outage. Today's mock arm ran with no provider at all. On a seed-locked pipeline, equal scores prove the modelling path is provider-independent: an outage costs narrative richness, never the score.*

| Dataset | Metric | Fresh (no provider) | Archival (provider failing) | Δ | Equal |
|---|---|---|---|---|---|
| credit-g | roc_auc | 0.7913 | 0.7913 | 0.00e+00 | **yes** |
| diabetes | roc_auc | 0.8150 | 0.8150 | 0.00e+00 | **yes** |
| vehicle | f1_macro | 0.8326 | 0.8326 | 0.00e+00 | **yes** |
| cpu_small | r2 | 0.9750 | 0.9750 | 1.17e-07 | **NO** |
| kin8nm | r2 | 0.8182 | 0.8182 | 0.00e+00 | **yes** |

**4/5 datasets bit-identical; the remainder differ by ≤ 1.2e-07** — below the 1e-06 float-noise line (thread-level reduction order in parallel learners), not a modelling difference. Quality is provider-independent; bit-level reproducibility of the parallel learners is a Phase-4 (Day 23) item.

### Blocked — what the live comparison still needs

- **groq** — probe result: `BadRequestError: Error code: 400 - {'error': {'message': 'Organization has been restricted. Please reach out to support if you believe this was in error.', 'type': 'invalid_request_error', 'code': 'organization_restricted'}}`. Unblock: GROQ_API_KEY (org must not be restricted).
- **anthropic** — probe result: `not_configured`. Unblock: ANTHROPIC_API_KEY.

*When either provider comes back, `python scripts/run_provider_study.py` re-runs the probes, adds the live arm(s), prices them and rewrites this board — nothing else to change.*

*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).

