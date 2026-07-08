# Day 3 — Phase 1 (Foundation & Baselines)

**Baseline 1: the solo agent — one LLM, one shot, one sklearn script.**

> ⚠️ **This run is MOCK.** No LLM key is configured, so the solo agent used a
> fixed offline single-shot script instead of a live model. Per
> [EVAL_PROTOCOL.md §5](../docs/EVAL_PROTOCOL.md), these numbers are labelled mock
> and are **not** the headline. They establish the harness and the honesty
> boundary; the real solo number lands once a Groq key is set and the same driver
> is re-run.

## What shipped

- **`crewml/llm.py`** — a thin provider abstraction. Every LLM call routes through
  `chat(system, user)` and returns an `LLMResult` carrying the text **and** token
  accounting (for run budgets and the Day 16 provider study). Groq (Llama 3.3 70B)
  is the default; Anthropic is wired as the optional study provider. In mock mode
  `chat` raises `MockModeError` — callers must branch on `is_mock_mode()` and take
  a deterministic offline path, so no network is ever required to run. Includes
  `extract_python()` to pull the fenced code block out of a reply.
- **`crewml/solo_agent.py`** — the solo baseline end to end:
  - `build_profile_summary()` — a compact, **train-only** description (shapes,
    dtypes, missingness, target distribution) handed to the agent. It mirrors what
    the Phase-2 Profiler will surface, so solo and crew start from equal knowledge
    (EVAL_PROTOCOL §4).
  - Prompts instructing the agent to emit exactly one `solve(train_df)` module
    returning a **fitted** sklearn estimator (Pipeline preferred), with all
    preprocessing baked in, `predict` (+ `predict_proba` for binary), seed 42, no
    file/network access.
  - `mock_solo_script()` — the offline single-shot solution: leakage-safe
    impute + one-hot preprocessing feeding an untuned `HistGradientBoosting`
    model. Deterministic.
  - **The honesty boundary.** The generated `solve` only ever receives `train`. It
    is executed in a **subprocess** by a *trusted* runner we write
    (`RUNNER_TEMPLATE`) — not by the agent — which fits on `train` and calls
    `predict` on held-out **features only** (labels never enter that process, and
    it never fits on the holdout). Scoring against the held-out labels happens back
    in the trusted parent via `crewml.scoring`, and the SHA-256 seal is re-verified
    after (`verify_holdout_untouched`).
- **`scripts/run_solo_agent.py`** — the Day 3 driver → `results/solo_agent_metrics.json`.
  Records seed, sklearn version, provider, the `mock` flag, per-dataset scores, and
  a `failures` map (no silent drops).
- **`tests/test_solo_agent.py`** — 23 new tests: LLM code-extraction + the
  mock-mode refusal contract; profile-is-train-only; the mock `solve` module
  compiles and honours the estimator contract (proba+classes for binary, finite
  numerics for regression); and integration checks that the metrics file is
  complete, finite, mock-labelled, clears the Dummy floor, and leaves every
  holdout seal intact.

## Held-out results (MOCK solo vs Day 2 baselines)

Primary metric per dataset; higher is better. `Δ` = solo − default RF.

| Dataset    | Metric   | Dummy (floor) | default RF | **Solo (mock)** | Δ vs RF |
|------------|----------|--------------:|-----------:|----------------:|--------:|
| credit-g   | ROC AUC  |        0.5000 |     0.7783 |      **0.7521** |  −0.0262 |
| diabetes   | ROC AUC  |        0.5000 |     0.8118 |      **0.7987** |  −0.0131 |
| vehicle    | macro-F1 |        0.1028 |     0.7260 |      **0.7763** |  +0.0502 |
| cpu_small  | R²       |       −0.0029 |     0.9726 |      **0.9747** |  +0.0021 |
| kin8nm     | R²       |       −0.0002 |     0.6948 |      **0.8120** |  +0.1172 |

**Reading it.** The mock solo agent clears the Dummy floor everywhere (as any
working system must) and trades blows with the untuned RandomForest: it wins
clearly where a gradient booster's smooth fit helps (kin8nm +0.117, vehicle
+0.050) and trails slightly on the two small binary sets where the extra
flexibility doesn't pay (credit-g, diabetes). That mixed picture is exactly what a
naive single-shot attempt should look like — a competent-but-uncoached solver, no
imbalance handling on credit-g, no disguised-missing detection on diabetes. **This
is the bar the crew's Critic loop has to raise**, and the two binary datasets are
where a real crew should earn its keep.

## Honesty / protocol adherence

- The agent (prompt + generated code) saw **only** `train`; the holdout entered
  only the trusted runner's `predict` call and the parent's scorer.
- Every one of the 5 holdout seals re-verified intact after scoring.
- Mock numbers are stamped `"mock": true` per dataset and at the run level; the
  report banner and `note` field both flag it.
- All 5 datasets scored; `failures = {}`.

## Verification

`python -m pytest tests/` → **59 passed** (36 prior + 23 new).

**Next:** Day 4 — Baseline 2: classical AutoML (FLAML) as the strong non-agent
ceiling; assemble the full baselines table; write the Phase 1 Wrap-Up and merge
the Phase 1 PR.
