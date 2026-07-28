"""Day 21 — the budget study: measure the run-budget layer doing its job, live.

Day 21 made ``CREWML_RUN_TOKEN_BUDGET`` real (see :mod:`crewml.budget`): every LLM
call is gated pre-call and charged post-call, the Critic finalises when the budget
is spent or unaffordable, and an exhausted budget degrades a run to its
deterministic core instead of crashing it. This study runs the full crew on one
dataset under three budget regimes and records what the ledger actually did:

* ``reference`` — config-default caps. The expected shape: zero refusals; the
  ledger is simply the honest itemised cost (tokens, calls, per-agent) of one
  production crew run.
* ``tight_tokens`` — a token cap far below a run's appetite. Expected: the first
  call(s) are granted, the rest are refused, refused narratives degrade to
  ``unavailable``, and the run still completes with a model.
* ``tight_time`` — a wall-clock cap below the run's duration. Expected: calls
  after the cap are refused and the Critic sees an exhausted budget at its pass
  boundary.

Honesty rules, same as every measured surface in the project:

* **Provider liveness is probed first** (an 8-token real call) — a configured key
  can be revoked or quota-blocked and look exactly like an outage (the Day-20
  lesson), so "live" is asserted by observation, never by ``is_mock_mode()``.
* **A budget-starved run is not a mock run.** Zero live narratives *because calls
  were refused* is the feature working; the record keeps ``n_refused`` next to
  ``llm_narratives_live`` so the two cases can never be conflated.
* **Expected shapes are not asserted shapes.** The record states what happened;
  if a tight cap never actually bound (e.g. the run finished under the wire), the
  study says so rather than dressing the scenario up as a demonstration.
* **The holdout stays sealed** — verified after every scenario; a broken seal
  raises rather than reports.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from crewml import budget as budget_mod
from crewml import config, llm
from crewml.config import RESULTS_DIR
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY, verify_holdout_untouched

BUDGET_STUDY_SCHEMA_VERSION = 1
BUDGET_STUDY_PATH = RESULTS_DIR / "day21_budget_study.json"
BUDGET_STUDY_MD_PATH = RESULTS_DIR / "day21_budget_study.md"

DATASET = "credit-g"

# (name, token_budget, time_budget_s) — None = the config default for that cap.
# The tight caps are calibrated FROM the reference run's measured appetite (a first
# pass of this study observed ~2,057 tokens / ~28s for a full live credit-g run;
# caps guessed above that — 2,500 tok / 20s-with-calls-landing-early — bound
# nothing and demonstrated nothing). 1,200 tokens grants the first call(s) and
# refuses the rest; 10s expires while the crew is still consulting the provider.
SCENARIOS: tuple[tuple[str, Optional[int], Optional[float]], ...] = (
    ("reference", None, None),
    ("tight_tokens", 1200, None),
    ("tight_time", None, 10.0),
)


def probe_provider() -> dict[str, Any]:
    """One real 8-token call to prove the provider is alive, not merely configured.

    ``is_mock_mode()`` only checks that a key is *present*; a present-but-revoked or
    quota-exhausted key is the dangerous case because it looks live. Runs outside
    any budget so the probe itself is never refused.
    """
    rec: dict[str, Any] = {
        "provider": config.LLM_PROVIDER,
        "model": config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.ANTHROPIC_MODEL,
        "key_present": not config.is_mock_mode(),
    }
    if config.is_mock_mode():
        rec.update(live=False, error="no key configured (mock mode)")
        return rec
    started = time.perf_counter()
    try:
        result = llm.chat(
            "You are a liveness probe for a measurement harness.",
            "Reply with the single word OK.",
            temperature=0.0, max_tokens=8, agent="probe",
        )
    except Exception as exc:  # the failure text IS the finding — keep it verbatim
        rec.update(live=False, latency_s=round(time.perf_counter() - started, 3),
                   error=f"{type(exc).__name__}: {exc}")
        return rec
    rec.update(live=True, latency_s=round(time.perf_counter() - started, 3),
               probe_tokens=result.total_tokens, error=None)
    return rec


def run_scenario(
    name: str,
    token_budget: Optional[int],
    time_budget_s: Optional[float],
    *,
    max_iterations: int = config.MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run the full crew on ``DATASET`` under one budget regime; distil the record."""
    spec = REGISTRY[DATASET]
    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10

    started = time.time()
    with budget_mod.run_budget(token_budget, time_budget_s):
        final = app.invoke(state, config={"recursion_limit": limit})
    crew_seconds = round(time.time() - started, 2)

    if not verify_holdout_untouched(DATASET):
        raise RuntimeError(f"{name}: holdout seal broken DURING the crew run — aborting")

    report = final.get("report") or {}
    ledger = report.get("run_budget") or {}
    llm_usage = report.get("llm_usage") or {}
    critiques = final.get("critiques") or []
    last = critiques[-1] if critiques else {}
    final_reason = last.get("reason") or ""
    final_model = report.get("final_model") or {}
    budget_warning = next(
        (w for w in (report.get("warnings") or []) if "BUDGET-CONSTRAINED" in w), None,
    )

    return {
        "scenario": name,
        # The caps as the ledger resolved them (config defaults filled in, <=0 -> None).
        "token_budget": ledger.get("token_budget"),
        "time_budget_s": ledger.get("time_budget_s"),
        "crew_seconds": crew_seconds,
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "final_decision": last.get("decision"),
        "final_reason": last.get("reason"),
        # Did the budget actually bind anywhere? (refused a call, or stopped the loop)
        "budget_stopped_loop": bool(last.get("decision") == "finalize"
                                    and "run budget" in final_reason),
        "n_refused": ledger.get("n_refused"),
        "budget_bound": bool((ledger.get("n_refused") or 0) > 0
                             or (last.get("decision") == "finalize"
                                 and "run budget" in final_reason)),
        "tokens_spent": ledger.get("tokens_spent"),
        "n_calls": ledger.get("n_calls"),
        "per_agent": ledger.get("per_agent"),
        "elapsed_s_ledger": ledger.get("elapsed_s"),
        "stop_reason": ledger.get("stop_reason"),
        "llm_narratives_live": llm_usage.get("n_live"),
        "llm_narratives_requested": llm_usage.get("n_requested"),
        "final_model_kind": final_model.get("kind"),
        "cv_score": final_model.get("cv_score"),
        "metric": report.get("metric"),
        "cv_score_is_holdout": False,
        "budget_warning": budget_warning,
        "holdout_untouched": True,  # verified above; a broken seal raised instead
    }


def run_study(*, max_iterations: int = config.MAX_ITERATIONS) -> dict[str, Any]:
    """Probe the provider, run all three scenarios, and return the committed record."""
    probe = probe_provider()
    scenarios = [
        run_scenario(name, tok, sec, max_iterations=max_iterations)
        for name, tok, sec in SCENARIOS
    ]
    notes = [
        "Scores are CV estimates on train (cv_score_is_holdout: false); the sealed "
        "holdout was verified untouched after every scenario.",
        "A budget-starved run is not a mock run: zero live narratives beside a "
        "non-zero n_refused is the enforcement working, and is labelled as such.",
        "Grid search was disabled (CREWML_TRAINER_PARAM_SEARCH=0) for every scenario "
        "so wall-clock differences come from LLM calls and CV, not the search grid.",
    ]
    if not probe.get("live"):
        notes.insert(0, (
            "PROVIDER NOT LIVE for this study run — token-cost numbers are not "
            "measurable and the scenarios exercise only the deterministic core and "
            "the wall-clock cap. No number below is presented as a live-provider cost."
        ))
    return {
        "schema_version": BUDGET_STUDY_SCHEMA_VERSION,
        "day": 21,
        "dataset": DATASET,
        "max_iterations": max_iterations,
        "config_defaults": {
            "token_budget": config.RUN_TOKEN_BUDGET,
            "time_budget_s": config.RUN_TIME_BUDGET_S,
        },
        "probe": probe,
        "scenarios": scenarios,
        "notes": notes,
    }


# --- The committed table (registered in crewml.artifact_registry) -------------

def _fmt_cap(tokens: Optional[int], seconds: Optional[float]) -> str:
    tok = f"{tokens:,} tok" if tokens else "∞ tok"
    sec = f"{seconds:.0f}s" if seconds else "∞s"
    return f"{tok} / {sec}"


def _fmt_score(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def render_markdown(data: dict[str, Any]) -> str:
    """Render the committed study table from the committed JSON (pure, deterministic)."""
    probe = data.get("probe") or {}
    live = bool(probe.get("live"))
    provider_line = (
        f"live **{probe.get('provider')}** ({probe.get('model')}), probe "
        f"{probe.get('latency_s')}s" if live else
        f"**NOT LIVE** — {probe.get('error')}"
    )

    rows = []
    for s in data.get("scenarios") or []:
        refused = s.get("n_refused") or 0
        bound = ("loop-stop" if s.get("budget_stopped_loop")
                 else ("refusals" if refused else "never bound"))
        rows.append(
            f"| {s['scenario']} | {_fmt_cap(s.get('token_budget'), s.get('time_budget_s'))} "
            f"| {s.get('tokens_spent', 0) or 0:,} | {s.get('n_calls', 0) or 0} live / {refused} refused "
            f"| {s.get('crew_seconds')}s | {s.get('iterations_run')}/{s.get('max_iterations')} "
            f"| {s.get('final_decision')} | {bound} | {_fmt_score(s.get('cv_score'))} |"
        )
    table = "\n".join(rows) or "| — | — | — | — | — | — | — | — | — |"

    notes = "\n".join(f"- {n}" for n in (data.get("notes") or []))

    return f"""# Day 21 — Run-budget study: cost & latency caps, enforced and measured

The full crew ran on `{data.get('dataset')}` under three budget regimes
(provider: {provider_line}). The run budget (Day 21, `crewml.budget`) gates every
LLM call pre-call, charges it post-call, and lets the Critic finalise on an
exhausted or unaffordable budget — so a spent budget degrades a run to its
deterministic core; it never crashes it.

| scenario | caps (tok/time) | tokens spent | LLM calls | wall-clock | passes | final decision | budget bound? | CV score |
|----------|-----------------|-------------:|-----------|-----------:|:------:|----------------|---------------|:--------:|
{table}

**Reading the table.** "Budget bound?" reports whether enforcement actually fired:
`refusals` = calls turned away by the gate, `loop-stop` = the Critic finalised on
budget grounds, `never bound` = the run finished inside its caps (the expected
shape for `reference`). CV scores are estimates on train, never holdout.

## Notes

{notes}
"""
