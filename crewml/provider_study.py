"""Day 16 — the provider study: what does the LLM provider change, and at what price?

The crew's LLM surface is deliberately thin (:mod:`crewml.llm`): every narrative and
code-generation call goes through one ``chat()`` with per-call token accounting, and every
node has a deterministic offline path. That design makes the provider a *swappable arm* of
a study rather than a rewrite: run the identical crew under each provider and compare
quality (LOCKED-holdout score), cost (tokens × published price) and latency (probe
round-trip + crew wall-clock).

The three arms:

* **groq** — Llama 3.3 70B via Groq, the project default.
* **anthropic** — Claude (``config.ANTHROPIC_MODEL``), the optional premium arm.
* **mock** — the deterministic offline core; zero tokens, zero dollars, always available.

Session reality (2026-07-21): the Groq organisation has been RESTRICTED since at least the
Day-14 runs (every archived ``llm_usage`` narrative is ``unavailable`` with
``organization_restricted``; no live LLM call exists anywhere in the artifact history), and
no Anthropic key is configured. The study is honest about that instead of working around
it: each provider is **probed live** and the probe outcome is committed evidence; fresh
crew arms run only for providers whose probe passes (plus the mock arm, which needs no
provider); the cost model is wired and tested so the moment a live arm is possible the
same runner prices it with no code change.

What *is* measurable today, and is a real finding rather than a consolation prize:
**provider-outage resilience**. The Day-14 archival runs executed with a configured-but-
failing provider (every LLM call errored mid-run); a fresh mock arm runs with no provider
at all. If the two produce identical holdout scores, the crew's modelling quality is
provably independent of provider availability — an outage costs narrative richness and
LLM-generated FE candidates, never the score. The study measures that equality per
dataset rather than asserting it.

Honesty rules, unchanged: the holdout is scored outside the graph with the seal re-verified
per run; mock-arm numbers are flagged ``mock`` and never presented as live-provider results
(EVAL_PROTOCOL.md §5); a cost is only computed from *measured* tokens, never estimated ones.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import date
from typing import Any, Iterable, Optional

from crewml import config, llm
from crewml.ablation import run_variant, LOOPED
from crewml.config import RESULTS_DIR
from crewml.datasets import REGISTRY, load_manifest

PROVIDER_STUDY_SCHEMA_VERSION = 1
PROVIDER_STUDY_RESULT_PATH = RESULTS_DIR / "day16_provider_study.json"
PROVIDER_STUDY_TABLE_MD_PATH = RESULTS_DIR / "day16_provider_study.md"

ARTIFACT_PREFIX = "day16"

# Two scores are "the same run" when they differ by less than this — the pipeline is
# seed-locked end to end, so genuine equality is exact and the epsilon only absorbs
# float round-tripping through JSON.
EQUALITY_EPS = 1e-9

# Below this, a score difference is float noise (thread-level reduction order in
# parallel learners such as HistGradientBoosting), not a modelling difference. The
# equality check stays strict; only the *verdict wording* uses this tier, so a 1e-7
# wobble is reported as what it is without being promoted to "identical".
FLOAT_NOISE_EPS = 1e-6

# --- The provider registry ----------------------------------------------------
#
# Published on-demand prices per 1M tokens, hard-coded with their source and date so the
# committed board is auditable. Prices are for the *listed* model; if the configured
# model ever diverges from the priced one the board must say so (``priced_model`` is
# carried on every cost row for exactly that check).
PRICING_AS_OF = "2026-07-21"

PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "label": "Groq — Llama 3.3 70B",
        "model": "llama-3.3-70b-versatile",
        "usd_per_mtok_in": 0.59,
        "usd_per_mtok_out": 0.79,
        "pricing_note": "Groq published on-demand pricing",
        "needs": "GROQ_API_KEY (org must not be restricted)",
    },
    "anthropic": {
        "label": "Anthropic — Claude Sonnet 5",
        "model": "claude-sonnet-5",
        # Introductory pricing in effect through 2026-08-31 (sticker: $3.00 / $15.00).
        "usd_per_mtok_in": 2.00,
        "usd_per_mtok_out": 10.00,
        "pricing_note": "Anthropic introductory pricing through 2026-08-31 (sticker 3.00/15.00)",
        "needs": "ANTHROPIC_API_KEY",
    },
    "mock": {
        "label": "Mock — deterministic offline core",
        "model": None,
        "usd_per_mtok_in": 0.0,
        "usd_per_mtok_out": 0.0,
        "pricing_note": "no LLM calls; always available",
        "needs": "nothing",
    },
}


def cost_usd(prompt_tokens: Optional[int], completion_tokens: Optional[int], provider: str) -> Optional[float]:
    """Price a measured token count under one provider's published rates (pure).

    Returns ``None`` unless *both* counts are real numbers — a cost is only ever
    computed from measured tokens, never from a guess or a half-measurement.
    """
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    p = PROVIDERS[provider]
    return round(
        prompt_tokens / 1e6 * p["usd_per_mtok_in"]
        + completion_tokens / 1e6 * p["usd_per_mtok_out"],
        6,
    )


# --- Forcing a provider per run ----------------------------------------------

@contextmanager
def forced_provider(name: str):
    """Temporarily force ``config.LLM_PROVIDER`` (module attribute + env), restoring after.

    ``config`` resolves the provider once at import, so the env var alone would not reach
    an already-imported process — the module attribute is the live switch and the env var
    keeps any subprocess consistent. Scoped exactly like :func:`crewml.ablation._handicap`
    so one arm can never leak its provider into the next.
    """
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}")
    prev_env = os.environ.get("CREWML_LLM_PROVIDER")
    prev_attr = config.LLM_PROVIDER
    os.environ["CREWML_LLM_PROVIDER"] = name
    config.LLM_PROVIDER = name
    try:
        yield
    finally:
        config.LLM_PROVIDER = prev_attr
        if prev_env is None:
            os.environ.pop("CREWML_LLM_PROVIDER", None)
        else:
            os.environ["CREWML_LLM_PROVIDER"] = prev_env


# --- Live availability probes -------------------------------------------------

PROBE_SYSTEM = "You are a connectivity probe. Reply with exactly one word."
PROBE_USER = "Reply with exactly: PONG"

def probe_provider(name: str) -> dict[str, Any]:
    """One cheap live round-trip against a provider; the outcome is committed evidence.

    Statuses:

    * ``ok`` — the call returned; latency and token counts are measured, the echo is
      recorded (a live provider that can't follow a one-word instruction is worth seeing).
    * ``offline`` — the mock arm: structurally available, nothing to probe.
    * ``not_configured`` — no key for this provider (``is_mock_mode()`` under the forced
      provider); no network call is attempted.
    * ``error`` — the call raised; the exception text is the evidence (this is where
      Groq's ``organization_restricted`` lands).
    """
    rec: dict[str, Any] = {
        "provider": name,
        "label": PROVIDERS[name]["label"],
        "model": None,
        "status": None,
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
        "reply": None,
        "error": None,
        "checked": date.today().isoformat(),
    }
    if name == "mock":
        rec["status"] = "offline"
        return rec
    with forced_provider(name):
        rec["model"] = config.GROQ_MODEL if name == "groq" else config.ANTHROPIC_MODEL
        if config.is_mock_mode():
            rec["status"] = "not_configured"
            return rec
        started = time.perf_counter()
        try:
            # 256, not 8: a reasoning model spends completion budget on
            # reasoning before it emits content, so a tiny cap returns a
            # blank reply and the board records reachability without
            # evidence of it. Still a trivial probe cost.
            result = llm.chat(PROBE_SYSTEM, PROBE_USER, max_tokens=256)
        except Exception as exc:  # the failure text IS the finding — keep it verbatim
            rec["status"] = "error"
            rec["latency_s"] = round(time.perf_counter() - started, 3)
            rec["error"] = f"{type(exc).__name__}: {exc}"
            return rec
        rec["status"] = "ok"
        rec["latency_s"] = round(time.perf_counter() - started, 3)
        rec["reply"] = result.text.strip()[:80]
        rec["prompt_tokens"] = result.prompt_tokens
        rec["completion_tokens"] = result.completion_tokens
        rec["cost_usd"] = cost_usd(result.prompt_tokens, result.completion_tokens, name)
        return rec


def probe_all(progress=None) -> dict[str, dict[str, Any]]:
    """Probe every registered provider; returns {name: probe record}."""
    out = {}
    for name in PROVIDERS:
        if progress:
            progress(f"[day16] probing provider: {name} ...")
        out[name] = probe_provider(name)
        if progress:
            progress("  " + summarise_probe(out[name]))
    return out


def summarise_probe(rec: dict[str, Any]) -> str:
    if rec["status"] == "ok":
        return (f"{rec['provider']}: OK {rec['latency_s']}s "
                f"({rec['prompt_tokens']}+{rec['completion_tokens']} tok, reply={rec['reply']!r})")
    if rec["status"] == "error":
        return f"{rec['provider']}: ERROR after {rec['latency_s']}s — {rec['error']}"
    return f"{rec['provider']}: {rec['status']}"


# --- Crew arms ----------------------------------------------------------------

def run_provider_arm(
    provider: str,
    keys: Iterable[str],
    manifest: Optional[dict] = None,
    *,
    progress=None,
) -> dict[str, dict[str, Any]]:
    """Run the full crew on each dataset under one forced provider; return {key: record}.

    Every run goes through :func:`crewml.ablation.run_variant` — same recursion budget,
    same post-run seal check, same outside-the-graph holdout scoring as Days 12-15 — so
    a provider-arm number means what every other scored number in the project means.
    Each record is annotated with the provider and its priced cost (``None`` when the
    run made no live calls — a mock run costs nothing *and prices nothing*).
    """
    manifest = manifest or load_manifest()
    out: dict[str, dict[str, Any]] = {}
    with forced_provider(provider):
        for key in keys:
            if progress:
                progress(f"[day16] arm={provider} === {key} ===")
            rec = run_variant(
                key, LOOPED, manifest,
                artifact_prefix=ARTIFACT_PREFIX,
                artifact_tag=provider,
            )
            rec["provider"] = provider
            rec["llm_cost_usd"] = cost_usd(
                rec.get("llm_prompt_tokens"), rec.get("llm_completion_tokens"), provider
            ) if (rec.get("llm_narratives_live") or 0) > 0 else None
            out[key] = rec
            if progress:
                v = f"{rec['value']:.4f}" if rec.get("ok") else f"FAILED({rec.get('error')})"
                progress(f"  {key}: {rec.get('metric')}={v} ({rec.get('crew_seconds')}s, "
                         f"live_narratives={rec.get('llm_narratives_live')})")
    return out


# --- Outage-resilience equality check ----------------------------------------

def load_day14_archival() -> dict[str, dict[str, Any]]:
    """The Day-14 ``full``-arm records: the crew with a configured-but-failing provider.

    Those runs had a Groq key present, so every node *attempted* its live call and every
    call errored (``organization_restricted``) mid-run — the harshest realistic outage
    shape. Returns {} if the Day-14 board is absent; the equality check then reports
    itself not-computable rather than inventing a baseline.
    """
    path = RESULTS_DIR / "day14_agent_ablation.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for key, row in (data.get("results") or {}).items():
        rec = (row.get("arms") or {}).get("full")
        if rec:
            out[key] = rec
    return out


def equality_check(
    fresh_mock: dict[str, dict[str, Any]],
    archival: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare fresh no-provider scores against archival failing-provider scores (pure).

    Equality (within :data:`EQUALITY_EPS`) on a seed-locked pipeline means the modelling
    path is provably provider-independent: an outage mid-run and no provider at all land
    on the same model and the same holdout score. Any inequality is surfaced per dataset
    with the actual diff — never averaged away.
    """
    rows = []
    for key, mock_rec in fresh_mock.items():
        arch = archival.get(key)
        mv = mock_rec.get("value") if mock_rec.get("ok") else None
        av = arch.get("value") if (arch and arch.get("ok")) else None
        diff = round(abs(mv - av), 12) if (mv is not None and av is not None) else None
        rows.append({
            "dataset": key,
            "metric": mock_rec.get("metric"),
            "fresh_mock_value": mv,
            "archival_failing_provider_value": av,
            "abs_diff": diff,
            "equal": (diff is not None and diff <= EQUALITY_EPS),
        })
    comparable = [r for r in rows if r["abs_diff"] is not None]
    return {
        "rows": rows,
        "n_compared": len(comparable),
        "n_equal": sum(1 for r in comparable if r["equal"]),
        "all_equal": bool(comparable) and all(r["equal"] for r in comparable),
        "max_abs_diff": max((r["abs_diff"] for r in comparable), default=None),
        # Strictly-unequal pairs can still sit below FLOAT_NOISE_EPS — that is float
        # noise from parallel learners, not a modelling difference, and the verdict
        # wording distinguishes the two rather than crying wolf.
        "all_within_noise": bool(comparable)
        and all(r["abs_diff"] <= FLOAT_NOISE_EPS for r in comparable),
    }


# --- Assembly -----------------------------------------------------------------

def assemble_report(
    probes: dict[str, dict[str, Any]],
    arms: dict[str, dict[str, dict[str, Any]]],
    resilience: dict[str, Any],
) -> dict[str, Any]:
    """Bundle probes + arms + the resilience check into the committed object."""
    any_mock = any(
        rec.get("mock") for arm in arms.values() for rec in arm.values()
    )
    live_arms = [p for p, probe in probes.items() if probe["status"] == "ok"]
    blocked = {
        p: (probes[p].get("error") or probes[p]["status"])
        for p in probes
        if probes[p]["status"] in ("error", "not_configured")
    }
    return {
        "schema_version": PROVIDER_STUDY_SCHEMA_VERSION,
        "day": 16,
        "phase": 3,
        "study": "provider_study",
        "any_mock": any_mock,
        "pricing_as_of": PRICING_AS_OF,
        "providers": {
            name: {k: v for k, v in spec.items()}
            for name, spec in PROVIDERS.items()
        },
        "probes": probes,
        "live_arms_run": live_arms,
        "blocked_providers": blocked,
        "arms": arms,
        "resilience": resilience,
    }


# --- Rendering ----------------------------------------------------------------

def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def render_markdown(report: dict) -> str:
    """Render the Day-16 provider board as committed markdown."""
    probes = report["probes"]
    lines = [
        "# Day 16 — provider study: Groq vs Claude vs mock",
        "",
        "*The identical crew under each LLM provider, compared on quality (LOCKED-holdout, "
        "scored outside the graph, seal re-verified per run), cost (measured tokens × "
        "published price) and latency. Providers are probed live before any arm runs; a "
        "provider that fails its probe contributes its failure as evidence, never an "
        "imagined number.*",
        "",
        "### Provider availability (live probes, "
        f"{next(iter(probes.values()))['checked']})",
        "",
        "| Provider | Model | Status | Probe latency | Evidence |",
        "|---|---|---|---|---|",
    ]
    for name, p in probes.items():
        status = {
            "ok": "**OK**", "offline": "always available (offline)",
            "not_configured": "NOT CONFIGURED", "error": "**UNAVAILABLE**",
        }[p["status"]]
        evidence = (
            f"reply `{p['reply']}`" if p["status"] == "ok"
            else f"`{p['error']}`" if p["status"] == "error"
            else f"needs {report['providers'][name]['needs']}" if p["status"] == "not_configured"
            else "deterministic core; no LLM calls"
        )
        latency = f"{p['latency_s']:.2f}s" if isinstance(p["latency_s"], (int, float)) else "—"
        lines.append(
            f"| {p['label']} | {p['model'] or '—'} | {status} | {latency} | {evidence} |"
        )
    lines += [
        "",
        "### Cost model (published on-demand prices, as of "
        f"{report['pricing_as_of']})",
        "",
        "| Provider | Priced model | $ / 1M input | $ / 1M output | Source |",
        "|---|---|---|---|---|",
    ]
    for name, spec in report["providers"].items():
        lines.append(
            f"| {spec['label']} | {spec['model'] or '—'} | "
            f"{spec['usd_per_mtok_in']:.2f} | {spec['usd_per_mtok_out']:.2f} | "
            f"{spec['pricing_note']} |"
        )
    any_live_cost = any(
        rec.get("llm_cost_usd") is not None
        for arm in report["arms"].values() for rec in arm.values()
    )
    cost_note = (
        "*A run's cost is only ever computed from tokens the accounting actually measured "
        "(`llm_usage`, live calls only). "
        + ("Live-arm costs below are measured-token totals priced at the rates above."
           if any_live_cost else
           "No live arm ran this session, so no cost is reported — the model above is "
           "wired into the runner and prices any future live arm with no code change.")
        + "*"
    )
    lines += ["", cost_note, ""]

    # Per-arm quality/latency tables.
    for provider, arm in report["arms"].items():
        label = report["providers"][provider]["label"]
        mock_flag = " *(mock — deterministic core, no LLM; never a headline result)*" if any(
            rec.get("mock") for rec in arm.values()
        ) else ""
        lines += [
            f"### Arm — {label}{mock_flag}",
            "",
            "| Dataset | Metric | Holdout score | Crew seconds | Live LLM calls | Tokens | Cost |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, rec in arm.items():
            toks = (
                (rec.get("llm_prompt_tokens") or 0) + (rec.get("llm_completion_tokens") or 0)
                if (rec.get("llm_narratives_live") or 0) > 0 else None
            )
            lines.append(
                f"| {key} | {rec.get('metric')} | {_fmt(rec.get('value') if rec.get('ok') else None)} | "
                f"{_fmt(rec.get('crew_seconds'), '.1f')} | {rec.get('llm_narratives_live') or 0} | "
                f"{toks if toks is not None else '—'} | "
                f"{('$' + format(rec['llm_cost_usd'], '.4f')) if rec.get('llm_cost_usd') is not None else '—'} |"
            )
        # A failed run is evidence, not a blank cell — say what happened, in the
        # study's own words (the scoring error + the Critic's final verdict).
        for key, rec in arm.items():
            if not rec.get("ok"):
                codes = ", ".join(rec.get("final_finding_codes") or []) or "none recorded"
                lines += [
                    "",
                    f"*`{key}` shipped no scorable model: {rec.get('error')} — Critic verdict: "
                    f"\"{rec.get('final_reason')}\" (finding codes: {codes}). The full failing "
                    "state is archived; the failure stands on the board as a result in itself.*",
                ]
        lines.append("")

    # Resilience equality.
    res = report["resilience"]
    lines += [
        "### Provider-outage resilience — fresh no-provider vs archival failing-provider",
        "",
        "*The Day-14 archival runs executed with a Groq key configured and every live call "
        "failing mid-run (`organization_restricted`) — the harshest realistic outage. "
        "Today's mock arm ran with no provider at all. On a seed-locked pipeline, equal "
        "scores prove the modelling path is provider-independent: an outage costs "
        "narrative richness, never the score.*",
        "",
        "| Dataset | Metric | Fresh (no provider) | Archival (provider failing) | Δ | Equal |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(
            f"| {r['dataset']} | {r['metric']} | {_fmt(r['fresh_mock_value'])} | "
            f"{_fmt(r['archival_failing_provider_value'])} | "
            f"{_fmt(r['abs_diff'], '.2e')} | {'**yes**' if r['equal'] else ('**NO**' if r['abs_diff'] is not None else '—')} |"
        )
    if res["all_equal"]:
        verdict = (f"**{res['n_equal']}/{res['n_compared']} datasets bit-identical** — the "
                   "crew's holdout quality is provably independent of provider availability.")
    elif res.get("all_within_noise"):
        verdict = (f"**{res['n_equal']}/{res['n_compared']} datasets bit-identical; the "
                   f"remainder differ by ≤ {res['max_abs_diff']:.1e}** — below the "
                   f"{FLOAT_NOISE_EPS:.0e} float-noise line (thread-level reduction order "
                   "in parallel learners), not a modelling difference. Quality is "
                   "provider-independent; bit-level reproducibility of the parallel "
                   "learners is a Phase-4 (Day 23) item.")
    elif res["n_compared"]:
        verdict = (f"**{res['n_equal']}/{res['n_compared']} datasets identical — the runs "
                   "are NOT equivalent; investigate before claiming resilience.**")
    else:
        verdict = "*(not computable — no archival baseline found)*"
    lines += ["", verdict, ""]

    if report["blocked_providers"]:
        lines += [
            "### Blocked — what the live comparison still needs",
            "",
        ]
        for p, why in report["blocked_providers"].items():
            needs = report["providers"][p]["needs"]
            lines.append(f"- **{p}** — probe result: `{why}`. Unblock: {needs}.")
        lines += [
            "",
            "*When either provider comes back, `python scripts/run_provider_study.py` "
            "re-runs the probes, adds the live arm(s), prices them and rewrites this "
            "board — nothing else to change.*",
            "",
        ]
    if report["any_mock"]:
        lines += ["*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).", ""]
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> dict:
    """Persist the report as JSON + markdown; return it unchanged."""
    PROVIDER_STUDY_RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    PROVIDER_STUDY_TABLE_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report
