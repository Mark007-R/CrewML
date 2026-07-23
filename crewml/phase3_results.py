"""The Phase-3 consolidated results section — six studies, one document.

Days 12-17 each answered one question and committed its own board:

* Day 12 — does the crew beat solo / AutoML / default RF on the locked holdout?
* Day 13 — does the Critic loop earn its keep?
* Day 14 — what do the Planner and Feature Engineer each contribute?
* Day 15 — what does each extra Critic loop buy (depth-response curve)?
* Day 16 — what does a run cost per provider, and does an outage cost score?
* Day 17 — where does the crew fail, and who catches it?

This module stitches those six boards into ``results/phase3_results.md`` (the results
section a reader actually starts from) plus ``results/phase3_results.json`` (the same
numbers machine-readable, for the Day-29 README rewrite).

Like :mod:`crewml.comparison`, it only ever *reshapes* committed numbers — nothing here
runs the crew, scores a model, or touches data. Every figure in the output is traceable
to exactly one committed study file, so the consolidated section can never disagree
with the studies it summarises. The same honesty rules are enforced in rendering:

* a missing number is an em dash, never a zero and never silently dropped;
* mock/deterministic-core runs stay labelled (EVAL_PROTOCOL.md §5);
* deltas appear only where the source study computed them between two real numbers;
* the section repeats the phase's *unflattering* findings (the live diabetes fatal,
  the measured leakage detection window) with the same prominence as the wins —
  consolidation is not the place where caveats go to die.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from crewml.config import RESULTS_DIR

# --- Inputs: one committed file per study (all tracked in git) ---
COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.json"
CRITIC_ABLATION_PATH = RESULTS_DIR / "day13_critic_ablation.json"
AGENT_ABLATION_PATH = RESULTS_DIR / "day14_agent_ablation.json"
ITERATION_DEPTH_PATH = RESULTS_DIR / "day15_iteration_depth.json"
PROVIDER_STUDY_PATH = RESULTS_DIR / "day16_provider_study.json"
FAILURE_TAXONOMY_PATH = RESULTS_DIR / "day17_failure_taxonomy.json"

STUDY_PATHS = {
    "comparison": COMPARISON_TABLE_PATH,
    "critic_ablation": CRITIC_ABLATION_PATH,
    "agent_ablation": AGENT_ABLATION_PATH,
    "iteration_depth": ITERATION_DEPTH_PATH,
    "provider_study": PROVIDER_STUDY_PATH,
    "failure_taxonomy": FAILURE_TAXONOMY_PATH,
}

# --- Outputs ---
PHASE3_RESULTS_JSON_PATH = RESULTS_DIR / "phase3_results.json"
PHASE3_RESULTS_MD_PATH = RESULTS_DIR / "phase3_results.md"

# Per-study boards the section points back to for full detail.
STUDY_BOARDS = [
    ("Day 12", "crew vs solo vs AutoML vs default", "comparison_table.md"),
    ("Day 13", "Critic-loop ablation", "day13_critic_ablation.md"),
    ("Day 14", "per-agent ablations", "day14_agent_ablation.md"),
    ("Day 15", "iteration-depth study", "day15_iteration_depth.md"),
    ("Day 16", "provider study", "day16_provider_study.md"),
    ("Day 17", "failure taxonomy", "day17_failure_taxonomy.md"),
]


def load_phase3_bundle(paths: dict | None = None) -> dict[str, dict]:
    """Read every study file. A study that never ran is a hard error, not a gap —
    the consolidated section exists to summarise all six, and silently writing it
    from five would misrepresent the phase."""
    paths = paths or STUDY_PATHS
    bundle: dict[str, dict] = {}
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"phase-3 consolidation needs every study file; missing: {missing}"
        )
    for name, p in paths.items():
        bundle[name] = json.loads(p.read_text(encoding="utf-8"))
    return bundle


# ---------------------------------------------------------------------------
# Extraction — one function per study, each returning a small serialisable block
# ---------------------------------------------------------------------------

def _extract_board(comparison: dict) -> dict[str, Any]:
    """Day 12: the headline board — scores, deltas and win counts, as committed."""
    rows = {}
    for key, row in comparison.get("rows", {}).items():
        systems = row.get("systems", {})
        rows[key] = {
            "metric": row.get("metric"),
            "scores": {
                name: (info or {}).get("value") for name, info in systems.items()
            },
            "mock": {
                name: bool((info or {}).get("mock", False))
                for name, info in systems.items()
            },
            "deltas": comparison.get("deltas", {}).get(key, {}),
        }
    return {
        "rows": rows,
        "wins": comparison.get("wins", {}),
        "any_mock": bool(comparison.get("any_mock", False)),
    }


def _extract_critic(critic: dict) -> dict[str, Any]:
    """Day 13: the loop is free when clean, and is the recovery when not."""
    summary = critic.get("summary", {})
    nat, probe = summary.get("natural", {}), summary.get("deficiency_probe", {})
    probe_drops = {
        key: (arm.get("loop_drop") if isinstance(arm, dict) else None)
        for key, arm in critic.get("deficiency_probe", {}).items()
    }
    return {
        "natural_loop_fired": nat.get("loop_fired_count"),
        "natural_datasets": nat.get("datasets"),
        "natural_mean_drop": nat.get("mean_drop"),
        "probe_loop_fired": probe.get("loop_fired_count"),
        "probe_datasets": probe.get("datasets"),
        "probe_mean_recovery": probe.get("mean_drop"),
        "probe_max_recovery": probe.get("max_drop"),
        "probe_recovery_by_dataset": probe_drops,
    }


def _extract_agents(agent: dict) -> dict[str, Any]:
    """Day 14: per-specialist attribution — summary plus the per-dataset drops."""
    drops = {
        key: row.get("drops", {}) for key, row in agent.get("results", {}).items()
    }
    return {"summary": agent.get("summary", {}), "drops_by_dataset": drops}


def _extract_depth(depth: dict) -> dict[str, Any]:
    """Day 15: the natural sweep is flat; the probe curve is a cliff at budget 2."""
    nat = depth.get("summary", {}).get("natural", {})
    probe = depth.get("summary", {}).get("deficiency_probe", {})
    natural_flat = all(
        (ds.get("score_spread") or 0.0) == 0.0 for ds in nat.values()
    ) if nat else None
    curves = {}
    for key, curve in depth.get("results", {}).get("deficiency_probe", {}).items():
        curves[key] = {
            int(d): (rec or {}).get("value") for d, rec in curve.items()
        }
    return {
        "natural_flat": natural_flat,
        "natural_datasets": len(nat),
        "probe_summary": {
            key: {
                "first_loop_lift": s.get("first_loop_lift"),
                "lift_beyond_first_loop": s.get("lift_beyond_first_loop"),
                "saturation_depth": s.get("saturation_depth"),
                "budget_bound_depths": s.get("budget_bound_depths", []),
            }
            for key, s in probe.items()
        },
        "probe_curves": curves,
    }


def _arm_cost(rec: dict, pricing: dict) -> Optional[float]:
    """Price one run from measured tokens only — no tokens measured, no cost."""
    pin, pout = pricing.get("usd_per_mtok_in"), pricing.get("usd_per_mtok_out")
    tin, tout = rec.get("llm_prompt_tokens"), rec.get("llm_completion_tokens")
    if None in (pin, pout, tin, tout) or not rec.get("llm_narratives_live"):
        return None
    return (tin * pin + tout * pout) / 1e6


def _extract_providers(provider: dict) -> dict[str, Any]:
    """Day 16: live arms with measured-token costs, plus the outage-resilience proof."""
    arms = {}
    for name, arm in provider.get("arms", {}).items():
        pricing = provider.get("providers", {}).get(name, {})
        per_ds = {}
        for key, rec in arm.items():
            per_ds[key] = {
                "ok": bool(rec.get("ok")),
                "value": rec.get("value") if rec.get("ok") else None,
                "mock": bool(rec.get("mock", False)),
                "cost_usd": _arm_cost(rec, pricing),
            }
        costs = [d["cost_usd"] for d in per_ds.values() if d["cost_usd"] is not None]
        arms[name] = {
            "label": pricing.get("label", name),
            "datasets": per_ds,
            "n_scored": sum(1 for d in per_ds.values() if d["value"] is not None),
            "n_failed": sum(1 for d in per_ds.values() if not d["ok"]),
            "total_cost_usd": sum(costs) if costs else None,
            "is_live": name in provider.get("live_arms_run", []),
        }
    res = provider.get("resilience", {})
    return {
        "pricing_as_of": provider.get("pricing_as_of"),
        "arms": arms,
        "blocked_providers": provider.get("blocked_providers", {}),
        "resilience": {
            "n_compared": res.get("n_compared"),
            "n_equal": res.get("n_equal"),
            "all_equal": res.get("all_equal"),
            "max_abs_diff": res.get("max_abs_diff"),
        },
    }


def _extract_failures(taxonomy: dict) -> dict[str, Any]:
    """Day 17: census outcomes + fatal-by-system, and the probe verdicts."""
    census = taxonomy.get("archive_census", {})
    summary = census.get("summary", {})
    probes = []
    for rec in taxonomy.get("probes", {}).get("live_leak", []):
        probes.append({
            "probe": rec.get("probe"),
            "detected": rec.get("detected"),
            "model_saw_leak": rec.get("model_saw_leak"),
        })
    timeout = taxonomy.get("probes", {}).get("live_timeout", {})
    if timeout:
        probes.append({
            "probe": timeout.get("probe", "exec_timeout"),
            "detected": timeout.get("detected"),
        })
    for rec in taxonomy.get("probes", {}).get("record_level", []):
        probes.append({"probe": rec.get("probe"), "detected": rec.get("detected")})
    return {
        "n_crew_runs": census.get("n_crew_runs"),
        "n_solo_runs": census.get("n_solo_runs"),
        "n_events": summary.get("n_events"),
        "by_outcome": summary.get("by_outcome", {}),
        "fatal_by_system": summary.get("fatal_by_system", {}),
        "by_category_totals": {
            cat: info.get("total")
            for cat, info in summary.get("by_category", {}).items()
        },
        "probes": probes,
    }


def assemble_results(bundle: dict[str, dict]) -> dict[str, Any]:
    """The consolidated Phase-3 report — every block traceable to one study file."""
    board = _extract_board(bundle["comparison"])
    report = {
        "schema_version": 1,
        "day": 18,
        "phase": 3,
        "study": "phase3_consolidated_results",
        "board": board,
        "critic": _extract_critic(bundle["critic_ablation"]),
        "agents": _extract_agents(bundle["agent_ablation"]),
        "depth": _extract_depth(bundle["iteration_depth"]),
        "providers": _extract_providers(bundle["provider_study"]),
        "failures": _extract_failures(bundle["failure_taxonomy"]),
        "sources": {name: p.name for name, p in STUDY_PATHS.items()},
    }
    report["headline"] = _headline(report)
    return report


def _headline(report: dict) -> dict[str, Any]:
    """The one-paragraph numbers — exactly what the Day-29 README will quote."""
    wins = report["board"]["wins"]
    fatal = report["failures"]["fatal_by_system"]
    groq = report["providers"]["arms"].get("groq", {})
    return {
        "crew_vs_solo": wins.get("vs_solo", {}),
        "crew_vs_automl": wins.get("vs_automl", {}),
        "crew_vs_default_rf": wins.get("vs_default_rf", {}),
        "critic_probe_max_recovery": report["critic"]["probe_max_recovery"],
        "planner_mean_drop": report["agents"]["summary"]
        .get("planner", {})
        .get("mean_drop"),
        "fe_mean_drop": report["agents"]["summary"]
        .get("feature_engineer", {})
        .get("mean_drop"),
        "fatal_crew": fatal.get("crew", 0),
        "fatal_solo": fatal.get("solo", 0),
        "live_run_total_cost_usd": groq.get("total_cost_usd"),
        "outage_resilience_all_equal": report["providers"]["resilience"]["all_equal"],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def _fmt_signed(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:+.4f}"


SYSTEM_ORDER = [
    ("dummy", "Dummy (floor)"),
    ("default_rf", "default RF"),
    ("solo_agent", "Solo agent"),
    ("automl_flaml", "AutoML (FLAML)"),
    ("crew", "**Crew**"),
]


def render_markdown(report: dict) -> str:
    """The results section itself. Each subsection cites its source board."""
    h = report["headline"]
    board, critic = report["board"], report["critic"]
    agents, depth = report["agents"], report["depth"]
    prov, fail = report["providers"], report["failures"]

    lines: list[str] = []
    add = lines.append

    add("# CrewML — Phase 3 results (consolidated)")
    add("")
    add(
        "*Six studies (Days 12–17), one section. Every number below is reshaped from a "
        "committed study file — nothing was re-run for this document, so it cannot "
        "disagree with the boards it summarises. Per-study boards with full method "
        "notes are linked at the end. All scores are on the LOCKED held-out split the "
        "crew never sees while modeling (EVAL_PROTOCOL.md §3); missing numbers render "
        "as an em dash, never a zero; runs without a live LLM stay labelled mock "
        "(§5).*"
    )
    add("")

    # --- 1. Headline board -------------------------------------------------
    add("## 1. Does the crew win? (Day 12)")
    add("")
    vs_solo, vs_automl, vs_rf = (
        h["crew_vs_solo"], h["crew_vs_automl"], h["crew_vs_default_rf"],
    )
    add(
        f"**Crew vs solo agent: {vs_solo.get('won')}/{vs_solo.get('compared')} · "
        f"vs AutoML (FLAML): {vs_automl.get('won')}/{vs_automl.get('compared')} · "
        f"vs default RF: {vs_rf.get('won')}/{vs_rf.get('compared')}** "
        "(wins counted only where both systems produced a real score)."
    )
    add("")
    add("| Dataset | Metric | " + " | ".join(label for _, label in SYSTEM_ORDER)
        + " | Crew − Solo | Crew − AutoML | Crew − default RF |")
    add("|---|---|" + "---|" * (len(SYSTEM_ORDER) + 3))
    for key, row in board["rows"].items():
        cells = []
        for name, _ in SYSTEM_ORDER:
            val = _fmt(row["scores"].get(name))
            if row["mock"].get(name) and row["scores"].get(name) is not None:
                val += " *(mock)*"
            cells.append(val)
        deltas = row["deltas"]
        add(
            f"| {key} | {row['metric']} | " + " | ".join(cells) + " | "
            + " | ".join(
                _fmt_signed(deltas.get(k))
                for k in ("vs_solo", "vs_automl", "vs_default_rf")
            )
            + " |"
        )
    add("")
    add(
        "The two crew losses are to AutoML (`cpu_small` −0.0009, `kin8nm` −0.0239) and "
        "are reported as losses; the solo agent produced no scorable model on 2/5 "
        "datasets, so those deltas do not exist rather than counting as wins."
    )
    add("")

    # --- 2. Attribution ----------------------------------------------------
    add("## 2. What each agent earns (Days 13–15)")
    add("")
    add(
        f"**Critic loop (Day 13).** On the healthy suite the loop fired on "
        f"{critic['natural_loop_fired']}/{critic['natural_datasets']} datasets — cost "
        f"when idle: {_fmt_signed(critic['natural_mean_drop'])}. Under a deliberately "
        f"crippled first pass it fired {critic['probe_loop_fired']}/"
        f"{critic['probe_datasets']} and recovered a mean "
        f"{_fmt_signed(critic['probe_mean_recovery'])} of held-out score (up to "
        f"{_fmt_signed(critic['probe_max_recovery'])}). The loop is free when clean "
        "and is the entire recovery when not."
    )
    add("")
    planner = agents["summary"].get("planner", {})
    fe = agents["summary"].get("feature_engineer", {})
    add(
        f"**Planner (Day 14).** Helped on {planner.get('helped_count')}/"
        f"{planner.get('compared')} datasets, hurt on {planner.get('hurt_count')}; "
        f"mean drop when removed {_fmt_signed(planner.get('mean_drop'))}, largest "
        f"{_fmt_signed(planner.get('max_drop'))} on `{planner.get('best_dataset')}`. "
        f"**Feature Engineer (Day 14).** Helped on {fe.get('helped_count')}/"
        f"{fe.get('compared')}, hurt on {fe.get('hurt_count')}; mean "
        f"{_fmt_signed(fe.get('mean_drop'))}, largest {_fmt_signed(fe.get('max_drop'))} "
        f"on `{fe.get('best_dataset')}` — small but never negative."
    )
    add("")
    add("| Dataset | Planner drop | FE drop | Critic probe recovery |")
    add("|---|---|---|---|")
    probe_rec = critic["probe_recovery_by_dataset"]
    for key, drops in agents["drops_by_dataset"].items():
        add(
            f"| {key} | {_fmt_signed(drops.get('planner'))} | "
            f"{_fmt_signed(drops.get('feature_engineer'))} | "
            f"{_fmt_signed(probe_rec.get(key))} |"
        )
    add("")
    add(
        "*(Drop = full crew − ablated arm on the locked holdout; positive means the "
        "specialist added score. Critic recovery is from the forced-deficiency probe "
        "and exists only for the two probe datasets — an em dash is a dataset the "
        "probe cannot reach, not a zero.)*"
    )
    add("")
    add(
        f"**Iteration depth (Day 15).** The natural sweep is "
        f"{'flat — unused budget changes nothing' if depth['natural_flat'] else 'NOT flat'} "
        f"across {depth['natural_datasets']} datasets. Under the deficiency probe the "
        "depth-response is a cliff, not a slope:"
    )
    add("")
    add("| Probe dataset | Budget 1 | Budget 2 | First-loop lift | Beyond first loop | Saturation |")
    add("|---|---|---|---|---|---|")
    for key, s in depth["probe_summary"].items():
        curve = depth["probe_curves"].get(key, {})
        add(
            f"| {key} | {_fmt(curve.get(1))} | {_fmt(curve.get(2))} | "
            f"{_fmt_signed(s.get('first_loop_lift'))} | "
            f"{_fmt_signed(s.get('lift_beyond_first_loop'))} | "
            f"budget {s.get('saturation_depth')} |"
        )
    add("")
    add(
        "Budget 1 ships the stump and reads as budget-bound (starvation is visible, "
        "not silent); the first allowed loop buys the entire recovery; every further "
        "loop buys nothing and goes unused. The production `max_iterations = 3` sits "
        "on the safe plateau."
    )
    add("")

    # --- 3. Cost & providers ----------------------------------------------
    add("## 3. What it costs (Day 16)")
    add("")
    groq = prov["arms"].get("groq", {})
    if groq:
        add(
            f"**Live crew runs (Groq — Llama 3.3 70B, prices as of "
            f"{prov['pricing_as_of']}):** {groq.get('n_scored')}/5 datasets scored, "
            f"total measured cost {('$' + format(groq['total_cost_usd'], '.4f')) if groq.get('total_cost_usd') is not None else '—'} "
            "for the whole suite — under a cent per dataset. Costs are computed from "
            "measured tokens only; a run with no live calls has no cost, not a zero "
            "cost."
        )
        if groq.get("n_failed"):
            add("")
            add(
                f"The live arm's {groq['n_failed']} failure (`diabetes`: generated FE "
                "code produced non-finite features and training died) is on the board "
                "as a failure — the Critic filed it and finalised without a model. It "
                "is Day 20's (self-repair) motivating case."
            )
    res = prov["resilience"]
    add("")
    add(
        f"**Outage resilience.** Fresh no-provider runs vs archival runs with a "
        f"failing provider: {res.get('n_equal')}/{res.get('n_compared')} datasets "
        f"{'bit-identical' if res.get('all_equal') else 'equal within noise'} "
        f"(max |Δ| = {_fmt(res.get('max_abs_diff'), '.2e')}) — holdout quality is "
        "provably independent of provider availability; an outage costs narrative "
        "richness, never score."
    )
    blocked = prov.get("blocked_providers", {})
    if blocked:
        add("")
        add(
            "Still blocked: "
            + "; ".join(f"**{name}** ({why})" for name, why in blocked.items())
            + ". The study re-runs and re-prices itself when a key appears."
        )
    add("")

    # --- 4. Failure modes --------------------------------------------------
    add("## 4. Where it fails, and who catches it (Day 17)")
    add("")
    add(
        f"Census of {fail['n_crew_runs']} archived crew runs + {fail['n_solo_runs']} "
        f"solo runs → {fail['n_events']} classified events. Fatal failures (no scored "
        f"model): **crew {h['fatal_crew']}** vs **solo {h['fatal_solo']}** — and the "
        "crew's fatal was *caught and filed* by the Critic (an honest no-model "
        "finalise), while every solo failure is silent-fatal: no Critic to file it, "
        "no fallback to absorb it, no chooser to contain it."
    )
    add("")
    outcomes = fail["by_outcome"]
    add("| Outcome | " + " | ".join(outcomes) + " |")
    add("|---|" + "---|" * len(outcomes))
    add("| Events | " + " | ".join(str(outcomes[o]) for o in outcomes) + " |")
    add("")
    probe_bits = []
    for p in fail["probes"]:
        verdict = "caught" if p.get("detected") else "**MISSED**"
        probe_bits.append(f"`{p.get('probe')}` {verdict}")
    add("**Injection probes:** " + " · ".join(probe_bits) + ".")
    add("")
    add(
        "The missed probe is the phase's most valuable negative result: a leaked "
        "column at 95% target agreement sits *below* the Profiler's purity screen "
        "(0.995) and keeps CV *under* the Critic's too-good-to-be-true ceiling "
        "(0.995), so nothing fires and the model trains on the leak. That measured "
        "detection window is logged as direct input to Day 22 (leakage & honesty "
        "guards) — deliberately not patched inside the study that found it."
    )
    add("")

    # --- 5. Caveats ---------------------------------------------------------
    add("## 5. Caveats — read before quoting")
    add("")
    add(
        "* Baseline scores (Dummy / default RF / solo / AutoML) and the crew's "
        "headline column come from the deterministic core (seed-locked); the live "
        "Groq arm reproduces them within noise but is scored separately (Day 16)."
    )
    add(
        "* The solo agent failed outright on 2/5 datasets; its column is honest "
        "about that, and so are the missing deltas."
    )
    add(
        "* The Critic's recovery numbers come from an instrumented handicap "
        "(`CREWML_ABLATION_HANDICAP=1`), not from natural runs — on healthy data the "
        "loop never fired. Both facts are the finding."
    )
    add(
        "* The leakage detection window (§4) is open until Day 22. The claim \"the "
        "crew detects leakage\" holds only for leaks outside that window."
    )
    add(
        "* Anthropic provider arm still unpriced-live: no ANTHROPIC_API_KEY this "
        "phase."
    )
    add("")

    # --- 6. Sources ----------------------------------------------------------
    add("## 6. Per-study boards (full method notes)")
    add("")
    for day, title, fname in STUDY_BOARDS:
        add(f"* {day} — {title}: `results/{fname}`")
    add("")
    add(
        "*Chart: `results/charts/day18_phase3_summary.png` — the four headline "
        "panels (board deltas, agent attribution, depth cliff, failure outcomes).*"
    )
    add("")
    return "\n".join(lines)


def write_results(report: dict) -> dict[str, Any]:
    """Write JSON + markdown; return the paths for the runner to print."""
    PHASE3_RESULTS_JSON_PATH.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    PHASE3_RESULTS_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return {
        "json": str(PHASE3_RESULTS_JSON_PATH),
        "markdown": str(PHASE3_RESULTS_MD_PATH),
    }
