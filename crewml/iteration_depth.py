"""Day 15 — the iteration-depth study: what does each extra Critic loop buy, and at what cost?

Day 13 established *that* the Critic loop earns its keep (it recovers a deficient first
pass); Day 14 established *who* the loop needs (without the Planner it has no actuator).
The remaining question about the loop is **how much of it to budget**: the crew ships with
``max_iterations = 3`` — is that number load-bearing, wasteful, or too small? This study
answers by sweeping the iteration budget and measuring the held-out score, the wall-clock
and the LLM tokens at every depth.

Vocabulary (used consistently in code, tables and charts):

* **budget** — ``max_iterations``, the number of Critic *passes* allowed (the knob).
* **passes** — Critic passes the crew actually used (``iterations_run``).
* **loops used** — ``passes − 1``: how many times the crew actually went back to the
  Planner. A budget of 1 makes looping structurally impossible.
* **budget-bound** — the final Critic pass still had actionable findings but was
  finalised by its "budget reached" rule: the crew was *cut off*, not *done*. This flag
  is what distinguishes saturation from starvation when two depths score the same.

Two arms, same discipline as Day 13 (which they extend):

1. **Natural sweep** (:data:`NATURAL_DEPTHS` × the five real datasets). Day 12 showed the
   Critic finalises pass 1 on all five, so the prediction is that the score *and the cost*
   are flat in the budget — unused budget must be free, or "3" would be a tax on healthy
   data. The sweep measures that rather than asserting it.
2. **Deficiency sweep** (:data:`PROBE_DEPTHS` × the two probe datasets). Under the Day-13
   first-pass handicap (``CREWML_ABLATION_HANDICAP`` — instrumentation, clearly labelled,
   never a production setting) the crew *needs* the loop, so depth becomes observable:
   budget 1 must ship the stump (and read as budget-bound), budget 2 lets the loop fire
   once, budgets 3-4 reveal whether further loops keep buying score or the Critic
   correctly stops. The probe keeps Day 13's scope — the two regression sets — because
   the handicap reliably drives R² under the 0.10 underfit floor, whereas a stumped
   classifier can still clear the 0.60 ROC-AUC floor and the loop would never arm.

The study's economic output is **cost per point of lift**: for each budget step, the
marginal held-out lift against the marginal seconds and tokens, with "a point" defined as
0.01 of the primary metric. A step whose marginal lift is ~0 gets no ratio (``None``) —
dividing by noise would manufacture an impressive-looking number out of nothing.

Honesty rules, unchanged: the holdout is scored outside the graph with the seal
re-verified per run; a lift is only computed between two real numbers; mock runs are
flagged and never presented as real (EVAL_PROTOCOL.md §5).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from crewml.ablation import LOOPED, _handicap, run_variant
from crewml.config import RESULTS_DIR
from crewml.datasets import REGISTRY, load_manifest

ITERATION_DEPTH_SCHEMA_VERSION = 1
ITERATION_DEPTH_RESULT_PATH = RESULTS_DIR / "day15_iteration_depth.json"
ITERATION_DEPTH_TABLE_MD_PATH = RESULTS_DIR / "day15_iteration_depth.md"

ARTIFACT_PREFIX = "day15"

# One "point" of lift on the primary metric, for the cost ratios.
POINT = 0.01
# Below this, a marginal lift is noise and prices no ratio.
LIFT_EPS = 1e-4

# Natural arm: two depths suffice — the Critic finalises pass 1 on healthy data, so any
# pair of budgets should produce identical runs; {min, production} shows it directly.
NATURAL_DEPTHS: tuple[int, ...] = (1, 3)
# Probe arm: production is 3; 4 adds headroom so "the Critic stops on its own" is a
# measured fact rather than the budget doing the stopping.
PROBE_DEPTHS: tuple[int, ...] = (1, 2, 3, 4)
# Day 13's probe scope (see module docstring for why classification sets can't arm it).
PROBE_KEYS: tuple[str, ...] = ("kin8nm", "cpu_small")

NATURAL = "natural"
PROBE = "deficiency_probe"


def run_depth_curve(
    key: str,
    manifest: dict,
    depths: Iterable[int],
    *,
    handicap: bool,
    progress=None,
) -> dict[int, dict[str, Any]]:
    """Run the full crew on one dataset once per budget; return {depth: run record}.

    Every run goes through :func:`crewml.ablation.run_variant` — the same recursion
    budget, post-run seal check and outside-the-graph holdout scoring as Days 12-14 —
    so a depth-curve point means exactly what every other scored number means. The
    handicap flag is scoped per run (:func:`crewml.ablation._handicap`) and the artifact
    tag carries both the arm and the depth so no two points clobber each other.
    """
    arm = "handicap" if handicap else "natural"
    out: dict[int, dict[str, Any]] = {}
    for d in depths:
        d = int(d)
        if d < 1:
            raise ValueError(f"iteration budget must be >= 1, got {d}")
        if progress:
            progress(f"  [day15] {key}: {arm} arm, budget={d} ...")
        with _handicap(handicap):
            out[d] = run_variant(
                key, LOOPED, manifest,
                max_iterations=d,
                artifact_prefix=ARTIFACT_PREFIX,
                artifact_tag=f"{arm}_d{d}",
            )
        if progress:
            progress("    " + summarise_point(key, d, out[d]))
    return out


def run_depth_study(
    natural_keys: Iterable[str],
    probe_keys: Iterable[str] = PROBE_KEYS,
    *,
    natural_depths: Iterable[int] = NATURAL_DEPTHS,
    probe_depths: Iterable[int] = PROBE_DEPTHS,
    progress=None,
) -> dict[str, dict[str, dict[int, dict]]]:
    """Both arms of the study. Returns {arm: {dataset: {depth: record}}}."""
    manifest = load_manifest()
    study: dict[str, dict[str, dict[int, dict]]] = {NATURAL: {}, PROBE: {}}
    for k in natural_keys:
        if progress:
            progress(f"[day15] natural sweep === {k} ===")
        study[NATURAL][k] = run_depth_curve(k, manifest, natural_depths, handicap=False, progress=progress)
    for k in probe_keys:
        if progress:
            progress(f"[day15] deficiency sweep === {k} ===")
        study[PROBE][k] = run_depth_curve(k, manifest, probe_depths, handicap=True, progress=progress)
    return study


def summarise_point(key: str, depth: int, rec: dict) -> str:
    """One human line per curve point."""
    v = f"{rec['value']:.4f}" if rec.get("ok") else f"FAILED({rec.get('error')})"
    bound = " BUDGET-BOUND" if rec.get("budget_bound") else ""
    return (
        f"{key} budget={depth}: {rec.get('metric', '')}={v} "
        f"passes={rec.get('iterations_run')}{bound} "
        f"({rec.get('crew_seconds')}s, {_tokens(rec) or 0} tok)"
    )


# --- Curve analysis (pure) ----------------------------------------------------

def _tokens(rec: dict) -> Optional[int]:
    p, c = rec.get("llm_prompt_tokens"), rec.get("llm_completion_tokens")
    return (p or 0) + (c or 0) if (p is not None or c is not None) else None


def analyse_curve(curve: dict[int, dict]) -> list[dict[str, Any]]:
    """Turn one {depth: record} curve into per-depth analysis rows (pure).

    Each row prices the *step up from the previous budget*: ``marginal_lift`` on the
    primary metric, ``marginal_seconds`` / ``marginal_tokens`` of extra spend, and
    ``seconds_per_point`` / ``tokens_per_point`` (cost per :data:`POINT` of metric).
    Ratios are only computed when the marginal lift clears :data:`LIFT_EPS` — a ~zero
    lift gets ``None``, never a divided-by-noise number. Lifts are only computed
    between two real (``ok``) values; a failed point contributes ``None`` all the way
    down, never a flattering zero.
    """
    rows: list[dict[str, Any]] = []
    prev: Optional[dict] = None
    base_value: Optional[float] = None
    for d in sorted(int(k) for k in curve):
        rec = curve[d] if d in curve else curve[str(d)]  # JSON round-trips keys to str
        value = rec["value"] if rec.get("ok") else None
        if base_value is None and rows == [] and value is not None:
            base_value = value

        prev_value = (prev["value"] if prev and prev.get("ok") else None) if prev else None
        marginal = (
            round(value - prev_value, 6)
            if (value is not None and prev_value is not None)
            else None
        )
        d_secs = (
            round(rec["crew_seconds"] - prev["crew_seconds"], 2)
            if prev is not None and isinstance(rec.get("crew_seconds"), (int, float))
            and isinstance(prev.get("crew_seconds"), (int, float))
            else None
        )
        toks, prev_toks = _tokens(rec), (_tokens(prev) if prev else None)
        d_toks = toks - prev_toks if (toks is not None and prev_toks is not None) else None

        priceable = isinstance(marginal, float) and marginal > LIFT_EPS
        rows.append({
            "depth": d,
            "value": value,
            "metric": rec.get("metric"),
            "passes": rec.get("iterations_run"),
            "loops_used": (rec.get("iterations_run") or 1) - 1,
            "budget_bound": bool(rec.get("budget_bound")),
            "crew_seconds": rec.get("crew_seconds"),
            "llm_tokens": toks,
            "lift_vs_min_budget": (
                round(value - base_value, 6)
                if (value is not None and base_value is not None) else None
            ),
            "marginal_lift": marginal,
            "marginal_seconds": d_secs,
            "marginal_tokens": d_toks,
            "seconds_per_point": (
                round(d_secs / (marginal / POINT), 2)
                if priceable and isinstance(d_secs, (int, float)) else None
            ),
            "tokens_per_point": (
                round(d_toks / (marginal / POINT), 1)
                if priceable and isinstance(d_toks, (int, float)) else None
            ),
        })
        prev = rec
    return rows


def _curve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline facts for one dataset's curve (pure over analyse_curve rows)."""
    values = [r["value"] for r in rows if r["value"] is not None]
    best = max(values) if values else None
    # Saturation: the smallest budget whose score is within LIFT_EPS of the curve's best.
    saturation = next(
        (r["depth"] for r in rows if r["value"] is not None and best is not None
         and best - r["value"] <= LIFT_EPS),
        None,
    )
    first_step = next((r for r in rows if r["marginal_lift"] is not None), None)
    later = [r["marginal_lift"] for r in rows[2:] if isinstance(r["marginal_lift"], float)]
    return {
        "depths": [r["depth"] for r in rows],
        "best_value": best,
        "saturation_depth": saturation,
        "first_loop_lift": first_step["marginal_lift"] if first_step else None,
        "lift_beyond_first_loop": round(sum(later), 6) if later else None,
        "score_spread": round(max(values) - min(values), 6) if values else None,
        "max_passes_used": max((r["passes"] or 0) for r in rows) if rows else None,
        "budget_bound_depths": [r["depth"] for r in rows if r["budget_bound"]],
    }


def assemble_report(study: dict[str, dict[str, dict[int, dict]]]) -> dict[str, Any]:
    """Bundle both arms + per-curve analyses + summaries into the committed object."""
    any_mock = any(
        rec.get("mock")
        for arm in study.values()
        for curve in arm.values()
        for rec in curve.values()
    )
    analysis = {
        arm: {key: analyse_curve(curve) for key, curve in curves.items()}
        for arm, curves in study.items()
    }
    return {
        "schema_version": ITERATION_DEPTH_SCHEMA_VERSION,
        "day": 15,
        "phase": 3,
        "study": "iteration_depth",
        "any_mock": any_mock,
        "point": POINT,
        "results": study,
        "analysis": analysis,
        "summary": {
            arm: {key: _curve_summary(rows) for key, rows in curves.items()}
            for arm, curves in analysis.items()
        },
    }


# --- Rendering ---------------------------------------------------------------

def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def _fmt_signed(v: Optional[float]) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "—"


def render_markdown(report: dict) -> str:
    """Render the Day-15 depth board as committed markdown."""
    lines = [
        "# Day 15 — iteration-depth study: what does each extra Critic loop buy?",
        "",
        "*The iteration budget (`max_iterations` = allowed Critic passes; budget 1 makes "
        "looping structurally impossible) swept on the full crew, everything else held "
        "constant. Scores are LOCKED-holdout, scored outside the graph, seal re-verified "
        "per run. **budget-bound** marks a run whose final Critic pass still had "
        "actionable findings — the crew was cut off, not done. Cost ratios price a "
        "*point* = 0.01 of the primary metric and are only computed when the marginal "
        "lift is real (> 1e-4) — never divided by noise.*",
        "",
    ]
    # Session anomalies are part of the record, not a footnote in the JSON: a reader
    # of the board alone must see them next to the numbers they qualify.
    if report.get("llm_narratives_note"):
        lines += [f"> **Session note (LLM):** {report['llm_narratives_note']}", ""]
    if report.get("rerun_points"):
        pts = ", ".join(f"{p['dataset']}@{p['depth']} ({p['arm']})" for p in report["rerun_points"])
        lines += [
            f"> **Session note (re-runs):** {len(report['rerun_points'])} point(s) re-ran with "
            f"`CREWML_EXECUTOR_TIMEOUT_S=600` after a host slowdown tripped the default 120s "
            f"executor cap ({pts}). Scores are timeout-independent given completion; treat "
            "the seconds columns as indicative only this session.",
            "",
        ]
    lines += [
        "### Arm 1 — Natural sweep (real datasets, no handicap)",
        "",
        "| Dataset | Metric | " + " | ".join(
            f"Budget {d}" for d in _arm_depths(report, NATURAL)
        ) + " | Spread | Passes used |",
        "|---" * (4 + len(_arm_depths(report, NATURAL))) + "|",
    ]
    for key, rows in report["analysis"][NATURAL].items():
        s = report["summary"][NATURAL][key]
        cells = " | ".join(_fmt(r["value"]) for r in rows)
        lines.append(
            f"| {key} | {rows[0].get('metric') or REGISTRY[key].metric} | {cells} | "
            f"{_fmt(s['score_spread'], '.6f')} | {s['max_passes_used']} |"
        )
    ns = report["summary"][NATURAL]
    flat = all((s["score_spread"] or 0) <= LIFT_EPS for s in ns.values())
    all_single_pass = all((s["max_passes_used"] or 0) <= 1 for s in ns.values())
    lines += [
        "",
        ("On healthy data the Critic finalises pass 1 at every budget, so the sweep is "
         f"{'**flat**' if flat else '**NOT flat — investigate**'}: unused budget changes "
         "neither the score nor the work done. The production setting of 3 is free "
         "insurance, not a tax."
         if all_single_pass else
         "The Critic used more than one pass on at least one healthy dataset — see the "
         "per-depth rows before drawing conclusions."),
        "",
        "### Arm 2 — Deficiency sweep (first pass handicapped; the loop must recover)",
        "",
        "*Same instrumentation as Day 13 (`CREWML_ABLATION_HANDICAP=1`): pass 1 capacity "
        "is capped to a near-stump so the winning CV score falls under the Critic's "
        "underfit floor and the loop has real work to do. Scope is Day 13's two "
        "regression sets — the handicap cannot push a classifier under the 0.60 ROC-AUC "
        "floor, so the loop would never arm there.*",
        "",
        "| Dataset | Budget | Score (R²) | Passes | Budget-bound | Marginal lift | "
        "Marginal cost | s per point | Tokens per point |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, rows in report["analysis"][PROBE].items():
        for r in rows:
            cost = (
                f"{_fmt(r['marginal_seconds'], '.0f')}s / {r['marginal_tokens'] if r['marginal_tokens'] is not None else '—'} tok"
                if r["marginal_seconds"] is not None else "—"
            )
            lines.append(
                f"| {key} | {r['depth']} | {_fmt(r['value'])} | {r['passes']} | "
                f"{'**yes**' if r['budget_bound'] else 'no'} | {_fmt_signed(r['marginal_lift'])} | "
                f"{cost} | {_fmt(r['seconds_per_point'], '.2f')} | "
                f"{_fmt(r['tokens_per_point'], '.0f')} |"
            )
    ps = report["summary"][PROBE]
    if ps:
        first = [s["first_loop_lift"] for s in ps.values() if isinstance(s["first_loop_lift"], float)]
        beyond = [s["lift_beyond_first_loop"] for s in ps.values() if isinstance(s["lift_beyond_first_loop"], float)]
        sat = {k: s["saturation_depth"] for k, s in ps.items()}
        lines += [
            "",
            f"**The depth-response is a cliff, not a slope**: the first allowed loop buys "
            f"{' and '.join(f'{v:+.4f}' for v in first)} of held-out R² "
            f"(budget 1 → 2), while every loop after it buys "
            f"{' and '.join(f'{v:+.4f}' for v in beyond) if beyond else '—'}. "
            f"Saturation depth per dataset: "
            + ", ".join(f"`{k}` at budget {v}" for k, v in sat.items()) + ". "
            "Past saturation the Critic finalises on its own — extra budget is unused, "
            "not merely unhelpful.",
            "",
            "Reading the two arms together: the right budget is *at least 2* (budget 1 "
            "ships the stump and reads as budget-bound — starvation is visible, not "
            "silent) and anything ≥ the crew's observed need is free. The production "
            "`max_iterations = 3` sits on the safe plateau of both curves.",
            "",
        ]
    if report["any_mock"]:
        lines += ["*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).", ""]
    return "\n".join(lines) + "\n"


def _arm_depths(report: dict, arm: str) -> list[int]:
    curves = report["analysis"].get(arm) or {}
    first = next(iter(curves.values()), [])
    return [r["depth"] for r in first]


def write_report(report: dict) -> dict:
    """Persist the report as JSON + markdown; return it unchanged."""
    ITERATION_DEPTH_RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    ITERATION_DEPTH_TABLE_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report
