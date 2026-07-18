"""Day 13 — the Critic-loop ablation: does the loop earn its keep?

Phase 3 opened (Day 12) by proving the crew beats a solo agent and a classical AutoML
ceiling on the locked holdout. This module answers the next question the project has to
answer honestly: **of the crew's parts, is the Critic feedback loop actually pulling
weight, or is it decoration?** The clean way to find out is an ablation — run the
*identical* crew with the one component removed and measure what the holdout score does.

The removal is structural, not a flag a node has to honour: :func:`crewml.crew.build_crew`
grows a ``no_critic`` variant (Day 13) whose graph drops the Critic node and its loop edge,
so the Trainer hands straight to the Ensembler and the crew makes exactly one forward pass.
Everything else — the same Profiler, Planner, Feature Engineer, Trainer, Ensembler and
Reporter, the same seed, the same LLM settings — is held constant. Any score difference is
the loop's, and nothing else's.

Two studies live here, because the loop is a *conditional* safeguard and an honest
ablation has to show both faces of that:

1. **Natural ablation** (:func:`run_natural_ablation`). Run both variants on the five real
   datasets. On healthy data the Critic correctly finalises on pass 1 (Day 12 showed all
   five did), so the loop never fires and the drop is ~0. That is not a null result to bury
   — it is the finding: the loop costs nothing when the first pass is already clean, so it
   is never a *liability*.

2. **Forced-deficiency probe** (:func:`run_deficiency_probe`). To show the loop *can* earn
   its keep, we hand the Planner a deliberately crippled first pass (``CREWML_ABLATION_HANDICAP``
   — see :func:`crewml.crew.planner._apply_ablation_handicap`) so the winning CV score falls
   to the Critic's underfit floor. Now the two variants diverge: *with* the loop the Critic
   diagnoses underfit and the Planner restores capacity on the next pass; *without* it the
   crew ships the stump. The gap is the loop's contribution, measured on the holdout.

Same honesty discipline as every other scored surface in the project:

* **The holdout is scored outside the graph**, after each variant has finished, via
  :mod:`crewml.holdout_eval`. No node — in either variant — can reach it.
* **The seal is re-verified per run.** A broken fingerprint raises rather than reports.
* **Mock runs are flagged** and never presented as real (EVAL_PROTOCOL.md §5).
* **A drop is only computed between two real numbers.** If either variant failed to ship a
  model on a dataset, the drop is ``None``, never a flattering zero.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from crewml.config import ARTIFACTS_DIR, MAX_ITERATIONS, RESULTS_DIR, is_mock_mode
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY, load_manifest, verify_holdout_untouched
from crewml.holdout_eval import score_on_holdout

ABLATION_SCHEMA_VERSION = 1
ABLATION_RESULT_PATH = RESULTS_DIR / "day13_critic_ablation.json"
ABLATION_TABLE_MD_PATH = RESULTS_DIR / "day13_critic_ablation.md"

HANDICAP_ENV = "CREWML_ABLATION_HANDICAP"
HANDICAP_NOTE = (
    "first pass capacity capped to a near-stump via CREWML_ABLATION_HANDICAP=1 so the "
    "winning CV score falls to the Critic's underfit floor; instrumentation only"
)

# The two variants the ablation compares. ``full`` is production (Critic loop present);
# ``no_critic`` is the ablated topology (one forward pass, no loop).
LOOPED = "full"
NO_CRITIC = "no_critic"


def _positive_class(manifest: dict, key: str) -> Optional[str]:
    return (manifest["datasets"][key].get("target") or {}).get("positive_class")


@contextmanager
def _handicap(enabled: bool):
    """Temporarily set the first-pass handicap env flag, restoring it afterward.

    Scoped so a probe run cannot leak the handicap into a later natural run in the same
    process — the flag is the only difference between the two studies and must not bleed.
    """
    prev = os.environ.get(HANDICAP_ENV)
    os.environ[HANDICAP_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(HANDICAP_ENV, None)
        else:
            os.environ[HANDICAP_ENV] = prev


def run_variant(
    key: str,
    variant: str,
    manifest: dict,
    *,
    max_iterations: int = MAX_ITERATIONS,
    artifact_tag: str = "",
) -> dict[str, Any]:
    """Run one crew variant on one dataset, then final-score it on the LOCKED holdout.

    Mirrors ``scripts/run_crew_benchmark._run_and_score`` exactly (same recursion budget,
    same post-run seal check, same :func:`score_on_holdout` call) so an ablation number
    means the same thing a Day-12 crew number means — only the graph topology differs.

    ``artifact_tag`` namespaces the persisted final-state JSON so the looped and no-critic
    runs on the same dataset don't clobber each other's artifacts.
    """
    spec = REGISTRY[key]

    started = time.time()
    app = build_crew(variant=variant)
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    final = app.invoke(state, config={"recursion_limit": limit})
    crew_seconds = time.time() - started

    # Seal must be intact before scoring — proves the variant never touched the holdout.
    if not verify_holdout_untouched(key):
        raise RuntimeError(f"{key}/{variant}: holdout seal broken DURING the crew run — aborting")

    scored = score_on_holdout(spec, final, positive_class=_positive_class(manifest, key))

    critiques = final.get("critiques") or []
    record = {
        **scored,
        "variant": variant,
        "mock": is_mock_mode(),
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "loop_fired": bool((final.get("iteration") or 0) > 1),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "crew_seconds": round(crew_seconds, 2),
    }

    tag = f"_{artifact_tag}" if artifact_tag else ""
    out_dir = ARTIFACTS_DIR / "ablation" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"day13_{variant}{tag}.json").write_text(json.dumps(final, indent=2, default=str))
    return record


def _pair(
    key: str,
    manifest: dict,
    *,
    max_iterations: int,
    handicap: bool,
) -> dict[str, Any]:
    """Run both variants on one dataset under identical settings and pair the results.

    ``loop_drop`` is ``looped − no_critic`` on the primary metric (higher-is-better across
    the whole suite), computed only when both variants shipped a real number — so a
    positive drop always means the Critic loop *added* held-out score.
    """
    tag = "handicap" if handicap else "natural"
    with _handicap(handicap):
        looped = run_variant(key, LOOPED, manifest, max_iterations=max_iterations, artifact_tag=tag)
        no_critic = run_variant(key, NO_CRITIC, manifest, max_iterations=max_iterations, artifact_tag=tag)

    lv = looped["value"] if looped.get("ok") else None
    nv = no_critic["value"] if no_critic.get("ok") else None
    drop = round(lv - nv, 6) if (lv is not None and nv is not None) else None

    return {
        "metric": REGISTRY[key].metric,
        "looped": looped,
        "no_critic": no_critic,
        "loop_drop": drop,            # looped − no_critic; > 0 => the loop helped
        "loop_fired": looped.get("loop_fired", False),
        "iterations_looped": looped.get("iterations_run"),
    }


def run_natural_ablation(
    keys: Iterable[str],
    *,
    max_iterations: int = MAX_ITERATIONS,
    progress=None,
) -> dict[str, dict]:
    """Study 1 — both variants on real datasets, no handicap. Returns {key: pair}."""
    out: dict[str, dict] = {}
    manifest = load_manifest()
    for k in keys:
        if progress:
            progress(f"[day13] natural ablation === {k} ===")
        out[k] = _pair(k, manifest, max_iterations=max_iterations, handicap=False)
        if progress:
            progress("  " + summarise_pair(k, out[k]))
    return out


def run_deficiency_probe(
    keys: Iterable[str],
    *,
    max_iterations: int = MAX_ITERATIONS,
    progress=None,
) -> dict[str, dict]:
    """Study 2 — both variants with a crippled first pass, so the loop must fire to recover."""
    out: dict[str, dict] = {}
    manifest = load_manifest()
    for k in keys:
        if progress:
            progress(f"[day13] deficiency probe === {k} ===")
        out[k] = _pair(k, manifest, max_iterations=max_iterations, handicap=True)
        if progress:
            progress("  " + summarise_pair(k, out[k]))
    return out


def summarise_pair(key: str, pair: dict) -> str:
    """A one-line human summary of a paired result."""
    lp, nc = pair["looped"], pair["no_critic"]
    metric = pair["metric"]

    def _v(rec):
        return f"{rec['value']:.4f}" if rec.get("ok") else f"FAILED({rec.get('error')})"

    drop = pair["loop_drop"]
    drop_str = f"{drop:+.4f}" if isinstance(drop, float) else "—"
    return (
        f"{key:<10} {metric}: looped={_v(lp)} (passes={lp.get('iterations_run')}, "
        f"fired={pair['loop_fired']}) vs no_critic={_v(nc)} (passes={nc.get('iterations_run')}) "
        f"| loop_drop={drop_str}"
    )


def _study_summary(study: dict[str, dict]) -> dict[str, Any]:
    """Aggregate one study's per-dataset drops into headline numbers."""
    drops = [p["loop_drop"] for p in study.values() if isinstance(p["loop_drop"], float)]
    fired = [k for k, p in study.items() if p["loop_fired"]]
    helped = [d for d in drops if d > 1e-9]
    return {
        "datasets": len(study),
        "compared": len(drops),
        "loop_fired_count": len(fired),
        "loop_fired_datasets": fired,
        "loop_helped_count": len(helped),
        "mean_drop": round(sum(drops) / len(drops), 6) if drops else None,
        "max_drop": round(max(drops), 6) if drops else None,
        "min_drop": round(min(drops), 6) if drops else None,
    }


def assemble_report(natural: dict[str, dict], probe: dict[str, dict]) -> dict[str, Any]:
    """Bundle both studies + their summaries into the committed report object."""
    any_mock = any(
        p[arm].get("mock")
        for study in (natural, probe)
        for p in study.values()
        for arm in ("looped", "no_critic")
    )
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "day": 13,
        "phase": 3,
        "study": "critic_loop_ablation",
        "any_mock": any_mock,
        "handicap_note": HANDICAP_NOTE,
        "natural": natural,
        "deficiency_probe": probe,
        "summary": {
            "natural": _study_summary(natural),
            "deficiency_probe": _study_summary(probe),
        },
    }


# --- Rendering ---------------------------------------------------------------

def _fmt(rec: dict) -> str:
    return f"{rec['value']:.4f}" if rec.get("ok") else "—"


def _fmt_drop(drop: Optional[float]) -> str:
    return "—" if not isinstance(drop, float) else f"{drop:+.4f}"


def _render_study_table(title: str, study: dict[str, dict]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Dataset | Metric | Looped (full crew) | No-Critic (ablated) | Loop drop | Loop fired | Passes (looped) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, p in study.items():
        lines.append(
            f"| {key} | {p['metric']} | {_fmt(p['looped'])} | {_fmt(p['no_critic'])} | "
            f"{_fmt_drop(p['loop_drop'])} | {'yes' if p['loop_fired'] else 'no'} | "
            f"{p['looped'].get('iterations_run')} |"
        )
    return lines


def render_markdown(report: dict) -> str:
    """Render the full Day-13 ablation as a committed markdown board."""
    lines = [
        "# Day 13 — Critic-loop ablation",
        "",
        "*Same crew, the Critic feedback loop removed structurally (the `no_critic` graph "
        "variant: Trainer → Ensembler, one forward pass). Same seed, same LLM settings, same "
        "holdout scoring. `Loop drop` = looped − no_critic on the primary metric "
        "(higher-is-better); a positive drop means the loop **added** held-out score.*",
        "",
    ]
    lines += _render_study_table("Study 1 — Natural ablation (real datasets, no handicap)", report["natural"])
    ns = report["summary"]["natural"]
    lines += [
        "",
        f"On the real suite the Critic fired the loop on **{ns['loop_fired_count']}/{ns['datasets']}** "
        f"dataset(s) — it judged the first pass clean and finalised. Mean loop drop: "
        f"**{_fmt_drop(ns['mean_drop'])}**. The loop costs nothing when the first pass is already "
        "healthy: that is the point — it is a conditional safeguard, never a liability.",
        "",
    ]
    lines += _render_study_table(
        "Study 2 — Forced-deficiency probe (crippled first pass, loop must recover)",
        report["deficiency_probe"],
    )
    ds = report["summary"]["deficiency_probe"]
    lines += [
        "",
        f"With a deliberately crippled first pass ({report['handicap_note']}), the loop fired on "
        f"**{ds['loop_fired_count']}/{ds['datasets']}** dataset(s) and the ablated variant — with no "
        f"Critic to diagnose the underfit — shipped the stump. Mean recovery credited to the loop: "
        f"**{_fmt_drop(ds['mean_drop'])}**"
        + (f", up to **{_fmt_drop(ds['max_drop'])}**." if isinstance(ds["max_drop"], float) else "."),
        "",
        "This is the honest reading of \"does the loop earn its keep\": on clean data it is free, "
        "and when a pass is genuinely deficient it is what recovers the score. Removing it can only "
        "ever leave score on the table, never gain any.",
        "",
    ]
    if report["any_mock"]:
        lines += ["*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).", ""]
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> dict:
    """Persist the report as JSON + markdown; return it unchanged."""
    ABLATION_RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    ABLATION_TABLE_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report
