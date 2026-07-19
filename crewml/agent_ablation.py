"""Day 14 — per-agent ablations: what do the Planner and the Feature Engineer earn?

Day 13 asked whether the Critic *loop* earns its keep and could answer by deleting it
outright — the loop is optional topology. The Planner and the Feature Engineer are not:
the Trainer cannot run without a plan or an ``add_features``, so "remove the agent" has
to mean **replace the specialist with its naive floor** and measure what the holdout
score does. That is what the two Day-14 variants are (see :mod:`crewml.crew.graph`):

* ``no_planner`` — the Planner node body is swapped for
  :func:`crewml.crew.planner.build_naive_plan`: a profile-blind plan with no leakage
  drops, cardinality-blind one-hot, no imbalance strategy, a single library-default
  RandomForest with no search, and no response to critiques. The Critic loop edge
  still exists — but it points at a stand-in that rebuilds the same plan, so the
  variant also measures that the Critic's instructions have **no actuator** without
  a Planner.
* ``no_feature_engineer`` — the FE node body is swapped for the identity transform
  (:data:`crewml.crew.feature_engineer.IDENTITY_FE_SOURCE`): the model trains on raw
  features only, with no generated or default engineered columns.

Every arm — including a **fresh** ``full`` reference run, so the three are paired
under identical same-day conditions — goes through :func:`crewml.ablation.run_variant`:
the same recursion budget, the same post-run holdout-seal check, and the same
:func:`crewml.holdout_eval.score_on_holdout` call as Day 12/13. The attribution for
agent X is then simply ``full − no_X`` on the primary metric (higher-is-better across
the suite), computed only when both arms shipped a real number.

Honesty notes, same discipline as Day 13:

* A **negative** drop (the naive floor beating the specialist) is reported as-is —
  that is the point of running the ablation rather than asserting the architecture.
* The eval protocol itself is held constant across arms: the naive plan still carries
  the protocol's positive class (a property of the task, not a planning decision), so
  no drop can be an artifact of a flipped ROC-AUC orientation.
* Mock runs are flagged and never presented as real (EVAL_PROTOCOL.md §5).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from crewml.ablation import run_variant
from crewml.config import MAX_ITERATIONS, RESULTS_DIR
from crewml.crew.graph import VARIANTS
from crewml.datasets import REGISTRY, load_manifest

AGENT_ABLATION_SCHEMA_VERSION = 1
AGENT_ABLATION_RESULT_PATH = RESULTS_DIR / "day14_agent_ablation.json"
AGENT_ABLATION_TABLE_MD_PATH = RESULTS_DIR / "day14_agent_ablation.md"

ARTIFACT_PREFIX = "day14"

# The reference arm and the two ablated arms this study compares. Order matters for
# rendering; every name must be a variant the graph builder knows.
FULL = "full"
ABLATED_ARMS: tuple[str, ...] = ("no_planner", "no_feature_engineer")
ARMS: tuple[str, ...] = (FULL,) + ABLATED_ARMS
assert all(a in VARIANTS for a in ARMS)

# arm -> the agent whose contribution the arm isolates (rendering + summary keys).
AGENT_FOR_ARM = {"no_planner": "planner", "no_feature_engineer": "feature_engineer"}


def run_dataset(
    key: str,
    manifest: dict,
    *,
    max_iterations: int = MAX_ITERATIONS,
    progress=None,
) -> dict[str, Any]:
    """Run all three arms on one dataset and attribute the drops.

    ``drops[agent]`` is ``full − no_<agent>`` on the primary metric — positive means
    the specialist *added* held-out score over its naive floor; ``None`` whenever
    either arm failed to ship a model (never a flattering zero).
    """
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        if progress:
            progress(f"  [day14] {key}: running arm {arm!r} ...")
        arms[arm] = run_variant(
            key, arm, manifest,
            max_iterations=max_iterations,
            artifact_prefix=ARTIFACT_PREFIX,
        )

    full_v = arms[FULL]["value"] if arms[FULL].get("ok") else None
    drops: dict[str, Optional[float]] = {}
    for arm, agent in AGENT_FOR_ARM.items():
        av = arms[arm]["value"] if arms[arm].get("ok") else None
        drops[agent] = round(full_v - av, 6) if (full_v is not None and av is not None) else None

    return {"metric": REGISTRY[key].metric, "arms": arms, "drops": drops}


def run_agent_ablation(
    keys: Iterable[str],
    *,
    max_iterations: int = MAX_ITERATIONS,
    progress=None,
) -> dict[str, dict]:
    """The full study: every dataset through all three arms. Returns {key: result}."""
    out: dict[str, dict] = {}
    manifest = load_manifest()
    for k in keys:
        if progress:
            progress(f"[day14] agent ablation === {k} ===")
        out[k] = run_dataset(k, manifest, max_iterations=max_iterations, progress=progress)
        if progress:
            progress("  " + summarise_dataset(k, out[k]))
    return out


def summarise_dataset(key: str, result: dict) -> str:
    """A one-line human summary of one dataset's three arms."""
    def _v(rec):
        return f"{rec['value']:.4f}" if rec.get("ok") else f"FAILED({rec.get('error')})"

    def _d(x):
        return f"{x:+.4f}" if isinstance(x, float) else "—"

    a = result["arms"]
    d = result["drops"]
    return (
        f"{key:<10} {result['metric']}: full={_v(a[FULL])} "
        f"no_planner={_v(a['no_planner'])} (planner {_d(d['planner'])}) "
        f"no_fe={_v(a['no_feature_engineer'])} (fe {_d(d['feature_engineer'])})"
    )


def _agent_summary(study: dict[str, dict], agent: str) -> dict[str, Any]:
    """Aggregate one agent's per-dataset drops into headline numbers."""
    drops = {k: r["drops"].get(agent) for k, r in study.items()}
    real = [d for d in drops.values() if isinstance(d, float)]
    return {
        "datasets": len(study),
        "compared": len(real),
        "helped_count": len([d for d in real if d > 1e-9]),
        "hurt_count": len([d for d in real if d < -1e-9]),
        "mean_drop": round(sum(real) / len(real), 6) if real else None,
        "max_drop": round(max(real), 6) if real else None,
        "min_drop": round(min(real), 6) if real else None,
        "best_dataset": max((k for k, d in drops.items() if isinstance(d, float)),
                            key=lambda k: drops[k], default=None),
    }


def assemble_report(study: dict[str, dict]) -> dict[str, Any]:
    """Bundle the study + per-agent summaries into the committed report object."""
    any_mock = any(rec.get("mock") for r in study.values() for rec in r["arms"].values())
    return {
        "schema_version": AGENT_ABLATION_SCHEMA_VERSION,
        "day": 14,
        "phase": 3,
        "study": "agent_ablation",
        "arms": list(ARMS),
        "any_mock": any_mock,
        "results": study,
        "summary": {agent: _agent_summary(study, agent) for agent in AGENT_FOR_ARM.values()},
    }


# --- Rendering ---------------------------------------------------------------

def _fmt(rec: dict) -> str:
    return f"{rec['value']:.4f}" if rec.get("ok") else "—"


def _fmt_drop(drop: Optional[float]) -> str:
    return "—" if not isinstance(drop, float) else f"{drop:+.4f}"


def render_markdown(report: dict) -> str:
    """Render the Day-14 attribution board as committed markdown."""
    study = report["results"]
    lines = [
        "# Day 14 — per-agent ablations: Planner and Feature Engineer",
        "",
        "*Each ablated arm replaces exactly one specialist with its naive floor — the "
        "Planner with a profile-blind default plan (no leakage drops, no cardinality or "
        "imbalance awareness, one library-default RandomForest, critique-deaf), the "
        "Feature Engineer with the identity transform (raw features only). Topology, "
        "seed, LLM settings and holdout scoring are identical across arms; the `full` "
        "reference was re-run in the same session so all three are paired. "
        "`drop` = full − ablated on the primary metric (higher-is-better): positive "
        "means the specialist added held-out score.*",
        "",
        "| Dataset | Metric | Full crew | No-Planner | Planner drop | No-FE | FE drop | Full model | Naive model | FE cols (full) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, r in study.items():
        a, d = r["arms"], r["drops"]
        lines.append(
            f"| {key} | {r['metric']} | {_fmt(a[FULL])} | {_fmt(a['no_planner'])} | "
            f"{_fmt_drop(d['planner'])} | {_fmt(a['no_feature_engineer'])} | "
            f"{_fmt_drop(d['feature_engineer'])} | {a[FULL].get('best_model') or '—'} | "
            f"{a['no_planner'].get('best_model') or '—'} | {a[FULL].get('n_engineered', '—')} |"
        )

    for agent, label in (("planner", "Planner"), ("feature_engineer", "Feature Engineer")):
        s = report["summary"][agent]
        lines += [
            "",
            f"**{label}** — compared on {s['compared']}/{s['datasets']} dataset(s): "
            f"helped on {s['helped_count']}, hurt on {s['hurt_count']}; "
            f"mean drop **{_fmt_drop(s['mean_drop'])}**, "
            f"range {_fmt_drop(s['min_drop'])} … {_fmt_drop(s['max_drop'])}"
            + (f" (largest on `{s['best_dataset']}`)." if s.get("best_dataset") else "."),
        ]

    lines += [
        "",
        "A negative drop means the naive floor beat the specialist on that dataset and "
        "is reported as-is — the ablation exists to *measure* the architecture, not to "
        "flatter it. In the `no_planner` arm the Critic loop still exists but points at "
        "a critique-deaf stand-in, so any iterate decision there changes nothing: without "
        "the Planner, the Critic's instructions have no actuator.",
        "",
    ]
    if report["any_mock"]:
        lines += ["*(mock)* — a run without a live LLM key; never a headline result (EVAL_PROTOCOL.md §5).", ""]
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> dict:
    """Persist the report as JSON + markdown; return it unchanged."""
    AGENT_ABLATION_RESULT_PATH.write_text(json.dumps(report, indent=2, default=str))
    AGENT_ABLATION_TABLE_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return report
