"""The derived-artifact registry: every committed file that is *generated*, and how.

Day 23 (reproducibility) deliverable, landed early on Day 20 because a Day-20 audit
found **four** committed artifacts that silently disagreed with the data they claimed
to present:

* ``results/charts/day12_*.png`` — were TEST-FIXTURE output. A test called
  ``render_all(table)`` with no ``out_dir``, so it wrote its fixture numbers over the
  committed board (a flat 0.500/0.700/0.850/0.900, no solo-agent bar, and both AutoML
  losses painted green) — and those bytes were what shipped.
* ``results/day20_self_repair.md`` — said "2/2 = 100%" while its own JSON, the chart
  and the report all said 18/18, because ``--table-only`` refreshed the chart but
  never rewrote the table.
* ``results/charts/day16_provider_study.png`` — captioned "no live-provider arm ran
  this session" over the live Groq arm, because the caption was fixed in code and the
  PNG was never re-rendered.
* ``results/sample_model_card.md`` — still described an unhardened executor and
  self-repair as future work, after both had shipped.

Every one of those is the same defect: **a committed artifact drifting from its own
generator.** Vigilance does not catch that class; a check does. This module states the
derivation for each artifact, and :mod:`tests.test_artifact_reproducibility` re-derives
them and fails on any mismatch.

The comparison is exact — byte-for-byte for charts, string-for-string for markdown.
That is only sound because both are deterministic here: matplotlib's PNG writer emits
no timestamp, so re-rendering identical data on the same library versions reproduces
identical bytes (verified). If a library upgrade ever breaks that, the right response
is to regenerate and commit the new bytes, not to loosen the check into uselessness.

**What this deliberately does NOT cover.** ``sample_model_card.md`` cannot be
re-derived: the committed Day-11 run record is a trimmed summary that no longer holds
the per-model rows and token counts the card reports, so regenerating it would replace
real measured numbers with em-dashes. Its *boilerplate* is checked against the live
Reporter template instead — which is exactly the part that went stale. Hand-authored
records (``dataset_manifest.json``, the raw per-day capture files) are not derived and
are out of scope by design; they are covered by the holdout seal and their own tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from crewml.config import RESULTS_DIR


@dataclass(frozen=True)
class DerivedArtifact:
    """One committed file, plus everything needed to re-derive it."""

    path: Path                      # the committed artifact
    source: Path                    # the JSON it is rendered from
    kind: str                       # "markdown" | "chart"
    render: Callable[[Any], Any]    # source-dict -> str (markdown) or writes (chart)
    why: str                        # what drifted here before, or what it guards

    @property
    def name(self) -> str:
        return str(self.path.relative_to(RESULTS_DIR)).replace("\\", "/")


def _markdown_artifacts() -> list[DerivedArtifact]:
    from crewml import (
        ablation, agent_ablation, budget_study, comparison, failure_taxonomy,
        iteration_depth, leaderboard, phase3_results, provider_study,
        self_repair_study,
    )

    specs = [
        (leaderboard.TABLE_MD_PATH, RESULTS_DIR / "baselines_table.json",
         leaderboard.render_markdown, "Day 4 baselines board"),
        (comparison.TABLE_MD_PATH, RESULTS_DIR / "comparison_table.json",
         comparison.render_markdown, "Day 12 crew-vs-rivals board"),
        (ablation.ABLATION_TABLE_MD_PATH, RESULTS_DIR / "day13_critic_ablation.json",
         ablation.render_markdown, "Day 13 Critic ablation"),
        (agent_ablation.AGENT_ABLATION_TABLE_MD_PATH,
         RESULTS_DIR / "day14_agent_ablation.json",
         agent_ablation.render_markdown, "Day 14 per-agent ablations"),
        (iteration_depth.ITERATION_DEPTH_TABLE_MD_PATH,
         RESULTS_DIR / "day15_iteration_depth.json",
         iteration_depth.render_markdown, "Day 15 iteration-depth study"),
        (provider_study.PROVIDER_STUDY_TABLE_MD_PATH,
         RESULTS_DIR / "day16_provider_study.json",
         provider_study.render_markdown, "Day 16 provider study"),
        (failure_taxonomy.TAXONOMY_TABLE_MD_PATH,
         RESULTS_DIR / "day17_failure_taxonomy.json",
         failure_taxonomy.render_markdown, "Day 17 failure taxonomy"),
        (phase3_results.PHASE3_RESULTS_MD_PATH, RESULTS_DIR / "phase3_results.json",
         phase3_results.render_markdown, "Day 18 consolidated Phase-3 results"),
        (self_repair_study.SELF_REPAIR_TABLE_MD_PATH,
         RESULTS_DIR / "day20_self_repair.json",
         self_repair_study.render_table_md,
         "Day 20 self-repair table — shipped as '2/2' against an 18/18 JSON"),
        (budget_study.BUDGET_STUDY_MD_PATH,
         RESULTS_DIR / "day21_budget_study.json",
         budget_study.render_markdown, "Day 21 run-budget study"),
    ]
    return [
        DerivedArtifact(path=p, source=s, kind="markdown", render=r, why=w)
        for p, s, r, w in specs
    ]


def _chart_artifacts() -> list[DerivedArtifact]:
    from crewml import charts

    specs = [
        (charts.SCORES_CHART_PATH, RESULTS_DIR / "comparison_table.json",
         charts.plot_holdout_scores,
         "Day 12 scores — shipped as TEST-FIXTURE output for weeks"),
        (charts.DELTAS_CHART_PATH, RESULTS_DIR / "comparison_table.json",
         charts.plot_crew_deltas,
         "Day 12 deltas — fixture output hid both AutoML losses"),
        (charts.CRITIC_ABLATION_CHART_PATH, RESULTS_DIR / "day13_critic_ablation.json",
         charts.plot_critic_ablation, "Day 13 chart"),
        (charts.AGENT_ABLATION_CHART_PATH, RESULTS_DIR / "day14_agent_ablation.json",
         charts.plot_agent_ablation, "Day 14 chart"),
        (charts.ITERATION_DEPTH_CHART_PATH, RESULTS_DIR / "day15_iteration_depth.json",
         charts.plot_iteration_depth, "Day 15 chart"),
        (charts.PROVIDER_STUDY_CHART_PATH, RESULTS_DIR / "day16_provider_study.json",
         charts.plot_provider_study,
         "Day 16 chart — captioned 'no live arm ran' over the live Groq arm"),
        (charts.FAILURE_TAXONOMY_CHART_PATH, RESULTS_DIR / "day17_failure_taxonomy.json",
         charts.plot_failure_taxonomy, "Day 17 chart"),
        (charts.PHASE3_SUMMARY_CHART_PATH, RESULTS_DIR / "phase3_results.json",
         charts.plot_phase3_summary, "Day 18 chart"),
        (charts.SELF_REPAIR_CHART_PATH, RESULTS_DIR / "day20_self_repair.json",
         charts.plot_self_repair, "Day 20 chart"),
    ]
    return [
        DerivedArtifact(path=p, source=s, kind="chart", render=r, why=w)
        for p, s, r, w in specs
    ]


def derived_artifacts() -> list[DerivedArtifact]:
    """Every committed artifact that can be re-derived from a committed source."""
    return _markdown_artifacts() + _chart_artifacts()


# Committed generated files that are deliberately NOT re-derivable, with the reason.
# Listed explicitly so "not checked" is a recorded decision rather than an oversight.
NOT_REDERIVABLE: dict[str, str] = {
    "sample_model_card.md": (
        "the committed Day-11 run record is a trimmed summary and no longer carries the "
        "per-model rows / token counts the card reports, so a re-render would replace "
        "real measured numbers with em-dashes. Its boilerplate is checked against the "
        "live Reporter template instead — the part that actually went stale."
    ),
}
