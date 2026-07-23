"""Day 18 guards: the consolidated section cannot drift from, or flatter, the studies.

The consolidation is pure reshaping, so the failure modes worth testing are the lying
ones: inventing a number a study never committed, dropping a study silently, turning a
missing value into a zero, rendering a loss as a win, or letting the phase's negative
results (the live fatal, the missed leak probe) fall out of the summary.
"""
from __future__ import annotations

import pytest

from crewml.charts import plot_phase3_summary
from crewml.phase3_results import (
    STUDY_PATHS,
    assemble_results,
    load_phase3_bundle,
    render_markdown,
)


# ---------------------------------------------------------------------------
# Synthetic bundle — minimal but shape-faithful to the committed files
# ---------------------------------------------------------------------------

def _bundle(solo_missing_on=("vehicle",), crew_loses_to_automl_on=("kin8nm",)):
    datasets = ["credit-g", "vehicle", "kin8nm"]
    rows, deltas = {}, {}
    for key in datasets:
        solo = None if key in solo_missing_on else 0.80
        crew = 0.90
        automl = 0.95 if key in crew_loses_to_automl_on else 0.85
        rows[key] = {
            "metric": "roc_auc",
            "systems": {
                "dummy": {"value": 0.5, "mock": False},
                "default_rf": {"value": 0.7, "mock": False},
                "solo_agent": ({"value": solo, "mock": False} if solo else {}),
                "automl_flaml": {"value": automl, "mock": False},
                "crew": {"value": crew, "mock": False},
            },
        }
        deltas[key] = {
            "vs_solo": (crew - solo) if solo is not None else None,
            "vs_automl": crew - automl,
            "vs_default_rf": crew - 0.7,
        }
    comparison = {
        "rows": rows,
        "deltas": deltas,
        "wins": {"vs_solo": {"won": 2, "compared": 2},
                 "vs_automl": {"won": 2, "compared": 3},
                 "vs_default_rf": {"won": 3, "compared": 3}},
        "any_mock": False,
    }
    critic = {
        "deficiency_probe": {"kin8nm": {"loop_drop": 0.82}},
        "summary": {
            "natural": {"datasets": 3, "loop_fired_count": 0, "mean_drop": 0.0},
            "deficiency_probe": {"datasets": 1, "loop_fired_count": 1,
                                 "mean_drop": 0.82, "max_drop": 0.82},
        },
    }
    agent = {
        "results": {k: {"drops": {"planner": 0.02, "feature_engineer": 0.01}}
                    for k in datasets},
        "summary": {
            "planner": {"compared": 3, "helped_count": 3, "hurt_count": 0,
                        "mean_drop": 0.02, "max_drop": 0.05, "best_dataset": "kin8nm"},
            "feature_engineer": {"compared": 3, "helped_count": 1, "hurt_count": 0,
                                 "mean_drop": 0.01, "max_drop": 0.01,
                                 "best_dataset": "credit-g"},
        },
    }
    depth = {
        "results": {"deficiency_probe": {
            "kin8nm": {"1": {"value": 0.004}, "2": {"value": 0.83},
                       "3": {"value": 0.83}},
        }},
        "summary": {
            "natural": {k: {"score_spread": 0.0} for k in datasets},
            "deficiency_probe": {"kin8nm": {"first_loop_lift": 0.82,
                                            "lift_beyond_first_loop": 0.0,
                                            "saturation_depth": 2,
                                            "budget_bound_depths": [1]}},
        },
    }
    provider = {
        "pricing_as_of": "2026-07-21",
        "providers": {"groq": {"label": "Groq", "usd_per_mtok_in": 0.59,
                               "usd_per_mtok_out": 0.79},
                      "mock": {"label": "Mock", "usd_per_mtok_in": 0.0,
                               "usd_per_mtok_out": 0.0}},
        "live_arms_run": ["groq"],
        "blocked_providers": {"anthropic": "not_configured"},
        "arms": {
            "groq": {
                "credit-g": {"ok": True, "value": 0.79, "mock": False,
                             "llm_prompt_tokens": 1000, "llm_completion_tokens": 500,
                             "llm_narratives_live": 4},
                "vehicle": {"ok": False, "value": None, "mock": False,
                            "llm_prompt_tokens": 900, "llm_completion_tokens": 400,
                            "llm_narratives_live": 4},
            },
            "mock": {
                "credit-g": {"ok": True, "value": 0.79, "mock": True,
                             "llm_narratives_live": 0},
            },
        },
        "resilience": {"n_compared": 3, "n_equal": 3, "all_equal": True,
                       "max_abs_diff": 0.0},
    }
    taxonomy = {
        "archive_census": {
            "n_crew_runs": 10, "n_solo_runs": 5,
            "summary": {
                "n_events": 12,
                "by_category": {"exec_error": {"total": 1,
                                               "by_outcome": {"fatal": 1}}},
                "by_outcome": {"fatal": 3, "handled": 8, "degraded": 1},
                "fatal_by_system": {"crew": 1, "solo": 2},
            },
        },
        "probes": {
            "live_leak": [
                {"probe": "leak_blatant", "detected": True, "model_saw_leak": False},
                {"probe": "leak_subtle", "detected": False, "model_saw_leak": True},
            ],
            "live_timeout": {"probe": "exec_timeout", "detected": True},
            "record_level": [{"probe": "wrong_metric", "detected": True}],
        },
    }
    return {
        "comparison": comparison,
        "critic_ablation": critic,
        "agent_ablation": agent,
        "iteration_depth": depth,
        "provider_study": provider,
        "failure_taxonomy": taxonomy,
    }


# ---------------------------------------------------------------------------
# Assembly honesty
# ---------------------------------------------------------------------------

def test_missing_study_file_is_a_hard_error(tmp_path):
    paths = dict(STUDY_PATHS)
    paths["provider_study"] = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="provider_study"):
        load_phase3_bundle(paths)


def test_headline_matches_source_wins_verbatim():
    report = assemble_results(_bundle())
    assert report["headline"]["crew_vs_solo"] == {"won": 2, "compared": 2}
    assert report["headline"]["crew_vs_automl"] == {"won": 2, "compared": 3}


def test_missing_solo_score_stays_none_not_zero():
    report = assemble_results(_bundle(solo_missing_on=("vehicle",)))
    row = report["board"]["rows"]["vehicle"]
    assert row["scores"]["solo_agent"] is None
    assert row["deltas"]["vs_solo"] is None


def test_failed_live_arm_has_no_value_but_counts_as_failed():
    report = assemble_results(_bundle())
    groq = report["providers"]["arms"]["groq"]
    assert groq["datasets"]["vehicle"]["value"] is None
    assert groq["n_failed"] == 1


def test_cost_only_from_measured_live_tokens():
    report = assemble_results(_bundle())
    arms = report["providers"]["arms"]
    # mock arm has no live narratives -> no cost, not $0.00
    assert arms["mock"]["total_cost_usd"] is None
    expected = (1000 * 0.59 + 500 * 0.79) / 1e6 + (900 * 0.59 + 400 * 0.79) / 1e6
    assert arms["groq"]["total_cost_usd"] == pytest.approx(expected)


def test_fatal_counts_survive_consolidation():
    report = assemble_results(_bundle())
    assert report["headline"]["fatal_crew"] == 1
    assert report["headline"]["fatal_solo"] == 2


def test_probe_verdicts_include_the_miss():
    report = assemble_results(_bundle())
    verdicts = {p["probe"]: p["detected"] for p in report["failures"]["probes"]}
    assert verdicts["leak_subtle"] is False
    assert verdicts["leak_blatant"] is True


# ---------------------------------------------------------------------------
# Rendering honesty
# ---------------------------------------------------------------------------

def test_markdown_renders_missing_as_em_dash_never_zero():
    md = render_markdown(assemble_results(_bundle()))
    row = next(l for l in md.splitlines() if l.startswith("| vehicle |"))
    assert "—" in row
    assert "0.0000 |" not in row.split("Solo agent")[0]  # no fabricated solo score


def test_markdown_repeats_the_negative_results():
    md = render_markdown(assemble_results(_bundle()))
    assert "MISSED" in md            # the leak_subtle probe verdict
    assert "detection window" in md  # the Day-22 handoff
    assert "Caveats" in md


def test_markdown_cites_every_study_board():
    md = render_markdown(assemble_results(_bundle()))
    for fname in ("comparison_table.md", "day13_critic_ablation.md",
                  "day14_agent_ablation.md", "day15_iteration_depth.md",
                  "day16_provider_study.md", "day17_failure_taxonomy.md"):
        assert fname in md


def test_chart_renders_from_synthetic_report(tmp_path):
    report = assemble_results(_bundle())
    out = plot_phase3_summary(report, path=tmp_path / "summary.png")
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Against the real committed files — the section cannot drift from the studies
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_report():
    if not all(p.exists() for p in STUDY_PATHS.values()):
        pytest.skip("committed study files not present")
    return assemble_results(load_phase3_bundle())


def test_real_headline_is_internally_consistent(real_report):
    h = real_report["headline"]
    for key in ("crew_vs_solo", "crew_vs_automl", "crew_vs_default_rf"):
        assert h[key]["won"] <= h[key]["compared"]
    assert h["fatal_crew"] is not None and h["fatal_solo"] is not None


def test_real_board_scores_match_comparison_table(real_report):
    import json

    from crewml.phase3_results import COMPARISON_TABLE_PATH

    source = json.loads(COMPARISON_TABLE_PATH.read_text())
    for key, row in real_report["board"]["rows"].items():
        for system, value in row["scores"].items():
            assert value == source["rows"][key]["systems"].get(system, {}).get("value")


def test_real_critic_probe_recovery_survives_extraction(real_report):
    # Regression guard: the Day-13 field is ``loop_drop``; a rename there must fail
    # here rather than silently rendering the recovery column as all em dashes.
    recoveries = real_report["critic"]["probe_recovery_by_dataset"]
    assert recoveries and all(v is not None for v in recoveries.values())


def test_real_markdown_renders_all_sections(real_report):
    md = render_markdown(real_report)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in md
