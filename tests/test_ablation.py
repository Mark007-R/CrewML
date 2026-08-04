"""Day 13 guards: the Critic-loop ablation is structural, honest, and correctly scored.

The ablation's whole validity rests on two properties this file pins down:

  * the ``no_critic`` variant is the full crew with the Critic node and its loop edge
    removed — nothing else — so a score difference is attributable to the loop alone;
  * the first-pass handicap that makes the loop's value observable is OFF by default and,
    when on, cripples only the first pass (recovery on later passes is the loop's job);
  * a "loop drop" is only computed between two real held-out numbers, and the aggregate
    summaries count fires/helps honestly (never a flattering zero for a failed run).

The end-to-end ``no_critic`` run is kept offline + fast (no LLM, no grid search), mirroring
``test_graph.py``'s fixture; the drop arithmetic and rendering are pure and tested in memory.
"""
from __future__ import annotations

import pytest

from crewml import ablation
from crewml.crew import VARIANTS, build_crew, build_graph, initial_state
from crewml.crew.graph import _NO_CRITIC_NODES
from crewml.crew.planner import build_plan
from crewml.crew.profiler import run_profiler
from crewml.datasets import REGISTRY

# Full-crew / model-fit module: minute-scale by nature (Day 28 speed lanes).
pytestmark = pytest.mark.slow

SPEC = REGISTRY["credit-g"]


# --- Variant topology -------------------------------------------------------

def test_variants_declared():
    # Day 13 shipped no_critic; Day 14 added the two per-agent removals.
    assert VARIANTS == ("full", "no_critic", "no_planner", "no_feature_engineer")


def test_no_critic_graph_drops_only_the_critic():
    full = set(build_graph("full").nodes)
    nc = set(build_graph("no_critic").nodes)
    # Exactly the Critic is gone; every other specialist remains.
    assert full - nc == {"critic"}
    assert "critic" not in _NO_CRITIC_NODES
    assert set(_NO_CRITIC_NODES) == nc


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        build_graph("bogus")
    with pytest.raises(ValueError):
        build_crew(variant="bogus")


# --- The first-pass handicap: off by default, first-pass only ---------------

@pytest.fixture(scope="module")
def profile():
    return run_profiler("kin8nm")


def test_handicap_off_by_default(profile, monkeypatch):
    monkeypatch.delenv("CREWML_ABLATION_HANDICAP", raising=False)
    plan = build_plan(profile, iteration=0)
    assert "ablation_handicap" not in plan
    gb = plan["candidate_models"][0]["param_grid"]
    assert gb["model__max_iter"] != [1]  # real capacity, untouched


def test_handicap_cripples_first_pass_when_enabled(profile, monkeypatch):
    monkeypatch.setenv("CREWML_ABLATION_HANDICAP", "1")
    plan = build_plan(profile, iteration=0)
    assert "ablation_handicap" in plan
    for m in plan["candidate_models"]:
        g = m["param_grid"]
        if "model__max_iter" in g:
            assert g["model__max_iter"] == [1]
        if "model__min_samples_leaf" in g:
            assert g["model__min_samples_leaf"] == [10_000_000]


def test_handicap_leaves_later_passes_at_full_capacity(profile, monkeypatch):
    # Recovery is the loop's job — the handicap must not touch iteration > 0.
    monkeypatch.setenv("CREWML_ABLATION_HANDICAP", "1")
    plan = build_plan(profile, iteration=1)
    assert "ablation_handicap" not in plan
    gb = plan["candidate_models"][0]["param_grid"]
    assert gb["model__max_iter"] != [1]


# --- Drop arithmetic + honest aggregation (pure, in-memory) -----------------

def _rec(value, *, ok=True, iters=1, mock=False):
    return {"ok": ok, "value": value, "iterations_run": iters, "mock": mock,
            "loop_fired": iters > 1, "error": None if ok else "boom"}


def _pair_obj(metric, looped_v, nc_v, *, looped_iters=1, looped_ok=True, nc_ok=True):
    lp = _rec(looped_v, ok=looped_ok, iters=looped_iters)
    nc = _rec(nc_v, ok=nc_ok, iters=1)
    lv = looped_v if looped_ok else None
    nv = nc_v if nc_ok else None
    drop = round(lv - nv, 6) if (lv is not None and nv is not None) else None
    return {"metric": metric, "looped": lp, "no_critic": nc, "loop_drop": drop,
            "loop_fired": looped_iters > 1, "iterations_looped": looped_iters}


def test_summary_counts_fires_and_helps():
    study = {
        "clean": _pair_obj("r2", 0.80, 0.80, looped_iters=1),           # loop didn't fire
        "recovered": _pair_obj("r2", 0.79, 0.02, looped_iters=2),        # fired + helped a lot
    }
    s = ablation._study_summary(study)
    assert s["datasets"] == 2 and s["compared"] == 2
    assert s["loop_fired_count"] == 1 and s["loop_fired_datasets"] == ["recovered"]
    assert s["loop_helped_count"] == 1
    assert s["max_drop"] == pytest.approx(0.77)
    assert s["min_drop"] == pytest.approx(0.0)


def test_drop_is_none_when_a_variant_failed():
    study = {"broken": _pair_obj("r2", 0.5, 0.0, nc_ok=False)}
    assert study["broken"]["loop_drop"] is None
    s = ablation._study_summary(study)
    # A failed variant contributes no drop — never a flattering zero.
    assert s["compared"] == 0 and s["mean_drop"] is None


def test_assemble_and_render_cover_both_studies():
    natural = {"diabetes": _pair_obj("roc_auc", 0.815, 0.815, looped_iters=1)}
    probe = {"kin8nm": _pair_obj("r2", 0.79, 0.02, looped_iters=2)}
    report = ablation.assemble_report(natural, probe)
    assert report["study"] == "critic_loop_ablation"
    assert report["summary"]["natural"]["loop_fired_count"] == 0
    assert report["summary"]["deficiency_probe"]["loop_fired_count"] == 1
    md = ablation.render_markdown(report)
    assert "Study 1" in md and "Study 2" in md
    assert "loop drop" in md.lower()


def test_any_mock_flag_propagates():
    natural = {"d": {"metric": "r2", "looped": _rec(0.5, mock=True), "no_critic": _rec(0.5),
                     "loop_drop": 0.0, "loop_fired": False, "iterations_looped": 1}}
    report = ablation.assemble_report(natural, {})
    assert report["any_mock"] is True


# --- End-to-end: the no_critic variant runs one pass, no Critic -------------

@pytest.fixture(scope="module")
def no_critic_final():
    mp = pytest.MonkeyPatch()
    for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
        mp.setenv(var, "0")
    mp.setenv("CREWML_TRAINER_PARAM_SEARCH", "0")
    app = build_crew(variant="no_critic")
    final = app.invoke(initial_state(SPEC, max_iterations=3), config={"recursion_limit": 50})
    mp.undo()
    return final


def test_no_critic_run_never_visits_the_critic(no_critic_final):
    assert "critic" not in no_critic_final["trace"]
    assert no_critic_final["trace"] == [
        "profiler", "planner", "feature_engineer", "trainer", "ensembler", "reporter",
    ]


def test_no_critic_run_makes_no_loop_and_still_ships(no_critic_final):
    # No Critic => iteration counter never increments, no critiques recorded...
    assert no_critic_final["iteration"] == 0
    assert (no_critic_final.get("critiques") or []) == []
    # ...but the crew still finalises a real model + report.
    assert no_critic_final["report"] is not None
    assert no_critic_final["ensemble"] is not None
