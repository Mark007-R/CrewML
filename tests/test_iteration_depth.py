"""Day 15 guards: the iteration-depth study measures honestly and prices no noise.

The study's claims rest on the properties pinned here:

  * ``extract_run_facts`` reads cost + loop-outcome facts correctly — in particular
    ``budget_bound`` is True exactly when the final Critic pass finalised on its
    "budget reached" rule (cut off), never when the run finalised clean (done);
  * curve analysis computes lifts only between two real numbers (a failed point is
    ``None`` all the way down, never a flattering zero) and prices a cost ratio only
    when the marginal lift is real — a ~zero lift never divides into an impressive
    seconds-per-point figure;
  * the per-curve summary finds the saturation depth and splits the first loop's lift
    from everything after it — the diminishing-returns headline is computed, not typed;
  * budgets below 1 are rejected (budget 1 = looping structurally impossible is the
    floor of the sweep, not budget 0);
  * rendering and the chart survive the real report shapes, including the mock caveat.

Everything here is pure/in-memory except the offline end-to-end check, which mirrors
``test_ablation.py``'s cost profile (no LLM, no grid search, one small dataset): a
budget-1 handicapped run must ship the stump AND self-report as budget-bound — the
starvation case the whole study turns on.
"""
from __future__ import annotations

import pytest

from crewml import iteration_depth as itd
from crewml.ablation import extract_run_facts


# --- extract_run_facts: budget_bound and token accounting (pure) --------------

def _final_state(*, decision="finalize", reason="", tokens=(100, 20), critiques=True):
    state = {
        "report": {"llm_usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "n_live": 2}},
        "critiques": [],
    }
    if critiques:
        state["critiques"] = [
            {"decision": "iterate", "reason": "actionable issue(s) [underfit] under budget",
             "finding_codes": ["underfit"]},
            {"decision": decision, "reason": reason, "finding_codes": ["underfit"]},
        ]
    return state


def test_budget_bound_true_only_on_the_budget_reached_finalise():
    cut_off = _final_state(reason="iteration budget reached (2/2) — finalising")
    done = _final_state(reason="no actionable failure modes found — the run is clean, finalising")
    assert extract_run_facts(cut_off)["budget_bound"] is True
    assert extract_run_facts(done)["budget_bound"] is False


def test_budget_bound_false_with_no_critiques_at_all():
    # The no_critic variant has no Critic passes — it can never read as budget-bound.
    facts = extract_run_facts(_final_state(critiques=False))
    assert facts["budget_bound"] is False
    assert facts["final_reason"] is None
    assert facts["final_finding_codes"] == []


def test_token_totals_come_from_the_reporter_aggregation():
    facts = extract_run_facts(_final_state(tokens=(1234, 56)))
    assert facts["llm_prompt_tokens"] == 1234
    assert facts["llm_completion_tokens"] == 56
    assert facts["llm_narratives_live"] == 2


# --- Curve analysis: honest lifts, no noise pricing (pure) --------------------

def _rec(value, *, ok=True, passes=1, bound=False, seconds=100.0, tokens=(1000, 100)):
    return {
        "ok": ok, "value": value if ok else None, "error": None if ok else "boom",
        "metric": "r2", "mock": False,
        "iterations_run": passes, "budget_bound": bound, "crew_seconds": seconds,
        "llm_prompt_tokens": tokens[0], "llm_completion_tokens": tokens[1],
    }


def test_analyse_curve_marginals_and_cost_per_point():
    curve = {
        1: _rec(0.02, passes=1, bound=True, seconds=60.0, tokens=(1000, 100)),
        2: _rec(0.82, passes=2, seconds=140.0, tokens=(2200, 220)),
        3: _rec(0.82, passes=2, seconds=141.0, tokens=(2200, 220)),
    }
    rows = itd.analyse_curve(curve)
    assert [r["depth"] for r in rows] == [1, 2, 3]
    assert rows[0]["marginal_lift"] is None                  # nothing below budget 1
    assert rows[1]["marginal_lift"] == pytest.approx(0.80)
    assert rows[1]["lift_vs_min_budget"] == pytest.approx(0.80)
    # 80 extra seconds for 80 points of lift -> 1.0 s per point.
    assert rows[1]["seconds_per_point"] == pytest.approx(1.0)
    assert rows[1]["marginal_tokens"] == 1320
    assert rows[1]["tokens_per_point"] == pytest.approx(1320 / 80)
    assert rows[0]["loops_used"] == 0 and rows[1]["loops_used"] == 1


def test_analyse_curve_never_prices_a_noise_lift():
    curve = {1: _rec(0.80, seconds=100.0), 2: _rec(0.80005, seconds=200.0)}
    row = itd.analyse_curve(curve)[1]
    # The lift is under LIFT_EPS: the 100 extra seconds bought nothing, and the study
    # must say "None", not "0.05 points for 100s".
    assert row["seconds_per_point"] is None
    assert row["tokens_per_point"] is None


def test_analyse_curve_failed_point_is_none_never_zero():
    curve = {1: _rec(None, ok=False), 2: _rec(0.82)}
    rows = itd.analyse_curve(curve)
    assert rows[0]["value"] is None
    assert rows[1]["marginal_lift"] is None          # no lift against a failed run
    assert rows[1]["lift_vs_min_budget"] is None
    assert rows[1]["seconds_per_point"] is None


def test_analyse_curve_accepts_json_string_keys():
    # A --table-only re-render reads depths back as JSON string keys.
    curve = {"1": _rec(0.02), "2": _rec(0.82)}
    rows = itd.analyse_curve(curve)
    assert [r["depth"] for r in rows] == [1, 2]
    assert rows[1]["marginal_lift"] == pytest.approx(0.80)


def test_curve_summary_saturation_and_the_first_loop_split():
    curve = {
        1: _rec(0.02, bound=True),
        2: _rec(0.82, passes=2),
        3: _rec(0.82, passes=2),
        4: _rec(0.82, passes=2),
    }
    s = itd._curve_summary(itd.analyse_curve(curve))
    assert s["saturation_depth"] == 2                # first budget within eps of the best
    assert s["first_loop_lift"] == pytest.approx(0.80)
    assert s["lift_beyond_first_loop"] == pytest.approx(0.0)
    assert s["budget_bound_depths"] == [1]
    assert s["max_passes_used"] == 2


def test_run_depth_curve_rejects_budget_below_one():
    with pytest.raises(ValueError, match="must be >= 1"):
        itd.run_depth_curve("kin8nm", {"datasets": {}}, [0], handicap=False)


# --- Report assembly + rendering (pure) ---------------------------------------

def _study(mock=False):
    probe_curve = {
        1: _rec(0.02, bound=True, seconds=60.0),
        2: _rec(0.82, passes=2, seconds=140.0),
        3: _rec(0.82, passes=2, seconds=141.0),
    }
    natural_curve = {1: _rec(0.975), 3: _rec(0.975)}
    if mock:
        natural_curve[1] = {**natural_curve[1], "mock": True}
    return {"natural": {"cpu_small": natural_curve}, "deficiency_probe": {"kin8nm": probe_curve}}


def test_assemble_report_shapes_and_mock_flag():
    report = itd.assemble_report(_study())
    assert report["day"] == 15 and report["study"] == "iteration_depth"
    assert report["any_mock"] is False
    assert report["summary"]["deficiency_probe"]["kin8nm"]["saturation_depth"] == 2
    assert itd.assemble_report(_study(mock=True))["any_mock"] is True


def test_render_markdown_carries_the_load_bearing_facts():
    md = itd.render_markdown(itd.assemble_report(_study()))
    assert "budget-bound" in md            # starvation is named, not hidden
    assert "+0.8000" in md                 # the first loop's lift, printed signed
    assert "cliff" in md                   # the headline is present
    assert "(mock)" not in md
    md_mock = itd.render_markdown(itd.assemble_report(_study(mock=True)))
    assert "(mock)" in md_mock             # a mock run can never pass as real


def test_chart_renders_from_the_report(tmp_path):
    from crewml import charts
    out = charts.plot_iteration_depth(itd.assemble_report(_study()), path=tmp_path / "d15.png")
    assert out.exists() and out.stat().st_size > 0


# --- Offline end-to-end: budget 1 under the handicap ships the stump ----------

def test_budget_one_handicapped_run_is_budget_bound(monkeypatch):
    for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
        monkeypatch.setenv(var, "0")
    # Param search must stay ON: the handicap lives in the grid values, and with search
    # disabled the Trainer fits library defaults and the stump never happens (the grids
    # are all singletons under the handicap, so pass 1 stays cheap regardless).
    monkeypatch.setenv("CREWML_TRAINER_PARAM_SEARCH", "1")
    monkeypatch.setenv("CREWML_ABLATION_HANDICAP", "1")

    from crewml.crew import build_crew, initial_state
    from crewml.datasets import REGISTRY

    app = build_crew(variant="full")
    final = app.invoke(initial_state(REGISTRY["kin8nm"], max_iterations=1),
                       config={"recursion_limit": 50})
    facts = extract_run_facts(final)
    assert final["iteration"] == 1                       # the guard bit: no loop happened
    assert facts["budget_bound"] is True                 # ...and the run says so honestly
    assert "underfit" in facts["final_finding_codes"]    # because work was left undone
    assert final["plan"].get("ablation_handicap")        # the instrumentation is labelled
