"""Day 12 guards: the comparison board cannot flatter the crew.

The board is the artifact people will actually read, so the tests here target the ways
a results table lies: scoring a failed rival as zero, computing a delta against a number
that does not exist, presenting a mock run as real, or getting the direction of a
comparison backwards.
"""
from __future__ import annotations

import pytest

from crewml.charts import render_all
from crewml.comparison import (
    HEADLINE_DELTAS,
    SYSTEMS,
    _delta,
    assemble_comparison,
    render_markdown,
)
from crewml.datasets import REGISTRY


def _reports(crew_value=0.90, solo_value=0.80, solo_mock=False):
    """Hand-built metrics reports — no disk, no scoring."""
    baseline = {"datasets": {k: {"dummy": {"value": 0.5}, "default_rf": {"value": 0.70}}
                             for k in REGISTRY}}
    solo = {"datasets": {k: {"ok": True, "value": solo_value, "mock": solo_mock}
                         for k in REGISTRY}}
    automl = {"datasets": {k: {"ok": True, "value": 0.85} for k in REGISTRY}}
    crew = {"datasets": {k: {"ok": True, "value": crew_value} for k in REGISTRY}}
    return baseline, solo, automl, crew


def test_board_shapes_every_dataset_and_system():
    table = assemble_comparison(*_reports())
    assert set(table["rows"]) == set(REGISTRY)
    for row in table["rows"].values():
        assert set(row["systems"]) == {name for name, _ in SYSTEMS}


def test_crew_column_is_present_and_populated():
    table = assemble_comparison(*_reports(crew_value=0.91))
    assert all(row["systems"]["crew"]["value"] == 0.91 for row in table["rows"].values())


def test_deltas_are_positive_when_the_crew_wins_and_negative_when_it_loses():
    won = assemble_comparison(*_reports(crew_value=0.90, solo_value=0.80))
    assert all(d["vs_solo"] == pytest.approx(0.10) for d in won["deltas"].values())
    assert won["wins"]["vs_solo"] == {"won": len(REGISTRY), "compared": len(REGISTRY)}

    # Losing must be recorded as losing — every metric in the suite is higher-is-better.
    lost = assemble_comparison(*_reports(crew_value=0.70, solo_value=0.80))
    assert all(d["vs_solo"] == pytest.approx(-0.10) for d in lost["deltas"].values())
    assert lost["wins"]["vs_solo"]["won"] == 0


def test_a_failed_rival_yields_no_delta_rather_than_a_free_win():
    baseline, solo, automl, crew = _reports()
    solo = {"datasets": {k: {"ok": False, "error": "boom"} for k in REGISTRY}}
    table = assemble_comparison(baseline, solo, automl, crew)

    for key in REGISTRY:
        assert table["rows"][key]["systems"]["solo_agent"]["value"] is None
        assert table["deltas"][key]["vs_solo"] is None  # not a win by default
    # A dataset the crew was never compared on cannot count toward its record.
    assert table["wins"]["vs_solo"] == {"won": 0, "compared": 0}
    assert "—" in render_markdown(table)


def test_a_failed_crew_run_yields_no_score_and_no_deltas():
    baseline, solo, automl, _ = _reports()
    crew = {"datasets": {k: {"ok": False, "error": "boom"} for k in REGISTRY}}
    table = assemble_comparison(baseline, solo, automl, crew)

    for key in REGISTRY:
        assert table["rows"][key]["systems"]["crew"]["value"] is None
        assert all(table["deltas"][key][name] is None for name, _, _ in HEADLINE_DELTAS)


def test_delta_needs_two_real_numbers():
    assert _delta(0.9, 0.8) == pytest.approx(0.1)
    assert _delta(None, 0.8) is None
    assert _delta(0.9, None) is None


def test_mock_runs_are_flagged_and_marked_in_the_render():
    table = assemble_comparison(*_reports(solo_mock=True))
    assert table["any_mock"] is True
    md = render_markdown(table)
    assert "(mock)" in md


def test_render_reports_the_win_record_for_every_headline_comparison():
    md = render_markdown(assemble_comparison(*_reports()))
    for _, _, label in HEADLINE_DELTAS:
        assert label.replace("*", "") in md.replace("*", "")
    assert "LOCKED held-out" in md


def test_charts_render_from_a_board_without_a_display(tmp_path):
    # Headless render must work on a scheduled run; also covers the missing-value path.
    #
    # MUST render into tmp_path. Passing no out_dir writes to the DEFAULT paths —
    # results/charts/day12_*.png — so this test used to overwrite the committed
    # board with its own fixture numbers (a flat 0.500/0.700/0.850/0.900, no solo
    # bar, both AutoML losses green). Those fixture bytes are what got committed.
    baseline, solo, automl, crew = _reports()
    solo = {"datasets": {k: {"ok": False} for k in REGISTRY}}
    table = assemble_comparison(baseline, solo, automl, crew)
    paths = render_all(table, out_dir=tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
        assert p.parent == tmp_path      # never the committed results/charts dir
