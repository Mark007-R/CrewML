"""Day 4 guards: the AutoML ceiling scores sanely and the leaderboard is honest.

Three layers:
  * unit tests on the metric mapping + the leaderboard assembler/renderer with
    hand-built reports (no FLAML, no datasets needed);
  * integration checks on ``results/automl_metrics.json`` — present, complete,
    every dataset beats the Dummy floor;
  * integration checks on ``results/baselines_table.json`` — the board agrees with
    the per-system files and flags mock columns.
"""
from __future__ import annotations

import json

import pytest

from crewml.automl_baseline import AUTOML_SYSTEM, FLAML_METRIC
from crewml.config import RESULTS_DIR
from crewml.datasets import BENCHMARK_KEYS, REGISTRY
from crewml.leaderboard import SYSTEMS, assemble_table, render_markdown

AUTOML_METRICS_PATH = RESULTS_DIR / "automl_metrics.json"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
TABLE_JSON_PATH = RESULTS_DIR / "baselines_table.json"
# Benchmark-scoped: a Day-26 upload restored into REGISTRY at import (via
# crewml.api.app) must not widen what "complete over the suite" means.
KEYS = list(BENCHMARK_KEYS)


# --- metric mapping (unit) --------------------------------------------------

def test_flaml_metric_covers_every_registry_metric():
    """Every primary metric in the suite must map to a FLAML objective."""
    for spec in REGISTRY.values():
        assert spec.metric in FLAML_METRIC


def test_flaml_metric_targets_are_class_balanced_where_it_matters():
    """Multiclass must optimise macro-F1, not accuracy (EVAL_PROTOCOL §2)."""
    assert FLAML_METRIC["f1_macro"] == "macro_f1"
    assert FLAML_METRIC["roc_auc"] == "roc_auc"
    assert FLAML_METRIC["r2"] == "r2"


# --- leaderboard assembler (unit, hand-built reports) -----------------------

def _fake_reports():
    """Minimal in-memory reports spanning all 5 datasets for the assembler."""
    baseline = {"datasets": {k: {
        "dummy": {"value": 0.10}, "default_rf": {"value": 0.50},
    } for k in REGISTRY}}
    solo = {"datasets": {k: {"ok": True, "value": 0.55, "mock": True} for k in REGISTRY}}
    automl = {"datasets": {k: {"ok": True, "value": 0.60} for k in REGISTRY}}
    return baseline, solo, automl


def test_assemble_table_shapes_every_dataset_and_system():
    table = assemble_table(*_fake_reports())
    assert set(table["rows"]) == set(REGISTRY)
    for row in table["rows"].values():
        assert set(row["systems"]) == {s for s, _ in SYSTEMS}


def test_assemble_table_propagates_mock_flag():
    table = assemble_table(*_fake_reports())
    assert table["any_mock"] is True
    # every solo cell carries mock=True in this fixture
    assert all(r["systems"]["solo_agent"]["mock"] for r in table["rows"].values())


def test_assemble_table_missing_system_is_none_not_dropped():
    # Empty (not None) reports mean "system produced no rows" — None would fall
    # back to reading the on-disk results, which is the script path, not this test.
    baseline, _, _ = _fake_reports()
    table = assemble_table(baseline, solo={}, automl={})
    row = table["rows"][KEYS[0]]
    assert row["systems"]["solo_agent"]["value"] is None
    assert row["systems"]["automl_flaml"]["value"] is None
    # a missing system renders as an em dash, never a fabricated number
    assert "—" in render_markdown(table)


def test_render_markdown_marks_mock_columns():
    md = render_markdown(assemble_table(*_fake_reports()))
    assert "(mock)" in md
    for _, label in SYSTEMS:
        assert label in md


# --- automl_metrics.json integration ----------------------------------------

def _load_automl() -> dict:
    if not AUTOML_METRICS_PATH.exists():
        pytest.skip("automl_metrics.json missing — run scripts/run_automl.py")
    return json.loads(AUTOML_METRICS_PATH.read_text())


def test_automl_metrics_complete():
    report = _load_automl()
    assert report["system"] == AUTOML_SYSTEM
    assert report["failures"] == {}
    assert set(report["datasets"]) == set(BENCHMARK_KEYS)


@pytest.mark.parametrize("key", KEYS)
def test_automl_entry_is_wellformed(key):
    report = _load_automl()
    entry = report["datasets"][key]
    assert entry["ok"] is True
    assert entry["metric"] == REGISTRY[key].metric
    assert isinstance(entry["value"], float)
    assert entry["value"] == entry["value"]  # not NaN
    assert entry["best_estimator"]  # a concrete learner was chosen


@pytest.mark.parametrize("key", KEYS)
def test_automl_beats_dummy_floor(key):
    """A serious AutoML ceiling must clear the feature-blind floor everywhere."""
    automl = _load_automl()
    baseline = json.loads(BASELINE_METRICS_PATH.read_text())
    dummy = baseline["datasets"][key]["dummy"]["value"]
    assert automl["datasets"][key]["value"] > dummy


# --- baselines_table.json integration ---------------------------------------

def test_baselines_table_matches_source_metrics():
    if not TABLE_JSON_PATH.exists():
        pytest.skip("baselines_table.json missing — run build_baselines_table.py")
    table = json.loads(TABLE_JSON_PATH.read_text())
    automl = _load_automl()
    assert set(table["rows"]) == set(BENCHMARK_KEYS)
    for key in BENCHMARK_KEYS:
        board = table["rows"][key]["systems"]["automl_flaml"]["value"]
        assert board == pytest.approx(automl["datasets"][key]["value"])
