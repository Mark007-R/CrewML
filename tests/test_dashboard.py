"""Day 26 — dashboard client helpers (headless: no Streamlit, no HTTP).

The UI file stays a thin view; everything with logic worth breaking lives in
``crewml.dashboard.client`` and is pinned here — especially the rule that the
target picker computes NO ranking and NO default: guessing the target is the
bug the Day-26 spec forbids.
"""
from __future__ import annotations

import pandas as pd

from crewml.dashboard.client import (
    column_options,
    derivation_summary,
    format_column_option,
    is_finished,
    run_label,
    trace_rows,
)


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [30, 40, 50, None],
        "city": ["a", "b", "a", "b"],
        "churned": ["yes", "no", "no", "no"],
    })


def test_column_options_reports_facts_without_ranking():
    opts = column_options(_df())
    # column order preserved verbatim — no "likely target" reordering, no score
    assert [o["name"] for o in opts] == ["age", "city", "churned"]
    assert all(set(o) == {"name", "dtype", "n_unique", "n_missing"} for o in opts)
    age = opts[0]
    assert age["n_missing"] == 1 and age["n_unique"] == 3


def test_format_column_option_mentions_missingness_only_when_present():
    opts = column_options(_df())
    assert "missing" in format_column_option(opts[0])       # age has a NaN
    assert "missing" not in format_column_option(opts[2])   # churned has none


def test_derivation_summary_flattens_the_upload_manifest():
    manifest = {
        "key": "upload-abc123def456",
        "derivation": {"task": "classification", "subtype": "binary",
                       "metric": "roc_auc", "rule": "2 distinct values",
                       "warnings": ["w1"]},
        "source": {"target_column_as_uploaded": "churned",
                   "n_rows_uploaded": 100,
                   "n_rows_dropped_missing_target": 2},
        "n_train": 78, "n_holdout": 20,
        "train_sha256": "t" * 64, "holdout_sha256": "h" * 64,
        "already_ingested": True,
    }
    s = derivation_summary(manifest)
    assert s["dataset_key"] == "upload-abc123def456"
    assert (s["task"], s["metric"]) == ("classification", "roc_auc")
    assert s["target_column"] == "churned"
    assert s["warnings"] == ["w1"]
    assert s["holdout_sha256"] == "h" * 64
    assert s["already_ingested"] is True


def test_trace_rows_labels_known_nodes_and_passes_unknown_through():
    rows = trace_rows({"trace": ["profiler", "critic", "mystery_node"]})
    assert rows[0]["step"] == "1" and "Profiler" in rows[0]["label"]
    assert rows[1]["node"] == "critic"
    assert rows[2]["label"] == "mystery_node"  # graceful for future nodes
    assert trace_rows(None) == []


def test_is_finished_only_on_terminal_statuses():
    assert not is_finished({"status": "queued"})
    assert not is_finished({"status": "running"})
    assert is_finished({"status": "succeeded"})
    assert is_finished({"status": "failed"})


def test_run_label_includes_score_only_when_present():
    with_score = run_label({"run_id": "abc", "dataset_key": "credit-g",
                            "status": "succeeded",
                            "headline": {"final_cv_score": 0.7971}})
    assert "cv=0.7971" in with_score
    without = run_label({"run_id": "abc", "dataset_key": "credit-g",
                         "status": "running"})
    assert "cv=" not in without
