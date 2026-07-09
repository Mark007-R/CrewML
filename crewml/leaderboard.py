"""Assemble the Phase-1 baselines leaderboard from the per-system metrics files.

Day 4 closes Phase 1 by putting every non-crew system on one board, on the same
LOCKED holdout and the same primary metric per dataset: the Dummy floor, the
default RandomForest anchor (Day 2), the solo agent (Day 3) and the FLAML AutoML
ceiling (Day 4). Phase 2's crew reads exactly these rows to answer, number vs
number, "did the team beat the solo player and the AutoML ceiling?".

This module only *reads* the committed ``results/*_metrics.json`` files and
reshapes them — it never re-scores, so the leaderboard can never disagree with the
per-system results it summarises. Missing systems are rendered as ``—`` rather than
dropped, and any system that ran in mock mode is flagged, so the board can never
quietly present a mock number as real (EVAL_PROTOCOL.md §5).
"""
from __future__ import annotations

import json
from pathlib import Path

from crewml.config import RESULTS_DIR
from crewml.datasets import REGISTRY

# System id -> human label + column order on the board (floor → ceiling).
SYSTEMS = [
    ("dummy", "Dummy (floor)"),
    ("default_rf", "default RF"),
    ("solo_agent", "Solo agent"),
    ("automl_flaml", "AutoML (FLAML)"),
]

BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
SOLO_METRICS_PATH = RESULTS_DIR / "solo_agent_metrics.json"
AUTOML_METRICS_PATH = RESULTS_DIR / "automl_metrics.json"
TABLE_JSON_PATH = RESULTS_DIR / "baselines_table.json"
TABLE_MD_PATH = RESULTS_DIR / "baselines_table.md"

# Pretty metric labels for the rendered table header.
METRIC_LABEL = {"roc_auc": "ROC AUC", "f1_macro": "macro-F1", "r2": "R²"}


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _system_value(baseline, solo, automl, key: str, system: str):
    """Pull one (value, mock) pair for a system on a dataset, or (None, False)."""
    if system in ("dummy", "default_rf"):
        entry = (baseline or {}).get("datasets", {}).get(key, {}).get(system)
        return (entry["value"], False) if entry else (None, False)
    if system == "solo_agent":
        entry = (solo or {}).get("datasets", {}).get(key)
        if entry and entry.get("ok"):
            return entry["value"], bool(entry.get("mock", False))
        return None, False
    if system == "automl_flaml":
        entry = (automl or {}).get("datasets", {}).get(key)
        return (entry["value"], False) if entry and entry.get("ok") else (None, False)
    return None, False


def assemble_table(
    baseline: dict | None = None,
    solo: dict | None = None,
    automl: dict | None = None,
) -> dict:
    """Build the leaderboard structure from the three metrics reports.

    Any argument left ``None`` is read from its canonical results path, so the
    function is both a script helper and directly unit-testable with in-memory
    reports. Returns ``{"metric_by_dataset", "rows", "any_mock"}``.
    """
    baseline = baseline if baseline is not None else _read(BASELINE_METRICS_PATH)
    solo = solo if solo is not None else _read(SOLO_METRICS_PATH)
    automl = automl if automl is not None else _read(AUTOML_METRICS_PATH)

    rows: dict[str, dict] = {}
    any_mock = False
    for key, spec in REGISTRY.items():
        cells: dict[str, dict] = {}
        for system, _ in SYSTEMS:
            value, mock = _system_value(baseline, solo, automl, key, system)
            any_mock = any_mock or mock
            cells[system] = {"value": value, "mock": mock}
        rows[key] = {"metric": spec.metric, "systems": cells}

    return {
        "metric_by_dataset": {k: REGISTRY[k].metric for k in REGISTRY},
        "rows": rows,
        "any_mock": any_mock,
    }


def render_markdown(table: dict) -> str:
    """Render the leaderboard as a GitHub-flavoured markdown table."""
    headers = ["Dataset", "Metric"] + [label for _, label in SYSTEMS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for key, row in table["rows"].items():
        metric = row["metric"]
        cells = [key, METRIC_LABEL.get(metric, metric)]
        for system, _ in SYSTEMS:
            cell = row["systems"][system]
            if cell["value"] is None:
                cells.append("—")
            else:
                mark = " *(mock)*" if cell["mock"] else ""
                cells.append(f"{cell['value']:.4f}{mark}")
        lines.append("| " + " | ".join(cells) + " |")
    if table["any_mock"]:
        lines.append("")
        lines.append("*(mock)* — solo agent ran without an LLM key; that column is "
                     "MOCK and not a headline number (EVAL_PROTOCOL.md §5).")
    return "\n".join(lines) + "\n"


def write_table() -> dict:
    """Assemble, persist (JSON + markdown), and return the leaderboard."""
    table = assemble_table()
    TABLE_JSON_PATH.write_text(json.dumps(table, indent=2))
    TABLE_MD_PATH.write_text(render_markdown(table))
    return table
