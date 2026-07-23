"""The Day-12 comparison board — crew vs solo vs AutoML vs default, one holdout.

Phase 1 built a leaderboard of everything that is *not* the crew (:mod:`crewml.leaderboard`).
Phase 3 opens by adding the column the whole project exists to justify: the crew's own
held-out score, next to the solo agent it must beat and the classical AutoML ceiling it
is measured against.

This module deliberately only *reshapes* committed numbers — it never scores anything.
The crew's values come from ``results/day12_crew_holdout.json`` (written by
``scripts/run_crew_benchmark.py`` via :mod:`crewml.holdout_eval`), the rest from the
Phase-1 metrics files. That keeps one property that matters more than convenience: the
board can never disagree with the runs it summarises, and it cannot invent a number for
a system that failed — a missing system renders as an em dash, never a zero.

Two honesty rules are enforced in the rendering, not left to prose:

* **Mock runs are marked.** Any system that ran without a live LLM is flagged in its
  cell, so a mock number can never be read as a headline result (EVAL_PROTOCOL.md §5).
* **Deltas are only computed between two real numbers.** ``crew - solo`` on a dataset
  where the solo agent failed is ``None``, not a flattering win.

Every metric in the suite is higher-is-better (:data:`crewml.scoring.HIGHER_IS_BETTER`),
so a positive delta always means the crew won — no per-metric sign juggling.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from crewml.config import RESULTS_DIR
from crewml.datasets import REGISTRY
from crewml.leaderboard import METRIC_LABEL, _read, _system_value
from crewml.leaderboard import (
    AUTOML_METRICS_PATH,
    BASELINE_METRICS_PATH,
    SOLO_METRICS_PATH,
)

CREW_HOLDOUT_PATH = RESULTS_DIR / "day12_crew_holdout.json"
TABLE_JSON_PATH = RESULTS_DIR / "comparison_table.json"
TABLE_MD_PATH = RESULTS_DIR / "comparison_table.md"

# Floor -> ceiling -> the crew. Order is the story the table tells.
SYSTEMS = [
    ("dummy", "Dummy (floor)"),
    ("default_rf", "default RF"),
    ("solo_agent", "Solo agent"),
    ("automl_flaml", "AutoML (FLAML)"),
    ("crew", "**Crew**"),
]

# The comparisons the project is actually judged on.
HEADLINE_DELTAS = [
    ("vs_solo", "solo_agent", "Crew − Solo"),
    ("vs_automl", "automl_flaml", "Crew − AutoML"),
    ("vs_default_rf", "default_rf", "Crew − default RF"),
]


def _crew_value(crew: dict | None, key: str) -> tuple[Optional[float], bool]:
    """Pull the crew's held-out value for a dataset, or (None, False) if it failed."""
    entry = (crew or {}).get("datasets", {}).get(key)
    if entry and entry.get("ok"):
        return entry["value"], bool(entry.get("mock", False))
    return None, False


def _delta(crew_value: Optional[float], other: Optional[float]) -> Optional[float]:
    """Crew minus a rival — only when both numbers are real. Positive => crew wins."""
    if crew_value is None or other is None:
        return None
    return round(crew_value - other, 6)


def assemble_comparison(
    baseline: dict | None = None,
    solo: dict | None = None,
    automl: dict | None = None,
    crew: dict | None = None,
) -> dict:
    """Build the full crew-inclusive board from the four metrics reports.

    Any argument left ``None`` is read from its canonical results path, so this is both
    a script helper and directly unit-testable with in-memory reports.

    Returns ``{"rows", "deltas", "wins", "any_mock", "metric_by_dataset"}`` where
    ``wins`` counts, per rival, the datasets on which the crew's held-out score is
    strictly higher — computed only over datasets where both systems produced a number.
    """
    baseline = baseline if baseline is not None else _read(BASELINE_METRICS_PATH)
    solo = solo if solo is not None else _read(SOLO_METRICS_PATH)
    automl = automl if automl is not None else _read(AUTOML_METRICS_PATH)
    crew = crew if crew is not None else _read(CREW_HOLDOUT_PATH)

    rows: dict[str, dict] = {}
    deltas: dict[str, dict] = {}
    any_mock = False
    wins = {name: {"won": 0, "compared": 0} for name, _, _ in HEADLINE_DELTAS}

    for key, spec in REGISTRY.items():
        cells: dict[str, dict] = {}
        for system, _ in SYSTEMS:
            if system == "crew":
                value, mock = _crew_value(crew, key)
            else:
                value, mock = _system_value(baseline, solo, automl, key, system)
            any_mock = any_mock or mock
            cells[system] = {"value": value, "mock": mock}
        rows[key] = {"metric": spec.metric, "systems": cells}

        crew_value = cells["crew"]["value"]
        row_deltas: dict[str, Optional[float]] = {}
        for name, rival, _ in HEADLINE_DELTAS:
            d = _delta(crew_value, cells[rival]["value"])
            row_deltas[name] = d
            if d is not None:
                wins[name]["compared"] += 1
                wins[name]["won"] += int(d > 0)
        deltas[key] = row_deltas

    return {
        "metric_by_dataset": {k: REGISTRY[k].metric for k in REGISTRY},
        "rows": rows,
        "deltas": deltas,
        "wins": wins,
        "any_mock": any_mock,
    }


def _fmt(value: Optional[float], mock: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}{' *(mock)*' if mock else ''}"


def _fmt_delta(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:+.4f}"


def render_markdown(table: dict) -> str:
    """Render the comparison board (scores + headline deltas) as markdown."""
    headers = ["Dataset", "Metric"] + [label for _, label in SYSTEMS] + [
        label for _, _, label in HEADLINE_DELTAS
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for key, row in table["rows"].items():
        cells = [key, METRIC_LABEL.get(row["metric"], row["metric"])]
        for system, _ in SYSTEMS:
            cell = row["systems"][system]
            cells.append(_fmt(cell["value"], cell["mock"]))
        for name, _, _ in HEADLINE_DELTAS:
            cells.append(_fmt_delta(table["deltas"][key][name]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("All scores are on the LOCKED held-out split, higher is better. "
                 "The crew's column is a final score taken after the run finished; "
                 "the crew never saw this split while modeling (EVAL_PROTOCOL.md §3).")
    lines.append("")
    for name, _, label in HEADLINE_DELTAS:
        w = table["wins"][name]
        lines.append(f"* **{label}**: crew wins {w['won']}/{w['compared']} datasets.")
    if table["any_mock"]:
        lines.append("")
        lines.append("*(mock)* — that system ran without a live LLM key; a MOCK number, "
                     "never a headline result (EVAL_PROTOCOL.md §5).")
    return "\n".join(lines) + "\n"


def write_comparison() -> dict:
    """Assemble, persist (JSON + markdown), and return the comparison board."""
    table = assemble_comparison()
    TABLE_JSON_PATH.write_text(json.dumps(table, indent=2))
    TABLE_MD_PATH.write_text(render_markdown(table), encoding="utf-8")
    return table
