"""Day 4 — consolidate the Phase-1 baselines into one leaderboard.

Reads the per-system metrics files (Dummy + default RF, solo agent, FLAML AutoML)
and writes ``results/baselines_table.json`` + ``results/baselines_table.md`` — the
single board Phase 2's crew is measured against.

    python scripts/build_baselines_table.py

Run it after the three producers (``run_baselines.py``, ``run_solo_agent.py``,
``run_automl.py``). It only reshapes committed results, so it never re-scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.leaderboard import TABLE_JSON_PATH, TABLE_MD_PATH, render_markdown, write_table


def main() -> int:
    table = write_table()
    print(render_markdown(table))
    print(f"[table] wrote -> {TABLE_JSON_PATH}")
    print(f"[table] wrote -> {TABLE_MD_PATH}")
    if table["any_mock"]:
        print("[table] NOTE: solo column is MOCK (no LLM key) — not a headline number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
