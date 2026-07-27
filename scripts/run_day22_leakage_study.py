"""Run the Day-22 leakage & honesty guard study and persist its artifacts.

Usage:
    python scripts/run_day22_leakage_study.py [--skip-crew-probes]

``--skip-crew-probes`` drops the two full-crew injection runs (the slow, LLM-
touching part) and keeps the deterministic families: calibration, FE-gate
probes, the runtime no-peek probe, and the seal sweep. The committed record
should always come from a full run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crewml.charts import plot_leakage_window  # noqa: E402
from crewml.leakage_study import (  # noqa: E402
    REPORT_JSON_PATH,
    REPORT_MD_PATH,
    build_report,
    save_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-crew-probes", action="store_true",
                        help="skip the two full-crew injection runs")
    args = parser.parse_args()

    report = build_report(with_crew_probes=not args.skip_crew_probes)
    save_report(report)
    chart = plot_leakage_window(report)

    print(json.dumps({
        "all_guards_hold": report["all_guards_hold"],
        "n_checks": report["n_checks"],
        "mock": report["mock"],
        "crew_probes": [
            {k: p[k] for k in ("probe", "flagged_by_measure", "plan_dropped",
                               "model_saw_leak", "detected")}
            for p in report["crew_probes"]
        ],
        "fe_guard_probes": [
            {k: p[k] for k in ("probe", "ok", "detected")}
            for p in report["fe_guard_probes"]
        ],
        "no_peek_refused": report["no_peek_probe"]["refused"],
        "holdout_seals": report["holdout_seals"],
        "artifacts": [str(REPORT_JSON_PATH), str(REPORT_MD_PATH), str(chart)],
    }, indent=2))
    return 0 if report["all_guards_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
