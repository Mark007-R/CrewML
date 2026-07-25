"""Day 20 — the self-repair recovery study: inject faults, measure the comeback.

    python scripts/run_self_repair_study.py [--datasets credit-g,cpu_small]
                                            [--faults name_error,key_error,...]
                                            [--table-only]

Per the design in :mod:`crewml.self_repair_study`: for each study dataset, one
clean control run (repair must NOT fire) plus one Trainer run per injected fault
(a planted bug in the FE module the training script embeds), all with self-repair
ON against the live provider. Emits the recovery rate, attempts-to-recover, token
cost, and the score-fidelity check of every repaired run against its clean
control.

Outputs (committed): ``results/day20_self_repair.{json,md}`` and
``results/charts/day20_self_repair.png``. ``--table-only`` re-renders the table +
chart from the committed JSON without spending a token.

Refuses to run in mock mode — a recovery rate with no live provider is fiction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from crewml.charts import plot_self_repair
from crewml.self_repair_study import (
    FAULTS,
    SELF_REPAIR_RESULT_PATH,
    SELF_REPAIR_TABLE_MD_PATH,
    STUDY_DATASETS,
    render_table_md,
    run_self_repair_study,
    save_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(STUDY_DATASETS))
    parser.add_argument("--faults", default="")
    parser.add_argument("--table-only", action="store_true")
    parser.add_argument(
        "--repairer",
        choices=("live", "scripted"),
        default="live",
        help="'live' (default) measures the provider; 'scripted' measures only "
             "the harness with a deterministic stand-in and stamps the output "
             "as NOT an LLM-capability measurement.",
    )
    args = parser.parse_args()

    if args.table_only:
        report = json.loads(SELF_REPAIR_RESULT_PATH.read_text(encoding="utf-8"))
        # Re-render the MARKDOWN too, not just the chart. Previously this branch
        # printed the table and rewrote the PNG but left results/*.md untouched,
        # so a hand-merged JSON produced a correct chart beside a stale table —
        # the committed Day-20 table said "2/2" while the JSON and chart said 18/18.
        SELF_REPAIR_TABLE_MD_PATH.write_text(render_table_md(report), encoding="utf-8")
    else:
        datasets = tuple(k for k in args.datasets.split(",") if k)
        wanted = {k for k in args.faults.split(",") if k}
        faults = tuple(f for f in FAULTS if not wanted or f["key"] in wanted)
        report = run_self_repair_study(datasets, faults, repairer=args.repairer)
        save_report(report)

    print(render_table_md(report))
    chart = plot_self_repair(report)
    print(f"[chart] {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
