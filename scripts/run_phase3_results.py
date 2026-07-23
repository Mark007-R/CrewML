"""Day 18 runner: consolidate the six Phase-3 studies into one results section.

Reads the committed study files, writes ``results/phase3_results.{json,md}`` and the
four-panel summary chart. Re-runs are idempotent: nothing is scored, so the output
changes only when a study file does.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.charts import plot_phase3_summary
from crewml.phase3_results import assemble_results, load_phase3_bundle, write_results


def main() -> None:
    bundle = load_phase3_bundle()
    report = assemble_results(bundle)
    paths = write_results(report)
    chart = plot_phase3_summary(report)

    h = report["headline"]
    print("Phase 3 consolidated results written:")
    for kind, p in paths.items():
        print(f"  {kind}: {p}")
    print(f"  chart: {chart}")
    print()
    print("Headline:")
    print(f"  crew vs solo      {h['crew_vs_solo']}")
    print(f"  crew vs AutoML    {h['crew_vs_automl']}")
    print(f"  crew vs defaultRF {h['crew_vs_default_rf']}")
    print(f"  critic max probe recovery  {h['critic_probe_max_recovery']}")
    print(f"  planner mean drop {h['planner_mean_drop']}  FE mean drop {h['fe_mean_drop']}")
    print(f"  fatal crew {h['fatal_crew']} vs solo {h['fatal_solo']}")
    print(f"  live suite cost   ${h['live_run_total_cost_usd']:.4f}"
          if h["live_run_total_cost_usd"] is not None else "  live suite cost   —")
    print(f"  outage-resilient  {h['outage_resilience_all_equal']}")


if __name__ == "__main__":
    main()
