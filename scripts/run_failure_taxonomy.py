"""Day 17 runner — the failure taxonomy (census + injection probes).

Classifies every archived run record into the closed taxonomy (nothing re-run), then
runs the live injection probes — a blatant leaked column the screens must catch, a
subtle one engineered inside the detection window, and a starved executor timeout —
plus the record-level Critic-detector probes, and writes the committed board + chart.

Usage:  python scripts/run_failure_taxonomy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.charts import plot_failure_taxonomy
from crewml.config import ARTIFACTS_DIR
from crewml.failure_taxonomy import (
    TAXONOMY_RESULT_PATH,
    TAXONOMY_TABLE_MD_PATH,
    assemble_report,
    mine_archive,
    run_leak_probe,
    run_record_probes,
    run_timeout_probe,
    write_report,
)


def main() -> None:
    print("[day17] failure taxonomy — mining the archive ...")
    archive = mine_archive()
    print(f"[day17] census: {archive['n_crew_runs']} crew runs + {archive['n_solo_runs']} solo "
          f"runs -> {len(archive['events'])} events")

    leak_probes = []
    for kind in ("blatant", "subtle"):
        print(f"[day17] live leak probe '{kind}' — running the real crew on an injected dataset ...")
        p = run_leak_probe(kind)
        leak_probes.append(p)
        print(f"  detected={p['detected']} (profiler={p['profiler_flagged']}, "
              f"dropped={p['plan_dropped']}, critic={p['critic_leakage_finding']}) "
              f"cv={p['cv_score']} [{p['seconds']}s]")

    print("[day17] live timeout probe — starved executor cap in a child process ...")
    tp = run_timeout_probe()
    if tp.get("ok"):
        print(f"  detected={tp['detected']} (timed_out={tp['trainer_timed_out']}, "
              f"critic_blocker={tp['critic_filed_blocker']}) [{tp['seconds']}s]")
    else:
        print(f"  PROBE FAILED: {tp.get('error', '')[:200]}")

    # Record-level probes mutate a real archived final state; prefer a Day-14 full run.
    base_record_path = next(iter(sorted((ARTIFACTS_DIR / "ablation").glob("*/day14_full.json"))), None)
    record_probes = []
    if base_record_path:
        record = json.loads(base_record_path.read_text())
        record_probes = run_record_probes(record)
        for p in record_probes:
            print(f"[day17] record-level probe '{p['probe']}': detected={p['detected']}")
    else:
        print("[day17] no archived day14_full record found — skipping record-level probes")

    report = write_report(assemble_report(archive, leak_probes, tp, record_probes))
    chart = plot_failure_taxonomy(report)
    print(f"[day17] wrote {TAXONOMY_RESULT_PATH}")
    print(f"[day17] wrote {TAXONOMY_TABLE_MD_PATH}")
    print(f"[day17] wrote {chart}")


if __name__ == "__main__":
    main()
