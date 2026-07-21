"""Day 16 runner — the provider study (Groq vs Claude vs mock).

Probes every registered provider live, runs a fresh full-crew arm for each provider whose
probe passes (the mock arm always runs — it needs no provider), checks outage-resilience
against the Day-14 archival baseline, then writes the committed board + chart.

Usage:  python scripts/run_provider_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.charts import plot_provider_study
from crewml.datasets import REGISTRY
from crewml.provider_study import (
    PROVIDER_STUDY_RESULT_PATH,
    PROVIDER_STUDY_TABLE_MD_PATH,
    assemble_report,
    equality_check,
    load_day14_archival,
    probe_all,
    run_provider_arm,
    write_report,
)


def main() -> None:
    print("[day16] provider study — probing providers ...")
    probes = probe_all(progress=print)

    keys = list(REGISTRY)
    arms = {}
    # Live arms only for providers that proved reachable; the mock arm always runs.
    for name, probe in probes.items():
        if name != "mock" and probe["status"] != "ok":
            print(f"[day16] skipping live arm '{name}' — probe status: {probe['status']}")
            continue
        arms[name] = run_provider_arm(name, keys, progress=print)

    archival = load_day14_archival()
    resilience = equality_check(arms.get("mock", {}), archival)
    print(f"[day16] resilience: {resilience['n_equal']}/{resilience['n_compared']} datasets identical")

    report = write_report(assemble_report(probes, arms, resilience))
    chart = plot_provider_study(report)
    print(f"[day16] wrote {PROVIDER_STUDY_RESULT_PATH}")
    print(f"[day16] wrote {PROVIDER_STUDY_TABLE_MD_PATH}")
    print(f"[day16] wrote {chart}")


if __name__ == "__main__":
    main()
