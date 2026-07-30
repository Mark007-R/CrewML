"""Day 25 — run the cache & telemetry study.

    python scripts/run_day25_cache_study.py

Times the cached Profiler+Planner nodes cold vs warm on every locked dataset
(asserting the warm answer is byte-identical), then drives two identical crew
runs through the real RunStore + JobRunner on credit-g and records exactly what
``GET /metrics`` serves. Deterministic core only — LLM narratives are disabled,
so the savings measured are avoided recomputation, never provider latency.

Outputs (committed, registered in the artifact registry):
* ``results/day25_cache_telemetry.json`` — the full record.
* ``results/day25_cache_telemetry.md`` — the tables.
* ``results/charts/day25_cache_telemetry.png`` — the figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from crewml import cache_telemetry_study, charts


def main() -> int:
    print("[day25] node cache cold/warm across the locked suite, then two API "
          "runs on credit-g (deterministic core, LLM off)", flush=True)
    data = cache_telemetry_study.run_study()

    for r in data["node_timings"]:
        print(f"  {r['dataset']:<10} cold={r['cold_s']:>7}s warm={r['warm_s']:>6}s "
              f"saved={r['saved_s']}s identical={r['warm_answer_identical']}", flush=True)
    for r in data["api_round_trip"]["runs"]:
        print(f"  api/{r['label']:<5} status={r['status']} dur={r['duration_s']}s "
              f"hits={(r.get('cache') or {}).get('n_hits')} "
              f"cv={r['final_cv_score']}", flush=True)
    print(f"  same result fingerprint: "
          f"{data['api_round_trip']['same_result_fingerprint']}", flush=True)

    cache_telemetry_study.STUDY_PATH.write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    cache_telemetry_study.STUDY_MD_PATH.write_text(
        cache_telemetry_study.render_markdown(data), encoding="utf-8")
    charts.plot_cache_telemetry(data)
    print(f"[day25] wrote {cache_telemetry_study.STUDY_PATH}")
    print(f"[day25] wrote {cache_telemetry_study.STUDY_MD_PATH}")
    print(f"[day25] wrote {charts.CACHE_TELEMETRY_CHART_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
