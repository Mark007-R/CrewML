"""Day 12 — the full crew across all 5 datasets, scored on the LOCKED holdout.

    python scripts/run_crew_benchmark.py [--datasets credit-g,diabetes] [--no-search]
                                         [--no-llm] [--max-iterations 3] [--table-only]

Phase 3 opens by answering the question the whole project is built to answer: **does
the crew actually beat a single-shot solo agent and a classical AutoML ceiling on data
it never saw?** Day 11 got the crew running end-to-end, but every number it produced
was a cross-validated estimate on train — not comparable to the Day 2-4 baselines,
which are all scored on the locked holdout.

So this script does two things per dataset, in strict order:

1. **Run the crew** (Profiler → Planner → FE → Trainer → Critic → Ensembler → Reporter)
   on the train split alone, exactly as Day 11 does.
2. **Then, and only then, score what it shipped** on the holdout via
   :mod:`crewml.holdout_eval` — predictions only, no refit, features only, and the
   value computed by the same :mod:`crewml.scoring` call every baseline goes through.

The holdout step is deliberately *outside* the graph. No node can reach it, so the
crew's held-out number cannot leak backwards into its own modeling. Each run
re-verifies the split's SHA-256 fingerprint afterwards; a broken seal aborts the run
rather than reporting a score (EVAL_PROTOCOL.md §3).

``--table-only`` rebuilds the board and charts from the committed results without
re-running the crew — useful when only the presentation changed.

Outputs (all committed):
* ``results/day12_crew_holdout.json`` — per dataset: the held-out score, the CV
  estimate it is paired with, CV-vs-holdout optimism, model shipped, seal check.
* ``results/comparison_table.{json,md}`` — crew vs solo vs AutoML vs default RF vs dummy.
* ``results/charts/day12_*.png`` — the two figures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The board renders real typography (−, —, R²) that a cp1252 Windows console cannot
# encode; degrade those characters rather than kill a completed benchmark on a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already-wrapped or non-reconfigurable stream
        pass

from crewml import charts
from crewml.comparison import CREW_HOLDOUT_PATH, render_markdown, write_comparison
from crewml.config import ARTIFACTS_DIR, MAX_ITERATIONS, is_mock_mode
from crewml.crew import build_crew, initial_state
from crewml.datasets import REGISTRY, load_manifest, verify_holdout_untouched
from crewml.holdout_eval import score_on_holdout


def _positive_class(manifest: dict, key: str) -> str | None:
    return (manifest["datasets"][key].get("target") or {}).get("positive_class")


def _run_and_score(key: str, manifest: dict, *, max_iterations: int) -> dict:
    """Run the crew on one dataset, then final-score it on the holdout."""
    spec = REGISTRY[key]

    # --- 1. The crew works on train only ---
    started = time.time()
    app = build_crew()
    state = initial_state(spec, max_iterations=max_iterations)
    limit = 3 + max_iterations * 4 + 10
    final = app.invoke(state, config={"recursion_limit": limit})
    crew_seconds = time.time() - started

    # Seal must be intact *before* we score — proves the crew itself never touched it.
    if not verify_holdout_untouched(key):
        raise RuntimeError(f"{key}: holdout seal broken DURING the crew run — aborting")

    # --- 2. Only now does the holdout come out of the vault ---
    scored = score_on_holdout(spec, final, positive_class=_positive_class(manifest, key))

    critiques = final.get("critiques") or []
    record = {
        **scored,
        "mock": is_mock_mode(),
        "iterations_run": final.get("iteration"),
        "max_iterations": final.get("max_iterations"),
        "final_decision": critiques[-1].get("decision") if critiques else None,
        "trace": final.get("trace"),
        "crew_seconds": round(crew_seconds, 2),
    }

    out_dir = ARTIFACTS_DIR / "crew" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "day12_final_run.json").write_text(json.dumps(final, indent=2, default=str))
    return record


def _summarise(rec: dict) -> str:
    if not rec.get("ok"):
        return f"{rec['dataset_key']:<10} FAILED — {rec.get('error')}"
    cv = rec.get("cv_score")
    gap = rec.get("cv_minus_holdout")
    cv_str = f"{cv:.4f}" if isinstance(cv, float) else "-"
    gap_str = f"{gap:+.4f}" if isinstance(gap, float) else "-"
    return (
        f"{rec['dataset_key']:<10} holdout {rec['metric']}={rec['value']:.4f} "
        f"(cv={cv_str}, cv-minus-holdout={gap_str}) "
        f"shipped={rec.get('final_model_kind')} via {rec.get('final_model_source')} "
        f"passes={rec.get('iterations_run')} sealed={rec.get('holdout_untouched')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 12 — full crew benchmark + comparison board.")
    ap.add_argument("--datasets", default="", help="comma-separated subset (default: all 5)")
    ap.add_argument("--no-search", action="store_true", help="skip grid search (faster)")
    ap.add_argument("--no-llm", action="store_true", help="deterministic core only, no narratives")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild board + charts from committed results; no crew runs")
    args = ap.parse_args()

    if args.no_search:
        os.environ["CREWML_TRAINER_PARAM_SEARCH"] = "0"
    if args.no_llm:
        for var in ("CREWML_PROFILER_LLM", "CREWML_PLANNER_LLM", "CREWML_FE_LLM", "CREWML_CRITIC_LLM"):
            os.environ[var] = "0"

    if not args.table_only:
        keys = [k.strip() for k in args.datasets.split(",") if k.strip()] or list(REGISTRY)
        for k in keys:
            if k not in REGISTRY:
                raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

        mode = "MOCK (no LLM key)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
        print(f"[day12] full crew on {len(keys)} dataset(s), then LOCKED-holdout scoring; "
              f"max_iterations={args.max_iterations}, grid-search={not args.no_search}, narratives: {mode}")
        if is_mock_mode():
            print("[day12] WARNING: mock mode — these are NOT real LLM numbers.", flush=True)

        manifest = load_manifest()
        records: dict[str, dict] = {}
        # Merge onto any existing report so a subset re-run updates rather than truncates.
        if CREW_HOLDOUT_PATH.exists():
            records = json.loads(CREW_HOLDOUT_PATH.read_text()).get("datasets", {})

        for k in keys:
            print(f"\n[day12] === {k} ===", flush=True)
            rec = _run_and_score(k, manifest, max_iterations=args.max_iterations)
            print("  " + _summarise(rec), flush=True)
            records[k] = rec

        CREW_HOLDOUT_PATH.write_text(json.dumps({
            "any_mock": any(r.get("mock") for r in records.values()),
            "datasets": records,
        }, indent=2, default=str))
        print(f"\n[day12] wrote crew holdout scores -> {CREW_HOLDOUT_PATH}")

    table = write_comparison()
    paths = charts.render_all(table)
    print("\n" + render_markdown(table))
    for p in paths:
        print(f"[day12] chart -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
