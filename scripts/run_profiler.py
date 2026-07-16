"""Day 7 — run the Profiler agent on the benchmark suite and dump DataProfiles.

    python scripts/run_profiler.py [--dataset diabetes] [--no-llm]

The Profiler is the crew's first REAL node. For each dataset it loads the
train-only split, computes a deterministic DataProfile (schema, dtypes,
missingness incl. suspected disguised-missing zeros, target distribution +
imbalance, basic leakage checks), and — unless ``--no-llm`` or mock mode — layers
a short advisory LLM briefing for the Planner on top.

Outputs:
* ``results/day07_profiles.json`` — the DETERMINISTIC profiles only (LLM narrative
  stripped), committed as reproducible evidence.
* ``artifacts/crew/<key>/profile.json`` — the full profile incl. the LLM narrative
  (git-ignored; narratives are advisory and provider-specific).

Never touches the locked held-out split — the Profiler only ever loads ``train``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.config import ARTIFACTS_DIR, RESULTS_DIR, is_mock_mode
from crewml.crew.profiler import run_profiler
from crewml.datasets import REGISTRY

COMMITTED_PATH = RESULTS_DIR / "day07_profiles.json"


def _summarise(profile: dict) -> str:
    """A one-line human summary of a profile for the console."""
    t = profile["target"]
    flags = ",".join(profile["assessment"]["flags"]) or "none"
    if profile["task"] == "classification":
        head = f"{t['n_classes']}-class imb={t.get('imbalance_ratio')} pos={t.get('positive_class')}"
    else:
        head = f"reg mean={t['mean']} skew={t['skew']}"
    return (
        f"{profile['dataset_key']:<10} rows={profile['n_rows']:<5} "
        f"feats={profile['n_features']:<3} {head:<34} flags=[{flags}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Day-7 Profiler on the benchmark suite.")
    ap.add_argument("--dataset", default=None, help="single dataset key (default: all)")
    ap.add_argument("--no-llm", action="store_true", help="skip the advisory LLM narrative")
    args = ap.parse_args()

    keys = [args.dataset] if args.dataset else list(REGISTRY)
    for k in keys:
        if k not in REGISTRY:
            raise SystemExit(f"unknown dataset {k!r}; choose from {list(REGISTRY)}")

    with_llm = False if args.no_llm else None  # None => env/mock-aware default
    mode = "mock (no LLM)" if is_mock_mode() else ("LLM off" if args.no_llm else "LLM on")
    print(f"[profiler] Day 7 — profiling {len(keys)} dataset(s), narrative: {mode}")

    committed: dict[str, dict] = {}
    for k in keys:
        profile = run_profiler(k, with_llm=with_llm)
        print("  " + _summarise(profile))

        # Full profile (with narrative) -> git-ignored artifacts.
        art_dir = ARTIFACTS_DIR / "crew" / k
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "profile.json").write_text(json.dumps(profile, indent=2, default=str))

        # Deterministic-only copy -> committed results (reproducible).
        deterministic = {kk: vv for kk, vv in profile.items() if kk != "llm_narrative"}
        committed[k] = deterministic

    COMMITTED_PATH.write_text(json.dumps({"datasets": committed}, indent=2, default=str))
    print(f"[profiler] wrote deterministic profiles -> {COMMITTED_PATH}")
    print(f"[profiler] full profiles (with narrative) -> {ARTIFACTS_DIR / 'crew' / '<key>' / 'profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
