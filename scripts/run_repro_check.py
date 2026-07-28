"""Day 23 — run the run-level reproducibility study.

    python scripts/run_repro_check.py [--no-live]

Three arms (see :mod:`crewml.repro_study` for the full design):

1. **Deterministic core** — the crew with LLM narratives off, run twice per
   dataset in separate fresh processes under identical pins. The result
   fingerprints must be bit-identical.
2. **Seed sensitivity** — a third run with ``CREWML_SEED=43`` must MOVE the
   fingerprint, proving the seed is controlled rather than ignored.
3. **Live LLM** (labelled, only if the provider answers an 8-token preflight) —
   two live runs under identical pins; reports which layers reproduced
   (scored result / generated FE code / advisory prose). Never simulated.

Outputs:
* ``results/day23_reproducibility.json`` + ``.md`` — committed, registered in
  :mod:`crewml.artifact_registry`.
* ``artifacts/repro/*.json`` — git-ignored per-run manifests for inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crewml.repro_study import main

if __name__ == "__main__":
    raise SystemExit(main())
