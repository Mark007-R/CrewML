"""Day 24 — serve the CrewML API.

    python scripts/run_api.py [--host 127.0.0.1] [--port 8000] [--reload]

Then:

    curl http://127.0.0.1:8000/healthz
    curl -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" \
         -d '{"dataset_key": "credit-g", "param_search": false}'
    curl http://127.0.0.1:8000/status/<run_id>
    curl http://127.0.0.1:8000/report/<run_id>

The run-store lives at ``artifacts/api/runs.sqlite`` (git-ignored) unless
``CREWML_RUN_STORE`` points elsewhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the CrewML FastAPI app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="dev auto-reload")
    args = ap.parse_args()

    import uvicorn

    uvicorn.run("crewml.api.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
