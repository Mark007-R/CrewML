"""Launch the Day-26 Streamlit dashboard (expects the API to be up).

    python scripts/run_api.py            # terminal 1
    python scripts/run_dashboard.py      # terminal 2

Environment: CREWML_API_URL points the dashboard at a non-default API address.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "crewml" / "dashboard" / "app.py"


def main() -> int:
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(APP),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
