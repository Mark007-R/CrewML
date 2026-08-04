#!/usr/bin/env bash
# CrewML Space entrypoint — API privately on :8000, dashboard on :7860.
#
# The dashboard is a pure HTTP client of the API (Day 26), so ordering
# matters only for first paint: wait for /healthz before Streamlit starts,
# else the first page render shows a connection error until a manual reload.
set -euo pipefail

uvicorn crewml.api.app:app --host 127.0.0.1 --port 8000 &

for _ in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" 2>/dev/null; then
        break
    fi
    sleep 1
done

# CORS/XSRF off: the Space serves the app inside an iframe on a different
# origin; with XSRF protection on, Streamlit's file-upload widget rejects
# the cross-origin POST and the CSV upload flow dies silently.
exec streamlit run crewml/dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
