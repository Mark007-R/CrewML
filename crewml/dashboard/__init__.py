"""Day 26 — Streamlit dashboard, a pure client of the CrewML API.

``client.py`` holds everything testable (the HTTP client and the pure
formatting/validation helpers); ``app.py`` is the Streamlit script and stays as
thin as a view should be. The dashboard never imports the crew, the datasets
module, or the store — if it can't be done through the API, the dashboard
can't do it, which is exactly the boundary a production client should have.
"""
