# CrewML

**An autonomous multi-agent ML engineering crew.** Give it a raw tabular dataset and a task (classification or regression); a LangGraph crew of specialised agents profiles the data, plans an approach, engineers features, trains and critiques models in a loop, ensembles the best, and writes a model card.

Upload a raw CSV, choose the target column, and follow the run to its report.

---

## Stack

| Layer | What it is |
|---|---|
| Crew | LangGraph agents sharing one state object |
| Execution | Sandboxed Python executor — import allowlist, no network egress, filesystem jail, resource caps |
| API | FastAPI — `/run`, `/status`, `/report`, `/metrics`; async worker with a SQLite run-store |
| Cache | Redis, content-addressed node cache (JSON-file fallback outside compose) |
| UI | Streamlit dashboard, a pure HTTP client of the API |
| Packaging | Single secret-free Docker image serving both API and dashboard |

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                 # add GROQ_API_KEY, or leave blank for mock mode
python scripts/prepare_datasets.py   # download + split the datasets
python -m pytest tests/
```

Run the service:

```bash
uvicorn crewml.api.app:app --port 8000       # the API
streamlit run crewml/dashboard/app.py        # the dashboard (a pure API client)
```

or the whole stack:

```bash
docker compose up --build
```

---

## Deploy as a Hugging Face Space

`sdk: docker` — one container, API private on `:8000`, dashboard on the Space's `:7860`:

```bash
python scripts/deploy_hf_space.py --set-secret --wait 900
```

The script assembles the Space repo from [`deploy/hf_space/`](deploy/hf_space/), secret-scans the staging tree (key *values*, not names), uploads, and provisions `GROQ_API_KEY` as a Space **secret** — never into the image. With no secret the Space boots in clearly-labelled mock mode.

*One caveat outside the repo's control: hosting new Docker Spaces on free hardware requires HF PRO, so the upload step is billing-gated on the target account.*

---

## License

MIT. See `LICENSE`.
