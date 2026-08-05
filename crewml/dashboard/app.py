"""CrewML dashboard — Streamlit client of the Day-24/26 API.

Run (API first, then the dashboard):

    uvicorn crewml.api.app:app --port 8000
    streamlit run crewml/dashboard/app.py

Everything on screen came over HTTP; the dashboard holds no crew, dataset, or
store imports. Honesty rules surface in the UI: mock-mode runs are banner-
labelled (never presentable as real), every score is captioned CV-on-train,
and the upload flow makes the user CHOOSE the target column — the Run button
stays locked until a choice exists and the server has shown what it derived
and sealed from that choice.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file with its own directory on sys.path, not the
# repo root — same bootstrap the scripts/ entry points use.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crewml.dashboard.client import (
    ApiError,
    CrewApiClient,
    column_options,
    derivation_summary,
    format_column_option,
    is_finished,
    run_label,
    trace_rows,
)

POLL_SECONDS = 2.0

st.set_page_config(page_title="CrewML", page_icon="🤖", layout="wide")


# --- API connection ----------------------------------------------------------

api_url = st.sidebar.text_input(
    "API URL", os.getenv("CREWML_API_URL", "http://127.0.0.1:8000"),
)
client = CrewApiClient(api_url)

try:
    health = client.health()
except ApiError as exc:
    st.error(f"CrewML API unreachable — start it with "
             f"`uvicorn crewml.api.app:app`.\n\n{exc}")
    st.stop()

st.sidebar.success(f"API {health['version']} · provider: {health['provider']}")
MOCK = bool(health.get("mock_mode"))
if MOCK:
    st.sidebar.warning("MOCK MODE — no LLM key configured. Runs execute the "
                       "deterministic pipeline; **numbers are not real "
                       "LLM-crew results** and are labelled as such.")

st.title("CrewML — multi-agent ML crew")
st.caption("Profiler → Planner → Feature Engineer → Trainer → Critic loop → "
           "Ensembler → Reporter. All scores shown are CV-on-train; the "
           "holdout stays sealed (SHA-256) and untouched.")

tab_new, tab_runs, tab_report, tab_metrics = st.tabs(
    ["🚀 New run", "📡 Runs & live trace", "📄 Report", "📊 Service metrics"]
)


# --- New run -----------------------------------------------------------------

with tab_new:
    source = st.radio("Data source", ["Benchmark dataset", "Upload CSV"],
                      horizontal=True)

    dataset_key: str | None = None

    if source == "Benchmark dataset":
        specs = client.datasets()
        bench = {k: v for k, v in specs.items() if not k.startswith("upload-")}
        key = st.selectbox("Dataset", list(bench), index=None,
                           placeholder="— choose a dataset —")
        if key:
            spec = bench[key]
            st.info(f"**{key}** — {spec['task']} ({spec['subtype']}), "
                    f"metric **{spec['metric']}**. {spec['note']}")
            dataset_key = key

    else:
        uploaded = st.file_uploader("CSV file", type=["csv"])
        # The sealed-ingest panel below must never outlive the file it describes:
        # session state persists across reruns, so after ingesting file A and
        # then selecting file B (or clearing the uploader), A's derivation panel
        # and dataset_key would otherwise stay on screen with the Run button
        # enabled — a run against A while looking at B. Key the stored ingest to
        # the exact bytes+name it came from and drop it on any mismatch.
        file_key = None
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()
            file_key = (uploaded.name,
                        hashlib.sha256(csv_bytes).hexdigest())
        stale = st.session_state.get("upload")
        if stale is not None and stale.get("file_key") != file_key:
            del st.session_state["upload"]
        if uploaded is not None:
            try:
                preview = pd.read_csv(BytesIO(csv_bytes))
            except Exception as exc:
                st.error(f"Could not parse the CSV: {exc}")
                preview = None
            if preview is not None:
                st.dataframe(preview.head(20), use_container_width=True)
                st.caption(f"{len(preview)} rows × {preview.shape[1]} columns "
                           f"(first 20 rows shown)")

                # The target is CHOSEN, never guessed: no default, no ranking.
                opts = column_options(preview)
                labels = {format_column_option(o): o["name"] for o in opts}
                picked = st.selectbox(
                    "Target column — what should the crew predict?",
                    list(labels), index=None,
                    placeholder="— choose the target column (required) —",
                )
                target_column = labels.get(picked) if picked else None

                if st.button("Ingest & seal", type="primary",
                             disabled=target_column is None,
                             help="Splits server-side and SHA-256-seals the "
                                  "holdout before any agent can run."):
                    try:
                        resp = client.upload_csv(
                            csv_bytes, filename=uploaded.name,
                            target_column=target_column,
                        )
                        st.session_state["upload"] = {"file_key": file_key,
                                                      "resp": resp}
                    except ApiError as exc:
                        st.error(str(exc))

        up = st.session_state.get("upload")
        if up:
            s = derivation_summary(up["resp"]["manifest"])
            st.subheader("What the server derived from your choice")
            c1, c2, c3 = st.columns(3)
            c1.metric("Task", f"{s['task']} ({s['subtype']})")
            c2.metric("Metric", s["metric"])
            c3.metric("Split (train / sealed holdout)",
                      f"{s['n_train']} / {s['n_holdout']}")
            st.caption(f"Rule: {s['rule']} — derived from the column **you** "
                       f"picked (`{s['target_column']}`). Wrong column? "
                       f"Re-upload and pick again; nothing has run yet.")
            for w in s["warnings"]:
                st.warning(w)
            if s["n_rows_dropped_missing_target"]:
                st.caption(f"{s['n_rows_dropped_missing_target']} row(s) with a "
                           f"missing target were dropped (labels are never "
                           f"imputed).")
            st.code(f"holdout sha256 = {s['holdout_sha256']}", language=None)
            if s["already_ingested"]:
                st.info("This exact file + target was ingested before — the "
                        "existing sealed split is reused (one dataset, one seal).")
            dataset_key = s["dataset_key"]

    st.divider()
    with st.expander("Run options"):
        max_iterations = st.slider("Max Critic iterations", 1, 10, 3)
        param_search = st.checkbox("Trainer parameter search", value=True)
        use_llm = st.checkbox("Use LLM agents (else deterministic fallbacks)",
                              value=True)

    if st.button("Run the crew", type="primary", disabled=dataset_key is None,
                 help=None if dataset_key else
                 "Pick a dataset — or upload a CSV and choose its target — first."):
        try:
            run_id = client.submit_run(
                dataset_key, max_iterations=max_iterations,
                param_search=param_search, llm=use_llm,
            )
            st.session_state["watch_run_id"] = run_id
            st.success(f"Run `{run_id}` queued on `{dataset_key}` — follow it "
                       f"in **Runs & live trace**.")
        except ApiError as exc:
            st.error(str(exc))


# --- Runs & live trace -------------------------------------------------------

with tab_runs:
    try:
        runs = client.runs()
    except ApiError as exc:
        runs = []
        st.error(str(exc))

    if not runs:
        st.info("No runs yet — start one in **New run**.")
    else:
        st.dataframe(pd.DataFrame([{
            "run_id": r["run_id"], "dataset": r["dataset_key"],
            "status": r["status"], "metric": r.get("metric"),
            "cv_score": (r.get("headline") or {}).get("final_cv_score"),
            "created": r.get("created_at"), "finished": r.get("finished_at"),
        } for r in runs]), use_container_width=True, hide_index=True)
        st.caption("cv_score is a CV-on-train estimate, never a holdout score.")

        ids = [r["run_id"] for r in runs]
        default = st.session_state.get("watch_run_id")
        chosen = st.selectbox(
            "Follow a run", ids,
            index=ids.index(default) if default in ids else 0,
            format_func=lambda rid: run_label(
                next(r for r in runs if r["run_id"] == rid)),
        )
        live = st.toggle("Watch live", value=True,
                         help=f"Polls /status every {POLL_SECONDS:.0f}s until "
                              f"the run finishes.")
        box = st.empty()

        def _render(snap: dict) -> None:
            with box.container():
                st.markdown(f"**{snap['run_id']}** on `{snap['dataset_key']}` — "
                            f"status **{snap['status']}**")
                prog = snap.get("progress")
                if prog:
                    done = prog.get("nodes_visited") or 0
                    st.progress(min(done / 12.0, 1.0),
                                text=f"{prog.get('current_node') or 'starting'} "
                                     f"(node visit {done}, iteration "
                                     f"{prog.get('iteration')})")
                    rows = trace_rows(prog)
                    if rows:
                        st.table(pd.DataFrame(rows))
                    if prog.get("decisions"):
                        st.caption("Critic decisions so far: "
                                   + " → ".join(prog["decisions"]))
                if snap["status"] == "failed":
                    st.error(snap.get("error") or "run failed")
                head = snap.get("headline")
                if head:
                    st.metric("Final CV score (train-side estimate)",
                              head.get("final_cv_score"))
                    st.caption(
                        f"model: {head.get('final_model_kind')} · iterations: "
                        f"{head.get('iterations_run')} · holdout untouched: "
                        f"{head.get('holdout_untouched')}")
                tel = snap.get("telemetry")
                if tel:
                    st.caption(f"⏱ {tel.get('duration_s')}s · "
                               f"{tel.get('tokens_spent') or 0} tokens · "
                               f"cache hits {tel.get('cache_hits') or 0}")

        snap = client.status(chosen)
        _render(snap)
        while live and not is_finished(snap):
            time.sleep(POLL_SECONDS)
            snap = client.status(chosen)
            _render(snap)


# --- Report ------------------------------------------------------------------

with tab_report:
    finished = [r["run_id"] for r in runs if r.get("status") == "succeeded"]
    if not finished:
        st.info("No succeeded runs to report on yet.")
    else:
        rid = st.selectbox("Run", finished)
        try:
            rep = client.report(rid)
        except ApiError as exc:
            rep = None
            st.error(str(exc))
        if rep:
            pins = ((rep.get("manifest") or {}).get("pins") or {})
            if ((pins.get("llm") or {}).get("mock_mode")):
                st.warning("**MOCK RUN** — produced without a live LLM. These "
                           "numbers exercise the pipeline; they are not real "
                           "crew results.")
            record = rep.get("record") or {}
            fm = record.get("final_model") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Metric", record.get("metric"))
            c2.metric("Final CV score", fm.get("final_cv_score"))
            c3.metric("Iterations", record.get("iterations_run"))
            c4.metric("Holdout untouched", str(record.get("holdout_untouched")))
            st.caption("CV-on-train estimate (`cv_score_is_holdout: false`); "
                       "the manifest's seals prove the holdout stayed sealed.")
            if rep.get("model_card"):
                st.markdown(rep["model_card"])
            with st.expander("Run manifest (pins + seals + fingerprint)"):
                st.json(rep.get("manifest"))
            with st.expander("Full record"):
                st.json(record)
            with st.expander("Telemetry"):
                st.json(rep.get("telemetry"))


# --- Service metrics ---------------------------------------------------------

with tab_metrics:
    try:
        m = client.metrics()
    except ApiError as exc:
        m = None
        st.error(str(exc))
    if m:
        r, lat, llm, cache = (m.get("runs") or {}), (m.get("latency") or {}), \
                             (m.get("llm") or {}), (m.get("cache") or {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total runs", r.get("total"))
        c2.metric("Success rate", r.get("success_rate"))
        c3.metric("p50 / p95 latency (s)",
                  f"{lat.get('p50_s')} / {lat.get('p95_s')}")
        c4.metric("Cache hit rate", cache.get("hit_rate"))
        st.caption(f"LLM: {llm.get('n_calls')} calls · "
                   f"{llm.get('tokens_spent')} tokens · "
                   f"{llm.get('llm_time_s')}s provider time")
        ds = m.get("datasets") or {}
        if ds:
            st.dataframe(pd.DataFrame([
                {"dataset": k, **v} for k, v in ds.items()
            ]), use_container_width=True, hide_index=True)
            st.caption("Per-dataset scores are CV-on-train "
                       "(`cv_score_is_holdout: false`).")
