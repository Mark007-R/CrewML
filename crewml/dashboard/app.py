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
import html
import json
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
    NODE_SHORT,
    column_options,
    derivation_summary,
    format_column_option,
    is_finished,
    node_states,
    run_label,
)

POLL_SECONDS = 2.0

st.set_page_config(page_title="CrewML", page_icon="🤖", layout="wide")

# One small stylesheet: pipeline chips + soft cards. Colors are translucent so
# they read correctly on both Streamlit themes.
st.markdown("""
<style>
.chip-row { display:flex; flex-wrap:wrap; gap:6px; align-items:center;
            margin: 2px 0 10px 0; }
.chip { padding: 3px 12px; border-radius: 999px; font-size: 0.82rem;
        border: 1px solid rgba(128,128,128,.35); white-space: nowrap; }
.chip.done    { background: rgba(22,163,74,.16); border-color: rgba(22,163,74,.55); }
.chip.active  { background: rgba(37,99,235,.18); border-color: rgba(37,99,235,.75);
                font-weight: 600; }
.chip.pending { opacity: .55; }
.chip.decision{ background: rgba(217,119,6,.14); border-color: rgba(217,119,6,.5); }
.chip-arrow { opacity:.4; font-size:.8rem; }
.seal-ok  { color:#16a34a; font-weight:600; }
.seal-bad { color:#dc2626; font-weight:600; }
div[data-testid="stMetric"] { background: rgba(128,128,128,.07);
    border: 1px solid rgba(128,128,128,.18); border-radius: 10px;
    padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)


def pipeline_chips(states: list[dict], decisions: list[str] | None = None) -> None:
    """Render the seven-node pipeline as state-coloured chips."""
    bits = []
    for i, s in enumerate(states):
        label = html.escape(s["label"])
        if s["visits"] > 1:
            label += f" ×{s['visits']}"
        bits.append(f'<span class="chip {s["state"]}">{label}</span>')
        if i < len(states) - 1:
            bits.append('<span class="chip-arrow">→</span>')
    st.markdown(f'<div class="chip-row">{"".join(bits)}</div>',
                unsafe_allow_html=True)
    if decisions:
        dec = '<span class="chip-arrow">→</span>'.join(
            f'<span class="chip decision">Critic: {html.escape(d)}</span>'
            for d in decisions)
        st.markdown(f'<div class="chip-row">{dec}</div>',
                    unsafe_allow_html=True)


# --- API connection ----------------------------------------------------------

with st.sidebar:
    st.markdown("### 🤖 CrewML")
    api_url = st.text_input(
        "API URL", os.getenv("CREWML_API_URL", "http://127.0.0.1:8000"),
    )
    client = CrewApiClient(api_url)
    try:
        health = client.health()
    except ApiError as exc:
        st.error(f"CrewML API unreachable — start it with "
                 f"`uvicorn crewml.api.app:app`.\n\n{exc}")
        st.stop()

    MOCK = bool(health.get("mock_mode"))
    st.success(f"API {health['version']} · provider **{health['provider']}**"
               + (" · **MOCK MODE**" if MOCK else ""))
    if MOCK:
        st.warning("MOCK MODE — no LLM key configured. Runs execute the "
                   "deterministic pipeline; **numbers are not real "
                   "LLM-crew results** and are labelled as such.")
    st.caption(f"{len(health.get('datasets') or [])} datasets registered")
    st.divider()
    st.caption("Every score on this dashboard is a **CV-on-train** estimate — "
               "the SHA-256-sealed holdout is scored once, by the final "
               "scorer, never by the crew.")

st.title("CrewML — multi-agent ML crew")
pipeline_chips([{"node": n, "label": lbl, "state": "", "visits": 1}
                for n, lbl in NODE_SHORT.items()])
st.caption("Give the crew a dataset; it profiles, plans, engineers features, "
           "trains, critiques itself in a loop, ensembles and reports — with "
           "the holdout sealed (SHA-256) and untouched throughout.")

tab_new, tab_runs, tab_report, tab_metrics = st.tabs(
    ["🚀 New run", "📡 Runs & live trace", "📄 Report", "📊 Service metrics"]
)


# --- New run -----------------------------------------------------------------

with tab_new:
    col_data, col_opts = st.columns([7, 3], gap="large")

    dataset_key: str | None = None

    with col_opts:
        st.markdown("##### Run options")
        max_iterations = st.slider("Max Critic iterations", 1, 10, 3,
                                   help="The loop budget — the Critic can send "
                                        "the crew back at most this many times.")
        param_search = st.checkbox("Trainer parameter search", value=True)
        use_llm = st.checkbox("Use LLM agents", value=True,
                              help="Off = deterministic fallbacks only "
                                   "(no provider tokens spent).")

    with col_data:
        source = st.radio("Data source", ["Benchmark dataset", "Upload CSV"],
                          horizontal=True, label_visibility="collapsed")

        if source == "Benchmark dataset":
            specs = client.datasets()
            bench = {k: v for k, v in specs.items()
                     if not k.startswith("upload-")}
            key = st.selectbox("Dataset", list(bench), index=None,
                               placeholder="— choose a dataset —")
            if key:
                spec = bench[key]
                c1, c2, c3 = st.columns(3)
                c1.metric("Task", f"{spec['task']}")
                c2.metric("Subtype", spec["subtype"])
                c3.metric("Metric", spec["metric"])
                st.caption(spec["note"])
                dataset_key = key

        else:
            uploaded = st.file_uploader("CSV file", type=["csv"])
            # The sealed-ingest panel below must never outlive the file it
            # describes: session state persists across reruns, so after
            # ingesting file A and then selecting file B (or clearing the
            # uploader), A's derivation panel and dataset_key would otherwise
            # stay on screen with the Run button enabled — a run against A
            # while looking at B. Key the stored ingest to the exact
            # bytes+name it came from and drop it on any mismatch.
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
                    st.dataframe(preview.head(20), use_container_width=True,
                                 height=240)
                    st.caption(f"{len(preview)} rows × {preview.shape[1]} "
                               f"columns (first 20 rows shown)")

                    # The target is CHOSEN, never guessed: no default, no ranking.
                    opts = column_options(preview)
                    labels = {format_column_option(o): o["name"] for o in opts}
                    picked = st.selectbox(
                        "Target column — what should the crew predict?",
                        list(labels), index=None,
                        placeholder="— choose the target column (required) —",
                    )
                    target_column = labels.get(picked) if picked else None

                    if st.button("🔐 Ingest & seal", type="primary",
                                 disabled=target_column is None,
                                 help="Splits server-side and SHA-256-seals "
                                      "the holdout before any agent can run."):
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
                st.markdown("##### What the server derived from your choice")
                c1, c2, c3 = st.columns(3)
                c1.metric("Task", f"{s['task']} ({s['subtype']})")
                c2.metric("Metric", s["metric"])
                c3.metric("Split (train / sealed holdout)",
                          f"{s['n_train']} / {s['n_holdout']}")
                st.caption(f"Rule: {s['rule']} — derived from the column "
                           f"**you** picked (`{s['target_column']}`). Wrong "
                           f"column? Re-upload and pick again; nothing has "
                           f"run yet.")
                for w in s["warnings"]:
                    st.warning(w)
                if s["n_rows_dropped_missing_target"]:
                    st.caption(f"{s['n_rows_dropped_missing_target']} row(s) "
                               f"with a missing target were dropped (labels "
                               f"are never imputed).")
                st.code(f"holdout sha256 = {s['holdout_sha256']}",
                        language=None)
                if s["already_ingested"]:
                    st.info("This exact file + target was ingested before — "
                            "the existing sealed split is reused (one "
                            "dataset, one seal).")
                dataset_key = s["dataset_key"]

    st.divider()
    if st.button("🚀 Run the crew", type="primary",
                 disabled=dataset_key is None, use_container_width=True,
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
        ids = [r["run_id"] for r in runs]
        default = st.session_state.get("watch_run_id")
        sel, tog = st.columns([8, 2])
        chosen = sel.selectbox(
            "Follow a run", ids,
            index=ids.index(default) if default in ids else 0,
            format_func=lambda rid: run_label(
                next(r for r in runs if r["run_id"] == rid)),
        )
        live = tog.toggle("Watch live", value=True,
                          help=f"Polls /status every {POLL_SECONDS:.0f}s "
                               f"until the run finishes.")
        box = st.empty()

        def _render(snap: dict) -> None:
            with box.container():
                status = snap["status"]
                icon = {"succeeded": "✅", "failed": "❌",
                        "running": "⏳"}.get(status, "🕐")
                st.markdown(f"#### {icon} `{snap['run_id']}` on "
                            f"`{snap['dataset_key']}` — **{status}**")
                prog = snap.get("progress")
                finished = is_finished(snap)
                pipeline_chips(node_states(prog, finished=finished),
                               (prog or {}).get("decisions"))
                if prog and not finished:
                    done = prog.get("nodes_visited") or 0
                    st.progress(min(done / 12.0, 1.0),
                                text=f"{prog.get('current_node') or 'starting'}"
                                     f" (node visit {done}, iteration "
                                     f"{prog.get('iteration')})")
                if status == "failed":
                    st.error(snap.get("error") or "run failed")
                head = snap.get("headline")
                tel = snap.get("telemetry") or {}
                if head:
                    score = head.get("final_cv_score")
                    sealed = head.get("holdout_untouched")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("CV score (train-side)",
                              f"{score:.4f}" if score is not None else "—")
                    c2.metric("Final model", head.get("final_model_kind") or "—")
                    c3.metric("Iterations", head.get("iterations_run"))
                    c4.metric("Holdout sealed",
                              "✓ yes" if sealed else "✗ CHECK",
                              help="verify_holdout_untouched, re-checked "
                                   "after the run")
                    st.caption("CV-on-train estimate "
                               "(`cv_score_is_holdout: false`) — never a "
                               "holdout score.")
                if tel:
                    st.caption(f"⏱ {tel.get('duration_s')}s · "
                               f"{tel.get('tokens_spent') or 0} tokens · "
                               f"{tel.get('llm_calls') or 0} LLM calls · "
                               f"cache hits {tel.get('cache_hits') or 0}")

        snap = client.status(chosen)
        _render(snap)
        while live and not is_finished(snap):
            time.sleep(POLL_SECONDS)
            snap = client.status(chosen)
            _render(snap)

        st.divider()
        with st.expander("All runs"):
            st.dataframe(pd.DataFrame([{
                "run_id": r["run_id"], "dataset": r["dataset_key"],
                "status": r["status"], "metric": r.get("metric"),
                "cv_score": (r.get("headline") or {}).get("final_cv_score"),
                "created": r.get("created_at"),
                "finished": r.get("finished_at"),
            } for r in runs]), use_container_width=True, hide_index=True)
            st.caption("cv_score is a CV-on-train estimate, never a holdout "
                       "score.")


# --- Report ------------------------------------------------------------------

with tab_report:
    finished_runs = [r["run_id"] for r in runs if r.get("status") == "succeeded"]
    if not finished_runs:
        st.info("No succeeded runs to report on yet.")
    else:
        rid = st.selectbox("Run", finished_runs)
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
            score = fm.get("final_cv_score")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Metric", record.get("metric"))
            c2.metric("Final CV score",
                      f"{score:.4f}" if score is not None else "—")
            c3.metric("Iterations", record.get("iterations_run"))
            c4.metric("Holdout sealed",
                      "✓ yes" if record.get("holdout_untouched") else "✗ CHECK")
            st.caption("CV-on-train estimate (`cv_score_is_holdout: false`); "
                       "the manifest's seals prove the holdout stayed sealed.")
            for w in record.get("warnings") or []:
                st.caption(f"⚠️ {w}")

            card_col, dl_col = st.columns([8, 2])
            with dl_col:
                if rep.get("model_card"):
                    st.download_button("⬇ Model card (.md)", rep["model_card"],
                                       file_name=f"{rid}_model_card.md",
                                       use_container_width=True)
                st.download_button("⬇ Full report (.json)",
                                   json.dumps(rep, indent=2),
                                   file_name=f"{rid}_report.json",
                                   use_container_width=True)
            with card_col:
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
        sr = r.get("success_rate")
        c2.metric("Success rate", f"{sr:.0%}" if sr is not None else "—")
        c3.metric("p50 / p95 latency (s)",
                  f"{lat.get('p50_s')} / {lat.get('p95_s')}")
        hr = cache.get("hit_rate")
        c4.metric("Cache hit rate", f"{hr:.0%}" if hr is not None else "—")
        st.caption(f"LLM: {llm.get('n_calls')} calls · "
                   f"{llm.get('tokens_spent')} tokens · "
                   f"{llm.get('llm_time_s')}s provider time")
        ds = m.get("datasets") or {}
        if ds:
            df = pd.DataFrame([{"dataset": k, **v} for k, v in ds.items()])
            score_col = next((c for c in ("mean_cv_score", "best_cv_score")
                              if c in df.columns), None)
            if score_col is not None:
                chart_df = df.dropna(subset=[score_col])
                if not chart_df.empty:
                    st.bar_chart(chart_df.set_index("dataset")[score_col],
                                 horizontal=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Per-dataset scores are CV-on-train "
                       "(`cv_score_is_holdout: false`).")
