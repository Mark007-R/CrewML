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

from crewml.dashboard import ui_theme
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

# --- Shared paper theme ------------------------------------------------------
# mark.dev paper ground, Fraunces/Inter/JetBrains Mono, olive accent. The same
# five base colours are mirrored in .streamlit/config.toml, which is the only
# way to reach the canvas-rendered dataframe grid — change one, change both.
ui_theme.apply_theme()

# --- App-specific stylesheet -------------------------------------------------
# Only what the shared theme cannot know about: the hero lockup, the pipeline
# chips, the sidebar brand block and the run header / telemetry strip, plus the
# few layout fixes this screen genuinely needs (metric cards that must not clip
# a seal digest, and a code block that must wrap one).
#
# Everything is expressed in the tokens ui_theme declares on :root, so the
# accent only has to change in one place. Selectors hang off Streamlit's stable
# `data-testid` / `data-baseweb` hooks, never the generated emotion class names.
# Font-family is set on containers and inherited, never with a `*` or `span`
# rule, so the Material icon ligatures survive.
st.markdown("""
<style>
:root {
  --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --serif: 'Fraunces', Georgia, 'Times New Roman', serif;
  --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  --ok: #3f7a3a;   --ok-tint: rgba(63, 122, 58, .10);
  --bad: #b3261e;  --bad-tint: rgba(179, 38, 30, .08);
  --warn: #a86a12; --warn-tint: rgba(168, 106, 18, .10);
  --info: #2f5f8a; --info-tint: rgba(47, 95, 138, .09);
}

/* --- canvas ------------------------------------------------------------- */
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1180px; }
[data-testid="stSidebarUserContent"] { padding-top: 1.35rem; }

/* --- type --------------------------------------------------------------- */
h1, h2, h3 { line-height: 1.15; }
h1 { font-size: 2.15rem; }
h2 { font-size: 1.5rem; margin-top: 1.6rem; }
h3 { font-size: 1.2rem; }

/* --- hero lockup -------------------------------------------------------- */
.crew-hero { margin: 0 0 .35rem 0; }
.crew-eyebrow { display:flex; align-items:center; gap:9px;
  font-family:var(--mono); font-size:.68rem; font-weight:600;
  letter-spacing:.14em; color:var(--accent); text-transform:uppercase;
  margin-bottom:.7rem; }
.crew-eyebrow .dot { width:8px; height:8px; border-radius:50%;
  background:var(--accent); box-shadow:0 0 0 3px var(--accent-glow); }
.crew-h1 { font-family:var(--serif); font-size:2.05rem; line-height:1.12;
  font-weight:700; letter-spacing:-.018em; color:var(--ink);
  margin:0 0 .5rem; text-wrap:balance; }
.crew-sub { font-family:var(--sans); font-size:.97rem; line-height:1.62;
  color:var(--ink-2); max-width:82ch; margin:0; }

/* --- pipeline chips ----------------------------------------------------- */
.chip-row { display:flex; flex-wrap:wrap; gap:5px; align-items:center;
  margin:1.15rem 0 1.15rem; }
.chip { font-family:var(--sans); padding:5px 14px; border-radius:999px;
  font-size:.79rem; font-weight:500; white-space:nowrap; background:var(--card);
  border:1px solid var(--line); color:var(--ink-3);
  transition:background .25s ease, border-color .25s ease, color .25s ease; }
.chip.done { background:var(--ok-tint); border-color:rgba(63,122,58,.28);
  color:var(--ok); }
.chip.done::before { content:"\\2713"; margin-right:6px; font-weight:700; }
.chip.active { background:var(--accent-tint); border-color:var(--accent-line);
  color:var(--accent); font-weight:600;
  animation:crewPulse 1.9s ease-in-out infinite; }
@keyframes crewPulse {
  0%,100% { box-shadow:0 0 0 0 var(--accent-glow); }
  55%     { box-shadow:0 0 0 6px transparent; } }
.chip.pending { opacity:.55; }
.chip.decision { background:var(--paper-alt); border-color:var(--line-strong);
  color:var(--ink-2); font-weight:600; }
.chip-arrow { color:var(--line-strong); font-size:.72rem; padding:0 1px; }
.crew-card { background:var(--card); border:1px solid var(--line);
  border-radius:16px; padding:18px 20px; box-shadow:var(--shadow-sm); }
.seal-ok { color:var(--ok); font-weight:650; }
.seal-bad { color:var(--bad); font-weight:650; }

/* --- cards -------------------------------------------------------------- */
/* st.container(border=True) draws its frame from a generated class, and every
   vertical block shares this wrapper, so nothing marks the bordered ones. Only
   properties that stay invisible on a borderless wrapper are set: the paper
   line colour and the card radius. */
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line);
  border-radius:16px; }

/* --- metrics ------------------------------------------------------------ */
/* Streamlit truncates metric labels and values to one nowrap line, and its
   emotion <style> is injected after this one — so equal-!important rules lose
   on document order. These deliberately over-qualify to win on specificity;
   a truncated seal or split line is exactly the kind of number this screen
   exists to show. */
/* height:100% inside a stretched column makes every card in a row match the
   tallest one, while still growing to fit a value that wraps */
[data-testid="stApp"] div[data-testid="stMetric"] { height:100% !important; }
[data-testid="stApp"] div[data-testid="stMetric"] > div {
  height:auto !important; }
[data-testid="stApp"] div[data-testid="stMetric"],
[data-testid="stApp"] div[data-testid="stMetric"] > div {
  overflow:visible !important; }
[data-testid="stColumn"] > div { height:100%; }
/* NB: stMetricLabel is a <label>, not a <div> — qualifying it with `div`
   silently matches nothing, which is how the ellipsis survived two passes. */
[data-testid="stApp"] [data-testid="stMetricLabel"],
[data-testid="stApp"] [data-testid="stMetricLabel"] *,
[data-testid="stApp"] [data-testid="stMetricValue"],
[data-testid="stApp"] [data-testid="stMetricValue"] * {
  overflow:visible !important; text-overflow:clip !important;
  white-space:normal !important; overflow-wrap:anywhere; }
[data-testid="stApp"] [data-testid="stMetricLabel"] p {
  line-height:1.45 !important; }
[data-testid="stApp"] [data-testid="stMetricValue"] {
  font-size:1.6rem !important; line-height:1.28 !important;
  letter-spacing:-.015em; }

/* --- dropdown menus ----------------------------------------------------- */
ul[role="listbox"] { border-radius:12px !important;
  border:1px solid var(--line) !important; background:var(--card) !important;
  box-shadow:var(--shadow-md) !important; padding:5px !important; }
li[role="option"] { border-radius:999px !important; font-size:.9rem !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background:var(--accent-tint) !important; color:var(--accent) !important; }

/* --- file uploader ------------------------------------------------------ */
[data-testid="stFileUploaderFile"] { background:var(--card);
  border:1px solid var(--line); border-radius:12px; padding:9px 12px; }

/* --- code / seals ------------------------------------------------------- */
[data-testid="stCode"] pre { border-left:3px solid var(--accent);
  padding:13px 15px !important; }
/* the seal is the point of this block — wrap the digest, never clip it */
[data-testid="stApp"] [data-testid="stCode"] code {
  color:var(--ink) !important; font-size:.8rem; background:none; padding:0;
  white-space:pre-wrap !important; overflow-wrap:anywhere;
  word-break:break-all; }
[data-testid="stCodeCopyButton"] { color:var(--ink-3) !important; }
[data-testid="stCodeCopyButton"]:hover { color:var(--accent) !important; }

/* --- alerts ------------------------------------------------------------- */
/* status colours keep their meaning, always on their tint. Streamlit paints
   the tint on the outer container and only names the kind on an inner child,
   hence :has() — tinting the child would draw a second, inset box. */
[data-testid="stAlertContainer"] { border:1px solid transparent; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
  background-color:var(--info-tint); border-color:rgba(47,95,138,.22);
  color:var(--info); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  background-color:var(--ok-tint); border-color:rgba(63,122,58,.25);
  color:var(--ok); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
  background-color:var(--warn-tint); border-color:rgba(168,106,18,.25);
  color:var(--warn); }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  background-color:var(--bad-tint); border-color:rgba(179,38,30,.22);
  color:var(--bad); }
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li {
  color:inherit; font-size:.9rem; }

/* --- progress / misc ---------------------------------------------------- */
/* the bar only: stProgress also holds the text label as a sibling div, so a
   bare `> div > div` chain would squash that label to the bar's height */
[data-testid="stProgress"] [data-baseweb="progress-bar"] > div > div {
  background:var(--paper-alt) !important; border-radius:999px; height:9px; }
[data-testid="stProgress"] [data-baseweb="progress-bar"] > div > div > div {
  background:var(--accent) !important; border-radius:999px; }
[data-testid="stTooltipContent"] { background:var(--ink) !important;
  color:var(--paper) !important; border-radius:10px; font-size:.83rem; }

/* --- sidebar ------------------------------------------------------------ */
.sb-brand { display:flex; align-items:center; gap:11px; margin:0 0 .3rem; }
.sb-mark { width:34px; height:34px; border-radius:11px; flex:0 0 auto;
  background:var(--accent);
  display:flex; align-items:center; justify-content:center; color:var(--paper);
  font-family:var(--serif); font-size:1.05rem; font-weight:700;
  box-shadow:0 2px 10px var(--accent-glow); }
.sb-word { font-family:var(--serif); font-size:1.32rem; font-weight:700;
  color:var(--ink); letter-spacing:-.02em; line-height:1; }
.sb-tag { font-family:var(--mono); font-size:.62rem; font-weight:600;
  color:var(--ink-3); letter-spacing:.12em; margin:.3rem 0 1.1rem 45px; }
.sb-status { display:flex; align-items:center; gap:9px; background:var(--card);
  border:1px solid var(--line); border-radius:14px; padding:10px 12px;
  box-shadow:var(--shadow-sm); margin:.2rem 0 .55rem; }
.sb-status .live { width:9px; height:9px; border-radius:50%; flex:0 0 auto;
  background:var(--ok); animation:crewLive 2.1s ease-in-out infinite; }
.sb-status.mock .live { background:var(--bad); }
@keyframes crewLive {
  0%,100% { box-shadow:0 0 0 0 rgba(63,122,58,.45); }
  60%     { box-shadow:0 0 0 5px rgba(63,122,58,0); } }
.sb-status .txt { font-family:var(--sans); font-size:.8rem; color:var(--ink-2);
  line-height:1.4; }
.sb-status .txt b { color:var(--ink); font-weight:650; }
.sb-status .txt span { color:var(--ink-3); }
.sb-note { font-family:var(--sans); border-left:2.5px solid var(--accent);
  padding:2px 0 2px 12px; font-size:.79rem; line-height:1.58;
  color:var(--ink-3); }
.sb-note b { color:var(--ink-2); font-weight:640; }

/* --- run header + telemetry strip --------------------------------------- */
.run-head { display:flex; align-items:center; flex-wrap:wrap; gap:9px;
  margin:.1rem 0 .2rem; }
.run-pill { font-family:var(--mono); font-size:.64rem; font-weight:600;
  letter-spacing:.11em; text-transform:uppercase; padding:4px 12px;
  border-radius:999px; border:1px solid transparent; }
.run-pill.ok { background:var(--ok-tint); border-color:rgba(63,122,58,.3);
  color:var(--ok); }
.run-pill.bad { background:var(--bad-tint); border-color:rgba(179,38,30,.3);
  color:var(--bad); }
.run-pill.run { background:var(--accent-tint);
  border-color:var(--accent-line); color:var(--accent);
  animation:crewPulse 1.9s ease-in-out infinite; }
.run-id, .run-ds { font-family:var(--mono); font-size:1rem;
  font-weight:600; color:var(--ink); letter-spacing:-.01em; }
.run-ds { color:var(--accent); }
.run-on { font-family:var(--sans); color:var(--ink-3); font-size:.86rem; }
.tele { display:flex; flex-wrap:wrap; gap:.35rem 1.6rem; margin:.85rem 0 .1rem;
  padding-top:.8rem; border-top:1px solid var(--line); }
.tele span { font-family:var(--mono); font-size:.68rem; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3); }
.tele b { font-family:var(--serif); font-size:1.05rem; font-weight:600;
  letter-spacing:0; text-transform:none; color:var(--accent);
  margin-right:4px; }
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
    st.markdown(
        '<div class="sb-brand"><div class="sb-mark">C</div>'
        '<div class="sb-word">CrewML</div></div>'
        '<div class="sb-tag">MULTI-AGENT ML CREW</div>',
        unsafe_allow_html=True)
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
    n_datasets = len(health.get("datasets") or [])
    provider = html.escape(str(health.get("provider")))
    version = html.escape(str(health.get("version")))
    headline = ("<b>MOCK MODE</b> — no live LLM" if MOCK
                else f"Connected · provider <b>{provider}</b>")
    st.markdown(
        f'<div class="sb-status{" mock" if MOCK else ""}">'
        f'<div class="live"></div><div class="txt">{headline}<br>'
        f'<span>API {version} · {n_datasets} datasets registered</span>'
        f'</div></div>',
        unsafe_allow_html=True)
    if MOCK:
        st.warning("MOCK MODE — no LLM key configured. Runs execute the "
                   "deterministic pipeline; **numbers are not real "
                   "LLM-crew results** and are labelled as such.")
    st.divider()
    st.markdown(
        '<div class="sb-note">Every score on this dashboard is a '
        '<b>CV-on-train</b> estimate. The SHA-256-sealed holdout is scored '
        'once, by the final scorer, never by the crew.</div>',
        unsafe_allow_html=True)

st.markdown(
    '<div class="crew-hero">'
    '<div class="crew-eyebrow"><span class="dot"></span>'
    'CSV in · trained model out · holdout sealed</div>'
    '<h1 class="crew-h1">Seven agents, one sealed holdout, '
    'no one steering.</h1>'
    '<p class="crew-sub">Hand the crew a raw CSV. It profiles the data, plans '
    'the approach, engineers features, trains, critiques itself in a loop, '
    'ensembles and writes the model card — while the holdout stays '
    'SHA-256-sealed from the moment it is split.</p></div>',
    unsafe_allow_html=True)
pipeline_chips([{"node": n, "label": lbl, "state": "", "visits": 1}
                for n, lbl in NODE_SHORT.items()])

tab_new, tab_runs, tab_report, tab_metrics = st.tabs(
    ["New run", "Runs & live trace", "Report", "Service metrics"]
)


# --- New run -----------------------------------------------------------------

with tab_new:
    col_data, col_opts = st.columns([7, 3], gap="large")

    dataset_key: str | None = None

    with col_opts, st.container(border=True):
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
                with st.container(border=True):
                    st.markdown("##### What the server derived from your "
                                "choice")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Task", f"{s['task']} ({s['subtype']})")
                    c2.metric("Metric", s["metric"])
                    c3.metric("Split (train / sealed holdout)",
                              f"{s['n_train']} / {s['n_holdout']}")
                    st.caption(f"Rule: {s['rule']} — derived from the column "
                               f"**you** picked (`{s['target_column']}`). "
                               f"Wrong column? Re-upload and pick again; "
                               f"nothing has run yet.")
                    for w in s["warnings"]:
                        st.warning(w)
                    if s["n_rows_dropped_missing_target"]:
                        st.caption(f"{s['n_rows_dropped_missing_target']} "
                                   f"row(s) with a missing target were dropped "
                                   f"(labels are never imputed).")
                    st.code(f"holdout sha256 = {s['holdout_sha256']}",
                            language=None)
                    if s["already_ingested"]:
                        st.info("This exact file + target was ingested "
                                "before — the existing sealed split is reused "
                                "(one dataset, one seal).")
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
            status = snap["status"]
            prog = snap.get("progress")
            finished = is_finished(snap)
            if finished and not prog:
                # /status drops progress the moment a run ends, which left
                # every node reading "pending" on exactly the screen that
                # should show the pipeline complete. The finished trace is
                # still in the report, so read the real one — inventing a
                # full trace here would be a lie about which nodes ran and
                # how many times the Critic sent the crew back. Fetched
                # BEFORE the container opens: a blocking call in the middle
                # of a render flushes a half-drawn card to the browser.
                try:
                    rec = (client.report(snap["run_id"]) or {})
                    trace = ((rec.get("record") or {}).get("trace") or [])
                    if trace:
                        prog = {"trace": trace}
                except ApiError:
                    prog = None

            with box.container(border=True):
                tone = {"succeeded": "ok", "failed": "bad"}.get(status, "run")
                st.markdown(
                    f'<div class="run-head">'
                    f'<span class="run-pill {tone}">{html.escape(status)}</span>'
                    f'<span class="run-id">{html.escape(snap["run_id"])}</span>'
                    f'<span class="run-on">on</span>'
                    f'<span class="run-ds">'
                    f'{html.escape(snap["dataset_key"])}</span></div>',
                    unsafe_allow_html=True)
                if prog or not finished:
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
                    st.markdown(
                        f'<div class="tele">'
                        f'<span><b>{tel.get("duration_s")}</b> s wall clock'
                        f'</span>'
                        f'<span><b>{tel.get("llm_calls") or 0}</b> LLM calls'
                        f'</span>'
                        f'<span><b>{tel.get("tokens_spent") or 0}</b> tokens'
                        f'</span>'
                        f'<span><b>{tel.get("cache_hits") or 0}</b> cache hits'
                        f'</span></div>',
                        unsafe_allow_html=True)

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
                                 horizontal=True, color=ui_theme.ACCENT)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Per-dataset scores are CV-on-train "
                       "(`cv_score_is_holdout: false`).")
