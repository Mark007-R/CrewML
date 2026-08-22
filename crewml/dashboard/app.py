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

# --- Stylesheet --------------------------------------------------------------
# One design system, declared once: white ground, near-black ink, and a single
# green accent, because this screen is read, not skimmed — the numbers and the
# seal state have to be the loudest things on it, which only works if nothing
# else competes. Green is a FILL, never small text: #1DB954 on white is 2.9:1,
# so anything green and typographic uses --clay-deep (#0E7A3A, 4.6:1), and
# green buttons take a black label the way Spotify's do. The same five colours
# are mirrored in
# .streamlit/config.toml, which is the only way to reach the canvas-rendered
# dataframe grid — change one, change the other.
#
# Selectors hang off Streamlit's stable `data-testid` / `data-baseweb` hooks,
# never the generated emotion class names. Font-family is set on containers and
# inherited, never with a `*` rule, so the Material icon ligatures survive.
st.markdown("""
<style>
:root {
  --paper:#F7F7F7; --ivory:#FFFFFF; --card:#FFFFFF;
  --ink:#121212; --ink-2:#2E2E2E; --muted:#6A6A6A; --faint:#9B9B9B;
  --line:#E8E8E8; --line-2:#D4D4D4;
  --clay:#1DB954; --clay-deep:#0E7A3A; --clay-tint:rgba(29,185,84,.12);
  --kraft:#1DB954; --manilla:#E8F7EE;
  --moss:#0E7A3A; --moss-ink:#0B5F2C; --moss-tint:rgba(29,185,84,.16);
  --sans:-apple-system,"Segoe UI Variable Text","Segoe UI",Inter,system-ui,sans-serif;
  --serif:"Tiempos Text","Iowan Old Style",Charter,Georgia,"Times New Roman",serif;
  --mono:"Cascadia Code","JetBrains Mono",Consolas,ui-monospace,monospace;
  --shadow:0 1px 2px rgba(0,0,0,.04), 0 6px 18px rgba(0,0,0,.055);
}

/* --- canvas ------------------------------------------------------------- */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], .stMarkdown, p, li, label, button, input, select,
textarea, h4, h5, h6, [data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"] { font-family: var(--sans); }
[data-testid="stApp"], [data-testid="stAppViewContainer"] {
  background: var(--ivory); color: var(--ink);
  -webkit-font-smoothing: antialiased; }
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 3.6rem; padding-bottom: 4rem; max-width: 1180px; }
[data-testid="stSidebar"] { background: var(--paper);
  border-right: 1px solid var(--line); }
[data-testid="stSidebarUserContent"] { padding-top: 1.35rem; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 8px;
  border: 3px solid var(--ivory); }
::-webkit-scrollbar-thumb:hover { background: var(--kraft); }

/* --- type --------------------------------------------------------------- */
h1, h2, h3 { font-family: var(--serif); color: var(--ink);
  letter-spacing: -.012em; font-weight: 600; }
h1 { font-size: 2.15rem; line-height: 1.18; }
h2 { font-size: 1.5rem; margin-top: 1.6rem; }
h3 { font-size: 1.2rem; }
h4, h5, h6 { color: var(--ink); font-weight: 640; letter-spacing: -.005em; }
p, li { color: var(--ink-2); }
code, kbd, pre, [data-testid="stCode"] * { font-family: var(--mono); }
:not(pre) > code { background: rgba(29,185,84,.14); color: #0E7A3A;
  border-radius: 5px; padding: .1em .38em; font-size: .86em; }
[data-testid="stCaptionContainer"] p { color: var(--muted);
  font-size: .845rem; line-height: 1.55; }
a, a:visited { color: var(--clay-deep); text-decoration-color: var(--kraft); }

/* --- hero lockup -------------------------------------------------------- */
.crew-hero { margin: 0 0 .35rem 0; }
.crew-eyebrow { display:flex; align-items:center; gap:9px; font-size:.72rem;
  font-weight:700; letter-spacing:.15em; color:var(--muted);
  text-transform:uppercase; margin-bottom:.65rem; }
.crew-eyebrow .dot { width:8px; height:8px; border-radius:50%;
  background:var(--clay); box-shadow:0 0 0 3px var(--clay-tint); }
.crew-h1 { font-family:var(--serif); font-size:1.98rem; line-height:1.2;
  font-weight:600; letter-spacing:-.018em; color:var(--ink); margin:0 0 .45rem; }
.crew-sub { font-size:.97rem; line-height:1.58; color:var(--muted);
  max-width:82ch; margin:0; }

/* --- pipeline chips ----------------------------------------------------- */
.chip-row { display:flex; flex-wrap:wrap; gap:5px; align-items:center;
  margin:1rem 0 1.15rem; }
.chip { padding:5px 13px; border-radius:999px; font-size:.795rem;
  font-weight:550; white-space:nowrap; background:var(--card);
  border:1px solid var(--line-2); color:var(--muted);
  transition:background .2s ease, border-color .2s ease, color .2s ease; }
.chip.done { background:var(--moss-tint); border-color:rgba(29,185,84,.45);
  color:var(--moss-ink); }
.chip.done::before { content:"\\2713"; margin-right:6px; font-weight:700;
  color:var(--moss); }
.chip.active { background:var(--clay-tint); border-color:var(--clay);
  color:var(--clay-deep); font-weight:680;
  animation:crewPulse 1.9s ease-in-out infinite; }
@keyframes crewPulse {
  0%,100% { box-shadow:0 0 0 0 rgba(29,185,84,.36); }
  55%     { box-shadow:0 0 0 6px rgba(29,185,84,0); } }
.chip.pending { opacity:.5; }
.chip.decision { background:#EFEFEF; border-color:var(--line-2);
  color:var(--ink-2); font-weight:620; }
.chip-arrow { color:var(--faint); font-size:.72rem; padding:0 1px; }
.seal-ok { color:var(--moss); font-weight:650; }
.seal-bad { color:#E22134; font-weight:650; }

/* --- cards -------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  border-radius:16px; }
div[data-testid="stVerticalBlockBorderWrapper"][style*="border"] {
  background:var(--card); border:1px solid var(--line) !important;
  border-radius:16px; box-shadow:var(--shadow); }
.crew-card { background:var(--card); border:1px solid var(--line);
  border-radius:16px; padding:18px 20px; box-shadow:var(--shadow); }

/* --- metrics ------------------------------------------------------------ */
div[data-testid="stMetric"] { background:var(--card);
  border:1px solid var(--line); border-radius:14px; padding:14px 17px 16px;
  box-shadow:var(--shadow);
  transition:border-color .2s ease, box-shadow .2s ease; }
div[data-testid="stMetric"]:hover { border-color:var(--line-2);
  box-shadow:0 2px 4px rgba(0,0,0,.05), 0 10px 24px rgba(0,0,0,.08); }
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
  font-size:.695rem !important; font-weight:680 !important;
  letter-spacing:.085em; text-transform:uppercase;
  color:var(--muted) !important; line-height:1.45 !important; }
[data-testid="stApp"] [data-testid="stMetricValue"] {
  font-family:var(--serif) !important; font-size:1.6rem !important;
  line-height:1.28 !important; color:var(--ink) !important;
  letter-spacing:-.015em; }

/* --- buttons ------------------------------------------------------------ */
[data-testid="stBaseButton-primary"] { background:var(--clay);
  border:1px solid var(--clay); color:#0A0A0A; border-radius:11px;
  font-weight:700; letter-spacing:.005em; padding:.6rem 1.15rem;
  box-shadow:0 1px 2px rgba(0,0,0,.08);
  transition:background .16s ease, box-shadow .16s ease, transform .07s ease; }
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
  background:var(--clay-deep); border-color:var(--clay-deep);
  box-shadow:0 4px 14px rgba(29,185,84,.32); }
[data-testid="stBaseButton-primary"]:active:not(:disabled) {
  transform:translateY(1px); box-shadow:0 1px 2px rgba(0,0,0,.14); }
[data-testid="stBaseButton-primary"]:disabled { background:var(--paper);
  border-color:var(--line-2); color:var(--faint); box-shadow:none; }
[data-testid="stBaseButton-secondary"] { background:var(--card);
  border:1px solid var(--line-2); color:var(--ink-2); border-radius:11px;
  font-weight:580; padding:.55rem 1rem;
  transition:border-color .16s ease, background .16s ease, color .16s ease; }
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
  border-color:var(--clay); background:var(--clay-tint);
  color:var(--clay-deep); }

/* --- tabs --------------------------------------------------------------- */
[data-baseweb="tab-list"] { gap:2px; background:transparent; }
[data-testid="stTab"] { color:var(--muted); font-weight:580; font-size:.93rem;
  padding:9px 15px; border-radius:10px 10px 0 0;
  transition:color .16s ease, background .16s ease; }
[data-testid="stTab"]:hover { color:var(--ink); background:var(--clay-tint); }
[data-testid="stTab"][aria-selected="true"] { color:var(--ink);
  font-weight:660; }
[data-baseweb="tab-highlight"] { background:var(--clay); height:2.5px;
  border-radius:3px 3px 0 0; }
[data-baseweb="tab-border"] { background:var(--line); height:1px; }

/* --- inputs ------------------------------------------------------------- */
[data-testid="stTextInputRootElement"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-baseweb="input"] { background:var(--card) !important;
  border:1px solid var(--line-2) !important; border-radius:11px !important;
  box-shadow:none !important;
  transition:border-color .16s ease, box-shadow .16s ease; }
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"]:focus-within { border-color:var(--clay) !important;
  box-shadow:0 0 0 3px var(--clay-tint) !important; }
[data-testid="stWidgetLabel"] p { font-size:.855rem; font-weight:600;
  color:var(--ink-2); }
ul[role="listbox"] { border-radius:12px !important;
  border:1px solid var(--line-2) !important; background:var(--card) !important;
  box-shadow:0 14px 38px rgba(0,0,0,.15) !important; padding:5px !important; }
li[role="option"] { border-radius:8px !important; font-size:.9rem !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
  background:var(--clay-tint) !important; color:var(--clay-deep) !important; }

/* --- file uploader ------------------------------------------------------ */
[data-testid="stFileUploaderDropzone"] { background:var(--paper);
  border:1.5px dashed var(--line-2); border-radius:14px;
  transition:border-color .16s ease, background .16s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--clay);
  background:var(--clay-tint); }
[data-testid="stFileUploaderFile"] { background:var(--card);
  border:1px solid var(--line); border-radius:11px; padding:9px 12px; }

/* --- code / seals ------------------------------------------------------- */
[data-testid="stCode"] pre { background:#F6F6F6 !important;
  border:1px solid var(--line); border-left:3px solid var(--kraft);
  border-radius:11px; padding:13px 15px !important; }
/* the seal is the point of this block — wrap the digest, never clip it */
[data-testid="stApp"] [data-testid="stCode"] code {
  color:var(--ink) !important; font-size:.8rem; background:none; padding:0;
  white-space:pre-wrap !important; overflow-wrap:anywhere;
  word-break:break-all; }
[data-testid="stCodeCopyButton"] { color:var(--muted) !important; }
[data-testid="stCodeCopyButton"]:hover { color:var(--clay-deep) !important; }

/* --- alerts ------------------------------------------------------------- */
[data-testid="stAlert"] { border-radius:13px; border:1px solid var(--line);
  box-shadow:none; }
[data-testid="stAlertContentInfo"] { background:#F4F4F4;
  border-color:var(--line-2); }
[data-testid="stAlertContentSuccess"] { background:var(--moss-tint);
  border-color:rgba(29,185,84,.45); }
[data-testid="stAlertContentWarning"] { background:#FFF6E0;
  border-color:#F2D48A; }
[data-testid="stAlertContentError"] { background:rgba(226,33,52,.09);
  border-color:rgba(226,33,52,.35); }
[data-testid="stAlert"] p { color:var(--ink-2); font-size:.9rem; }

/* --- progress / misc ---------------------------------------------------- */
[data-testid="stProgress"] > div > div { background:#EAEAEA !important;
  border-radius:999px; height:9px; }
[data-testid="stProgress"] > div > div > div > div {
  background:var(--clay) !important;
  border-radius:999px; }
hr, [data-testid="stMarkdown"] hr { border-color:var(--line); }
details, [data-testid="stExpander"] details { background:var(--card);
  border:1px solid var(--line) !important; border-radius:13px !important;
  box-shadow:var(--shadow); }
[data-testid="stExpander"] summary:hover { color:var(--clay-deep); }
[data-testid="stDataFrame"] { border:1px solid var(--line);
  border-radius:12px; overflow:hidden; }
[data-testid="stTooltipContent"] { background:var(--ink) !important;
  color:var(--ivory) !important; border-radius:10px; font-size:.83rem; }

/* --- sidebar ------------------------------------------------------------ */
.sb-brand { display:flex; align-items:center; gap:11px; margin:0 0 .3rem; }
.sb-mark { width:34px; height:34px; border-radius:10px; flex:0 0 auto;
  background:var(--clay);
  display:flex; align-items:center; justify-content:center; color:#0A0A0A;
  font-family:var(--serif); font-size:1.05rem; font-weight:600;
  box-shadow:0 2px 6px rgba(29,185,84,.30); }
.sb-word { font-family:var(--serif); font-size:1.32rem; font-weight:600;
  color:var(--ink); letter-spacing:-.015em; line-height:1; }
.sb-tag { font-size:.735rem; color:var(--muted); letter-spacing:.04em;
  margin:.15rem 0 1.1rem 45px; }
.sb-status { display:flex; align-items:center; gap:9px; background:var(--card);
  border:1px solid var(--line); border-radius:12px; padding:10px 12px;
  box-shadow:var(--shadow); margin:.2rem 0 .55rem; }
.sb-status .live { width:9px; height:9px; border-radius:50%; flex:0 0 auto;
  background:var(--moss); animation:crewLive 2.1s ease-in-out infinite; }
.sb-status.mock .live { background:#E22134; }
@keyframes crewLive {
  0%,100% { box-shadow:0 0 0 0 rgba(29,185,84,.5); }
  60%     { box-shadow:0 0 0 5px rgba(29,185,84,0); } }
.sb-status .txt { font-size:.8rem; color:var(--ink-2); line-height:1.35; }
.sb-status .txt b { color:var(--ink); font-weight:660; }
.sb-status .txt span { color:var(--muted); }
.sb-note { border-left:2.5px solid var(--kraft); padding:2px 0 2px 12px;
  font-size:.79rem; line-height:1.55; color:var(--muted); }
.sb-note b { color:var(--ink-2); font-weight:640; }

/* --- run header + telemetry strip --------------------------------------- */
.run-head { display:flex; align-items:center; flex-wrap:wrap; gap:9px;
  margin:.1rem 0 .2rem; }
.run-pill { font-size:.7rem; font-weight:700; letter-spacing:.09em;
  text-transform:uppercase; padding:4px 11px; border-radius:999px;
  border:1px solid transparent; }
.run-pill.ok { background:var(--clay); border-color:var(--clay);
  color:#0A0A0A; }
.run-pill.bad { background:rgba(226,33,52,.12);
  border-color:rgba(226,33,52,.4); color:#B01B2B; }
.run-pill.run { background:var(--clay-tint); border-color:var(--clay);
  color:var(--clay-deep); animation:crewPulse 1.9s ease-in-out infinite; }
.run-id, .run-ds { font-family:var(--mono); font-size:1.02rem;
  font-weight:600; color:var(--ink); letter-spacing:-.01em; }
.run-ds { color:var(--clay-deep); }
.run-on { color:var(--faint); font-size:.86rem; }
.tele { display:flex; flex-wrap:wrap; gap:.35rem 1.6rem; margin:.85rem 0 .1rem;
  padding-top:.8rem; border-top:1px solid var(--line); }
.tele span { font-size:.78rem; color:var(--muted); letter-spacing:.01em; }
.tele b { font-family:var(--serif); font-size:1.02rem; font-weight:600;
  color:var(--ink); margin-right:3px; }
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
                                 horizontal=True, color="#1DB954")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption("Per-dataset scores are CV-on-train "
                       "(`cv_score_is_holdout: false`).")
