"""Shared visual theme for Streamlit apps — mark.dev paper with one project accent.

Copy this file into a repo as ``ui_theme.py`` (next to the app, or inside the
package), set the three ACCENT values from the palette, and call
``apply_theme()`` right after ``st.set_page_config(...)`` on EVERY page script
(multipage apps run each page on its own, so CSS does not carry over).

Pair it with ``.streamlit/config.toml`` at the repo root so native widgets
(sliders, checkboxes, focus rings) pick up the same accent:

    [theme]
    base = "light"
    primaryColor = "<ACCENT>"
    backgroundColor = "#fefaf5"
    secondaryBackgroundColor = "#faf2e9"
    textColor = "#1c1714"
    font = "sans serif"
"""
from __future__ import annotations

import streamlit as st

# ---- the only values that change between projects -------------------------
ACCENT = "#4D7C0F"         # main accent (buttons, links, active tab, metric values)
ACCENT_STRONG = "#3D630C"  # hover / pressed
ACCENT_SOFT = "#739E3A"    # lighter partner for gradients and secondary marks
# ---------------------------------------------------------------------------

PAPER = "#fefaf5"
PAPER_ALT = "#faf2e9"
CARD = "#fffdfa"
INK = "#1c1714"
INK_2 = "#57504a"
INK_3 = "#756c65"
LINE = "#e9dbcd"
LINE_STRONG = "#d8c3b2"
FONT_DISPLAY = "'Fraunces', Georgia, 'Times New Roman', serif"
FONT_BODY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Consolas, monospace"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def colorway() -> list[str]:
    """Categorical chart colours: accent first, then warm neutrals that sit on paper."""
    return [ACCENT, "#8c7b6b", ACCENT_SOFT, "#c9a27e", "#3b3230", "#b9ad9f"]


def sequential(n: int = 6) -> list[str]:
    """Light-to-accent ramp for heatmaps / ordered categories."""
    h = ACCENT.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    base = (250, 242, 233)  # PAPER_ALT
    out = []
    for i in range(n):
        t = 0.15 + 0.85 * i / max(n - 1, 1)
        out.append("#%02x%02x%02x" % tuple(round(base[k] + (c - base[k]) * t)
                                            for k, c in enumerate((r, g, b))))
    return out


def plotly_layout(**overrides) -> dict:
    """Transparent plotly layout that blends into the paper ground.

    Render with ``st.plotly_chart(fig, theme=None)`` — Streamlit's default chart
    theme otherwise repaints the traces in its own palette. Do not add a
    ``title_font`` without a title: Streamlit then prints "undefined".
    """
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        colorway=colorway(),
        xaxis=dict(gridcolor=LINE, linecolor=LINE_STRONG, zerolinecolor=LINE),
        yaxis=dict(gridcolor=LINE, linecolor=LINE_STRONG, zerolinecolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=LINE_STRONG, font=dict(color=INK)),
    )
    layout.update(overrides)
    return layout


# plotly's stock palette — traces still wearing these get repainted by style_fig()
_PLOTLY_DEFAULTS = {"#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
                    "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52"}


def style_fig(fig, **layout_overrides):
    """Apply the paper layout AND recolour traces, then render with theme=None.

    plotly express stamps its own colour onto each trace at creation time, so
    setting ``colorway`` afterwards does nothing — hence the second pass. Even
    better: pass ``color_discrete_sequence=ui_theme.colorway()`` to the px call.
    """
    fig.update_layout(**plotly_layout(**layout_overrides))
    palette, i = colorway(), 0
    for trace in fig.data:
        for holder in ("marker", "line"):
            obj = getattr(trace, holder, None)
            colour = getattr(obj, "color", None) if obj is not None else None
            if isinstance(colour, str) and colour.lower() in _PLOTLY_DEFAULTS:
                obj.color = palette[i % len(palette)]
                i += 1
    return fig


def _dots_svg() -> str:
    fill = ACCENT.replace("#", "%23")
    dots = [(52, 71, 1.4, .28), (178, 38, .9, .22), (312, 95, 1.6, .3), (458, 52, 1, .24),
            (546, 128, 1.3, .26), (96, 188, 1.1, .25), (236, 222, 1.5, .28), (392, 176, .9, .2),
            (508, 246, 1.4, .27), (28, 298, 1.2, .24), (158, 344, 1.6, .3), (286, 312, 1, .22),
            (430, 366, 1.3, .26), (566, 402, 1.1, .23), (72, 452, 1.5, .29), (204, 486, .9, .21),
            (348, 438, 1.4, .27), (482, 528, 1.2, .25), (128, 566, 1.3, .26), (268, 592, 1, .22),
            (412, 556, 1.5, .28), (588, 486, .9, .2)]
    circles = "".join(f"%3Ccircle cx='{x}' cy='{y}' r='{r}' opacity='{o}'/%3E" for x, y, r, o in dots)
    return ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600'%3E"
            f"%3Cg fill='{fill}'%3E{circles}%3C/g%3E%3C/svg%3E")


def theme_css() -> str:
    tint = _rgba(ACCENT, .06)
    glow = _rgba(ACCENT, .14)
    line_accent = _rgba(ACCENT, .24)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500;1,9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
  --paper: {PAPER}; --paper-alt: {PAPER_ALT}; --card: {CARD};
  --ink: {INK}; --ink-2: {INK_2}; --ink-3: {INK_3};
  --line: {LINE}; --line-strong: {LINE_STRONG};
  --accent: {ACCENT}; --accent-strong: {ACCENT_STRONG}; --accent-soft: {ACCENT_SOFT};
  --accent-tint: {tint}; --accent-glow: {glow}; --accent-line: {line_accent};
  --shadow-sm: 0 1px 2px rgba(60, 40, 20, .04), 0 4px 16px rgba(60, 40, 20, .05);
  --shadow-md: 0 2px 4px rgba(60, 40, 20, .05), 0 14px 38px rgba(60, 40, 20, .09);
}}

/* paper ground with the mark.dev dot texture */
.stApp {{
  background-color: var(--paper);
  background-image: url("{_dots_svg()}");
  background-repeat: repeat;
  color: var(--ink);
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, [data-testid="stMarkdownContainer"] {{ font-family: {FONT_BODY}; }}
::selection {{ background: var(--accent); color: var(--paper); }}

[data-testid="stHeader"] {{ background: rgba(254, 250, 245, .82); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }}
[data-testid="stDecoration"] {{ background-image: linear-gradient(90deg, var(--accent), var(--accent-soft)); height: 2px; }}

[data-testid="stSidebar"] {{ background-color: var(--paper-alt); border-right: 1px solid var(--line); }}
[data-testid="stSidebar"] > div:first-child {{ background: transparent; }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{ border-radius: 999px; }}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover {{ background: var(--accent-tint); }}
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {{ background: var(--accent-tint); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span, [data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--accent); font-weight: 600; }}

/* type */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONT_DISPLAY}; color: var(--ink); letter-spacing: -.015em; text-wrap: balance; }}
.stApp h1 {{ font-weight: 700; line-height: 1.08; }}
.stApp h2, .stApp h3 {{ font-weight: 600; }}
.stApp h4 {{ font-weight: 600; font-size: 1.1rem; }}
.stApp a {{ color: var(--accent); }}
.stApp a:hover {{ color: var(--accent-strong); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--ink-3); }}
.stApp hr {{ border-color: var(--line); }}
.stApp code {{ font-family: {FONT_MONO}; font-size: .85em; color: var(--accent); background: var(--accent-tint); border-radius: 6px; padding: .1em .35em; }}
.stApp pre code, [data-testid="stCode"] code {{ color: inherit; background: transparent; padding: 0; }}
[data-testid="stCode"] pre, .stCode pre {{ background: var(--paper-alt) !important; border: 1px solid var(--line); border-radius: 10px; }}

/* widget labels read like mark.dev form labels */
[data-testid="stWidgetLabel"] p {{ font-family: {FONT_MONO}; font-size: .7rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }}
.stCheckbox [data-testid="stWidgetLabel"] p, [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
.stToggle [data-testid="stWidgetLabel"] p, [data-testid="stToggle"] [data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label[data-baseweb="radio"] p {{
  font-family: {FONT_BODY}; font-size: .92rem; font-weight: 500; letter-spacing: 0; text-transform: none; color: var(--ink-2);
}}

/* buttons: pills */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button,
[data-testid="baseButton-secondary"], [data-testid="stBaseButton-secondary"] {{
  border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink-2);
  font-weight: 600; padding: .5rem 1.25rem; box-shadow: var(--shadow-sm);
  transition: background .25s ease, border-color .25s ease, color .25s ease, transform .25s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
[data-testid="baseButton-secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {{
  border-color: var(--accent); color: var(--accent); background: var(--accent-tint); transform: translateY(-1px);
}}
.stApp button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--accent); border-color: var(--accent); color: var(--paper);
  box-shadow: 0 2px 12px {_rgba(ACCENT, .2)};
}}
.stApp button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: var(--accent-strong); border-color: var(--accent-strong); color: var(--paper);
}}
.stApp button[kind="primary"] p, [data-testid="baseButton-primary"] p, [data-testid="stBaseButton-primary"] p {{ color: var(--paper); }}
.stApp button:focus-visible {{ box-shadow: 0 0 0 3px var(--accent-glow); outline: none; }}

/* inputs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {{
  background-color: var(--card); border: 1px solid var(--line); border-radius: 10px;
}}
.stApp [data-baseweb="base-input"], .stApp [data-baseweb="input"] input, .stApp [data-baseweb="textarea"] textarea {{ background-color: var(--card); color: var(--ink); }}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within, .stApp [data-baseweb="select"] > div:focus-within {{
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
}}
.stApp [data-baseweb="tag"] {{ background-color: var(--accent-tint); border: 1px solid var(--accent-line); border-radius: 999px; }}
.stApp [data-baseweb="tag"] span {{ color: var(--accent); }}
[data-testid="stSlider"] [role="slider"] {{ background-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-glow); }}
[data-testid="stThumbValue"], [data-testid="stSliderThumbValue"] {{ color: var(--accent); font-family: {FONT_MONO}; }}

/* metrics as mark.dev stat cards */
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1rem; box-shadow: var(--shadow-sm); }}
[data-testid="stMetricLabel"] p {{ font-family: {FONT_MONO}; font-size: .68rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-family: {FONT_DISPLAY}; color: var(--accent); font-weight: 600; }}
/* serif digits run wider than Streamlit's default face: scale with the viewport so 5-6 metrics in a row don't ellipsize */
[data-testid="stMetricValue"] {{ font-size: clamp(1.35rem, 1rem + 1.1vw, 2.1rem); }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; }}
.stTabs [data-baseweb="tab"] {{ color: var(--ink-2); font-weight: 500; padding-left: .15rem; padding-right: .15rem; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--accent); }}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color: var(--accent); font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent); }}
.stTabs [data-baseweb="tab-border"] {{ background-color: var(--line); }}

/* surfaces */
[data-testid="stExpander"] details {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow-sm); }}
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p {{ color: var(--accent); }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--card); border: 1.5px dashed var(--line-strong); border-radius: 16px; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--accent); background: var(--accent-tint); }}
[data-testid="stFileUploaderDropzone"] svg {{ color: var(--accent); fill: var(--accent); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: var(--card); }}
[data-testid="stAlert"], [data-testid="stAlert"] > div {{ border-radius: 12px; }}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{ border-radius: 12px; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
</style>
"""


def apply_theme() -> None:
    st.markdown(theme_css(), unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    """Small mono uppercase label above a heading, like mark.dev's status badge."""
    st.markdown(
        f"<div style=\"font-family:{FONT_MONO};font-size:.68rem;font-weight:600;letter-spacing:.14em;"
        f"text-transform:uppercase;color:{ACCENT};margin-bottom:.35rem\">{text}</div>",
        unsafe_allow_html=True,
    )
