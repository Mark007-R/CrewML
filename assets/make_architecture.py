"""Render assets/architecture.png.

Pillow, drawn at 2x and downsampled. Dark card with light text so it stays
readable on both the GitHub light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 960 * S, 620 * S
OUT = Path(__file__).with_name("architecture.png")

BG, FG, MUTED, LINE = (13, 17, 23), (201, 209, 217), (139, 148, 158), (110, 118, 129)
ACCENT, GREEN = (188, 140, 255), (63, 185, 80)
FONTS = r"C:\Windows\Fonts"


def font(n, s):
    return ImageFont.truetype(f"{FONTS}\\{n}", s * S)


f_title, f_head = font("seguisb.ttf", 15), font("seguisb.ttf", 12)
f_small, f_lbl = font("segoeui.ttf", 10), font("segoeuii.ttf", 9)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, c=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=c, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def _head(p0, p1, c, size=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * .5, y1 - uy * s + py * s * .5),
               (x1 - ux * s - px * s * .5, y1 - uy * s - py * s * .5)], fill=c)


def _dashed(p0, p1, c, w, on=6, off=4):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    pos = 0.
    while pos < dist:
        seg = min(on * S, dist - pos)
        d.line([(x0 + ux * pos, y0 + uy * pos),
                (x0 + ux * (pos + seg), y0 + uy * (pos + seg))], fill=c, width=int(w * S))
        pos += (on + off) * S


def arrow(pts, c=LINE, w=1.5, dash=False):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        (_dashed if dash else lambda a, b, cc, ww: d.line([a, b], fill=cc, width=int(ww * S)))(
            pts[i], pts[i + 1], c, w)
    _head(pts[-2], pts[-1], c)


text(480, 26, "CrewML — a LangGraph crew that trains a model against a sealed holdout",
     f_title, FG)

# ── ingest ──────────────────────────────────────────────────────────────────
box(30, 54, 200, 60)
text(130, 76, "Upload raw CSV", f_head, FG)
text(130, 95, "you choose the target column")

arrow([(230, 84), (268, 84)])
box(270, 54, 232, 60, GREEN)
text(386, 76, "Split & SHA-256 seal", f_head, FG)
text(386, 95, "80/20 · seed 42 · sealed before any agent runs")

arrow([(502, 84), (540, 84)])
box(542, 54, 200, 60, GREEN)
text(642, 76, "Sealed holdout", f_head, FG)
text(642, 95, "no agent ever holds its path")

text(756, 76, "re-fingerprinted", f_lbl, GREEN, anchor="lm")
text(756, 92, "after every scoring run", f_lbl, GREEN, anchor="lm")

# ── the crew ────────────────────────────────────────────────────────────────
arrow([(386, 114), (386, 146)])
box(30, 148, 900, 172, LINE)
text(52, 168, "LangGraph crew — one shared state object", f_head, FG, anchor="lm")

names = ["Profiler", "Planner", "Feature Eng.", "Trainer", "Critic"]
subs = ["EDA, leakage", "models, CV", "generates code", "CV per cand.", "diagnoses"]
x0, bw, gap = 54, 152, 18
for i, (n, s) in enumerate(zip(names, subs)):
    x = x0 + i * (bw + gap)
    c = ACCENT if n == "Critic" else LINE
    box(x, 188, bw, 54, c)
    text(x + bw / 2, 206, n, f_head, FG)
    text(x + bw / 2, 226, s)
    if i:
        arrow([(x - gap, 215), (x - 4, 215)])

# critic loop back to planner
arrow([(54 + 4 * (bw + gap) + bw / 2, 242), (54 + 4 * (bw + gap) + bw / 2, 262),
       (x0 + bw + gap + bw / 2, 262), (x0 + bw + gap + bw / 2, 246)], ACCENT, dash=True)
text(470, 276, "loop back with specific fix instructions · bounded by max_iterations",
     f_lbl, ACCENT)

text(52, 302, "then → Ensembler (combine best) → Reporter (model card)", f_small, MUTED, anchor="lm")

# ── sandbox ─────────────────────────────────────────────────────────────────
arrow([(x0 + 2 * (bw + gap) + bw / 2, 320), (x0 + 2 * (bw + gap) + bw / 2, 350)])
box(250, 352, 460, 62, ACCENT)
text(480, 374, "Sandboxed Python executor", f_head, FG)
text(480, 393, "import allowlist · no network egress · filesystem jail · resource caps")

# ── service layer ───────────────────────────────────────────────────────────
box(30, 442, 280, 82)
text(170, 464, "FastAPI  /run /status", f_head, FG)
text(170, 483, "/report /metrics")
text(170, 501, "async worker · SQLite run-store")

box(340, 442, 280, 82)
text(480, 464, "Redis", f_head, FG)
text(480, 483, "content-addressed node cache")
text(480, 501, "JSON-file fallback outside compose")

box(650, 442, 280, 82)
text(790, 464, "Streamlit dashboard", f_head, FG)
text(790, 483, "pure HTTP client of the API")
text(790, 501, "live per-agent trace")

arrow([(310, 483), (338, 483)])
arrow([(620, 483), (648, 483)])
arrow([(480, 414), (480, 440)])

text(30, 560, "One secret-free Docker image serves API + dashboard · docker compose adds Redis",
     f_lbl, MUTED, anchor="lm")
text(30, 580, "Crew scores in the README are deterministic-core runs, not live-LLM runs — see Provenance",
     f_lbl, GREEN, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
