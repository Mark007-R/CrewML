"""Charts for the comparison board — rendered from committed numbers only.

Two figures, each answering one question the Day-12 table answers in text:

* ``day12_holdout_scores.png`` — one panel per dataset, every system's held-out score
  side by side. Panels are per-dataset because the primary metric changes across the
  suite (ROC AUC / macro-F1 / R²); plotting them on one shared axis would imply a
  comparability that does not exist.
* ``day12_crew_deltas.png`` — crew minus each rival, per dataset, around a zero line.
  This is the honest view: bars below zero are losses and are drawn as such.

Like :mod:`crewml.comparison`, nothing here computes a score — the figures are a view
of ``results/comparison_table.json``. A system with no number is left out of the panel
rather than drawn at zero, since a failed run is not a score of nought.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: charts render on a scheduled run with no display
import matplotlib.pyplot as plt

from crewml.comparison import HEADLINE_DELTAS, METRIC_LABEL, SYSTEMS
from crewml.config import RESULTS_DIR

CHARTS_DIR = RESULTS_DIR / "charts"
SCORES_CHART_PATH = CHARTS_DIR / "day12_holdout_scores.png"
DELTAS_CHART_PATH = CHARTS_DIR / "day12_crew_deltas.png"

# The crew is the subject of every chart — give it the one saturated colour.
SYSTEM_COLOURS = {
    "dummy": "#c9ced6",
    "default_rf": "#9aa5b4",
    "solo_agent": "#6b7a8f",
    "automl_flaml": "#41618c",
    "crew": "#d1495b",
}

WIN_COLOUR = "#2a9d8f"
LOSS_COLOUR = "#d1495b"

# On the deltas chart, colour already carries win/loss, so the *rival* is carried by
# hatch instead. Without this the three comparisons are indistinguishable — and a
# dataset where one rival is missing (solo failed on vehicle/kin8nm) leaves a gap the
# reader cannot otherwise attribute.
DELTA_HATCHES = ["", "//", "xx"]


def _clean_label(label: str) -> str:
    """Table labels carry markdown bold; axes do not."""
    return label.replace("*", "")


def plot_holdout_scores(table: dict, path: Path = SCORES_CHART_PATH) -> Path:
    """One panel per dataset: every system's held-out score, crew highlighted."""
    keys = list(table["rows"])
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 4.1))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        row = table["rows"][key]
        names, values, colours = [], [], []
        for system, label in SYSTEMS:
            value = row["systems"][system]["value"]
            if value is None:  # a failed system is absent, never a zero bar
                continue
            names.append(_clean_label(label))
            values.append(value)
            colours.append(SYSTEM_COLOURS[system])

        bars = ax.bar(range(len(values)), values, color=colours)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(key, fontsize=10, fontweight="bold")
        ax.set_ylabel(METRIC_LABEL.get(row["metric"], row["metric"]), fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.16)

    fig.suptitle("CrewML — held-out scores by system (higher is better)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_crew_deltas(table: dict, path: Path = DELTAS_CHART_PATH) -> Path:
    """Crew minus each rival per dataset — bars below zero are losses, drawn as losses."""
    keys = list(table["rows"])
    width = 0.8 / len(HEADLINE_DELTAS)
    fig, ax = plt.subplots(figsize=(1.7 * len(keys) + 3.4, 4.3))

    for i, (name, _, label) in enumerate(HEADLINE_DELTAS):
        xs, ys = [], []
        for j, key in enumerate(keys):
            d = table["deltas"][key][name]
            if d is None:  # no comparison possible — leave the slot empty, never a 0 bar
                continue
            # Slot position is fixed per comparison, so a missing rival reads as a gap
            # in that rival's slot rather than shifting the remaining bars.
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(d)
        colours = [WIN_COLOUR if y > 0 else LOSS_COLOUR for y in ys]
        bars = ax.bar(xs, ys, width=width, color=colours, label=_clean_label(label),
                      edgecolor="white", linewidth=0.5, hatch=DELTA_HATCHES[i])
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{y:+.3f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=6.5)

    ax.axhline(0, color="#333", linewidth=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("crew − rival (primary metric)", fontsize=9)
    ax.set_title("CrewML — crew minus each rival on the locked holdout\n"
                 "(green = crew wins, red = crew loses)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.2)

    # Two legends: hatch identifies the rival, colour identifies the outcome. Colour
    # cannot key the rival here because it is already spent on win/loss.
    rival_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#dfe3e8", edgecolor="#41618c", hatch=h)
        for h in DELTA_HATCHES[:len(HEADLINE_DELTAS)]
    ]
    rival_legend = ax.legend(
        rival_handles, [_clean_label(l) for _, _, l in HEADLINE_DELTAS],
        fontsize=7, frameon=False, ncol=len(HEADLINE_DELTAS), loc="upper center",
        bbox_to_anchor=(0.5, -0.08), title="comparison (hatch)", title_fontsize=7,
    )
    ax.add_artist(rival_legend)

    outcome_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=WIN_COLOUR),
        plt.Rectangle((0, 0), 1, 1, facecolor=LOSS_COLOUR),
    ]
    ax.legend(outcome_handles, ["crew wins", "crew loses"], fontsize=7, frameon=False,
              ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              title="outcome (colour)", title_fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(table: dict) -> list[Path]:
    """Render every Day-12 figure; returns the written paths."""
    return [plot_holdout_scores(table), plot_crew_deltas(table)]


# --- Day 13: the Critic-loop ablation ---------------------------------------

CRITIC_ABLATION_CHART_PATH = CHARTS_DIR / "day13_critic_ablation.png"

LOOPED_COLOUR = "#d1495b"      # the full crew — the saturated colour, as everywhere
NO_CRITIC_COLOUR = "#9aa5b4"   # the ablated variant — muted, a baseline


def _ablation_panel(ax, study: dict, title: str) -> None:
    """One panel: paired looped-vs-ablated holdout bars per dataset (absent = failed)."""
    keys = list(study)
    x = range(len(keys))
    width = 0.38

    looped_v = [study[k]["looped"]["value"] if study[k]["looped"].get("ok") else 0.0 for k in keys]
    nc_v = [study[k]["no_critic"]["value"] if study[k]["no_critic"].get("ok") else 0.0 for k in keys]

    b1 = ax.bar([i - width / 2 for i in x], looped_v, width, color=LOOPED_COLOUR, label="Looped (full crew)")
    b2 = ax.bar([i + width / 2 for i in x], nc_v, width, color=NO_CRITIC_COLOUR, label="No-Critic (ablated)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

    # Annotate the loop's contribution above each dataset where the loop actually fired.
    for i, k in enumerate(keys):
        drop = study[k]["loop_drop"]
        if isinstance(drop, (int, float)) and drop > 1e-9:
            ax.annotate(f"loop +{drop:.3f}", (i, max(looped_v[i], nc_v[i])),
                        textcoords="offset points", xytext=(0, 14), ha="center",
                        fontsize=7, fontweight="bold", color="#2a9d8f")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k}\n({study[k]['metric']})" for k in keys], fontsize=7.5)
    ax.set_ylabel("held-out score (higher is better)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")


def plot_critic_ablation(report: dict, path: Path = CRITIC_ABLATION_CHART_PATH) -> Path:
    """Two panels — natural ablation (loop rarely fires) and the forced-deficiency probe
    (loop recovers the crippled first pass). Rendered from committed numbers only.
    """
    natural = report.get("natural") or {}
    probe = report.get("deficiency_probe") or {}
    n_panels = 1 + (1 if probe else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(1.9 * (len(natural) + len(probe)) + 3.5, 4.4))
    if n_panels == 1:
        axes = [axes]

    _ablation_panel(axes[0], natural, "Natural — real datasets\n(Critic finalises pass 1; loop is free)")
    if probe:
        _ablation_panel(axes[1], probe, "Forced deficiency — crippled first pass\n(loop diagnoses underfit and recovers)")

    fig.suptitle("CrewML — does the Critic loop earn its keep?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Day 14: per-agent ablations (Planner / Feature Engineer) ----------------

AGENT_ABLATION_CHART_PATH = CHARTS_DIR / "day14_agent_ablation.png"

NO_PLANNER_COLOUR = "#9aa5b4"   # naive-floor arms — muted, like every ablated baseline
NO_FE_COLOUR = "#c8b8a2"


def plot_agent_ablation(report: dict, path: Path = AGENT_ABLATION_CHART_PATH) -> Path:
    """One panel: per dataset, the full crew vs. its two single-agent removals.

    Bars are held-out scores (absent = that arm failed to ship); the annotation above
    each dataset names the larger of the two agent drops, sign included — a negative
    number (naive floor won) is drawn as prominently as a positive one.
    """
    study = report.get("results") or {}
    keys = list(study)
    x = range(len(keys))
    width = 0.26

    def _val(k, arm):
        rec = study[k]["arms"][arm]
        return rec["value"] if rec.get("ok") else 0.0

    fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 3.5, 4.6))
    bars = [
        ax.bar([i - width for i in x], [_val(k, "full") for k in keys], width,
               color=LOOPED_COLOUR, label="Full crew"),
        ax.bar(list(x), [_val(k, "no_planner") for k in keys], width,
               color=NO_PLANNER_COLOUR, label="No Planner (naive plan)"),
        ax.bar([i + width for i in x], [_val(k, "no_feature_engineer") for k in keys], width,
               color=NO_FE_COLOUR, label="No Feature Engineer (raw features)"),
    ]
    for group in bars:
        for bar in group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6)

    for i, k in enumerate(keys):
        drops = study[k]["drops"]
        named = [(a, d) for a, d in drops.items() if isinstance(d, (int, float))]
        if not named:
            continue
        agent, drop = max(named, key=lambda t: abs(t[1]))
        if abs(drop) <= 1e-9:
            continue
        colour = "#2a9d8f" if drop > 0 else "#b3423a"
        label = {"planner": "planner", "feature_engineer": "FE"}[agent]
        top = max(_val(k, arm) for arm in ("full", "no_planner", "no_feature_engineer"))
        ax.annotate(f"{label} {drop:+.3f}", (i, top),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=7, fontweight="bold", color=colour)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k}\n({study[k]['metric']})" for k in keys], fontsize=7.5)
    ax.set_ylabel("held-out score (higher is better)", fontsize=8)
    ax.set_title("CrewML — what do the Planner and the Feature Engineer earn?\n"
                 "(each arm removes exactly one specialist; drop = full − ablated)",
                 fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.22)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Day 15: iteration-depth study (score vs Critic-loop budget) -------------

ITERATION_DEPTH_CHART_PATH = CHARTS_DIR / "day15_iteration_depth.png"

# One line per probe dataset; the crew red stays the "subject" colour.
DEPTH_CURVE_COLOURS = ["#d1495b", "#41618c", "#2a9d8f", "#c8842a"]


def plot_iteration_depth(report: dict, path: Path = ITERATION_DEPTH_CHART_PATH) -> Path:
    """Two panels from the committed Day-15 numbers.

    Left — the deficiency-sweep depth-response: held-out score vs iteration budget, one
    line per probe dataset, budget-bound points ringed (the crew was cut off there, not
    done) and each point annotated with the passes actually used. The natural-arm score
    for the same dataset is drawn as a dashed reference — the ceiling a healthy first
    pass reaches without needing the loop at all.

    Right — the price of each budget step: marginal held-out lift per added unit of
    budget. A failed/absent point is left out, never drawn at zero.
    """
    probe = report.get("analysis", {}).get("deficiency_probe") or {}
    natural = report.get("analysis", {}).get("natural") or {}

    fig, (ax_curve, ax_marg) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for i, (key, rows) in enumerate(probe.items()):
        colour = DEPTH_CURVE_COLOURS[i % len(DEPTH_CURVE_COLOURS)]
        pts = [(r["depth"], r["value"], r) for r in rows if r["value"] is not None]
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax_curve.plot(xs, ys, marker="o", color=colour, label=key, linewidth=2, zorder=3)
        for x, y, r in pts:
            if r.get("budget_bound"):
                ax_curve.scatter([x], [y], s=170, facecolors="none",
                                 edgecolors=colour, linewidths=1.6, zorder=4)
            ax_curve.annotate(f"{r['passes']}p", (x, y), textcoords="offset points",
                              xytext=(0, -14), ha="center", fontsize=6.5, color=colour)
        nat_rows = natural.get(key) or []
        nat_vals = [r["value"] for r in nat_rows if r["value"] is not None]
        if nat_vals:
            ax_curve.axhline(nat_vals[-1], color=colour, linestyle="--",
                             linewidth=1, alpha=0.55, zorder=2)

    ax_curve.set_xlabel("iteration budget (max Critic passes)", fontsize=8)
    ax_curve.set_ylabel("held-out score", fontsize=8)
    ax_curve.set_title("Deficiency sweep — score vs budget\n(ringed = budget-bound: cut off, not done;\n"
                       "dashed = natural-run ceiling; 'Np' = passes used)",
                       fontsize=9, fontweight="bold")
    depths = sorted({r["depth"] for rows in probe.values() for r in rows})
    if depths:
        ax_curve.set_xticks(depths)
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(fontsize=7.5, frameon=False, loc="lower right")

    # Right panel: marginal lift per budget step, grouped by dataset.
    steps = [d for d in depths[1:]]
    width = 0.8 / max(len(probe), 1)
    for i, (key, rows) in enumerate(probe.items()):
        colour = DEPTH_CURVE_COLOURS[i % len(DEPTH_CURVE_COLOURS)]
        by_depth = {r["depth"]: r for r in rows}
        xs, ys = [], []
        for j, d in enumerate(steps):
            m = (by_depth.get(d) or {}).get("marginal_lift")
            if isinstance(m, (int, float)):
                xs.append(j + i * width)
                ys.append(m)
        bars = ax_marg.bar(xs, ys, width * 0.92, color=colour, label=key)
        for bar in bars:
            h = bar.get_height()
            ax_marg.text(bar.get_x() + bar.get_width() / 2, h, f"{h:+.3f}",
                         ha="center", va="bottom" if h >= 0 else "top", fontsize=6.5)

    ax_marg.axhline(0, color="#444444", linewidth=0.8)
    ax_marg.set_xticks([j + width * (len(probe) - 1) / 2 for j in range(len(steps))])
    ax_marg.set_xticklabels([f"{d - 1}→{d}" for d in steps], fontsize=8)
    ax_marg.set_xlabel("budget step", fontsize=8)
    ax_marg.set_ylabel("marginal held-out lift", fontsize=8)
    ax_marg.set_title("The price of each extra loop\n(first allowed loop buys the cliff; the rest buy ~0)",
                      fontsize=9, fontweight="bold")
    ax_marg.spines[["top", "right"]].set_visible(False)
    ax_marg.margins(y=0.25)
    ax_marg.legend(fontsize=7.5, frameon=False)

    fig.suptitle("CrewML — how deep should the Critic loop go?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Day 16: provider study (Groq vs Claude vs mock) --------------------------

PROVIDER_STUDY_CHART_PATH = CHARTS_DIR / "day16_provider_study.png"

PROBE_STATUS_COLOURS = {
    "ok": "#2a9d8f",
    "offline": "#41618c",
    "not_configured": "#c8842a",
    "error": "#d1495b",
}
ARM_COLOURS = {"groq": "#c8842a", "anthropic": "#41618c", "mock": "#d1495b"}


def plot_provider_study(report: dict, path: Path = PROVIDER_STUDY_CHART_PATH) -> Path:
    """Two panels from the committed Day-16 numbers.

    Left — provider availability: one bar per probed provider, colour = probe status.
    Unavailable providers are drawn *as unavailable* rather than omitted: the outage is
    the finding this session, so the chart must show it.

    Right — holdout score per dataset for each arm that actually ran, side by side with
    the Day-14 archival failing-provider baseline. Identical bars are the resilience
    result made visible. Absent = that arm failed to ship; never drawn at zero.
    """
    probes = report.get("probes") or {}
    arms = report.get("arms") or {}
    res_rows = (report.get("resilience") or {}).get("rows") or []

    fig, (ax_probe, ax_scores) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1, 2.2]}
    )

    names = list(probes)
    colours = [PROBE_STATUS_COLOURS.get(probes[n]["status"], "#9aa5b4") for n in names]
    # Availability is categorical — the bar height (1) carries nothing; colour + the
    # status label on the bar carry everything.
    bars = ax_probe.bar(range(len(names)), [1] * len(names), color=colours)
    for bar, n in zip(bars, names):
        p = probes[n]
        label = {"ok": "OK", "offline": "always\navailable", "not_configured": "no key",
                 "error": "restricted /\nunavailable"}.get(p["status"], p["status"])
        ax_probe.text(bar.get_x() + bar.get_width() / 2, 0.5, label, ha="center",
                      va="center", fontsize=8, fontweight="bold", color="white")
        if p.get("latency_s") is not None:
            ax_probe.text(bar.get_x() + bar.get_width() / 2, 1.02,
                          f"{p['latency_s']:.2f}s", ha="center", va="bottom", fontsize=7)
    ax_probe.set_xticks(range(len(names)))
    ax_probe.set_xticklabels(names, fontsize=8)
    ax_probe.set_yticks([])
    ax_probe.set_title("Provider availability\n(live probes, committed evidence)",
                       fontsize=9, fontweight="bold")
    ax_probe.spines[["top", "right", "left"]].set_visible(False)
    ax_probe.margins(y=0.15)

    # Right panel: per-dataset holdout bars — each run arm plus the archival baseline.
    series: list[tuple[str, dict[str, float], str, dict]] = []
    for provider, arm in arms.items():
        vals = {k: r["value"] for k, r in arm.items() if r.get("ok")}
        series.append((f"{provider} (fresh)", vals, ARM_COLOURS.get(provider, "#9aa5b4"), {}))
    arch_vals = {
        r["dataset"]: r["archival_failing_provider_value"]
        for r in res_rows if r["archival_failing_provider_value"] is not None
    }
    if arch_vals:
        series.append(("day-14 archival (provider failing)", arch_vals, "#9aa5b4", {"hatch": "//"}))

    keys = sorted({k for _, vals, _, _ in series for k in vals})
    width = 0.8 / max(len(series), 1)
    for i, (label, vals, colour, extra) in enumerate(series):
        xs = [j + i * width - 0.4 + width / 2 for j, k in enumerate(keys) if k in vals]
        ys = [vals[k] for k in keys if k in vals]
        bars = ax_scores.bar(xs, ys, width * 0.92, color=colour, label=label,
                             edgecolor="white", linewidth=0.5, **extra)
        for bar, y in zip(bars, ys):
            ax_scores.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}",
                           ha="center", va="bottom", fontsize=6)
    ax_scores.set_xticks(range(len(keys)))
    ax_scores.set_xticklabels(keys, fontsize=8)
    ax_scores.set_ylabel("held-out score (higher is better)", fontsize=8)
    ax_scores.set_title("Holdout quality by arm — identical bars = provider-independent core\n"
                        "(no live-provider arm ran this session; mock is never a headline)",
                        fontsize=9, fontweight="bold")
    ax_scores.spines[["top", "right"]].set_visible(False)
    ax_scores.margins(y=0.2)
    ax_scores.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("CrewML — provider study: availability, quality, resilience",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Day 17: failure taxonomy ------------------------------------------------

FAILURE_TAXONOMY_CHART_PATH = CHARTS_DIR / "day17_failure_taxonomy.png"

# Outcome severity is the story of the taxonomy chart, so colour carries outcome:
# red = fatal (no model), amber = degraded (shipped, knowably worse), green = handled
# (a guard absorbed it), slate = detected (impact indeterminate), purple = missed
# (ground-truth fault, no surface fired — only ever assignable by a probe).
OUTCOME_COLOURS = {
    "fatal": "#d1495b",
    "degraded": "#e8a23d",
    "handled": "#2a9d8f",
    "detected": "#6b7a8f",
    "missed": "#7b4b94",
}


def plot_failure_taxonomy(report: dict, path: Path = FAILURE_TAXONOMY_CHART_PATH) -> Path:
    """Two panels: the archive census by category (stacked by outcome), and the
    fatal-count-by-system headline. Probe-sourced ``missed`` events are merged into
    the left panel so the one gap the probes found is visible next to the census."""
    census = report["archive_census"]["summary"]
    probe_summary = report["probes"]["summary"]

    # Merge census + probe categories (probes contribute e.g. leakage_missed).
    merged: dict[str, dict] = {}
    for src in (census["by_category"], probe_summary["by_category"]):
        for code, c in src.items():
            slot = merged.setdefault(code, {"by_outcome": {}})
            for outcome, n in c["by_outcome"].items():
                slot["by_outcome"][outcome] = slot["by_outcome"].get(outcome, 0) + n
    codes = sorted(merged, key=lambda k: -sum(merged[k]["by_outcome"].values()))

    fig, (ax_census, ax_fatal) = plt.subplots(
        1, 2, figsize=(11.5, 0.42 * max(len(codes), 6) + 2.6),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    outcomes = [o for o in OUTCOME_COLOURS if any(merged[c]["by_outcome"].get(o) for c in codes)]
    left = [0.0] * len(codes)
    for outcome in outcomes:
        widths = [merged[c]["by_outcome"].get(outcome, 0) for c in codes]
        ax_census.barh(range(len(codes)), widths, left=left,
                       color=OUTCOME_COLOURS[outcome], label=outcome, height=0.62)
        left = [l + w for l, w in zip(left, widths)]
    for i, total in enumerate(left):
        ax_census.text(total + 0.15, i, str(int(total)), va="center", fontsize=7)
    ax_census.set_yticks(range(len(codes)))
    ax_census.set_yticklabels(codes, fontsize=8, family="monospace")
    ax_census.invert_yaxis()
    ax_census.set_xlabel("events (archive census + probes)", fontsize=8)
    ax_census.set_title("Failure events by category and outcome", fontsize=10, fontweight="bold")
    ax_census.spines[["top", "right"]].set_visible(False)
    ax_census.legend(fontsize=7, frameon=False, loc="lower right")

    fatal = census["fatal_by_system"]
    systems = ["crew", "solo"]
    values = [fatal.get(s, 0) for s in systems]
    bars = ax_fatal.bar(systems, values, color=[SYSTEM_COLOURS["crew"], SYSTEM_COLOURS["solo_agent"]])
    for bar, v in zip(bars, values):
        ax_fatal.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                      ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_fatal.set_title("Fatal failures\n(no scored model)", fontsize=10, fontweight="bold")
    ax_fatal.set_ylim(0, max(values + [1]) * 1.3)
    ax_fatal.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CrewML — Day 17 failure taxonomy: what fails, who catches it, what it costs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


PHASE3_SUMMARY_CHART_PATH = CHARTS_DIR / "day18_phase3_summary.png"


def plot_phase3_summary(report: dict, path: Path = PHASE3_SUMMARY_CHART_PATH) -> Path:
    """The Day-18 consolidation figure: the phase's four headline findings on one
    canvas, each panel a compact restatement of one study's committed chart.

    Like :mod:`crewml.phase3_results`, this draws from the consolidated report only —
    a rival with no score leaves a gap, a probe-only number is hatched as such, and
    the ``missed`` leakage probe is drawn in the failure panel rather than omitted.
    """
    board = report["board"]
    agents = report["agents"]
    critic = report["critic"]
    depth = report["depth"]
    fail = report["failures"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax_delta, ax_attr, ax_depth, ax_fail = axes.flat

    # --- Panel 1: crew − rival deltas (Day 12) ---
    keys = list(board["rows"])
    width = 0.8 / len(HEADLINE_DELTAS)
    for i, (name, _, label) in enumerate(HEADLINE_DELTAS):
        for j, key in enumerate(keys):
            d = board["rows"][key]["deltas"].get(name)
            if d is None:  # no comparison — a gap, never a zero bar
                continue
            ax_delta.bar(j + i * width - 0.4 + width / 2, d, width=width,
                         color=WIN_COLOUR if d > 0 else LOSS_COLOUR,
                         edgecolor="white", linewidth=0.5, hatch=DELTA_HATCHES[i])
    ax_delta.axhline(0, color="#333", linewidth=1)
    ax_delta.set_xticks(range(len(keys)))
    ax_delta.set_xticklabels(keys, fontsize=7)
    ax_delta.set_title("Crew − rival, locked holdout (Day 12)\n"
                       "hatch: none=solo, //=AutoML, xx=default RF",
                       fontsize=9, fontweight="bold")
    ax_delta.spines[["top", "right"]].set_visible(False)
    ax_delta.margins(y=0.15)

    # --- Panel 2: per-agent attribution (Days 13-14) ---
    drops = agents["drops_by_dataset"]
    probe_rec = critic["probe_recovery_by_dataset"]
    akeys = list(drops)
    aw = 0.26
    for j, key in enumerate(akeys):
        p = drops[key].get("planner")
        f = drops[key].get("feature_engineer")
        c = probe_rec.get(key)
        if p is not None:
            ax_attr.bar(j - aw, p, width=aw, color=SYSTEM_COLOURS["automl_flaml"],
                        label="Planner drop" if j == 0 else None)
        if f is not None:
            ax_attr.bar(j, f, width=aw, color=SYSTEM_COLOURS["solo_agent"],
                        label="FE drop" if j == 0 else None)
        if c is not None:  # probe-sourced, hatched: an instrumented number, not natural
            ax_attr.bar(j + aw, c, width=aw, color=SYSTEM_COLOURS["crew"],
                        hatch="//", edgecolor="white",
                        label="Critic recovery (probe)" if key == list(probe_rec)[0] else None)
    ax_attr.axhline(0, color="#333", linewidth=1)
    ax_attr.set_xticks(range(len(akeys)))
    ax_attr.set_xticklabels(akeys, fontsize=7)
    ax_attr.set_title("What removing each agent costs (Days 13–14)\n"
                      "drop = full − ablated; Critic bar is the forced-deficiency probe",
                      fontsize=9, fontweight="bold")
    ax_attr.legend(fontsize=7, frameon=False)
    ax_attr.spines[["top", "right"]].set_visible(False)

    # --- Panel 3: iteration-depth cliff (Day 15) ---
    for key, curve in depth["probe_curves"].items():
        ds = sorted(curve)
        ax_depth.plot(ds, [curve[d] for d in ds], marker="o", linewidth=2, label=key)
        bound = depth["probe_summary"].get(key, {}).get("budget_bound_depths", [])
        for b in bound:
            if b in curve:
                ax_depth.plot([b], [curve[b]], marker="x", markersize=11,
                              color=LOSS_COLOUR, markeredgewidth=2.5)
    ax_depth.set_xlabel("Critic-loop budget (max_iterations)", fontsize=8)
    ax_depth.set_ylabel("holdout R²", fontsize=8)
    ax_depth.set_xticks(sorted({d for c in depth["probe_curves"].values() for d in c}))
    ax_depth.set_title("Depth-response under forced deficiency (Day 15)\n"
                       "× = budget-bound (cut off, not done); natural sweep is flat",
                       fontsize=9, fontweight="bold")
    ax_depth.legend(fontsize=7, frameon=False, loc="lower right")
    ax_depth.spines[["top", "right"]].set_visible(False)

    # --- Panel 4: failure outcomes + fatal headline (Day 17) ---
    outcomes = [o for o in OUTCOME_COLOURS if fail["by_outcome"].get(o)]
    values = [fail["by_outcome"][o] for o in outcomes]
    bars = ax_fail.bar(outcomes, values, color=[OUTCOME_COLOURS[o] for o in outcomes])
    for bar, v in zip(bars, values):
        ax_fail.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    fatal = fail["fatal_by_system"]
    ax_fail.set_title(
        f"Failure outcomes, {fail['n_events']} archived events (Day 17)\n"
        f"fatal: crew {fatal.get('crew', 0)} vs solo {fatal.get('solo', 0)} "
        f"(of {fail['n_crew_runs']} / {fail['n_solo_runs']} runs)",
        fontsize=9, fontweight="bold")
    ax_fail.set_yscale("log")  # handled dwarfs the rest; log keeps the rare outcomes legible
    ax_fail.set_ylabel("events (log)", fontsize=8)
    ax_fail.tick_params(axis="x", labelsize=7)
    ax_fail.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CrewML — Phase 3 consolidated results (Days 12–17)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


SELF_REPAIR_CHART_PATH = CHARTS_DIR / "day20_self_repair.png"

# Attempt-count colours for the Day-20 repair chart: recovered first try, needed
# the second, or the budget was spent and the crash stood.
REPAIR_COLOURS = {1: "#2a9d8f", 2: "#e9c46a", None: "#d1495b"}


def plot_self_repair(report: dict, path: Path = SELF_REPAIR_CHART_PATH) -> Path:
    """Two panels from ``results/day20_self_repair.json``: which injected faults the
    repair loop recovered (and on which attempt), and how faithfully the repaired
    runs reproduce the clean run's CV score. Controls are excluded from the left
    panel (nothing was injected) but their scores anchor the fidelity deltas."""
    injected = [r for r in report["runs"] if r["fault"] != "none_control"]
    datasets = report["datasets"]
    faults = list(dict.fromkeys(r["fault"] for r in injected))
    by = {(r["dataset"], r["fault"]): r for r in injected}

    fig, (ax_rec, ax_fid) = plt.subplots(
        1, 2, figsize=(11.5, 0.5 * len(faults) + 2.8),
        gridspec_kw={"width_ratios": [5, 4]},
    )

    # --- Left: recovery matrix — one marker per (fault, dataset) ------------
    bar_h = 0.36
    for j, ds in enumerate(datasets):
        for i, fault in enumerate(faults):
            r = by.get((ds, fault))
            if r is None:
                continue
            attempt = r["recovered_on_attempt"] if r["recovered"] else None
            y = i + (j - (len(datasets) - 1) / 2) * bar_h
            width = attempt if attempt else report["max_attempts"]
            ax_rec.barh(y, width, height=bar_h * 0.9,
                        color=REPAIR_COLOURS.get(attempt, REPAIR_COLOURS[None]))
            label = f"attempt {attempt}" if attempt else "NOT recovered"
            ax_rec.text(width + 0.05, y, f"{ds}: {label}", va="center", fontsize=7)
    ax_rec.set_yticks(range(len(faults)))
    ax_rec.set_yticklabels(faults, fontsize=8, family="monospace")
    ax_rec.invert_yaxis()
    ax_rec.set_xlim(0, report["max_attempts"] + 1.6)
    ax_rec.set_xticks(range(0, report["max_attempts"] + 1))
    ax_rec.set_xlabel("repair attempts consumed", fontsize=8)
    rate = report["recovery_rate"]
    ax_rec.set_title(
        f"Recovery per injected fault — rate {report['recovered_runs']}"
        f"/{report['n_injected_runs']} = {rate:.0%}",
        fontsize=10, fontweight="bold",
    )
    ax_rec.spines[["top", "right"]].set_visible(False)

    # --- Right: score fidelity of recovered runs vs the clean control -------
    recovered = [r for r in injected if r["recovered"]
                 and r["score_fidelity_vs_clean"] is not None]
    ys, vals, labels = [], [], []
    for i, r in enumerate(recovered):
        ys.append(i)
        vals.append(r["score_fidelity_vs_clean"])
        labels.append(f"{r['dataset']}: {r['fault']}")
    ax_fid.barh(ys, vals, height=0.6,
                color=[WIN_COLOUR if v >= 0 else LOSS_COLOUR for v in vals])
    ax_fid.axvline(0, color="#333", lw=0.8)
    ax_fid.set_yticks(ys)
    ax_fid.set_yticklabels(labels, fontsize=7, family="monospace")
    ax_fid.invert_yaxis()
    # Exact fidelity (every Δ == 0) is the *good* outcome, and also a degenerate
    # axis — floor the span so the zero line stays visible instead of singular.
    span = max(max((abs(v) for v in vals), default=0.0), 0.001)
    ax_fid.set_xlim(-span * 1.6, span * 1.6)
    ax_fid.set_xlabel("repaired CV score − clean CV score", fontsize=8)
    ax_fid.set_title(
        f"Score fidelity of repaired runs\n(mean |Δ| = "
        f"{report['mean_abs_score_fidelity']})",
        fontsize=10, fontweight="bold",
    )
    ax_fid.spines[["top", "right"]].set_visible(False)

    # The honesty label travels with the figure: a chart gets separated from its
    # JSON and its report, and a scripted-mode panel read as a capability result
    # would be exactly the misreading the study's stamp exists to prevent.
    if report.get("is_measurement_of_llm_capability", True):
        fig.suptitle(
            "CrewML — Day 20 self-repair: crashed generated code reads its own "
            "traceback and comes back",
            fontsize=12, fontweight="bold",
        )
    else:
        fig.suptitle(
            "CrewML — Day 20 self-repair MECHANISM check (deterministic "
            "stand-in repairer — NOT an LLM measurement)\n"
            "Injected faults detonate in the real Trainer, the loop fires, the "
            "sandboxed re-run is adopted. Says nothing about model repair skill.",
            fontsize=11, fontweight="bold",
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
