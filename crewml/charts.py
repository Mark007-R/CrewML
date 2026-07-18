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
