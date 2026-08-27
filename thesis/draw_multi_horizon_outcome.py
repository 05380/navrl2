#!/usr/bin/env python3
"""Draw the multi-horizon action-conditioned outcome-learning diagram."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "foresightnav-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"

NAVY = "#173F6B"
BLUE = "#3D7EA6"
TEAL = "#009E86"
TEAL_LIGHT = "#E8F6F2"
ORANGE = "#D97818"
ORANGE_LIGHT = "#FFF2E6"
INK = "#202830"
MUTED = "#6F7C87"
GRID = "#CBD4DB"
PANEL = "#F5F8FA"
WHITE = "#FFFFFF"


def rounded_box(ax, xy, width, height, *, facecolor=WHITE, edgecolor=INK,
                linewidth=1.0, radius=0.02, linestyle="-", zorder=2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color=INK, linewidth=1.15, style="-|>",
          mutation_scale=8, linestyle="-", connectionstyle="arc3,rad=0", zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, *, size=7.4, color=INK, weight="normal",
          ha="center", va="center", style="normal", zorder=5):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def build_figure():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Native IEEE single-column width; vector exports remain editable/scalable.
    fig, ax = plt.subplots(figsize=(3.5, 2.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Figure heading and a compact training-only tag.
    label(ax, 0.025, 0.955, "MULTI-HORIZON OUTCOME LEARNING",
          size=7.2, color=NAVY, weight="bold", ha="left")
    rounded_box(ax, (0.79, 0.925), 0.18, 0.055, facecolor=TEAL_LIGHT,
                edgecolor=TEAL, linewidth=0.8, radius=0.013)
    label(ax, 0.88, 0.953, "TRAINING ONLY", size=5.3, color=TEAL,
          weight="bold")
    ax.plot([0.025, 0.975], [0.91, 0.91], color=GRID, linewidth=0.8)

    # Current representation and the fixed rollout action.
    rounded_box(ax, (0.035, 0.755), 0.205, 0.115, facecolor=PANEL,
                edgecolor=BLUE, linewidth=1.0, radius=0.018)
    label(ax, 0.138, 0.831, "shared representation", size=5.7, color=MUTED)
    label(ax, 0.138, 0.785, r"$\mathbf{z}_t=E_\omega(\mathbf{o}_t)$",
          size=8.4, color=NAVY, weight="bold")

    rounded_box(ax, (0.035, 0.615), 0.205, 0.10, facecolor=PANEL,
                edgecolor=BLUE, linewidth=1.0, radius=0.018)
    label(ax, 0.138, 0.681, "sampled rollout action", size=5.7, color=MUTED)
    label(ax, 0.138, 0.642, r"$\tilde{\mathbf{a}}_t$",
          size=8.3, color=NAVY, weight="bold")

    # Concatenation and action-conditioned decoder.
    concat = Circle((0.292, 0.755), 0.025, facecolor=WHITE,
                    edgecolor=NAVY, linewidth=1.0, zorder=4)
    ax.add_patch(concat)
    label(ax, 0.292, 0.755, r"$\oplus$", size=8.7, color=NAVY)
    arrow(ax, (0.24, 0.812), (0.268, 0.772), color=BLUE, linewidth=1.0)
    arrow(ax, (0.24, 0.665), (0.268, 0.738), color=BLUE, linewidth=1.0)

    rounded_box(ax, (0.34, 0.685), 0.285, 0.145, facecolor=TEAL_LIGHT,
                edgecolor=TEAL, linewidth=1.15, radius=0.022)
    label(ax, 0.4825, 0.790, "action-conditioned", size=5.8,
          color=TEAL, weight="bold")
    label(ax, 0.4825, 0.764, "outcome predictor", size=5.8,
          color=TEAL, weight="bold")
    label(ax, 0.4825, 0.720,
          r"$g_\phi(\mathbf{z}_t,\tilde{\mathbf{a}}_t)$",
          size=8.2, color=NAVY)
    arrow(ax, (0.317, 0.755), (0.34, 0.755), color=TEAL, linewidth=1.2)

    # Horizon-specific predictions.
    horizon_colors = [BLUE, TEAL, ORANGE]
    horizon_fills = ["#EBF3F8", TEAL_LIGHT, ORANGE_LIGHT]
    horizon_y = [0.833, 0.755, 0.677]
    for h, y, color, fill in zip((1, 3, 5), horizon_y, horizon_colors, horizon_fills):
        rounded_box(ax, (0.71, y - 0.031), 0.225, 0.062, facecolor=fill,
                    edgecolor=color, linewidth=0.9, radius=0.014)
        label(ax, 0.742, y, rf"$h={h}$", size=6.0, color=color, weight="bold")
        label(ax, 0.855, y, rf"$\hat{{\mathbf{{y}}}}_t^{{({h})}}$",
              size=8.0, color=INK)
        arrow(ax, (0.625, 0.755), (0.71, y), color=color, linewidth=0.9,
              connectionstyle="arc3,rad=0.08" if h == 1 else (
                  "arc3,rad=-0.08" if h == 5 else "arc3,rad=0"))

    # All prediction heads are collected without crossing the rollout diagram.
    collector_x = 0.958
    for y, color in zip(horizon_y, horizon_colors):
        ax.plot([0.943, collector_x], [y, y], color=color, linewidth=0.8,
                zorder=2)
    ax.plot([collector_x, collector_x], [horizon_y[-1], horizon_y[0]],
            color=MUTED, linewidth=0.9, zorder=2)

    # Collected policy rollout and temporal support.
    label(ax, 0.035, 0.565, "COLLECTED POLICY ROLLOUT", size=6.1,
          color=NAVY, weight="bold", ha="left")
    timeline_y = 0.485
    x_nodes = [0.115, 0.265, 0.415, 0.565, 0.715, 0.865]
    arrow(ax, (x_nodes[0], timeline_y), (x_nodes[-1], timeline_y),
          color=MUTED, linewidth=1.05, mutation_scale=7)
    for idx, x in enumerate(x_nodes):
        if idx == 0:
            face, edge = NAVY, NAVY
            node_label = r"$t$"
        else:
            face, edge = WHITE, MUTED
            node_label = rf"$t+{idx}$"
        ax.add_patch(Circle((x, timeline_y), 0.014, facecolor=face,
                            edgecolor=edge, linewidth=1.0, zorder=4))
        label(ax, x, timeline_y + 0.043, node_label, size=6.3,
              color=INK, weight="bold" if idx == 0 else "normal")
    label(ax, 0.49, timeline_y - 0.043,
          r"future outcomes: $(c_{t+k},\,b_{\mathrm{stuck},t+k},\,f_{t+k},\,\Delta d_{t+k})$",
          size=4.65, color=MUTED)

    # Nested target windows ending at each requested horizon.
    windows = [
        (1, x_nodes[1] - 0.018, x_nodes[1] + 0.018, 0.385, BLUE),
        (3, x_nodes[1], x_nodes[3], 0.330, TEAL),
        (5, x_nodes[1], x_nodes[5], 0.275, ORANGE),
    ]
    for h, x_start, x_end, y, color in windows:
        ax.plot([x_start, x_end], [y, y], color=color, linewidth=1.7,
                solid_capstyle="round", zorder=2)
        ax.plot([x_start, x_start], [y, y + 0.025], color=color,
                linewidth=1.0, zorder=2)
        ax.plot([x_end, x_end], [y, y + 0.025], color=color,
                linewidth=1.0, zorder=2)
        label(ax, x_nodes[1] - 0.052, y, rf"$h={h}$", size=5.8,
              color=color, weight="bold", ha="right")

    # Compact target aggregation used by all horizons.
    rounded_box(ax, (0.055, 0.075), 0.575, 0.125, facecolor=PANEL,
                edgecolor=GRID, linewidth=0.9, radius=0.018)
    label(ax, 0.3425, 0.173, r"target over the $h$-step window: $\mathbf{y}_t^{(h)}$",
          size=6.2, color=MUTED, weight="bold")
    chip_specs = [
        (0.075, r"$\max\,c$"),
        (0.215, r"$\max\,b_{\mathrm{stuck}}$"),
        (0.355, r"$\min\,f$"),
        (0.495, r"$p=\sum \Delta d$"),
    ]
    for x, text in chip_specs:
        rounded_box(ax, (x, 0.095), 0.115, 0.045, facecolor=WHITE,
                    edgecolor=GRID, linewidth=0.65, radius=0.01)
        label(ax, x + 0.0575, 0.1175, text,
              size=5.5 if "stuck" in text else 5.8, color=INK)

    # The nested brackets define the samples used to build each target.
    arrow(ax, (0.565, 0.275), (0.565, 0.200), color=MUTED,
          linewidth=0.8, mutation_scale=6)
    label(ax, 0.575, 0.232, "aggregate", size=4.3, color=MUTED,
          ha="left")

    # Prediction/target matching and representation update.
    rounded_box(ax, (0.69, 0.075), 0.26, 0.125, facecolor=TEAL_LIGHT,
                edgecolor=TEAL, linewidth=1.05, radius=0.018)
    label(ax, 0.82, 0.174, r"multi-horizon loss $\mathcal{L}_{\mathrm{MH}}$",
          size=6.1, color=TEAL, weight="bold")
    label(ax, 0.82, 0.133, r"BCE: $c,b_{\mathrm{stuck}}$", size=5.3, color=INK)
    label(ax, 0.82, 0.101, r"Smooth-L1: $f,p$", size=5.5, color=INK)
    arrow(ax, (0.63, 0.137), (0.69, 0.137), color=TEAL, linewidth=1.05)
    label(ax, 0.66, 0.151, "targets", size=4.2, color=TEAL)

    # Prediction collector follows the right margin and enters the loss from above.
    ax.plot([collector_x, collector_x], [horizon_y[-1], 0.225],
            color=TEAL, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.plot([collector_x, 0.82], [0.225, 0.225], color=TEAL,
            linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    label(ax, 0.89, 0.239, "predictions", size=4.2, color=TEAL)
    arrow(ax, (0.82, 0.225), (0.82, 0.20), color=TEAL, linewidth=0.95,
          linestyle=(0, (3, 2)), mutation_scale=7)

    # Text replaces the ambiguous long feedback arrow from the previous layout.
    label(ax, 0.5, 0.027,
          r"$\mathcal{L}_{\mathrm{MH}}$ updates $E_\omega$ and $g_\phi$; "
          r"rollout actions and targets are detached",
          size=5.0, color=TEAL, weight="bold")

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    base = OUT_DIR / "fig_multi_horizon_outcome_learning"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
