"""Draw the ForesightNav framework implemented at commit 6e137e9.

The PDF and SVG outputs remain vector graphics. A PNG is emitted only for
quick visual inspection while editing the figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"


COLORS = {
    "ink": "#20252B",
    "muted": "#68737D",
    "line": "#3D4852",
    "blue": "#3D8FC4",
    "blue_fill": "#EAF4FA",
    "orange": "#D97A24",
    "orange_fill": "#FFF1E3",
    "gold": "#C69A18",
    "gold_fill": "#FFF8DE",
    "navy": "#244F73",
    "navy_fill": "#E9F0F6",
    "green": "#5B8F68",
    "green_fill": "#EDF6EF",
    "teal": "#119A8D",
    "teal_fill": "#E8F7F4",
    "coral": "#C95537",
    "coral_fill": "#FCEDE8",
    "band": "#FAFAF8",
    "band_edge": "#B5BEC5",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def rounded_box(ax, x, y, w, h, edge, face="white", lw=1.15, ls="-", radius=0.12, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        linestyle=ls,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=None, lw=1.15, ls="-", mutation=8, connection="arc3", z=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=ls,
        color=color or COLORS["line"],
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points, color=None, lw=1.1, ls="-", mutation=8, z=3):
    color = color or COLORS["line"]
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, linestyle=ls, zorder=z)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ls=ls, mutation=mutation, z=z + 1)


def section_title(ax, x, text, color=None, y=8.18):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=color or COLORS["ink"],
    )


def draw_lidar_icon(ax, x, y, color):
    for row in range(3):
        for col in range(6):
            ax.add_patch(
                Rectangle(
                    (x + 0.13 * col, y + 0.13 * row),
                    0.09,
                    0.09,
                    facecolor="white" if (row + col) % 3 else color,
                    edgecolor=color,
                    linewidth=0.45,
                    zorder=5,
                )
            )


def draw_dynamic_icon(ax, x, y, color):
    centers = [(x, y), (x + 0.42, y + 0.19), (x + 0.76, y - 0.06)]
    for cx, cy in centers:
        ax.add_patch(Circle((cx, cy), 0.10, facecolor="white", edgecolor=color, linewidth=0.8, zorder=5))
    arrow(ax, (x + 0.07, y + 0.02), (x + 0.28, y + 0.13), color=color, lw=0.7, mutation=5, z=5)
    arrow(ax, (x + 0.49, y + 0.15), (x + 0.67, y), color=color, lw=0.7, mutation=5, z=5)


def draw_navigation_icon(ax, x, y, color):
    ax.add_patch(Circle((x + 0.08, y), 0.07, facecolor=color, edgecolor=color, zorder=5))
    ax.add_patch(Circle((x + 0.72, y + 0.13), 0.10, facecolor="white", edgecolor=color, linewidth=0.8, zorder=5))
    arrow(ax, (x + 0.17, y + 0.02), (x + 0.60, y + 0.11), color=color, lw=0.85, mutation=6, z=5)


def draw_encoder(ax, x, y, w, h, edge, fill, title, output_dim):
    rounded_box(ax, x, y, w, h, edge=edge, face=fill, lw=1.05, ls=(0, (4, 3)), radius=0.12)
    widths = [0.42, 0.34, 0.25]
    heights = [0.64, 0.53, 0.42]
    starts = [x + 0.23, x + 0.82, x + 1.31]
    for sx, pw, ph in zip(starts, widths, heights):
        cy = y + h / 2
        poly = Polygon(
            [
                (sx, cy - ph / 2),
                (sx + pw * 0.82, cy - ph / 2 + 0.05),
                (sx + pw, cy + ph / 2 - 0.05),
                (sx + pw * 0.18, cy + ph / 2),
            ],
            closed=True,
            facecolor=edge,
            edgecolor=COLORS["ink"],
            linewidth=0.55,
            alpha=0.88,
            zorder=4,
        )
        ax.add_patch(poly)
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=6.7, fontweight="bold", color=COLORS["ink"])
    ax.text(x + w - 0.13, y + 0.12, output_dim, ha="right", va="bottom", fontsize=5.8, color=edge)


def draw_network(ax, x, y, w, h, edge, fill, title, subtitle=None):
    rounded_box(ax, x, y, w, h, edge=edge, face=fill, lw=1.05, radius=0.12)
    columns = [3, 4, 2]
    xs = [x + 0.34, x + w / 2, x + w - 0.34]
    nodes = []
    for count, cx in zip(columns, xs):
        ys = [y + 0.27 + i * (h - 0.54) / max(count - 1, 1) for i in range(count)]
        nodes.append([(cx, cy) for cy in ys])
    for left, right in zip(nodes[:-1], nodes[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=edge, linewidth=0.35, alpha=0.7, zorder=3)
    for column in nodes:
        for cx, cy in column:
            ax.add_patch(Circle((cx, cy), 0.065, facecolor="white", edgecolor=edge, linewidth=0.65, zorder=5))
    ax.text(x + w / 2, y + h + 0.14, title, ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    if subtitle:
        ax.text(x + w / 2, y - 0.12, subtitle, ha="center", va="top", fontsize=5.7, color=COLORS["muted"])


def draw_vo_icon(ax, x, y):
    apex = (x, y)
    top = (x + 0.43, y + 0.20)
    bottom = (x + 0.43, y - 0.20)
    ax.add_patch(Polygon([apex, top, bottom], closed=True, facecolor="#F5C8B8", edgecolor=COLORS["coral"], linewidth=0.7, zorder=4))
    ax.add_patch(Circle((x + 0.42, y), 0.075, facecolor="white", edgecolor=COLORS["coral"], linewidth=0.7, zorder=5))
    arrow(ax, (x - 0.20, y - 0.16), (x + 0.24, y + 0.02), color=COLORS["navy"], lw=0.75, mutation=4.5, z=5)


def build_figure():
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Section headings.
    section_title(ax, 1.35, "Local observation", y=8.03)
    section_title(ax, 4.20, "Modality encoders", y=8.03)
    section_title(ax, 7.65, "Shared representation", y=8.03)
    section_title(ax, 11.35, "Predictive actor--critic", y=8.03)
    section_title(ax, 14.45, "Command", y=8.03)

    # Local observations.
    rounded_box(ax, 0.12, 3.35, 2.55, 4.25, edge="#7A4DC2", face="#FCFAFF", lw=1.15, ls=(0, (4, 3)), radius=0.16)

    rounded_box(ax, 0.32, 6.36, 2.15, 0.92, edge=COLORS["blue"], face=COLORS["blue_fill"], lw=0.9)
    draw_lidar_icon(ax, 0.43, 6.61, COLORS["blue"])
    ax.text(1.77, 6.86, r"LiDAR $\mathbf{L}_t$", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["blue"])
    ax.text(1.77, 6.58, r"$36\times4$ proximity", ha="center", va="center", fontsize=5.4, color=COLORS["muted"])

    rounded_box(ax, 0.32, 5.03, 2.15, 0.92, edge=COLORS["orange"], face=COLORS["orange_fill"], lw=0.9)
    ax.text(1.395, 5.55, r"Dynamic obstacles $\mathbf{D}_t$", ha="center", va="center", fontsize=6.1, fontweight="bold", color=COLORS["orange"])
    ax.text(1.395, 5.27, r"$5\times10$ ordered states", ha="center", va="center", fontsize=5.4, color=COLORS["muted"])

    rounded_box(ax, 0.32, 3.70, 2.15, 0.92, edge=COLORS["gold"], face=COLORS["gold_fill"], lw=0.9)
    ax.text(1.395, 4.24, r"Navigation state $\mathbf{s}_t$", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#8C6C0B")
    ax.text(1.395, 3.95, r"goal and UAV motion ($8$D)", ha="center", va="center", fontsize=5.4, color=COLORS["muted"])

    # Encoders and branch arrows.
    draw_encoder(ax, 3.05, 6.18, 2.30, 1.18, COLORS["blue"], COLORS["blue_fill"], "LiDAR CNN", "128-D")
    draw_encoder(ax, 3.05, 4.83, 2.30, 1.18, COLORS["orange"], COLORS["orange_fill"], "Dynamic-obstacle MLP", "64-D")
    arrow(ax, (2.47, 6.82), (3.05, 6.82), color=COLORS["blue"], lw=1.0)
    arrow(ax, (2.47, 5.49), (3.05, 5.42), color=COLORS["orange"], lw=1.0)

    # Concatenation and fusion.
    concat = (5.90, 5.30)
    routed_arrow(ax, [(5.35, 6.77), (5.62, 6.77), (5.62, 5.45), concat], color=COLORS["blue"], lw=0.9)
    arrow(ax, (5.35, 5.42), concat, color=COLORS["orange"], lw=0.9)
    routed_arrow(ax, [(2.47, 4.16), (5.52, 4.16), (5.52, 5.14), concat], color=COLORS["gold"], lw=0.9)
    ax.add_patch(Circle(concat, 0.18, facecolor="white", edgecolor=COLORS["navy"], linewidth=1.0, zorder=6))
    ax.text(*concat, r"$\oplus$", ha="center", va="center", fontsize=8.5, color=COLORS["navy"], zorder=7)

    rounded_box(ax, 6.30, 4.75, 1.58, 1.10, edge=COLORS["navy"], face=COLORS["navy_fill"], lw=1.0)
    ax.text(7.09, 5.46, "Fusion MLP", ha="center", va="center", fontsize=6.7, fontweight="bold", color=COLORS["navy"])
    ax.text(7.09, 5.13, "feature fusion", ha="center", va="center", fontsize=5.7, color=COLORS["muted"])
    arrow(ax, (6.08, 5.30), (6.30, 5.30), color=COLORS["navy"], lw=1.0)

    rounded_box(ax, 8.18, 4.75, 1.02, 1.10, edge=COLORS["navy"], face="white", lw=1.0)
    ax.text(8.69, 5.43, r"$\mathbf{z}_t$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLORS["navy"])
    ax.text(8.69, 5.09, "256-D", ha="center", va="center", fontsize=5.7, color=COLORS["muted"])
    arrow(ax, (7.88, 5.30), (8.18, 5.30), color=COLORS["navy"], lw=1.0)

    # Shared feature bus into actor, critic, and predictive head.
    ax.plot([9.20, 9.58], [5.30, 5.30], color=COLORS["line"], linewidth=1.0, zorder=3)
    ax.plot([9.58, 9.58], [3.41, 6.84], color=COLORS["line"], linewidth=1.0, zorder=3)

    draw_network(ax, 9.88, 6.20, 2.45, 1.15, COLORS["navy"], COLORS["navy_fill"], "Beta policy actor", r"$\alpha_t,\beta_t$")
    draw_network(ax, 9.88, 4.54, 2.45, 1.00, COLORS["green"], COLORS["green_fill"], "Observation-conditioned critic", r"$V_\psi(o_t)$")
    rounded_box(ax, 9.88, 2.78, 2.45, 1.22, edge=COLORS["teal"], face=COLORS["teal_fill"], lw=1.05, ls=(0, (4, 3)), radius=0.12)
    ax.text(11.105, 3.71, "Multi-horizon outcome", ha="center", va="center", fontsize=6.2, fontweight="bold", color=COLORS["teal"])
    ax.text(11.105, 3.50, "predictor", ha="center", va="center", fontsize=6.2, fontweight="bold", color=COLORS["teal"])
    ax.text(11.105, 3.26, r"$h\in\{1,3,5\}$", ha="center", va="center", fontsize=5.8, color=COLORS["ink"])
    ax.text(11.105, 3.03, "collision | stuck | clearance | progress", ha="center", va="center", fontsize=4.9, color=COLORS["muted"])

    arrow(ax, (9.58, 6.78), (9.88, 6.78), color=COLORS["navy"], lw=1.0)
    arrow(ax, (9.58, 5.04), (9.88, 5.04), color=COLORS["green"], lw=1.0)
    arrow(ax, (9.58, 3.39), (9.88, 3.39), color=COLORS["teal"], lw=1.0)

    # Bounded velocity action and action-conditioning branch.
    rounded_box(ax, 13.30, 6.27, 2.43, 1.02, edge=COLORS["navy"], face="white", lw=1.0)
    ax.text(14.515, 6.96, "Bounded 3-D velocity", ha="center", va="center", fontsize=6.6, fontweight="bold", color=COLORS["navy"])
    ax.text(14.515, 6.59, r"$\mathbf{a}_t^G=v_{\rm lim}(2\tilde{\mathbf{a}}_t-1)$", ha="center", va="center", fontsize=6.3, color=COLORS["ink"])
    ax.text(14.515, 6.36, r"$v_{\rm lim}=2\,\mathrm{m/s}$", ha="center", va="center", fontsize=5.3, color=COLORS["muted"])
    arrow(ax, (12.33, 6.78), (13.30, 6.78), color=COLORS["navy"], lw=1.1)
    routed_arrow(
        ax,
        [(12.73, 6.78), (12.73, 4.24), (12.54, 4.24), (12.54, 3.39), (12.33, 3.39)],
        color=COLORS["teal"],
        lw=0.85,
        ls=(0, (3, 2)),
        mutation=7,
    )
    ax.text(12.88, 4.45, r"current $\tilde{\mathbf{a}}_t$", ha="left", va="center", fontsize=5.2, color=COLORS["teal"])

    # Training-only supervision band.
    rounded_box(ax, 0.12, 0.12, 15.66, 2.30, edge=COLORS["band_edge"], face=COLORS["band"], lw=0.95, ls=(0, (4, 3)), radius=0.15, z=0)
    ax.text(0.40, 2.16, "Training-only supervision", ha="left", va="center", fontsize=7.0, fontweight="bold", color=COLORS["coral"])

    rounded_box(ax, 12.72, 0.50, 2.68, 1.35, edge=COLORS["line"], face="white", lw=0.95, radius=0.10)
    ax.text(14.06, 1.52, "Parallel curriculum rollouts", ha="center", va="center", fontsize=6.4, fontweight="bold", color=COLORS["ink"])
    ax.text(14.06, 1.22, "dynamic + non-convex obstacles", ha="center", va="center", fontsize=5.1, color=COLORS["muted"])
    # Minimal environment glyph.
    ax.add_patch(Rectangle((13.28, 0.70), 0.18, 0.35, facecolor=COLORS["blue"], edgecolor=COLORS["navy"], linewidth=0.5, zorder=4))
    ax.add_patch(Rectangle((13.62, 0.70), 0.48, 0.18, facecolor=COLORS["orange"], edgecolor=COLORS["coral"], linewidth=0.5, zorder=4))
    ax.add_patch(Polygon([(14.31, 0.70), (14.31, 1.03), (14.45, 1.03), (14.45, 0.84), (14.65, 0.84), (14.65, 0.70)], closed=False, fill=False, edgecolor=COLORS["gold"], linewidth=1.1, zorder=4))
    ax.add_patch(Circle((14.92, 0.88), 0.09, facecolor=COLORS["orange_fill"], edgecolor=COLORS["orange"], linewidth=0.6, zorder=4))

    rounded_box(ax, 9.75, 1.28, 2.22, 0.76, edge=COLORS["coral"], face=COLORS["coral_fill"], lw=0.95, radius=0.09)
    draw_vo_icon(ax, 9.90, 1.66)
    ax.text(11.28, 1.79, "TTC-aware 3D-VO", ha="center", va="center", fontsize=5.6, fontweight="bold", color=COLORS["coral"])
    ax.text(11.28, 1.53, "risk reward", ha="center", va="center", fontsize=5.2, fontweight="bold", color=COLORS["coral"])

    rounded_box(ax, 9.75, 0.35, 2.22, 0.68, edge=COLORS["teal"], face=COLORS["teal_fill"], lw=0.95, radius=0.09)
    ax.text(10.86, 0.76, "Multi-horizon targets", ha="center", va="center", fontsize=6.0, fontweight="bold", color=COLORS["teal"])
    ax.text(10.86, 0.51, "from rollout sequences", ha="center", va="center", fontsize=5.1, color=COLORS["muted"])

    rounded_box(ax, 7.15, 1.28, 1.95, 0.76, edge=COLORS["navy"], face=COLORS["navy_fill"], lw=0.95, radius=0.09)
    ax.text(8.125, 1.75, r"PPO loss $\mathcal{L}_{\rm PPO}$", ha="center", va="center", fontsize=6.0, fontweight="bold", color=COLORS["navy"])
    ax.text(8.125, 1.49, "policy and value learning", ha="center", va="center", fontsize=5.1, color=COLORS["muted"])

    rounded_box(ax, 7.15, 0.35, 1.95, 0.68, edge=COLORS["teal"], face=COLORS["teal_fill"], lw=0.95, radius=0.09)
    ax.text(8.125, 0.76, r"Outcome loss $\mathcal{L}_{\rm MH}$", ha="center", va="center", fontsize=5.9, fontweight="bold", color=COLORS["teal"])
    ax.text(8.125, 0.51, "predictive supervision", ha="center", va="center", fontsize=5.0, color=COLORS["muted"])

    rounded_box(ax, 3.95, 0.62, 2.32, 1.10, edge=COLORS["line"], face="white", lw=1.0, radius=0.10)
    ax.text(5.11, 1.34, "Joint optimization", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"])
    ax.text(5.11, 1.02, r"$\mathcal{L}=\mathcal{L}_{\rm PPO}+\lambda_{\rm MH}\mathcal{L}_{\rm MH}$", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
    ax.text(5.11, 0.76, "shared predictive representation", ha="center", va="center", fontsize=5.1, color=COLORS["muted"])

    # Training data flow.
    arrow(ax, (12.72, 1.55), (11.97, 1.66), color=COLORS["coral"], lw=0.95)
    arrow(ax, (12.72, 0.83), (11.97, 0.69), color=COLORS["teal"], lw=0.95)
    arrow(ax, (9.75, 1.66), (9.10, 1.66), color=COLORS["coral"], lw=0.95)
    arrow(ax, (9.75, 0.69), (9.10, 0.69), color=COLORS["teal"], lw=0.95)
    arrow(ax, (7.15, 1.66), (6.27, 1.36), color=COLORS["navy"], lw=0.95)
    arrow(ax, (7.15, 0.69), (6.27, 0.94), color=COLORS["teal"], lw=0.95)

    # Policy action closes the rollout loop.
    routed_arrow(ax, [(15.50, 6.27), (15.50, 2.12), (14.92, 2.12), (14.92, 1.85)], color=COLORS["line"], lw=0.85, mutation=7)
    ax.text(15.37, 2.28, "action", ha="right", va="center", fontsize=5.1, color=COLORS["muted"])

    # Joint objective shapes the shared predictive representation. The exact
    # per-head gradient paths are described in the text to keep this figure clear.
    routed_arrow(ax, [(5.11, 1.72), (5.11, 2.63), (8.69, 2.63), (8.69, 4.75)], color=COLORS["navy"], lw=0.9, ls=(0, (3, 2)), mutation=7)
    ax.text(6.90, 2.70, "joint representation update", ha="center", va="bottom", fontsize=5.2, color=COLORS["navy"])

    fig.subplots_adjust(left=0.008, right=0.992, top=0.985, bottom=0.015)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    base = OUT_DIR / "fig_foresightnav_framework"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_name(base.name + "_preview").with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
