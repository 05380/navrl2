#!/usr/bin/env python3
"""Overlay the four approved real-flight 3D-speed traces on one figure."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    color: str
    csv_path: Path
    start_s: float | None = None
    end_s: float | None = None


DEFAULT_SERIES = (
    SeriesSpec(
        "Static",
        "#D62728",
        REPO_ROOT
        / "analysis_outputs/cloud_velocity_01_4_env1/cloud_velocity_01.4_velocity.csv",
    ),
    SeriesSpec(
        "Dynamic",
        "#F28E2B",
        REPO_ROOT
        / "analysis_outputs/cloud_velocity_01_env2/cloud_velocity_01_velocity.csv",
    ),
    SeriesSpec(
        "Mixed",
        "#2CA02C",
        REPO_ROOT
        / "analysis_outputs/cloud_velocity_02_b1_env3/cloud_velocity_02.b1_velocity.csv",
        start_s=9.839821312,
        end_s=79.129344896,
    ),
    SeriesSpec(
        "Non-convex",
        "#1769AA",
        REPO_ROOT / "analysis_outputs/cloud_velocity_03/cloud_velocity_03_velocity.csv",
        start_s=40.0,
    ),
)


def read_speed_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"CSV file not found: {path}")

    elapsed: list[float] = []
    speed: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"elapsed_s", "speed_3d_mps"}
        if not required.issubset(reader.fieldnames or ()):
            raise SystemExit(f"{path} does not contain columns {sorted(required)}")
        for row in reader:
            elapsed.append(float(row["elapsed_s"]))
            speed.append(float(row["speed_3d_mps"]))

    return np.asarray(elapsed, dtype=float), np.asarray(speed, dtype=float)


def displayed_series(spec: SeriesSpec) -> tuple[np.ndarray, np.ndarray]:
    elapsed, speed = read_speed_csv(spec.csv_path)
    finite = np.isfinite(elapsed) & np.isfinite(speed)
    if spec.start_s is not None:
        finite &= elapsed >= spec.start_s
    if spec.end_s is not None:
        finite &= elapsed <= spec.end_s

    elapsed = elapsed[finite]
    speed = speed[finite]
    if elapsed.size == 0:
        raise SystemExit(f"No samples remain for {spec.label}")

    # Every previously approved single-environment graph starts its displayed
    # x-axis at zero after applying that environment's selected time window.
    return elapsed - elapsed[0], speed


def parse_args() -> argparse.Namespace:
    default_png = (
        REPO_ROOT
        / "analysis_outputs/env_speed_comparison/env1_env4_3d_speed_comparison.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_png)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=default_png.with_suffix(".pdf"),
        help="PDF copy; pass an empty string to disable it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = [(spec, *displayed_series(spec)) for spec in DEFAULT_SERIES]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
        }
    )
    fig, ax = plt.subplots(figsize=(18.5, 5.8), constrained_layout=True)

    for spec, elapsed, speed in traces:
        ax.plot(
            elapsed,
            speed,
            color=spec.color,
            linestyle="-",
            linewidth=1.4,
            alpha=0.92,
            label=spec.label,
        )

    # Intentionally no title or subtitle: the user requested a title-free plot.
    ax.set_xlabel("Time(s)")
    ax.set_ylabel("Linear velocity(m/s)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(1.0)
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#777777",
        framealpha=0.95,
        handlelength=2.6,
        borderpad=0.65,
        labelspacing=0.45,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight", facecolor="white")
    if args.pdf:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for spec, elapsed, speed in traces:
        print(
            f"{spec.label}: samples={elapsed.size}, duration={elapsed[-1]:.3f}s, "
            f"peak={np.max(speed):.3f}m/s"
        )
    print(f"PNG: {args.output}")
    if args.pdf:
        print(f"PDF: {args.pdf}")


if __name__ == "__main__":
    main()
