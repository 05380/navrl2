#!/usr/bin/env python3
"""Plot evaluation success/deadlock rates from NavRL console/W&B text logs.

The script parses lines like:

    [NavRL]: start evaluating policy at training step:  34000
    [NavRL]: random_crossing_eval metrics | eval/success_rate=...

It also recognizes exact W&B summary values. If the log only contains a
W&B sparkline for a metric, the script does not reconstruct it because the
sparkline is normalized and is not a precise data source.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"training step:\s*([0-9]+)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
SUMMARY_RE = re.compile(
    r"wandb:\s+([A-Za-z0-9_./-]+)\s+(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


SUCCESS_KEY = "eval/success_rate"
DEADLOCK_KEY = "eval/deadlock_rate"
ESCAPE_KEY = "eval/conditioned/success_given_deadlock"


def parse_log(path: Path) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    summary: dict[str, float] = {}
    current_step: int | None = None

    for line in path.read_text(errors="ignore").splitlines():
        step_match = STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group(1))

        if "[NavRL]:" in line and " metrics | " in line:
            values = {key: float(value) for key, value in KV_RE.findall(line)}
            if SUCCESS_KEY not in values and DEADLOCK_KEY not in values and ESCAPE_KEY not in values:
                continue
            if current_step is None:
                current_step = len(rows)
            row: dict[str, float] = {"step": float(current_step)}
            for key in (SUCCESS_KEY, DEADLOCK_KEY, ESCAPE_KEY):
                if key in values:
                    row[key] = values[key]
            rows.append(row)

        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            key, value = summary_match.groups()
            summary[key] = float(value)

    if ESCAPE_KEY in summary and rows and ESCAPE_KEY not in rows[-1]:
        # The exact final summary value belongs to the last logged step, but it
        # is only one point, not a full curve.
        rows[-1][ESCAPE_KEY] = summary[ESCAPE_KEY]
        rows[-1]["_escape_summary_only"] = 1.0

    return rows, summary


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or len(values) < 2:
        return values, np.zeros_like(values)
    window = max(1, int(window))
    half = window // 2
    means = np.empty_like(values, dtype=float)
    stds = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        chunk = values[start:end]
        means[index] = float(np.nanmean(chunk))
        stds[index] = float(np.nanstd(chunk))
    return means, stds


def finite_series(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(row["step"], row[key]) for row in rows if key in row and math.isfinite(row[key])]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def write_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "success_rate",
                "deadlock_rate",
                "escape_after_deadlock_rate",
                "escape_after_deadlock_source",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": int(row["step"]),
                    "success_rate": row.get(SUCCESS_KEY, ""),
                    "deadlock_rate": row.get(DEADLOCK_KEY, ""),
                    "escape_after_deadlock_rate": row.get(ESCAPE_KEY, ""),
                    "escape_after_deadlock_source": (
                        "summary_final" if row.get("_escape_summary_only") else "per_eval"
                    )
                    if ESCAPE_KEY in row
                    else "",
                }
            )


def plot_rates(
    rows: list[dict[str, float]],
    output_png: Path,
    band_window: int,
    annotate_missing: bool,
    legend_loc: str,
    x_tick_step: int,
    y_tick_step: float,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    series = [
        (SUCCESS_KEY, "Success rate", "#1f77b4", "^", "-"),
        (DEADLOCK_KEY, "Deadlock rate", "#d62728", "o", ":"),
        (ESCAPE_KEY, "Escape after deadlock", "#2ca02c", "s", "--"),
    ]

    exact_escape_points = len([row for row in rows if ESCAPE_KEY in row])

    for key, label, color, marker, linestyle in series:
        x, y = finite_series(rows, key)
        if len(x) == 0:
            continue
        if key == ESCAPE_KEY and exact_escape_points == 1:
            ax.scatter(x, y, s=72, marker=marker, color=color, edgecolor="white", zorder=4, label=label)
            if annotate_missing:
                ax.annotate(
                    "final summary only",
                    xy=(x[-1], y[-1]),
                    xytext=(-92, 18),
                    textcoords="offset points",
                    fontsize=10,
                    color=color,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
                )
            continue

        mean, std = rolling_mean_std(y, band_window)
        if band_window > 1 and len(y) > 2:
            lower = np.clip(mean - std, 0.0, 1.0)
            upper = np.clip(mean + std, 0.0, 1.0)
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x,
            mean,
            linestyle=linestyle,
            marker=marker,
            markersize=5.2,
            linewidth=1.8,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=label,
        )

    ax.set_xlabel("Iterations", fontsize=18)
    ax.set_ylabel("Rate", fontsize=18)
    ax.set_ylim(-0.02, 1.02)
    if y_tick_step > 0:
        ax.set_yticks(np.arange(0.0, 1.0 + y_tick_step, y_tick_step))
    max_step = max(row["step"] for row in rows)
    if x_tick_step > 0:
        last_tick = int(math.ceil(max_step / x_tick_step) * x_tick_step)
        ax.set_xticks(np.arange(0, last_tick + x_tick_step, x_tick_step))
    ax.grid(True, color="#b0b0b0", alpha=0.35, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=14, direction="in", top=True, right=True)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.94, edgecolor="#444", fontsize=12)

    if annotate_missing and exact_escape_points <= 1:
        ax.text(
            0.02,
            0.96,
            "No exact per-iteration escape-after-deadlock values in this text.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#444",
        )

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Path to pasted training/evaluation text log.")
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("plots/eval_rates_from_log.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("plots/eval_rates_from_log.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--band-window",
        type=int,
        default=5,
        help="Centered rolling window for the visual shaded band. Use 1 to disable.",
    )
    parser.add_argument(
        "--annotate-missing",
        action="store_true",
        help="Annotate missing per-iteration escape-after-deadlock values on the figure.",
    )
    parser.add_argument(
        "--legend-loc",
        default="upper right",
        help="Matplotlib legend location. Default avoids covering the deadlock-rate curve.",
    )
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=5000,
        help="X-axis major tick interval in iterations.",
    )
    parser.add_argument(
        "--y-tick-step",
        type=float,
        default=0.1,
        help="Y-axis major tick interval.",
    )
    args = parser.parse_args()

    rows, _summary = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No evaluation metric rows found in {args.log}")

    write_csv(rows, args.output_csv)
    plot_rates(
        rows,
        args.output_png,
        args.band_window,
        args.annotate_missing,
        args.legend_loc,
        args.x_tick_step,
        args.y_tick_step,
    )

    escape_rows = sum(1 for row in rows if ESCAPE_KEY in row)
    print(f"parsed_eval_points={len(rows)}")
    print(f"escape_after_deadlock_points={escape_rows}")
    print(f"saved_png={args.output_png}")
    print(f"saved_csv={args.output_csv}")
    if escape_rows == 0:
        print(
            "warning=The text does not contain exact "
            "eval/conditioned/success_given_deadlock values; "
            "escape-after-deadlock is not plotted."
        )
    elif escape_rows == 1:
        print(
            "warning=The text does not contain exact per-iteration "
            "eval/conditioned/success_given_deadlock values; only the final "
            "summary point can be plotted."
        )


if __name__ == "__main__":
    main()
