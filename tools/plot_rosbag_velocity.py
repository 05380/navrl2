#!/usr/bin/env python3
"""Extract and plot UAV velocity from a ROS1 bag, including unindexed bags.

The preferred source is ``/uav1/mavros/local_position/odom``.  The script
scans complete bag chunks directly, so an interrupted/unindexed recording can
still be analysed without modifying the original file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


def _dependency_error(exc: Exception) -> None:
    raise SystemExit(
        "Missing dependency 'rosbags'. Install it with:\n"
        "  python3 -m pip install rosbags\n"
        f"Original error: {exc}"
    )


try:
    from lz4.frame import decompress as lz4_decompress
    from rosbags.rosbag1.reader import Header, ReaderError, RecordType, read_bytes, read_uint32
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    from rosbags.typesys.msg import normalize_msgtype
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    _dependency_error(exc)


@dataclass
class ConnectionInfo:
    topic: str
    msgtype: str
    msgdef: str


@dataclass
class OdomSample:
    bag_ns: int
    header_ns: int
    vx: float
    vy: float
    vz: float
    px: float
    py: float
    pz: float


@dataclass
class PrometheusSample:
    bag_ns: int
    header_ns: int
    vx: float
    vy: float
    vz: float


def _header_from_blob(blob: bytes) -> Header:
    """Parse the data-header part of a ROS1 CONNECTION record."""
    return Header.read(BytesIO(struct.pack("<L", len(blob)) + blob))


def _stamp_ns(message: object, fallback_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns
    sec = int(getattr(stamp, "sec", 0))
    nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    value = sec * 1_000_000_000 + nsec
    return value if value > 0 else fallback_ns


class BagScan:
    def __init__(self) -> None:
        self.connections: dict[int, ConnectionInfo] = {}
        self.topic_counts: Counter[str] = Counter()
        self.odom: list[OdomSample] = []
        self.prometheus: list[PrometheusSample] = []
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.chunk_count = 0
        self.message_count = 0
        self.first_message_ns: int | None = None
        self.last_message_ns: int | None = None
        self.index_pos = 0
        self.declared_chunk_count = 0
        self.declared_connection_count = 0
        self.complete = True
        self.trailing_error = ""
        self.deserialize_errors: Counter[str] = Counter()

    def _remember_connection(self, header: Header, blob: bytes) -> None:
        details = _header_from_blob(blob)
        conn_id = header.get_uint32("conn")
        topic = header.get_string("topic")
        msgtype = normalize_msgtype(details.get_string("type"))
        msgdef = details.get_string("message_definition")
        self.connections[conn_id] = ConnectionInfo(topic, msgtype, msgdef)
        try:
            types = get_types_from_msg(msgdef, msgtype)
            new_types = {name: value for name, value in types.items() if name not in self.typestore.types}
            if new_types:
                self.typestore.register(new_types)
        except Exception as exc:  # Topic inventory can continue even for unknown custom messages.
            self.deserialize_errors[f"register:{topic}:{type(exc).__name__}"] += 1

    def _remember_message(self, header: Header, blob: bytes) -> None:
        conn_id = header.get_uint32("conn")
        bag_ns = header.get_time("time")
        info = self.connections.get(conn_id)
        topic = info.topic if info else f"<connection:{conn_id}>"
        self.topic_counts[topic] += 1
        self.message_count += 1
        self.first_message_ns = bag_ns if self.first_message_ns is None else min(self.first_message_ns, bag_ns)
        self.last_message_ns = bag_ns if self.last_message_ns is None else max(self.last_message_ns, bag_ns)

        if info is None or topic not in {
            "/uav1/mavros/local_position/odom",
            "/uav1/prometheus/state",
        }:
            return

        try:
            message = self.typestore.deserialize_ros1(blob, info.msgtype)
            header_ns = _stamp_ns(message, bag_ns)
            if topic == "/uav1/mavros/local_position/odom":
                linear = message.twist.twist.linear
                position = message.pose.pose.position
                self.odom.append(
                    OdomSample(
                        bag_ns,
                        header_ns,
                        float(linear.x),
                        float(linear.y),
                        float(linear.z),
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    )
                )
            else:
                velocity = np.asarray(message.velocity, dtype=float).reshape(-1)
                if velocity.size >= 3:
                    self.prometheus.append(
                        PrometheusSample(
                            bag_ns,
                            header_ns,
                            float(velocity[0]),
                            float(velocity[1]),
                            float(velocity[2]),
                        )
                    )
        except Exception as exc:
            self.deserialize_errors[f"message:{topic}:{type(exc).__name__}"] += 1

    def _scan_chunk(self, raw: bytes) -> None:
        stream = BytesIO(raw)
        while stream.tell() < len(raw):
            header = Header.read(stream)
            blob = read_bytes(stream, read_uint32(stream))
            op = header.get_uint8("op")
            if op == RecordType.CONNECTION:
                self._remember_connection(header, blob)
            elif op == RecordType.MSGDATA:
                self._remember_message(header, blob)

    def _skip_or_read_blob(self, stream: BinaryIO, size: int) -> bytes:
        return read_bytes(stream, size)

    def scan(self, path: Path) -> None:
        with path.open("rb") as stream:
            magic = stream.readline()
            if magic != b"#ROSBAG V2.0\n":
                raise SystemExit(f"Not a ROS1 bag v2.0 file: {path}")

            bag_header = Header.read(stream, RecordType.BAGHEADER)
            self.index_pos = bag_header.get_uint64("index_pos")
            self.declared_connection_count = bag_header.get_uint32("conn_count")
            self.declared_chunk_count = bag_header.get_uint32("chunk_count")
            stream.seek(read_uint32(stream), os.SEEK_CUR)

            while True:
                record_pos = stream.tell()
                if not stream.read(1):
                    break
                stream.seek(record_pos)
                try:
                    header = Header.read(stream)
                    size = read_uint32(stream)
                    blob = self._skip_or_read_blob(stream, size)
                    op = header.get_uint8("op")
                    if op == RecordType.CHUNK:
                        compression = header.get_string("compression")
                        if compression == "none":
                            raw = blob
                        elif compression == "lz4":
                            raw = lz4_decompress(blob)
                        else:
                            raise ReaderError(f"Unsupported compression {compression!r}")
                        self._scan_chunk(raw)
                        self.chunk_count += 1
                    elif op == RecordType.CONNECTION:
                        self._remember_connection(header, blob)
                except Exception as exc:
                    # An interrupted recorder can leave one incomplete trailing record.
                    self.complete = False
                    self.trailing_error = f"offset {record_pos}: {type(exc).__name__}: {exc}"
                    break


def _as_arrays(samples: Iterable[OdomSample]) -> dict[str, np.ndarray]:
    rows = list(samples)
    return {
        "bag_ns": np.asarray([x.bag_ns for x in rows], dtype=np.int64),
        "header_ns": np.asarray([x.header_ns for x in rows], dtype=np.int64),
        "vx": np.asarray([x.vx for x in rows], dtype=float),
        "vy": np.asarray([x.vy for x in rows], dtype=float),
        "vz": np.asarray([x.vz for x in rows], dtype=float),
        "px": np.asarray([x.px for x in rows], dtype=float),
        "py": np.asarray([x.py for x in rows], dtype=float),
        "pz": np.asarray([x.pz for x in rows], dtype=float),
    }


def _series_metrics(data: dict[str, np.ndarray], scan: BagScan) -> tuple[dict[str, object], np.ndarray]:
    header_ns = data["header_ns"]
    bag_ns = data["bag_ns"]
    elapsed_s = (header_ns - header_ns[0]) / 1e9
    dt = np.diff(header_ns) / 1e9
    positive_dt = dt[dt > 0]
    finite = np.isfinite(np.column_stack((data["vx"], data["vy"], data["vz"]))).all(axis=1)
    horizontal = np.hypot(data["vx"], data["vy"])
    speed_3d = np.sqrt(horizontal**2 + data["vz"] ** 2)
    delay_ms = (bag_ns - header_ns) / 1e6
    duration_s = float(elapsed_s[-1]) if len(elapsed_s) > 1 else 0.0

    metrics: dict[str, object] = {
        "bag_indexed": bool(scan.index_pos),
        "scan_complete_to_eof": scan.complete,
        "trailing_error": scan.trailing_error or None,
        "file_chunks_scanned": scan.chunk_count,
        "declared_chunks": scan.declared_chunk_count,
        "messages_scanned": scan.message_count,
        "bag_duration_s": (
            (scan.last_message_ns - scan.first_message_ns) / 1e9
            if scan.first_message_ns is not None and scan.last_message_ns is not None
            else 0.0
        ),
        "velocity_topic": "/uav1/mavros/local_position/odom",
        "velocity_messages": len(header_ns),
        "velocity_duration_s": duration_s,
        "mean_rate_hz": ((len(header_ns) - 1) / duration_s if duration_s > 0 else None),
        "median_period_ms": (float(np.median(positive_dt) * 1000) if positive_dt.size else None),
        "p95_period_ms": (float(np.percentile(positive_dt, 95) * 1000) if positive_dt.size else None),
        "max_period_ms": (float(np.max(positive_dt) * 1000) if positive_dt.size else None),
        "non_increasing_header_timestamps": int(np.sum(dt <= 0)),
        "finite_velocity_samples": int(np.sum(finite)),
        "nonfinite_velocity_samples": int(np.sum(~finite)),
        "header_to_bag_delay_ms": {
            "median": float(np.median(delay_ms)),
            "p95": float(np.percentile(delay_ms, 95)),
            "max": float(np.max(delay_ms)),
        },
        "axis_range_mps": {
            "vx": [float(np.min(data["vx"])), float(np.max(data["vx"]))],
            "vy": [float(np.min(data["vy"])), float(np.max(data["vy"]))],
            "vz": [float(np.min(data["vz"])), float(np.max(data["vz"]))],
        },
        "horizontal_speed_mps": {
            "mean": float(np.mean(horizontal)),
            "p95": float(np.percentile(horizontal, 95)),
            "max": float(np.max(horizontal)),
        },
        "speed_3d_mps": {
            "mean": float(np.mean(speed_3d)),
            "p95": float(np.percentile(speed_3d, 95)),
            "max": float(np.max(speed_3d)),
        },
        "samples_above_0_1_mps": int(np.sum(speed_3d > 0.1)),
        "prometheus_velocity_messages": len(scan.prometheus),
        "topic_counts": dict(sorted(scan.topic_counts.items())),
        "deserialize_errors": dict(scan.deserialize_errors),
    }
    return metrics, elapsed_s


def _write_csv(path: Path, data: dict[str, np.ndarray], elapsed_s: np.ndarray) -> None:
    horizontal = np.hypot(data["vx"], data["vy"])
    speed_3d = np.sqrt(horizontal**2 + data["vz"] ** 2)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "elapsed_s",
                "bag_time_s",
                "header_time_s",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "horizontal_speed_mps",
                "speed_3d_mps",
                "position_x_m",
                "position_y_m",
                "position_z_m",
            ]
        )
        for values in zip(
            elapsed_s,
            data["bag_ns"] / 1e9,
            data["header_ns"] / 1e9,
            data["vx"],
            data["vy"],
            data["vz"],
            horizontal,
            speed_3d,
            data["px"],
            data["py"],
            data["pz"],
        ):
            writer.writerow((f"{value:.9f}" for value in values))


def _plot(
    path: Path,
    source_name: str,
    data: dict[str, np.ndarray],
    elapsed_s: np.ndarray,
    plot_start_s: float,
    plot_end_s: float | None,
    series_label: str,
    line_color: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/navrl-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed_3d = np.sqrt(data["vx"] ** 2 + data["vy"] ** 2 + data["vz"] ** 2)
    if plot_end_s is not None and plot_end_s <= plot_start_s:
        raise SystemExit("--plot-end-s must be greater than --plot-start-s")
    visible = elapsed_s >= plot_start_s
    if plot_end_s is not None:
        visible &= elapsed_s <= plot_end_s
    if np.count_nonzero(visible) < 2:
        raise SystemExit(
            f"Plot start {plot_start_s:.3f} s leaves fewer than two samples; "
            f"available range is {elapsed_s[0]:.3f}--{elapsed_s[-1]:.3f} s."
        )
    retained_time = elapsed_s[visible]
    plot_time = retained_time - retained_time[0]
    plot_speed = speed_3d[visible]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axis = plt.subplots(figsize=(12, 5.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.79)
    axis.plot(
        plot_time,
        plot_speed,
        color=line_color,
        linewidth=1.5,
        linestyle="-",
        label=series_label,
    )
    axis.set_xlim(float(plot_time[0]), float(plot_time[-1]))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("3D speed (m/s)")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.legend(loc="upper right", frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.suptitle("SU17 3D speed over time", y=0.965, fontsize=15, fontweight="semibold")
    if plot_end_s is not None:
        time_note = (
            f"bag segment {plot_start_s:.2f}--{plot_end_s:.2f} s, "
            "displayed time rebased to 0"
        )
    elif plot_start_s > 0:
        time_note = f"first {plot_start_s:g} s omitted, displayed time rebased to 0"
    else:
        time_note = "full recording, time starts at 0"
    fig.text(
        0.5,
        0.89,
        (
            f"MAVROS local odometry | {source_name} | "
            f"{time_note} | raw samples, no smoothing"
        ),
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS1 .bag file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG, CSV, and JSON")
    parser.add_argument(
        "--plot-start-s",
        type=float,
        default=0.0,
        help="Hide samples before this elapsed bag time in the PNG (default: 0)",
    )
    parser.add_argument(
        "--plot-end-s",
        type=float,
        default=None,
        help="Hide samples after this elapsed bag time in the PNG (default: recording end)",
    )
    parser.add_argument("--series-label", default="3D speed", help="Legend label")
    parser.add_argument("--line-color", default="#1769AA", help="Matplotlib line color")
    args = parser.parse_args()

    if not args.bag.is_file():
        raise SystemExit(f"Bag does not exist: {args.bag}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scan = BagScan()
    scan.scan(args.bag)
    if not scan.odom:
        inventory = "\n".join(f"  {topic}: {count}" for topic, count in sorted(scan.topic_counts.items()))
        raise SystemExit(
            "No decodable /uav1/mavros/local_position/odom samples were found.\n"
            f"Scanned topics:\n{inventory or '  (none)'}"
        )

    data = _as_arrays(scan.odom)
    metrics, elapsed_s = _series_metrics(data, scan)
    stem = args.bag.stem
    csv_path = args.output_dir / f"{stem}_velocity.csv"
    start_label = f"{args.plot_start_s:g}".replace(".", "p")
    end_suffix = (
        f"_to_{f'{args.plot_end_s:g}'.replace('.', 'p')}s"
        if args.plot_end_s is not None
        else ""
    )
    png_path = args.output_dir / f"{stem}_3d_speed_from_{start_label}s{end_suffix}.png"
    json_path = args.output_dir / f"{stem}_summary.json"
    _write_csv(csv_path, data, elapsed_s)
    _plot(
        png_path,
        args.bag.name,
        data,
        elapsed_s,
        args.plot_start_s,
        args.plot_end_s,
        args.series_label,
        args.line_color,
    )
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"CSV={csv_path}")
    print(f"PNG={png_path}")
    print(f"SUMMARY={json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
