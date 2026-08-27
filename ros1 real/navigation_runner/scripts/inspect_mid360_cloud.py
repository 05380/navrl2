#!/usr/bin/env python3

"""Print near-range statistics for one MID-360 body-cloud frame."""

import argparse
import math
import sys

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return float("nan")
    index = int(round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/uav1/cloud_mid360_body")
    parser.add_argument("--timeout", type=float, default=5.0)
    args, _unknown = parser.parse_known_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("inspect_mid360_cloud", anonymous=True, disable_signals=True)
    try:
        message = rospy.wait_for_message(
            args.topic, PointCloud2, timeout=args.timeout
        )
    except rospy.ROSException as exc:
        print("cloud wait failed: {}".format(exc))
        return 1

    samples = []
    for x, y, z in point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    ):
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        distance = math.sqrt(x * x + y * y + z * z)
        samples.append((distance, x, y, z))

    samples.sort(key=lambda item: item[0])
    ranges = [item[0] for item in samples]
    print("topic={} frame={} valid_points={}".format(
        args.topic, message.header.frame_id, len(samples)
    ))
    if not samples:
        return 1

    print(
        "range_m min={:.3f} p01={:.3f} p05={:.3f} p50={:.3f} max={:.3f}".format(
            ranges[0],
            _percentile(ranges, 0.01),
            _percentile(ranges, 0.05),
            _percentile(ranges, 0.50),
            ranges[-1],
        )
    )
    for threshold in (0.30, 0.35, 0.45, 0.55, 1.00, 4.00, 5.00):
        count = sum(distance < threshold for distance in ranges)
        print("range<{:.2f}m: {} ({:.2f}%)".format(
            threshold, count, 100.0 * count / len(ranges)
        ))
    print("closest_points: distance x y z")
    for distance, x, y, z in samples[:10]:
        print("  {:.3f} {:+.3f} {:+.3f} {:+.3f}".format(distance, x, y, z))
    return 0


if __name__ == "__main__":
    sys.exit(main())
