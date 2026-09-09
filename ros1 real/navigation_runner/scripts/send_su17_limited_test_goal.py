#!/usr/bin/env python3

"""Publish the single guarded 0.55 m goal for the first tethered Move test."""

import math
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from prometheus_msgs.msg import UAVControlState, UAVState
from std_msgs.msg import Header, String


GOAL_TOPIC = "/uav1/navrl/limited_test_goal"
GOAL_DISTANCE = 0.55
GOAL_SENT_PARAM = "/navrl_limited_move_test/goal_sent"
ACK_VALUE = "SEND_ONE_055M_GOAL"
STABILITY_WINDOW = 2.0
MIN_STABILITY_SAMPLES = 20
MIN_HEIGHT = 0.75
MAX_HEIGHT = 1.15
MAX_XY_SPEED_RMS = 0.06
MAX_XY_SPEED_PEAK = 0.12
MAX_ABS_Z_SPEED = 0.05
MAX_XY_DRIFT = 0.12
MAX_Z_RANGE = 0.08
MAX_ABS_TILT = 0.15


def fail(message):
    raise RuntimeError(message)


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def roll_pitch_from_quaternion(q):
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch


def close_to(value, expected, tolerance=1e-6):
    return abs(float(value) - float(expected)) <= tolerance


def check_system_state(state, control, bridge, navigation):
    if not state.connected or not state.armed or not state.odom_valid:
        fail("UAV must be connected, armed, and odom_valid")
    if int(state.location_source) != int(UAVState.MID360):
        fail("location_source is not MID360")
    if control.control_state != UAVControlState.COMMAND_CONTROL:
        fail("control_state is not COMMAND_CONTROL")
    if control.pos_controller != UAVControlState.PX4_ORIGIN or control.failsafe:
        fail("controller is not healthy PX4_ORIGIN")
    if bridge.data != "hold_stale_policy":
        fail("bridge must be holding with no goal; got '{}'".format(bridge.data))
    if navigation.data != "idle_no_goal":
        fail("navigation status is not idle_no_goal; got '{}'".format(navigation.data))


def collect_stability_window():
    samples = []

    def odom_callback(msg):
        samples.append(msg)

    subscriber = rospy.Subscriber(
        "/uav1/mavros/local_position/odom",
        Odometry,
        odom_callback,
        queue_size=100,
    )
    try:
        deadline = time.monotonic() + STABILITY_WINDOW
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            rospy.sleep(0.02)
    finally:
        subscriber.unregister()

    if len(samples) < MIN_STABILITY_SAMPLES:
        fail(
            "only {} odometry samples arrived during the {:.1f} s stability "
            "window (need at least {})".format(
                len(samples), STABILITY_WINDOW, MIN_STABILITY_SAMPLES
            )
        )
    return samples


def validate_stability(samples):
    positions = []
    velocities = []
    tilts = []
    for odom in samples:
        position = odom.pose.pose.position
        velocity = odom.twist.twist.linear
        q = odom.pose.pose.orientation
        values = np.array(
            [
                position.x,
                position.y,
                position.z,
                q.x,
                q.y,
                q.z,
                q.w,
                velocity.x,
                velocity.y,
                velocity.z,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            fail("odometry contains non-finite values")
        roll, pitch = roll_pitch_from_quaternion(q)
        positions.append(values[:3])
        velocities.append(values[-3:])
        tilts.append([roll, pitch])

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    tilts = np.asarray(tilts, dtype=np.float64)
    heights = positions[:, 2]
    xy_speeds = np.linalg.norm(velocities[:, :2], axis=1)
    xy_speed_rms = float(np.sqrt(np.mean(np.square(xy_speeds))))
    xy_speed_peak = float(np.max(xy_speeds))
    z_speed_peak = float(np.max(np.abs(velocities[:, 2])))
    xy_drift = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    z_range = float(np.ptp(heights))
    tilt_peak = float(np.max(np.abs(tilts)))

    if float(np.min(heights)) < MIN_HEIGHT or float(np.max(heights)) > MAX_HEIGHT:
        fail(
            "height left [{:.2f}, {:.2f}] m during the {:.1f} s stability "
            "window; observed [{:.3f}, {:.3f}] m".format(
                MIN_HEIGHT,
                MAX_HEIGHT,
                STABILITY_WINDOW,
                float(np.min(heights)),
                float(np.max(heights)),
            )
        )
    if xy_speed_rms > MAX_XY_SPEED_RMS:
        fail(
            "horizontal RMS speed {:.3f} m/s exceeds {:.3f} m/s over {:.1f} s"
            .format(xy_speed_rms, MAX_XY_SPEED_RMS, STABILITY_WINDOW)
        )
    if xy_speed_peak > MAX_XY_SPEED_PEAK:
        fail(
            "horizontal peak speed {:.3f} m/s exceeds {:.3f} m/s"
            .format(xy_speed_peak, MAX_XY_SPEED_PEAK)
        )
    if z_speed_peak > MAX_ABS_Z_SPEED:
        fail(
            "vertical peak speed {:.3f} m/s exceeds {:.3f} m/s"
            .format(z_speed_peak, MAX_ABS_Z_SPEED)
        )
    if xy_drift > MAX_XY_DRIFT:
        fail(
            "horizontal drift {:.3f} m exceeds {:.3f} m over {:.1f} s"
            .format(xy_drift, MAX_XY_DRIFT, STABILITY_WINDOW)
        )
    if z_range > MAX_Z_RANGE:
        fail(
            "height range {:.3f} m exceeds {:.3f} m over {:.1f} s"
            .format(z_range, MAX_Z_RANGE, STABILITY_WINDOW)
        )
    if tilt_peak > MAX_ABS_TILT:
        fail(
            "roll/pitch peak {:.3f} rad exceeds {:.3f} rad"
            .format(tilt_peak, MAX_ABS_TILT)
        )

    return {
        "xy_speed_rms": xy_speed_rms,
        "xy_speed_peak": xy_speed_peak,
        "z_speed_peak": z_speed_peak,
        "xy_drift": xy_drift,
        "z_range": z_range,
        "tilt_peak": tilt_peak,
    }


def main():
    rospy.init_node("send_su17_limited_test_goal", anonymous=False)
    if os.environ.get("NAVRL_LIMITED_GOAL_ACK") != ACK_VALUE:
        fail(
            "goal interlock not acknowledged; export "
            "NAVRL_LIMITED_GOAL_ACK={}".format(ACK_VALUE)
        )
    if rospy.get_param(GOAL_SENT_PARAM, False):
        fail("the one allowed limited-test goal was already sent; restart the launcher")

    expected_params = {
        "/navigation_su17_phase1/max_xy_speed": 0.10,
        "/navrl_su17_bridge/max_xy_speed": 0.10,
        "/navrl_su17_bridge/max_xy_from_home": 0.75,
        "/navrl_su17_bridge/max_move_duration": 3.0,
    }
    for name, expected in expected_params.items():
        if not rospy.has_param(name) or not close_to(rospy.get_param(name), expected):
            fail("unsafe or missing parameter {} (expected {})".format(name, expected))
    if rospy.get_param("/navigation_su17_phase1/goal_topic", "") != GOAL_TOPIC:
        fail("navigation node is not using the private limited-test goal topic")
    if not rospy.get_param("/navrl_su17_bridge/output_enabled", False):
        fail("bridge output_enabled is not true")

    state = rospy.wait_for_message(
        "/uav1/prometheus/state", UAVState, timeout=2.0
    )
    control = rospy.wait_for_message(
        "/uav1/prometheus/control_state", UAVControlState, timeout=2.0
    )
    bridge = rospy.wait_for_message(
        "/uav1/navrl/bridge_status", String, timeout=2.0
    )
    navigation = rospy.wait_for_message(
        "/uav1/navrl/navigation_status", String, timeout=2.0
    )
    rospy.wait_for_message("/occupancy_map/update", Header, timeout=2.0)

    check_system_state(state, control, bridge, navigation)
    rospy.loginfo(
        "[limited-goal] checking a continuous %.1f s hover-stability window",
        STABILITY_WINDOW,
    )
    odom_samples = collect_stability_window()
    stability = validate_stability(odom_samples)

    # The RC mode, flight-controller health, bridge hold state, navigation idle
    # state and map heartbeat must still be valid after the stability window.
    state = rospy.wait_for_message(
        "/uav1/prometheus/state", UAVState, timeout=2.0
    )
    control = rospy.wait_for_message(
        "/uav1/prometheus/control_state", UAVControlState, timeout=2.0
    )
    bridge = rospy.wait_for_message(
        "/uav1/navrl/bridge_status", String, timeout=2.0
    )
    navigation = rospy.wait_for_message(
        "/uav1/navrl/navigation_status", String, timeout=2.0
    )
    rospy.wait_for_message("/occupancy_map/update", Header, timeout=2.0)
    check_system_state(state, control, bridge, navigation)

    odom = odom_samples[-1]
    position = odom.pose.pose.position
    q = odom.pose.pose.orientation
    velocity = odom.twist.twist.linear
    speed = float(
        np.linalg.norm([velocity.x, velocity.y, velocity.z])
    )

    yaw = yaw_from_quaternion(q)
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "world"
    goal.pose.position.x = float(position.x + GOAL_DISTANCE * math.cos(yaw))
    goal.pose.position.y = float(position.y + GOAL_DISTANCE * math.sin(yaw))
    goal.pose.position.z = float(position.z)
    goal.pose.orientation = q

    publisher = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1)
    master_code, master_message, system_state = rospy.get_master().getSystemState()
    if master_code != 1:
        fail("cannot query ROS master subscribers: {}".format(master_message))
    goal_subscribers = dict(system_state[1]).get(GOAL_TOPIC, [])
    if "/navigation_su17_phase1" not in goal_subscribers:
        fail("navigation_su17_phase1 is not subscribed to the private goal topic")
    unexpected_subscribers = [
        name
        for name in goal_subscribers
        if name != "/navigation_su17_phase1" and not name.startswith("/record_")
    ]
    if unexpected_subscribers:
        fail(
            "unexpected private-goal subscribers: {}".format(
                unexpected_subscribers
            )
        )
    deadline = time.monotonic() + 2.0
    while publisher.get_num_connections() < 1 and time.monotonic() < deadline:
        rospy.sleep(0.02)
    if publisher.get_num_connections() < 1:
        fail("private goal topic has no connected subscriber")

    # Latch the one-shot interlock before publishing.  If delivery fails, the
    # operator must restart the whole limited-test launch rather than retry.
    rospy.set_param(GOAL_SENT_PARAM, True)
    publisher.publish(goal)
    rospy.logwarn(
        "[limited-goal] SENT ONCE current=[%.3f %.3f %.3f] "
        "goal=[%.3f %.3f %.3f] yaw=%.3f speed=%.3f "
        "stability={xy_rms=%.3f xy_peak=%.3f z_peak=%.3f "
        "xy_drift=%.3f z_range=%.3f tilt_peak=%.3f}",
        position.x,
        position.y,
        position.z,
        goal.pose.position.x,
        goal.pose.position.y,
        goal.pose.position.z,
        yaw,
        speed,
        stability["xy_speed_rms"],
        stability["xy_speed_peak"],
        stability["z_speed_peak"],
        stability["xy_drift"],
        stability["z_range"],
        stability["tilt_peak"],
    )
    rospy.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logfatal("[limited-goal] REFUSED: %s", exc)
        raise
