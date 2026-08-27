#!/usr/bin/env python3

"""Watchdog bridge from NavRL's private target to Prometheus UAVCommand.

The bridge is disabled by default.  When enabled it only forwards finite,
fresh, bounded world-ENU velocity targets while the supplied SU17 controller
is in COMMAND_CONTROL.  Every unhealthy condition requests the native
Current_Pos_Hover behavior instead of continuing the previous velocity.
"""

import math
import threading

import numpy as np
import rospy
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import Odometry
from prometheus_msgs.msg import UAVCommand, UAVControlState, UAVState
from std_msgs.msg import String


def _bool_param(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    raise ValueError("{} must be a boolean, got {!r}".format(name, value))


class NavRLSU17Bridge:
    def __init__(self):
        self.lock = threading.RLock()
        self.uav_id = int(rospy.get_param("~uav_id", 1))
        prefix = "/uav{}".format(self.uav_id)

        self.output_enabled = _bool_param("~output_enabled", False)
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.command_timeout = float(rospy.get_param("~command_timeout", 0.25))
        self.state_timeout = float(rospy.get_param("~state_timeout", 0.20))
        self.control_state_timeout = float(
            rospy.get_param("~control_state_timeout", 0.20)
        )
        self.max_xy_speed = float(rospy.get_param("~max_xy_speed", 0.50))
        self.max_z_speed = float(rospy.get_param("~max_z_speed", 0.30))
        self.min_z = float(rospy.get_param("~min_z", 0.30))
        self.max_z = float(rospy.get_param("~max_z", 1.50))
        self.max_xy_from_home = float(
            rospy.get_param("~max_xy_from_home", 3.0)
        )
        self.fence_prediction_horizon = float(
            rospy.get_param("~fence_prediction_horizon", 1.0)
        )
        self.enable_odom_consistency_check = _bool_param(
            "~enable_odom_consistency_check", False
        )
        self.odom_consistency_timeout = float(
            rospy.get_param("~odom_consistency_timeout", 0.30)
        )
        self.max_odom_delta_change = float(
            rospy.get_param("~max_odom_delta_change", 0.75)
        )
        self.max_odom_yaw_delta_change = float(
            rospy.get_param("~max_odom_yaw_delta_change", 0.20)
        )

        numeric_params = (
            self.rate_hz,
            self.command_timeout,
            self.state_timeout,
            self.control_state_timeout,
            self.max_xy_speed,
            self.max_z_speed,
            self.min_z,
            self.max_z,
            self.max_xy_from_home,
            self.fence_prediction_horizon,
            self.odom_consistency_timeout,
            self.max_odom_delta_change,
            self.max_odom_yaw_delta_change,
        )
        if not all(math.isfinite(value) for value in numeric_params):
            raise ValueError("all watchdog and safety parameters must be finite")
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if min(
            self.command_timeout,
            self.state_timeout,
            self.control_state_timeout,
            self.odom_consistency_timeout,
        ) <= 0.0:
            raise ValueError("watchdog timeouts must be positive")
        if self.max_xy_speed <= 0.0 or self.max_z_speed <= 0.0:
            raise ValueError("velocity limits must be positive")
        if self.min_z >= self.max_z:
            raise ValueError("min_z must be smaller than max_z")
        if min(
            self.max_xy_from_home,
            self.fence_prediction_horizon,
            self.max_odom_delta_change,
            self.max_odom_yaw_delta_change,
        ) <= 0.0:
            raise ValueError("fence and odometry thresholds must be positive")

        # A self-filter makes points inside its box unobservable.  Active
        # control is only safe when the collision-inflation body encloses that
        # whole box; otherwise a real obstacle could be filtered before the
        # mapper accounts for the vehicle footprint.  Shadow mode keeps
        # running so the initial box can be measured and refined.
        self_filter_enabled = _bool_param(
            "/occupancy_map/self_filter_enabled", False
        )
        if self_filter_enabled:
            robot_size = np.asarray(
                rospy.get_param("/occupancy_map/robot_size", []),
                dtype=np.float64,
            )
            self_filter_min = np.asarray(
                rospy.get_param("/occupancy_map/self_filter_min", []),
                dtype=np.float64,
            )
            self_filter_max = np.asarray(
                rospy.get_param("/occupancy_map/self_filter_max", []),
                dtype=np.float64,
            )
            vectors = (robot_size, self_filter_min, self_filter_max)
            if any(value.shape != (3,) for value in vectors) or not all(
                self._finite(value) for value in vectors
            ):
                raise ValueError(
                    "robot_size and self_filter_min/max must be finite 3-vectors"
                )
            required_half_size = np.maximum(
                np.abs(self_filter_min), np.abs(self_filter_max)
            )
            available_half_size = 0.5 * robot_size
            if np.any(robot_size <= 0.0) or np.any(
                available_half_size + 1e-9 < required_half_size
            ):
                message = (
                    "collision robot_size={} does not enclose self-filter "
                    "half extents={}; measure the airframe and make these "
                    "settings consistent"
                ).format(robot_size.tolist(), required_half_size.tolist())
                if self.output_enabled:
                    raise ValueError(message)
                rospy.logwarn("[navrl-bridge] SHADOW ONLY: %s", message)

        self.desired = None
        self.desired_receipt = None
        self.control_state = None
        self.control_state_receipt = None
        self.uav_state = None
        self.uav_state_receipt = None
        self.fastlio_odom = None
        self.fastlio_receipt = None
        self.mavros_odom = None
        self.mavros_receipt = None
        self.odom_delta_baseline = None
        self.odom_yaw_delta_baseline = None
        self.home_xy = None
        self.last_control_state_value = UAVControlState.INIT
        self.last_status = None
        self.command_id = 0

        self.prometheus_command_topic = str(
            rospy.get_param("~prometheus_command_topic", prefix + "/prometheus/command")
        )
        self.desired_topic = str(
            rospy.get_param("~desired_topic", prefix + "/navrl/desired_setpoint")
        )
        # In shadow mode do not even register as a command-topic publisher.
        # Enabling output requires a node restart with an explicit true value.
        self.command_pub = None
        if self.output_enabled:
            self.command_pub = rospy.Publisher(
                self.prometheus_command_topic, UAVCommand, queue_size=2
            )
        self.safe_setpoint_pub = rospy.Publisher(
            prefix + "/navrl/safe_setpoint", PositionTarget, queue_size=2
        )
        self.status_pub = rospy.Publisher(
            prefix + "/navrl/bridge_status", String, queue_size=5, latch=True
        )

        rospy.Subscriber(
            self.desired_topic, PositionTarget, self._desired_cb, queue_size=2
        )
        rospy.Subscriber(
            prefix + "/prometheus/control_state",
            UAVControlState,
            self._control_state_cb,
            queue_size=5,
        )
        rospy.Subscriber(
            prefix + "/prometheus/state", UAVState, self._uav_state_cb, queue_size=5
        )
        rospy.Subscriber(
            prefix + "/Odometry", Odometry, self._fastlio_cb, queue_size=5
        )
        rospy.Subscriber(
            prefix + "/mavros/local_position/odom",
            Odometry,
            self._mavros_cb,
            queue_size=5,
        )

        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.rate_hz), self._timer_cb
        )
        rospy.on_shutdown(self._on_shutdown)
        mode = "ACTIVE" if self.output_enabled else "SHADOW"
        self._set_status("{}_waiting_for_inputs".format(mode.lower()))
        rospy.logwarn(
            "[navrl-bridge] mode=%s output=%s desired=%s; active flight requires explicit output_enabled:=true",
            mode,
            self.prometheus_command_topic,
            self.desired_topic,
        )

    @staticmethod
    def _finite(values):
        return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))

    @staticmethod
    def _position_from_odom(msg):
        return np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _set_status(self, status):
        if status != self.last_status:
            self.last_status = status
            self.status_pub.publish(String(data=status))
            rospy.loginfo("[navrl-bridge] status=%s", status)

    def _desired_cb(self, msg):
        with self.lock:
            self.desired = msg
            self.desired_receipt = rospy.Time.now()

    def _control_state_cb(self, msg):
        with self.lock:
            self.control_state = msg
            self.control_state_receipt = rospy.Time.now()
            self.last_control_state_value = msg.control_state

    def _uav_state_cb(self, msg):
        with self.lock:
            self.uav_state = msg
            self.uav_state_receipt = rospy.Time.now()
            position = np.asarray(msg.position, dtype=np.float64)
            if (
                self.home_xy is None
                and msg.odom_valid
                and self._finite(position)
            ):
                self.home_xy = position[:2].copy()
                rospy.loginfo(
                    "[navrl-bridge] local safety-fence origin=[%.3f %.3f]",
                    self.home_xy[0],
                    self.home_xy[1],
                )

    def _fastlio_cb(self, msg):
        with self.lock:
            self.fastlio_odom = msg
            self.fastlio_receipt = rospy.Time.now()

    def _mavros_cb(self, msg):
        with self.lock:
            self.mavros_odom = msg
            self.mavros_receipt = rospy.Time.now()

    def _next_command_id(self):
        self.command_id = (self.command_id + 1) & 0xFFFFFFFF
        return self.command_id

    def _publish_hover(self, reason):
        if not self.output_enabled:
            self._set_status("shadow_{}".format(reason))
            return
        msg = UAVCommand()
        msg.header.stamp = rospy.Time.now()
        msg.Agent_CMD = UAVCommand.Current_Pos_Hover
        msg.Control_Level = UAVCommand.DEFAULT_CONTROL
        msg.Move_mode = UAVCommand.XYZ_POS
        msg.Yaw_Rate_Mode = False
        msg.Command_ID = self._next_command_id()
        self.command_pub.publish(msg)
        self._set_status("hold_{}".format(reason))

    def _publish_move(self, desired, velocity):
        msg = UAVCommand()
        msg.header.stamp = rospy.Time.now()
        msg.Agent_CMD = UAVCommand.Move
        msg.Control_Level = UAVCommand.DEFAULT_CONTROL
        msg.Move_mode = UAVCommand.XYZ_VEL
        msg.velocity_ref[0] = float(velocity[0])
        msg.velocity_ref[1] = float(velocity[1])
        msg.velocity_ref[2] = float(velocity[2])
        msg.yaw_ref = float(desired.yaw)
        msg.Yaw_Rate_Mode = False
        msg.yaw_rate_ref = 0.0
        msg.Command_ID = self._next_command_id()
        self.command_pub.publish(msg)

        safe = PositionTarget()
        safe.header.stamp = msg.header.stamp
        safe.header.frame_id = desired.header.frame_id
        safe.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        safe.type_mask = desired.type_mask
        safe.velocity.x = msg.velocity_ref[0]
        safe.velocity.y = msg.velocity_ref[1]
        safe.velocity.z = msg.velocity_ref[2]
        safe.yaw = msg.yaw_ref
        self.safe_setpoint_pub.publish(safe)
        self._set_status("active_forwarding")

    def _control_state_is_fresh(self, now):
        return (
            self.control_state is not None
            and self.control_state_receipt is not None
            and (now - self.control_state_receipt).to_sec()
            <= self.control_state_timeout
        )

    def _state_is_fresh(self, now):
        return (
            self.uav_state is not None
            and self.uav_state_receipt is not None
            and (now - self.uav_state_receipt).to_sec() <= self.state_timeout
        )

    def _odom_consistent(self, now):
        if not self.enable_odom_consistency_check:
            return True
        if (
            self.fastlio_odom is None
            or self.mavros_odom is None
            or self.fastlio_receipt is None
            or self.mavros_receipt is None
            or (now - self.fastlio_receipt).to_sec() > self.odom_consistency_timeout
            or (now - self.mavros_receipt).to_sec() > self.odom_consistency_timeout
        ):
            return False
        delta = self._position_from_odom(self.mavros_odom) - self._position_from_odom(
            self.fastlio_odom
        )
        yaw_delta = self._wrap_angle(
            self._yaw_from_odom(self.mavros_odom)
            - self._yaw_from_odom(self.fastlio_odom)
        )
        if not self._finite(np.concatenate([delta, [yaw_delta]])):
            return False
        if self.odom_delta_baseline is None:
            self.odom_delta_baseline = delta.copy()
            self.odom_yaw_delta_baseline = yaw_delta
            rospy.loginfo(
                "[navrl-bridge] odom delta baseline=[%.3f %.3f %.3f] yaw=%.3f",
                delta[0],
                delta[1],
                delta[2],
                yaw_delta,
            )
            return True
        position_ok = (
            np.linalg.norm(delta - self.odom_delta_baseline)
            <= self.max_odom_delta_change
        )
        yaw_ok = (
            abs(self._wrap_angle(yaw_delta - self.odom_yaw_delta_baseline))
            <= self.max_odom_yaw_delta_change
        )
        return position_ok and yaw_ok

    def _bounded_velocity(self, desired, state):
        velocity = np.array(
            [desired.velocity.x, desired.velocity.y, desired.velocity.z],
            dtype=np.float64,
        )
        if not self._finite(np.concatenate([velocity, [desired.yaw]])):
            return None, "invalid_desired"
        horizontal = np.linalg.norm(velocity[:2])
        if horizontal > self.max_xy_speed:
            velocity[:2] *= self.max_xy_speed / horizontal
        velocity[2] = float(np.clip(velocity[2], -self.max_z_speed, self.max_z_speed))

        position = np.asarray(state.position, dtype=np.float64)
        if position.shape != (3,) or not self._finite(position):
            return None, "invalid_state_position"
        if self.home_xy is None:
            return None, "no_fence_origin"

        predicted = position + self.fence_prediction_horizon * velocity
        if np.linalg.norm(position[:2] - self.home_xy) > self.max_xy_from_home:
            return None, "outside_xy_fence"
        if np.linalg.norm(predicted[:2] - self.home_xy) > self.max_xy_from_home:
            return None, "predicted_xy_fence"
        if position[2] < self.min_z - 0.10 or position[2] > self.max_z + 0.10:
            return None, "outside_z_fence"
        if predicted[2] < self.min_z or predicted[2] > self.max_z:
            return None, "predicted_z_fence"
        return velocity, "ready"

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self.lock:
            control_fresh = self._control_state_is_fresh(now)
            last_was_command = (
                self.last_control_state_value == UAVControlState.COMMAND_CONTROL
            )
            if not control_fresh:
                if last_was_command:
                    self._publish_hover("stale_control_state")
                else:
                    self._set_status("waiting_for_control_state")
                return

            in_command = (
                self.control_state.control_state
                == UAVControlState.COMMAND_CONTROL
            )
            if not self.output_enabled:
                if self.desired is None:
                    self._set_status("shadow_waiting_for_policy")
                    return
                if (
                    self.desired_receipt is None
                    or (now - self.desired_receipt).to_sec() > self.command_timeout
                ):
                    self._set_status("shadow_stale_policy")
                    return
                if not self._state_is_fresh(now) or not self.uav_state.odom_valid:
                    self._set_status("shadow_unhealthy_state")
                    return
                if not self._odom_consistent(now):
                    self._set_status("shadow_odom_inconsistent")
                    return
                velocity, reason = self._bounded_velocity(self.desired, self.uav_state)
                if velocity is None:
                    self._set_status("shadow_{}".format(reason))
                    return
                # Shadow mode exposes the exact command that would be forwarded.
                safe = PositionTarget()
                safe.header.stamp = now
                safe.header.frame_id = self.desired.header.frame_id
                safe.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
                safe.type_mask = self.desired.type_mask
                safe.velocity.x = float(velocity[0])
                safe.velocity.y = float(velocity[1])
                safe.velocity.z = float(velocity[2])
                safe.yaw = float(self.desired.yaw)
                self.safe_setpoint_pub.publish(safe)
                self._set_status("shadow_ready")
                return

            if not in_command:
                self._set_status("active_waiting_for_command_control")
                return
            if self.control_state.failsafe:
                self._publish_hover("prometheus_failsafe_active")
                return
            if self.control_state.pos_controller != UAVControlState.PX4_ORIGIN:
                self._publish_hover("unsupported_pos_controller")
                return
            if not self._state_is_fresh(now):
                self._publish_hover("stale_uav_state")
                return
            if not self.uav_state.connected:
                self._publish_hover("px4_disconnected")
                return
            if not self.uav_state.odom_valid:
                self._publish_hover("invalid_odom")
                return
            if not self._odom_consistent(now):
                self._publish_hover("odom_inconsistent")
                return
            if (
                self.desired is None
                or self.desired_receipt is None
                or (now - self.desired_receipt).to_sec() > self.command_timeout
            ):
                self._publish_hover("stale_policy")
                return

            velocity, reason = self._bounded_velocity(self.desired, self.uav_state)
            if velocity is None:
                self._publish_hover(reason)
                return
            self._publish_move(self.desired, velocity)

    def _on_shutdown(self):
        if not self.output_enabled:
            return
        if self.last_control_state_value != UAVControlState.COMMAND_CONTROL:
            return
        try:
            for _ in range(3):
                self._publish_hover("bridge_shutdown")
        except Exception:
            pass


def main():
    rospy.init_node("navrl_su17_bridge")
    NavRLSU17Bridge()
    rospy.spin()


if __name__ == "__main__":
    main()
