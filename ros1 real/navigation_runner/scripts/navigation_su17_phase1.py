#!/usr/bin/env python3

"""SU17 NavRL phase-1 inference node.

This node never arms the aircraft, changes PX4 mode, or publishes to MAVROS.
It converts the MID-360 occupancy-map raycast into the observation used during
training and publishes a private, world-ENU velocity target.  A separate
watchdog bridge is the only component allowed to forward that target to the
Prometheus controller.
"""

import math
import os
import sys
import threading
import time

import numpy as np
import rospkg
import rospy
import tf.transformations
import torch
from geometry_msgs.msg import Point, PoseStamped
from map_manager.srv import RayCast
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from std_msgs.msg import Empty, Header, String
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type

PACKAGE_SCRIPTS_DIR = os.path.join(
    rospkg.RosPack().get_path("navigation_runner"), "scripts"
)
if PACKAGE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_SCRIPTS_DIR)

from ppo import PPO
from utils import vec_to_new_frame


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


class SU17Phase1Navigation:
    def __init__(self):
        self.lock = threading.RLock()

        self.uav_id = int(rospy.get_param("~uav_id", 1))
        prefix = "/uav{}".format(self.uav_id)
        self.world_frame = str(rospy.get_param("~world_frame", "world"))
        self.odom_topic = str(
            rospy.get_param("~odom_topic", prefix + "/mavros/local_position/odom")
        )
        self.point_cloud_topic = str(
            rospy.get_param("~point_cloud_topic", prefix + "/cloud_mid360_body")
        )
        self.goal_topic = str(rospy.get_param("~goal_topic", "/move_base_simple/goal"))
        self.command_topic = str(
            rospy.get_param("~command_topic", prefix + "/navrl/desired_setpoint")
        )
        self.raycast_service_name = str(
            rospy.get_param("~raycast_service", "/occupancy_map/raycast")
        )
        self.map_update_topic = str(
            rospy.get_param("~map_update_topic", "/occupancy_map/update")
        )

        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 10.0))
        self.odom_timeout = float(rospy.get_param("~odom_timeout", 0.20))
        self.cloud_timeout = float(rospy.get_param("~cloud_timeout", 0.25))
        self.map_update_timeout = float(rospy.get_param("~map_update_timeout", 0.30))
        self.max_xy_speed = float(rospy.get_param("~max_xy_speed", 0.50))
        self.max_z_speed = float(rospy.get_param("~max_z_speed", 0.30))
        self.min_z = float(rospy.get_param("~min_z", 0.30))
        self.max_z = float(rospy.get_param("~max_z", 1.50))
        self.goal_stop_radius = float(rospy.get_param("~goal_stop_radius", 0.40))
        self.goal_xy_settle_radius = float(
            rospy.get_param("~goal_xy_settle_radius", 0.30)
        )
        self.goal_height_tolerance = float(
            rospy.get_param("~goal_height_tolerance", 0.15)
        )
        self.goal_slow_radius = float(rospy.get_param("~goal_slow_radius", 1.50))
        self.goal_slow_min_speed = float(
            rospy.get_param("~goal_slow_min_speed", 0.10)
        )
        self.goal_slow_max_speed = float(
            rospy.get_param("~goal_slow_max_speed", 0.35)
        )
        self.goal_vertical_settle_speed = float(
            rospy.get_param("~goal_vertical_settle_speed", 0.20)
        )
        self.yaw_alignment_tolerance = float(
            rospy.get_param("~yaw_alignment_tolerance", 0.15)
        )
        self.yaw_settle_cycles = int(rospy.get_param("~yaw_settle_cycles", 3))
        self.emergency_stop_distance = float(
            rospy.get_param("~emergency_stop_distance", 0.55)
        )
        self.use_goal_z = _bool_param("~use_goal_z", True)
        self.odom_twist_in_body = _bool_param("~odom_twist_in_body", True)

        numeric_params = (
            self.control_rate_hz,
            self.odom_timeout,
            self.cloud_timeout,
            self.map_update_timeout,
            self.max_xy_speed,
            self.max_z_speed,
            self.min_z,
            self.max_z,
            self.goal_stop_radius,
            self.goal_xy_settle_radius,
            self.goal_height_tolerance,
            self.emergency_stop_distance,
        )
        if not all(math.isfinite(value) for value in numeric_params):
            raise ValueError("all safety and timing parameters must be finite")
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        if min(self.odom_timeout, self.cloud_timeout, self.map_update_timeout) <= 0.0:
            raise ValueError("input timeouts must be positive")
        if self.max_xy_speed <= 0.0 or self.max_z_speed <= 0.0:
            raise ValueError("velocity limits must be positive")
        if self.min_z >= self.max_z:
            raise ValueError("min_z must be smaller than max_z")
        if min(
            self.goal_stop_radius,
            self.goal_xy_settle_radius,
            self.goal_height_tolerance,
            self.emergency_stop_distance,
        ) < 0.0:
            raise ValueError("goal and clearance distances cannot be negative")

        torch_threads = int(rospy.get_param("~torch_num_threads", 2))
        torch.set_num_threads(max(1, torch_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.cfg = self._load_runtime_config()
        self.lidar_range = float(self.cfg.sensor.lidar_range)
        self.lidar_vbeams = int(self.cfg.sensor.lidar_vbeams)
        self.lidar_hres = float(self.cfg.sensor.lidar_hres)
        self.lidar_hbeams = int(round(360.0 / self.lidar_hres))
        self.expected_rays = self.lidar_hbeams * self.lidar_vbeams
        if self.expected_rays != 144:
            raise ValueError(
                "The deployed checkpoint expects 144 rays; configuration gives {}".format(
                    self.expected_rays
                )
            )

        self.policy = self._load_policy()
        self.policy.eval()

        self.odom = None
        self.odom_receipt_time = None
        self.cloud_receipt_time = None
        self.map_update_receipt_time = None
        self.goal = None
        self.target_dir = None
        self.navigation_frame_yaw = None
        self.goal_start_position = None
        self.yaw_stable_count = 0
        self.goal_active = False
        self.last_status = None

        self.command_pub = rospy.Publisher(
            self.command_topic, PositionTarget, queue_size=2
        )
        self.status_pub = rospy.Publisher(
            prefix + "/navrl/navigation_status", String, queue_size=5, latch=True
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=5
        )
        # AnyMsg records source health without deserializing thousands of points
        # into Python.  The C++ occupancy map remains the point-cloud consumer.
        self.cloud_sub = rospy.Subscriber(
            self.point_cloud_topic, rospy.AnyMsg, self._cloud_cb, queue_size=1
        )
        self.map_update_sub = rospy.Subscriber(
            self.map_update_topic, Header, self._map_update_cb, queue_size=2
        )
        self.goal_sub = rospy.Subscriber(
            self.goal_topic, PoseStamped, self._goal_cb, queue_size=2
        )
        self.cancel_sub = rospy.Subscriber(
            prefix + "/navrl/cancel", Empty, self._cancel_cb, queue_size=2
        )
        self.raycast_client = rospy.ServiceProxy(
            self.raycast_service_name, RayCast
        )
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / self.control_rate_hz), self._control_cb
        )

        self._set_status("waiting_for_odom_and_map")
        rospy.loginfo(
            "[navrl-su17] phase-1 inference ready: odom=%s cloud=%s command=%s",
            self.odom_topic,
            self.point_cloud_topic,
            self.command_topic,
        )

    def _load_runtime_config(self):
        cfg_dir = os.path.join(PACKAGE_SCRIPTS_DIR, "cfg")
        cfg = OmegaConf.merge(
            OmegaConf.load(os.path.join(cfg_dir, "drone.yaml")),
            OmegaConf.load(os.path.join(cfg_dir, "ppo.yaml")),
        )
        cfg.device = "cpu"
        return cfg

    def _load_policy(self):
        observation_dim = 8
        dynamic_state_dim = 10
        action_dim = 3
        device = torch.device("cpu")
        observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": CompositeSpec(
                            {
                                "state": UnboundedContinuousTensorSpec(
                                    (observation_dim,), device=device
                                ),
                                "lidar": UnboundedContinuousTensorSpec(
                                    (1, 36, 4), device=device
                                ),
                                "direction": UnboundedContinuousTensorSpec(
                                    (1, 3), device=device
                                ),
                                "dynamic_obstacle": UnboundedContinuousTensorSpec(
                                    (
                                        1,
                                        int(self.cfg.algo.feature_extractor.dyn_obs_num),
                                        dynamic_state_dim,
                                    ),
                                    device=device,
                                ),
                            }
                        )
                    }
                ).expand(1)
            },
            shape=[1],
            device=device,
        )
        action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "action": UnboundedContinuousTensorSpec(
                            (action_dim,), device=device
                        )
                    }
                )
            }
        ).expand(1, action_dim).to(device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, device)
        checkpoint_default = os.path.join(
            PACKAGE_SCRIPTS_DIR,
            "ckpts",
            "navrl_checkpoint.pt",
        )
        checkpoint_path = os.path.abspath(
            os.path.expanduser(rospy.get_param("~checkpoint", checkpoint_default))
        )
        strict = _bool_param("~strict_checkpoint", True)
        if not os.path.isfile(checkpoint_path):
            raise IOError("checkpoint not found: {}".format(checkpoint_path))
        checkpoint_state = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
            checkpoint_state = checkpoint_state["state_dict"]
        incompatible = policy.load_state_dict(checkpoint_state, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            rospy.logwarn(
                "[navrl-su17] non-strict checkpoint load: missing=%s unexpected=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        rospy.loginfo("[navrl-su17] checkpoint loaded on CPU: %s", checkpoint_path)
        return policy

    def _set_status(self, status):
        if status != self.last_status:
            self.last_status = status
            self.status_pub.publish(String(data=status))
            rospy.loginfo("[navrl-su17] status=%s", status)

    def _odom_cb(self, msg):
        with self.lock:
            self.odom = msg
            self.odom_receipt_time = rospy.Time.now()

    def _cloud_cb(self, _msg):
        with self.lock:
            self.cloud_receipt_time = rospy.Time.now()

    def _map_update_cb(self, _msg):
        with self.lock:
            self.map_update_receipt_time = rospy.Time.now()

    def _cancel_cb(self, _msg):
        with self.lock:
            self.goal_active = False
            self.goal = None
            self.yaw_stable_count = 0
        self._set_status("cancelled_hold_requested")

    @staticmethod
    def _is_finite(values):
        return bool(np.all(np.isfinite(np.asarray(values, dtype=np.float64))))

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        return tf.transformations.euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )[2]

    @staticmethod
    def _wrapped_angle_error(target, current):
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _goal_cb(self, msg):
        with self.lock:
            if self.odom is None:
                rospy.logwarn("[navrl-su17] goal rejected: no odometry")
                return
            current = self.odom.pose.pose.position
            requested_z = msg.pose.position.z if self.use_goal_z else current.z
            goal_xyz = np.array(
                [msg.pose.position.x, msg.pose.position.y, requested_z],
                dtype=np.float64,
            )
            if not self._is_finite(goal_xyz):
                rospy.logerr("[navrl-su17] goal rejected: non-finite coordinates")
                return
            if requested_z < self.min_z or requested_z > self.max_z:
                rospy.logerr(
                    "[navrl-su17] goal rejected: z %.3f outside [%.3f, %.3f]",
                    requested_z,
                    self.min_z,
                    self.max_z,
                )
                return
            if msg.header.frame_id not in ("", self.world_frame):
                rospy.logerr(
                    "[navrl-su17] goal rejected: frame '%s' is not '%s'",
                    msg.header.frame_id,
                    self.world_frame,
                )
                return

            current_xyz = np.array([current.x, current.y, current.z])
            direction = goal_xyz - current_xyz
            current_yaw = self._yaw_from_quaternion(self.odom.pose.pose.orientation)
            if np.linalg.norm(direction[:2]) > 1e-6:
                navigation_yaw = math.atan2(direction[1], direction[0])
            else:
                navigation_yaw = current_yaw
                direction[0] = math.cos(current_yaw)
                direction[1] = math.sin(current_yaw)

            self.goal = goal_xyz
            self.goal_start_position = current_xyz
            self.target_dir = torch.tensor(direction, dtype=torch.float32)
            self.navigation_frame_yaw = navigation_yaw
            self.yaw_stable_count = 0
            self.goal_active = True
        self._set_status("goal_received")

    def _inputs_are_fresh(self, now):
        if self.odom is None or self.odom_receipt_time is None:
            return False, "waiting_for_odom"
        if (now - self.odom_receipt_time).to_sec() > self.odom_timeout:
            return False, "stale_odom"
        if self.cloud_receipt_time is None:
            return False, "waiting_for_cloud"
        if (now - self.cloud_receipt_time).to_sec() > self.cloud_timeout:
            return False, "stale_cloud"
        if self.map_update_receipt_time is None:
            return False, "waiting_for_map_update"
        if (now - self.map_update_receipt_time).to_sec() > self.map_update_timeout:
            return False, "stale_map_update"
        return True, "ready"

    def _get_raycast(self, position):
        position_msg = Point(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        )
        response = self.raycast_client(
            position_msg,
            float(self.navigation_frame_yaw),
            self.lidar_range,
            float(self.cfg.sensor.lidar_vfov[0]),
            float(self.cfg.sensor.lidar_vfov[1]),
            self.lidar_vbeams,
            self.lidar_hres,
        )
        points = np.asarray(response.points, dtype=np.float32)
        if points.size != self.expected_rays * 3:
            raise RuntimeError(
                "raycast returned {} coordinates; expected {}".format(
                    points.size, self.expected_rays * 3
                )
            )
        points = points.reshape(self.expected_rays, 3)
        if not np.all(np.isfinite(points)):
            raise RuntimeError("raycast returned non-finite points")
        return points

    @staticmethod
    def _rotation_matrix(quaternion):
        matrix = tf.transformations.quaternion_matrix(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        return matrix[:3, :3]

    def _build_observation(self, position, velocity_world, goal, raypoints):
        position_t = torch.tensor(position, dtype=torch.float32)
        velocity_t = torch.tensor(velocity_world, dtype=torch.float32)
        goal_t = torch.tensor(goal, dtype=torch.float32)

        relative = goal_t - position_t
        distance = relative.norm().clamp_min(1e-6)
        distance_2d = relative[:2].norm().reshape(1)
        distance_z = relative[2].reshape(1)
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.0
        if target_dir_2d[:2].norm() < 1e-6:
            target_dir_2d = torch.tensor(
                [math.cos(self.navigation_frame_yaw), math.sin(self.navigation_frame_yaw), 0.0],
                dtype=torch.float32,
            )

        relative_goal_frame = vec_to_new_frame(
            relative / distance, target_dir_2d
        ).squeeze(0)
        velocity_goal_frame = vec_to_new_frame(
            velocity_t, target_dir_2d
        ).squeeze(0)
        drone_state = torch.cat(
            [relative_goal_frame, distance_2d, distance_z, velocity_goal_frame],
            dim=-1,
        ).unsqueeze(0)

        distances_np = np.linalg.norm(raypoints - position.reshape(1, 3), axis=1)
        distances_np = np.clip(distances_np, 0.0, self.lidar_range)
        lidar_state = torch.tensor(
            self.lidar_range - distances_np, dtype=torch.float32
        ).reshape(1, 1, self.lidar_hbeams, self.lidar_vbeams)

        dynamic_state = torch.zeros(
            (
                1,
                1,
                int(self.cfg.algo.feature_extractor.dyn_obs_num),
                10,
            ),
            dtype=torch.float32,
        )
        observation = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": TensorDict(
                            {
                                "state": drone_state,
                                "lidar": lidar_state,
                                "direction": target_dir_2d.unsqueeze(0),
                                "dynamic_obstacle": dynamic_state,
                            },
                            batch_size=[1],
                        )
                    },
                    batch_size=[1],
                )
            },
            batch_size=[1],
        )
        return observation, distances_np

    def _limit_velocity(self, velocity, current_z, goal_distance):
        velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        horizontal_speed = np.linalg.norm(velocity[:2])
        if horizontal_speed > self.max_xy_speed:
            velocity[:2] *= self.max_xy_speed / horizontal_speed
        velocity[2] = float(np.clip(velocity[2], -self.max_z_speed, self.max_z_speed))
        if current_z <= self.min_z and velocity[2] < 0.0:
            velocity[2] = 0.0
        if current_z >= self.max_z and velocity[2] > 0.0:
            velocity[2] = 0.0
        if goal_distance < self.goal_slow_radius:
            ratio = max(0.0, goal_distance / max(self.goal_slow_radius, 1e-6))
            speed_limit = self.goal_slow_min_speed + ratio * (
                self.goal_slow_max_speed - self.goal_slow_min_speed
            )
            norm = np.linalg.norm(velocity)
            if norm > speed_limit:
                velocity *= speed_limit / norm
        return velocity

    def _publish_velocity(self, velocity, yaw):
        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_PX
            | PositionTarget.IGNORE_PY
            | PositionTarget.IGNORE_PZ
            | PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        msg.velocity.x = float(velocity[0])
        msg.velocity.y = float(velocity[1])
        msg.velocity.z = float(velocity[2])
        msg.yaw = float(yaw)
        self.command_pub.publish(msg)

    def _control_cb(self, _event):
        started = time.perf_counter()
        now = rospy.Time.now()
        try:
            with self.lock:
                fresh, reason = self._inputs_are_fresh(now)
                if not fresh:
                    self._set_status(reason)
                    return
                if not self.goal_active or self.goal is None:
                    self._set_status("idle_no_goal")
                    return
                odom = self.odom
                goal = self.goal.copy()

            position = np.array(
                [
                    odom.pose.pose.position.x,
                    odom.pose.pose.position.y,
                    odom.pose.pose.position.z,
                ],
                dtype=np.float64,
            )
            quaternion = odom.pose.pose.orientation
            velocity = np.array(
                [
                    odom.twist.twist.linear.x,
                    odom.twist.twist.linear.y,
                    odom.twist.twist.linear.z,
                ],
                dtype=np.float64,
            )
            if self.odom_twist_in_body:
                velocity = self._rotation_matrix(quaternion).dot(velocity)
            if not self._is_finite(np.concatenate([position, velocity])):
                self._set_status("invalid_odom_values")
                return

            relative = goal - position
            goal_distance = float(np.linalg.norm(relative))
            goal_xy_distance = float(np.linalg.norm(relative[:2]))
            height_error = float(relative[2])
            frame_direction_xy = np.array(
                [
                    math.cos(self.navigation_frame_yaw),
                    math.sin(self.navigation_frame_yaw),
                ],
                dtype=np.float64,
            )
            along_track_remaining = float(
                np.dot(relative[:2], frame_direction_xy)
            )
            cross_track_error = float(
                abs(
                    frame_direction_xy[0] * relative[1]
                    - frame_direction_xy[1] * relative[0]
                )
            )
            passed_goal_plane = (
                along_track_remaining <= 0.0
                and cross_track_error <= self.goal_xy_settle_radius
                and abs(height_error) <= self.goal_height_tolerance
            )
            if (
                goal_distance <= self.goal_stop_radius
                or passed_goal_plane
                or (
                    goal_xy_distance <= self.goal_xy_settle_radius
                    and abs(height_error) <= self.goal_height_tolerance
                )
            ):
                self._publish_velocity(np.zeros(3), self.navigation_frame_yaw)
                with self.lock:
                    self.goal_active = False
                self._set_status("goal_reached_hold_requested")
                return

            current_yaw = self._yaw_from_quaternion(quaternion)
            yaw_error = self._wrapped_angle_error(
                self.navigation_frame_yaw, current_yaw
            )
            if abs(yaw_error) > self.yaw_alignment_tolerance:
                self.yaw_stable_count = 0
                self._publish_velocity(np.zeros(3), self.navigation_frame_yaw)
                self._set_status("aligning_yaw")
                return
            self.yaw_stable_count += 1
            if self.yaw_stable_count <= self.yaw_settle_cycles:
                self._publish_velocity(np.zeros(3), self.navigation_frame_yaw)
                self._set_status("settling_yaw")
                return

            if goal_xy_distance <= self.goal_xy_settle_radius:
                vertical = float(
                    np.clip(
                        height_error,
                        -self.goal_vertical_settle_speed,
                        self.goal_vertical_settle_speed,
                    )
                )
                command = self._limit_velocity(
                    np.array([0.0, 0.0, vertical]), position[2], goal_distance
                )
                self._publish_velocity(command, self.navigation_frame_yaw)
                self._set_status("vertical_settle")
                return

            raypoints = self._get_raycast(position)
            observation, ray_distances = self._build_observation(
                position, velocity, goal, raypoints
            )
            if (
                self.emergency_stop_distance > 0.0
                and float(np.min(ray_distances)) < self.emergency_stop_distance
            ):
                self._publish_velocity(np.zeros(3), self.navigation_frame_yaw)
                self._set_status("emergency_clearance_stop")
                return

            with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
                output = self.policy(observation)
                command = (
                    output["agents", "action"]
                    .squeeze(0)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            if command.shape != (3,) or not self._is_finite(command):
                self._set_status("invalid_policy_output")
                return
            command = self._limit_velocity(command, position[2], goal_distance)
            self._publish_velocity(command, self.navigation_frame_yaw)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._set_status("navigating")
            rospy.loginfo_throttle(
                1.0,
                "[navrl-su17] inference/control %.1f ms cmd=[%.2f %.2f %.2f]",
                elapsed_ms,
                command[0],
                command[1],
                command[2],
            )
        except rospy.ServiceException as exc:
            self._set_status("raycast_service_error")
            rospy.logwarn_throttle(1.0, "[navrl-su17] raycast failed: %s", exc)
        except Exception as exc:
            self._set_status("control_exception")
            rospy.logerr_throttle(1.0, "[navrl-su17] control error: %s", exc)


def main():
    rospy.init_node("navigation_su17_phase1")
    SU17Phase1Navigation()
    rospy.spin()


if __name__ == "__main__":
    main()
