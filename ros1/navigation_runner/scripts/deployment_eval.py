#!/usr/bin/env python3

import colorsys
import csv
import math
import os
import random
import time

import rospy
import tf.transformations
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Point, PoseStamped
from map_manager.srv import RayCast
from nav_msgs.msg import Odometry
from onboard_detector.srv import GetDynamicObstacles
from std_msgs.msg import ColorRGBA, Int32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


class DeploymentEvaluator:
    def __init__(self):
        self.num_trials = int(rospy.get_param("~num_trials", 100))
        self.model_name = rospy.get_param("~model_name", "quadcopter")
        self.random_seed = int(rospy.get_param("~random_seed", 0))
        self.rng = random.Random(self.random_seed)

        legacy_boundary_half_size = rospy.get_param("~boundary_half_size", None)
        if legacy_boundary_half_size is None:
            self.start_boundary_half_size = float(rospy.get_param("~start_boundary_half_size", 11.0))
            self.goal_boundary_half_size = float(rospy.get_param("~goal_boundary_half_size", 11.0))
        else:
            legacy_boundary_half_size = float(legacy_boundary_half_size)
            self.start_boundary_half_size = float(rospy.get_param("~start_boundary_half_size", legacy_boundary_half_size))
            self.goal_boundary_half_size = float(rospy.get_param("~goal_boundary_half_size", legacy_boundary_half_size))
        self.boundary_axis = str(rospy.get_param("~boundary_axis", "y")).strip().lower()
        if self.boundary_axis == "y":
            self.boundary_sides = (0, 1)
        elif self.boundary_axis == "x":
            self.boundary_sides = (2, 3)
        else:
            raise ValueError("~boundary_axis must be either 'x' or 'y'")
        self.random_height = bool(rospy.get_param("~random_height", True))
        self.height_min = float(rospy.get_param("~height_min", 0.5))
        self.height_max = float(rospy.get_param("~height_max", 2.5))
        if self.height_max < self.height_min:
            raise ValueError("~height_max must be greater than or equal to ~height_min")
        self.start_z = float(rospy.get_param("~start_z", 1.0))
        self.goal_z = float(rospy.get_param("~goal_z", self.start_z))

        # Keep the deployment metric defaults aligned with training env.py.
        self.success_radius = float(rospy.get_param("~success_radius", 0.5))
        self.success_xy_radius = float(
            rospy.get_param(
                "~success_xy_radius",
                rospy.get_param("rl/goal_xy_settle_radius", self.success_radius),
            )
        )
        self.success_height_tolerance = float(
            rospy.get_param(
                "~success_height_tolerance",
                rospy.get_param("rl/goal_stop_height_tolerance", 0.2),
            )
        )
        # Collision radius used for static raycast hits and dynamic obstacle
        # inflation during deployment evaluation.
        self.collision_radius = float(rospy.get_param("~collision_radius", 0.15))
        self.dynamic_collision_range = float(rospy.get_param("~dynamic_collision_range", 4.0))
        self.timeout = float(rospy.get_param("~timeout", 180.0))
        self.eval_rate_hz = float(rospy.get_param("~eval_rate_hz", 20.0))
        self.stuck_window = int(rospy.get_param("~stuck_window", 40))
        self.stuck_progress_eps = float(rospy.get_param("~stuck_progress_eps", 0.005))
        self.stuck_front_distance = float(rospy.get_param("~stuck_front_distance", 1.5))
        self.stuck_front_angle_deg = float(rospy.get_param("~stuck_front_angle_deg", 35.0))
        self.stuck_front_tan = math.tan(math.radians(self.stuck_front_angle_deg))
        self.stuck_front_height = float(rospy.get_param("~stuck_front_height", 0.75))
        self.deadlock_recovery_window = max(
            1, int(rospy.get_param("~deadlock_recovery_window", 180))
        )
        self.deadlock_recovery_min_displacement = max(
            0.0,
            float(rospy.get_param("~deadlock_recovery_min_displacement", 0.3)),
        )
        self.deadlock_recovery_min_goal_progress = max(
            0.0,
            float(rospy.get_param("~deadlock_recovery_min_goal_progress", 0.2)),
        )
        self.deadlock_recovery_min_clearance_gain = max(
            0.0,
            float(rospy.get_param("~deadlock_recovery_min_clearance_gain", 0.2)),
        )
        self.lidar_range = float(rospy.get_param("~lidar_range", 4.0))
        self.lidar_vfov_min = float(rospy.get_param("~lidar_vfov_min", -10.0))
        self.lidar_vfov_max = float(rospy.get_param("~lidar_vfov_max", 20.0))
        self.lidar_vbeams = int(rospy.get_param("~lidar_vbeams", 4))
        self.lidar_hres = float(rospy.get_param("~lidar_hres", 10.0))
        self.reset_wait = float(rospy.get_param("~reset_wait", 2.0))
        self.reset_position_tolerance = float(rospy.get_param("~reset_position_tolerance", 0.15))
        self.reset_speed_tolerance = float(rospy.get_param("~reset_speed_tolerance", 0.10))
        self.reset_stable_time = float(rospy.get_param("~reset_stable_time", 0.30))
        self.goal_publish_time = float(rospy.get_param("~goal_publish_time", 1.0))
        self.csv_path = rospy.get_param("~csv_path", "")
        self.publish_trajectories = bool(rospy.get_param("~publish_trajectories", True))
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/deployment_eval/trajectories"
        )
        self.trajectory_selection_topic = rospy.get_param(
            "~trajectory_selection_topic", "/deployment_eval/trajectory_selection"
        )
        self.trajectory_frame = rospy.get_param("~trajectory_frame", "map")
        self.trajectory_line_width = float(
            rospy.get_param("~trajectory_line_width", 0.06)
        )
        self.trajectory_alpha = float(rospy.get_param("~trajectory_alpha", 0.90))
        self.trajectory_color_r = float(rospy.get_param("~trajectory_color_r", 0.0))
        self.trajectory_color_g = float(rospy.get_param("~trajectory_color_g", 0.0))
        self.trajectory_color_b = float(rospy.get_param("~trajectory_color_b", 0.50))
        self.failed_trajectory_color_r = float(
            rospy.get_param("~failed_trajectory_color_r", 0.90)
        )
        self.failed_trajectory_color_g = float(
            rospy.get_param("~failed_trajectory_color_g", 0.05)
        )
        self.failed_trajectory_color_b = float(
            rospy.get_param("~failed_trajectory_color_b", 0.05)
        )
        self.trajectory_min_point_distance = float(
            rospy.get_param("~trajectory_min_point_distance", 0.05)
        )
        self.trajectory_endpoint_scale = float(
            rospy.get_param("~trajectory_endpoint_scale", 0.20)
        )
        self.publish_trajectories_during_trials = bool(
            rospy.get_param("~publish_trajectories_during_trials", False)
        )
        self.publish_full_environment = bool(
            rospy.get_param("~publish_full_environment", True)
        )
        self.environment_topic = rospy.get_param(
            "~environment_topic", "/deployment_eval/environment"
        )
        self.full_map_pcd = os.path.abspath(
            os.path.expanduser(
                rospy.get_param("~full_map_pcd", self.default_full_map_pcd())
            )
        )
        self.full_map_voxel_size = float(
            rospy.get_param("~full_map_voxel_size", 0.10)
        )
        if self.full_map_voxel_size <= 0.0:
            raise ValueError("~full_map_voxel_size must be positive")
        self.static_map_alpha = float(rospy.get_param("~static_map_alpha", 1.0))
        self.static_map_color_min_z = rospy.get_param(
            "~static_map_color_min_z", None
        )
        self.static_map_color_max_z = rospy.get_param(
            "~static_map_color_max_z", None
        )
        self.publish_dynamic_bounding_boxes = bool(
            rospy.get_param("~publish_dynamic_bounding_boxes", True)
        )
        self.dynamic_bbox_line_width = float(
            rospy.get_param("~dynamic_bbox_line_width", 0.06)
        )
        self.dynamic_bbox_padding = float(
            rospy.get_param("~dynamic_bbox_padding", 0.03)
        )
        self.keep_trajectory_publisher_alive = bool(
            rospy.get_param("~keep_trajectory_publisher_alive", True)
        )

        self.latest_odom = None
        self.latest_model_states = None
        self.trajectory_markers = []
        self.selected_trajectory_trials = None
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self.odom_callback)
        self.model_states_sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_callback, queue_size=1
        )
        self.trajectory_pub = None
        self.trajectory_selection_sub = None
        if self.publish_trajectories:
            self.trajectory_pub = rospy.Publisher(
                self.trajectory_topic,
                MarkerArray,
                queue_size=10,
                latch=True,
            )
            self.clear_trajectory_markers()
            self.trajectory_selection_sub = rospy.Subscriber(
                self.trajectory_selection_topic,
                Int32MultiArray,
                self.trajectory_selection_callback,
                queue_size=1,
            )
        self.environment_pub = None
        if self.publish_full_environment:
            self.environment_pub = rospy.Publisher(
                self.environment_topic,
                MarkerArray,
                queue_size=1,
                latch=True,
            )
            self.clear_environment_markers()

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.wait_for_service("occupancy_map/raycast")
        self.raycast = rospy.ServiceProxy("occupancy_map/raycast", RayCast)
        self.dynamic_obstacle_service = None
        try:
            rospy.wait_for_service("onboard_detector/get_dynamic_obstacles", timeout=5.0)
            self.dynamic_obstacle_service = rospy.ServiceProxy(
                "onboard_detector/get_dynamic_obstacles", GetDynamicObstacles
            )
        except rospy.ROSException:
            rospy.logwarn(
                "[deployment-eval] onboard_detector/get_dynamic_obstacles unavailable; "
                "dynamic collision checking is disabled."
            )

    def odom_callback(self, msg):
        self.latest_odom = msg

    def model_states_callback(self, msg):
        self.latest_model_states = msg

    @staticmethod
    def default_full_map_pcd():
        relative_path = os.path.join(
            "uav_simulator", "worlds", "generated_env", "generated_env.pcd"
        )
        candidates = [
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", relative_path)
            )
        ]
        for root in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
            if root:
                candidates.append(os.path.join(root, relative_path))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return candidates[0]

    def clear_trajectory_markers(self):
        if self.trajectory_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL
        message = MarkerArray()
        message.markers = [marker]
        self.trajectory_pub.publish(message)

    def trajectory_selection_callback(self, msg):
        requested_trials = sorted(set(int(value) for value in msg.data))
        if not requested_trials:
            self.selected_trajectory_trials = None
            rospy.loginfo(
                "[deployment-eval] trajectory selection cleared; displaying all trials"
            )
        else:
            valid_trials = [
                trial
                for trial in requested_trials
                if 1 <= trial <= self.num_trials
            ]
            invalid_trials = sorted(set(requested_trials) - set(valid_trials))
            if invalid_trials:
                rospy.logwarn(
                    "[deployment-eval] ignoring invalid trajectory trial IDs: %s "
                    "(valid range: 1-%d)",
                    invalid_trials,
                    self.num_trials,
                )
            self.selected_trajectory_trials = {
                trial - 1 for trial in valid_trials
            }
            rospy.loginfo(
                "[deployment-eval] displaying trajectory trials: %s",
                valid_trials if valid_trials else "none",
            )

        if self.trajectory_markers:
            self.publish_selected_trajectory_markers(clear_existing=True)

    def clear_environment_markers(self):
        if self.environment_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL
        message = MarkerArray()
        message.markers = [marker]
        self.environment_pub.publish(message)

    def append_trajectory_point(self, points, odom, force=False):
        if not self.publish_trajectories:
            return
        position = odom.pose.pose.position
        point = Point(x=position.x, y=position.y, z=position.z)
        if not points:
            points.append(point)
            return

        last = points[-1]
        distance = math.sqrt(
            (point.x - last.x) ** 2
            + (point.y - last.y) ** 2
            + (point.z - last.z) ** 2
        )
        if force or distance >= self.trajectory_min_point_distance:
            if distance > 1e-6:
                points.append(point)

    def add_trajectory_marker(self, trial_idx, points, success, collision):
        if self.trajectory_pub is None or not points:
            return

        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = rospy.Time.now()
        marker.id = int(trial_idx)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.trajectory_line_width
        marker.color.a = self.trajectory_alpha
        marker.points = points
        marker.lifetime = rospy.Duration(0)

        if success:
            marker.ns = "deployment_eval_success"
            marker.color.r = self.trajectory_color_r
            marker.color.g = self.trajectory_color_g
            marker.color.b = self.trajectory_color_b
        elif collision:
            marker.ns = "deployment_eval_collision"
            marker.color.r = self.failed_trajectory_color_r
            marker.color.g = self.failed_trajectory_color_g
            marker.color.b = self.failed_trajectory_color_b
        else:
            marker.ns = "deployment_eval_timeout"
            marker.color.r = self.failed_trajectory_color_r
            marker.color.g = self.failed_trajectory_color_g
            marker.color.b = self.failed_trajectory_color_b

        start_marker = self.make_trajectory_endpoint_marker(
            trial_idx,
            points[0],
            namespace="deployment_eval_start",
            marker_type=Marker.SPHERE,
            color=(0.05, 0.75, 0.20),
        )
        end_marker = self.make_trajectory_endpoint_marker(
            trial_idx,
            points[-1],
            namespace="deployment_eval_end",
            marker_type=Marker.CUBE,
            color=(0.90, 0.10, 0.10),
        )
        trial_markers = [marker, start_marker, end_marker]
        self.trajectory_markers.extend(trial_markers)
        trial_selected = (
            self.selected_trajectory_trials is None
            or trial_idx in self.selected_trajectory_trials
        )
        if self.publish_trajectories_during_trials and trial_selected:
            message = MarkerArray()
            message.markers = trial_markers
            self.trajectory_pub.publish(message)

    def make_trajectory_endpoint_marker(
        self, trial_idx, point, namespace, marker_type, color
    ):
        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = int(trial_idx)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.trajectory_endpoint_scale
        marker.scale.y = self.trajectory_endpoint_scale
        marker.scale.z = self.trajectory_endpoint_scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0)
        return marker

    def publish_all_trajectory_markers(self):
        self.publish_selected_trajectory_markers(clear_existing=True)

    def publish_selected_trajectory_markers(self, clear_existing=False):
        if self.trajectory_pub is None or not self.trajectory_markers:
            return
        if clear_existing:
            self.clear_trajectory_markers()

        markers = self.trajectory_markers
        if self.selected_trajectory_trials is not None:
            markers = [
                marker
                for marker in markers
                if marker.id in self.selected_trajectory_trials
            ]
        if not markers:
            rospy.logwarn(
                "[deployment-eval] trajectory selection contains no available trials"
            )
            return

        stamp = rospy.Time.now()
        for marker in markers:
            marker.header.stamp = stamp
        message = MarkerArray()
        message.markers = list(markers)
        self.trajectory_pub.publish(message)

    def load_full_map_points(self):
        if not os.path.isfile(self.full_map_pcd):
            rospy.logwarn(
                "[deployment-eval] full-map PCD not found: %s", self.full_map_pcd
            )
            return []

        fields = []
        data_is_ascii = False
        points_by_voxel = {}
        try:
            with open(self.full_map_pcd, "r") as pcd_file:
                for raw_line in pcd_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if not data_is_ascii:
                        tokens = line.split()
                        key = tokens[0].upper()
                        if key == "FIELDS":
                            fields = tokens[1:]
                        elif key == "DATA":
                            if len(tokens) < 2 or tokens[1].lower() != "ascii":
                                rospy.logwarn(
                                    "[deployment-eval] only ASCII PCD files are supported: %s",
                                    self.full_map_pcd,
                                )
                                return []
                            data_is_ascii = True
                        continue

                    values = line.split()
                    if fields:
                        x_idx = fields.index("x")
                        y_idx = fields.index("y")
                        z_idx = fields.index("z")
                    else:
                        x_idx, y_idx, z_idx = 0, 1, 2
                    x = float(values[x_idx])
                    y = float(values[y_idx])
                    z = float(values[z_idx])
                    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                        continue
                    voxel = (
                        int(math.floor(x / self.full_map_voxel_size)),
                        int(math.floor(y / self.full_map_voxel_size)),
                        int(math.floor(z / self.full_map_voxel_size)),
                    )
                    if voxel not in points_by_voxel:
                        points_by_voxel[voxel] = Point(x=x, y=y, z=z)
        except (OSError, ValueError, IndexError) as exc:
            rospy.logwarn(
                "[deployment-eval] failed to read full-map PCD %s: %s",
                self.full_map_pcd,
                exc,
            )
            return []

        if not data_is_ascii:
            rospy.logwarn(
                "[deployment-eval] PCD has no ASCII DATA section: %s", self.full_map_pcd
            )
            return []
        return list(points_by_voxel.values())

    @staticmethod
    def dynamic_model_geometry(model_name):
        parts = model_name.rsplit("_", 3)
        if len(parts) == 4:
            prefix = parts[0]
            try:
                size_x, size_y, size_z = (float(value) for value in parts[1:])
            except ValueError:
                return None
            if prefix.startswith("dynamic_box"):
                return Marker.CUBE, (size_x, size_y, size_z)
            if prefix.startswith("dynamic_cylinder"):
                return Marker.CYLINDER, (size_x, size_y, size_z)
        if model_name.lower().startswith("person"):
            return Marker.CYLINDER, (0.5, 0.5, 1.8)
        return None

    def make_static_environment_marker(self, points, stamp):
        if not points:
            return None

        min_z = (
            float(self.static_map_color_min_z)
            if self.static_map_color_min_z is not None
            else min(point.z for point in points)
        )
        max_z = (
            float(self.static_map_color_max_z)
            if self.static_map_color_max_z is not None
            else max(point.z for point in points)
        )
        color_span = max(max_z - min_z, 1e-6)

        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = stamp
        marker.ns = "deployment_eval_static_map"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.full_map_voxel_size
        marker.scale.y = self.full_map_voxel_size
        marker.scale.z = self.full_map_voxel_size
        # Match RViz's height-based rainbow rendering used for the live
        # inflated voxel map: low voxels are red and high voxels are magenta.
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = self.static_map_alpha
        marker.points = points
        marker.colors = []
        for point in points:
            normalized_height = min(max((point.z - min_z) / color_span, 0.0), 1.0)
            red, green, blue = colorsys.hsv_to_rgb(
                normalized_height * (5.0 / 6.0), 1.0, 1.0
            )
            marker.colors.append(
                ColorRGBA(
                    r=red,
                    g=green,
                    b=blue,
                    a=self.static_map_alpha,
                )
            )
        marker.lifetime = rospy.Duration(0)
        return marker

    def make_dynamic_environment_markers(self, stamp):
        model_states = self.latest_model_states
        if model_states is None:
            rospy.logwarn(
                "[deployment-eval] /gazebo/model_states is unavailable; "
                "the final environment snapshot contains static obstacles only."
            )
            return []

        markers = []
        obstacle_id = 0
        for model_name, pose in zip(model_states.name, model_states.pose):
            geometry = self.dynamic_model_geometry(model_name)
            if geometry is None:
                continue
            _, dimensions = geometry
            if self.publish_dynamic_bounding_boxes:
                markers.append(
                    self.make_dynamic_bounding_box_marker(
                        obstacle_id, pose, dimensions, stamp
                    )
                )
            obstacle_id += 1
        return markers

    def make_dynamic_bounding_box_marker(self, marker_id, pose, dimensions, stamp):
        half_x = dimensions[0] / 2.0 + self.dynamic_bbox_padding
        half_y = dimensions[1] / 2.0 + self.dynamic_bbox_padding
        half_z = dimensions[2] / 2.0 + self.dynamic_bbox_padding
        vertices = [
            Point(x=-half_x, y=-half_y, z=-half_z),
            Point(x=-half_x, y=half_y, z=-half_z),
            Point(x=half_x, y=half_y, z=-half_z),
            Point(x=half_x, y=-half_y, z=-half_z),
            Point(x=-half_x, y=-half_y, z=half_z),
            Point(x=-half_x, y=half_y, z=half_z),
            Point(x=half_x, y=half_y, z=half_z),
            Point(x=half_x, y=-half_y, z=half_z),
        ]
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        marker = Marker()
        marker.header.frame_id = self.trajectory_frame
        marker.header.stamp = stamp
        marker.ns = "deployment_eval_dynamic_bboxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = self.dynamic_bbox_line_width
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = []
        for start_idx, end_idx in edges:
            marker.points.append(vertices[start_idx])
            marker.points.append(vertices[end_idx])
        marker.lifetime = rospy.Duration(0)
        return marker

    def publish_environment_snapshot(self):
        if self.environment_pub is None:
            return
        stamp = rospy.Time.now()
        markers = []
        static_marker = self.make_static_environment_marker(
            self.load_full_map_points(), stamp
        )
        if static_marker is not None:
            markers.append(static_marker)
        dynamic_markers = self.make_dynamic_environment_markers(stamp)
        markers.extend(dynamic_markers)
        if not markers:
            rospy.logwarn("[deployment-eval] final environment snapshot is empty")
            return
        message = MarkerArray()
        message.markers = markers
        self.environment_pub.publish(message)
        rospy.loginfo(
            "[deployment-eval] published final environment snapshot on %s "
            "(%d static voxels, %d dynamic obstacles)",
            self.environment_topic,
            len(static_marker.points) if static_marker is not None else 0,
            sum(
                marker.ns == "deployment_eval_dynamic_bboxes"
                for marker in dynamic_markers
            ),
        )

    def make_pose_msg(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg

    def reset_robot(self, start, goal):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        state.pose = self.make_pose_msg(start[0], start[1], start[2], yaw).pose
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        rate = rospy.Rate(20)
        end_time = time.time() + self.reset_wait
        stable_since = None
        stable = False
        while not rospy.is_shutdown() and time.time() < end_time:
            self.set_model_state(state)
            odom = self.latest_odom
            if odom is not None:
                pos_error = self.distance_to_point(odom, start)
                curr_speed = self.speed(odom)
                if (
                    pos_error <= self.reset_position_tolerance
                    and curr_speed <= self.reset_speed_tolerance
                ):
                    if stable_since is None:
                        stable_since = time.time()
                    if time.time() - stable_since >= self.reset_stable_time:
                        stable = True
                        break
                else:
                    stable_since = None
            rate.sleep()
        if not stable:
            rospy.logwarn(
                "[deployment-eval] reset did not fully settle before goal publish: "
                "pos_tolerance=%.3f speed_tolerance=%.3f",
                self.reset_position_tolerance,
                self.reset_speed_tolerance,
            )

    def sample_boundary_point(self, side, z, half):
        offset = self.rng.uniform(-half, half)
        if side == 0:
            return (offset, half, z)
        if side == 1:
            return (offset, -half, z)
        if side == 2:
            return (half, offset, z)
        return (-half, offset, z)

    def sample_trial_task(self):
        start_side = self.rng.choice(self.boundary_sides)
        goal_side = self.boundary_sides[1] if start_side == self.boundary_sides[0] else self.boundary_sides[0]
        if self.random_height:
            start_z = self.rng.uniform(self.height_min, self.height_max)
            goal_z = self.rng.uniform(self.height_min, self.height_max)
        else:
            start_z = self.start_z
            goal_z = self.goal_z
        start = self.sample_boundary_point(start_side, start_z, self.start_boundary_half_size)
        goal = self.sample_boundary_point(goal_side, goal_z, self.goal_boundary_half_size)
        return start, goal, start_side, goal_side

    def publish_goal_for_a_moment(self, goal, trajectory_points=None):
        goal_msg = self.make_pose_msg(goal[0], goal[1], goal[2], 0.0)
        rate = rospy.Rate(10)
        end_time = time.time() + self.goal_publish_time
        while not rospy.is_shutdown() and time.time() < end_time:
            goal_msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal_msg)
            if trajectory_points is not None and self.latest_odom is not None:
                self.append_trajectory_point(trajectory_points, self.latest_odom)
            rate.sleep()

    def wait_for_odom(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.latest_odom is None:
            rospy.loginfo("[deployment-eval] waiting for /CERLAB/quadcopter/odom ...")
            rate.sleep()

    def distance_to_goal(self, odom, goal):
        pos = odom.pose.pose.position
        dx = goal[0] - pos.x
        dy = goal[1] - pos.y
        dz = goal[2] - pos.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def goal_reached(self, odom, start, goal):
        pos = odom.pose.pose.position
        remaining_x = goal[0] - pos.x
        remaining_y = goal[1] - pos.y
        remaining_z = goal[2] - pos.z
        distance = math.sqrt(
            remaining_x * remaining_x
            + remaining_y * remaining_y
            + remaining_z * remaining_z
        )
        if distance <= self.success_radius:
            return True

        goal_direction_x = goal[0] - start[0]
        goal_direction_y = goal[1] - start[1]
        goal_direction_norm = math.sqrt(
            goal_direction_x * goal_direction_x
            + goal_direction_y * goal_direction_y
        )
        if goal_direction_norm <= 1e-6:
            return False

        frame_x = goal_direction_x / goal_direction_norm
        frame_y = goal_direction_y / goal_direction_norm
        horizontal_distance = math.sqrt(
            remaining_x * remaining_x + remaining_y * remaining_y
        )
        height_reached = abs(remaining_z) <= self.success_height_tolerance
        if horizontal_distance <= self.success_xy_radius and height_reached:
            return True

        along_track_remaining = remaining_x * frame_x + remaining_y * frame_y
        cross_track_error = abs(frame_x * remaining_y - frame_y * remaining_x)
        return (
            along_track_remaining <= 0.0
            and cross_track_error <= self.success_xy_radius
            and height_reached
        )

    def distance_to_point(self, odom, point):
        pos = odom.pose.pose.position
        dx = point[0] - pos.x
        dy = point[1] - pos.y
        dz = point[2] - pos.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def speed(self, odom):
        vel = odom.twist.twist.linear
        return math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)

    def get_static_front_state(self, odom, goal):
        pos = odom.pose.pose.position
        goal_dx = goal[0] - pos.x
        goal_dy = goal[1] - pos.y
        goal_norm = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)
        if goal_norm <= 1e-6:
            return False, self.lidar_range
        forward_x = goal_dx / goal_norm
        forward_y = goal_dy / goal_norm
        lateral_x = -forward_y
        lateral_y = forward_x
        start_angle = math.atan2(forward_y, forward_x)

        try:
            points = self.raycast(
                pos,
                start_angle,
                self.lidar_range,
                self.lidar_vfov_min,
                self.lidar_vfov_max,
                self.lidar_vbeams,
                self.lidar_hres,
            ).points
        except rospy.ServiceException:
            return False, self.lidar_range

        front_obstacle = False
        front_clearance = self.lidar_range
        for i in range(0, len(points), 3):
            rel_x = points[i] - pos.x
            rel_y = points[i + 1] - pos.y
            rel_z = points[i + 2] - pos.z
            forward = rel_x * forward_x + rel_y * forward_y
            lateral = abs(rel_x * lateral_x + rel_y * lateral_y)
            vertical = abs(rel_z)
            horizontal_distance = math.sqrt(rel_x * rel_x + rel_y * rel_y)
            if (
                forward > 0.0
                and horizontal_distance <= self.stuck_front_distance
                and lateral <= self.stuck_front_tan * max(forward, 1e-6)
                and vertical <= self.stuck_front_height
            ):
                front_obstacle = True
                point_distance = math.sqrt(
                    rel_x * rel_x + rel_y * rel_y + rel_z * rel_z
                )
                front_clearance = min(front_clearance, point_distance)
        return front_obstacle, front_clearance

    def get_static_collision(self, odom):
        pos = odom.pose.pose.position
        try:
            points = self.raycast(
                pos,
                0.0,
                self.lidar_range,
                self.lidar_vfov_min,
                self.lidar_vfov_max,
                self.lidar_vbeams,
                self.lidar_hres,
            ).points
        except rospy.ServiceException:
            return False

        for i in range(0, len(points), 3):
            dx = points[i] - pos.x
            dy = points[i + 1] - pos.y
            dz = points[i + 2] - pos.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= self.collision_radius:
                return True
        return False

    def get_dynamic_collision(self, odom):
        if self.dynamic_obstacle_service is None:
            return False

        pos = odom.pose.pose.position
        query = Point(x=pos.x, y=pos.y, z=pos.z)
        try:
            response = self.dynamic_obstacle_service(query, self.dynamic_collision_range)
        except rospy.ServiceException:
            return False

        total_obs_num = min(len(response.position), len(response.velocity), len(response.size))
        for i in range(total_obs_num):
            obs_pos = response.position[i]
            obs_size = response.size[i]
            if obs_size.x == 0.0 and obs_size.y == 0.0 and obs_size.z == 0.0:
                continue

            obstacle_width = max(obs_size.x, obs_size.y)
            horizontal_distance = math.sqrt((obs_pos.x - pos.x) ** 2 + (obs_pos.y - pos.y) ** 2)
            vertical_distance = abs(obs_pos.z - pos.z)

            collision_xy = horizontal_distance <= obstacle_width * 0.5 + self.collision_radius
            collision_z = vertical_distance <= obs_size.z * 0.5 + self.collision_radius
            if collision_xy and collision_z:
                return True
        return False

    def get_collision(self, odom):
        static_collision = self.get_static_collision(odom)
        dynamic_collision = self.get_dynamic_collision(odom)
        if static_collision and dynamic_collision:
            return True, "static+dynamic"
        if static_collision:
            return True, "static"
        if dynamic_collision:
            return True, "dynamic"
        return False, ""

    def run_one_trial(self, trial_idx):
        start, goal, start_side, goal_side = self.sample_trial_task()
        self.reset_robot(start, goal)
        trajectory_points = []
        if self.latest_odom is not None:
            self.append_trajectory_point(trajectory_points, self.latest_odom)
        # Publish the goal only after the reset pose has settled, so the
        # navigation node does not act on a new task while Gazebo is still being
        # forced to the start state.
        self.publish_goal_for_a_moment(goal, trajectory_points)

        start_time = rospy.Time.now().to_sec()
        prev_goal_distance = None
        stuck_counter = 0
        stuck_steps = 0
        deadlock_seen = False
        deadlock_time = None
        deadlock_event_count = 0
        recovered_deadlock_event_count = 0
        recovered_from_deadlock = False
        first_recovery_latency = None
        recovery_armed = False
        recovery_age = 0
        recovery_anchor_xy = None
        recovery_anchor_goal_distance = None
        recovery_anchor_clearance = None
        recovery_event_start_time = None
        success = False
        collision = False
        collision_type = ""
        timed_out = False

        rate = rospy.Rate(self.eval_rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - start_time
            odom = self.latest_odom
            if odom is None:
                rate.sleep()
                continue

            self.append_trajectory_point(trajectory_points, odom)
            distance = self.distance_to_goal(odom, goal)
            if prev_goal_distance is None:
                prev_goal_distance = distance

            goal_progress = prev_goal_distance - distance
            front_obstacle, front_clearance = self.get_static_front_state(odom, goal)
            blocked_low_progress = (
                goal_progress <= self.stuck_progress_eps and front_obstacle
            )
            previous_stuck_counter = stuck_counter
            if blocked_low_progress:
                stuck_counter += 1
            else:
                stuck_counter = 0

            stuck_active = stuck_counter >= self.stuck_window
            if stuck_active:
                stuck_steps += 1
            crossed_deadlock_threshold = (
                previous_stuck_counter < self.stuck_window
                and stuck_counter >= self.stuck_window
            )
            if crossed_deadlock_threshold:
                deadlock_seen = True
                deadlock_event_count += 1
                if deadlock_time is None:
                    deadlock_time = elapsed
                recovery_armed = True
                recovery_age = 0
                recovery_anchor_xy = (odom.pose.pose.position.x, odom.pose.pose.position.y)
                recovery_anchor_goal_distance = distance
                recovery_anchor_clearance = front_clearance
                recovery_event_start_time = elapsed

            if recovery_armed:
                recovery_age += 1
                displacement = math.sqrt(
                    (odom.pose.pose.position.x - recovery_anchor_xy[0]) ** 2
                    + (odom.pose.pose.position.y - recovery_anchor_xy[1]) ** 2
                )
                goal_progress_since_deadlock = (
                    recovery_anchor_goal_distance - distance
                )
                clearance_gain = front_clearance - recovery_anchor_clearance
                recovery_evidence = (
                    displacement >= self.deadlock_recovery_min_displacement
                    and (
                        goal_progress_since_deadlock
                        >= self.deadlock_recovery_min_goal_progress
                        or clearance_gain
                        >= self.deadlock_recovery_min_clearance_gain
                    )
                )
                if not blocked_low_progress and recovery_evidence:
                    recovered_deadlock_event_count += 1
                    recovered_from_deadlock = True
                    if first_recovery_latency is None:
                        first_recovery_latency = elapsed - recovery_event_start_time
                    recovery_armed = False
                elif recovery_age >= self.deadlock_recovery_window:
                    recovery_armed = False

            collision, collision_type = self.get_collision(odom)
            if collision:
                break

            if self.goal_reached(odom, start, goal):
                success = True
                break

            if elapsed >= self.timeout:
                timed_out = True
                break

            prev_goal_distance = distance
            rate.sleep()

        if self.latest_odom is not None:
            self.append_trajectory_point(trajectory_points, self.latest_odom, force=True)
        self.add_trajectory_marker(
            trial_idx,
            trajectory_points,
            success=success,
            collision=collision,
        )

        return {
            "trial": trial_idx + 1,
            "start_x": start[0],
            "start_y": start[1],
            "start_z": start[2],
            "start_side": start_side,
            "goal_x": goal[0],
            "goal_y": goal[1],
            "goal_z": goal[2],
            "goal_side": goal_side,
            "success": success,
            "collision": collision,
            "collision_type": collision_type,
            "timeout": timed_out,
            "deadlock": deadlock_seen,
            "recovered_from_deadlock": recovered_from_deadlock,
            "post_deadlock_success": deadlock_seen and success,
            "deadlock_time": deadlock_time if deadlock_time is not None else "",
            "first_recovery_latency": (
                first_recovery_latency if first_recovery_latency is not None else ""
            ),
            "deadlock_events": deadlock_event_count,
            "recovered_deadlock_events": recovered_deadlock_event_count,
            "deadlock_steps": stuck_steps,
            "duration": rospy.Time.now().to_sec() - start_time,
        }

    def write_csv(self, results):
        if not self.csv_path:
            return
        csv_path = os.path.abspath(os.path.expanduser(self.csv_path))
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        rospy.loginfo("[deployment-eval] wrote %s", csv_path)

    def run(self):
        self.wait_for_odom()
        results = []
        for i in range(self.num_trials):
            result = self.run_one_trial(i)
            results.append(result)
            rospy.loginfo(
                "[deployment-eval] trial %02d/%02d | success=%s | collision=%s(%s) | timeout=%s | deadlock=%s | recovered_from_deadlock=%s | deadlock_events=%d/%d recovered | post_deadlock_success=%s | duration=%.1fs | start=(%.1f, %.1f, %.1f) side=%d | goal=(%.1f, %.1f, %.1f) side=%d",
                result["trial"],
                self.num_trials,
                result["success"],
                result["collision"],
                result["collision_type"],
                result["timeout"],
                result["deadlock"],
                result["recovered_from_deadlock"],
                result["recovered_deadlock_events"],
                result["deadlock_events"],
                result["post_deadlock_success"],
                result["duration"],
                result["start_x"],
                result["start_y"],
                result["start_z"],
                result["start_side"],
                result["goal_x"],
                result["goal_y"],
                result["goal_z"],
                result["goal_side"],
            )

        success_count = sum(1 for item in results if item["success"])
        collision_count = sum(1 for item in results if item["collision"])
        timeout_count = sum(1 for item in results if item["timeout"])
        deadlock_count = sum(1 for item in results if item["deadlock"])
        recovered_trial_count = sum(
            1 for item in results if item["recovered_from_deadlock"]
        )
        post_deadlock_success_count = sum(
            1 for item in results if item["post_deadlock_success"]
        )
        deadlock_event_count = sum(item["deadlock_events"] for item in results)
        recovered_deadlock_event_count = sum(
            item["recovered_deadlock_events"] for item in results
        )
        total_deadlock_steps = sum(item["deadlock_steps"] for item in results)
        deadlock_steps_mean_all = total_deadlock_steps / float(self.num_trials)
        deadlock_steps_mean_detected = (
            total_deadlock_steps / float(deadlock_count)
            if deadlock_count > 0
            else 0.0
        )

        success_rate = success_count / float(self.num_trials)
        failure_count = self.num_trials - success_count
        failure_rate = failure_count / float(self.num_trials)
        collision_rate = collision_count / float(self.num_trials)
        timeout_rate = timeout_count / float(self.num_trials)
        deadlock_rate = deadlock_count / float(self.num_trials)
        # Match utils.conditional_rate: no conditioned samples returns 0.0.
        deadlock_recovery_rate = (
            recovered_trial_count / float(deadlock_count)
            if deadlock_count > 0
            else 0.0
        )
        deadlock_event_recovery_rate = (
            recovered_deadlock_event_count / float(deadlock_event_count)
            if deadlock_event_count > 0
            else 0.0
        )
        post_deadlock_success_rate = (
            post_deadlock_success_count / float(deadlock_count)
            if deadlock_count > 0
            else 0.0
        )

        print("")
        print("[deployment-eval] summary")
        print(f"  trials: {self.num_trials}")
        print(f"  success_rate: {success_count}/{self.num_trials} = {success_rate:.4f}")
        print(f"  failure_rate: {failure_count}/{self.num_trials} = {failure_rate:.4f}")
        print(f"  collision_rate: {collision_count}/{self.num_trials} = {collision_rate:.4f}")
        print(f"  timeout_rate: {timeout_count}/{self.num_trials} = {timeout_rate:.4f}")
        print(f"  deadlock_rate: {deadlock_count}/{self.num_trials} = {deadlock_rate:.4f}")
        print(
            "  deadlock_recovery_rate: "
            f"{recovered_trial_count}/{deadlock_count} = "
            f"{deadlock_recovery_rate:.4f}"
        )
        print(
            "  deadlock_event_recovery_rate: "
            f"{recovered_deadlock_event_count}/{deadlock_event_count} = "
            f"{deadlock_event_recovery_rate:.4f}"
        )
        print(
            "  post_deadlock_success_rate: "
            f"{post_deadlock_success_count}/{deadlock_count} = "
            f"{post_deadlock_success_rate:.4f}"
        )
        print(f"  deadlock_steps_mean_all_trials: {deadlock_steps_mean_all:.4f}")
        print(
            "  deadlock_steps_mean_deadlocked_trials: "
            f"{deadlock_steps_mean_detected:.4f}"
        )

        self.write_csv(results)
        self.publish_all_trajectory_markers()
        self.publish_environment_snapshot()
        visualization_active = (
            self.trajectory_pub is not None or self.environment_pub is not None
        )
        if self.keep_trajectory_publisher_alive and visualization_active:
            rospy.loginfo(
                "[deployment-eval] final visualization remains available on %s and %s; "
                "select trials via %s; press Ctrl-C to exit",
                self.trajectory_topic,
                self.environment_topic,
                self.trajectory_selection_topic,
            )
            rospy.spin()


def main():
    rospy.init_node("deployment_eval", anonymous=True)
    evaluator = DeploymentEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
