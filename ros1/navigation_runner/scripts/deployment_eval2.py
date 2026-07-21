#!/usr/bin/env python3

"""Deterministic evaluator for the LP-Nav 10 m x 5 m Gazebo benchmark."""

import csv
import json
import math
import os
import random
import threading
import time

import rospy
import tf.transformations
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

try:
    import yaml
except ImportError:
    yaml = None


def load_scenario(path):
    with open(path, "r") as stream:
        text = stream.read()
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("PyYAML is required to read %s" % path) from error


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def distance_3d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance_2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class PolylineWalker:
    def __init__(self, config):
        self.name = str(config["name"])
        self.size = tuple(float(value) for value in config["size"])
        self.speed = float(config["speed"])
        self.path = [tuple(float(value) for value in point) for point in config["path"]]
        if len(self.path) < 2:
            raise ValueError("Pedestrian %s requires at least two path points" % self.name)

        self.segment_lengths = []
        self.cumulative_lengths = [0.0]
        for index in range(1, len(self.path)):
            length = distance_3d(self.path[index - 1], self.path[index])
            if length <= 1e-6:
                continue
            self.segment_lengths.append((index - 1, index, length))
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        if not self.segment_lengths:
            raise ValueError("Pedestrian %s has a zero-length path" % self.name)
        self.total_length = self.cumulative_lengths[-1]
        self.progress = 0.0
        self.position, self.yaw = self.pose_at(0.0)
        self.velocity = (0.0, 0.0, 0.0)

    def pose_at(self, progress):
        progress = progress % self.total_length
        traversed = 0.0
        for start_index, end_index, length in self.segment_lengths:
            if progress <= traversed + length or (start_index, end_index, length) == self.segment_lengths[-1]:
                ratio = clamp((progress - traversed) / length, 0.0, 1.0)
                start = self.path[start_index]
                end = self.path[end_index]
                position = (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                    start[2] + ratio * (end[2] - start[2]),
                )
                yaw = math.atan2(end[1] - start[1], end[0] - start[0])
                return position, yaw
            traversed += length
        return self.path[0], 0.0

    def reset(self, phase):
        self.progress = (phase % 1.0) * self.total_length
        self.position, self.yaw = self.pose_at(self.progress)
        self.velocity = (0.0, 0.0, 0.0)

    def candidate(self, dt):
        next_progress = (self.progress + self.speed * max(0.0, dt)) % self.total_length
        position, yaw = self.pose_at(next_progress)
        return next_progress, position, yaw

    def advance(self, next_progress, position, yaw, dt):
        previous = self.position
        self.progress = next_progress
        self.position = position
        self.yaw = yaw
        if dt > 1e-6:
            self.velocity = (
                (position[0] - previous[0]) / dt,
                (position[1] - previous[1]) / dt,
                (position[2] - previous[2]) / dt,
            )
        else:
            self.velocity = (0.0, 0.0, 0.0)

    def pause(self):
        self.velocity = (0.0, 0.0, 0.0)


class PedestrianController:
    def __init__(self, pedestrian_configs, state_publisher, state_service):
        self.walkers = [PolylineWalker(config) for config in pedestrian_configs]
        self.state_publisher = state_publisher
        self.state_service = state_service

    def reset(self, rng, forbidden_points, forbidden_clearance):
        phases = []
        for walker in self.walkers:
            selected_phase = None
            for _ in range(500):
                phase = rng.random()
                position, _ = walker.pose_at(phase * walker.total_length)
                if any(distance_2d(position, point) < forbidden_clearance for point in forbidden_points):
                    continue
                selected_phase = phase
                break
            if selected_phase is None:
                raise RuntimeError("Could not place pedestrian %s away from the trial endpoints" % walker.name)
            walker.reset(selected_phase)
            phases.append(round(selected_phase, 6))
        self.publish(use_service=True)
        return phases

    def advance(self, dt):
        if not self.walkers:
            return
        for walker in self.walkers:
            next_progress, position, yaw = walker.candidate(dt)
            walker.advance(next_progress, position, yaw, dt)
        self.publish(use_service=False)

    @staticmethod
    def model_state(walker):
        state = ModelState()
        state.model_name = walker.name
        state.reference_frame = "world"
        state.pose.position.x = walker.position[0]
        state.pose.position.y = walker.position[1]
        state.pose.position.z = walker.position[2]
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, walker.yaw)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        state.twist.linear.x = walker.velocity[0]
        state.twist.linear.y = walker.velocity[1]
        state.twist.linear.z = walker.velocity[2]
        return state

    def publish(self, use_service=False):
        for walker in self.walkers:
            state = self.model_state(walker)
            if use_service:
                try:
                    self.state_service(state)
                except rospy.ServiceException as error:
                    rospy.logwarn("[deployment-eval2] failed to reset %s: %s", walker.name, error)
            else:
                self.state_publisher.publish(state)

    def stop(self):
        for walker in self.walkers:
            walker.pause()
        self.publish(use_service=True)


class CorridorDeploymentEvaluator:
    def __init__(self):
        self.scenario_file = os.path.abspath(os.path.expanduser(rospy.get_param("~scenario_file", self.default_scenario_file())))
        self.scenario = load_scenario(self.scenario_file)

        evaluation = self.scenario["evaluation"]
        flight = self.scenario["flight"]
        arena = self.scenario["arena"]
        task = self.scenario["task"]
        self.num_trials = int(rospy.get_param("~num_trials", 100))
        self.random_seed = int(rospy.get_param("~random_seed", self.scenario["random_seed"] + 1000))
        self.model_name = str(rospy.get_param("~model_name", "quadcopter"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/CERLAB/quadcopter/odom"))
        self.csv_path = str(rospy.get_param("~csv_path", ""))

        self.eval_rate_hz = float(rospy.get_param("~eval_rate_hz", evaluation["rate_hz"]))
        self.timeout = float(rospy.get_param("~timeout", evaluation["timeout"]))
        self.success_radius = float(rospy.get_param("~success_radius", evaluation["success_radius"]))
        self.collision_radius = float(rospy.get_param("~collision_radius", evaluation["collision_radius"]))
        self.deadlock_duration = float(rospy.get_param("~deadlock_duration", evaluation["deadlock_duration"]))
        self.stuck_window = max(1, int(math.ceil(self.deadlock_duration * self.eval_rate_hz)))
        self.progress_epsilon = float(rospy.get_param("~progress_epsilon", evaluation["progress_epsilon"]))
        self.front_distance = float(rospy.get_param("~front_obstacle_distance", evaluation["front_obstacle_distance"]))
        self.front_tan = math.tan(math.radians(float(rospy.get_param("~front_obstacle_angle_deg", evaluation["front_obstacle_angle_deg"]))))
        self.front_height = float(rospy.get_param("~front_obstacle_height", evaluation["front_obstacle_height"]))
        self.reset_wait = float(rospy.get_param("~reset_wait", evaluation["reset_wait"]))
        self.goal_publish_time = float(rospy.get_param("~goal_publish_time", 1.0))
        self.reset_position_tolerance = float(rospy.get_param("~reset_position_tolerance", 0.15))
        self.reset_speed_tolerance = float(rospy.get_param("~reset_speed_tolerance", 0.10))
        self.reset_stable_time = float(rospy.get_param("~reset_stable_time", 0.30))
        self.dependency_timeout = float(rospy.get_param("~dependency_timeout", 60.0))
        self.require_dynamic_detector = bool(rospy.get_param("~require_dynamic_detector", True))

        self.min_altitude = float(rospy.get_param("~min_altitude", flight["min_altitude"]))
        self.max_altitude = float(rospy.get_param("~max_altitude", flight["max_altitude"]))
        self.height_range = tuple(float(value) for value in flight["start_goal_height_range"])
        self.interior_x = tuple(float(value) for value in arena["interior_bounds"]["x"])
        self.interior_y = tuple(float(value) for value in arena["interior_bounds"]["y"])
        self.left_region = task["left_region"]
        self.right_region = task["right_region"]
        legacy_spawn_clearance = float(task.get("spawn_clearance", 0.50))
        self.static_spawn_clearance = float(task.get("static_spawn_clearance", legacy_spawn_clearance))
        self.pedestrian_spawn_clearance = float(task.get("pedestrian_spawn_clearance", legacy_spawn_clearance))
        self.alternate_direction = bool(task.get("alternate_direction", True))
        self.static_geometry = list(self.scenario["walls"]) + list(self.scenario["static_obstacles"])

        self.lock = threading.Lock()
        self.latest_odom = None
        self.model_names = set()
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=20)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.model_states_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback, queue_size=1)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.pedestrians = PedestrianController(
            self.scenario["pedestrians"],
            self.model_state_pub,
            self.set_model_state,
        )

    @staticmethod
    def default_scenario_file():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "uav_simulator",
                "worlds",
                "lpnav_corridor",
                "lpnav_corridor_seed_7.yaml",
            )
        )

    def odom_callback(self, message):
        with self.lock:
            self.latest_odom = message

    def model_states_callback(self, message):
        with self.lock:
            self.model_names = set(message.name)

    def current_odom(self):
        with self.lock:
            return self.latest_odom

    @staticmethod
    def pose_message(point, yaw=0.0):
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = rospy.Time.now()
        message.pose.position.x = point[0]
        message.pose.position.y = point[1]
        message.pose.position.z = point[2]
        quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        message.pose.orientation.x = quaternion[0]
        message.pose.orientation.y = quaternion[1]
        message.pose.orientation.z = quaternion[2]
        message.pose.orientation.w = quaternion[3]
        return message

    @staticmethod
    def odom_point(odom):
        position = odom.pose.pose.position
        return (position.x, position.y, position.z)

    @staticmethod
    def odom_speed(odom):
        velocity = odom.twist.twist.linear
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def point_box_signed_clearance(self, point, box):
        center = box["center"]
        size = box["size"]
        dx = abs(point[0] - center[0]) - size[0] * 0.5
        dy = abs(point[1] - center[1]) - size[1] * 0.5
        outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
        inside = min(max(dx, dy), 0.0)
        return outside + inside - self.collision_radius

    def point_collides_box(self, point, box):
        center = box["center"]
        size = box["size"]
        if abs(point[2] - center[2]) > size[2] * 0.5 + self.collision_radius:
            return False
        nearest_x = clamp(point[0], center[0] - size[0] * 0.5, center[0] + size[0] * 0.5)
        nearest_y = clamp(point[1], center[1] - size[1] * 0.5, center[1] + size[1] * 0.5)
        return math.hypot(point[0] - nearest_x, point[1] - nearest_y) <= self.collision_radius

    def static_collision_and_clearance(self, point):
        clearances = [self.point_box_signed_clearance(point, box) for box in self.static_geometry]
        collision = any(self.point_collides_box(point, box) for box in self.static_geometry)
        return collision, min(clearances) if clearances else float("inf")

    def dynamic_collision_and_clearance(self, point):
        collision = False
        minimum = float("inf")
        for walker in self.pedestrians.walkers:
            obstacle_center_z = walker.position[2] + walker.size[2] * 0.5
            vertical_overlap = abs(point[2] - obstacle_center_z) <= walker.size[2] * 0.5 + self.collision_radius
            horizontal_clearance = distance_2d(point, walker.position) - max(walker.size[0], walker.size[1]) * 0.5 - self.collision_radius
            if vertical_overlap:
                minimum = min(minimum, horizontal_clearance)
                collision = collision or horizontal_clearance <= 0.0
        return collision, minimum

    def front_static_obstacle(self, point, goal):
        goal_dx = goal[0] - point[0]
        goal_dy = goal[1] - point[1]
        goal_norm = math.hypot(goal_dx, goal_dy)
        if goal_norm <= 1e-6:
            return False
        forward_x, forward_y = goal_dx / goal_norm, goal_dy / goal_norm
        lateral_x, lateral_y = -forward_y, forward_x
        for box in self.static_geometry:
            center = box["center"]
            size = box["size"]
            if abs(point[2] - center[2]) > size[2] * 0.5 + self.front_height:
                continue
            nearest_x = clamp(point[0], center[0] - size[0] * 0.5, center[0] + size[0] * 0.5)
            nearest_y = clamp(point[1], center[1] - size[1] * 0.5, center[1] + size[1] * 0.5)
            rel_x, rel_y = nearest_x - point[0], nearest_y - point[1]
            forward = rel_x * forward_x + rel_y * forward_y
            lateral = abs(rel_x * lateral_x + rel_y * lateral_y)
            if forward > 0.0 and math.hypot(rel_x, rel_y) <= self.front_distance and lateral <= self.front_tan * forward:
                return True
        return False

    def point_is_spawn_safe(self, point):
        for box in self.static_geometry:
            if self.point_box_signed_clearance(point, box) < self.static_spawn_clearance:
                return False
        return True

    def sample_region(self, region, rng):
        for _ in range(500):
            point = (
                rng.uniform(float(region["x"][0]), float(region["x"][1])),
                rng.uniform(float(region["y"][0]), float(region["y"][1])),
                rng.uniform(self.height_range[0], self.height_range[1]),
            )
            if self.point_is_spawn_safe(point):
                return point
        raise RuntimeError("Could not sample a collision-free endpoint from %s" % region)

    def sample_task(self, trial_index, rng):
        if self.alternate_direction:
            left_to_right = trial_index % 2 == 1
        else:
            left_to_right = rng.random() < 0.5
        start_region = self.left_region if left_to_right else self.right_region
        goal_region = self.right_region if left_to_right else self.left_region
        start = self.sample_region(start_region, rng)
        goal = self.sample_region(goal_region, rng)
        return start, goal, "left_to_right" if left_to_right else "right_to_left"

    def wait_for_dependencies(self):
        rospy.loginfo("[deployment-eval2] scenario: %s", self.scenario_file)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=self.dependency_timeout)
        rospy.wait_for_service("/occupancy_map/raycast", timeout=self.dependency_timeout)
        if self.require_dynamic_detector:
            rospy.wait_for_service("/onboard_detector/get_dynamic_obstacles", timeout=self.dependency_timeout)

        deadline = time.time() + self.dependency_timeout
        required_models = {self.model_name} | {walker.name for walker in self.pedestrians.walkers}
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                odom_ready = self.latest_odom is not None
                models_ready = required_models.issubset(self.model_names)
            goal_ready = self.goal_pub.get_num_connections() > 0
            if odom_ready and models_ready and goal_ready:
                return
            rospy.sleep(0.1)
        missing = required_models - self.model_names
        raise RuntimeError(
            "Benchmark dependencies were not ready: odom=%s goal_subscriber=%s missing_models=%s"
            % (self.latest_odom is not None, self.goal_pub.get_num_connections() > 0, sorted(missing))
        )

    def reset_robot(self, start, goal):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        state.pose = self.pose_message(start, yaw).pose

        rate = rospy.Rate(20.0)
        deadline = time.time() + self.reset_wait
        stable_since = None
        while not rospy.is_shutdown() and time.time() < deadline:
            self.set_model_state(state)
            self.pedestrians.publish(use_service=False)
            odom = self.current_odom()
            if odom is not None:
                position_error = distance_3d(self.odom_point(odom), start)
                if position_error <= self.reset_position_tolerance and self.odom_speed(odom) <= self.reset_speed_tolerance:
                    if stable_since is None:
                        stable_since = time.time()
                    elif time.time() - stable_since >= self.reset_stable_time:
                        return True
                else:
                    stable_since = None
            rate.sleep()
        rospy.logwarn("[deployment-eval2] UAV reset did not settle within %.1f s", self.reset_wait)
        return False

    def publish_goal(self, goal):
        message = self.pose_message(goal)
        rate = rospy.Rate(10.0)
        deadline = time.time() + self.goal_publish_time
        while not rospy.is_shutdown() and time.time() < deadline:
            message.header.stamp = rospy.Time.now()
            self.goal_pub.publish(message)
            self.pedestrians.publish(use_service=False)
            rate.sleep()

    def run_one_trial(self, trial_index):
        trial_seed = self.random_seed + trial_index * 100003
        rng = random.Random(trial_seed)
        start, goal, direction = self.sample_task(trial_index, rng)
        phases = self.pedestrians.reset(rng, [start, goal], self.pedestrian_spawn_clearance)
        reset_stable = self.reset_robot(start, goal)
        self.publish_goal(goal)

        rate = rospy.Rate(self.eval_rate_hz)
        start_time = rospy.Time.now().to_sec()
        previous_time = start_time
        previous_position = None
        previous_goal_distance = None
        stuck_counter = 0
        deadlock_seen = False
        deadlock_active_previous = False
        deadlock_recovered = False
        deadlock_time = None
        deadlock_steps = 0
        path_length = 0.0
        speed_sum = 0.0
        speed_samples = 0
        min_static_clearance = float("inf")
        min_dynamic_clearance = float("inf")
        success = False
        collision = False
        collision_type = ""
        timed_out = False
        altitude_violation = False
        boundary_violation = False

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            dt = clamp(now - previous_time, 0.0, 0.25)
            previous_time = now
            self.pedestrians.advance(dt)

            odom = self.current_odom()
            if odom is None:
                rate.sleep()
                continue
            point = self.odom_point(odom)
            elapsed = now - start_time
            goal_distance = distance_3d(point, goal)
            speed = self.odom_speed(odom)
            speed_sum += speed
            speed_samples += 1
            if previous_position is not None:
                path_length += distance_3d(point, previous_position)
            previous_position = point

            static_collision, static_clearance = self.static_collision_and_clearance(point)
            dynamic_collision, dynamic_clearance = self.dynamic_collision_and_clearance(point)
            min_static_clearance = min(min_static_clearance, static_clearance)
            min_dynamic_clearance = min(min_dynamic_clearance, dynamic_clearance)

            if static_collision or dynamic_collision:
                collision = True
                if static_collision and dynamic_collision:
                    collision_type = "static+dynamic"
                elif static_collision:
                    collision_type = "static"
                else:
                    collision_type = "dynamic"
                break

            if point[2] < self.min_altitude or point[2] > self.max_altitude:
                altitude_violation = True
                break
            if not (self.interior_x[0] <= point[0] <= self.interior_x[1] and self.interior_y[0] <= point[1] <= self.interior_y[1]):
                boundary_violation = True
                break
            if goal_distance <= self.success_radius:
                success = True
                break

            if previous_goal_distance is None:
                goal_progress = 0.0
            else:
                goal_progress = previous_goal_distance - goal_distance
            front_obstacle = self.front_static_obstacle(point, goal)
            if goal_progress <= self.progress_epsilon and front_obstacle:
                stuck_counter += 1
            else:
                stuck_counter = 0
            stuck_active = stuck_counter >= self.stuck_window
            if stuck_active:
                deadlock_steps += 1
                if not deadlock_seen:
                    deadlock_seen = True
                    deadlock_time = elapsed
            if deadlock_seen and deadlock_active_previous and not stuck_active:
                deadlock_recovered = True
            deadlock_active_previous = stuck_active
            previous_goal_distance = goal_distance

            if elapsed >= self.timeout:
                timed_out = True
                break
            rate.sleep()

        duration = max(0.0, rospy.Time.now().to_sec() - start_time)
        outcome = "success"
        if not success:
            if collision:
                outcome = "collision"
            elif altitude_violation:
                outcome = "altitude_violation"
            elif boundary_violation:
                outcome = "boundary_violation"
            elif timed_out:
                outcome = "timeout"
            else:
                outcome = "interrupted"
        return {
            "trial": trial_index,
            "trial_seed": trial_seed,
            "scenario": self.scenario["name"],
            "direction": direction,
            "start_x": start[0],
            "start_y": start[1],
            "start_z": start[2],
            "goal_x": goal[0],
            "goal_y": goal[1],
            "goal_z": goal[2],
            "pedestrian_phases": json.dumps(phases, separators=(",", ":")),
            "reset_stable": reset_stable,
            "outcome": outcome,
            "success": success,
            "collision": collision,
            "collision_type": collision_type,
            "timeout": timed_out,
            "altitude_violation": altitude_violation,
            "boundary_violation": boundary_violation,
            "deadlock": deadlock_seen,
            "deadlock_recovered": deadlock_recovered,
            "escaped_after_deadlock": deadlock_seen and success,
            "deadlock_time": deadlock_time if deadlock_time is not None else "",
            "deadlock_steps": deadlock_steps,
            "duration": duration,
            "path_length": path_length,
            "mean_speed": speed_sum / speed_samples if speed_samples else 0.0,
            "min_static_clearance": min_static_clearance,
            "min_dynamic_clearance": min_dynamic_clearance,
        }

    def write_csv(self, results):
        if not self.csv_path or not results:
            return
        csv_path = os.path.abspath(os.path.expanduser(self.csv_path))
        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(csv_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        rospy.loginfo("[deployment-eval2] wrote %s", csv_path)

    @staticmethod
    def rate(count, denominator):
        return float(count) / denominator if denominator else 0.0

    def print_summary(self, results):
        total = len(results)
        success_count = sum(bool(result["success"]) for result in results)
        collision_count = sum(bool(result["collision"]) for result in results)
        static_collision_count = sum("static" in result["collision_type"] for result in results)
        dynamic_collision_count = sum("dynamic" in result["collision_type"] for result in results)
        deadlock_count = sum(bool(result["deadlock"]) for result in results)
        recovered_count = sum(bool(result["deadlock_recovered"]) for result in results)
        escaped_count = sum(bool(result["escaped_after_deadlock"]) for result in results)
        timeout_count = sum(bool(result["timeout"]) for result in results)
        altitude_count = sum(bool(result["altitude_violation"]) for result in results)
        successful = [result for result in results if result["success"]]

        print("")
        print("[deployment-eval2] summary")
        print("  scenario: %s" % self.scenario["name"])
        print("  trials: %d" % total)
        print("  success_rate: %.4f" % self.rate(success_count, total))
        print("  collision_rate: %.4f" % self.rate(collision_count, total))
        print("  static_collision_rate: %.4f" % self.rate(static_collision_count, total))
        print("  dynamic_collision_rate: %.4f" % self.rate(dynamic_collision_count, total))
        print("  deadlock_rate: %.4f" % self.rate(deadlock_count, total))
        print("  deadlock_recovery_rate: %.4f" % self.rate(recovered_count, deadlock_count))
        print("  escape_after_deadlock_rate: %.4f" % self.rate(escaped_count, deadlock_count))
        print("  timeout_rate: %.4f" % self.rate(timeout_count, total))
        print("  altitude_violation_rate: %.4f" % self.rate(altitude_count, total))
        if successful:
            print("  mean_success_duration: %.4f" % (sum(result["duration"] for result in successful) / len(successful)))
            print("  mean_success_path_length: %.4f" % (sum(result["path_length"] for result in successful) / len(successful)))

    def run(self):
        self.wait_for_dependencies()
        rospy.loginfo(
            "[deployment-eval2] starting %d trials; timeout=%.1fs deadlock_window=%d steps max_altitude=%.2fm",
            self.num_trials,
            self.timeout,
            self.stuck_window,
            self.max_altitude,
        )
        results = []
        try:
            for trial_index in range(1, self.num_trials + 1):
                result = self.run_one_trial(trial_index)
                results.append(result)
                rospy.loginfo(
                    "[deployment-eval2] %d/%d outcome=%s deadlock=%s escaped=%s duration=%.2fs path=%.2fm",
                    trial_index,
                    self.num_trials,
                    result["outcome"],
                    result["deadlock"],
                    result["escaped_after_deadlock"],
                    result["duration"],
                    result["path_length"],
                )
        finally:
            self.pedestrians.stop()
        self.write_csv(results)
        self.print_summary(results)


def main():
    rospy.init_node("deployment_eval2", anonymous=False)
    evaluator = CorridorDeploymentEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
