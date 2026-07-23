#!/usr/bin/env python3

import csv
import math
import os
import random
import time

import rospy
import tf.transformations
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Point, PoseStamped
from map_manager.srv import RayCast
from nav_msgs.msg import Odometry
from onboard_detector.srv import GetDynamicObstacles


class DeploymentEvaluator:
    def __init__(self):
        self.num_trials = int(rospy.get_param("~num_trials", 1000))
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

        self.latest_odom = None
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.odom_sub = rospy.Subscriber("/CERLAB/quadcopter/odom", Odometry, self.odom_callback)

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

    def publish_goal_for_a_moment(self, goal):
        goal_msg = self.make_pose_msg(goal[0], goal[1], goal[2], 0.0)
        rate = rospy.Rate(10)
        end_time = time.time() + self.goal_publish_time
        while not rospy.is_shutdown() and time.time() < end_time:
            goal_msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal_msg)
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

    def distance_to_point(self, odom, point):
        pos = odom.pose.pose.position
        dx = point[0] - pos.x
        dy = point[1] - pos.y
        dz = point[2] - pos.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def speed(self, odom):
        vel = odom.twist.twist.linear
        return math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)

    def get_static_front_obstacle(self, odom, goal):
        pos = odom.pose.pose.position
        goal_dx = goal[0] - pos.x
        goal_dy = goal[1] - pos.y
        goal_norm = math.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)
        if goal_norm <= 1e-6:
            return False
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
            return False

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
                return True
        return False

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
        # Publish the goal only after the reset pose has settled, so the
        # navigation node does not act on a new task while Gazebo is still being
        # forced to the start state.
        self.publish_goal_for_a_moment(goal)

        start_time = rospy.Time.now().to_sec()
        prev_goal_distance = None
        stuck_counter = 0
        stuck_steps = 0
        deadlock_seen = False
        deadlock_time = None
        success = False
        collision = False
        collision_type = ""

        rate = rospy.Rate(self.eval_rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            elapsed = now - start_time
            odom = self.latest_odom
            if odom is None:
                rate.sleep()
                continue

            distance = self.distance_to_goal(odom, goal)
            if prev_goal_distance is None:
                prev_goal_distance = distance

            goal_progress = prev_goal_distance - distance
            front_obstacle = self.get_static_front_obstacle(odom, goal)
            small_progress_with_obstacle = goal_progress <= self.stuck_progress_eps and front_obstacle
            if small_progress_with_obstacle:
                stuck_counter += 1
            else:
                stuck_counter = 0

            stuck_active = stuck_counter >= self.stuck_window
            if stuck_active:
                stuck_steps += 1
                if not deadlock_seen:
                    deadlock_seen = True
                    deadlock_time = elapsed

            collision, collision_type = self.get_collision(odom)
            if collision:
                break

            if distance <= self.success_radius:
                success = True
                break

            if elapsed >= self.timeout:
                break

            prev_goal_distance = distance
            rate.sleep()

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
            "deadlock": deadlock_seen,
            "escaped_after_deadlock": deadlock_seen and success,
            "deadlock_time": deadlock_time if deadlock_time is not None else "",
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
                "[deployment-eval] trial %02d/%02d | success=%s | collision=%s(%s) | deadlock=%s | escaped_after_deadlock=%s | duration=%.1fs | start=(%.1f, %.1f, %.1f) side=%d | goal=(%.1f, %.1f, %.1f) side=%d",
                result["trial"],
                self.num_trials,
                result["success"],
                result["collision"],
                result["collision_type"],
                result["deadlock"],
                result["escaped_after_deadlock"],
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
        deadlock_count = sum(1 for item in results if item["deadlock"])
        escaped_count = sum(1 for item in results if item["escaped_after_deadlock"])
        deadlock_steps_mean = sum(item["deadlock_steps"] for item in results) / float(self.num_trials)

        success_rate = success_count / float(self.num_trials)
        failure_count = self.num_trials - success_count
        failure_rate = failure_count / float(self.num_trials)
        collision_rate = collision_count / float(self.num_trials)
        deadlock_rate = deadlock_count / float(self.num_trials)
        # Match utils.conditional_rate: no conditioned samples returns 0.0.
        escape_rate = escaped_count / float(deadlock_count) if deadlock_count > 0 else 0.0

        print("")
        print("[deployment-eval] summary")
        print(f"  trials: {self.num_trials}")
        print(f"  success_rate: {success_count}/{self.num_trials} = {success_rate:.4f}")
        print(f"  failure_rate: {failure_count}/{self.num_trials} = {failure_rate:.4f}")
        print(f"  collision_rate: {collision_count}/{self.num_trials} = {collision_rate:.4f}")
        print(f"  deadlock_rate: {deadlock_count}/{self.num_trials} = {deadlock_rate:.4f}")
        print(f"  escape_after_deadlock_rate: {escaped_count}/{deadlock_count} = {escape_rate:.4f}")
        print(f"  deadlock_steps_mean: {deadlock_steps_mean:.4f}")

        self.write_csv(results)


def main():
    rospy.init_node("deployment_eval", anonymous=True)
    evaluator = DeploymentEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()
