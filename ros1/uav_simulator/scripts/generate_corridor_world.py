#!/usr/bin/env python3

"""Generate a deterministic 50 m x 5 m Gazebo navigation benchmark."""

import argparse
import heapq
import json
import math
import os
import random
from pathlib import Path

try:
    import yaml
except ImportError:  # ROS normally provides python3-yaml; JSON remains a portable fallback.
    yaml = None


def load_structured_file(path):
    text = Path(path).read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("PyYAML is required to read %s" % path) from error


def dump_structured_data(data):
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)
    return json.dumps(data, indent=2) + "\n"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def pairwise_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_length(points):
    return sum(pairwise_distance(points[i - 1], points[i]) for i in range(1, len(points)))


def ranges_overlap(a_min, a_max, b_min, b_max, margin=0.0):
    return not (a_max + margin <= b_min or b_max + margin <= a_min)


class OccupancyGrid:
    NEIGHBORS = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, bounds, resolution, obstacles, clearance):
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.resolution = resolution
        self.nx = int(math.floor((self.x_max - self.x_min) / resolution)) + 1
        self.ny = int(math.floor((self.y_max - self.y_min) / resolution)) + 1
        self.obstacles = obstacles
        self.clearance = clearance
        self.free_cells = [
            (ix, iy)
            for ix in range(self.nx)
            for iy in range(self.ny)
            if self.is_free((ix, iy))
        ]

    def point(self, cell):
        return (
            self.x_min + cell[0] * self.resolution,
            self.y_min + cell[1] * self.resolution,
        )

    def cell(self, point):
        ix = int(round((point[0] - self.x_min) / self.resolution))
        iy = int(round((point[1] - self.y_min) / self.resolution))
        return (clamp(ix, 0, self.nx - 1), clamp(iy, 0, self.ny - 1))

    def is_free(self, cell):
        ix, iy = cell
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return False
        x, y = self.point(cell)
        for obstacle in self.obstacles:
            half_x = obstacle["size"][0] * 0.5 + self.clearance
            half_y = obstacle["size"][1] * 0.5 + self.clearance
            if abs(x - obstacle["center"][0]) <= half_x and abs(y - obstacle["center"][1]) <= half_y:
                return False
        return True

    def astar(self, start_point, goal_point):
        start = self.cell(start_point)
        goal = self.cell(goal_point)
        if not self.is_free(start) or not self.is_free(goal):
            return None

        frontier = [(0.0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy, step_cost in self.NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_free(neighbor):
                    continue
                if dx and dy:
                    if not self.is_free((current[0] + dx, current[1])):
                        continue
                    if not self.is_free((current[0], current[1] + dy)):
                        continue
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
                    heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return None
        cells = []
        current = goal
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        return [self.point(cell) for cell in cells]

    def line_is_free(self, start, goal):
        distance = pairwise_distance(start, goal)
        samples = max(1, int(math.ceil(distance / (self.resolution * 0.5))))
        for i in range(samples + 1):
            ratio = float(i) / samples
            point = (
                start[0] + ratio * (goal[0] - start[0]),
                start[1] + ratio * (goal[1] - start[1]),
            )
            if not self.is_free(self.cell(point)):
                return False
        return True

    def simplify(self, path):
        if not path or len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1 and not self.line_is_free(path[anchor], path[candidate]):
                candidate -= 1
            simplified.append(path[candidate])
            anchor = candidate
        return simplified


class CorridorWorldGenerator:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.package_root = self.config_path.parent.parent
        self.seed = int(config["random_seed"])
        self.rng = random.Random(self.seed)

        arena = config["arena"]
        self.length = float(arena["length"])
        self.width = float(arena["width"])
        self.wall_thickness = float(arena["wall_thickness"])
        self.wall_height = float(arena["wall_height"])
        self.interior_bounds = (
            -self.length * 0.5,
            self.length * 0.5,
            -self.width * 0.5,
            self.width * 0.5,
        )
        self.validate_non_overflyable_geometry()

    def validate_non_overflyable_geometry(self):
        max_altitude = float(self.config["flight"]["max_altitude"])
        collision_radius = float(self.config["evaluation"]["collision_radius"])
        required_height = max_altitude + collision_radius
        minimum_prism_height = float(self.config["static_obstacles"]["height_range"][0])
        pedestrian_height = float(self.config["pedestrians"]["size"][2])
        if min(self.wall_height, minimum_prism_height, pedestrian_height) < required_height:
            raise ValueError(
                "All obstacle heights must reach max_altitude + collision_radius (%.3f m)" % required_height
            )

    def make_walls(self):
        half_l = self.length * 0.5
        half_w = self.width * 0.5
        thickness = self.wall_thickness
        height = self.wall_height
        return [
            self.box_record("wall_left", -half_l - thickness * 0.5, 0.0, thickness, self.width + 2.0 * thickness, height, "wall"),
            self.box_record("wall_right", half_l + thickness * 0.5, 0.0, thickness, self.width + 2.0 * thickness, height, "wall"),
            self.box_record("wall_bottom", 0.0, -half_w - thickness * 0.5, self.length, thickness, height, "wall"),
            self.box_record("wall_top", 0.0, half_w + thickness * 0.5, self.length, thickness, height, "wall"),
        ]

    @staticmethod
    def box_record(name, x, y, size_x, size_y, height, kind="prism"):
        return {
            "name": name,
            "kind": kind,
            "center": [round(x, 4), round(y, 4), round(height * 0.5, 4)],
            "size": [round(size_x, 4), round(size_y, 4), round(height, 4)],
        }

    def obstacle_valid(self, candidate, obstacles):
        cfg = self.config["static_obstacles"]
        separation = float(cfg["separation"])
        cx, cy, _ = candidate["center"]
        side = candidate["size"][0]
        c_min_x, c_max_x = cx - side * 0.5, cx + side * 0.5
        c_min_y, c_max_y = cy - side * 0.5, cy + side * 0.5

        left = self.config["task"]["left_region"]
        right = self.config["task"]["right_region"]
        spawn_margin = float(cfg["spawn_region_clearance"])
        for region in (left, right):
            if ranges_overlap(c_min_x, c_max_x, region["x"][0], region["x"][1], spawn_margin) and ranges_overlap(
                c_min_y, c_max_y, region["y"][0], region["y"][1], spawn_margin
            ):
                return False

        for obstacle in obstacles:
            ox, oy, _ = obstacle["center"]
            osx, osy, _ = obstacle["size"]
            if ranges_overlap(c_min_x, c_max_x, ox - osx * 0.5, ox + osx * 0.5, separation) and ranges_overlap(
                c_min_y, c_max_y, oy - osy * 0.5, oy + osy * 0.5, separation
            ):
                return False
        return True

    def generate_obstacle_layout(self):
        cfg = self.config["static_obstacles"]
        count = int(cfg["count"])
        edge_margin = float(cfg["edge_margin"])
        side_min, side_max = map(float, cfg["side_range"])
        height_min, height_max = map(float, cfg["height_range"])
        placement_attempts = int(cfg["placement_attempts"])
        max_layout_attempts = int(cfg["max_layout_attempts"])
        uav_clearance = float(self.config["planning"]["uav_clearance"])
        resolution = float(self.config["planning"]["grid_resolution"])
        left = self.config["task"]["left_region"]
        right = self.config["task"]["right_region"]
        start = ((left["x"][0] + left["x"][1]) * 0.5, 0.0)
        goal = ((right["x"][0] + right["x"][1]) * 0.5, 0.0)

        for _ in range(max_layout_attempts):
            obstacles = []
            for index in range(count):
                placed = False
                for _ in range(placement_attempts):
                    side = self.rng.uniform(side_min, side_max)
                    height = self.rng.uniform(height_min, height_max)
                    half_side = side * 0.5
                    x = self.rng.uniform(-self.length * 0.5 + edge_margin + half_side, self.length * 0.5 - edge_margin - half_side)
                    y = self.rng.uniform(-self.width * 0.5 + edge_margin + half_side, self.width * 0.5 - edge_margin - half_side)
                    candidate = self.box_record("static_prism_%02d" % index, x, y, side, side, height)
                    if self.obstacle_valid(candidate, obstacles):
                        obstacles.append(candidate)
                        placed = True
                        break
                if not placed:
                    break
            if len(obstacles) != count:
                continue

            bounds = (
                -self.length * 0.5 + uav_clearance,
                self.length * 0.5 - uav_clearance,
                -self.width * 0.5 + uav_clearance,
                self.width * 0.5 - uav_clearance,
            )
            grid = OccupancyGrid(bounds, resolution, obstacles, uav_clearance)
            if grid.astar(start, goal):
                return obstacles
        raise RuntimeError("Could not generate a connected static-obstacle layout")

    def generate_pedestrians(self):
        cfg = self.config["pedestrians"]
        count = int(cfg["count"])
        size = [float(value) for value in cfg["size"]]
        wall_clearance = float(cfg["wall_clearance"])
        body_radius = max(size[0], size[1]) * 0.5
        center_margin = body_radius + wall_clearance
        bounds = (
            -self.length * 0.5 + center_margin,
            self.length * 0.5 - center_margin,
            -self.width * 0.5 + center_margin,
            self.width * 0.5 - center_margin,
        )
        speed_min, speed_max = map(float, cfg["speed_range"])
        target_count = int(cfg["random_targets_per_path"])
        min_path_length = float(cfg["min_path_length"])
        min_leg_length = float(cfg["min_leg_length"])
        path_attempts = int(cfg["path_attempts"])

        pedestrians = []
        for index in range(count):
            accepted = None
            for _ in range(path_attempts):
                path = [
                    (
                        self.rng.uniform(bounds[0], bounds[1]),
                        self.rng.uniform(bounds[2], bounds[3]),
                    )
                ]
                for _ in range(target_count):
                    for _ in range(100):
                        candidate = (
                            self.rng.uniform(bounds[0], bounds[1]),
                            self.rng.uniform(bounds[2], bounds[3]),
                        )
                        if pairwise_distance(path[-1], candidate) >= min_leg_length:
                            path.append(candidate)
                            break
                    else:
                        break
                if len(path) != target_count + 1:
                    continue
                if pairwise_distance(path[-1], path[0]) < min_leg_length:
                    continue
                path.append(path[0])
                if polyline_length(path) < min_path_length:
                    continue
                accepted = path
                break
            if accepted is None:
                raise RuntimeError("Could not generate pedestrian path %d" % (index + 1))

            name = "person%d_%.1f_%.1f_%.1f" % (index + 1, size[0], size[1], size[2])
            pedestrians.append(
                {
                    "name": name,
                    "size": [round(value, 3) for value in size],
                    "speed": round(self.rng.uniform(speed_min, speed_max), 3),
                    "path": [[round(point[0], 3), round(point[1], 3), 0.0] for point in accepted],
                }
            )
        return pedestrians

    def scenario(self, walls, obstacles, pedestrians):
        flight = self.config["flight"]
        return {
            "version": 1,
            "name": self.config["output"]["basename"],
            "random_seed": self.seed,
            "arena": {
                "length": self.length,
                "width": self.width,
                "interior_bounds": {
                    "x": [-self.length * 0.5, self.length * 0.5],
                    "y": [-self.width * 0.5, self.width * 0.5],
                },
                "wall_thickness": self.wall_thickness,
                "wall_height": self.wall_height,
            },
            "flight": {
                "min_altitude": float(flight["min_altitude"]),
                "max_altitude": float(flight["max_altitude"]),
                "start_goal_height_range": [float(value) for value in flight["start_goal_height_range"]],
                "overflight_is_failure": True,
            },
            "task": self.config["task"],
            "evaluation": self.config["evaluation"],
            "walls": walls,
            "static_obstacles": obstacles,
            "pedestrians": pedestrians,
        }

    @staticmethod
    def box_sdf(box, color):
        cx, cy, cz = box["center"]
        sx, sy, sz = box["size"]
        return """
    <model name="{name}">
      <static>true</static>
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material>
            <ambient>{color}</ambient>
            <diffuse>{color}</diffuse>
          </material>
        </visual>
      </link>
    </model>""".format(name=box["name"], cx=cx, cy=cy, cz=cz, sx=sx, sy=sy, sz=sz, color=color)

    @staticmethod
    def pedestrian_sdf(pedestrian):
        x, y, _ = pedestrian["path"][0]
        radius = max(pedestrian["size"][0], pedestrian["size"][1]) * 0.5
        height = pedestrian["size"][2]
        return """
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} 0 0 0 0</pose>
      <link name="link">
        <collision name="body">
          <pose>0 0 {half_height} 0 0 0</pose>
          <geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <pose>0 0 -0.02 0 0 1.570796</pose>
          <geometry><mesh><uri>model://person/meshes/walking.dae</uri></mesh></geometry>
        </visual>
      </link>
    </model>""".format(
            name=pedestrian["name"],
            x=x,
            y=y,
            half_height=height * 0.5,
            radius=radius,
            height=height,
        )

    def world_text(self, walls, obstacles, pedestrians):
        models = []
        for wall in walls:
            models.append(self.box_sdf(wall, "0.55 0.75 0.90 1"))
        palette = (
            "0.25 0.55 0.75 1",
            "0.75 0.45 0.20 1",
            "0.35 0.65 0.35 1",
            "0.65 0.35 0.55 1",
        )
        for index, obstacle in enumerate(obstacles):
            models.append(self.box_sdf(obstacle, palette[index % len(palette)]))
        models.extend(self.pedestrian_sdf(pedestrian) for pedestrian in pedestrians)
        ground_size_x = self.length + 4.0
        ground_size_y = self.width + 4.0
        return """<?xml version="1.0"?>
<sdf version="1.7">
  <world name="default">
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.80 0.86 0.92 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>
    <include><uri>model://sun</uri></include>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>{ground_size_x} {ground_size_y}</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>{ground_size_x} {ground_size_y}</size></plane></geometry>
          <material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material>
        </visual>
      </link>
    </model>
{models}
  </world>
</sdf>
""".format(
            ground_size_x=ground_size_x,
            ground_size_y=ground_size_y,
            models="\n".join(models),
        )

    @staticmethod
    def axis_samples(center, size, resolution):
        minimum = center - size * 0.5
        count = max(1, int(math.ceil(size / resolution)))
        return [minimum + size * float(index) / count for index in range(count + 1)]

    def sample_box_surface(self, box, resolution):
        cx, cy, cz = box["center"]
        sx, sy, sz = box["size"]
        xs = self.axis_samples(cx, sx, resolution)
        ys = self.axis_samples(cy, sy, resolution)
        zs = self.axis_samples(cz, sz, resolution)
        x_min, x_max = xs[0], xs[-1]
        y_min, y_max = ys[0], ys[-1]
        z_min, z_max = zs[0], zs[-1]
        points = set()
        for x in xs:
            for y in ys:
                points.add((round(x, 4), round(y, 4), round(z_min, 4)))
                points.add((round(x, 4), round(y, 4), round(z_max, 4)))
        for x in xs:
            for z in zs:
                points.add((round(x, 4), round(y_min, 4), round(z, 4)))
                points.add((round(x, 4), round(y_max, 4), round(z, 4)))
        for y in ys:
            for z in zs:
                points.add((round(x_min, 4), round(y, 4), round(z, 4)))
                points.add((round(x_max, 4), round(y, 4), round(z, 4)))
        return points

    def pcd_text(self, static_geometry):
        resolution = float(self.config["map"]["resolution"])
        points = set()
        for box in static_geometry:
            points.update(self.sample_box_surface(box, resolution))
        ordered = sorted(points)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            "WIDTH {count}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS {count}\n"
            "DATA ascii\n"
        ).format(count=len(ordered))
        return header + "".join("%.4f %.4f %.4f\n" % point for point in ordered)

    def output_directory(self, override=None):
        if override:
            return Path(override).expanduser().resolve()
        configured = Path(self.config["output"]["directory"])
        if configured.is_absolute():
            return configured
        return self.package_root / configured

    def generate(self, output_override=None, basename_override=None):
        basename = basename_override or self.config["output"]["basename"]
        output_dir = self.output_directory(output_override)
        output_dir.mkdir(parents=True, exist_ok=True)

        walls = self.make_walls()
        obstacles = self.generate_obstacle_layout()
        pedestrians = self.generate_pedestrians()
        scenario = self.scenario(walls, obstacles, pedestrians)
        scenario["name"] = basename

        world_path = output_dir / (basename + ".world")
        pcd_path = output_dir / (basename + ".pcd")
        scenario_path = output_dir / (basename + ".yaml")
        world_path.write_text(self.world_text(walls, obstacles, pedestrians))
        pcd_path.write_text(self.pcd_text(walls + obstacles))
        scenario_path.write_text(dump_structured_data(scenario))
        return world_path, pcd_path, scenario_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).with_name("corridor_world_generator.yaml")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--basename")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_structured_file(config_path)
    if args.seed is not None:
        config["random_seed"] = args.seed
        if args.basename is None:
            config["output"]["basename"] = "lpnav_corridor_seed_%d" % args.seed
    generator = CorridorWorldGenerator(config, config_path)
    paths = generator.generate(args.output_dir, args.basename)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
