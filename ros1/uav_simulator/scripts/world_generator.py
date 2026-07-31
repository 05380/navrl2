'''
    worldGenerator class for random (dynamic) world generation
    The generator will generate a gazebo world file which can be included in the start.launch
'''
import numpy as np
import os
import time


class worldGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.obstacle_dist = 2.0
        self.curr_obstacle_dist = self.obstacle_dist
        self.static_footprints = []
        self.large_structure_segments = []
        np.random.seed(self.cfg["random_seed"])

    def write_world_file(self):
        static_models, points = self.load_static_obstacles()
        structure_models, structure_points = self.load_large_static_structures()
        points.extend(structure_points)
        dynamic_models = self.load_dyanmic_obtacles()
        world_models = self.create_world_file(
            static_models + structure_models + dynamic_models
        )
        curr_path = os.path.dirname(os.path.abspath(__file__))
        parent_path = os.path.dirname(curr_path)
        os.makedirs(os.path.join(parent_path, "worlds/generated_env"), exist_ok=True)
        with open(os.path.join(parent_path, "worlds/generated_env/generated_env.world"), "w") as f:
            f.write(world_models)
        if (self.cfg["map"]["generate_map"]):
            if (self.cfg["map"]["save_directory"] == "default"):
                self.create_pcd(points, os.path.join(parent_path, "worlds/generated_env/generated_env.pcd"))
            else:
                self.create_pcd(points, self.cfg["map"]["save_directory"])

    def create_pcd(self, points, filename):
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            f"WIDTH {len(points)}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {len(points)}\n"
            "DATA ascii\n"
        )

        with open(filename, 'w') as file:
            file.write(header)
            for point in points:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")        

    def check_pos_validity(self, prev_pos_list, curr_pos):
        for prev_pos in prev_pos_list:
            if (np.linalg.norm(curr_pos - prev_pos) <= self.curr_obstacle_dist):
                return False
        return True

    @staticmethod
    def _sample_range(value, name):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("%s must be a two-element range" % name)
        low, high = float(value[0]), float(value[1])
        if high < low:
            raise ValueError("%s must be ordered as [min, max]" % name)
        return float(np.random.uniform(low=low, high=high))

    @staticmethod
    def _segment(center, local_offset, length, thickness, height, yaw, local_yaw=0.0):
        c = np.cos(yaw)
        s = np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        center_xy = np.asarray(center, dtype=float) + rotation.dot(
            np.asarray(local_offset, dtype=float)
        )
        return {
            "center": center_xy,
            "length": float(length),
            "thickness": float(thickness),
            "height": float(height),
            "yaw": float(yaw + local_yaw),
        }

    @staticmethod
    def _segment_axes(segment):
        yaw = float(segment["yaw"])
        longitudinal = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        lateral = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)
        return longitudinal, lateral

    @classmethod
    def _segment_corners(cls, segment):
        longitudinal, lateral = cls._segment_axes(segment)
        half_length = 0.5 * float(segment["length"])
        half_thickness = 0.5 * float(segment["thickness"])
        center = np.asarray(segment["center"], dtype=float)
        return np.asarray(
            [
                center + sx * half_length * longitudinal + sy * half_thickness * lateral
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )

    @classmethod
    def _segments_intersect(cls, first, second, clearance=0.0):
        first_long, first_lat = cls._segment_axes(first)
        second_long, second_lat = cls._segment_axes(second)
        center_delta = np.asarray(second["center"]) - np.asarray(first["center"])
        padding = 0.5 * max(float(clearance), 0.0)
        first_half = (
            0.5 * float(first["length"]) + padding,
            0.5 * float(first["thickness"]) + padding,
        )
        second_half = (
            0.5 * float(second["length"]) + padding,
            0.5 * float(second["thickness"]) + padding,
        )

        for axis in (first_long, first_lat, second_long, second_lat):
            center_distance = abs(float(np.dot(center_delta, axis)))
            first_radius = (
                first_half[0] * abs(float(np.dot(first_long, axis)))
                + first_half[1] * abs(float(np.dot(first_lat, axis)))
            )
            second_radius = (
                second_half[0] * abs(float(np.dot(second_long, axis)))
                + second_half[1] * abs(float(np.dot(second_lat, axis)))
            )
            if center_distance > first_radius + second_radius:
                return False
        return True

    @classmethod
    def _segment_intersects_circle(cls, segment, circle, clearance=0.0):
        longitudinal, lateral = cls._segment_axes(segment)
        relative = np.asarray(circle["center"], dtype=float) - np.asarray(
            segment["center"], dtype=float
        )
        local_x = float(np.dot(relative, longitudinal))
        local_y = float(np.dot(relative, lateral))
        half_length = 0.5 * float(segment["length"])
        half_thickness = 0.5 * float(segment["thickness"])
        closest_x = np.clip(local_x, -half_length, half_length)
        closest_y = np.clip(local_y, -half_thickness, half_thickness)
        distance = np.hypot(local_x - closest_x, local_y - closest_y)
        return distance <= float(circle["radius"]) + max(float(clearance), 0.0)

    def _segments_inside_bounds(self, segments, range_x, range_y):
        for segment in segments:
            corners = self._segment_corners(segment)
            if (
                np.min(corners[:, 0]) < float(range_x[0])
                or np.max(corners[:, 0]) > float(range_x[1])
                or np.min(corners[:, 1]) < float(range_y[0])
                or np.max(corners[:, 1]) > float(range_y[1])
            ):
                return False
        return True

    def _segments_clear_of_static_obstacles(self, segments, clearance):
        for segment in segments:
            for footprint in self.static_footprints:
                if footprint["shape"] == "circle":
                    if self._segment_intersects_circle(segment, footprint, clearance):
                        return False
                else:
                    static_segment = {
                        "center": footprint["center"],
                        "length": footprint["size"][0],
                        "thickness": footprint["size"][1],
                        "yaw": 0.0,
                    }
                    if self._segments_intersect(segment, static_segment, clearance):
                        return False
        return True

    def _segments_clear_of_large_structures(self, segments, clearance):
        return not any(
            self._segments_intersect(candidate, placed, clearance)
            for candidate in segments
            for placed in self.large_structure_segments
        )

    def _sample_large_structure(self, structure_type, info, common):
        thickness = self._sample_range(
            info.get("thickness", common["thickness"]),
            "%s.thickness" % structure_type,
        )
        height = self._sample_range(
            info.get("height", common["height"]),
            "%s.height" % structure_type,
        )
        yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
        range_x = common["range_x"]
        range_y = common["range_y"]
        center = np.array(
            [
                np.random.uniform(float(range_x[0]), float(range_x[1])),
                np.random.uniform(float(range_y[0]), float(range_y[1])),
            ],
            dtype=float,
        )

        if structure_type == "single_wall":
            length = self._sample_range(info["length"], "single_wall.length")
            return [self._segment(center, (0.0, 0.0), length, thickness, height, yaw)]

        if structure_type == "l_wall":
            arm_x = self._sample_range(info["arm_length"], "l_wall.arm_length")
            arm_y = self._sample_range(info["arm_length"], "l_wall.arm_length")
            sign_x = float(np.random.choice((-1.0, 1.0)))
            sign_y = float(np.random.choice((-1.0, 1.0)))
            return [
                self._segment(
                    center,
                    (0.0, -sign_y * arm_y * 0.5),
                    arm_x,
                    thickness,
                    height,
                    yaw,
                ),
                self._segment(
                    center,
                    (-sign_x * arm_x * 0.5, 0.0),
                    arm_y,
                    thickness,
                    height,
                    yaw,
                    np.pi * 0.5,
                ),
            ]

        if structure_type == "u_wall":
            width = self._sample_range(info["width"], "u_wall.width")
            depth = self._sample_range(info["depth"], "u_wall.depth")
            return [
                self._segment(
                    center,
                    (0.0, -depth * 0.5),
                    width + thickness,
                    thickness,
                    height,
                    yaw,
                ),
                self._segment(
                    center,
                    (-width * 0.5, 0.0),
                    depth,
                    thickness,
                    height,
                    yaw,
                    np.pi * 0.5,
                ),
                self._segment(
                    center,
                    (width * 0.5, 0.0),
                    depth,
                    thickness,
                    height,
                    yaw,
                    np.pi * 0.5,
                ),
            ]

        raise ValueError("Unsupported large structure type: %s" % structure_type)

    @staticmethod
    def _large_structure_model(name, segments):
        collision_visuals = []
        for index, segment in enumerate(segments):
            center = segment["center"]
            pose = "%f %f %f 0 0 %f" % (
                center[0],
                center[1],
                0.5 * segment["height"],
                segment["yaw"],
            )
            size = "%f %f %f" % (
                segment["length"],
                segment["thickness"],
                segment["height"],
            )
            collision_visuals.append(
                """
                <collision name='collision_{index}'>
                    <pose>{pose}</pose>
                    <geometry><box><size>{size}</size></box></geometry>
                </collision>
                <visual name='visual_{index}'>
                    <pose>{pose}</pose>
                    <geometry><box><size>{size}</size></box></geometry>
                    <material>
                        <ambient>0.32 0.36 0.42 1</ambient>
                        <diffuse>0.42 0.47 0.54 1</diffuse>
                        <specular>0.12 0.12 0.12 1</specular>
                    </material>
                </visual>
                """.format(index=index, pose=pose, size=size)
            )
        return """
        <model name='{name}'>
            <static>true</static>
            <link name='link'>
                {geometry}
            </link>
        </model>
        """.format(name=name, geometry="".join(collision_visuals))

    @classmethod
    def _large_structure_points(cls, segments, resolution):
        points = []
        resolution = float(resolution)
        if resolution <= 0.0:
            raise ValueError("large_static_structures.pcd_resolution must be positive")
        for segment in segments:
            longitudinal, lateral = cls._segment_axes(segment)
            center = np.asarray(segment["center"], dtype=float)
            local_x_values = np.arange(
                -0.5 * segment["length"],
                0.5 * segment["length"] + 0.5 * resolution,
                resolution,
            )
            local_y_values = np.arange(
                -0.5 * segment["thickness"],
                0.5 * segment["thickness"] + 0.5 * resolution,
                resolution,
            )
            z_values = np.arange(
                0.0,
                segment["height"] + 0.5 * resolution,
                resolution,
            )
            for local_x in local_x_values:
                for local_y in local_y_values:
                    xy = center + local_x * longitudinal + local_y * lateral
                    for z in z_values:
                        points.append([xy[0], xy[1], z])
        return points

    def load_large_static_structures(self):
        config = self.cfg.get("large_static_structures")
        if not config or not bool(config.get("enabled", True)):
            return [], []

        common = {
            "range_x": config.get("range_x", [-9.0, 9.0]),
            "range_y": config.get("range_y", [-9.0, 9.0]),
            "thickness": config.get("thickness", [0.2, 0.35]),
            "height": config.get("height", [3.0, 4.0]),
        }
        for axis_name in ("range_x", "range_y"):
            axis_range = common[axis_name]
            if len(axis_range) != 2 or float(axis_range[1]) <= float(axis_range[0]):
                raise ValueError(
                    "large_static_structures.%s must be an increasing range" % axis_name
                )

        obstacle_clearance = float(config.get("obstacle_clearance", 0.35))
        structure_clearance = float(config.get("structure_clearance", 0.75))
        sampling_attempts = int(config.get("sampling_attempts", 2000))
        pcd_resolution = float(config.get("pcd_resolution", 0.1))
        if sampling_attempts <= 0:
            raise ValueError("large_static_structures.sampling_attempts must be positive")

        models = []
        points = []
        structure_types = ("single_wall", "l_wall", "u_wall")
        for structure_type in structure_types:
            info = config.get(structure_type, {})
            count = int(info.get("num", 0))
            if count < 0:
                raise ValueError(
                    "large_static_structures.%s.num cannot be negative" % structure_type
                )
            for index in range(count):
                segments = None
                for _ in range(sampling_attempts):
                    candidate = self._sample_large_structure(
                        structure_type, info, common
                    )
                    if not self._segments_inside_bounds(
                        candidate, common["range_x"], common["range_y"]
                    ):
                        continue
                    if not self._segments_clear_of_static_obstacles(
                        candidate, obstacle_clearance
                    ):
                        continue
                    if not self._segments_clear_of_large_structures(
                        candidate, structure_clearance
                    ):
                        continue
                    segments = candidate
                    break

                if segments is None:
                    raise RuntimeError(
                        "Unable to place %s %d after %d attempts; reduce obstacle "
                        "density/counts or clearances"
                        % (structure_type, index, sampling_attempts)
                    )

                name = "large_%s_%02d" % (structure_type, index)
                models.append(self._large_structure_model(name, segments))
                points.extend(self._large_structure_points(segments, pcd_resolution))
                self.large_structure_segments.extend(segments)

        return models, points

    def load_static_obstacles(self):
        static_obstacles = self.cfg["static_objects"]
        
        static_models = []
        prev_pos_list = [] # 2d
        points = [] # a list of points
        for obstacle_type in static_obstacles:
            obstacle_info = static_obstacles[obstacle_type]
            num_obstacles = obstacle_info["num"]
            range_x = obstacle_info["range_x"]
            range_y = obstacle_info["range_y"]

            if (obstacle_type == "box"):
                range_z = obstacle_info["range_z"]
                width_x_range = obstacle_info["width_x"]
                width_y_range = obstacle_info["width_y"]
            else:
                range_z = [0.0, 0.0]
                radius_range = obstacle_info["radius"]


            obstacle_height_range = obstacle_info["height"]        
            check_validity = self.cfg["even_distribution"]

            i = 0
            start_time = time.time()
            while i < num_obstacles:
                ox = np.random.uniform(low=range_x[0], high=range_x[1])
                oy = np.random.uniform(low=range_y[0], high=range_y[1])
                oz = np.random.uniform(low=range_z[0], high=range_z[1])
                height = np.random.uniform(low=obstacle_height_range[0], high=obstacle_height_range[1])
                curr_pos = np.array([ox, oy])
                
                if (check_validity):
                    valid = self.check_pos_validity(prev_pos_list, curr_pos)
                    curr_time = time.time()
                    if (valid):
                        start_time = time.time()
                    else:
                        if ((curr_time - start_time > 0.1)):
                            self.curr_obstacle_dist *= 0.8
                            start_time = time.time()
                        continue

                if (obstacle_type == "box"):
                    ob_size = (np.random.uniform(low=width_x_range[0], high=width_x_range[1]), np.random.uniform(low=width_y_range[0], high=width_y_range[1]))
                    self.static_footprints.append(
                        {
                            "shape": "box",
                            "center": curr_pos.copy(),
                            "size": ob_size,
                        }
                    )
                    static_models.append(
                            f"""
                            <model name='box_{i}_{ob_size[0]:.1f}_{ob_size[1]:.1f}_{height:.1f}'>
                            <static>true</static>
                            <pose>{ox} {oy} {oz+height/2.} 0 0 0</pose> <!-- X, Y, Z, Roll, Pitch, Yaw -->
                            <link name='link'>
                                <visual name='visual'>
                                <geometry>
                                    <box>
                                        <size>{ob_size[0]:.1f} {ob_size[1]:.1f} {height:.1f}</size> <!-- Width, Depth, Height -->
                                    </box>
                                </geometry>
                                </visual>
                            </link>
                            </model> 
                            """
                    )
                else:
                    ob_size = (np.random.uniform(low=radius_range[0], high=radius_range[1]))
                    self.static_footprints.append(
                        {
                            "shape": "circle",
                            "center": curr_pos.copy(),
                            "radius": ob_size,
                        }
                    )
                    static_models.append(
                            f"""
                            <model name='cylinder_{i}_{2*ob_size:.1f}_{2*ob_size:.1f}_{height:.1f}'>
                            <static>true</static>
                            <pose>{ox} {oy} {oz+height/2.} 0 0 0</pose> <!-- X, Y, Z, Roll, Pitch, Yaw -->
                            <link name='link'>
                                <visual name='visual'>
                                <geometry>
                                    <cylinder>
                                        <radius>{ob_size}</radius>
                                        <length>{height}</length>
                                    </cylinder>
                                </geometry>
                                </visual>
                            </link>
                            </model> 
                            """
                    )
                prev_pos_list.append(curr_pos)
                i += 1

                # map generation
                if (obstacle_type == "box"):
                    start_x = ox - ob_size[0]/2.
                    start_y = oy - ob_size[1]/2.
                    start_z = oz
                    end_x = ox + ob_size[0]/2.
                    end_y = oy + ob_size[1]/2.
                    end_z = oz + height
                else:
                    start_x = ox - ob_size
                    start_y = oy - ob_size
                    start_z = oz
                    end_x = ox + ob_size
                    end_y = oy + ob_size
                    end_z = oz + height
                
                for px in np.arange(start_x, end_x+0.1, step=0.1):
                    for py in np.arange(start_y, end_y+0.1, step=0.1):
                        for pz in np.arange(start_z, end_z+0.1, step=0.1):
                            if (obstacle_type == "box"):
                                points.append([px, py, pz])
                            else:
                                dist = ((px - ox)**2 + (py - oy)**2)**0.5
                                if (dist < ob_size):
                                    points.append([px, py, pz])
        self.curr_obstacle_dist = self.obstacle_dist
        return static_models, points

    def load_dyanmic_obtacles(self):
        dynamic_obstacles = self.cfg["dynamic_objects"]
        
        dynamic_models = []
        prev_pos_list = [] # 2d
        for obstacle_type in dynamic_obstacles:
            obstacle_info = dynamic_obstacles[obstacle_type]
            num_obstacles = obstacle_info["num"]
            range_x = obstacle_info["range_x"]
            range_y = obstacle_info["range_y"]
            
            if (obstacle_type == "box"):
                range_z = obstacle_info["range_z"]
                width_x_range = obstacle_info["width_x"]
                width_y_range = obstacle_info["width_y"]
            else:
                range_z = [0.0, 0.0]
                radius_range = obstacle_info["radius"]


            obstacle_height_range = obstacle_info["height"]
            velocity_range = obstacle_info["velocity"]        
            check_validity = self.cfg["even_distribution"]

            i = 0
            start_time = time.time()
            while i < num_obstacles:
                ox = np.random.uniform(low=range_x[0], high=range_x[1])
                oy = np.random.uniform(low=range_y[0], high=range_y[1])
                oz = np.random.uniform(low=range_z[0], high=range_z[1])
                gx = np.random.uniform(low=range_x[0], high=range_x[1])
                gy = np.random.uniform(low=range_y[0], high=range_y[1])
                gz = np.random.uniform(low=range_z[0], high=range_z[1])                
                height = np.random.uniform(low=obstacle_height_range[0], high=obstacle_height_range[1])
                velocity = np.random.uniform(low=velocity_range[0], high=velocity_range[1])
                curr_pos = np.array([ox, oy])

                if (check_validity):
                    valid = self.check_pos_validity(prev_pos_list, curr_pos)
                    curr_time = time.time()
                    if (valid):
                        start_time = time.time()
                    else:
                        if ((curr_time - start_time > 0.1)):
                            self.curr_obstacle_dist *= 0.8
                        continue

                if (obstacle_type == "box"):
                    ob_size = (np.random.uniform(low=width_x_range[0], high=width_x_range[1]), np.random.uniform(low=width_y_range[0], high=width_y_range[1]))
                    dynamic_models.append(
                            f"""
                            <model name='dynamic_box_{i}_{ob_size[0]:.1f}_{ob_size[1]:.1f}_{height:.1f}'>
                            <static>true</static>
                            <pose>{ox} {oy} {oz+height/2.} 0 0 0</pose> <!-- X, Y, Z, Roll, Pitch, Yaw -->
                            <link name='link'>
                                <visual name='visual'>
                                    <geometry>
                                        <box>
                                            <size>{ob_size[0]} {ob_size[1]} {height}</size> <!-- Width, Depth, Height -->
                                        </box>
                                    </geometry>
                                    <material>
                                        <ambient>1 0 0 1</ambient> <!-- Red color -->
                                        <diffuse>1 0 0 1</diffuse> <!-- Red color -->
                                        <specular>0.5 0.5 0.5 1</specular> <!-- Specular highlight -->
                                    </material>
                                </visual>
                            </link>
                            <plugin name="obstacle_motion" filename="libobstaclePathPlugin.so">
                                <orientation>false</orientation>
                                <loop>0</loop>
                                <velocity>{velocity}</velocity>
                                <path>
                                <waypoint>{ox} {oy} {oz+height/2.}</waypoint>
                                <waypoint>{gx} {gy} {gz+height/2.}</waypoint>
                                </path>
                            </plugin>
                            </model> 
                            """
                    )
                else:
                    ob_size = (np.random.uniform(low=radius_range[0], high=radius_range[1]))
                    dynamic_models.append(
                            f"""
                            <model name='dynamic_cylinder_{i}_{2*ob_size:.1f}_{2*ob_size:.1f}_{height:.1f}'>
                            <static>true</static>
                            <pose>{ox} {oy} {oz+height/2.} 0 0 0</pose> <!-- X, Y, Z, Roll, Pitch, Yaw -->
                            <link name='link'>
                                <visual name='visual'>
                                    <geometry>
                                        <cylinder>
                                            <radius>{ob_size}</radius>
                                            <length>{height}</length>
                                        </cylinder>
                                    </geometry>
                                    <material>
                                        <ambient>1 0 0 1</ambient> <!-- Red color -->
                                        <diffuse>1 0 0 1</diffuse> <!-- Red color -->
                                        <specular>0.5 0.5 0.5 1</specular> <!-- Specular highlight -->
                                    </material>
                                </visual>
                            </link>
                            <plugin name="obstacle_motion" filename="libobstaclePathPlugin.so">
                                <orientation>false</orientation>
                                <loop>0</loop>
                                <velocity>{velocity}</velocity>
                                <path>
                                <waypoint>{ox} {oy} {oz+height/2.}</waypoint>
                                <waypoint>{gx} {gy} {gz+height/2.}</waypoint>
                                </path>
                            </plugin>
                            </model> 
                            """
                    )
                prev_pos_list.append(curr_pos)
                i += 1
        self.curr_obstacle_dist = self.obstacle_dist
        return dynamic_models 
    
    def create_world_file(self, models):
        # print(models)
        # models = "\n".join(models)
        world_model = f"""
            <sdf version='1.7'>
            <world name='default'>
                <light name='sun' type='directional'>
                <cast_shadows>1</cast_shadows>
                <pose>0 0 10 0 -0 0</pose>
                <diffuse>0.8 0.8 0.8 1</diffuse>
                <specular>0.2 0.2 0.2 1</specular>
                <attenuation>
                    <range>1000</range>
                    <constant>0.9</constant>
                    <linear>0.01</linear>
                    <quadratic>0.001</quadratic>
                </attenuation>
                <direction>-0.5 0.1 -0.9</direction>
                <spot>
                    <inner_angle>0</inner_angle>
                    <outer_angle>0</outer_angle>
                    <falloff>0</falloff>
                </spot>
                </light>
                <model name='ground_plane'>
                <static>1</static>
                <link name='link'>
                    <collision name='collision'>
                    <geometry>
                        <plane>
                        <normal>0 0 1</normal>
                        <size>100 100</size>
                        </plane>
                    </geometry>
                    <surface>
                        <contact>
                        <collide_bitmask>65535</collide_bitmask>
                        <ode/>
                        </contact>
                        <friction>
                        <ode>
                            <mu>100</mu>
                            <mu2>50</mu2>
                        </ode>
                        <torsional>
                            <ode/>
                        </torsional>
                        </friction>
                        <bounce/>
                    </surface>
                    <max_contacts>10</max_contacts>
                    </collision>
                    <visual name='visual'>
                    <cast_shadows>0</cast_shadows>
                    <geometry>
                        <plane>
                        <normal>0 0 1</normal>
                        <size>100 100</size>
                        </plane>
                    </geometry>
                    <material>
                        <script>
                        <uri>file://media/materials/scripts/gazebo.material</uri>
                        <name>Gazebo/Grey</name>
                        </script>
                    </material>
                    </visual>
                    <self_collide>0</self_collide>
                    <enable_wind>0</enable_wind>
                    <kinematic>0</kinematic>
                </link>
                </model>
                <gravity>0 0 -9.8</gravity>
                <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
                <atmosphere type='adiabatic'/>
                <physics type='ode'>
                <max_step_size>0.001</max_step_size>
                <real_time_factor>1</real_time_factor>
                <real_time_update_rate>1000</real_time_update_rate>
                </physics>
                <scene>
                <ambient>0.4 0.4 0.4 1</ambient>
                <background>0.7 0.7 0.7 1</background>
                <shadows>1</shadows>
                </scene>
                <wind/>
                <spherical_coordinates>
                <surface_model>EARTH_WGS84</surface_model>
                <latitude_deg>0</latitude_deg>
                <longitude_deg>0</longitude_deg>
                <elevation>0</elevation>
                <heading_deg>0</heading_deg>
                </spherical_coordinates>
                {''.join(models)}
            </world>
            </sdf>
            """
        return world_model
