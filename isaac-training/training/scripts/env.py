import torch
import torch.nn.functional as F
import einops
import numpy as np
import trimesh
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni.isaac.orbit.terrains.height_field.utils import convert_height_field_to_mesh
from omni.isaac.orbit.utils import configclass
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time


def _range_to_tuple(value):
    return (float(value[0]), float(value[1]))


def _sample_range(value):
    low, high = _range_to_tuple(value)
    return float(np.random.uniform(low, high))


def _sequence_to_tuple(value):
    return tuple(float(item) for item in value)


def _sample_piecewise_range(range_edges, probabilities):
    range_edges = _sequence_to_tuple(range_edges)
    probabilities = np.asarray(_sequence_to_tuple(probabilities), dtype=np.float64)
    if len(range_edges) != len(probabilities) + 1:
        raise ValueError(
            "Piecewise range sampling expects len(range_edges) == len(probabilities) + 1, "
            f"got {len(range_edges)} and {len(probabilities)}."
        )
    probability_sum = probabilities.sum()
    if probability_sum <= 0.0:
        raise ValueError("Piecewise range probabilities must sum to a positive value.")
    probabilities = probabilities / probability_sum
    interval_idx = int(np.random.choice(len(probabilities), p=probabilities))
    return float(np.random.uniform(range_edges[interval_idx], range_edges[interval_idx + 1]))


def _sample_wall_height(cfg):
    if len(cfg.wall_height_range) > 2:
        return _sample_piecewise_range(cfg.wall_height_range, cfg.wall_height_probability)
    return _sample_range(cfg.wall_height_range)


def _sample_wall_center(size, max_extent, placement_margin):
    margin = max(float(placement_margin), 0.5 * float(max_extent) + 1.0)
    margin_x = min(margin, float(size[0]) * 0.45)
    margin_y = min(margin, float(size[1]) * 0.45)
    return np.array(
        [
            np.random.uniform(margin_x, float(size[0]) - margin_x),
            np.random.uniform(margin_y, float(size[1]) - margin_y),
        ],
        dtype=np.float32,
    )


def _rotation_2d(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _wall_segment_spec(center_xy, local_offset_xy, length, thickness, height, yaw, local_yaw=0.0):
    segment_yaw = yaw + local_yaw
    segment_xy = np.asarray(center_xy) + _rotation_2d(yaw) @ np.asarray(local_offset_xy, dtype=np.float32)
    return {
        "center_xy": np.asarray(segment_xy, dtype=np.float32),
        "length": float(length),
        "thickness": float(thickness),
        "height": float(height),
        "yaw": float(segment_yaw),
    }


def _wall_segment_mesh(segment):
    center_xy = segment["center_xy"]
    length = segment["length"]
    thickness = segment["thickness"]
    height = segment["height"]
    segment_yaw = segment["yaw"]
    transform = np.eye(4)
    transform[0:3, -1] = (float(center_xy[0]), float(center_xy[1]), float(height) * 0.5)
    transform[0:3, 0:3] = np.array(
        [
            [np.cos(segment_yaw), -np.sin(segment_yaw), 0.0],
            [np.sin(segment_yaw), np.cos(segment_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return trimesh.creation.box((float(length), float(thickness), float(height)), transform=transform)


def _make_single_wall(center, cfg):
    length = _sample_range(cfg.wall_length_range)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    max_extent = length + thickness
    segments = [_wall_segment_spec(center, (0.0, 0.0), length, thickness, height, yaw)]
    return segments, max_extent


def _make_l_wall(center, cfg):
    arm_x = _sample_range(cfg.wall_l_length_range)
    arm_y = _sample_range(cfg.wall_l_length_range)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    sign_x, sign_y = signs[int(np.random.randint(0, len(signs)))]
    max_extent = max(arm_x, arm_y) + thickness
    segments = [
        _wall_segment_spec(center, (0.0, -sign_y * arm_y * 0.5), arm_x, thickness, height, yaw),
        _wall_segment_spec(center, (-sign_x * arm_x * 0.5, 0.0), arm_y, thickness, height, yaw, np.pi * 0.5),
    ]
    return segments, max_extent


def _make_u_wall(center, cfg):
    width = _sample_range(cfg.wall_u_width_range)
    depth = _sample_range(cfg.wall_u_depth_range)
    thickness = _sample_range(cfg.wall_thickness_range)
    height = _sample_wall_height(cfg)
    opening_direction = int(np.random.randint(0, 4))
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi) + opening_direction * np.pi * 0.5)
    max_extent = max(width, depth) + thickness
    segments = [
        _wall_segment_spec(center, (0.0, -depth * 0.5), width + thickness, thickness, height, yaw),
        _wall_segment_spec(center, (-width * 0.5, 0.0), depth, thickness, height, yaw, np.pi * 0.5),
        _wall_segment_spec(center, (width * 0.5, 0.0), depth, thickness, height, yaw, np.pi * 0.5),
    ]
    return segments, max_extent


def _grid_coordinates(size, horizontal_scale):
    width_pixels = int(float(size[0]) / float(horizontal_scale)) + 1
    length_pixels = int(float(size[1]) / float(horizontal_scale)) + 1
    x = np.linspace(0.0, float(size[0]), width_pixels, dtype=np.float32)
    y = np.linspace(0.0, float(size[1]), length_pixels, dtype=np.float32)
    return np.meshgrid(x, y, indexing="ij")


def _segments_mask(grid_x, grid_y, segments, clearance=0.0):
    mask = np.zeros_like(grid_x, dtype=bool)
    for segment in segments:
        center_xy = segment["center_xy"]
        rel_x = grid_x - float(center_xy[0])
        rel_y = grid_y - float(center_xy[1])
        c = float(np.cos(segment["yaw"]))
        s = float(np.sin(segment["yaw"]))
        local_x = c * rel_x + s * rel_y
        local_y = -s * rel_x + c * rel_y
        half_length = 0.5 * float(segment["length"]) + float(clearance)
        half_thickness = 0.5 * float(segment["thickness"]) + float(clearance)
        mask |= (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_thickness)
    return mask


def _generate_discrete_obstacles_height_field(cfg):
    width_pixels = int(float(cfg.size[0]) / float(cfg.horizontal_scale)) + 1
    length_pixels = int(float(cfg.size[1]) / float(cfg.horizontal_scale)) + 1
    obs_width_min = max(1, int(float(cfg.obstacle_width_range[0]) / float(cfg.horizontal_scale)))
    obs_width_max = max(obs_width_min + 1, int(float(cfg.obstacle_width_range[1]) / float(cfg.horizontal_scale)))
    platform_width = int(float(cfg.platform_width) / float(cfg.horizontal_scale))

    obs_width_range = np.arange(obs_width_min, obs_width_max + 1, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max + 1, 4)
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)
    obstacles_history = []
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    probability_length = len(cfg.obstacle_height_probability)

    def good_distance(x, y, width, length, obstacles_hist, bad_range=(2, 10)):
        lower_bound_pixels = bad_range[0]
        upper_bound_pixels = bad_range[1]
        for (xp, yp, wp, lp) in obstacles_hist:
            dx = abs(xp - x) - width
            dy = abs(yp - y) - length
            if dx < 0 and dy < 0:
                continue
            distance = np.sqrt(dx**2 + dy**2)
            if lower_bound_pixels <= distance <= upper_bound_pixels:
                return False
        return True

    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-2, -1, 1, 2])
        elif cfg.obstacle_height_mode == "fixed":
            height = float(cfg.obstacle_height_range[0]) / float(cfg.vertical_scale)
        elif cfg.obstacle_height_mode == "range":
            random_roll = int(np.random.choice(probability_length, 1, p=cfg.obstacle_height_probability))
            low = float(cfg.obstacle_height_range[random_roll]) / float(cfg.vertical_scale)
            high = float(cfg.obstacle_height_range[random_roll + 1]) / float(cfg.vertical_scale)
            height = np.random.uniform(low, high)
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice', 'fixed' or 'range'."
            )

        attempts = 0
        while attempts < 100000:
            width = int(np.random.choice(obs_width_range))
            length = int(np.random.choice(obs_length_range))
            x_start = int(np.random.choice(obs_x_range))
            y_start = int(np.random.choice(obs_y_range))
            if x_start + width > width_pixels:
                x_start = width_pixels - width
            if y_start + length > length_pixels:
                y_start = length_pixels - length
            if good_distance(x_start, y_start, width, length, obstacles_history) or not obstacles_history:
                break
            attempts += 1

        obstacles_history.append((x_start, y_start, width, length))
        hf_raw[x_start : x_start + width, y_start : y_start + length] = int(np.rint(height))

    if platform_width > 0:
        x1 = (width_pixels - platform_width) // 2
        x2 = (width_pixels + platform_width) // 2
        y1 = (length_pixels - platform_width) // 2
        y2 = (length_pixels + platform_width) // 2
        hf_raw[x1:x2, y1:y2] = 0

    return hf_raw


def _sample_valid_wall_segments(wall_type, cfg, placed_centers, static_occupied, wall_occupied, grid_x, grid_y):
    for _ in range(int(cfg.wall_sampling_attempts)):
        if wall_type == "single":
            segments_builder = _make_single_wall
        elif wall_type == "l":
            segments_builder = _make_l_wall
        elif wall_type == "u":
            segments_builder = _make_u_wall
        else:
            raise ValueError(f"Unknown wall type: {wall_type}")

        provisional_center = _sample_wall_center(cfg.size, 0.0, cfg.wall_placement_margin)
        segments, _ = segments_builder(provisional_center, cfg)
        center = np.mean([segment["center_xy"] for segment in segments], axis=0)

        if any(np.linalg.norm(center - prev) < float(cfg.wall_min_separation) for prev in placed_centers):
            continue

        obstacle_clearance_mask = _segments_mask(
            grid_x, grid_y, segments, clearance=float(cfg.wall_obstacle_clearance)
        )
        if np.any(static_occupied & obstacle_clearance_mask):
            continue

        wall_clearance_mask = _segments_mask(grid_x, grid_y, segments, clearance=float(cfg.wall_min_separation))
        if np.any(wall_occupied & wall_clearance_mask):
            continue

        exact_mask = _segments_mask(grid_x, grid_y, segments, clearance=0.0)
        if not np.any(exact_mask):
            continue

        placed_centers.append(center)
        return segments, exact_mask

    return None, None


def _sample_valid_wall_center(cfg, placed_centers, max_extent):
    min_separation = max(float(cfg.wall_min_separation), float(max_extent))
    center = _sample_wall_center(cfg.size, max_extent, cfg.wall_placement_margin)
    for _ in range(100):
        if all(np.linalg.norm(center - prev) >= min_separation for prev in placed_centers):
            placed_centers.append(center)
            return center
        center = _sample_wall_center(cfg.size, max_extent, cfg.wall_placement_margin)
    placed_centers.append(center)
    return center


def discrete_obstacles_with_curriculum_walls_terrain(difficulty, cfg):
    heights = _generate_discrete_obstacles_height_field(cfg)
    vertices, triangles = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    meshes = [terrain_mesh]
    origin = np.array([0.5 * float(cfg.size[0]), 0.5 * float(cfg.size[1]), 0.0], dtype=np.float32)

    static_occupied = heights > 0
    wall_occupied = np.zeros_like(static_occupied, dtype=bool)
    grid_x, grid_y = _grid_coordinates(cfg.size, cfg.horizontal_scale)
    wall_style = int(cfg.wall_style)

    cfg._terrain_occupancy = static_occupied.copy()
    cfg._terrain_size = np.array(cfg.size, dtype=np.float32)
    cfg._terrain_resolution = float(cfg.horizontal_scale)

    if wall_style == 0:
        return meshes, origin

    # Walls are appended to the terrain mesh so static LiDAR and collision use the same geometry.
    if wall_style == 1:
        wall_plan = [("single", 1)]
    elif wall_style == 2:
        wall_plan = [("single", 2)]
    elif wall_style == 3:
        wall_plan = [("l", 1)]
    elif wall_style == 4:
        wall_plan = [("l", 2)]
    elif wall_style == 5:
        wall_plan = [("u", 1)]
    elif wall_style == 6:
        wall_plan = [("u", 2)]
    else:
        raise ValueError(f"Unsupported env.wall_style={wall_style}. Expected an integer in [0, 6].")

    placed_centers = []
    for wall_type, wall_count in wall_plan:
        for _ in range(wall_count):
            segments, exact_mask = _sample_valid_wall_segments(
                wall_type, cfg, placed_centers, static_occupied, wall_occupied, grid_x, grid_y
            )
            if segments is None:
                continue
            meshes.extend(_wall_segment_mesh(segment) for segment in segments)
            wall_occupied |= exact_mask

    cfg._terrain_occupancy = static_occupied | wall_occupied
    return meshes, origin


@configclass
class HfDiscreteObstaclesWithWallsTerrainCfg(HfDiscreteObstaclesTerrainCfg):
    function = discrete_obstacles_with_curriculum_walls_terrain

    wall_style: int = 0
    wall_length_range: tuple[float, float] = (4.0, 8.0)
    wall_l_length_range: tuple[float, float] = (3.5, 6.5)
    wall_u_width_range: tuple[float, float] = (4.0, 7.0)
    wall_u_depth_range: tuple[float, float] = (3.5, 6.5)
    wall_thickness_range: tuple[float, float] = (0.35, 0.65)
    wall_height_range: tuple[float, ...] = (2.0, 4.0, 6.0)
    wall_height_probability: tuple[float, ...] = (0.10, 0.90)
    wall_placement_margin: float = 4.0
    wall_min_separation: float = 4.0
    wall_obstacle_clearance: float = 0.4
    wall_sampling_attempts: int = 200


class NavigationEnv(IsaacEnv):

    # In one step:
    # 1. _pre_sim_step (apply action) -> step isaac sim
    # 2. _post_sim_step (update lidar)
    # 3. increment progress_buf
    # 4. _compute_state_and_obs (get observation and states, update stats)
    # 5. _compute_reward_and_done (update reward and calculate returns)

    def __init__(self, cfg):
        print("[Navigation Environment]: Initializing Env...")
        # LiDAR params:
        self.lidar_range = cfg.sensor.lidar_range
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))
        self.lidar_vbeams = cfg.sensor.lidar_vbeams
        self.lidar_hres = cfg.sensor.lidar_hres
        self.lidar_hbeams = int(360/self.lidar_hres)
        stuck_cfg = cfg.get("stuck", {})
        self.stuck_window = max(1, int(stuck_cfg.get("window", 40)))
        self.stuck_progress_eps = float(stuck_cfg.get("progress_eps", 0.005))
        self.stuck_front_distance = min(self.lidar_range, float(stuck_cfg.get("front_obstacle_distance", 1.5)))
        self.stuck_front_tan = float(np.tan(np.deg2rad(stuck_cfg.get("front_angle_deg", 35.0))))
        self.stuck_front_height = float(stuck_cfg.get("front_height", 0.75))
        reward_cfg = cfg.get("reward", {})
        goal_progress_cfg = reward_cfg.get("goal_progress", {})
        self.goal_radius = float(goal_progress_cfg.get("goal_radius", 0.5))
        self.goal_arrival_reward = float(goal_progress_cfg.get("arrival_reward", 5.0))
        self.goal_timeout_penalty = float(goal_progress_cfg.get("timeout_penalty", self.goal_arrival_reward))
        self.goal_progress_scale = float(goal_progress_cfg.get("progress_scale", 10.0))
        stall_reward_cfg = reward_cfg.get("stall", {})
        self.stall_reward_weight = float(stall_reward_cfg.get("weight", 0.5))
        self.stall_reward_window = max(1, int(stall_reward_cfg.get("window", self.stuck_window)))
        self.stall_reward_ramp_window = max(1, int(stall_reward_cfg.get("ramp_window", self.stall_reward_window)))
        self.stall_distance_threshold = float(stall_reward_cfg.get("distance_threshold", 0.25))
        self.stall_speed_threshold = float(stall_reward_cfg.get("speed_threshold", 0.15))
        self.stall_progress_threshold = float(stall_reward_cfg.get("progress_eps", self.stuck_progress_eps))
        vo_cfg = reward_cfg.get("vo", {})
        self.vo_weight = float(vo_cfg.get("weight", 1.0))
        self.vo_tau = max(float(vo_cfg.get("tau", 0.75)), 1e-6)
        self.vo_horizon = max(float(vo_cfg.get("horizon", 2.5)), 1e-6)
        self.vo_xy_margin = float(vo_cfg.get("xy_margin", 0.3))
        self.vo_z_margin = float(vo_cfg.get("z_margin", 0.3))
        self.vo_reward_topk = max(1, int(vo_cfg.get("reward_topk", 10)))
        self.vo_reward_range = max(self.lidar_range, float(max(cfg.env_dyn.local_range)))
        self.vo_warmup_steps = max(0, int(vo_cfg.get("warmup_steps", 0)))
        self.vo_step_count = 0

        super().__init__(cfg, cfg.headless)
        
        # Drone Initialization
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())


        # LiDAR Intialization
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            # attach_yaw_only=False,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # horizontal default is set to 10
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) 
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            # mesh_prim_paths=["/World"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # start and target 
        with torch.device(self.device):
            # self.start_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            
            # Coordinate change: add target direction variable
            self.target_dir = torch.zeros(self.num_envs, 1, 3)
            self.height_range = torch.zeros(self.num_envs, 1, 2)
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1 , 3)
            self.prev_goal_distance = torch.zeros(self.num_envs, 1)
            self.stuck_counter = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.stall_counter = torch.zeros(self.num_envs, 1, dtype=torch.long)
            self.stall_anchor_pos = torch.zeros(self.num_envs, 1, 3)
            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.     


    def _design_scene(self):
        # Initialize a drone in prim /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] # drone model class
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        # drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 1.0)])[0]
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]

        # lighting
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # Ground Plane
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        self.map_range = [20.0, 20.0, 4.5]
        wall_cfg = self.cfg.env.get("wall", {})
        self.terrain_occupancy = None
        self.terrain_occupancy_resolution = 0.1
        self.terrain_occupancy_size = torch.tensor(
            [self.map_range[0] * 2.0, self.map_range[1] * 2.0], dtype=torch.float32, device=self.device
        )
        self.terrain_clearance_cache = {}
        self.spawn_clearance_radius = float(wall_cfg.get("spawn_clearance_radius", 1.5))
        self.target_clearance_radius = float(wall_cfg.get("target_clearance_radius", self.spawn_clearance_radius))
        self.position_sample_attempts = max(1, int(wall_cfg.get("position_sample_attempts", 128)))

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesWithWallsTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],
                        platform_width=0.0,
                        wall_style=int(self.cfg.env.get("wall_style", 0)),
                        wall_length_range=_range_to_tuple(wall_cfg.get("length_range", [4.0, 8.0])),
                        wall_l_length_range=_range_to_tuple(wall_cfg.get("l_length_range", [3.5, 6.5])),
                        wall_u_width_range=_range_to_tuple(wall_cfg.get("u_width_range", [4.0, 7.0])),
                        wall_u_depth_range=_range_to_tuple(wall_cfg.get("u_depth_range", [3.5, 6.5])),
                        wall_thickness_range=_range_to_tuple(wall_cfg.get("thickness_range", [0.35, 0.65])),
                        wall_height_range=_sequence_to_tuple(wall_cfg.get("height_range", [2.0, 4.0, 6.0])),
                        wall_height_probability=_sequence_to_tuple(wall_cfg.get("height_probability", [0.10, 0.90])),
                        wall_placement_margin=float(wall_cfg.get("placement_margin", 4.0)),
                        wall_min_separation=float(wall_cfg.get("min_separation", 4.0)),
                        wall_obstacle_clearance=float(wall_cfg.get("obstacle_clearance", 0.4)),
                        wall_sampling_attempts=int(wall_cfg.get("sampling_attempts", 200)),
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)
        obstacle_terrain_cfg = terrain_cfg.terrain_generator.sub_terrains["obstacles"]
        terrain_occupancy = getattr(obstacle_terrain_cfg, "_terrain_occupancy", None)
        terrain_size = getattr(obstacle_terrain_cfg, "_terrain_size", None)
        terrain_resolution = getattr(obstacle_terrain_cfg, "_terrain_resolution", None)
        if terrain_occupancy is not None:
            self.terrain_occupancy = torch.as_tensor(
                terrain_occupancy.astype(np.float32), device=self.device, dtype=torch.float32
            )
        if terrain_size is not None:
            self.terrain_occupancy_size = torch.as_tensor(terrain_size[:2], device=self.device, dtype=torch.float32)
        if terrain_resolution is not None:
            self.terrain_occupancy_resolution = float(terrain_resolution)

        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # Dynamic Obstacles
        # NOTE: we use cuboid to represent 3D dynamic obstacles which can float in the air 
        # and the long cylinder to represent 2D dynamic obstacles for which the drone can only pass in 2D 
        # The width of the dynamic obstacles is divided into N_w=4 bins
        # [[0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]]
        # The height of the dynamic obstacles is divided into N_h=2 bins
        # [[0, 0.5], [0.5, inf]] we want to distinguish 3D obstacles and 2d obstacles
        N_w = 4 # number of width intervals between [0, 1]
        N_h = 2 # number of height: current only support binary
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # in case of the roundup error


        # Dynamic obstacle info
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) # 13 is based on the states from sim, we only care the first three which is position
        self.dyn_obs_state[:, 3] = 1. # Quaternion
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # dynamic obstacle motion step count
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) # size of dynamic obstacles


        # helper function to check pos validity for even distribution condition
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) # prefered distance between each dynamic obstacle
        curr_obs_dist = obs_dist
        prev_pos_list = [] # for distance check
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h)
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # create all origins for 3D dynamic obstacles of this category (size)
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # random sample an origin until satisfy the evenly distributed condition
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2. # half of the height
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # Spawn various sizes of dynamic obstacles 
            if (category_idx < cuboid_category_num):
                # spawn for 3D dynamic obstacles
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # spawn for 2D dynamic obstacles
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)



    def move_dynamic_obstacle(self):
        # Step 1: Random sample new goals for required update dynamic obstacles
        # Check whether the current dynamic obstacles need new goals
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 # change to a new goal if less than the threshold
        
        # sample new goals in local range
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # apply local goal to the global range
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # clamp the range if out of the static env range
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles


        # Step 2: Random sample velocity for roughly every 2 seconds
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # Step 3: Calculate new position update for current timestep
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt


        # Step 4: Update Visualized Location in Simulation
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    def _in_stuck_front_region(self, rel_pos, lateral_inflation=None, vertical_inflation=None):
        forward = rel_pos[..., 0]
        lateral = rel_pos[..., 1].abs()
        vertical = rel_pos[..., 2].abs()
        horizontal_distance = rel_pos[..., :2].norm(dim=-1)

        if lateral_inflation is None:
            lateral_inflation = 0.0
        if vertical_inflation is None:
            vertical_inflation = 0.0

        return (
            (forward > 0.0)
            & (horizontal_distance <= self.stuck_front_distance + lateral_inflation)
            & (lateral <= self.stuck_front_tan * forward.clamp_min(1e-6) + lateral_inflation)
            & (vertical <= self.stuck_front_height + vertical_inflation)
        )

    def _compute_goal_progress_reward(self, goal_progress, reach_goal, time_limit):
        reward_goal_progress = self.goal_progress_scale * goal_progress
        reward_goal_progress = torch.where(
            time_limit,
            torch.full_like(reward_goal_progress, -self.goal_timeout_penalty),
            reward_goal_progress,
        )
        reward_goal_progress = torch.where(
            reach_goal,
            torch.full_like(reward_goal_progress, self.goal_arrival_reward),
            reward_goal_progress,
        )
        return reward_goal_progress

    def _compute_stall_reward(self, current_pos, current_vel, goal_progress, reach_goal):
        stall_displacement = (current_pos[..., :2] - self.stall_anchor_pos[..., :2]).norm(dim=-1)
        stall_speed = current_vel.norm(dim=-1)
        low_progress = goal_progress.abs() <= self.stall_progress_threshold
        low_motion = (stall_displacement <= self.stall_distance_threshold) & (stall_speed <= self.stall_speed_threshold)
        stall_candidate = low_motion & low_progress & (~reach_goal)

        self.stall_counter = torch.where(
            stall_candidate,
            self.stall_counter + 1,
            torch.zeros_like(self.stall_counter),
        )
        self.stall_anchor_pos = torch.where(
            stall_candidate.unsqueeze(-1),
            self.stall_anchor_pos,
            current_pos.clone(),
        )

        stall_active = self.stall_counter >= self.stall_reward_window
        stall_excess = (self.stall_counter - self.stall_reward_window).clamp(min=0).float()
        stall_scale = (stall_excess / float(self.stall_reward_ramp_window)).clamp(max=1.0)
        reward_stall = -self.stall_reward_weight * stall_scale
        return reward_stall, stall_active

    def _compute_vo_scale(self, dyn_obs_size):
        effective_radius_xy = dyn_obs_size[..., 0].unsqueeze(-1) / 2.0 + self.vo_xy_margin
        effective_radius_z = dyn_obs_size[..., 2].unsqueeze(-1) / 2.0 + self.vo_z_margin
        scale = torch.cat(
            [effective_radius_xy, effective_radius_xy, effective_radius_z],
            dim=-1,
        ).clamp_min(1e-6)
        return scale

    def _compute_vo_reward(self, dyn_obs_rpos, dyn_obs_vel, dyn_obs_size, dyn_obs_range_mask, drone_vel):
        drone_vel_expanded = drone_vel.expand(-1, dyn_obs_rpos.size(1), -1)
        # dyn_obs_rpos is obstacle position minus drone position, so its time derivative is v_obs - v_drone.
        rel_vel = dyn_obs_vel - drone_vel_expanded

        # Scale the 3D relative motion by an anisotropic safety ellipsoid.
        scale = self._compute_vo_scale(dyn_obs_size)

        scaled_rel_pos = dyn_obs_rpos / scale
        scaled_rel_vel = rel_vel / scale

        quad_a = (scaled_rel_vel * scaled_rel_vel).sum(dim=-1, keepdim=True)
        quad_b = 2.0 * (scaled_rel_pos * scaled_rel_vel).sum(dim=-1, keepdim=True)
        quad_c = (scaled_rel_pos * scaled_rel_pos).sum(dim=-1, keepdim=True) - 1.0
        discriminant = quad_b.square() - 4.0 * quad_a * quad_c

        valid_mask = (~dyn_obs_range_mask).unsqueeze(-1)
        approaching = quad_b < 0.0
        overlapping = quad_c <= 0.0
        discriminant_positive = discriminant >= 0.0
        moving = quad_a > 1e-6
        ttc = (-quad_b - torch.sqrt(discriminant.clamp_min(0.0))) / (2.0 * quad_a.clamp_min(1e-6))
        valid_ttc = valid_mask & approaching & discriminant_positive & moving & (~overlapping) & (ttc > 0.0) & (ttc <= self.vo_horizon)

        risk = torch.zeros_like(ttc)
        risk = torch.where(valid_ttc, torch.exp(-ttc / self.vo_tau), risk)
        risk = torch.where(valid_mask & overlapping, torch.ones_like(risk), risk)
        vo_risk = risk.max(dim=1, keepdim=True).values.squeeze(-1)

        if self.vo_warmup_steps > 0:
            warmup = min(float(self.vo_step_count) / float(self.vo_warmup_steps), 1.0)
        else:
            warmup = 1.0
        vo_warmup = torch.full_like(vo_risk, warmup)
        reward_vo = -self.vo_weight * vo_warmup * vo_risk
        return reward_vo, vo_risk, vo_warmup


    def _set_specs(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10

        # Observation Spec
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # Action Spec
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # number of motor
            })
        }).expand(self.num_envs).to(self.device)
        
        # Reward Spec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)

        # Done Spec
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device) 


        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "goal_distance": UnboundedContinuousTensorSpec(1),
            "reward_goal_progress": UnboundedContinuousTensorSpec(1),
            "penalty_safety_static": UnboundedContinuousTensorSpec(1),
            "penalty_safety_dynamic": UnboundedContinuousTensorSpec(1),
            "reward_vel": UnboundedContinuousTensorSpec(1),
            "penalty_height": UnboundedContinuousTensorSpec(1),
            "reward_stall": UnboundedContinuousTensorSpec(1),
            "stall": UnboundedContinuousTensorSpec(1),
            "stall_steps": UnboundedContinuousTensorSpec(1),
            "reward_vo": UnboundedContinuousTensorSpec(1),
            "vo_risk": UnboundedContinuousTensorSpec(1),
            "vo_warmup": UnboundedContinuousTensorSpec(1),
            "reach_goal": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "below_bound": UnboundedContinuousTensorSpec(1),
            "above_bound": UnboundedContinuousTensorSpec(1),
            "stuck": UnboundedContinuousTensorSpec(1),
            "stuck_steps": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _get_static_clearance_map(self, clearance_radius: float):
        if self.terrain_occupancy is None:
            return None
        clearance_radius = max(float(clearance_radius), 0.0)
        pixel_radius = int(np.ceil(clearance_radius / self.terrain_occupancy_resolution))
        if pixel_radius not in self.terrain_clearance_cache:
            kernel_size = 2 * pixel_radius + 1
            occupancy = self.terrain_occupancy.unsqueeze(0).unsqueeze(0)
            expanded = F.max_pool2d(occupancy, kernel_size=kernel_size, stride=1, padding=pixel_radius)
            self.terrain_clearance_cache[pixel_radius] = expanded[0, 0] > 0.5
        return self.terrain_clearance_cache[pixel_radius]

    def _points_have_static_clearance(self, points_xy: torch.Tensor, clearance_radius: float):
        clearance_map = self._get_static_clearance_map(clearance_radius)
        if clearance_map is None:
            return torch.ones(points_xy.shape[0], dtype=torch.bool, device=self.device)

        half_size = 0.5 * self.terrain_occupancy_size
        local_xy = points_xy + half_size.unsqueeze(0)
        valid = (
            (local_xy[:, 0] >= 0.0)
            & (local_xy[:, 0] <= self.terrain_occupancy_size[0])
            & (local_xy[:, 1] >= 0.0)
            & (local_xy[:, 1] <= self.terrain_occupancy_size[1])
        )
        if not torch.any(valid).item():
            return torch.ones(points_xy.shape[0], dtype=torch.bool, device=self.device)

        map_x = clearance_map.shape[0] - 1
        map_y = clearance_map.shape[1] - 1
        grid_xy = torch.round(local_xy / self.terrain_occupancy_resolution).long()
        grid_xy[:, 0] = grid_xy[:, 0].clamp_(0, map_x)
        grid_xy[:, 1] = grid_xy[:, 1].clamp_(0, map_y)
        occupied = clearance_map[grid_xy[:, 0], grid_xy[:, 1]]
        return (~valid) | (~occupied)

    def _sample_boundary_positions(self, count: int):
        masks = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float,
            device=self.device,
        )
        shifts = torch.tensor(
            [[0.0, 24.0, 0.0], [0.0, -24.0, 0.0], [24.0, 0.0, 0.0], [-24.0, 0.0, 0.0]],
            dtype=torch.float,
            device=self.device,
        )
        mask_indices = torch.randint(0, masks.size(0), (count,), device=self.device)
        selected_masks = masks[mask_indices].unsqueeze(1)
        selected_shifts = shifts[mask_indices].unsqueeze(1)

        positions = 48.0 * torch.rand(count, 1, 3, dtype=torch.float, device=self.device) - 24.0
        heights = 0.5 + torch.rand(count, dtype=torch.float, device=self.device) * 2.0
        positions[:, 0, 2] = heights
        return positions * selected_masks + selected_shifts

    def _sample_training_boundary_positions(self, count: int, clearance_radius: float):
        positions = torch.zeros(count, 1, 3, dtype=torch.float, device=self.device)
        remaining = torch.arange(count, device=self.device)
        fallback = None

        for _ in range(self.position_sample_attempts):
            if remaining.numel() == 0:
                break
            candidates = self._sample_boundary_positions(int(remaining.numel()))
            fallback = candidates
            valid = self._points_have_static_clearance(candidates[:, 0, :2], clearance_radius)
            if not torch.any(valid).item():
                continue
            accepted = remaining[valid]
            positions[accepted] = candidates[valid]
            remaining = remaining[~valid]

        if remaining.numel() > 0:
            if fallback is None or fallback.shape[0] != remaining.numel():
                fallback = self._sample_boundary_positions(int(remaining.numel()))
            positions[remaining] = fallback
        return positions

    
    def reset_target(self, env_ids: torch.Tensor):
        if (self.training):
            self.target_pos[env_ids] = self._sample_training_boundary_positions(
                env_ids.size(0), self.target_clearance_radius
            )

            # self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            # self.target_pos[:, 0, 1] = 24.
            # self.target_pos[:, 0, 2] = 2.    
        else:
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = -24.
            self.target_pos[:, 0, 2] = 2.            


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.reset_target(env_ids)
        if (self.training):
            pos = self._sample_training_boundary_positions(env_ids.size(0), self.spawn_clearance_radius)
            
            # pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            # pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            # pos[:, 0, 1] = -24.
            # pos[:, 0, 2] = 2.
        else:
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # Coordinate change: after reset, the drone's target direction should be changed
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # Coordinate change: after reset, the drone's facing direction should face the current goal
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.prev_drone_vel_w[env_ids] = 0.
        init_goal_distance = (self.target_pos[env_ids] - pos).norm(dim=-1)
        self.prev_goal_distance[env_ids] = init_goal_distance
        self.stuck_counter[env_ids] = 0
        self.stall_counter[env_ids] = 0
        self.stall_anchor_pos[env_ids] = pos
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])

        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] 
        if self.training:
            self.vo_step_count += 1
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    # get current states/observation
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state(env_frame=False) # (world_pos, orientation (quat), world_vel_and_angular, heading, up, 4motorsthrust)
        self.info["drone_state"][:] = self.root_state[..., :13] # info is for controller

        # >>>>>>>>>>>>The relevant code starts from here<<<<<<<<<<<<
        # -----------Network Input I: LiDAR range data--------------
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)
            .clamp_max(self.lidar_range)
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) # lidar scan store the data that is range - distance and it is in lidar's local frame

        # Optional render for LiDAR
        if self._should_render(0):
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            # set_camera_view(
            #     eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
            #     target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)                        
            # )
            v = (self.lidar.data.ray_hits_w[0] - x).reshape(*self.lidar_resolution, 3)
            # self.debug_draw.vector(x.expand_as(v[:, 0]), v[:, 0])
            # self.debug_draw.vector(x.expand_as(v[:, -1]), v[:, -1])
            self.debug_draw.vector(x.expand_as(v[:, 0])[0], v[0, 0])

        # ---------Network Input II: Drone's internal states---------
        # a. distance info in horizontal and vertical plane
        rpos = self.target_pos - self.root_state[..., :3]        
        distance = rpos.norm(dim=-1, keepdim=True) # start to goal distance
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)
        
        
        # b. unit direction vector to goal
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0

        rpos_clipped = rpos / distance.clamp(1e-6) # unit vector: start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d) # express in the goal coodinate
        stuck_dir_2d = rpos.clone()
        stuck_dir_2d[..., 2] = 0
        stuck_dir_2d = torch.where(stuck_dir_2d.norm(dim=-1, keepdim=True) > 1e-6, stuck_dir_2d, target_dir_2d)

        lidar_hit_rpos_w = self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)
        lidar_hit_rpos_g = vec_to_new_frame(lidar_hit_rpos_w, stuck_dir_2d)
        static_front_obstacle = self._in_stuck_front_region(lidar_hit_rpos_g).any(dim=1, keepdim=True)
        
        # c. velocity in the goal frame
        vel_w = self.root_state[..., 7:10] # world vel
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)   # coordinate change for velocity

        # final drone's internal states
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        if (self.cfg.env_dyn.num_obstacles != 0):
            # ---------Network Input III: Dynamic obstacle states--------
            # ------------------------------------------------------------
            # a. Closest N obstacles relative position in the goal frame 
            # Find the N closest and within range obstacles for each drone
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3] 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            dyn_obs_rpos_expanded_g_stuck = vec_to_new_frame(dyn_obs_rpos_expanded, stuck_dir_2d)
            dynamic_front_obstacle = self._in_stuck_front_region(
                dyn_obs_rpos_expanded_g_stuck,
                lateral_inflation=self.dyn_obs_size[:, 0].unsqueeze(0) / 2.,
                vertical_inflation=self.dyn_obs_size[:, 2].unsqueeze(0) / 2.,
            ).any(dim=1, keepdim=True)
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index for RL observation
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # relative distance of obstacles in the goal frame
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # exclude out of range obstacles
            closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. Velocity in the goal frame for the dynamic obstacles
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. Size of dynamic obstacles in category
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] # the acutal size

            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)
            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # convert to category: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # concatenate all for dynamic obstacles
            # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)
            dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, closest_dyn_obs_width_category, closest_dyn_obs_height_category], dim=-1).unsqueeze(1)

            # check dynamic obstacle collision for later reward
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # distance to dynamic obstacle for reward calculation (not 100% correct in math but should be good enough for approximation)
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range

            # Keep reward candidates aligned with the observed dynamic obstacles, then
            # re-rank only within that set using the 3D VO metric.
            closest_dyn_obs_distance_3d = closest_dyn_obs_rpos.norm(dim=-1)
            reward_dyn_obs_metric = torch.norm(
                closest_dyn_obs_rpos / self._compute_vo_scale(closest_dyn_obs_size),
                dim=2,
            )
            reward_dyn_obs_metric = reward_dyn_obs_metric.masked_fill(
                dyn_obs_range_mask,
                float("inf"),
            )
            reward_dyn_obs_metric = reward_dyn_obs_metric.masked_fill(
                closest_dyn_obs_distance_3d > self.vo_reward_range,
                float("inf"),
            )
            reward_topk = min(self.vo_reward_topk, reward_dyn_obs_metric.size(1))
            _, reward_dyn_obs_idx = torch.topk(reward_dyn_obs_metric, reward_topk, dim=1, largest=False)
            reward_dyn_obs_distance_3d = torch.gather(closest_dyn_obs_distance_3d, 1, reward_dyn_obs_idx)
            reward_dyn_obs_range_mask = torch.gather(dyn_obs_range_mask, 1, reward_dyn_obs_idx) | (reward_dyn_obs_distance_3d > self.vo_reward_range)
            reward_dyn_obs_rpos = torch.gather(closest_dyn_obs_rpos, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_dyn_obs_vel = torch.gather(closest_dyn_obs_vel, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_dyn_obs_size = torch.gather(closest_dyn_obs_size, 1, reward_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            reward_vo, vo_risk, vo_warmup = self._compute_vo_reward(
                reward_dyn_obs_rpos,
                reward_dyn_obs_vel,
                reward_dyn_obs_size,
                reward_dyn_obs_range_mask,
                vel_w,
            )
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            dynamic_front_obstacle = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)
            closest_dyn_obs_distance_reward = torch.full(
                (self.num_envs, self.cfg.algo.feature_extractor.dyn_obs_num),
                self.lidar_range,
                device=self.cfg.device,
            )
            reward_vo = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            vo_risk = torch.zeros(self.num_envs, 1, device=self.cfg.device)
            vo_warmup = torch.zeros(self.num_envs, 1, device=self.cfg.device)

        goal_distance = distance.squeeze(-1)
        reach_goal = goal_distance <= self.goal_radius
        time_limit = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        goal_progress = self.prev_goal_distance - goal_distance
        reward_goal_progress = self._compute_goal_progress_reward(goal_progress, reach_goal, time_limit)
        reward_stall, stall_active = self._compute_stall_reward(
            self.root_state[..., :3],
            self.drone.vel_w[..., :3],
            goal_progress,
            reach_goal,
        )
        front_obstacle = static_front_obstacle | dynamic_front_obstacle
        small_progress_with_obstacle = (goal_progress <= self.stuck_progress_eps) & front_obstacle
        self.stuck_counter = torch.where(
            small_progress_with_obstacle,
            self.stuck_counter + 1,
            torch.zeros_like(self.stuck_counter),
        )
        stuck = self.stuck_counter >= self.stuck_window
        self.prev_goal_distance = goal_distance.clone()
            
        # -----------------Network Input Final--------------
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }
####

        # -----------------Reward Calculation-----------------
        # a. penalize static obstacles only when they get too close.
        dist_static = self.lidar_range - self.lidar_scan
        safe_margin = 1.1
        penalty_safety_static = torch.relu(safe_margin - dist_static) / safe_margin
        penalty_safety_static = penalty_safety_static.mean(dim=(2, 3))
        

        # b. penalize dynamic obstacles only when they enter the same safety margin.
        penalty_safety_dynamic = torch.relu(safe_margin - closest_dyn_obs_distance_reward) / safe_margin
        penalty_safety_dynamic = penalty_safety_dynamic.mean(dim=-1, keepdim=True)

        # c. reward forward progress only in the horizontal plane.
        goal_direction_xy = rpos[..., :2] / distance_2d.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :2] * goal_direction_xy).sum(-1)
        
        # d. smoothness reward for action smoothness
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. softly keep altitude close to the goal altitude.
        target_z = self.target_pos[..., 2]
        z_error = self.drone.pos[..., 2] - target_z
        penalty_height = torch.relu(z_error.abs() - 0.4).square()


        # f. Collision condition with its penalty
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision | dynamic_collision
        below_bound = self.drone.pos[..., 2] < 0.2
        above_bound = self.drone.pos[..., 2] > 4.
        
        # Final reward calculation
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = reward_vel*0.05 + 0.1 - penalty_safety_static * 0.5 - penalty_safety_dynamic * 0.6 - penalty_smooth * 0.1 - penalty_height * 0.5
        else:
            self.reward = reward_vel*0.05 + 0.1 - penalty_safety_static * 0.5 - penalty_smooth * 0.1 - penalty_height * 0.5
        self.reward = self.reward + reward_goal_progress + reward_stall + reward_vo

        # Terminal penalties make failure modes explicitly costly.
        self.reward[collision] -= 45.0
        self.reward[below_bound] -= 20.0
        self.reward[above_bound] -= 20.0

        # Terminate Conditions
        self.terminated = reach_goal | below_bound | above_bound | collision
        self.truncated = time_limit # progress buf is to track the step number

        # update previous velocity for smoothness calculation in the next ieteration
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # # -----------------Training Stats-----------------
        self.stats["return"] += self.reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["goal_distance"] = goal_distance
        self.stats["reward_goal_progress"] = reward_goal_progress
        self.stats["penalty_safety_static"] = penalty_safety_static
        self.stats["penalty_safety_dynamic"] = penalty_safety_dynamic
        self.stats["reward_vel"] = reward_vel
        self.stats["penalty_height"] = penalty_height
        self.stats["reward_stall"] = reward_stall
        self.stats["stall"] = torch.maximum(self.stats["stall"], stall_active.float())
        self.stats["stall_steps"] += stall_active.float()
        self.stats["reward_vo"] = reward_vo
        self.stats["vo_risk"] = vo_risk
        self.stats["vo_warmup"] = vo_warmup
        self.stats["reach_goal"] = reach_goal.float()
        self.stats["collision"] = collision.float()
        self.stats["below_bound"] = below_bound.float()
        self.stats["above_bound"] = above_bound.float()
        self.stats["stuck"] = torch.maximum(self.stats["stuck"], stuck.float())
        self.stats["stuck_steps"] += stuck.float()
        self.stats["truncated"] = self.truncated.float()

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
