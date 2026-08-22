# LP-Nav Gazebo Corridor Benchmark

This benchmark is independent of the Isaac Sim training environment. The default
scenario contains a 50 m x 5 m enclosure, eight non-overflyable square prisms,
and four pedestrians following reproducible random loops within the enclosure.
Pedestrian paths may cross static obstacles and one another, and the Gazebo scene
grid is disabled for a cleaner overhead view.
The evaluator restricts the UAV center to 0.35--1.50 m altitude; together with
the configured collision radius, both the prisms and pedestrians are
non-overflyable.

## Generate scenarios

Generate the default scenario:

```bash
rosrun uav_simulator generate_corridor_world.py
```

The generator requires the ROS `python3-yaml` package.

Generate another reproducible layout:

```bash
rosrun uav_simulator generate_corridor_world.py --seed 42
```

Each generation produces matching `.world`, `.pcd`, and `.yaml` files in this
directory. Keep all three files together. Configuration is in
`uav_simulator/scripts/corridor_world_generator.yaml`.

## Run the benchmark

Terminal 1 starts Gazebo, online mapping, the fake dynamic-obstacle detector,
the VO shield, and RViz:

```bash
roslaunch navigation_runner lpnav_corridor_sim.launch \
  scenario_name:=lpnav_corridor_seed_7
```

This corridor launch enables global occupancy-map visualization so the final
RViz view can show the complete accumulated perceived map rather than only the
local area around the UAV.

Terminal 2 starts the policy using the checkpoint selected by
`navigation_runner/scripts/navigation.py`:

```bash
conda activate NavRL
rosrun navigation_runner navigation_node.py
```

Terminal 3 starts the dedicated evaluator and pedestrian controller:

```bash
rosrun navigation_runner deployment_eval2.py \
  _scenario_file:=$(rospack find uav_simulator)/worlds/lpnav_corridor/lpnav_corridor_seed_7.yaml \
  _num_trials:=100 \
  _random_seed:=1007 \
  _csv_path:=/tmp/lpnav_corridor_lpnav.csv \
  _keep_trajectory_publisher_alive:=true
```

The evaluator waits for the navigation goal subscriber before starting. It also
resets the pedestrian phases deterministically for every trial. Use the same
scenario and random seed for every compared method.

After all trials finish, leave the evaluator terminal running and open the
final corridor environment and all trajectories in another terminal:

```bash
roslaunch navigation_runner lpnav_corridor_eval_rviz.launch
```

The final RViz configuration displays `/occupancy_map/inflated_voxel_map` with
the same Z-axis rainbow coloring as the live perception view. The ground-truth
environment marker is available as a disabled fallback display. A one-trial
test only shows the portion perceived during that trial; use the complete trial
set to accumulate the perceived map, or enable `Ground-truth Environment
(fallback)` in RViz to show the full height-colored scenario immediately.

The evaluator publishes latched marker arrays on
`/deployment_eval/trajectories` and `/deployment_eval/environment`. The static
environment is loaded from the `.pcd` file matching the scenario YAML and is
rendered as height-colored voxels, matching the live occupancy-map rainbow
style. To return to all trajectories after selecting a subset, publish an empty
selection:

```bash
rostopic pub -1 /deployment_eval/trajectory_selection \
  std_msgs/Int32MultiArray "data: []"
```

Keep `deployment_eval2.py` alive until RViz has received the markers; press
Ctrl-C in the evaluator terminal only after visualization is no longer needed.

To let the launch file start the evaluator, set `run_evaluator:=true`; start the
navigation node in another terminal before the 120-second dependency timeout.

## Mapping modes

The default `use_prebuilt_map:=false` builds the static map online from the
simulated depth camera. This is the preferred cross-simulator test.

For a controlled policy-only diagnostic, use the matching generated PCD:

```bash
roslaunch navigation_runner lpnav_corridor_sim.launch \
  scenario_name:=lpnav_corridor_seed_7 \
  use_prebuilt_map:=true
```

Pedestrians move only while `deployment_eval2.py` is running. Their Gazebo model
names retain the `personN_0.5_0.5_1.8` convention required by the fake detector.
