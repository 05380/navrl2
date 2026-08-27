# SU17 + MID-360：NavRL 方案 B 第一阶段部署说明

## 当前结论

这台 SU17 具备第一阶段部署条件，但当前交付只允许以**影子模式**运行。影子模式会完成点云建图、144 条虚拟激光、CPU 策略推理和安全限幅，但不会向 `/uav1/prometheus/command` 发布任何指令。

已确认的数据链：

- `/uav1/cloud_mid360_body`：`sensor_msgs/PointCloud2`，约 10 Hz，实测每帧约 4400–5200 点；
- `/uav1/mavros/local_position/odom`：`nav_msgs/Odometry`，约 30 Hz，`world -> base_link`；
- `/uav1/Odometry`：FAST-LIO 里程计，约 10 Hz；
- `/uav1/prometheus/state`：约 50 Hz；
- `/uav1/prometheus/control_state`：约 100 Hz；
- 实机 CPU 为 i7-1165G7、15 GiB 内存，无 NVIDIA GPU，适合 CPU 推理；
- 模型输入不是 RGB-D 图像，而是地图生成的 `36 × 4 = 144` 条虚拟激光，训练量程固定为 4 m；
- 第一阶段将 `5 × 10` 动态障碍物状态置零。移动物体仍会以点云占据物的形式触发反应式避让，但此阶段不预测其速度和未来轨迹。

## 两个旧问题的最终结论

### 1. `setpoint_position/local`：已经解决，不再是部署条件

实体机没有 `/uav1/mavros/setpoint_position/local` 不会阻塞这版程序。检查原机 Prometheus 控制器源码后已经确认，它本来就是向 `/uav1/mavros/setpoint_raw/local` 发布 `mavros_msgs/PositionTarget`；实机存在这个话题，与控制器实现一致。

本次新代码取消了旧导航程序对 `PoseStamped -> setpoint_position/local` 的依赖：

- 不再自动解锁、自动起飞，起飞和降落继续使用原机人工流程；
- NavRL 只生成私有话题 `/uav1/navrl/desired_setpoint`；
- 影子模式完全不注册 `/uav1/prometheus/command` 发布器；
- 以后主动模式下，航向、速度和悬停请求均转换成 `prometheus_msgs/UAVCommand`；
- 原机 Prometheus 再负责将 Move、yaw 和 `Current_Pos_Hover` 转成 `/uav1/mavros/setpoint_raw/local`。

因此不用安装 `setpoint_position` MAVROS 插件，也不用修改 MAVROS。部署前只需确认原控制链仍正常：

```bash
rostopic type /uav1/mavros/setpoint_raw/local
rostopic info /uav1/mavros/setpoint_raw/local
```

正常结果应为 `mavros_msgs/PositionTarget`，并能看到原 Prometheus 控制节点作为 publisher。`setpoint_position/local` 的两个检查命令可以从本项目的阻塞项中删除。

仓库中保留的旧仿真脚本 `scripts/navigation.py` 仍包含原来的 `setpoint_position/local` publisher，用于兼容旧实验，但 `su17_phase1.launch` 不会安装或启动它。实机只运行本文给出的 `su17_phase1.launch`/影子启动脚本，不要手动运行旧 `navigation.py` 或旧仿真 launch。

### 2. MID-360 外参与物理碰撞包络：已经确认

`rosparam get /uav_control_main_1/Lidar` 返回“未设置”，说明旧配置中的 `z=0.1、pitch=0.26` 没有加载到当前控制节点；它不能作为实机运行值或物理测量值。但结合 Livox 驱动、FAST-LIO 源码与实际配置，已经能够确认本项目订阅的 `/uav1/cloud_mid360_body` 在发布前完成了原始安装角和雷达到 IMU/机体系的处理。因此，当前“body 点云 + 由 FAST-LIO 融合得到的 MAVROS odom”组合在地图端必须保持附加变换全零，不能再套用“俯仰 15°”或“x=0.13、z=0.23”这两组原始安装值。

机身相对 MID-360 **雷达原点**的六向包络已经实测：前 0.22 m、后 0.22 m、左 0.25 m、右 0.25 m、上 0.11 m、下 0.20 m（`x 前、y 左、z 上`）。CropBox 使用的却是 `/uav1/cloud_mid360_body` 的 IMU/body 原点，因此还要使用 FAST-LIO 的 `extrinsic_T=[-0.011,-0.02329,0.04412] m` 换算。当前 Prometheus 把 FAST-LIO 位姿直接送入 PX4，所以 MAVROS 返回的数值坐标与这套 IMU/body 位姿链一致，地图端不应再额外补物理安装平移。

### 点云外参结论

不要被消息的 `frame_id: uav1/lidar_link` 误导。实体机 FAST-LIO 的 `publish_frame_body()` 在发布 `/uav1/cloud_mid360_body` 前调用 `RGBpointBodyLidarToIMU()`；其配置为：

```text
extrinsic_T = [-0.011, -0.02329, 0.04412]
extrinsic_R = identity
scan_bodyframe_pub_en = true
```

因此这个话题里的点已经处于 IMU/机体系，地图端必须使用单位外参：

```text
lidar_x/y/z/roll/pitch/yaw = 0
```

只有以后改订阅原始雷达坐标系点云时，才填写非零的 body-to-lidar 外参。重复施加 FAST-LIO 外参会造成地图偏移。

### `mid360_level_check.bag` 离线核验结果

2026-08-25 对开发机上当前这份 `/Users/yoloflps/Desktop/mid360_level_check.bag` 进行了逐帧解析。该文件实际时长 16.69 s，包含 170 帧 `/uav1/livox/lidar`、170 帧 `/uav1/cloud_mid360_body` 和 498 帧 `/uav1/mavros/local_position/odom`。结论如下：

- 飞机在采集期间基本静止：位置三轴跨度约 3.1/2.4/9.0 mm，线速度 95 分位约 0.0013 m/s；
- 原始点云和 body 点云拟合地面/天花板后，倾角均只有约 0.62–0.68°；人为再加正负 15° 后误差会变成约 14.4–15.6°。因此照片里可见的物理倾角已被当前数据链补偿，NavRL 对 `/uav1/cloud_mid360_body` 继续使用全零附加旋转；
- body 点云相对原始点云的上下平面平移约 4.4–4.6 cm，与 FAST-LIO 配置的 `extrinsic_T.z=0.04412 m` 一致；
- body 点云与最近 MAVROS odom 的时间戳差在稳定段中位数约 15.83 ms、95 分位约 16.04 ms。170 帧中有 4 帧超过 50 ms，均位于录包首尾缺少配对消息的边界；
- 旧 mapper 风格的宽 CropBox `[-0.60,-0.60,-0.20] → [0.60,0.60,0.65]` 每帧中位数剔除约 384 点，占有效近远距点约 8.79%，会滤掉大量实际机体盒外的近点；
- 将雷达原点测量换算到 IMU/body 原点并给自滤盒每面增加 0.02 m 后，新 CropBox `[-0.26,-0.30,-0.18] → [0.23,0.25,0.18]` 在整包 745753 个有效近远距点中只剔除 83 点（约 0.01113%，每帧中位数为 0）。这与 MID-360/FAST-LIO 已有约 0.3–0.35 m 盲区/近距过滤相符，也说明不应继续使用旧的 1.2 m 宽盒子。

这份包没有 `/uav1/Odometry`，所以它确认了安装倾角、body 点云平移、近点分布和点云/里程计同步裕量，但不能完成两套里程计的运动漂移核验。

## 正式控制前新增的两项地图保护

### 1. 可配置机身 CropBox

`map_manager` 会先用 `body_to_sensor` 把输入点变换到 IMU/body 机体系，在机体系中剔除机身盒内的点，再执行 0.35–5.0 m 距离过滤。雷达原点下实测物理包络是：

```text
physical_min_lidar = [-0.22, -0.25, -0.20]
physical_max_lidar = [ 0.22,  0.25,  0.11]
```

FAST-LIO 源码执行 `p_imu = R * p_lidar + T`，当前 `R=identity`、`T=[-0.011,-0.02329,0.04412]`，所以精确换算后的 IMU/body 包络为：

```text
physical_min_body = [-0.23100, -0.27329, -0.15588]
physical_max_body = [ 0.20900,  0.22671,  0.15412]
```

自滤盒只增加 0.02 m 余量并按厘米向外取整，避免过度删除真实障碍：

```yaml
self_filter_enabled: true
self_filter_min: [-0.26, -0.30, -0.18]
self_filter_max: [ 0.23,  0.25,  0.18]
```

这比把 `point_cloud_min_range` 直接提高到 0.6 m 更合适：盒外距离雷达不足 0.6 m 的真实障碍仍会保留。盒子始终按机体系定义；当前 `/uav1/cloud_mid360_body` 的附加外参为单位阵，若以后更换原始点云话题，必须先给出正确的 `body_to_sensor`。启动日志每 5 s 输出一次 `kept/self/range/invalid` 数量，可用来发现盒子过大、过小或点云异常。

自滤盒内部对地图不可见，因此主动控制时碰撞膨胀体必须完整包住 CropBox。代码会检查：

```text
robot_x / 2 >= max(abs(self_filter_min_x), abs(self_filter_max_x))
robot_y / 2 >= max(abs(self_filter_min_y), abs(self_filter_max_y))
robot_z / 2 >= max(abs(self_filter_min_z), abs(self_filter_max_z))
```

碰撞膨胀比自滤盒保守：在精确 IMU/body 包络每个方向增加至少 0.05 m，再对称化并向外取整，最终设置为 `robot_size=[0.58,0.66,0.42]`。它完整包住 CropBox，同时保留比自滤盒更大的碰撞余量。默认配置现在能通过碰撞体/CropBox 一致性检查，但这只解除“尺寸未知”这一项阻塞，不代表可以跳过影子测试和两套里程计运动一致性检查。

### 2. 50 ms 同步硬上限和时间差日志

点云与 pose/odom 的 `ApproximateTime` 同步器现在设置：

```yaml
point_cloud_sync_max_interval: 0.05
point_cloud_sync_log_interval: 5.0
```

只有 header 时间戳差不超过 50 ms 的消息对才会进入地图；回调内还有同样的防御性检查。正常配对时间差会按配置节流输出，单位为 ms。大包稳定段约 16 ms，因而当前 50 ms 上限有约 34 ms 裕量；若地图更新频率明显低于点云 10 Hz，同时没有时间差日志，应检查时钟源、消息 header 和里程计延迟，而不是先放宽上限。

## 第一阶段数据流和安全边界

```text
MID-360 body cloud + MAVROS world odom
                 |
                 v
        0.1 m occupancy map
                 |
                 v
       36 x 4 virtual raycast (4 m)
                 |
                 v
       NavRL CPU policy, dynamic=zero
                 |
                 v
 /uav1/navrl/desired_setpoint       <- 策略私有输出
                 |
                 v
      watchdog + clamp + fence
                 |
        +--------+---------+
        |                  |
  shadow mode          active mode
        |                  |
 safe_setpoint only    Prometheus UAVCommand
```

代码不会解锁、起飞、切 PX4 模式，也不直接向 MAVROS setpoint 发布。未来启用输出后仍由原机 Prometheus 控制器负责 RC 状态机、OFFBOARD、飞控下发和失效保护。

首轮限值：水平 0.50 m/s、垂直 0.30 m/s、高度 0.30–1.50 m、启动点水平半径 3 m。任一输入过期、定位无效、Prometheus failsafe、控制器不匹配、策略输出非法或越界时，桥接器请求 `Current_Pos_Hover`。

## 正式控制前仍需完成的三项核验

1. **核对测量起点和方向**：已按“从 MID-360 雷达原点起量、`x 前、y 左、z 上`”记录并完成 IMU/body 换算；需要确认实际测量不是从机身几何中心、雷达外壳边缘或圆顶最高点起量。
2. **两套里程计运动时是否保持对齐**：`/uav1/Odometry` 和 `/uav1/mavros/local_position/odom` 可以有固定原点差，但位置差和 yaw 差不能随运动漂移。影子测试时设置 `enable_odom_consistency_check:=true`，桥接器会记录初始差值并监控变化。
3. **验证新 CropBox 不过滤盒外障碍**：代码已使用 `[-0.26,-0.30,-0.18]` 到 `[0.23,0.25,0.18]`。静止场景运行 `rosrun navigation_runner inspect_mid360_cloud.py` 并观察 RViz/过滤统计；再用纸板从六个方向缓慢接近，确认纸板在盒外能进入地图。不能直接把最小距离提高到 0.6 m，也不能把紧急停车距离调小来掩盖自点。

当前已分析的大包只有 MAVROS odom，没有同时包含 `/uav1/Odometry`，因此两套里程计运动漂移仍需在影子测试中核验。

## 机载电脑是否需要改原代码

不需要修改 `/home/amov/su17_experiment`、MAVROS、FAST-LIO 或 PX4，也不需要在机载电脑上重新手写 Python/C++。需要新增的地图、策略、安全桥和检查代码都已经放在 `ros1 real` 的三个 ROS 包里；机载电脑只做四件事：

1. 复制本交付文件；
2. 安装 CPU 版 PyTorch 和 Python 依赖；
3. 在独立的 `/home/amov/navrl_ws` overlay 中编译；
4. 先运行固定为 `output_enabled=false` 的影子模式。

交付中另有两个机载脚本：

- `tools/setup_su17_phase1_onboard.sh`：检查路径、创建 overlay/venv、安装依赖、编译并运行环境自检；
- `tools/run_su17_phase1_shadow.sh`：按正确顺序 source 环境并启动影子模式，脚本会拒绝任何 `output_enabled:=...` 参数。

## 将代码放入机载电脑

建议保留原有 `/home/amov/su17_experiment`，单独建立 overlay 工作空间，避免覆盖实体机已经能飞的代码。

### 推荐方法：复制已经打好的部署包

开发机执行：

```bash
scp "/Users/yoloflps/Downloads/study/navrl2/su17_phase1_onboard_bundle.tar.gz" \
  amov@192.168.2.97:/home/amov/
```

机载电脑执行：

```bash
mkdir -p /home/amov/navrl2
tar -xzf /home/amov/su17_phase1_onboard_bundle.tar.gz \
  -C /home/amov/navrl2

sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential ninja-build \
  ros-noetic-vision-msgs

bash /home/amov/navrl2/ros1_real/navigation_runner/tools/setup_su17_phase1_onboard.sh
```

安装脚本默认使用 `/home/amov/su17_experiment/devel/setup.bash`。若原机 Prometheus 实际在别处，使用下面的方式指定，不要移动或覆盖原工作空间：

```bash
PROMETHEUS_SETUP=/实际路径/devel/setup.bash \
  bash /home/amov/navrl2/ros1_real/navigation_runner/tools/setup_su17_phase1_onboard.sh
```

### 备选方法：使用 rsync

从开发机只复制三个 ROS 包和训练时使用的 TensorDict/TorchRL 源码，不需要复制 82 MB 的仿真器或完整 Isaac 工程：

```bash
ssh amov@192.168.2.97 \
  'mkdir -p /home/amov/navrl2/ros1_real /home/amov/navrl2/isaac-training/third_party'

rsync -av "/Users/yoloflps/Downloads/study/navrl2/ros1 real/map_manager" \
  "/Users/yoloflps/Downloads/study/navrl2/ros1 real/onboard_detector" \
  "/Users/yoloflps/Downloads/study/navrl2/ros1 real/navigation_runner" \
  amov@192.168.2.97:/home/amov/navrl2/ros1_real/

rsync -av "/Users/yoloflps/Downloads/study/navrl2/isaac-training/third_party/tensordict" \
  "/Users/yoloflps/Downloads/study/navrl2/isaac-training/third_party/rl" \
  amov@192.168.2.97:/home/amov/navrl2/isaac-training/third_party/
```

若使用 rsync，则在机载电脑手动执行：

```bash
mkdir -p /home/amov/navrl_ws/src
ln -s /home/amov/navrl2/ros1_real/map_manager \
  /home/amov/navrl_ws/src/map_manager
ln -s /home/amov/navrl2/ros1_real/onboard_detector \
  /home/amov/navrl_ws/src/onboard_detector
ln -s /home/amov/navrl2/ros1_real/navigation_runner \
  /home/amov/navrl_ws/src/navigation_runner

source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
rospack find prometheus_msgs
```

`rospack find prometheus_msgs` 必须成功。如果实体机实际 Prometheus 工作空间不在 `/home/amov/su17_experiment`，把第二个 `source` 改成 `rospack find prometheus_msgs` 能成功的现有工作空间。先完成下一节的 venv，再编译 overlay，确保 ROS 启动的 Python 节点确实使用含 PyTorch 的解释器。

## 安装 CPU 推理环境

如果已经成功运行推荐方法中的 `setup_su17_phase1_onboard.sh`，本节和后面的手动 `catkin_make` 命令可以跳过；下面保留的是脚本执行内容，便于定位安装失败原因。

工程自带的 `isaac-training/setup_deployment.sh` 表明模型原环境使用 PyTorch 2.0.1，并使用仓库内的 TensorDict/TorchRL 0.4.0。ROS Noetic 使用 Python 3.8，因此创建带系统 ROS 包的 venv：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential ninja-build \
  ros-noetic-vision-msgs

python3 -m venv --system-site-packages /home/amov/navrl_venv
source /home/amov/navrl_venv/bin/activate
python -m pip install --upgrade "pip<25" setuptools wheel ninja

python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.0.1

python -m pip install -r \
  /home/amov/navrl2/ros1_real/navigation_runner/requirements_su17_phase1.txt

python -m pip install --no-build-isolation --no-deps -e \
  /home/amov/navrl2/isaac-training/third_party/tensordict
python -m pip install --no-build-isolation --no-deps -e \
  /home/amov/navrl2/isaac-training/third_party/rl
```

使用 `--no-build-isolation --no-deps` 是为了直接使用已安装的 CPU PyTorch 2.0.1，并防止第三方包元数据自动升级 PyTorch。无需安装 CUDA、torchvision 或 torchaudio。

现在用 venv 的 Python 编译 overlay：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
source /home/amov/navrl_venv/bin/activate
cd /home/amov/navrl_ws
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/home/amov/navrl_venv/bin/python3
source /home/amov/navrl_ws/devel/setup.bash
```

每次启动前的 source 顺序：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
source /home/amov/navrl_venv/bin/activate
source /home/amov/navrl_ws/devel/setup.bash
```

离线检查依赖和 checkpoint：

```bash
rosrun navigation_runner check_su17_phase1_env.py
```

原机实时雷达启动后，另开终端按同样顺序 source 环境，再检查近距离点云：

```bash
rosrun navigation_runner inspect_mid360_cloud.py
```

期望最后一行：

```text
SU17 phase-1 environment: OK
```

## 第 0 步：回放现有 bag，不连接控制输出

先停止实时雷达/控制启动项，避免 live topic 和 bag topic 混在一起。终端 A：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
source /home/amov/navrl_venv/bin/activate
source /home/amov/navrl_ws/devel/setup.bash
rosparam set use_sim_time true
roslaunch navigation_runner su17_phase1.launch \
  output_enabled:=false \
  enable_odom_consistency_check:=true \
  map_visualization:=true
```

终端 B：

```bash
rosbag play --clock --pause /home/amov/mid360_navrl_check.bag
```

按空格开始播放。另一个终端检查：

```bash
rostopic echo /uav1/navrl/navigation_status
rostopic echo /uav1/navrl/bridge_status
rostopic hz /occupancy_map/update
rostopic hz /occupancy_map/voxel_map
rosservice type /occupancy_map/raycast
```

发送一个仅供计算的目标。把 x/y/z 改为 bag 当前位姿附近、且 z 在 0.30–1.50 m 内的点：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  "header: {frame_id: 'world'}
pose:
  position: {x: 1.0, y: 0.0, z: 1.0}
  orientation: {w: 1.0}"
```

检查策略输出和安全限幅输出：

```bash
rostopic echo /uav1/navrl/desired_setpoint
rostopic echo /uav1/navrl/safe_setpoint
rostopic info /uav1/prometheus/command
```

影子模式下，`/uav1/prometheus/command` 的 publisher 列表中不能出现 `/navrl_su17_bridge`。bag 只有约 9.4 秒，目标和检查命令要在播放窗口内执行，也可以反复回放。

测试结束恢复实时 ROS 时间：

```bash
rosparam set use_sim_time false
```

## 第 1 步：实体机影子模式

1. 使用原 SU17 流程启动雷达、FAST-LIO、MAVROS 和 Prometheus；
2. 不启动 EGO planner 或其他会向 `/uav1/prometheus/command` 发布 Move 的节点；
3. 启动本 launch，保持 `output_enabled:=false`；
4. 先拆桨或保持未解锁，验证地图方向；
5. 再由人工按原机流程起飞悬停，NavRL 仍只计算、不控机，验证两套里程计运动对齐和策略输出。

从已采集的实机进程记录看，原系统分别由下面三个 launch 启动。如果地面站/开机脚本已经让相关话题正常发布，就跳过这些命令，不能重复启动同名节点；否则可在三个终端中执行：

```bash
# 终端 1：MAVROS + Prometheus 控制/状态
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment su17_onboard.launch
```

```bash
# 终端 2：MID-360 驱动
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment msg_MID360.launch
```

```bash
# 终端 3：FAST-LIO
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment mapping_mid360_y.launch
```

启动 NavRL 前先确认，五项都有输出才继续：

```bash
timeout 6 rostopic hz /uav1/cloud_mid360_body
timeout 6 rostopic hz /uav1/Odometry
timeout 6 rostopic hz /uav1/mavros/local_position/odom
timeout 6 rostopic hz /uav1/prometheus/state
timeout 6 rostopic hz /uav1/prometheus/control_state
```

然后在终端 4 启动影子模式：

```bash
bash /home/amov/navrl2/ros1_real/navigation_runner/tools/run_su17_phase1_shadow.sh \
  robot_x:=0.58 robot_y:=0.66 robot_z:=0.42
```

也可以不用脚本，先按上文顺序 source 四个环境，再执行等价命令：

```bash
roslaunch navigation_runner su17_phase1.launch \
  output_enabled:=false \
  enable_odom_consistency_check:=true \
  map_visualization:=true \
  robot_x:=0.58 robot_y:=0.66 robot_z:=0.42
```

启动后先核对本次新增参数确实进入了地图节点：

```bash
rosparam get /occupancy_map/self_filter_enabled
rosparam get /occupancy_map/self_filter_min
rosparam get /occupancy_map/self_filter_max
rosparam get /occupancy_map/point_cloud_sync_max_interval
rosparam get /occupancy_map/point_cloud_sync_log_interval
```

预期依次得到 `true`、两个三元素数组、`0.05` 和 `5.0`。另开终端查看节流日志和地图心跳：

```bash
rostopic echo /rosout | grep -E 'timestamp delta|Point-cloud filter|Rejecting pointcloud'
timeout 10 rostopic hz /occupancy_map/update
```

正常时日志中的 `timestamp delta` 应接近 16 ms，`/occupancy_map/update` 接近 10 Hz。launch 已默认使用实测值；如需显式覆盖，可追加以下参数，改完必须重新启动节点才生效：

```bash
self_filter_min_x:=-0.26 self_filter_min_y:=-0.30 self_filter_min_z:=-0.18 \
self_filter_max_x:=0.23 self_filter_max_y:=0.25 self_filter_max_z:=0.18 \
robot_x:=0.58 robot_y:=0.66 robot_z:=0.42
```

影子模式通过标准：

- 静止时地图墙面/地面不随 yaw 旋转漂移；
- `/uav1/navrl/bridge_status` 能到 `shadow_ready`；
- 飞行移动后不出现 `shadow_odom_inconsistent`；
- raycast 始终返回 432 个浮点数，即 144 个三维端点；
- `/occupancy_map/update` 稳定接近点云的 10 Hz；这个心跳中断时策略必须停止输出；
- 策略循环稳定约 10 Hz，日志中的 inference/control 显著小于 100 ms；
- 目标在前方且无障碍时，`safe_setpoint` 的水平速度方向正确；
- 人或纸板进入 4 m 范围时，对应虚拟激光距离缩短，策略减速或改向；
- 停点云、停里程计、停止目标输出后，状态分别进入 stale/hold 路径；
- 全程 `/navrl_su17_bridge` 不出现在 Prometheus command 的 publisher 列表。

首次检查建议同时观察负载：

```bash
top -H -p $(pgrep -d, -f 'navigation_su17_phase1|occupancy_map_node')
```

完成地图目视确认后，可用 `map_visualization:=false` 降低 CPU 和 ROS 网络负载，raycast 与策略仍正常工作。

结束 NavRL 时只在终端 4 按 `Ctrl-C`。影子模式不会发布飞行指令，也不会替你执行起飞、降落或急停；这些动作仍全部使用原 SU17/遥控器流程。

## 未来小范围系留飞行

只有上述项目全部通过、实际机体尺寸已填写、CropBox 已缩到实测机体且通过“碰撞体包住自滤盒”检查、场地已清空并准备好 RC 人工接管后，才讨论：

```text
output_enabled:=true
```

启用后仍不自动解锁或起飞。人工先按原 SU17 流程进入稳定悬停，再切入 `COMMAND_CONTROL`。桥接器只在 `COMMAND_CONTROL + PX4_ORIGIN + connected + odom_valid + no failsafe` 时转发 `XYZ_VEL`；否则不发送 Move，已处于控制状态时则请求当前位置悬停。

第一阶段不应在人群、狭窄室内或高速动态障碍场景使用。当前 `navigation_su17_phase1.py` 明确将模型的 `5 × 10` 动态状态置零，移动物体只能作为“此刻点云中的占据物”触发反应式减速或绕行。mapper 的消失占据清理（且现配置未启用 `dynamic_environment`）既不估计目标速度，也不预测轨迹，不能作为动态预测能力的证据；真正的目标检测/跟踪、速度估计和未来轨迹输入属于方案 B 第二阶段。

## 本次新增或修改的核心文件

- `launch/su17_phase1.launch`：默认影子模式的一键启动；
- `scripts/navigation_su17_phase1.py`：CPU 策略推理与 144-ray 观测构建；
- `scripts/navrl_su17_bridge.py`：Prometheus 安全桥、watchdog、限幅和围栏；
- `scripts/check_su17_phase1_env.py`：依赖与 checkpoint 自检；
- `tools/setup_su17_phase1_onboard.sh`：机载 overlay、venv、依赖和编译脚本；
- `tools/run_su17_phase1_shadow.sh`：禁止控制输出的实体机影子启动入口；
- `cfg/mapping/real/su17_mid360.yaml`：MID-360 实机地图配置；
- `map_manager/occupancyMap.*`：点云无效值/近远距/CropBox 过滤、50 ms 同步上限与日志、显式坐标变换、边界修复和可关闭的高负载可视化。


目前进度：安装、编译、实时数据、地图同步、CropBox和影子模式都已通过；尚未完成六向纸板测试和空中影子测试，因此还不能开启主动控制。

## 现在安全关机

1. 无人机保持未解锁。
2. 在NavRL影子模式终端按 `Ctrl-C`。
3. 在RViz终端按 `Ctrl-C`。
4. 在你手动启动的FAST-LIO和MID-360终端分别按 `Ctrl-C`。
5. 如果手动启动了 `su17_onboard.launch`，最后再停止它。
6. 执行：

```bash
sync
sudo poweroff
```

等Ubuntu完全关闭、屏幕熄灭后再断开无人机电源。不要直接拔电池。

---

# 以后每次使用的命令

安装和编译只需做一次。以后开机不需要再次运行setup脚本、pip或catkin_make。

## 第一步：检查原系统是否已经自动启动

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash

rosparam set use_sim_time false

rostopic list | grep -E \
'livox/lidar|cloud_mid360_body|Odometry|mavros/local_position/odom|prometheus/state|prometheus/control_state'
```

检查频率：

```bash
timeout 6 rostopic hz /uav1/cloud_mid360_body
timeout 6 rostopic hz /uav1/Odometry
timeout 6 rostopic hz /uav1/mavros/local_position/odom
timeout 6 rostopic hz /uav1/prometheus/state
timeout 6 rostopic hz /uav1/prometheus/control_state
```

预期约为：

```text
10 / 10 / 30 / 50 / 100 Hz
```

如果全部正常，跳过原系统手动启动步骤。

## 第二步：缺哪个就启动哪个

Prometheus/MAVROS没有启动时：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment su17_onboard.launch
```

MID-360没有启动时：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment msg_MID360.launch
```

FAST-LIO或body点云没有启动时：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
roslaunch su17_experiment mapping_mid360_y.launch
```

不要在话题已经正常时重复启动。

## 第三步：启动NavRL影子模式

```bash
bash /home/amov/navrl2/ros1_real/navigation_runner/tools/run_su17_phase1_shadow.sh
```

这条脚本固定：

```text
output_enabled=false
```

不会向无人机发送控制指令。

## 第四步：检查NavRL

另开终端：

```bash
source /opt/ros/noetic/setup.bash
source /home/amov/su17_experiment/devel/setup.bash
source /home/amov/navrl_venv/bin/activate
source /home/amov/navrl_ws/devel/setup.bash
```

检查：

```bash
timeout 10 rostopic hz /occupancy_map/update

rostopic echo -n 1 /uav1/navrl/navigation_status
rostopic echo -n 1 /uav1/navrl/bridge_status

rostopic info /uav1/prometheus/command
```

没有发送目标时：

```text
navigation_status: idle_no_goal
bridge_status: shadow_waiting_for_policy
```

影子模式下，`/navrl_su17_bridge`不能出现在Prometheus command的publisher列表中。

## 第五步：启动RViz

```bash
roslaunch navigation_runner rviz.launch
```

将RViz的 `Fixed Frame`改为：

```text
world
```

观察：

```text
/occupancy_map/depth_cloud
/occupancy_map/voxel_map
/occupancy_map/inflated_voxel_map
```

`depth_cloud`是CropBox和量程过滤后的当前帧点云。`local_cloud`是旧静态聚类路径遗留的话题，第一阶段不会发布，不用于纸板测试。

---

# 下一步1：完成六向纸板测试

下次开机后，先启动原系统、NavRL影子模式和RViz。无人机保持未解锁，最好拆桨。

纸板依次从以下方向靠近：

- 前
- 后
- 左
- 右
- 上
- 下

每个方向测试：

```text
1.0 m
0.6 m
0.4 m
```

通过标准：

- `depth_cloud`能看到纸板；
- `voxel_map`产生障碍；
- `inflated_voxel_map`产生更大的膨胀障碍；
- 纸板在约`0.4 m`不会突然完全消失；
- 无人机自身没有形成大片固定障碍。

小于`0.35 m`属于当前近距过滤范围，不作为失败依据。

纸板测试时可观察：

```bash
timeout 60 rostopic echo /rosout | grep -E \
'timestamp delta|Point-cloud filter|Rejecting pointcloud'
```

---

# 下一步2：空中影子测试

只有六向纸板全部通过后才进行。

注意：此次仍使用影子模式，NavRL不控制无人机；全部飞行动作由原遥控器和原SU17流程完成。

起飞前开始录包：

```bash
rosbag record --lz4 \
  -O /home/amov/navrl_air_shadow_01.bag \
  /uav1/cloud_mid360_body \
  /uav1/Odometry \
  /uav1/mavros/local_position/odom \
  /uav1/prometheus/state \
  /uav1/prometheus/control_state \
  /uav1/navrl/navigation_status \
  /uav1/navrl/bridge_status \
  /uav1/navrl/desired_setpoint \
  /uav1/navrl/safe_setpoint \
  /occupancy_map/update
```

人工起飞至约 `0.8～1.0 m`稳定悬停，读取当前位置：

```bash
rostopic echo -n 1 /uav1/mavros/local_position/odom
```

发送一个仅供计算、不会控制飞机的近距离目标。把坐标替换为当前点附近约`0.5～1.0 m`的空旷位置：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  "header: {frame_id: 'world'}
pose:
  position: {x: 目标X, y: 目标Y, z: 当前高度}
  orientation: {w: 1.0}"
```

然后人工缓慢完成：

- 悬停；
- 前后小幅移动；
- 左右小幅移动；
- 缓慢改变航向；
- 始终保持在起飞点约2m内。

监控：

```bash
rostopic echo /uav1/navrl/bridge_status
```

目标激活且输入正常时应出现：

```text
shadow_ready
```

不得出现：

```text
shadow_odom_inconsistent
```

完成后：

1. 人工悬停并降落；
2. 停止录包；
3. 停止NavRL；
4. 把bag复制回Mac：

```bash
scp amov@192.168.2.97:/home/amov/navrl_air_shadow_01.bag \
  "/Users/yoloflps/Downloads/"
```

把这个bag交给我分析两套里程计的运动一致性。通过后，我们再准备首次系留主动飞行；在此之前不要使用 `output_enabled:=true`。
