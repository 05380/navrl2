# 人形具身智能实习学习与求职路线

> 适用背景：已完成基于 PPO、Isaac 仿真和 ROS/Gazebo 部署的无人机局部导航项目，希望转向人形机器人或具身智能相关实习。
>
> 信息核验日期：2026-09-25。企业岗位和开源项目更新很快，投递或安装前应再次查看官方页面。
>
> 目标：不是短期内“学完具身智能”，而是在 8-12 周内形成一条可信的技术主线、一项可演示的新增作品和一套能应对面试追问的知识体系。


招聘要求的学习内容融入进项目；招人渠道；师兄经验


## 0. 先理解“具身智能”包括什么

具身智能不是某一个模型名称，也不等于人形机器人或 VLA。更实用的定义是：**智能体通过一个具有物理约束的身体，在感知-决策-行动闭环中与环境交互并完成任务。** 真机是最终载体，高保真仿真则是训练和验证的重要工具。

在人形机器人公司里，技术栈常被粗略分成四层：

| 层次 | 主要问题 | 典型技术/岗位 |
|---|---|---|
| 本体与执行器 | 身体能否可靠地产生目标运动 | 机械、关节、电机、驱动、嵌入式、传感器 |
| “小脑”与技能 | 如何站稳、行走、恢复和操作 | 动力学、WBC/MPC、locomotion RL、motion imitation、灵巧操作 |
| “大脑”与自主性 | 做什么、去哪里、如何理解任务 | 感知、导航、规划、world model、VLM/VLA、Agent |
| 数据与基础设施 | 如何高效训练、评测和部署 | 仿真、遥操作、数据闭环、训练平台、模型部署、机器人中间件 |

你当前最接近的是“小脑/技能 + 自主导航 + 仿真基础设施”的交界处。短期内应先把这条连接做深，再逐步进入 manipulation/VLA；不需要同时补齐机械、电机、视觉大模型和全身控制的所有细节。

## 1. 先确定求职主线

### 1.1 最适合你的方向

建议将目标岗位按以下优先级排列：

| 优先级 | 岗位方向 | 当前匹配度 | 选择理由 |
|---|---|---:|---|
| A | 人形/足式机器人强化学习运动控制 | 高 | PPO、并行仿真、奖励设计、课程训练和部署经验可以直接迁移 |
| A | 机器人仿真与 sim-to-real 算法 | 高 | 已使用 Isaac、Gazebo、ROS，并处理过训练/部署差异 |
| A- | 人形机器人导航与规划 | 中高 | 无人机局部导航经验可迁移，但要补足足式底层控制和整机约束 |
| B+ | 机器人学习/模仿学习 | 中 | 已有深度学习与控制基础，需要补数据采集、BC、ACT、Diffusion Policy |
| B | 具身智能算法工程化 | 中 | 需要加强 ROS 2、C++、推理部署、数据闭环和真实硬件调试 |
| C | VLA/机器人基础模型研究 | 中低 | 需要额外补 Transformer、多模态预训练、大规模数据与分布式训练，不宜作为第一主线 |

因此，推荐定位是：

> **主线：人形机器人运动控制、强化学习和 sim-to-real。**
>
> **副线：操作学习、模仿学习与 VLA 的基本原理和小规模复现。**

这比直接把自己包装成 VLA 研究者更可信，也更能发挥 ForesightNav 的已有积累。

### 1.2 你已有的可迁移能力

- 使用 PPO 训练连续控制策略，理解 actor、critic、GAE、clipping 和并行 rollout；
- 在 Isaac 中构建向量化环境，设计 observation、action、reward、termination 和 curriculum；
- 处理动态障碍物、局部观测、坐标变换和速度命令；
- 使用 ROS/Gazebo 做部署评估，已经意识到频率、单位、坐标系、感知误差和动力学差异的重要性；
- 能进行消融实验、失败案例分析和论文写作；
- 有从算法训练到系统部署的完整项目，而不只是跑过单个 benchmark。

### 1.3 需要重点补齐的能力

1. **多刚体机器人学：** SE(3)、正/逆运动学、Jacobian、刚体动力学和浮动基系统。
2. **接触与足式控制：** 接触力、摩擦锥、支撑多边形、CoM、ZMP、centroidal dynamics、步态和动态平衡。
3. **经典控制链路：** joint PD、阻抗控制、逆动力学、WBC/QP 和 MPC 的基本作用与边界。
4. **人形 RL 工程：** 关节动作、低层 PD、控制频率、运动模仿、domain randomization、延迟和 actuator model。
5. **机器人软件工程：** ROS 2、现代 C++、URDF/MJCF/USD、通信、日志、实时性和安全状态机。
6. **操作学习：** demonstration 数据、behavior cloning、ACT、Diffusion Policy、action chunking 和闭环评测。
7. **VLA 基础：** VLM backbone、动作表示、跨本体数据、微调与部署；达到“能解释和运行”即可，暂不追求从头预训练。

## 2. 面试需要掌握的知识地图

### 2.1 机器人学基础

至少要能不看资料解释以下内容：

- 坐标系、旋转矩阵、四元数和 SE(3) 变换分别解决什么问题；
- 正运动学、逆运动学和 Jacobian 的输入输出；
- Jacobian 奇异性为什么会导致速度或力映射异常；
- 关节空间和任务空间控制的区别；
- 刚体动力学方程各项的物理意义：

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=S^\top\tau+J_c(q)^\top\lambda_c.
\]

- 人形机器人为什么是 floating-base、underactuated system；
- 接触约束和摩擦锥为什么决定机器人能否站稳；
- forward dynamics、inverse dynamics 和 system identification 的区别。

推荐资源：

- [Modern Robotics 免费教材与课程](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)：重点学习 Ch. 3-6、8、11；
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)：用 Python 完成 FK、Jacobian、inverse dynamics 小练习；
- [MuJoCo 官方文档](https://mujoco.readthedocs.io/en/stable/)：理解 MJCF、actuator、contact、solver 和 simulation loop。

### 2.2 足式与人形控制

必须理解以下层次，而不是把“控制”全部等同于神经网络：

| 层次 | 典型输入 | 典型输出 | 频率示例 | 作用 |
|---|---|---|---:|---|
| 任务/导航层 | 目标、地图、感知 | 基座速度或局部目标 | 1-20 Hz | 决定去哪里 |
| 技能/运动策略 | proprioception、命令、历史 | 关节目标或残差 | 20-100 Hz | 决定如何走或执行动作 |
| WBC/关节控制 | 目标姿态、接触状态 | torque/position target | 100-1000 Hz | 满足动力学和接触约束 |
| 电机驱动 | 电流、编码器反馈 | 实际力矩 | kHz 级 | 执行底层控制 |

要会回答：

- RL locomotion policy 为什么常输出关节位置偏移，而不是直接输出电机电流；
- policy frequency、physics step、control decimation 和电机环频率有何区别；
- joint PD 如何把策略动作转成力矩；
- 为什么需要 gait、contact schedule 或 reference motion，它们何时可以不显式使用；
- WBC/QP 如何把多个任务和约束统一到一个优化问题；
- 学习控制和经典控制为什么通常是互补关系，而不是二选一。

### 2.3 强化学习运动控制

#### 为什么人形运动控制常用 PPO

- 动作连续且维度较高，policy gradient 可直接优化随机连续策略；
- Isaac Lab 一类 GPU 并行仿真能快速生成大量新鲜 on-policy 数据，缓解 PPO 样本效率偏低的问题；
- clipped update、advantage normalization 和 gradient clipping 使训练相对稳定；
- 实现成熟，`rsl_rl`、Isaac Lab 和多个足式开源项目都有验证过的训练链路；
- 对复杂 reward shaping、curriculum 和 domain randomization 的工程兼容性较好。

#### PPO 的不足

- on-policy 数据通常只使用少数轮，样本效率低；
- 对 reward、observation normalization、termination 和超参数仍敏感；
- clipping 只是经验性的更新限制，不保证单调改进或真实机器人安全；
- 训练稳定不代表 sim-to-real 稳定；
- 长时域稀疏任务仍可能探索困难。

#### 与其他方法的比较

| 方法 | 优点 | 局限 | 在人形项目中的常见位置 |
|---|---|---|---|
| PPO | 并行训练成熟、较稳定、实现简单 | 样本效率低 | locomotion、tracking、motion imitation |
| SAC/TD3 | replay 提高样本效率，适合连续动作 | 大规模并行下 replay 成本高，训练和分布偏移更难调 | 小规模仿真、真实数据昂贵的任务 |
| Behavior Cloning | 简单、稳定、可直接利用示范 | covariate shift，超出示范分布时易失效 | 操作技能初始化、遥操作数据学习 |
| ACT | action chunk 可缓解高频误差积累，适合双臂示范 | 依赖高质量数据，对分布外状态仍脆弱 | 精细操作、双臂任务 |
| Diffusion Policy | 能表达多模态动作分布，轨迹平滑 | 推理和训练更重，部署延迟需控制 | 视觉操作、复杂接触任务 |
| MPC/WBC | 显式模型、约束清晰、可解释 | 模型和在线优化成本高 | 安全约束、全身协调、策略下层 |
| VLA | 语义泛化和多任务潜力强 | 数据/算力大，低层稳定控制仍需专门模块 | 高层技能选择或操作策略 |

### 2.4 Sim-to-real 必须会讲的内容

训练成功不等于能上真机。面试中应能按以下链路分析 sim-to-real gap：

1. **模型误差：** 质量、惯量、关节零位、摩擦、背隙、电机饱和与柔性。
2. **接触误差：** 地面摩擦、恢复系数、接触刚度、solver 和时间步差异。
3. **观测误差：** IMU bias、encoder noise、状态估计延迟、丢包和坐标偏差。
4. **执行误差：** command latency、控制频率抖动、PD 参数和 torque limit。
5. **软件误差：** observation 顺序、归一化统计、单位、frame、模型导出和 dtype。
6. **真实安全：** 启动姿态、急停、限幅、fall detection、watchdog 和恢复状态机。

常用措施：

- domain randomization；
- actuator network 或更准确的 motor model；
- observation history、delay randomization 和噪声注入；
- privileged teacher + student adaptation；
- system identification；
- sim-to-sim（例如 Isaac Lab -> MuJoCo）交叉验证；
- 逐级测试：离线网络输出、悬空、保护架、低速、受限空间、完整任务。

### 2.5 操作学习与 VLA

你不需要一开始就训练 7B VLA，但必须理解完整数据闭环：

> 遥操作/脚本采集 -> 时间同步与标定 -> 数据清洗 -> 训练 -> 离线诊断 -> 闭环评测 -> 失败回流。

需要掌握的概念：

- state action 与 end-effector delta action 的差别；
- 单步动作与 action chunk；
- image/state normalization；
- teacher forcing、behavior cloning loss 和 covariate shift；
- diffusion/flow matching 为什么适合多模态连续动作；
- VLA 如何把图像、语言和机器人状态映射到动作；
- cross-embodiment 时不同自由度、坐标系和控制接口如何对齐；
- 成功率为何必须通过闭环 rollout 而不是仅看训练 loss 评估。

## 3. 12 周标准学习计划

建议每周投入 15-20 小时。每周都必须有可检查的输出，避免只看论文或视频。

### 第 0 周：确定岗位和环境

**目标**

- 收集 15 个真实 JD，按“运控/RL、仿真、操作学习/VLA”分类；
- 从 JD 统计出现频率最高的 20 个关键词；
- 新建独立项目仓库，不在 ForesightNav 仓库内堆叠无关代码；
- 固定 Ubuntu、CUDA、Isaac Sim、Isaac Lab、PyTorch 和驱动版本。

**输出**

- `target_roles.md`；
- 环境锁定文件和一次可重复的 smoke test；
- 一页个人能力 gap 表。

### 第 1 周：运动学与坐标系统

**学习**

- SO(3)、SE(3)、指数坐标；
- FK、IK、body/space Jacobian；
- URDF 的 link、joint、inertial 和 frame。

**实践**

- 用 Pinocchio 载入一个 G1/H1 或机械臂 URDF；
- 输出指定 link 的位姿和 Jacobian；
- 实现一个阻尼最小二乘 IK，并画出误差收敛曲线。

**验收**

- 能解释 world、base、body、end-effector frame；
- 能定位一个“坐标系方向写反”的具体 bug。

### 第 2 周：动力学、接触与经典控制

**学习**

- (M,C,g,J,\lambda) 的物理意义；
- joint PD、computed torque、impedance control；
- contact、friction cone、CoM、ZMP 和 centroidal momentum；
- WBC/QP 的 objective、equality constraint 和 inequality constraint。

**实践**

- 在 MuJoCo 中搭建倒立摆或简化双足；
- 比较 position control、torque PD 和错误 PD 参数的行为；
- 记录不同时间步、摩擦和延迟对稳定性的影响。

**验收**

- 能解释“策略输出关节目标 + PD”为什么不是纯 position control；
- 能说清接触力为何通过 (J_c^\top\lambda_c) 进入动力学。

### 第 3 周：Isaac Lab 与人形环境

**学习/实践**

- 安装并跑通 [Isaac Lab](https://github.com/isaac-sim/IsaacLab)；
- 阅读 manager-based environment 的 observation、action、reward、event、termination 配置；
- 跑通 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) 的 G1 velocity task；
- 对照 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 找到 rollout、GAE、PPO update 和 checkpoint 路径。

**输出**

- 环境数据流图；
- observation/action/reward/termination 表；
- 一段 60 秒 baseline 视频和训练曲线。

### 第 4 周：训练并解释速度跟踪策略

**实践**

- 训练 G1 或 H1 平地速度跟踪；
- 只做少量、可解释的配置变化：command range、reward scale、control decimation；
- 比较至少 3 个随机种子，报告均值和波动；
- 记录 velocity tracking RMSE、fall rate、episode length、energy/torque proxy。

**面试准备**

- 为什么选择 PPO；
- observation history 是否必要；
- action scale、PD gains 和 policy frequency 如何共同影响运动；
- reward 各项如何冲突，怎样防止 reward hacking。

### 第 5 周：鲁棒性与 sim-to-sim

**实践**

- 随机化 mass、friction、motor strength、sensor noise、latency 和 external push；
- 做“无随机化 vs 有随机化”的消融；
- 按 Unitree RL Lab 流程导出策略，在 MuJoCo 做 sim-to-sim；
- 汇总 Isaac 与 MuJoCo 中 tracking error 和 fall rate 的差异。

**输出**

- 一张随机化范围表；
- 一张跨 simulator 结果表；
- 3-5 个失败案例及原因分类。

### 第 6 周：运动模仿与动作重定向

**阅读**

- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html)；
- [Adversarial Motion Priors](https://xbpeng.github.io/projects/AMP/index.html)；
- [PHC](https://github.com/ZhengyiLuo/PHC)；
- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym)。

**实践**

- 跑通一条 reference motion 的 tracking 或 imitation；
- 理解 motion retargeting、root pose、joint mapping、phase 和 reference state initialization；
- 比较 task reward、style/imitation reward 和 termination 的作用。

**验收**

- 能解释 locomotion command tracking 与 motion imitation 的差异；
- 能说明 MoCap 人体骨架为什么不能直接映射到机器人关节。

### 第 7 周：WBC、状态估计与 ROS 2

**学习/实践**

- 阅读一个 WBC/QP 示例，能写出基本目标和约束；
- 了解 base state estimation、IMU、kinematic contact 和 EKF 的角色；
- 完成 ROS 2 publisher/subscriber、service、action、launch 和 rosbag2；
- 用 C++ 写一个固定频率控制节点，加入 timeout、限幅和 watchdog；
- 理解 DDS/QoS 对机器人通信的影响。

官方入口：[ROS 2 Documentation](https://docs.ros.org/)。

### 第 8 周：模仿学习与数据闭环

**实践**

- 使用 [LeRobot](https://github.com/huggingface/lerobot) 或 [ACT](https://github.com/tonyzhaozh/act)；
- 在仿真任务上完成数据查看、训练和闭环评测；
- 明确 episode、frame、state、action、camera、timestamp 和 task metadata；
- 对比不同示范数量或 action chunk 长度。

**输出**

- 数据 schema；
- train/validation loss 与真实 rollout success rate 对照；
- 失败案例：视角变化、物体位置变化、动作累积误差。

### 第 9 周：Diffusion Policy 与 VLA 入门

**阅读顺序**

1. [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)；
2. [Open X-Embodiment](https://robotics-transformer-x.github.io/)；
3. [Octo](https://octo-models.github.io/)；
4. [OpenVLA](https://openvla.github.io/)；
5. [openpi](https://github.com/Physical-Intelligence/openpi)；
6. [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)。

**实践边界**

- 优先做 inference、dataset adapter 或小规模 LoRA；
- 不从头训练大模型；
- 显存不足时只分析架构和数据流，继续完成 ACT/小型 Diffusion Policy 闭环实验。

**算力取舍**

- locomotion 训练先通过减少并行环境数适配现有 NVIDIA GPU，优先保证流程和评测正确；
- VLA 只做 inference 或参数高效微调。[openpi](https://github.com/Physical-Intelligence/openpi) 当前给出的单卡参考门槛约为 inference 8 GB 以上、LoRA 22.5 GB 以上，具体占用仍取决于模型和 batch；
- 本地 macOS 适合阅读、轻量分析和文档工作，Isaac Lab、人形训练和 CUDA 推理统一放在版本锁定的 Ubuntu/NVIDIA 环境；
- 不为了“用满显卡”盲目增大 `num_envs`，先监控吞吐、显存、simulation FPS 和 policy update 时间。

**验收**

- 能画出 VLM/VLA 的输入、backbone、action head 和 action chunk 数据流；
- 能解释“语义规划能力”和“高频稳定控制”为什么通常需要分层。

### 第 10 周：主作品集集成

完成第 5 节定义的“导航命令 + G1 运动策略”项目：

- 基础速度跟踪；
- 障碍环境中的高层命令；
- domain randomization；
- Isaac -> MuJoCo sim-to-sim；
- 统一日志和评测脚本。

### 第 11 周：评测、消融和工程整理

- 至少 3 个随机种子；
- baseline 与 2 个有效消融；
- 固定测试集和随机种子；
- 报告均值、标准差和失败分布；
- README 一键命令、环境版本、硬件、checkpoint 和常见错误；
- 2 分钟演示视频与一页技术海报。

### 第 12 周：面试与集中投递

- 完成 30 秒、2 分钟和 10 分钟三版项目介绍；
- 每天手写回答 3 个第 8 节问题；
- 针对每个 JD 调整项目 bullet 和技能顺序；
- 每家公司至少了解一个产品、一个技术栈和一个公开项目；
- 先投匹配岗位，再投跨度较大的 VLA 岗位。

## 4. 8 周压缩方案

若距离面试不足两个月，按下面顺序压缩：

| 周 | 内容 | 不可删的输出 |
|---:|---|---|
| 1 | 运动学 + 动力学 + PD | Pinocchio IK、动力学解释 |
| 2 | Isaac Lab + Unitree RL Lab | 跑通 G1 baseline |
| 3 | PPO locomotion | 三种子曲线、指标表 |
| 4 | domain randomization + sim-to-sim | Isaac/MuJoCo 对照 |
| 5 | DeepMimic/RMA/Humanoid-Gym | 论文卡片、imitation demo |
| 6 | ROS 2/C++ + WBC 概念 | 控制节点、watchdog、WBC 框图 |
| 7 | LeRobot + ACT | 一个闭环 manipulation baseline |
| 8 | 项目包装 + 面试 + 投递 | README、视频、简历、问答 |

压缩时先删 VLA 微调，不要删动力学、控制、评测和 sim-to-real。

## 5. 推荐的主作品集项目

### 5.1 项目题目

**Hierarchical Humanoid Navigation with Robust RL Locomotion**

中文可写：**基于鲁棒强化学习运动控制的分层人形机器人导航**。

### 5.2 为什么适合你

它把已有的无人机导航经验与新学习的人形运动控制连接起来：

\[
\text{local observation/goal}
\rightarrow \text{high-level velocity command}
\rightarrow \text{G1 locomotion policy}
\rightarrow \text{joint targets + PD}.
\]

你不需要立刻解决完整 VLA，也不需要拥有真机，就能展示：

- 分层系统设计；
- PPO locomotion；
- 动态/静态障碍导航；
- sim-to-sim 与 domain randomization；
- 对控制频率、坐标系和安全接口的理解。

### 5.3 实现阶段

1. 复现 Unitree G1 velocity tracking baseline。
2. 给定手柄或脚本速度命令，验证前进、侧移、转向和停止。
3. 加入高层局部导航器：先用简单 geometric baseline，再接入自己熟悉的学习式导航策略。
4. 限制命令变化率，处理 stop、turn-in-place 和 fall recovery。
5. 加入质量、摩擦、延迟、motor strength、观测噪声和外力随机化。
6. 导出到 MuJoCo 做 sim-to-sim。
7. 对高层导航与低层 tracking 分别评测，避免只报端到端成功率。

### 5.4 最低评测集

| 类别 | 指标 |
|---|---|
| 低层 tracking | linear/angular velocity RMSE、command response time |
| 稳定性 | fall rate、mean episode length、push recovery success |
| 导航 | success、collision、timeout、path length/SPL |
| 工程 | inference latency、control-loop jitter、real-time factor |
| 代价 | torque/energy proxy、foot slip |
| 泛化 | 未见 friction、mass、terrain、obstacle layout |

### 5.5 必做消融

- 无 domain randomization；
- 无 latency/noise randomization；
- 高层命令不做 rate limit；
- 可选：无 observation history 或不同 history length。

不要把“训练后的视频看起来能走”当作完整结论。

### 5.6 简历 bullet 模板

完成后再按真实结果填写数字：

> Built a hierarchical humanoid navigation stack in Isaac Lab, coupling a goal-conditioned local planner with a PPO-based Unitree G1 locomotion policy; evaluated velocity tracking, navigation success, and robustness under dynamics and latency randomization, and validated policy transfer in MuJoCo sim-to-sim.

若没有真机，不要写 `sim-to-real deployment`；应写 `sim-to-sim validation` 或 `sim-to-real-oriented robustness evaluation`。

## 6. 推荐的副作品集项目

### 6.1 题目

**Closed-Loop Manipulation with ACT or Diffusion Policy**。

### 6.2 最小方案

- 使用 LeRobot、ACT 自带仿真任务或 LIBERO；
- 检查/转换 demonstration dataset；
- 训练一个 ACT baseline；
- 评测至少 50 次闭环 rollout；
- 改变物体初始位置和视觉条件，分析泛化；
- 有余力再加入 Diffusion Policy 比较。

### 6.3 重点不是模型规模

面试价值主要来自你能解释：

- 数据怎样采集和同步；
- action chunk 为什么有效；
- loss 下降为何不等于成功率上升；
- 如何处理相机标定、动作归一化和失败数据；
- 怎样建立训练、评测、失败回流的数据闭环。

## 7. 论文与开源项目阅读顺序

### 7.1 必读并动手

| 资源 | 学到什么 | 预期动作 |
|---|---|---|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | 当前主流 GPU 机器人学习框架 | 跑通 task、读配置和 manager 数据流 |
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | G1/H1 训练、MuJoCo 验证和部署链路 | 训练一个 G1 task，完成 sim-to-sim |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | 机器人 PPO 实现 | 跟一遍 rollout 到 update |
| [MuJoCo](https://mujoco.org/) | 接触动力学与独立仿真验证 | 修改 friction、actuator、delay |
| [LeRobot](https://github.com/huggingface/lerobot) | 数据、示范、训练和部署闭环 | 完成一个 imitation baseline |

### 7.2 运动控制核心论文

| 论文/项目 | 阅读重点 | 深度 |
|---|---|---|
| [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) | reference motion、imitation reward、robust recovery | 精读 |
| [Adversarial Motion Priors](https://xbpeng.github.io/projects/AMP/index.html) | 无需手工风格奖励的 motion prior | 精读方法，跑代码可选 |
| [RMA](https://www.roboticsproceedings.org/rss17/p011.pdf) | privileged training、adaptation module、真实泛化 | 精读 |
| [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) | humanoid locomotion 与 zero-shot sim-to-real pipeline | 精读并运行 |
| [PHC](https://github.com/ZhengyiLuo/PHC) | 大规模 motion tracking 与失稳恢复 | 选读 |

阅读每篇论文只记录六项：问题、观测、动作、目标函数、部署链路、证据边界。

### 7.3 操作学习与 VLA

| 论文/项目 | 阅读重点 | 优先级 |
|---|---|---:|
| [ACT / ALOHA](https://github.com/tonyzhaozh/act) | action chunk、CVAE、双臂示范 | A |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 条件扩散动作生成、receding horizon | A |
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | 跨数据集与跨本体数据问题 | B |
| [Octo](https://octo-models.github.io/) | 通用策略预训练与任务适配 | B |
| [OpenVLA](https://github.com/openvla/openvla) | VLM 到连续机器人动作的开源范式 | B |
| [openpi](https://github.com/Physical-Intelligence/openpi) | flow-based VLA、微调和推理 | B |
| [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 面向通用/人形机器人的 VLA 工具链 | B |

### 7.4 可作为企业调研的开源项目

- [Unitree 官方 GitHub](https://github.com/unitreerobotics)：RL、MuJoCo、仿真、遥操作、LeRobot 和 VLA 项目；
- [Fourier 官方 GitHub](https://github.com/FFTAI)：GRx 模型、Isaac Lab、MuJoCo、部署和 LeRobot；
- [OpenLoong](https://www.openloong.org.cn/)：开源人形机器人生态；
- [MagicLab 开源中心](https://www.magiclab.top/about)：人形、四足和 ROS 2 SDK；
- [NVIDIA Isaac Lab humanoid workflow](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/tutorial_humanoid_workflow.html)：从机器人资产到训练/部署的官方流程。

## 8. 高频面试问题清单

### 8.1 项目迁移类

1. 你的无人机项目哪些能力能迁移到人形机器人，哪些不能？
2. 为什么导航策略不能直接输出人形机器人每个关节的动作？
3. 你会如何把 ForesightNav 接到 G1 的 locomotion policy？
4. 高层导航与低层运动策略的频率、坐标系和命令接口怎样定义？
5. UAV 是全驱动还是欠驱动？人形 floating base 为什么更难？

回答重点：明确层级边界，不要声称无人机速度控制经验等价于人形全身控制。

### 8.2 PPO 与强化学习

1. PPO clipping 限制的到底是什么，是否等价于硬 KL 约束？
2. 为什么并行仿真适合 PPO？
3. PPO 与 SAC 在样本效率、稳定性和 replay 分布方面有什么差异？
4. GAE 中 \(\gamma\) 和 \(\lambda\) 如何影响 bias/variance？
5. reward scale 改变为什么可能影响 policy 和 critic 的训练？
6. early termination 既是工程设计也是隐式 reward，为什么？
7. 怎样发现 reward hacking？
8. 三个种子够不够，如何报告波动？

### 8.3 人形运动控制

1. 为什么人形机器人是 underactuated？
2. 什么是 support polygon、ZMP 和 centroidal momentum？
3. joint-space PD 与 task-space impedance 有何区别？
4. WBC 为什么常被写成 QP？
5. policy action 是 joint position、joint delta 还是 torque，各有什么利弊？
6. policy 50 Hz、simulation 500 Hz 时 control decimation 是多少？
7. foot slip 怎样检测，reward 怎样设计？
8. reference motion retargeting 为什么困难？
9. 人体 MoCap 与 G1 的 DoF、比例和接触如何对齐？

### 8.4 Sim-to-real 与系统

1. domain randomization 应随机哪些量，范围过大有什么问题？
2. 为什么先做 Isaac -> MuJoCo sim-to-sim？它不能证明什么？
3. observation latency 与 action latency 应怎样建模？
4. 模型在仿真稳定、真机振荡，先查什么？
5. 怎样设计急停、watchdog、fall detection 和限幅？
6. ROS 2 QoS 的 reliability、history、deadline 对控制链路有什么影响？
7. Python 训练代码如何迁移到 C++/TensorRT 实时部署？

### 8.5 模仿学习与 VLA

1. behavior cloning 的 covariate shift 是什么？
2. ACT 为什么预测 action chunk，temporal ensembling 起什么作用？
3. Diffusion Policy 为什么能表达多模态动作？
4. VLA 的语言理解能力是否等于低层控制能力？
5. 跨本体训练时如何统一 action space？
6. 为什么闭环 success rate 比 validation loss 更重要？
7. 数据数量、数据覆盖和数据质量哪个更重要？如何诊断？

## 9. 三地企业与目标岗位表

以下是基于公开产品、开源项目和招聘页整理的**目标池**，不是对公司经营状况、待遇或文化的排名。优先级表示与你当前背景的技术匹配度。

### 9.1 杭州

| 优先级 | 企业/团队 | 方向 | 建议关注岗位 | 为什么适合你 | 官方入口 |
|---|---|---|---|---|---|
| A | 宇树科技 Unitree | 四足、人形、运动控制、具身模型 | 深度强化学习、运动控制、仿真、规划、具身软件 | 官方岗位直接要求 PPO/多自由度机器人/真机验证，与你的 RL 和 Isaac 经历最接近 | [招聘](https://www.unitree.com/position/) / [GitHub](https://github.com/unitreerobotics) |
| A- | 云深处科技 Deep Robotics | 四足、人形、行业落地 | 运控、强化学习、规划、仿真、系统软件 | 足式机器人主线明确，可把无人机导航与足式自主移动结合 | [官网与招聘入口](https://deeprobotics.cn/) |
| B+ | 千寻智能 Spirit AI | 具身 Agent、人形机器人 | 具身算法、RL/VLA、仿真、数据闭环 | 同时覆盖具身模型与机器人平台；具体工作城市和岗位需按批次核验 | [招聘示例](https://nwd4iy9rd2s.jobs.feishu.cn/campusofSpiritAI/m/position/7672031985867295002/detail) |
| B | 阿里通义机器人方向 | VLA、操作基础模型、RobotNav/Manip | 机器人学习、VLA、数据与训练基础设施 | 适合作为副线目标；更看重多模态、数据规模和模型训练能力，杭州是否开放对应岗位需按批次核验 | [Qwen-RobotManip](https://github.com/QwenLM/Qwen-RobotManip) |
| B | 宇泛智能 UNIUBI | 服务机器人、具身智能 | 导航、机器人软件、算法工程 | 杭州岗位与产品工程机会较多，适合导航/系统转型 | [加入我们](https://www.uniubi.com/about/join) |
| C+ | 强脑科技 BrainCo | 脑机接口、智能仿生手与康复机器人 | 控制、感知、人机交互、机器人软件 | 更偏脑机接口和仿生硬件，不是纯人形 locomotion，但可拓展灵巧操作视角 | [招聘](https://www.brainco.cn/recruit) |

### 9.2 上海

| 优先级 | 企业/团队 | 方向 | 建议关注岗位 | 为什么适合你 | 官方入口 |
|---|---|---|---|---|---|
| A | 智元机器人 AgiBot | 人形本体、运控、操作、具身模型与数据 | 运控/RL、仿真、具身算法、数据闭环 | 产品和岗位覆盖完整，官网明确提供实习招聘 | [实习/招聘](https://www.agibot.com.cn/join_us) |
| A | 傅利叶智能 Fourier | 通用人形、康复机器人、RL/IL 工具链 | 人形运控、Isaac Lab、部署、操作学习 | 官方开源了 GRx 的训练、MuJoCo、部署和 LeRobot 相关项目，便于针对性准备 | [官网](https://fftai.com/) / [GitHub](https://github.com/FFTAI) |
| A | 国家地方共建人形机器人创新中心 / OpenLoong | 开源人形平台、具身数据、VLA | 具身模型、强化学习、仿真、软件系统 | 偏研究与公共平台，适合希望兼顾论文、开源和整机系统的人 | [OpenLoong](https://www.openloong.org.cn/) / [招聘信息](https://forum.openloong.org.cn/forum.php?mod=viewthread&tid=140) |
| A- | 卓益得 DroidUp | 双足人形、运动控制、模仿学习 | 运动控制、DRL、模仿学习、机械臂具身 | 官方 JD 明确要求 PPO/SAC、Isaac/MuJoCo、动力学和 sim-to-real | [招聘](https://www.droidup.com/jiaruwomen) |
| B+ | 开普勒机器人 Kepler | 通用人形、工业场景 | 规划控制、C++、嵌入式、系统集成 | 与导航/控制背景匹配，但实习和具体算法岗需实时确认 | [官网](https://www.gotokepler.com/about) |

### 9.3 苏州

| 优先级 | 企业/团队 | 方向 | 建议关注岗位 | 为什么适合你 | 官方入口 |
|---|---|---|---|---|---|
| A | 魔法原子 MagicLab | 通用人形、四足、灵巧手 | 运控/RL、仿真、机器人软件、操作 | 人形和四足主线明确，并提供 SDK/开源中心 | [官网](https://www.magiclab.top/about) / [招聘](https://magiclab.zhiye.com/) |
| A | 至简动力 Simplexity Robotics | 通用人形、统一模型、数据闭环 | 运动控制、RL、仿真、VLA、感知 | 官网列出的算法和仿真岗位均向实习生开放；公司较新，应同时考察导师和项目成熟度 | [官网与岗位](https://www.simplexityrobotics.com/) |
| B+ | 艾利特机器人 Elite Robots | 协作、复合与人形操作机器人 | 运动控制、操作、机器人软件、具身模型 | 适合补机械臂控制和工业操作，产品落地属性强 | [官网](https://www.elibot.com/) / [招聘](https://www.elibot.com/about/join) |
| B | 双子智擎 Gemini AI Bot | 工业人形、运动控制与 AI 决策 | 运控、规划、工业场景集成 | 可接触人形在工业任务中的工程落地；公开招聘入口较少，需主动联系核验 | [官网](https://geminiaibot.com/) |
| B- | 科沃斯 Ecovacs | 服务机器人、感知、导航与产品化 | 导航、SLAM、感知、机器人算法 | 不以人形为主，但与你的自主导航背景高度相邻，工程和量产经验有价值 | [招聘](https://hr.ecovacs.cn/) |

### 9.4 投递顺序建议

1. **第一批：** Unitree、Deep Robotics、AgiBot、Fourier、DroidUp、MagicLab、Simplexity。
2. **第二批：** OpenLoong、Kepler、Elite、Spirit AI、UNIUBI。
3. **扩展批：** Qwen 机器人方向、BrainCo、Ecovacs、Gemini AI Bot，以及高校/研究院的联合实习。

每个岗位投递前核对：

- 工作地点是否真在目标城市；
- 是实习、校招还是要求 3-5 年经验；
- 团队做 locomotion、manipulation、VLA、导航还是平台；
- 是否能接触真机、数据和完整评测；
- 导师/汇报对象、实习时长和转正机会；
- 初创公司需要额外了解融资、产品交付、团队稳定性和工作节奏。

## 10. 简历与自我介绍策略

### 10.1 不要把方向写得过宽

不建议写：

> 求职方向：具身智能、大模型、强化学习、机器人控制、规划、感知、SLAM。

建议写：

> 求职方向：人形/足式机器人强化学习运动控制、机器人仿真与 sim-to-real；同时具备自主导航和 ROS 部署经验。

### 10.2 60 秒转型说明

> 我之前主要做基于强化学习的无人机局部导航，完成了从 Isaac 并行训练、PPO 和奖励设计，到 ROS/Gazebo 部署评估的完整链路。这个经历让我比较熟悉连续控制、课程训练、动态环境和 sim-to-real 中的观测、坐标系与频率一致性问题。现在我希望把这些能力迁移到人形机器人，重点补足浮动基动力学、接触控制、WBC 和人形 locomotion。我正在用 Isaac Lab 和 Unitree G1 完成速度跟踪、domain randomization 和 MuJoCo sim-to-sim，并同步学习 ACT/LeRobot 的操作学习流程。

最后一句必须与你届时真实完成的进度一致。

### 10.3 面试证据优先级

从强到弱依次是：

1. 能运行的代码、配置、checkpoint 和视频；
2. 固定协议下的多种子指标和消融；
3. 能解释的失败案例和修复过程；
4. 读过并能联系实现的论文；
5. 只在简历技能栏列出名词。

## 11. 前 7 天立即执行清单

### Day 1

- 下载 15 个目标岗位 JD；
- 确定主投岗位名称；
- 建立技能关键词统计表。

### Day 2

- 阅读 Modern Robotics 的 SE(3) 与 FK 部分；
- 用 Python 验证 rotation、transform composition 和 frame conversion。

### Day 3

- 用 Pinocchio 载入机器人模型；
- 输出关节、link 位姿和 Jacobian。

### Day 4

- 阅读人形 floating-base 动力学、接触和 joint PD；
- 在 MuJoCo 修改摩擦、质量和 actuator 参数。

### Day 5

- 安装版本匹配的 Isaac Lab；
- 跑通官方 cartpole 或 humanoid 示例。

### Day 6

- 安装 Unitree RL Lab；
- 跑通 G1 play 或预训练策略；
- 记录 observation、action、reward、frequency。

### Day 7

- 画出 G1 训练到 MuJoCo sim-to-sim 的链路；
- 录制 baseline；
- 写一页“我当前不理解的问题”，作为下一周任务。

## 12. 学习完成度检查

达到以下标准后，可以把“人形强化学习/仿真”作为简历重点：

- [ ] 能从公式解释 floating-base dynamics 和 contact force；
- [ ] 能用 Pinocchio 做 FK/Jacobian/inverse dynamics；
- [ ] 能从代码定位 PPO rollout、GAE 和 update；
- [ ] 能解释 humanoid observation、action、PD、decimation 和 reward；
- [ ] 独立训练过一个 G1/H1 locomotion policy；
- [ ] 做过多种子评测和至少两个消融；
- [ ] 做过 Isaac -> MuJoCo sim-to-sim；
- [ ] 能列出 sim-to-real gap 和逐级上真机流程；
- [ ] 能用 ROS 2/C++ 写带 watchdog 的控制节点；
- [ ] 跑通过一个 ACT/LeRobot 闭环操作任务；
- [ ] 能解释 VLA 的数据、动作表示和部署边界；
- [ ] 有 2 分钟视频、清晰 README 和一页结果表。

## 13. 几条重要取舍

- 不要一开始购买昂贵人形机器人；先用成熟开源模型完成训练、评测和 sim-to-sim。
- 不要把所有时间投入安装过时的 Isaac Gym 项目；优先使用持续维护的 Isaac Lab，旧项目仅用于理解论文。
- 不要从头训练大型 VLA；先完成 ACT 或小型 Diffusion Policy，再做现成模型的 inference/adapter/LoRA。
- 不要只展示最好的一条视频；保留失败案例、测试协议和统计结果。
- 不要把 NavRL 继承组件或团队工作说成个人原创；面试官更重视边界清楚和真正理解。
- 不要把“具身智能”等同于“VLA”。机器人动力学、控制、数据、系统和安全同样是核心。
- 不要同时准备五个方向。先靠运动控制/RL/仿真拿到面试，再根据 JD 深挖操作学习或 VLA。

---

**最终建议：** 你的最短有效路径不是抛弃无人机项目重新开始，而是把已有的 PPO、并行仿真、局部导航和部署经验升级为“高层自主导航 + 人形低层运动控制”的分层作品。完成 G1 locomotion、domain randomization、MuJoCo sim-to-sim 和一个小型 ACT 项目后，你对人形运控、仿真和机器人学习实习都会具备更可信的竞争力。
