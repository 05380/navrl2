# ForesightNav 项目面试复习手册

> 适用仓库：`navrl2`，分支 `2-14-7-1`，整理时 HEAD 为 `23807b9`。  
> 论文核心实现的历史基线：`6e137e952fc075b3804c9addadc074691482df02`。  
> 本文档按“能解释设计、能讲清代码、能诚实说明证据边界”的标准编写，不应代替对实际实验记录和个人贡献边界的确认。

## 0. 面试前先统一口径

这个仓库目前存在三个容易混淆的层次：

| 层次 | 核心内容 | 面试口径 |
|---|---|---|
| 论文核心方法 | TTC-aware 3D-VO 训练奖励 + action-conditioned multi-horizon outcome learning | 项目的主要方法贡献 |
| NavRL 继承部分 | PPO、局部感知编码、Beta 连续动作策略、部署端 safety shield | 基线框架或已有组件，不要说成自己的原创 |
| 当前分支后续增强 | wall-following teacher、demonstration buffer、behavior cloning、噪声注入 | 当前工程探索；论文结果若未使用，就不要混入论文贡献 |

面试前必须确认你实际提交的 checkpoint 属于哪一版，并保存对应的：

- Git commit；
- 完整 Hydra 配置；
- checkpoint 路径和 W&B run；
- 训练阶段及随机种子；
- 是否启用 teacher、BC、shield 和 observation/action noise。

回答任何“你具体做了什么”的问题时，只讲自己确实完成并能展开解释的部分。团队完成的内容可以说“我们”，个人负责的部分再说“我负责”。

## 1. 先背熟的项目介绍

### 1.1 30 秒版本

ForesightNav 是一个面向无人机局部导航的强化学习系统，重点处理两类问题：动态障碍物需要提前规避，而大型墙体和非凸结构容易让只看局部距离的策略陷入停滞。系统在 NavRL 式 PPO 导航框架上加入两种训练期机制：有限时域、TTC-aware 的 3D 速度障碍风险奖励，用相对运动而非仅距离来刻画动态碰撞风险；以及动作条件的多时间尺度结果预测，用碰撞、局部受困、最小净空和目标进度监督共享表示。辅助预测器只在训练时使用，部署时仍由轻量策略直接输出三维速度。

### 1.2 2 分钟版本

项目解决的是部分可观测环境中的无人机点到点局部导航。输入包括局部距离扫描、目标与自身运动状态，以及最近动态障碍物的相对位置、速度和尺寸；输出是目标对齐坐标系中的三维速度命令，经过坐标变换和低层飞控执行。

普通距离奖励有两个缺陷。第一，相同距离的障碍物可能正在靠近、远离或从高度上错开，危险程度并不相同。第二，面对长墙或 U 形结构时，朝目标方向移动可能长期不可行，而瞬时“安全”的停住或来回摆动又不一定受到充分惩罚。因此，ForesightNav 用 3D-VO 根据相对位置、相对速度、障碍物尺度和 TTC 构造有限时域风险；同时让共享编码器预测多个未来窗口内的碰撞、受困、净空和进度，使表示包含短期避碰与较长期绕行所需的信息。

训练采用 PPO，是因为动作连续、Isaac Sim 能提供大量并行 on-policy 样本，而 PPO 的 clipped update 在课程变化和随机环境中较稳定、实现成本也低。仿真训练使用 1,024 个并行环境，部署时采用 Beta 策略均值直接输出速度；可选的 VO/ORCA safety shield 只负责修正残余危险动作。Gazebo 中的消融结果显示，3D-VO 对动态避碰提升更明显，MH 对大型非凸结构中的超时和死锁改善更明显，两者互补但并非完全互斥。

### 1.3 一句话技术主线

**局部感知 -> 共享 256 维表示 -> Beta actor 输出三维速度；训练时再用 TTC 风险奖励和多时域结果监督，让策略既能提前避开动态冲突，也更容易持续绕过大型非凸结构。**

## 2. 问题定义与系统边界

### 2.1 这是规划还是控制

它更准确地说是**学习式局部导航策略**：

- 输入局部观测和目标信息；
- 直接输出三维速度命令；
- 不显式构建全局路径，也不在线展开未来轨迹树；
- 低层姿态/推力控制由 `LeePositionController + VelController` 等控制器完成；
- 部署端可选 shield 对策略动作做安全修正。

因此不要把它描述成全局规划器，也不要声称解决了未知大尺度环境中的完备路径规划。

### 2.2 为什么是 POMDP

策略只能看到有限量程的局部感知、有限数量的动态障碍物和当前运动状态，看不到完整地图、障碍物真实意图或未来运动，因此同一观测可能对应不同的全局状态。形式上可写为：

\[
o_t \sim \mathcal O(s_t),\qquad
a_t\sim\pi_\theta(\cdot\mid o_t),\qquad
\max_\theta\;\mathbb E\!\left[\sum_t\gamma^t r_t\right].
\]

项目没有通过显式 belief filter 完整求解 POMDP，而是依靠局部运动描述、课程随机化和辅助预测监督学习实用策略。

### 2.3 项目不保证什么

- 没有形式化安全保证；shield 也不等于绝对安全。
- 3D-VO 假设有限时域内相对速度近似恒定。
- 局部观测下仍可能发生感知混淆、长时域错误和无解场景。
- 实飞结果主要证明可部署性；若没有大规模实飞统计，不应把它说成严格性能验证。

## 3. 端到端数据流

### 3.1 训练链路

1. Isaac Sim 并行生成 UAV、静态几何、动态障碍物和任务。
2. `NavigationEnv` 计算局部观测、碰撞、进度、VO 风险和其他奖励。
3. 共享编码器融合 LiDAR、导航状态和动态障碍物描述。
4. Beta actor 采样归一化动作，映射为目标坐标系速度，再旋转到世界坐标系。
5. `VelController` 和 Lee 控制器将速度命令转为飞行器执行量。
6. 每个并行环境收集 32 步 rollout，使用 GAE 计算 advantage。
7. PPO、critic 和辅助结果预测联合更新共享编码器。
8. 周期性使用策略均值评估，并保存 checkpoint。

### 3.2 部署链路

1. ROS 节点获得里程计、局部距离信息和动态障碍物跟踪结果。
2. 按训练时相同的观测结构和坐标定义进行预处理。
3. actor 直接输出速度，辅助 outcome decoder 不部署。
4. 可选 safety shield 构造 ORCA/VO 约束并最小修正危险动作。
5. 飞控跟踪最终速度命令。

训练与部署最容易出问题的不是神经网络本身，而是**观测语义、单位、坐标系、更新频率、速度限制和障碍物尺寸定义不一致**。

## 4. 观测、动作与网络

### 4.1 三类策略观测

当前 `env.py` 定义的主要输入为：

| 输入 | 维度 | 含义 |
|---|---:|---|
| 局部距离扫描 | `1 x 36 x 4` | 4 m 量程；水平 10 度分辨率，共 36 束；垂直视场 `[-10,20]` 度，共 4 束 |
| navigation state | `8` | 目标单位方向 3、水平目标距离 1、带符号高度差 1、目标对齐坐标系速度 3 |
| dynamic obstacle descriptors | `1 x 5 x 10` | 最近 5 个动态障碍物，每个含相对单位方向 3、水平距离 1、垂直距离 1、目标对齐坐标系中的障碍物速度 3、宽度类别 1、高度描述 1 |

另有一个 `direction (1 x 3)` 字段用于把局部动作旋转回世界坐标系；它没有直接拼接进共享特征。

### 4.2 为什么使用目标对齐坐标系

将局部 x 轴固定为起点到目标的水平朝向，有三个工程收益：

- 相同几何关系在不同世界朝向下具有相近表示，降低旋转方差；
- actor 的前进、侧移和垂直动作语义稳定；
- 有利于 sim-to-sim 和 sim-to-real 时统一坐标处理。

代价是坐标变换必须完全一致，并需处理目标方向接近零向量的数值问题。

### 4.3 编码网络

- 距离扫描：三层卷积后展平，投影为 128 维。
- 动态障碍物：`5 x 10` 展平，经 MLP 投影为 64 维。
- 拼接：`128 + 8 + 64 = 200` 维。
- 共享编码：两层 MLP 得到 256 维 \(z_t\)。
- actor：从 256 维特征产生每个动作维度的 Beta 分布参数。
- critic：从 256 维特征估计标量状态价值。
- outcome decoder：输入 `256 维特征 + 3 维已采样动作`，预测各 horizon 的四类结果。

### 4.4 为什么用 Beta 策略

每个动作分量服从 Beta 分布：

\[
u_t^k\sim\mathrm{Beta}(\alpha_k,\beta_k),\quad u_t^k\in[0,1],
\]

再映射为：

\[
a_t^k=v_{\mathrm{axis}}(2u_t^k-1),\quad
v_{\mathrm{axis}}=2.0\ \mathrm{m/s}.
\]

优势：

- 分布天然有界，不需要采样后硬裁剪；
- 避免高斯动作越界后，执行动作与 log-prob 对应动作不一致；
- 均值可直接用于确定性评估。

局限：

- 三个轴独立建模，不能显式表达轴间相关性；
- 单个 Beta 分布难以表达多峰动作选择，例如“向左绕或向右绕”两个同等模式；
- 参数限制为大于 1 时偏向单峰内部动作，对边界动作的表达更保守。

**速度上限口径：** `2.0 m/s` 是仿真中每个笛卡尔分量的上限，不是速度模长上限；理论最大模长是 \(2\sqrt 3\approx3.46\,\mathrm{m/s}\)。实飞部分另对平移速度模长限制为 `1.5 m/s`。

## 5. 为什么使用 PPO

### 5.1 PPO 在这里解决什么

PPO 是 on-policy actor-critic 方法：actor 表示动作分布 \(\pi_\theta(a_t\mid o_t)\)，critic 估计当前观测下的期望折扣回报 \(V_\psi(o_t)\)，advantage \(A_t\) 则衡量某个动作相对当前平均水平好多少。`on-policy` 表示更新数据来自当前策略或非常接近的旧策略，更新后不会长期反复使用历史 replay 数据。

动作是连续三维速度，奖励来自长时序导航结果，环境又能在 GPU 上并行产生大量轨迹。PPO 的 clipped surrogate objective 为：

\[
r_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)},
\]

\[
L_{\mathrm{clip}}=\mathbb E_t\left[
\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
\]

当新策略相对旧策略变化过大时，裁剪项限制继续增加 surrogate gain，从而降低一次更新把策略推坏的概率。这里 \(\epsilon=0.1\)，属于相对保守的更新。

### 5.2 GAE 是什么

TD residual：

\[
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
\]

GAE：

\[
\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
\]

本项目使用 \(\gamma=0.99\)、\(\lambda=0.95\)。较大的 \(\lambda\) 使用更长的回报信息、降低 value bias，但会增加方差；GAE 在二者之间折中。代码还会标准化 advantage，并对 return 做运行统计归一化。

一次更新中的主要优化项是：

\[
\mathcal L_{\mathrm{total}}=
\mathcal L_{\mathrm{actor}}+
\mathcal L_{\mathrm{critic}}+
\mathcal L_{\mathrm{entropy}}+
\mathcal L_{\mathrm{MH}}
\quad(+\mathcal L_{\mathrm{BC}}\text{，若启用后续增强}).
\]

critic 使用 value clipping 和 Huber loss；entropy 项鼓励探索；所有网络梯度范数裁剪为 5。注意代码以“最小化 loss”实现，所以 entropy loss 前带负系数。

### 5.3 为什么 PPO 适合本项目

- **连续控制自然：** 可直接优化随机连续动作策略。
- **训练稳定性较好：** clipping 比无约束 policy gradient 更抗大步更新。
- **并行仿真匹配：** PPO 样本效率不高，但 1,024 个 GPU 环境能快速产生新鲜 on-policy 数据。
- **实现和调试成本适中：** 相比 TRPO 不需要二阶优化和显式 KL 约束求解。
- **便于联合辅助任务：** 共享编码器可同时接收 PPO 与 MH loss 的梯度。
- **课程训练更直接：** 不使用长期 replay buffer，减少旧课程数据与当前环境分布不一致的问题。

### 5.4 PPO 的缺点

- on-policy，采样效率通常低于 SAC/TD3；
- 对奖励尺度、学习率、clip ratio、熵系数和 curriculum 较敏感；
- clipping 是启发式近似，不提供严格单调改进保证；
- 仍可能局部最优、停滞或学到 reward hacking；
- PPO 本身不提供安全保证。

### 5.5 与常见算法比较

| 方法 | 相对 PPO 的优点 | 在本项目中的主要代价 |
|---|---|---|
| SAC | off-policy、样本复用高、熵正则探索强 | replay 中旧课程样本可能过时；Q 学习和温度参数更难调，吞吐未必是瓶颈 |
| TD3 | 样本效率较高，缓解 Q 过估计 | 确定性策略探索依赖噪声，对多模态绕行和分布变化较敏感 |
| DDPG | 结构简单、连续动作 | 对超参数和 Q 误差敏感，训练稳定性通常更弱 |
| TRPO | trust region 理论约束更直接 | 二阶计算和实现复杂，大规模并行训练成本高 |
| A2C/A3C | 简单、on-policy | 没有 PPO clipping，更新稳定性通常较弱 |
| DQN | 离散动作上成熟 | 不能原生处理连续三维速度，离散化会带来组合爆炸和控制粗糙 |

正确回答不是“PPO 永远优于 SAC”，而是：**本项目的主要瓶颈不是仿真样本数量，而是大规模随机课程中的稳定训练和工程可控性，所以 PPO 是合理折中。**

### 5.6 PPO 关键参数如何解释

| 参数 | 值 | 面试解释 |
|---|---:|---|
| 并行环境 | 1,024（正式训练覆盖配置） | 提高 on-policy 吞吐和样本多样性；仓库默认 `2` 仅便于开发 |
| rollout length | 32 | 每次每环境收集 32 个 transition |
| 每批 transition | `1024 x 32 = 32768` | 若启用 32 个 teacher 环境，纯 PPO 部分为 `992 x 32 = 31744` |
| PPO epochs | 4 | 每批数据重复优化 4 轮 |
| minibatches | 16 | 每轮切 16 份，共 64 次梯度更新 |
| actor clip | 0.1 | 限制策略概率比更新幅度 |
| critic clip | 0.1 | 限制 value 相对旧预测的变化 |
| \(\gamma\) | 0.99 | 长期回报权重 |
| GAE \(\lambda\) | 0.95 | advantage 偏差/方差折中 |
| entropy coefficient | `1e-3` | 防止策略过早塌缩，保持探索 |
| 初始学习率 | encoder/actor/critic 均 `5e-4` | 课程初期快速学习 |
| 适应阶段学习率 | `1e-5/1e-5/1e-4` | 保护已学表示和策略，允许 critic 更快适应新分布 |
| gradient clip | 5.0 | 防止联合损失导致梯度爆炸 |

## 6. TTC-Aware 3D Velocity Obstacle

### 6.1 为什么仅距离奖励不够

距离相同的两个障碍物可能：

- 一个高速靠近，另一个正在远离；
- 一个与 UAV 高度重叠，另一个从上方或下方通过；
- 一个尺寸很大，另一个很小。

只按距离惩罚无法区分这些情况，容易等到障碍物很近才反应。VO/TTC 将相对运动和几何纳入风险估计。

### 6.2 相对运动和各向异性膨胀

令障碍物相对 UAV 的位置和速度为：

\[
\mathbf p=\mathbf p_o-\mathbf p_u,\qquad
\mathbf v=\mathbf v_o-\mathbf v_u.
\]

根据障碍物水平尺寸和高度构造安全尺度：

\[
\mathbf r=[r_{xy},r_{xy},r_z],
\]

其中当前配置在水平和竖直方向各增加 `0.3 m` margin。归一化后：

\[
\bar{\mathbf p}=\mathbf p\oslash\mathbf r,\qquad
\bar{\mathbf v}=\mathbf v\oslash\mathbf r.
\]

碰撞边界满足：

\[
\|\bar{\mathbf p}+T\bar{\mathbf v}\|_2^2=1,
\]

展开得到：

\[
AT^2+BT+C=0,
\]

\[
A=\|\bar{\mathbf v}\|^2,\quad
B=2\bar{\mathbf p}^{\top}\bar{\mathbf v},\quad
C=\|\bar{\mathbf p}\|^2-1.
\]

若障碍物正在靠近、判别式非负且最早正根处于有限 horizon 内，则：

\[
T_{\mathrm{TTC}}=\frac{-B-\sqrt{B^2-4AC}}{2A}.
\]

### 6.3 风险函数

当前实现的核心风险为：

\[
\rho=\begin{cases}
1, & C\le0,\\
\exp(-T_{\mathrm{TTC}}/\tau_{\mathrm{VO}}), & 0<T_{\mathrm{TTC}}\le H_{\mathrm{VO}},\\
0, & \text{otherwise}.
\end{cases}
\]

并取所有保留 primitive 的最大风险，奖励为：

\[
r_{\mathrm{VO}}=-\lambda_{\mathrm{VO}}w_{\mathrm{warmup}}\rho.
\]

关键参数：

- \(H_{\mathrm{VO}}=2.1\,\mathrm{s}\)：只关心有限时域冲突；
- \(\tau_{\mathrm{VO}}=0.75\,\mathrm{s}\)：TTC 越小，风险越接近 1；
- \(\lambda_{\mathrm{VO}}=1.3\)：奖励权重；
- warmup `20,000` steps：逐渐引入 VO 奖励，避免训练初期压过基础学习信号；
- reward top-k `10`：限制参与风险计算的近邻障碍物数量。

风险在 \(H_{\mathrm{VO}}\) 处采用硬截断，因此应称为 **horizon-truncated, TTC-graded risk**，不能说成数学上处处连续的风险函数。

### 6.4 为什么用有限 horizon

- 远期基于恒速假设的预测误差更大；
- 过长 horizon 会对很多实际上可轻松修正的轨迹过度惩罚；
- 局部策略更适合处理近期冲突，远期绕行由持续闭环决策和 MH 表示学习承担。

### 6.5 高障碍物如何近似

代码可把高棱柱沿 z 轴展开为 sphere stack，再对相关 primitive 计算风险。这样比把高障碍物压成单个球更能保留垂直几何，同时仍可使用统一的二次方程 TTC 检验。未来垂直运动预筛选会排除在 horizon 内始终不可能发生高度重叠的球，从而降低无关计算和误惩罚。

### 6.6 3D-VO 的创新边界

不要说“我们首次提出 3D VO”。更严谨的说法是：

> 项目的贡献是把有限时域、各向异性障碍物几何和 earliest-TTC 风险组合成训练期监督，并与策略表示学习结合；它不是在线 VO 规划器，也不是对经典 VO 概念本身的发明。

### 6.7 训练奖励和部署 shield 的区别

| TTC-aware 3D-VO reward | Safety shield |
|---|---|
| 只在训练时参与 reward | 部署时位于 actor 之后 |
| 使策略参数内化提前避碰倾向 | 对单次危险命令做约束修正 |
| 不直接保证每个动作安全 | 提高残余安全性，但仍受感知和模型假设限制 |
| 不增加部署端预测器开销 | 增加在线障碍物处理和优化开销 |

当前奖励总式里真正启用的是基于实际相对运动的 `reward_vo`。代码虽然还计算 action-VO、static-VO 和 safe-action 项，但在最终 reward 中乘了 `0`；面试时不能仅看到 YAML 中有非零 weight 就说这些项已经生效。

## 7. Action-Conditioned Multi-Horizon Outcome Learning

### 7.1 动机

标量总奖励把碰撞、受困、净空和进度混在一起，编码器只能间接学习这些因素。MH 直接要求共享表示回答：

> 在当前观测下执行这个动作后，未来一段时间是否会碰撞或受困，最小净空是多少，能取得多少目标进度？

这种训练信号比单一即时 reward 更结构化，同时不要求部署时在线 rollout。

### 7.2 四类预测目标

对每个 horizon，decoder 预测：

1. 是否发生 collision；
2. 是否发生 local trapping/stuck；
3. 窗口内最小 forward clearance；
4. 窗口内累计 goal progress。

分类项使用 BCE，连续项先 `symlog` 再使用 Smooth L1：

\[
\operatorname{symlog}(x)=\operatorname{sign}(x)\log(1+|x|).
\]

`symlog` 压缩大数值，同时保留正负号；Smooth L1 对异常 target 比 MSE 更稳健。

### 7.3 为什么要 action-conditioned

只从 \(z_t\) 预测未来，网络容易学习“这个场景通常危险”，却不区分当前选了什么动作。将已采样动作 \(a_t\) 拼到 decoder 输入后，预测目标与当前决策建立联系。

但它不是严格的反事实模型：只显式条件化当前动作，后续动作仍来自采集 rollout 的策略。正确表述是“action-conditioned outcome prediction”，不是“能够任意模拟所有候选动作的 world model”。

### 7.4 梯度如何流动

\[
\mathcal L=\mathcal L_{\mathrm{PPO}}+\lambda_{\mathrm{MH}}\mathcal L_{\mathrm{MH}}.
\]

- outcome targets `detach`，不通过标签反向传播；
- rollout 中的当前动作视为固定数据；
- MH loss 更新 decoder 和共享 encoder；
- actor/critic head 不直接接收 MH loss；
- 但 actor 和 critic 会在之后使用被 MH 塑造过的共享表示；
- decoder 在部署时移除，因此不增加在线推理路径。

### 7.5 为什么多个 horizon

- 短 horizon 更关注立即碰撞和净空；
- 中期 horizon 捕捉避让趋势；
- 长 horizon 更适合描述持续受阻、绕行和累计进度；
- 多尺度共同监督可减少只优化单一时间尺度造成的短视。

### 7.6 物理时间尺度与跨 rollout 实现

当前训练代码已经与论文的方法定义对齐：

- 配置使用 `{0.25,1.0,2.0} s`，并根据策略转移间隔 `env.dt * env.substeps` 转换为步数；
- 在 `dt=0.016 s` 下对应 `{16,63,125}` 步；
- 每个环境维护 episode-aware sequence buffer，未成熟样本跨 PPO rollout 保留；
- 最长未来窗口完整或 episode 提前结束时才生成监督，标签不会跨 reset；
- trapping target 由未来窗口的 blockage 比例、净水平位移速度和目标进展速度共同定义；
- PPO loss 仍只使用当前 rollout，延迟成熟样本只参与辅助损失并重新通过共享 encoder。

必须区分“实现已修正”和“实验已验证”。旧 checkpoint 和既有表格数据可能来自早期 `[1,3,5]` 实现；在完成新实现的重新训练与消融复评前，不能声称旧数据证明了 2 秒跨 rollout 监督。面试时应明确版本和结果来源：

> 当前分支已经实现论文定义的物理时间多尺度监督和跨 rollout 标签缓冲；下一步是用该版本重新训练完整方法与消融模型，并将 checkpoint、resolved config、commit 和结果绑定归档。

## 8. 奖励与终止条件

### 8.1 当前代码的有效奖励

当前实现可概括为：

\[
\begin{aligned}
r_t={}&0.1
-0.6p_{\mathrm{static}}
-0.6p_{\mathrm{dynamic}}
-0.1p_{\mathrm{smooth}}
-0.5p_{\mathrm{alt}}\\
&+r_{\mathrm{goal}}
+r_{\mathrm{escape}}
+r_{\mathrm{stall}}
+r_{\mathrm{VO}}
+r_{\mathrm{terminal}}.
\end{aligned}
\]

| 项 | 当前含义 |
|---|---|
| alive | 每步 `+0.1` |
| static proximity | 1.1 m margin 内按距离线性惩罚，并对扫描束求均值 |
| dynamic proximity | 1.1 m margin 内按最近动态障碍物距离惩罚 |
| smoothness | `-0.1 ||v_t-v_{t-1}||` |
| altitude | 目标高度误差超过 0.4 m 后平方惩罚 |
| goal progress | `35 * (d_{t-1}-d_t)` |
| arrival | 进入 0.5 m 目标半径时该项设为 `+50` |
| timeout | 达到时间上限时该项设为 `-45` |
| collision | 额外 `-45` 并终止 |
| vertical/horizontal out of bound | 各额外 `-20` 并终止 |
| escape | 识别到受阻后满足恢复证据时 `+3` |
| stall | 持续低运动、低进度后渐增到每步 `-1` |
| TTC-aware VO | `-1.3 * warmup * risk` |

代码中的 `reward_vel` 当前乘以 `0`，不能说它是有效的朝向速度奖励。action-VO、static-VO 和 safe-action reward 也在最终求和时乘以 `0`。

### 8.2 为什么同时需要 progress、stall、escape 和 MH

- progress 是通用的稠密任务信号；
- stall penalty 直接惩罚长期几乎不动；
- escape reward 对“已经受阻但成功恢复”给出稀疏正反馈；
- MH 不是额外行为 reward，而是让共享表示显式编码未来结果。

它们存在语义重叠，因此必须通过消融验证，而不能仅凭设计宣称每项都独立有效。

### 8.3 当前 stuck 与恢复定义

当前训练配置中，若前方存在障碍且单步目标进度不超过 `0.005 m`，连续 `60` 步后进入 stuck。恢复事件在受阻触发后最多观察 `180` 步，要求：

- 离开低进度阻塞状态；
- 水平位移至少 `0.3 m`；
- 并且目标距离减少至少 `0.2 m`，或前方净空增加至少 `0.2 m`。

论文评估指标使用 20 Hz 下连续 40 周期，即 2 秒的事件定义。训练实现与评估协议的周期数目前并不完全一致，回答时应说明它们用途不同，或者在最终版本中统一。

## 9. 任务采样、环境和课程训练

### 9.1 默认起终点规则

- 从四条边均匀随机选择起点所在边；
- 边界坐标为 `x=+/-21 m` 或 `y=+/-21 m`；
- 切向坐标在 `[-18,18] m` 采样；
- 目标位于正对边，切向位置接近中心对称点，并加入 `+/-4 m` jitter；
- 起点和目标高度分别在 `[0.5,2.5] m` 采样；
- 起终点尝试满足静态和动态障碍物 `1.5 m` clearance；
- 最多拒绝采样 128 轮；
- 水平坐标超过 `+/-22 m` 会终止，防止策略沿障碍区外缘绕开任务。

`wall_crossing` 模式在上述基础上，进一步要求起终点线段穿过大墙体的膨胀影响区。它用于增加大型结构任务的针对性，不是默认模式。

代码在采样次数耗尽后存在 fallback，因此“1.5 m clearance”是正常采样目标，不是任何情况下的数学硬保证。

### 9.2 动态障碍物

- 当前开发配置默认 80 个，论文初始 curriculum 环境描述为 60 个；
- 速度范围 `0.5-1.5 m/s`；
- 宽度类别约为 `0.25/0.50/0.75/1.0 m`；
- 同时包含可三维绕行的矮 cuboid 和近似贯穿高度的长 cylinder；
- 障碍物在局部范围内随机选择目标，并约每 2 秒重采样速度大小；
- 数量会按 8 个尺寸/高度类别取整为 8 的倍数。

### 9.3 为什么做 curriculum

直接在最高密度动态障碍物和复杂非凸结构中从随机策略开始，碰撞和无效轨迹占比高，回报信号质量差。课程训练先学习基本目标趋近和局部避障，再逐渐增加动态密度和大型结构，可改善早期探索和训练稳定性。

当前课程更接近“分阶段调整配置并从 checkpoint 继续训练”，不是一个自动 curriculum scheduler。适应阶段把 encoder/actor 学习率降到 `1e-5`，critic 保持 `1e-4`，目的是减少对已有策略的破坏，同时让价值函数更快适应新任务分布。

### 9.4 当前分支的 teacher/BC

当前默认配置还包含后续探索：

- 最多 32 个环境由局部 wall-following teacher 生成 demonstration；
- 这些环境从 PPO/GAE 数据中排除，避免破坏 on-policy 假设；
- demonstration 存入最多 100,000 条的 buffer；
- 每轮最多 8 次 BC 更新，loss weight 为 `0.05`；
- 其余环境继续纯 on-policy PPO。

这是合理的工程设计，因为 teacher action 不能冒充 PPO 自己采样的动作。但它不属于当前论文的两个核心机制，除非对应实验确实重新训练并报告，否则不要写进论文结果解释。

## 10. 实验设计与结果怎么讲

### 10.1 训练和测试设置

- Isaac Sim 训练区域：`40 m x 40 m`；
- 训练正式配置：1,024 个并行 UAV，RTX 4090；
- 静态障碍物数量：当前训练配置 350；
- 动态障碍物：论文课程从 60 开始，当前默认配置 80；
- Gazebo 测试区域：`20 m x 20 m`；
- 每种方法、每种环境 1,000 次 matched trials；
- 测试终止：成功、碰撞或 `30 s` timeout；
- 起点在四边随机，目标位于正对边；
- PE-Planner 使用原生 pipeline，学习方法使用相同 shield；另有 shield on/off 实验。

### 10.2 主要指标

- Success Rate：在碰撞或超时前到达目标；
- Collision Rate：因障碍物碰撞终止；
- Timeout Rate：时限内既未到达也未碰撞；
- Deadlock Event Count：进入持续阻塞的事件数；
- Deadlock-Event Recovery Rate：死锁事件中满足恢复标准的比例。

前三项应满足：

\[
\mathrm{Success}+\mathrm{Collision}+\mathrm{Timeout}=100\%.
\]

恢复率分母是 deadlock event，而不是 trial；无事件时应写 `N/A`，不能写 0%。

### 10.3 消融结果支持什么

把 `Backbone` 记为两个机制都关闭：

| 环境 | Backbone | 仅 3D-VO（w/o MH） | 仅 MH（w/o 3D-VO） | Full |
|---|---:|---:|---:|---:|
| Dynamic success | 68.3 | 81.6 | 69.7 | 83.3 |
| Large non-convex success | 64.5 | 68.0 | 75.9 | 78.1 |
| Large non-convex timeout | 20.2 | 16.9 | 10.4 | 6.3 |
| Large non-convex deadlocks | 102 | 113 | 72 | 53 |

可支持的结论：

- 在动态环境中，加入 3D-VO 的两个匹配对分别提升 13.3 和 13.6 个百分点，说明它是动态避碰的主要贡献者；
- 在大型非凸环境中，加入 MH 的两个匹配对分别提升 11.4 和 10.1 个百分点，并显著降低 timeout/deadlock，说明它更主要作用于持续绕行和局部受困；
- full method 在两类任务中总体最好，支持二者互补；
- 不能说 3D-VO 只影响动态避障、MH 只影响非凸绕行，因为两者在另一类任务中也可能产生影响。

### 10.4 Shield 结果怎么解释

动态环境中：

- NavRL：shield off/on success 为 `63.5/69.9%`；
- ForesightNav：shield off/on success 为 `71.4/83.3%`。

这说明：

- shield 对两种学习策略都有效；
- 无 shield 的 ForesightNav 仍高于有 shield 的 NavRL，收益不完全由 shield 造成；
- full result 明显受 shield 帮助，因此主表必须明确比较协议，不能暗示完全是裸策略性能。

更完整的 shield 分析还应统计介入频率、平均修正幅度和耗时；当前论文只报告终端结果，证据仍有限。

### 10.5 实验局限要主动承认

- 1,000 trials 降低测试随机性，但不能替代多个独立训练 seed；
- 外部 baseline 数量有限，“全面优于所有 SOTA”是过强结论；
- learned methods 使用共同 shield，而 PE-Planner 使用原生 pipeline，属于系统级比较，不是纯策略组件比较；
- 实飞主要为四种场景的定性验证，缺少大规模成功率与置信区间；
- dynamic obstacle 的恒速近似、感知误差和控制延迟可能影响真实部署。

## 11. Sim-to-Sim 与 Sim-to-Real

### 11.1 为什么 Isaac 评估好，Gazebo 可能明显变差

按排查优先级：

1. **观测定义不一致：** 距离值是 `distance` 还是 `range-distance`，动态障碍物尺寸是半径还是直径，越界目标如何补零。
2. **坐标系错误：** world/body/goal frame、yaw 正方向、z 轴和左右方向不一致。
3. **控制频率不一致：** 训练 `dt=0.016 s`，ROS 常见 20 Hz；同一动作持续时间和滤波效果不同。
4. **动作限制不一致：** per-axis 2 m/s、速度模长 1.5 m/s、飞控饱和或加速度限制混用。
5. **动力学和控制延迟：** Isaac 中理想速度跟踪与 Gazebo/实机惯性、超调和通信延迟不同。
6. **感知差异：** detector 噪声、遮挡、追踪 ID 跳变、速度估计抖动和更新延迟。
7. **任务分布不一致：** 起终点、地图边界、障碍密度、高度和速度分布不同。
8. **shield 参数不一致：** horizon、安全半径、最大速度、静态障碍物开关不同。
9. **checkpoint/config 不匹配：** 动态障碍物数、网络输出维度或归一化定义改变，却使用 `strict=False` 静默加载。

### 11.2 为什么加噪声不等于完成 sim-to-real

Domain randomization 只能覆盖预先定义的误差。真实系统还包含结构性偏差，例如时间戳错位、控制闭环动态、传感器视场、检测漏失和障碍物行为变化。因此还需要接口一致性测试、延迟测量、闭环日志和逐层消融。

### 11.3 部署时应记录什么

- actor 原始命令和 shield 后命令；
- shield 介入率和修正量；
- 里程计、目标、观测与命令时间戳；
- 实际速度、加速度和控制饱和；
- 最近障碍物距离、预测 TTC；
- 成功/碰撞/超时/人工接管；
- 单次推理耗时和完整闭环频率。

## 12. 关键参数速查表

| 类别 | 参数 | 值/说明 |
|---|---|---|
| 仿真 | physics `dt` | `0.016 s`，约 62.5 Hz |
| 仿真 | max episode | 2,200 steps，按该 dt 为 35.2 s；Gazebo 论文协议为 30 s |
| 感知 | local range | 4 m |
| 感知 | LiDAR shape | `1 x 36 x 4` |
| 感知 | dynamic descriptors | 最近 5 个，每个 10 维 |
| 动作 | policy action | 三维速度，目标坐标系输出后转到世界系 |
| 动作 | simulation bound | 每轴 `+/-2.0 m/s` |
| 动作 | physical bound | 平移速度模长 `1.5 m/s` |
| PPO | rollout/epochs/minibatches | `32 / 4 / 16` |
| PPO | gamma/lambda | `0.99 / 0.95` |
| PPO | actor/critic clip | `0.1 / 0.1` |
| PPO | entropy coefficient | `1e-3` |
| 网络 | shared latent | 256 维 |
| MH | loss weight | `0.1` |
| MH | decoder hidden | 128 |
| MH | physical horizons | `{0.25,1.0,2.0} s`，在 `dt=0.016 s` 下为 `{16,63,125}` 步 |
| VO | horizon/tau | `2.1 / 0.75 s` |
| VO | xy/z margin | `0.3 / 0.3 m` |
| VO | weight/warmup | `1.3 / 20,000 steps` |
| goal | radius/arrival | `0.5 m / +50` |
| failure | collision/timeout | `-45 / -45` |
| task | start/goal boundary | `+/-21 m`，目标在正对边 |
| task | tangent/jitter | `+/-18 m / +/-4 m` |
| task | horizontal termination | `+/-22 m` |
| obstacles | static count | 当前配置 350 |
| obstacles | dynamic speed | `0.5-1.5 m/s` |

## 13. 高频面试问题与参考回答

### Q1：你的核心贡献是什么

不是重新发明 PPO 或 VO，而是针对局部学习导航的两个具体缺口：用有限时域、各向异性 3D-VO 的 TTC 风险作为训练监督，提升动态冲突的提前响应；用动作条件的多时域结果任务塑造共享表示，改善长墙和非凸结构附近的持续绕行与局部受困。两者只用于训练，部署仍保持直接 policy inference。

### Q2：为什么普通距离奖励不能解决动态避障

距离没有方向和时间信息。相同距离下，靠近、远离和高度错开的风险完全不同。TTC 基于相对位置和速度估计首次冲突时间，能在障碍物尚未很近时区分风险。

### Q3：为什么不用直接把 TTC 输入网络

把 TTC 作为输入可以提供显式信息，但仍需要策略自己从稀疏回报中学会如何使用，而且部署依赖 TTC 估计质量。这里把它作为训练 reward，直接塑造行为；也可以把 TTC 输入作为后续对照实验，但需防止与 reward 同时使用后无法分离贡献。

### Q4：VO 的恒速假设现实吗

它在短 horizon 内是常用近似，不是对未来运动的精确保证。有限 `2.1 s` horizon、闭环重复计算和 safety shield 能降低误差影响；对于强加速或意图变化的障碍物，仍需要更好的运动预测或不确定性建模。

### Q5：3D-VO 为什么不是单纯二维 VO 加 z

实现按水平尺寸和竖直高度分别归一化，构造各向异性安全椭球，并检查三维相对轨迹进入该椭球的最早时间。高障碍物还可用 sphere stack 近似，垂直错开会改变是否存在有效 TTC。

### Q6：为什么 MH 能帮助绕长墙

单步朝目标进度会偏好直冲、停住或局部摆动。MH 强迫共享表示区分一段时间内的净空、累计进度和受困结果，能让 critic/actor 获得更有行为意义的特征。消融中 MH 对大型非凸任务的 success、timeout 和 deadlock 改善更明显。

### Q7：MH 是 world model 吗

不是。它不重建下一状态，也不在部署时递归 rollout；它只从当前表示和动作回归几个任务相关的未来统计量，是训练期辅助监督。

### Q8：为什么不用 LSTM 解决部分可观测

LSTM 可以利用历史，但增加训练和部署状态管理，且并不保证学到碰撞或受困语义。MH 提供直接的未来结果监督且部署时可移除。二者并不互斥，未来可比较 feed-forward+MH、RNN、RNN+MH。

### Q9：为什么选择 PPO 而不是 SAC

Isaac Sim 能以 1,024 个环境提供大量新鲜样本，因此样本复用不是首要瓶颈；更重要的是课程随机环境下更新稳定和实现可控。PPO 的 clipped on-policy 更新适合这一点。SAC 可能更省样本，但 replay 分布、Q 估计和温度调参会增加复杂度。这个选择是工程折中，不是说 PPO 普遍更优。

### Q10：PPO clipping 真能保证稳定吗

不能形式化保证。它限制 surrogate objective 中概率比带来的收益，通常减少大更新，但还需学习率、advantage normalization、value clipping、gradient clipping 和监控 KL/entropy 配合。

### Q11：为什么采用 Beta 而不是高斯+tanh

Beta 天然定义在有界区间，动作和 log-prob 一致，适合明确速度边界。tanh-Gaussian 同样可用，但需正确处理变量变换 Jacobian；Beta 的不足是独立单峰分布难表达相关或多峰动作。

### Q12：训练时随机采样，评估为什么用均值

训练采样提供探索；评估使用 Beta 均值可以减少随机方差，使方法间比较更稳定，也符合部署希望输出确定命令的需求。

### Q13：你的 action limit 是 2 m/s 吗

仿真动作域是每个笛卡尔分量 `+/-2 m/s`，并非速度模长不超过 2。实飞另把平移速度模长限制为 1.5 m/s。两者要分开解释。

### Q14：为什么起终点放在正对边

它迫使轨迹穿过环境主体，减少无人机一直在边缘活动或采到过多简单短任务的问题；边界外再设置 `+/-22 m` 越界终止，防止从障碍区域外侧投机绕行。

### Q15：如何证明两个机制各自有效

用同训练协议的 2x2 消融：Backbone、仅 3D-VO、仅 MH、Full。在动态环境看 3D-VO 的 matched-pair 增益；在大型非凸环境看 MH 对 success、timeout 和 deadlock 的 matched-pair 增益，而不是只比较 full 与一个外部 baseline。

### Q16：为什么 full 的某个指标不一定全部最好

多目标导航存在权衡。例如更主动绕行可能提升 success 和减少 timeout，却不保证 collision 或 recovery ratio 每项都达到数值最优。应以预先定义的主指标和完整终止分布解释，避免挑选性汇报。

### Q17：shield 会不会掩盖策略能力

会影响最终系统表现，所以主表明确说明 learned methods 共同使用 shield，并另做 on/off 实验。无 shield 的 ForesightNav 仍超过有 shield 的 NavRL，说明收益不完全来自 shield；但更完整证据仍应报告 shield 介入率和修正幅度。

### Q18：训练奖励和 shield 都用 VO，是否重复

功能层级不同。训练 reward 改变策略分布，使其更早选择安全动作；shield 是部署时最后一道动作修正。理想策略会降低 shield 介入频率，但 shield 处理剩余风险。

### Q19：如何处理类别不平衡

collision 和 stuck 事件通常远少于安全帧。当前实现直接 BCE，可能被负样本主导。可改用正样本权重、focal loss、事件附近重采样，并分别监控 precision/recall，而不是只看总辅助 loss。

### Q20：为什么 continuous outcome 用 symlog

clearance 和累计 progress 的尺度与长尾不同。symlog 在零附近近似线性、对大值近似对数，并保留负进度，因此能降低少量大 target 对梯度的支配。

### Q21：你如何保证辅助 loss 不破坏 PPO

使用较小权重 `0.1`，分类/回归 target detach，梯度裁剪到 5，并监控 actor、critic、auxiliary loss 和 explained variance。更严格的做法是扫描 loss weight、检查 encoder 梯度夹角，或用分阶段/自适应权重。

### Q22：为什么训练要 1,024 个环境

PPO 需要大量新鲜 on-policy 样本。并行环境提高吞吐，同时让每批覆盖更多任务、障碍位置和运动状态，降低单条轨迹相关性。代价是显存、仿真资源和大 batch 下更新响应变慢。

### Q23：如何判断 reward 设计是否合理

不只看 return，要分别看 success/collision/timeout、平均进度、VO risk、stall、动作平滑度和 shield 介入；再做单项消融和轨迹可视化。若 return 上升但成功率不升，可能发生 reward hacking 或奖励尺度失衡。

### Q24：项目最难的工程问题是什么

可以从你真实经历中选一个深入讲，例如：训练与 Gazebo 的观测坐标和时间语义对齐；设计严格正对边任务避免边缘轨迹；把 teacher 环境从 PPO 数据中隔离以保持 on-policy；或定位 checkpoint/config 不匹配。回答需包含现象、假设、诊断日志、修复和验证，而不只说“调参数”。

### Q25：系统失败时你怎么排查

先分层：观测是否正确、坐标和单位是否一致、策略原始动作是否合理、shield 是否过度修正、低层控制是否跟踪、终止统计是否准确。记录中间张量和时间戳，用固定场景 replay；不要一开始就重新训练。

### Q26：当前方法最大的局限是什么

局部观测和恒相对速度假设限制长时域预测；没有形式安全保证；新版跨 rollout 监督仍需完整重训和多 seed 验证；实飞统计仍不足。未来可加入不确定运动预测、显式记忆、多机交互和更系统的安全验证。

### Q27：如果重新做一次，你会改什么

第一，建立配置/checkpoint/commit 的强绑定；第二，用已实现的物理时间跨 rollout MH buffer 重训完整方法和消融模型；第三，报告多训练 seed 和置信区间；第四，记录 shield 介入率和路径效率；第五，增加 sim-to-real 的延迟、噪声和动力学随机化。

### Q28：如何保证论文 horizon 和代码一致

配置保存物理时间 `{0.25,1.0,2.0} s`，策略用运行时控制间隔 `env.dt * env.substeps` 做半入舍入，在 `0.016 s` 下得到 `{16,63,125}` 步。最长时域超过 32 步 PPO rollout，因此按环境缓存未成熟样本，并在未来完整或 episode 结束时构造标签。运行开始会打印时间与步数映射；最终还应把 resolved config、checkpoint 和 commit 一起归档。

## 14. 白板上要能写出的内容

不看资料应能写出并解释：

1. PPO probability ratio 和 clipped objective；
2. TD residual 与 GAE；
3. Beta `[0,1] -> [-v_axis,v_axis]` 动作映射；
4. 3D 相对运动二次方程 `A T^2 + B T + C = 0`；
5. earliest positive TTC 和有限 horizon 条件；
6. `exp(-TTC/tau)` 风险的趋势与硬截断；
7. MH 的四个目标和 BCE/Smooth L1；
8. 完整训练 loss 的组成和梯度流向；
9. `Backbone / only 3D-VO / only MH / Full` 的 2x2 消融逻辑。

## 15. 代码地图

| 文件 | 面试前至少要会解释的内容 |
|---|---|
| `isaac-training/training/scripts/env.py` | 观测、起终点采样、动态障碍物、奖励、3D-VO、终止条件 |
| `isaac-training/training/scripts/ppo.py` | 编码器、Beta actor、critic、PPO loss、MH targets/loss、BC |
| `isaac-training/training/scripts/utils.py` | GAE、Beta 分布、坐标变换、评估统计 |
| `isaac-training/training/scripts/train.py` | 并行 collector、teacher 隔离、训练/评估/保存循环 |
| `isaac-training/training/scripts/eval.py` | checkpoint 加载、deterministic mean evaluation |
| `isaac-training/training/cfg/train.yaml` | 环境、任务、reward、teacher 和噪声参数 |
| `isaac-training/training/cfg/ppo.yaml` | 学习率、PPO、MH 和 BC 参数 |
| `ros1/navigation_runner/scripts/navigation.py` | 部署观测预处理、策略调用和 ROS 数据流 |
| `ros1/navigation_runner/include/navigation_runner/safeAction.cpp` | ORCA/VO plane、线性规划和 shield 修正 |
| `thesis/icra.tex` | 方法定义、实验协议、表格结果和论文声明边界 |

## 16. 简历写法模板

只有在与你的实际职责一致时才使用。推荐写成“任务 + 技术动作 + 可核验结果”，不要堆名词。

### 16.1 项目标题

**ForesightNav：面向动态障碍物与大型非凸结构的无人机强化学习导航**

### 16.2 三条项目描述模板

- 基于 PyTorch、TorchRL 与 NVIDIA Isaac Sim 搭建 GPU 并行 UAV 导航训练管线，使用 PPO/Beta 连续策略在 1,024 个随机环境中学习三维速度控制，并完成 Gazebo/ROS 部署接口。
- 设计有限时域 TTC-aware 3D-VO 训练奖励，将相对运动、障碍物三维尺度和 earliest TTC 转化为风险信号；动态环境消融中，相对匹配配置将成功率提升约 13 个百分点。
- 设计动作条件的多时域结果辅助任务，对未来碰撞、局部受困、最小净空和目标进度进行监督；大型非凸环境中将 full model 的成功率提升至 78.1%，并将 timeout 降至 6.3%。

如果你没有独立负责 ROS、实飞或某个算法模块，应改为“参与”并明确自己的具体部分。不要写“实现绝对安全”“解决通用自主导航”或“全面超越 SOTA”。

### 16.3 技术栈关键词

`Python`、`PyTorch`、`TorchRL`、`PPO`、`Isaac Sim`、`ROS1`、`Gazebo`、`CUDA/GPU parallel simulation`、`Velocity Obstacle`、`sim-to-real`。

## 17. 建议的一周复习顺序

### Day 1：讲清系统

- 不看稿完成 30 秒和 2 分钟介绍；
- 手画训练/部署数据流；
- 明确个人贡献、团队贡献和继承组件。

### Day 2：PPO

- 推导 clipped objective、GAE、critic loss；
- 解释所有 PPO 参数；
- 比较 PPO、SAC、TD3、TRPO。

### Day 3：3D-VO

- 从相对运动推导二次方程和 TTC；
- 解释各向异性尺度、sphere stack、horizon 和 tau；
- 准备恒速假设、硬截断和安全边界的回答。

### Day 4：MH 与网络

- 画出 200 -> 256 shared feature 和三个 head；
- 解释四类 targets、symlog、loss 和梯度流；
- 处理论文/代码 horizon 不一致，固定最终口径。

### Day 5：环境与部署

- 逐项核对 observation/action 的单位和坐标系；
- 解释起终点采样、课程训练和 reward；
- 复盘 Isaac->Gazebo/实机可能掉点的原因。

### Day 6：实验

- 背熟四组消融关系，不必死背全表；
- 能解释 success/collision/timeout/deadlock/recovery；
- 主动说出公平性、多 seed 和实飞统计局限。

### Day 7：模拟面试

- 随机抽取本文件 10 个问题，每题 1-2 分钟；
- 选择一个真实 bug 用 STAR 方式讲 5 分钟；
- 白板推导 PPO、TTC 和 MH loss；
- 最后检查简历中的每个数字都能指向实验表或日志。

## 18. 面试前最终检查清单

- [ ] 我能区分论文核心机制、NavRL 继承组件和当前分支后续增强。
- [ ] 我能解释 8 维 state、`36 x 4` 扫描和 `5 x 10` 动态描述。
- [ ] 我知道 `2 m/s` 是 per-axis bound，而实飞 `1.5 m/s` 是速度模长限制。
- [ ] 我能推导 PPO clipping、GAE 和 3D-VO TTC。
- [ ] 我能解释 Beta policy 的优缺点。
- [ ] 我能解释 MH 的四类 target、梯度流和部署时移除原因。
- [ ] 我已经解决或能诚实说明 MH horizon 的论文/代码版本差异。
- [ ] 我不会把 safety shield 或 teacher/BC 说成论文原创贡献。
- [ ] 我能用 matched pairs 解读消融，而不是只背 full model 数字。
- [ ] 我能说明单 seed、baseline 数量和实飞统计的证据边界。
- [ ] 我准备了一个真实工程 bug、一个失败实验和一个取舍案例。
- [ ] 简历上每项“负责、设计、实现、提升”的说法都与事实一致。
