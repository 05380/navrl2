# ForesightNav 项目交接文档

最后复核：2026-09-11（HEAD/upstream 为 `5e6540b`；当前 8 页论文与本次文档刷新尚未提交）

项目根目录：`/Users/yoloflps/Downloads/study/navrl2`

本文档用于在更换 Codex 账号或开启新会话后恢复项目上下文。它区分以下三类信息：

- **[当前文件]**：已经写入当前工作树，可直接从代码或论文源文件核对。
- **[已确认口径]**：对话中已经确认的研究、实验或写作口径，但未必全部固化在代码中。
- **[待完成/待验证]**：仍需实验、数据或作者决策才能完成，不能写成已证实结论。

若本文档与当前文件冲突，以当前代码、实验原始记录和 `thesis/icra.tex` 为准。不要把讨论设想当成已经实现或已经验证的结果。

### 本轮对话结论速览

- 论文主稿是 `thesis/icra.tex`；论文方法叙述的指定代码基准是 commit `6e137e9`，当前工作分支/HEAD 则是 `2-14-7-1` / `5e6540b`。
- 当前工作树中的论文已压缩到 8 页，正文无 Algorithm；这批论文精简和最新预览尚未提交，不能只检出 HEAD 就认为拿到了当前稿。
- 训练端 `opposite_crossing_eval` 是“四边均匀随机起点、目标在正对边且靠近中心对称点”的模式；训练边界为 `±21 m`，切向范围 `±18 m`，目标切向抖动 `±4 m`。
- 当前最新可见的本地仿真部署快照是 `/Users/yoloflps/Downloads/ros1 2 18.27.45`。旧的 `/Users/yoloflps/Downloads/ros1 2` 在最后复核时已经不存在，不能继续作为可核验路径引用。
- Ubuntu 真正运行的代码位于 `/home/wzf/navrl1_ws/src/ros1`。本机修改不会自动同步到 Ubuntu；每次正式评估都必须核对 `rospack find` 和 SHA-256。
- 时间戳部署快照的 `deployment_eval.py` 当前默认 `collision_radius=0.15 m`，不是 `0.02 m`。静态判碰撞使用已经按 occupancy-map `robot_size` 膨胀过的地图，而且射线每 `0.1 m` 才检查一次；直接设置 `0.02 m` 会使静态判碰撞基本失效。
- “trial 结束后停止并切换下一条轨迹”的完整组合是：evaluator 调用 `/rl_navigation/stop`，navigation 节点以零速度保持，然后下一 trial 用 `/gazebo/set_model_state` 复位同一架无人机并发布新目标。Gazebo 和 RViz 进程不会重启；画面中的无人机模型会复位。
- 用户遇到的 `self.target_dir.clone()` / `NoneType` 异常是停止服务与控制定时器的并发竞态。完整 `RLock + navigation_epoch` 修复目前只在时间戳快照的 `navigation.py`；仓库内 `ros1/navigation_runner/scripts/navigation.py` 仍可能复现该竞态。
- 障碍物生成器模板已经包含与 `<visual>` 相同的 `<collision>`，但时间戳快照中由 `start.launch` 加载的现有 `generated_env.world` 仍只有地面 collision。必须重新生成 world 并重启 Gazebo，障碍物物理碰撞才会实际进入运行场景。

### 文档导航

- 第 1 节：仓库、论文和所有 ROS1 副本的路径/版本指纹。
- 第 2--5 节：论文定位、结构、师兄意见和指定基准 commit 的方法定义。
- 第 6 节：当前训练端起终点、课程和教师策略。
- 第 7 节：传感器链、ROS1 已改内容、精确碰撞规则、Gazebo 尺寸和 trial-stop 并发修复。
- 第 8--9 节：指标分母、实验设置和当前论文表格数据。
- 第 10 节：训练/评估命令、Ubuntu 同步、六终端流程、world 重生成和故障判断。
- 第 11--13 节：sim-to-sim 差异、已完成工作和按优先级排列的后续事项。
- 第 14--18 节：验证边界、新会话提示、推荐文件、对话时间线和最短执行清单。

## 1. 最重要的接手信息

### 1.1 当前仓库与版本

- **[当前文件]** 当前分支：`2-14-7-1`
- **[当前文件]** 当前 HEAD：`5e6540bbd822f73b2ff5ae076c9f561d9fef1086`
- **[当前文件]** 当前 upstream：`navrl2/2-14-7-1`；remote URL：`https://github.com/05380/navrl2.git`。
- **[已确认口径]** 论文的方法描述必须以提交
  `6e137e952fc075b3804c9addadc074691482df02`
  为主要代码依据。该提交的信息为：
  `2-14-7-1-14 修改论文 部署端新增大障碍物。`
- **[已确认口径]** 当前分支在 `6e137e9` 之后又加入教师策略、起终点重构、ROS1 评估修订和实机部署文件。因此，不能直接用当前 HEAD 的全部实现反向解释论文中的既有方法。
- **[已确认口径]** 提交
  `37d1092f8057a45baac49cd78488d03837af2c92`
  引入了大障碍物教师引导。用户用该版本训练后观察到：直接碰撞死亡比例下降，但因死锁和超时死亡的比例显著上升。
- **[当前文件]** 提交 `26f3a6a` 重构了训练起终点采样规则，详见第 6 节。

关键版本时间线：

| Commit | 日期 | 交接意义 |
|---|---|---|
| `6e137e9` | 2026-07-31 | 当前论文方法说明的指定代码基准 |
| `84e7b72` / `f970735` | 2026-08-01 | 部署端轨迹选择和测试后轨迹显示调整 |
| `37d1092` | 2026-08-02 | 加入训练端教师引导；出现碰撞下降但死锁/超时上升的实测现象 |
| `5fe8480` | 2026-08-06 | 在教师版本之后继续修正训练端 |
| `26f3a6a` | 2026-08-22 | 默认改为四边起点到正对边目标，并增加 wall-crossing/越界规则 |
| `e045060` | 2026-08-27 | 论文调整并加入 `ros1 real/` 实机部署文档 |
| `e0cf610` | 2026-09-09 | 整理 real ROS 目录；上一版交接生成时的 HEAD |
| `ba0a7e8` | 2026-09-11 | 提交论文、ROS1 bug 修改、生成 world、两张训练图和第一版交接文档 |
| `5e6540b` | 2026-09-11 | 当前 HEAD/upstream；更新交接文档，位于 `ba0a7e8` 之后 |

接手后先执行：

```bash
cd /Users/yoloflps/Downloads/study/navrl2
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
```

### 1.2 当前提交与工作树状态

用户已在 2026-09-11 14:15:39 +0800 创建并推送提交：

```text
ba0a7e84249c3ccdd28b1b4e4169354ab90899e7
修改论文 、ros1bug修改 、交接文档
```

该提交一次性纳入了此前工作树中的 14 个文件，包括：

- 新增 `PROJECT_HANDOFF.md`；
- 提交 `ros1/navigation_runner` 的 evaluator、简单 stop 版 navigation 和 `std_srvs` 依赖；
- 提交 `world_generator.py` 与带 107 个 collision 的 Git 内 `generated_env.world`；
- 提交论文主稿、架构图、预览 PDF/辅助文件；
- 新增并跟踪 `fig_training_initial.png` 与 `fig_training_final.png`。

随后又创建并推送了提交：

```text
5e6540bbd822f73b2ff5ae076c9f561d9fef1086
更新交接文档
```

最后复核时，当前分支与 upstream `navrl2/2-14-7-1` 都指向 `5e6540b`，ahead/behind 为 `0/0`。两张训练图已经存在于当前 HEAD 中。

当前工作树并非 clean。尚未提交的有效论文改动包括 `thesis/icra.tex`、Fig. 2 的 TikZ/PDF 和最新的 `thesis/icra.pdf`；原 `thesis/icra_preview.pdf` 已按用户要求删除，只保留 `icra.pdf` 作为论文输出。两套 jobname 产生的辅助文件仍可能存在，提交前应区分正文、正式 PDF 和构建产物，不要把辅助文件误当成研究数据。

仍然不要使用 `git reset --hard`、`git checkout -- .` 或 `git clean` 来处理状态；先运行 `git status --short` 并确认文件归属。尤其不要为了让文档 clean 而覆盖本次交接更新。

### 1.3 论文主文件

- **[当前文件]** 主 LaTeX：`/Users/yoloflps/Downloads/study/navrl2/thesis/icra.tex`
- **[当前文件]** 当前论文 PDF：`/Users/yoloflps/Downloads/study/navrl2/thesis/icra.pdf`
- **[当前文件]** 参考文献数据库：`thesis/ref.bib`；图片目录：`thesis/figures/`。正文目前是单体 `icra.tex`，没有通过 `\input` / `\include` 拆分章节。
- **[当前文件]** IEEE 模板：
  - `thesis/support/IEEEtran.cls`：IEEE 论文版式类文件。
  - `thesis/support/IEEEtran.bst`：IEEE 参考文献排序和显示格式。
- **[重要]** 另有一份师兄批注过的外部副本：
  `/Users/yoloflps/Downloads/thesis/icra.tex`。
  它与仓库内主文件内容不同。除非用户明确要求，不要把外部副本当成当前投稿源文件直接覆盖仓库版本。
- **[历史参考]** NavRL 原始训练代码曾位于：
  `/Users/yoloflps/Downloads/NavRL-main 2/isaac-training`。
- **[仿真部署快照]** 当前最新可见副本：
  `/Users/yoloflps/Downloads/ros1 2 18.27.45`。
  它包含本轮对话最后版本的 opposite-crossing evaluator 和并发安全 navigation 修复，但不受本仓库 Git 管理。
- **[历史副本]** `/Users/yoloflps/Downloads/ros1` 仍存在，但没有正对边目标和 trial-stop 完整逻辑，不应误同步到 Ubuntu。
- **[已消失路径]** 对话前段直接修改过 `/Users/yoloflps/Downloads/ros1 2`；最后复核时该目录已不存在。尚不能证明它一定只是被重命名成时间戳目录，因此只把它作为历史路径记录。
- **[Ubuntu 运行路径]** `/home/wzf/navrl1_ws/src/ros1` 才是用户命令实际加载的工作空间源文件。当前无法从 macOS 工作区直接确认 Ubuntu 文件内容。

论文图源补充：`thesis/fig_3d_vo_geometry.tex` 是独立 TikZ 源，可在 thesis 目录单独运行 `pdflatex fig_3d_vo_geometry.tex` 生成一页 PDF；`draw_algorithm_framework.py`、`create_algorithm_framework_pptx.py` 和 `draw_multi_horizon_outcome.py` 保留了部分可编辑图源，但当前正文架构图实际引用的是 `figures/fig_foresightnav_architecture.png`，不要误以为那些输出都已被论文使用。

论文编译命令：

```bash
cd /Users/yoloflps/Downloads/study/navrl2/thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  icra.tex
```

最近一次编译状态：成功，8 页；未发现未解析引用、未解析文献或 overfull box。仍有非致命问题：作者信息尚未填写，并存在少量 underfull box。补入正式作者与单位后可能改变分页，必须重新检查是否仍满足 8 页限制。

### 1.4 代码副本、职责与版本指纹

本项目当前最大的操作风险是把不同目录误认为同一版本。最后复核时的关系如下：

| 角色 | 路径 | 状态与用途 | 关键指纹/风险 |
|---|---|---|---|
| Git 管理的研究主仓库 | `/Users/yoloflps/Downloads/study/navrl2` | 训练、论文、`ros1/` 参考实现、`ros1 real/` | branch `2-14-7-1`；HEAD/upstream 均为 `5e6540bbd822f73b2ff5ae076c9f561d9fef1086`；当前论文、预览、辅助文件和交接文档有未提交修改 |
| Git 内 ROS1 参考实现 | `.../navrl2/ros1` | opposite-crossing、stop service 和 world-generator 修改已提交于 `ba0a7e8` | evaluator `ad09030f...`; navigation `6f5f96f3...`；navigation 仍有 stop/rollout 竞态 |
| 旧 ROS1 下载副本 | `/Users/yoloflps/Downloads/ros1` | 仅保留作历史对照 | evaluator `fcbf5ffd...`；目标可落相邻边；无完整 stop 机制 |
| 对话前段修改路径 | `/Users/yoloflps/Downloads/ros1 2` | 最后复核时不存在 | 不可作为交付源；需人工确认是否被重命名/移动 |
| 最新仿真部署传输快照 | `/Users/yoloflps/Downloads/ros1 2 18.27.45` | 下一次同步 Ubuntu 时应优先核对的版本 | evaluator `70c284aa...`；navigation `6ff3e0de...`；后者含完整竞态修复 |
| Ubuntu 实际运行树 | `/home/wzf/navrl1_ws/src/ros1` | ROS Noetic/Gazebo 真正执行的源代码 | 内容未知，必须在 Ubuntu 用 `rospack find`、`sha256sum` 和 `stat` 确认 |
| 实机部署路径 | `.../navrl2/ros1 real` | SU17 + MID-360 Phase 1/shadow-mode 相关文件 | 与本轮 Gazebo evaluator 不同，不能混用 |

功能矩阵：

| 功能 | Git 内 `navrl2/ros1` | 旧 `/Downloads/ros1` | 时间戳快照 | Ubuntu 运行树 |
|---|---|---|---|---|
| 四边严格正对边 | 有，默认边界 `±11` | 无；目标只排除同边，仍可到相邻边 | 有，默认边界 `±12.5` | 未知，待核验 |
| trial 结束调用 stop | 有 | 无 | 有 | 未知，待核验 |
| navigation 零速保持 | 有，简单版本 | 无 | 有 | 未知，待核验 |
| `RLock + epoch` 竞态修复 | 无 | 无 | 有 | 未知，必须核对 hash |
| evaluator 默认碰撞半径 | `0.15` | `0.15` | `0.15` | 可能被源码或 ROS 参数覆盖 |
| 生成器障碍物 visual/collision 对应 | 模板有 | 旧模板不完整 | 模板有 | 未知 |
| 目录内 `generated_env.world` 的 visual/collision 计数 | `107/107` | 未作为本轮目标审计 | `71/1`，未重生成 | 未知；以 `start.launch` 实际路径为准 |

完整 SHA-256（2026-09-11 最后复核）：

```text
训练 train.py:       8297d543c598d62dde58f40219c0ad0109468e8948a7d0252be902eb31dbd1da
训练 env.py:         bac5937b4d3ec04edad80a116c19eff82a8636c64b54e15574af5ea9c294d201
训练 train.yaml:     5f6314c2dd964f7cb90219f0fdfc7851b1e780e90e0c9e6050eb1c403d446754
Git ROS evaluator:   ad09030fd680149cad4b0f6067d7e514788605c640d48d59acba4038e1ef69d6
Git ROS navigation:  6f5f96f38eff7b9c0ab7fdbafc72bb3f51f862be33c14a262966a9df73e3226e
Git ROS package.xml: d621cd868fec4275303cf0470aa4add0c4356c90c52ffbc21549e2ffe9706ea6
Git ROS 生成 world:  2ea25da8f5884bcd8794466a4e1b606b66cba57b4ae7c45d590f3e37c41dbea4
时间戳 evaluator:    70c284aa7fa8c0f838501d003510296734d4e5014f38becf318da6d1bf61070c
时间戳 navigation:   6ff3e0de6a0f068873a97b616a64547ebd9244765b6c6d194efc8ff2f01f628d
时间戳 package.xml:  c36a91429a5146bc217c63ac0f3b45e4d4b952b99d59dd75d6c0db966082b9df
时间戳 world 生成器: 0b58f2654a84ce50097ec1bd08810d1aaf16933ce07d563a4cba13df265d3db4
时间戳旧 world:      b5827e180bc749c2b59334d21e8ad400902b65beaec8e9224e12031926dfdfaf
论文 icra.tex:       478d4c0712dce77a076810e39df801ed41bcef803c34001ff98e6b4dc5fe6203
论文 PDF:             51275da743aa4561dbe2bf2edee700c000b32993a095a27e7035b595e00f3ddc
论文架构图:          e7ee61f8b54ee9a8171772f90d62713df9a202d0c4e2436d3bdeb26819d80c17
训练初始场景图:      131c25540662d5ba93f8cd1dd34ba0e185fa764319547b13fdbec6a1868dca6c
训练最终场景图:      ac096b70707e73aa0bad0575c9bd1f71fad54dc027becf91be779d76d2270178
```

哈希只描述最后审计时的文件。任何后续有意修改都会改变哈希；此时应记录新哈希、日期和改动原因，而不是强行恢复旧哈希。

## 2. 论文定位与写作边界

### 2.1 当前题目

> ForesightNav: Anticipatory Reinforcement Learning for UAV Navigation in Environments with Dynamic Obstacles and Large Non-Convex Structures

标题中的 `with` 曾被讨论是否改成 `under`。当前使用 `with`，含义是环境中包含这些障碍物；`under` 更适合 conditions/constraints，不适合直接修饰 obstacles。

### 2.2 方法定位

- **[已确认口径]** ForesightNav 基于 NavRL 的学习导航框架继续解决其未充分处理的问题，但论文不要反复或过于明显地写成“we extend NavRL”。
- **[已确认口径]** 更自然的逻辑是：先说明 NavRL 体现了端到端、并行仿真训练和反应式控制的潜力；随后指出它没有显式建模有限时域运动冲突，也缺少针对未来导航结果的分时域监督，对大型非凸结构附近的持续绕行和局部停滞处理不足；最后引出 ForesightNav。
- **[已确认口径]** 不要详细列举“保留了 NavRL 的哪些模块”，只需在必要位置以引用说明继承关系。
- **[当前论文]** 核心新增内容：
  1. 有限时域、TTC-aware 的 3D velocity-obstacle 风险奖励；
  2. action-conditioned multi-horizon outcome learning，在训练阶段预测碰撞、停滞、净空和进度；
  3. 面向动态障碍物和大型非凸结构的课程训练与评估。
- **[当前论文]** Outcome predictor 只用于训练，部署时移除，不增加在线预测或规划计算。
- **[当前论文]** 部署端可选 safety shield 与训练期 3D-VO 奖励是互补模块，不应混写成同一个创新。

### 2.3 当前摘要口径

- 摘要已经从“现有强化学习方法”扩大为“现有方法”，避免问题定义过窄。
- 实验结论使用定性写法，不在摘要中逐项列出 83.3%、69.9% 或提升 13.4 个百分点。
- 当前结尾强调：仿真中相较基线具有更好的整体导航表现；零样本实飞验证可部署性；不增加在线预测或规划。
- **不要重新加入 Index Terms。** 用户曾要求加入，随后明确要求撤销；当前论文没有 `IEEEkeywords`。

## 3. 论文当前结构、图、表和算法

### 3.1 章节结构

1. Introduction
2. Related Work
   - Classical, Learning-Based, and Hybrid Navigation
   - Velocity-Obstacle-Guided Learning
   - Predictive Models and Auxiliary Outcome Learning
3. Methodology
   - Finite-Horizon TTC-Aware 3D Velocity-Obstacle Modeling
   - RL Formulation
   - Reward Function
   - Network Design and Policy Training
   - Deployment-Time Policy Action Safety Shield
4. Experiment Results
   - Training Details and Setup
   - Simulation Results
   - Physical Flight Tests
5. Conclusion

### 3.2 当前图编号

1. **Fig. 1：总体架构图**，标签 `fig_zong`
   - 文件：`thesis/figures/fig_foresightnav_architecture.png`
   - 最近一次由 `/Users/yoloflps/Desktop/图片11.png` 替换。
   - 内容：系统输入、感知系统、动态障碍物 MLP、LiDAR-like CNN、共享嵌入、Predictive Actor-Critic、TTC-aware 3D-VO、multi-horizon predictor、训练/部署分支和可选 shield。
   - 图注已按 NavRL 论文的叙述风格重写，正文已引用。

2. **Fig. 2：3D-VO 几何图**，标签 `fig_3d_vo_geometry`
   - 文件：`thesis/fig_3d_vo_geometry.pdf`
   - 源文件通常为 `thesis/fig_3d_vo_geometry.tex`。
   - 子图 (a) 表示相对运动首次进入膨胀区域和 TTC；子图 (b) 表示有限时域 3D-VO 速度集合。
   - 已按师兄建议在正文中引用，并去掉两个子图之间的竖线。
   - 无人机图标已调整为灰色、细机臂和较大旋翼圆。

3. **Fig. 3：课程训练环境**，标签 `fig:training_environments`
   - 文件：`thesis/figures/fig_training_initial.png`、`fig_training_final.png`
   - 半栏图，(a) 初始环境包含 60 个动态障碍物；(b) 最终阶段包含多个大型非凸结构。
   - 图注将 `map_range=[20,20,4.5]` 对应的障碍核心区表述为 40 m x 40 m，并概括课程中逐步提升动态密度和非凸结构复杂度；含每侧 5 m border 的总地形仍约为 50 m x 50 m。

4. **Fig. 4：四类仿真轨迹**，标签 `fig_sim_trajectories`
   - (a) 静态障碍物；(b) 动态障碍物；(c) 混合障碍物；(d) 大型非凸结构。
   - 当前图只展示 ForesightNav 轨迹。
   - 当前图例顺序为 Start、Goal、Trajectory、Static obstacle、Dynamic obstacle；四幅子图已统一尺寸并对齐。
   - 师兄建议在同一图中加入多个方法轨迹作直观比较，用户当时明确暂缓这项修改，因此仍是待办。

5. **Fig. 5：Gazebo 走廊导航可视化**，标签 `fig:corridor_navigation`
   - 上图为含静态障碍物和动态行人的 Gazebo 走廊。
   - 下图为环境地图与 ForesightNav 导航轨迹。

6. **Fig. 6：物理飞行实验**，标签 `fig_real_world`
   - (a)--(d) 为静态、动态、混合和大型非凸结构四类环境。
   - (e) 为四个环境中的三维线速度变化曲线。
   - 论文实际引用：`thesis/figures/fig_real_velocity.pdf`；它与分析输出
     `analysis_outputs/env_speed_comparison/env1_env4_3d_speed_comparison.pdf`
     内容一致，使用 PDF 版本以保持清晰度。
   - 图已移动到 Physical Flight Tests 中 “We construct four physical environments ...” 之前。

历史上曾存在课程成功率占位图、额外 Fig. 4 和其他占位图，已经删除并完成后续编号修正。不要根据旧对话截图恢复旧编号。

### 3.3 当前算法框

- 当前正文中没有 Algorithm 环境、算法编号或算法引用。
- Multi-Horizon Outcome Learning 仍由预测输出、目标聚合和多任务损失公式完整描述，不再保留重复的伪代码图。
- `algorithm2e` 依赖以及相关辅助命令已经从 `icra.tex` 删除。

### 3.4 当前表格

- **Table I**：训练超参数。
- **Table II**：四种环境中的仿真定量结果。
- **Table III**：混合环境中的 safety-shield 消融。
- Table II、III 的最优值已加粗。
- Table II 方法顺序固定为：PE-Planner、NavRL、ForesightNav w/o TTC-aware 3D-VO reward、ForesightNav。
- Table III 的 Success、Collision 和 Timeout 为互斥终止统计；当前四行 Timeout Rate 均为 `0`，不再写 `N/A`。
- Ego-Planner 已由 PE-Planner 替换。相关论文历史文件：
  `/Users/yoloflps/Desktop/2403.12865v1.pdf`。

## 4. 师兄意见的落实状态

### 4.1 已落实

- 标题和摘要范围不再局限于 RL 方法。
- Introduction 重新组织为问题、NavRL 潜力、未解决问题、本文方案。
- 解释了仅用距离风险无法区分 closing velocity、TTC 和三维重叠关系。
- 大型非凸结构中的失败现象与期望绕行行为拆开描述。
- 首次出现 VO 时定义并引用经典 VO；区分经典 VO、3D-VO 和 TTC-aware 3D-VO。
- 贡献列表删除直接与 NavRL 比较的措辞，改为陈述本文完成的工作。
- PPO 首次出现时同时引用 NavRL 和 PPO 原始文献。
- 将不准确的 “recurrent architectures” 改为 predictive/model-based architectures；按用户要求删除了
  `General latent models can introduce prediction error, while online rollout evaluation adds deployment computation.`
- Fiorini 文献只作为经典 VO 概念来源，不暗示其提出本文的有限时域各向异性 3D-VO。
- 在继承 NavRL 的观测/动作、编码器和 safety shield 处增加适当引用。
- Methodology 中的主要符号定义已补全。
- 3D-VO 图已在正文引用，子图中间竖线已删除。
- 实验章节删除可能暗示多机集群的 `swarm sizes`。
- 全文图引用格式统一为 `Fig.~\ref{...}`。
- 仿真轨迹文字明确为 ForesightNav 生成。
- 实验章节再次引用 NavRL 和 PE-Planner。
- Table II 和 Table III 最优值已经加粗。

### 4.2 暂未落实或仍需复核

- **[待完成]** Fig. 4 尚未加入多方法轨迹对比，这是师兄意见中用户明确暂缓的一项。
- **[待验证]** 投稿前需再次逐条对照师兄原始注释副本，确认没有在后续压缩篇幅和图编号调整中发生回退。
- **[当前完成]** 当前无作者信息的预览已经压缩到 8 页。
- **[待完成]** 作者信息和最终投稿元数据尚未补齐；加入后可能重新溢出到第 9 页，需要再次排版。

## 5. 以 `6e137e9` 为准的训练端方法定义

本节用于回答“论文中的模型实际吃什么输入、输出什么动作”。若当前 HEAD 与此不同，论文解释仍优先回到 `6e137e9` 核对。

### 5.1 策略观测

策略的直接输入不是原始 RGB 图像，而是经过感知或仿真环境生成的结构化观测：

1. **`state`，8 维导航状态**
   - 目标相对方向的三维单位向量：3 维；
   - 水平目标距离：1 维；
   - 垂直目标误差：1 维；
   - 无人机速度转换到目标/导航坐标系：3 维。

2. **`lidar`，静态几何邻近观测**
   - 形状为 `(1, 36, 4)`；
   - 水平 360 度、10 度分辨率，共 36 个方位；
   - 垂直 4 层，约覆盖 -10 度到 20 度；
   - 量程约 4 m；
   - 名称虽为 LiDAR，但它是“LiDAR-like/raycast proximity tensor”，可由深度相机构图后执行 ray casting 生成，并不等于实机必须安装激光雷达。

3. **`dynamic_obstacle`，动态障碍物描述符**
   - 形状为 `(1, 5, 10)`，保留最近的 5 个动态目标；
   - 每个目标包含：三维相对方向、水平距离、垂直距离、目标坐标系内速度以及尺寸/类别相关量。

4. **`direction`，3 维方向信息**
   - 主要用于坐标系转换。

因此，“state”狭义上指 8 维无人机与目标导航状态；完整策略观测还包括静态几何和动态障碍物特征。

### 5.2 动作

- Actor 使用 Beta 分布输出归一化动作。
- 动作从 `[0,1]` 映射为目标/局部坐标系内的三维速度命令，再旋转到世界坐标系。
- 训练配置中的最大速度通常为 `2.0 m/s`；论文实飞实验使用 `1.2 m/s` 上限，两者不能混写。

### 5.3 网络

- 静态几何分支：LiDAR CNN，卷积后形成约 128 维表示。
- 动态障碍物分支：展平后经 MLP，形成约 64 维表示。
- 两种感知特征与 8 维导航状态拼接后进入共享 MLP，主干隐藏层为 `[256, 256]`。
- Actor 输出 Beta 分布参数，Critic 输出状态价值。
- Outcome predictor 接收共享特征与采样动作，隐藏层约 128，分别预测 `h={1,3,5}` 的结果。

### 5.4 Multi-Horizon Outcome Learning

每个预测时域聚合四类训练目标：

- `collision`：窗口内是否发生碰撞，取最大值；
- `stuck`：窗口内是否停滞，取最大值；
- `clearance`：窗口内前方最小净空；
- `progress`：窗口内累计目标进度。

二元目标使用 BCE，连续目标经过适当变换后使用 Smooth-L1。采样动作和 rollout target 使用 stop-gradient；辅助损失更新 outcome decoder 和共享编码器，不直接通过 actor/critic 输出头反向传播。预测器只参与训练。

### 5.5 TTC-aware 3D-VO 奖励

论文当前必须保留三部分核心定义：

- 有限时域 3D velocity-obstacle 集合；
- 相对运动二次方程与首次碰撞时间 TTC；
- 将 TTC 和三维重叠风险映射为连续奖励/惩罚。

`6e137e9` 附近的代表性配置口径：

- reward weight：`1.3`
- TTC decay/temperature：`0.75 s`
- horizon：`2.1 s`
- XY/Z safety margin：约 `0.3 m`
- top-k obstacles：`10`
- warm-up：`20,000` steps

正式写作或复现实验前仍应以对应 checkpoint 使用的 YAML 为准。

### 5.6 动态障碍物是否具有 XYZ 速度

- 一部分动态障碍物会在三维空间随机采样局部目标，因此同时具有 x、y、z 速度分量。
- 另一部分障碍物的高度固定，主要在水平面运动，因此 z 速度为 0。
- 不能笼统写成“所有动态障碍物都做三维运动”。更准确的说法是训练环境包含三维运动目标和受限于水平运动的目标。

## 6. 当前分支新增的起终点和教师策略

这些内容主要来自 `6e137e9` 之后的提交，必须与论文基准版本区分。

### 6.1 起终点采样

**[当前文件，提交 `26f3a6a` 后]** 默认模式为 opposite crossing：

- 起点从四条边随机选择；
- 目标位于起点正对边，不保留相邻边任务；
- 训练大地图中边界为 `x=±21 m` 或 `y=±21 m`；
- 目标以起点关于中心的对称位置为中心，并可沿边切向加入有限抖动；
- `target_jitter=0` 时为精确对称；
- 起点和目标都会进入静态/动态障碍物净空筛选流程，但当前静态 raster 边界处理和最终 fallback 会削弱这一保证，详见下文；
- `wall_crossing` 模式还要求起终点连线与大墙体扩展影响区域相交；
- 超出 `|x|` 或 `|y| > 22 m` 会按越界统计。

当前 `isaac-training/training/cfg/train.yaml` 的代表值：

```yaml
eval_style: opposite_crossing_eval
train_style: opposite_crossing
env:
  task_sampling:
    boundary_coordinate: 21.0
    tangent_limit: 18.0
    target_jitter: 4.0
    horizontal_limit: 22.0
    dynamic_clearance_radius: 1.5
    wall_influence_radius: 2.0
    wall_line_sample_spacing: 0.25
```

旧名称 `random`、`random_crossing` 等在当前代码中会映射到 opposite crossing。评估模型时，实际起终点规则由所运行 checkout 的 `eval_style` 和配置决定，不能仅凭 `eval.py` 命令判断。

训练端的精确采样公式为：

```text
side ~ Uniform{0,1,2,3}
t     ~ U[-18,18]
eps   ~ U[-4,4]
t_goal = clamp(-t + eps, -18,18)
z_start, z_goal 分别独立采样自 U[0.5,2.5]
```

| `side` | 起点 | 正对边目标 |
|---:|---|---|
| 0，上边 | `(t, +21, z_start)` | `(t_goal, -21, z_goal)` |
| 1，下边 | `(t, -21, z_start)` | `(t_goal, +21, z_goal)` |
| 2，右边 | `(+21, t, z_start)` | `(-21, t_goal, z_goal)` |
| 3，左边 | `(-21, t, z_start)` | `(+21, t_goal, z_goal)` |

因此，“正对边”不等于“目标在对边再次独立均匀随机”：目标应靠近起点关于地图中心的中心对称点，只在切向叠加有限抖动。

候选任务会调用以下合法性筛选：起点和终点的静态净空配置默认 `1.5 m`，动态净空也默认 `1.5 m`，最多尝试 128 轮。静态净空检查使用 `0.1 m` 栅格上的二维方形膨胀，`1.5 m` 对应 15 个栅格。

但当前实现有两个必须记录的例外：

1. 静态 occupancy raster 保存的是 `40 m x 40 m` 核心区，即约 `[-20,20]^2`；任务边界却是 `±21 m`。`_points_have_static_clearance()` 将 raster 外的点直接视为 clear，因此当前边界起终点的 `1.5 m` 静态净空实际上通常被旁路。外层 `5 m` border 按设计应为空，但这仍不等价于严格检查到核心边缘障碍物的距离。
2. 普通 opposite-crossing 在 128 次都失败时会采用最后一批 fallback 候选，动态净空等约束也可能被放弃。

所以更准确的说法是“训练代码存在 `1.5 m` 静态/动态净空筛选流程”，而不是“每个最终起终点都被数学保证有 `1.5 m` 净空”。这应作为后续采样器修复与单元测试事项。

### 6.2 课程训练

- 课程训练不仅增加动态障碍物密度，也在后期逐渐引入或加强大型非凸结构。
- 论文只做概括说明，不需要逐项写难度调度公式。
- 当前论文 Table I 采用的适应阶段学习率为：
  - feature extractor：`1e-5`
  - actor：`1e-5`
  - critic：`1e-4`
- 基础阶段的 encoder/actor/critic 初始学习率均为 `5e-4`。

用户曾比较两种课程二继续训练方案：

1. 三者都保持 `5e-4`；
2. encoder/actor=`1e-5`，critic=`1e-4`。

第二组被用于当前论文超参数表，但最终选择仍应结合原始训练日志验证。历史日志附件：
`/Users/yoloflps/.codex/attachments/e54dba4f-8345-44aa-9684-a1af7cc36681/pasted-text.txt`。

### 6.3 大障碍物教师策略

- `37d1092` 首次加入教师引导与行为克隆。
- 用户的实测现象是碰撞减少、死锁和超时增加。这说明教师可能过度偏向即时避碰或局部墙体跟随，未保证持续进度。
- 后续代码已经优先加入两项修复：
  1. **连续点段分割**：根据量程跳变和相邻点距离分割扫描点，并处理 360 度首尾段合并，减少相邻静态物体对大墙法向估计的污染；
  2. **教师动作静态扫掠检查**：沿候选教师动作的短时轨迹检查三维点云净空，拒绝会扫入静态障碍物的动作。
- 代表性配置/实现参数包括：range jump `0.65`、point gap `0.85`、最小段长度 `5`、扫掠时域 `1.0 s`、10 个检查步、safety radius `0.40 m`。
- 相关代码：
  - `isaac-training/training/scripts/wall_follow_teacher.py`
  - `isaac-training/training/scripts/demonstration_buffer.py`
  - `isaac-training/training/scripts/ppo.py`
  - `isaac-training/training/scripts/train.py`
  - `isaac-training/training/tests/test_wall_teacher.py`

当前配置中通常已有以下键：

```yaml
wall_teacher:
  enabled: true

algo:
  feature_extractor:
    behavior_cloning:
      enabled: true
```

用户曾在旧 checkout 中遇到 Hydra 错误：`Key 'wall_teacher' is not in struct`。旧配置临时追加键可写 `+wall_teacher.enabled=false`，但若代码本身没有读取该配置，仅加 `+` 不会实现教师开关。当前分支已有正式配置键，应使用 `wall_teacher.enabled=false`。

## 7. 训练、仿真部署与实机部署的传感器关系

### 7.1 是否必须使用雷达

- **结论：算法不要求实机必须安装物理 LiDAR。**
- 训练端静态分支需要的是 36x4 的局部几何射线距离表示，而不是特定传感器品牌。
- 标准 ROS1 部署可以用 RGB-D 深度图/点云构建占据地图，再对地图 ray casting，生成与训练端近似的 LiDAR-like tensor。
- Ray casting 的作用是把稠密地图转换成策略训练时使用的固定角度、固定维度局部距离观测，不代表原算法必须使用激光雷达。

### 7.2 标准 ROS1 感知链

- RGB 与对齐深度可用于动态目标检测、三维定位、尺寸估计和速度跟踪。
- 深度图或深度点云也可用于静态占据地图。
- Intel RealSense D435i 与该接口兼容，但代码依赖的是 ROS topic 和数据格式，不是硬编码相机型号。
- `use_real_detector=false` 时，Gazebo 评估一般使用仿真真值/假检测器，而非真实视觉检测结果。

因此，架构图中可以展示 RGB-D 作为感知源，但不应画成“动态障碍物只靠 RGB、静态障碍物必须靠物理雷达”的硬性分工。

### 7.3 `ros1 real/` 的 SU17 + MID-360 路径

当前实机部署目录：

`/Users/yoloflps/Downloads/study/navrl2/ros1 real`

详细操作说明：

`ros1 real/navigation_runner/SU17_PHASE1_DEPLOYMENT.md`

该 Phase 1 路径与论文的完整动态感知配置不同：

- 使用 MID-360 点云构建静态几何观测；
- ray casting 生成 144 个虚拟射线（36x4）；
- 5x10 动态障碍物描述符被置零；
- 移动物体只能通过占据变化被间接感知，不具备论文训练端那样的显式速度预测；
- 文档当前强调 shadow mode；不会自动解锁、起飞，也不直接向 MAVROS 写 setpoint；
- 控制接口采用 Prometheus `XY_VEL_Z_POS`。

因此，使用该 Phase 1 实机结果支撑“显式动态障碍速度感知”前，必须完成相应检测器/跟踪器接入和验证。

### 7.4 部署端 safety shield 的维度

- ROS1 shield 虽然部分数据结构包含 z，但当前基于 VO 的动作修正主要是水平面/二维避障。
- 论文的训练期 TTC-aware 3D-VO 奖励不能等同于部署 shield。
- 写作时宜称：训练期 3D-VO 提供预测性风险塑形；可选部署 shield 对策略动作做最低限度的安全修正。

### 7.5 本轮 ROS1 修改的落点与完成状态

本轮对话围绕四个问题进行了修改和核对：部署碰撞口径、opposite-crossing 起终点、trial 结束后的停止/复位，以及 Gazebo 障碍物物理 collision。它们落在不同副本，不能笼统写成“ROS1 已全部改好”。

#### Git 主仓库内 `ros1/`（已提交于 `ba0a7e8`）

- `deployment_eval.py` 已改成四边随机起点、正对边目标；默认边界 `±11 m`、切向范围 `±9 m`、目标抖动 `±2 m`。
- evaluator 已接入 `/rl_navigation/stop`，每个 trial 的碰撞、超时或成功终止后都尝试调用。
- `navigation.py` 已提供 stop 服务和零速度保持。
- `package.xml` 增加了 `std_srvs` 运行依赖。
- `world_generator.py` 已为随机生成的静态/动态 box 和 cylinder 增加与 visual 相同的 collision 几何。
- 仓库内当前 `generated_env.world` 已有 107 个 `<visual>` 和 107 个 `<collision>`，说明这份生成产物已更新。

但是，仓库内 `navigation.py` 只是早期简单修复：stop callback 仍会把 `goal` 和 `target_dir` 清成 `None`。若控制 timer 已进入 rollout，它仍可能在 `self.target_dir.clone()` 处触发相同竞态。该文件不能标为“最终修复版”。

#### 最新时间戳部署快照的修改

`/Users/yoloflps/Downloads/ros1 2 18.27.45` 中：

- `deployment_eval.py` 已采用相同的四边/正对边结构，部署尺度默认是边界 `±12.5 m`、切向 `±9 m`、目标抖动 `±2 m`、高度 `0.5--2.5 m`。
- `deployment_eval.py` 已在每个 trial 终止后调用 `/rl_navigation/stop`。
- `navigation.py` 是本轮最后的并发安全版本，包含 `threading.RLock`、单调 `navigation_epoch`、目标状态快照、发布前 epoch 校验以及 rollout 失效检查。
- 该目录的 `package.xml` 已经有 `std_srvs`，无需为本次修复再次增加依赖。
- 该目录的 `world_generator.py` 模板也已包含静态/动态障碍物 collision。

尚未完成的关键点：

- 时间戳快照的 `deployment_eval.py` 权限是 `664`，没有执行位；同步到 Ubuntu 后必须使用 `chmod +x`。
- 时间戳快照当前的 `generated_env.world` 只有 71 个 `<visual>`、1 个 `<collision>`；唯一 collision 基本是地面。`start.launch` 恰好加载这份旧 world，因此障碍物模板的修改尚未进入实际场景。
- 本轮没有在 Ubuntu ROS/Gazebo 中跑完端到端 trial；本机完成的是源码审计、Python 语法检查和隔离的状态切换测试。

#### 关于“增加 collision”和“撤销尺寸统一”的边界

- 给障碍物添加与 visual 相同的 collision，作用是让 Gazebo 物理引擎知道障碍物实体边界，支持接触、阻挡和物理碰撞。它不会自动改变 occupancy-map evaluator 的 `success=false` 判定，因为 evaluator 没有订阅 Gazebo contact，而是查询深度构建的 occupancy map。
- 随后讨论过把 Gazebo 无人机物理尺寸与 occupancy-map `robot_size` 统一，用户又要求撤销该尺寸改动。最后审计时 `no_map.yaml` 仍是 `robot_size: [0.2,0.2,0.1]`，无人机 URDF 也仍使用原始 mesh；所以“尺寸已统一”不是当前事实。
- 障碍物 collision 模板本身仍然存在，并没有随着尺寸统一尝试一起消失；真正欠缺的是重新生成并部署 world。
- 对话中也讨论过把 evaluator 半径从 `0.15 m` 改成 `0.02 m`，但最后可核验的 Git 内和时间戳 evaluator 默认值都是 `0.15 m`。后续报告结果时不得写成默认 `0.02 m`。

### 7.6 ROS1 静态碰撞判定的精确规则

以下结论以时间戳部署快照和用户给出的默认启动命令为准。

1. `deployment_eval.py` 默认读取：

   ```python
   collision_radius = rospy.get_param("~collision_radius", 0.15)
   ```

2. `get_static_collision()` 从无人机中心调用 `occupancy_map/raycast`，对返回命中点计算三维欧氏距离；任意点满足：

   ```text
   ||hit_point - drone_center||_2 <= 0.15 m
   ```

   就把 trial 判为静态碰撞。

3. 默认 `safety_and_perception_sim.launch` 使用 `use_px4=false`、`use_prebuilt_map=false`，因此加载 `navigation_runner/cfg/mapping/sim/no_map.yaml`。关键参数是：

   ```yaml
   robot_size: [0.2, 0.2, 0.1]
   map_resolution: 0.1
   ```

4. map manager 的膨胀格数为：

   ```text
   Nx = ceil(0.2 / (2*0.1)) = 1
   Ny = ceil(0.2 / (2*0.1)) = 1
   Nz = ceil(0.1 / (2*0.1)) = 1
   ```

   每个原始占据体素会扩成 `3 x 3 x 3` 的轴对齐体素块。这里 `robot_size` 是完整尺寸，源码内部除以 2；这种膨胀是方盒/Chebyshev 近似，不是欧氏球。

5. `RayCast` 明确调用 `castRay(..., true)` 并查询 `isInflatedOccupied()`，所以 evaluator 检查的是**已经膨胀的 occupancy map**，不是原始障碍表面。

6. raycast 从 `i=1` 开始，每隔 `map_resolution=0.1 m` 检查一次。因此返回的命中距离按理想情况量化为 `0.1, 0.2, 0.3, ... m`。默认半径 `0.15 m` 时，只有第一步 `0.1 m` 命中会触发，`0.2 m` 已不会触发。

因此不能把当前规则简单说成“无人机中心距离原始静态障碍物 `0.15 m` 就碰撞”。更准确的三层口径是：

- **代码硬判据**：144 条离散射线中至少一条的 `0.1 m` 首采样点落入膨胀占据体素；
- **相对未膨胀体素的轴向近似**：地图膨胀约 `0.1 m`，再加有效的一步射线距离约 `0.1 m`，通常理解为约 `0.2 m`；
- **相对 Gazebo 原始表面**：没有唯一连续数值，还会受到体素归属的约半格误差、深度噪声、遮挡、旧地图和 `10° x 4 beams` 射线角分辨率影响；理想情况下常落在约 `0.2--0.3 m`，将半格误差计入时可粗略写约 `0.25 m`，但这不是代码中的硬阈值。

`local_bound_inflation: 3.0` 只是扩大地图更新/重新膨胀的局部索引范围，不是额外增加 `3 m` 碰撞安全距离。

极其重要：如果只运行 `_collision_radius:=0.02`，raycast 的最小检查距离仍是 `0.1 m`，所以静态碰撞条件几乎不可能成立。这不是“2 cm 精确碰撞”，而是近似关闭静态碰撞检测。要得到真正的 2 cm 判据，需要提高地图与 raycast 分辨率，改用中心体素/邻域查询、ESDF 连续距离，或直接使用 Gazebo contact。

### 7.7 Gazebo 无人机物理尺寸与地图尺寸

非 PX4 的 `start.launch` 加载 `uav_simulator/urdf/quadcopter.urdf`。其 collision 是 `CERLAB_quadcopter.stl` mesh，缩放为 `1.5`，而不是一个球。对二进制 STL 顶点做只读边界审计得到：

```text
缩放后轴对齐包围盒约：0.509 x 0.490 x 0.150 m
相对模型原点的最大实际水平顶点半径约：0.315 m
```

这与 occupancy map 的 `robot_size=[0.2,0.2,0.1]` 并不相等。navigation safety shield 里另有 `self.robot_size=0.3`（半径）这一第三个尺寸。当前至少有四个不同概念：

1. Gazebo mesh 的真实物理 collision；
2. occupancy map 的 `[0.2,0.2,0.1]` 方盒膨胀；
3. evaluator 在膨胀地图之外再用的 `collision_radius=0.15`；
4. safety shield 的 `robot_size=0.3`。

论文、日志和代码注释里不得把它们统称为同一个“无人机碰撞半径”。若未来要统一，应先决定基准是物理最大外廓、等效圆半径，还是希望的安全包络，并用同一组场景做接触和 evaluator 的标定，而不是只改一个 YAML 数字。

### 7.8 训练端与 ROS1 碰撞/成功口径对照

训练端碰撞终止实际在 `isaac-training/training/scripts/env.py`，而不是由 `train.py` 自己计算；`train.py` 负责训练循环、调用评估和汇总指标。

| 项目 | 训练端 Isaac | 时间戳 ROS1 evaluator |
|---|---|---|
| 静态数据源 | 直接对静态三角网格的 LiDAR ray hits | 深度相机在线构建的膨胀 occupancy map |
| 静态判据 | 任意干净射线命中距离严格 `< 0.3 m` | 命中膨胀体素点的距离 `<= 0.15 m`，但距离以 `0.1 m` 步进 |
| occupancy `robot_size` | 不参与训练静态碰撞 | `[0.2,0.2,0.1]`，先做体素膨胀 |
| 动态水平判据 | `d_xy <= width/2 + 0.3` | `d_xy <= max(size.x,size.y)/2 + 0.15` |
| 动态垂直判据 | `|d_z| <= height/2 + 0.3` | `|d_z| <= height/2 + 0.15` |
| 成功 | 三维距离 `<=0.5 m` | 三维 `<=0.5 m`；或 XY `<=0.5 m` 且高度差 `<=0.2 m`；或越过目标平面并满足横向/高度容差 |
| 越界 | `z<0.2`, `z>4`, `|x|>22` 或 `|y|>22` 会终止 | evaluator 当前没有同类水平/高度越界终止 |
| 超时 | 2,200 episode steps | 180 s |
| 同步满足成功和碰撞 | 同一步理论上可同时进入两个统计标志 | evaluator 先查碰撞，碰撞优先，success 保持 false |
| 起终点净空 | 配置/流程中静态和动态默认各 `1.5 m`、最多 128 次；但 `±21 m` 位于静态 raster 外而被视为 clear，fallback 也可能放弃约束 | 当前只采几何位置，不做障碍净空重采样 |

训练端静态条件来自：

```python
lidar_scan_clean = lidar_range - hit_distance
max(lidar_scan_clean) > lidar_range - 0.3
```

代数上等价于 `min(hit_distance) < 0.3 m`。它使用无观测噪声的 `lidar_scan_clean`，但仍只有水平 `10°`、垂直 4 束、垂直视场 `[-10°,20°]`，所以也不是连续实体接触检测。

训练端动态碰撞只检查筛选出的最近 5 个障碍，超过 `4 m` 的障碍被屏蔽；部分高圆柱型二维动态障碍物的相对 z 被置零。ROS1 dynamic service 不可用或调用失败时，evaluator 当前直接按“没有动态碰撞”继续，这也是后续需要增加 fail-closed/fail-visible 处理的风险。

### 7.9 trial 停止、复位和 `NoneType.clone()` 竞态

#### 为什么原来显示碰撞后无人机仍飞向终点

旧 evaluator 在检测到碰撞时只做两件事：把当前 trial 标为 `success=false`，然后退出自己的监测循环。navigation 节点是另一个独立 ROS 进程，仍保存原目标并继续发速度命令；红色 RViz 轨迹只是 evaluator 的失败可视化，不是取消目标指令。因此“终端已经判碰撞”和“Gazebo 仍继续飞”可以同时发生。

#### 修改后的预期时序

```text
碰撞 / 超时 / 成功
  -> deployment_eval 调用 /rl_navigation/stop
  -> navigation 使当前 epoch 失效并发布零速度
  -> evaluator 返回本 trial、进入 for 循环下一项
  -> /gazebo/set_model_state 反复设置新起点、朝向和零速度直到稳定
  -> 发布新目标
  -> navigation 建立新 epoch 并恢复控制
```

Gazebo、occupancy-map 节点和 RViz 都不会被杀掉再启动。Gazebo 里的同一个 `quadcopter` 被瞬移到新起点；RViz 根据 odometry/marker 更新画面。若用户说“重新启动下一个轨迹”，在当前实现里应理解为“复位模型状态并开始新 trial”，不是重启第 1、2 个终端。

#### 已出现异常及最后修复

用户运行 stop 版本后遇到：

```text
AttributeError: 'NoneType' object has no attribute 'clone'
get_action -> self.target_dir.clone()
```

根因是 rospy Timer 控制回调与 stop 服务并行：timer 已通过开头检查，stop 随后把 `target_dir=None`，timer 又继续做 rollout。时间戳快照最终修复包括：

- 使用 `threading.RLock` 保护目标和发布状态；
- 每个真正的新目标增加 `navigation_epoch`；evaluator 短时间重复发布同一目标不会反复创建 epoch；
- stop 只使 epoch 失效，不再清空正在被旧 callback 引用的 `goal/target_dir`；
- 控制 callback 一次性快照 goal、target direction、yaw 和 epoch；
- 推理后的非零速度、定高命令、姿态命令和 rollout 发布前均重新验证 epoch；
- stop 的零速度发布与非零速度最终发布使用同一把锁，保证旧命令不可能在 stop 之后覆盖零速度；
- `get_action(target_dir=None)` 仍有防御性零动作返回。

针对这次特定 `NoneType.clone()` 报错，只替换最终版 `navigation.py` 并重启 terminal 3 即可，因为日志已经证明 evaluator 能调用 stop 服务。若要从旧 ROS1 副本获得完整的 opposite-crossing + 自动 stop 功能，则仍需同时使用对应的 `deployment_eval.py`。

## 8. 评估指标的统一定义

### 8.1 部署端终端输出

用户确认希望完整评估结束后输出：

```text
success_rate
failure_rate
collision_count / collision_rate
timeout_count / timeout_rate
deadlocked_trial_count / deadlocked_trial_rate
deadlock_event_count
deadlock_event_recovery_rate
post_deadlock_success_rate
```

当前 `ros1/navigation_runner/scripts/deployment_eval.py` 已计算并打印这些指标。

### 8.2 定义

- `success_rate = successful_trial_count / trial_count`
- `failure_rate = failed_trial_count / trial_count`
- 若所有试验只可能以成功、碰撞或超时结束，则：
  `failure_rate = collision_rate + timeout_rate`。
- `collision_count`：因碰撞结束的轨迹数。
- `timeout_count`：达到时间上限的轨迹数。
- `deadlocked_trial_count`：至少发生过一次死锁事件的轨迹数；一条轨迹最多计一次。
- `deadlock_event_count`：所有轨迹中的死锁事件总数；同一轨迹可以有多次，因此可大于 deadlocked trial 数，甚至可能超过试验总数。
- `recovered_deadlock_event_count`：被判定为恢复的死锁事件数。
- `deadlock_event_recovery_rate = recovered_deadlock_event_count / deadlock_event_count`。
- `post_deadlock_success_rate`：发生过死锁的轨迹中最终成功到达目标的比例。它是 trial-level 指标，不等于 event recovery rate。

### 8.3 论文当前定义

- 每种方法、每个环境执行 1,000 次独立试验。
- 成功、碰撞、180 s 超时为轨迹终止结果。
- Deadlock Count 是 1,000 次试验中的事件总数，不是比率。
- 死锁检测：前方存在障碍，同时单周期目标进度不超过 `0.005 m`，连续 40 个 20 Hz 评估周期，即约 2 s。
- 论文 Table II 当前最后一列名为 **Post-Deadlock Escape Rate**，定义为
  `recovered deadlock event count / deadlock event count`；无死锁事件时写 N/A。

**命名风险：** 用户曾要求把论文中的 `post_deadlock_success_rate` 改成 `recovered_deadlock_event_count`，但当前论文表格最后一列展示的是事件恢复率，而不是单独的恢复事件计数；代码又同时保留了 event-level recovery 和 trial-level post-deadlock success。投稿前必须最终统一术语，建议使用：

- 表内：`Deadlock Event Count` 和 `Deadlock Event Recovery Rate (%)`；
- 若要显示恢复数量：写成 `Recovered Events / Deadlock Events (Rate)`，例如 `24/53 (45.3%)`；
- 不要把它称为 `post_deadlock_success_rate`，除非分母确实是发生死锁的轨迹数。

## 9. 实验设置与当前表格数据

### 9.1 训练设置

- Isaac Sim + PyTorch。
- `env.py` 中障碍核心区由 `map_range=[20,20,4.5]` 生成，即 `40 m x 40 m`；TerrainGenerator 另设每侧 `5 m` border，因此包含边界的总水平地形约为 `50 m x 50 m`。当前论文和 Fig. 3 使用 40 m x 40 m，明确指 obstacle workspace；不要把它与含 border 的总地形混为一谈。
- 论文和正式训练命令使用并行 UAV 数 1,024；当前 `train.yaml` 文件里的开发默认值却是 `env.num_envs: 2`，必须通过命令行显式覆盖并保存 resolved config，不能只根据 YAML 默认声称运行了 1,024。
- 训练最大速度：2.0 m/s。
- 动态圆柱半径集合：`{0.125, 0.25, 0.375, 0.50} m`。
- 动态障碍物速度：约 `0.5--1.5 m/s`。
- 当前配置的代表性障碍数为静态 350、动态 80；课程各阶段和论文图片可使用不同数量，例如初始图说明为 60 个动态障碍，必须以每次运行的 resolved Hydra config 为准。
- 当前最大 episode 长度：2,200 steps；仿真时间上限由实际 control/simulation dt 决定，不能在不知道 dt 时直接等同于 180 s。
- 训练硬件：NVIDIA RTX 4090。
- 每个课程阶段约 12 小时。
- rollout length：32；PPO epochs：4；mini-batches：16。
- `gamma=0.99`，`GAE lambda=0.95`。
- multi-horizon：`{1,3,5}`，辅助损失权重 `0.1`。

### 9.2 仿真评估设置

- 论文当前写作口径：工作空间 20 m x 20 m。
- 时间戳 Gazebo YAML 的静态障碍物采样核心区是 `x,y in [-10,10]`，确实为 20 m x 20 m；但时间戳 evaluator 的默认起点/目标边界是 `±12.5 m`，任务端点横跨 25 m。Git 内 evaluator 默认又是 `±11 m`。因此“工作空间”可能分别指障碍核心区、可飞区或起终点边界；正式复现实验必须保存实际参数并明确论文采用哪个定义，不能只写 20 m x 20 m 后默认所有几何都一致。
- 每个方法、每个环境：1,000 次独立试验。
- 方法：PE-Planner、NavRL、ForesightNav w/o TTC-aware 3D-VO reward、ForesightNav。
- 环境：
  1. Static obstacles，`N_stat=70`；
  2. Dynamic obstacles，`N_dyn=60`；
  3. Mixed static and dynamic obstacles，`N_stat=20, N_dyn=40`；
  4. Static obstacles and large non-convex structures。

### 9.3 Table II 当前数据

| Environment | Method | Success % | Collision % | Timeout % | Deadlock events | Recovered / total |
|---|---|---:|---:|---:|---:|---:|
| Static | PE-Planner | 85.5 | 14.5 | 0.0 | 0 | N/A |
| Static | NavRL | 94.1 | 5.9 | 0.0 | 0 | N/A |
| Static | ForesightNav w/o TTC-aware 3D-VO reward | 92.7 | 7.3 | 0.0 | 0 | N/A |
| Static | ForesightNav | 94.4 | 5.6 | 0.0 | 0 | N/A |
| Dynamic | PE-Planner | 65.6 | 34.4 | 0.0 | 0 | N/A |
| Dynamic | NavRL | 69.9 | 30.1 | 0.0 | 0 | N/A |
| Dynamic | ForesightNav w/o TTC-aware 3D-VO reward | 71.2 | 28.8 | 0.0 | 0 | N/A |
| Dynamic | ForesightNav | 83.3 | 16.7 | 0.0 | 0 | N/A |
| Mixed | PE-Planner | 63.2 | 36.8 | 0.0 | 0 | N/A |
| Mixed | NavRL | 61.8 | 38.2 | 0.0 | 0 | N/A |
| Mixed | ForesightNav w/o TTC-aware 3D-VO reward | 60.3 | 39.7 | 0.0 | 0 | N/A |
| Mixed | ForesightNav | 76.5 | 23.5 | 0.0 | 0 | N/A |
| Large non-convex | PE-Planner | 58.9 | 15.9 | 25.2 | 127 | 21/127 (16.5%) |
| Large non-convex | NavRL | 66.7 | 13.6 | 19.7 | 93 | 23/93 (24.7%) |
| Large non-convex | ForesightNav w/o TTC-aware 3D-VO reward | 77.5 | 14.0 | 8.5 | 42 | 18/42 (42.9%) |
| Large non-convex | ForesightNav | 78.1 | 15.6 | 6.3 | 53 | 24/53 (45.3%) |

这些数字满足 `success + collision + timeout = 100%`。但表格合理性不等于数据真实性，投稿前必须保留每组 1,000 次试验的 CSV/日志和统计脚本以便复核。

**重要写作要求：** 不要强调完整 ForesightNav 在 Env. 4 优于 `w/o TTC-aware 3D-VO reward`。两者在成功率、碰撞率、死锁数等指标上呈现混合差异。当前正文应把结论限制为整体预测设计在大型结构环境中改善持续导航和恢复能力，不能将所有优势单独归因于 TTC-aware 3D-VO reward。

### 9.4 Table III 当前数据

Table III 位于混合静态/动态环境，1,000 次试验：

| Method | Shield | Success % | Collision % | Timeout |
|---|---|---:|---:|---:|
| NavRL | Off | 57.4 | 42.6 | 0 |
| NavRL | On | 62.7 | 37.3 | 0 |
| ForesightNav | Off | 64.2 | 35.8 | 0 |
| ForesightNav | On | 75.3 | 24.7 | 0 |

这组数据曾被误填成 NavRL 与 ForesightNav 对调，当前表格已按上述顺序纠正。由于每行 Success 与 Collision 之和为 100%，Timeout Rate 按互斥终止定义记为 `0`。

### 9.5 物理飞行实验

- 总实验场地：28 m x 15 m；Fig. 6(a)--(d) 中每个障碍课程布置在 5 m x 15 m 的测试区域内。
- 最大速度：1.2 m/s。
- 四类环境：静态、动态、混合、大型非凸结构。
- Fig. 6(e) 显示各环境三维线速度随时间变化。
- 当前正文把实飞结果表述为定性、零样本部署可行性证据，不把有限的代表性轨迹包装成统计显著性结论。
- **[待验证]** 投稿前核对 Fig. 6(a)--(d) 是否均为实际飞行截图、轨迹是否来自真实日志、速度曲线是否与对应四次飞行一一匹配。

## 10. 典型运行命令和已知问题

### 10.1 训练示例

```bash
cd /Users/yoloflps/Downloads/study/navrl2/isaac-training/training/scripts
python train.py \
  headless=false \
  env.num_envs=1024 \
  env.num_obstacles=350 \
  env.wall_style=0 \
  env_dyn.num_obstacles=60 \
  wall_teacher.enabled=false \
  algo.feature_extractor.behavior_cloning.enabled=false \
  wandb.mode=offline \
  wandb.name=ForesightNav_C1_latest
```

如在旧 checkout 中出现 Hydra struct 错误，先确认其 `train.yaml` 是否真的包含 `wall_teacher`，不要只机械添加 `+`。

### 10.2 Isaac 评估示例

```bash
cd /Users/yoloflps/Downloads/study/navrl2/isaac-training/training/scripts
python eval.py \
  headless=false \
  env.num_envs=1024 \
  env.num_obstacles=350 \
  env_dyn.num_obstacles=80 \
  wandb.mode=offline \
  checkpoint=/path/to/checkpoint_29000.pt \
  seed=1
```

注意：该命令没有显式覆盖起终点风格，所以规则取决于当前 checkout 的 `eval_style`。记录结果时必须同时保存 Git commit、完整 Hydra 配置和 checkpoint 来源。

### 10.3 Gazebo 走廊生成与运行

走廊生成配置：

`ros1/uav_simulator/scripts/corridor_world_generator.yaml`

最近一次记录的代表值：

- arena：50 m x 5 m；
- static obstacle count：8；
- pedestrians：4；
- pedestrian speed：0.55--1.10 m/s；
- left/right start-goal x 区间约 `[-23.5,-22.5]` 和 `[22.5,23.5]`；
- timeout：120 s；评估频率：20 Hz。

修改 YAML 后必须重新运行生成脚本，YAML 不会被 launch 自动转换为新 world：

```bash
cd ~/navrl1_ws/src/ros1/uav_simulator/scripts
python3 generate_corridor_world.py
```

然后重新 source 工作空间并启动生成后的场景。用户历史运行流程为：

```bash
# Terminal 1: Gazebo + perception/safety（单独使用 Terminal 4 RViz 时关闭内置 RViz）
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
roslaunch navigation_runner lpnav_corridor_sim.launch \
  scenario_name:=lpnav_corridor_seed_7 \
  rviz:=false

# Terminal 2: policy node
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
conda activate NavRL
rosrun navigation_runner navigation_node.py

# Terminal 3: evaluator
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
rosrun navigation_runner deployment_eval2.py \
  _scenario_file:=$(rospack find uav_simulator)/worlds/lpnav_corridor/lpnav_corridor_seed_7.yaml \
  _num_trials:=100 \
  _random_seed:=1007 \
  _csv_path:=/tmp/lpnav_corridor_lpnav.csv \
  _keep_trajectory_publisher_alive:=true

# Terminal 4: RViz
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
roslaunch navigation_runner lpnav_corridor_eval_rviz.launch
```

当前仓库的 `deployment_eval.py` 比历史 `deployment_eval2.py` 包含更完整的新指标和 stop-service 修复。迁移命令时先核对 launch 文件和节点参数，不能只替换脚本名。

### 10.4 将本轮最终 ROS1 修复同步到 Ubuntu

本机时间戳目录只是传输快照；ROS Noetic 实际运行树在 Ubuntu。同步后先做以下检查：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash

rospack find navigation_runner
sha256sum \
  ~/navrl1_ws/src/ros1/navigation_runner/scripts/deployment_eval.py \
  ~/navrl1_ws/src/ros1/navigation_runner/scripts/navigation.py
stat -c '%A %a %n' \
  ~/navrl1_ws/src/ros1/navigation_runner/scripts/deployment_eval.py \
  ~/navrl1_ws/src/ros1/navigation_runner/scripts/navigation_node.py
```

预期 `rospack find` 必须输出：

```text
/home/wzf/navrl1_ws/src/ros1/navigation_runner
```

若要确认 Ubuntu 已收到本轮最后核验的组合，预期哈希为：

```text
deployment_eval.py  70c284aa7fa8c0f838501d003510296734d4e5014f38becf318da6d1bf61070c
navigation.py       6ff3e0de6a0f068873a97b616a64547ebd9244765b6c6d194efc8ff2f01f628d
```

时间戳快照的 evaluator 当前没有执行位。Ubuntu 上应执行：

```bash
chmod +x ~/navrl1_ws/src/ros1/navigation_runner/scripts/deployment_eval.py
chmod +x ~/navrl1_ws/src/ros1/navigation_runner/scripts/navigation_node.py
```

不要再用 `chmod -x`；`-x` 是删除执行权限，正是 `rosrun` 报 “Found ... but ... not executable” 的原因。`navigation.py` 是由 `navigation_node.py` import 的模块，本身通常无需执行位。

这次是纯 Python 源码替换时通常不需要重新 `catkin_make`，但必须完整退出并重启 navigation 和 evaluator 进程。若同时改动 ROS 消息、服务定义、CMake 或工作空间依赖，再重新构建并 source `devel/setup.bash`。

### 10.5 本轮用户使用的标准六终端评估流程

以下命令对应 `start.launch + safety_and_perception_sim.launch + navigation_node.py + deployment_eval.py`。为使结果可复现，建议显式写出随机种子、collision radius 和 CSV 路径。

Terminal 1，Gazebo：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
roslaunch uav_simulator start.launch
```

Terminal 2，占据地图、动态检测、safety shield；若准备在 Terminal 6 单独开 RViz，建议在这里禁用内置 RViz：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
roslaunch navigation_runner safety_and_perception_sim.launch rviz:=false
```

若保留 launch 的默认 `rviz:=true`，则不需要 Terminal 6，否则会打开两个 RViz。

Terminal 3，策略/navigation 节点：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
conda activate NavRL
rosrun navigation_runner navigation_node.py
```

启动 evaluator 前可在任意已 source 的终端确认停止服务：

```bash
rosservice list | grep '^/rl_navigation/stop$'
rosservice type /rl_navigation/stop
```

Terminal 4，评估器：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
rosrun navigation_runner deployment_eval.py \
  _num_trials:=50 \
  _random_seed:=0 \
  _eval_style:=opposite_crossing_eval \
  _start_boundary_half_size:=12.5 \
  _goal_boundary_half_size:=12.5 \
  _tangent_limit:=9.0 \
  _target_jitter:=2.0 \
  _collision_radius:=0.15 \
  _csv_path:=/tmp/deployment_eval_seed0.csv \
  _keep_trajectory_publisher_alive:=true
```

不要在反斜杠后再留空格，也不要在反斜杠续行之间插入空行。若只想复用源码默认值，可省略多数参数，但正式实验最好显式保存。

Terminal 5，评估完成后选择显示轨迹 1 和 3：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
rostopic pub -1 /deployment_eval/trajectory_selection \
  std_msgs/Int32MultiArray "data: [1, 3]"
```

Terminal 6，单独 RViz：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
rosrun rviz rviz \
  -d "$(rospack find navigation_runner)/rviz/navigation_runner.rviz"
```

`_keep_trajectory_publisher_alive:=true` 只是在全部 trial 完成后让 evaluator 保持运行，以便发布/筛选历史轨迹；它不是让失败 trial 继续导航的开关。

### 10.6 重新生成带障碍物 collision 的当前 world

时间戳部署快照的生成器已修改，但实际 `generated_env.world` 尚未更新。应在 Ubuntu 同步正确生成器后执行：

```bash
source /opt/ros/noetic/setup.bash
source ~/navrl1_ws/devel/setup.bash
source ~/navrl1_ws/src/ros1/uav_simulator/gazeboSetup.bash
cd ~/navrl1_ws/src/ros1/uav_simulator/scripts
python3 generate_random_world.py
```

然后在启动 Gazebo 前核对：

```bash
grep -c '<visual' \
  ~/navrl1_ws/src/ros1/uav_simulator/worlds/generated_env/generated_env.world
grep -c '<collision' \
  ~/navrl1_ws/src/ros1/uav_simulator/worlds/generated_env/generated_env.world
```

当前 YAML 是 35 个静态 cylinder + 35 个静态 box、0 个动态物体，再加地面，因此理想情况下两项都应约为 71。只看到 `visual=71, collision=1` 表示仍是旧 world。重新生成会按 `random_seed` 重建 world/PCD；正式实验前应保存 YAML、world、PCD 和 hash，避免场景悄悄变化。生成后必须重启 Terminal 1，正在运行的 Gazebo 不会热加载磁盘上的新 world。

### 10.7 常见报错的直接判断

- `rosrun` 找到脚本但说 `not executable`：运行 `chmod +x deployment_eval.py`，不是 `chmod -x`。
- `rospack find navigation_runner` 指向别的工作空间：source 顺序或 overlay 错误；先解决路径，不要继续跑实验。
- evaluator 启动时提示 stop service unavailable：应先启动 Terminal 3。当前 evaluator 只在初始化时等待 30 s，一旦失败，之后不会自动重连。
- 碰撞后仍继续飞：检查 `/rl_navigation/stop` 是否存在、evaluator 是否打印 `navigation stopped`、Terminal 3 是否运行带 stop service 的 `navigation.py`。
- 再次出现 `target_dir.clone()` 的 `NoneType`：Ubuntu 仍是旧的简单 stop 版本；核对 `navigation.py` SHA-256。
- evaluator 已进入下一 trial，但 Gazebo 未复位：检查 `/gazebo/set_model_state`、`model_name=quadcopter`、reset-settle 警告和 odometry 更新。
- Gazebo 可视障碍物没有物理阻挡：检查实际加载 world 的 `<collision>` 计数；生成器源码已改不代表 world 产物已重建。
- evaluator 报静态碰撞但 Gazebo 看似没接触：这是膨胀 occupancy-map 判据与 Gazebo mesh contact 的定义差异，不应仅凭视觉断言 evaluator 错误。

## 11. Isaac 与 Gazebo 成功率差异的排查框架

用户观察到：相同模型、相近障碍密度和相同起终点规则下，Isaac `eval.py` 成功率明显高于 Gazebo，且 Gazebo 使用 `use_real_detector=false` 仍较差。

优先排查：

1. **观测几何不一致**：Isaac 直接 raycast 与 Gazebo 占据地图分辨率、膨胀、盲区、射线角度和量程不同。
2. **坐标系/航向不一致**：world/body/goal frame 的旋转方向、yaw 符号、ENU/NED 约定可能错位。
3. **动作缩放不一致**：Beta 动作映射、速度上限、z 控制方式、控制频率和加速度限制不同。
4. **时延和动态学差异**：ROS topic 延迟、地图更新、控制器响应、Gazebo 动力学和 Isaac 简化动力学不同。
5. **动态目标语义不一致**：真值检测器输出的尺寸、速度、排序、坐标系或 padding 与训练端 5x10 描述符不完全一致。
6. **地图和任务尺度不一致**：训练障碍核心区为 40 m x 40 m（含 border 总地形约 50 m x 50 m）、任务边界约 ±21 m；用户的 Gazebo 定量地图为 20 m x 20 m。即使障碍密度相同，视野/路程/边界效应仍不同。
7. **碰撞和超时定义不一致**：碰撞半径、模型 collision 几何、成功半径、timeout 和 deadlock 检测频率可能不同。
8. **静态/动态模型生成物只有 visual 无 collision**：生成器模板已补 collision，但时间戳部署快照正在被 `start.launch` 加载的 world 仍是 `71 visual / 1 collision`，必须重新生成才生效。
9. **试验切换残留控制命令**：时间戳 evaluator 与 navigation 已加入 stop/zero-velocity 和并发 epoch 修复，但尚需 Ubuntu 端实测；仓库内 navigation 仍不是最终并发安全版。
10. **成功定义不一致**：Isaac 只使用三维 `0.5 m` 球形成功区，ROS1 还允许 XY/高度组合和越过目标平面；成功率不能直接横向比较。
11. **起终点合法性不一致**：训练端有静态/动态 `1.5 m` 净空重采样流程，但静态 raster 外点和 fallback 会削弱保证；ROS1 当前连这套筛选流程也没有，可能从障碍物附近甚至膨胀区内起飞。

建议对同一条轨迹逐帧保存以下对照，而不是只比较最终成功率：原始深度/点云、36x4 raycast、5x10 动态描述符、8D state、归一化动作、世界速度命令、时间戳、碰撞/成功判定。

## 12. 已完成修改的总体摘要

### 12.1 论文

- 依据 `6e137e9` 多轮校正 NavRL 和 ForesightNav 的方法表述。
- 重写标题、摘要、Introduction、Related Work 和 Methodology 的关键逻辑。
- 精简 State/Action Space、3D-VO 推导和 Safety Shield，保留方法核心。
- 统一 1,000 次试验口径、deadlock event 定义和表格数据。
- 用 PE-Planner 替换 Ego-Planner。
- 重写 Simulation Results，使其更接近 IEEE 论文的概括式结果分析，避免逐行堆数据。
- 编写 Physical Flight Tests，加入场地、速度上限、四类环境和速度曲线解释。
- 多轮插入、替换、裁剪和重新排版 Fig. 1--6。
- 删除历史上的旧 Fig. 4 和全部伪代码 Algorithm，完成图表编号及正文引用修正；当前正文没有 Algorithm 环境或算法编号。
- 将 Observation 的三组公式合并为一个紧凑公式，同时保留三类观测维度、navigation state 和动态障碍物描述符定义。
- 压缩 3D-VO 的球堆构造与预筛选细节，保留障碍物近似、各向异性膨胀、有限时域 VO、TTC 和风险函数。
- 压缩 Multi-Horizon Outcome Learning 的重复说明，并将联合优化目标改为行内公式；保留预测输出、目标构造、多任务损失、共享编码器更新和部署时移除预测器的说明。
- 精简 deadlock 指标说明、Physical Flight Tests 和 Conclusion；当前无作者版本已从 9 页压缩到 8 页。
- 按师兄意见修正引用归属、术语、图引用和表格加粗。

### 12.2 训练端

- 分析 `37d1092` 教师引导导致碰撞下降但死锁/超时上升的问题。
- 优先实现连续点段分割和教师动作静态扫掠检查。
- 重构默认起终点为四边到正对边的强制穿越任务。
- 保留可选 wall-crossing 模式，用于要求路径与大墙影响区域相交。

### 12.3 ROS1

- 统一部署统计为 trial-level 和 event-level 两套 deadlock 指标。
- 增加 success/failure/collision/timeout/deadlock/recovery 输出。
- 在 Git 内 ROS1 和时间戳部署快照中对齐了 opposite-crossing 的几何结构；部署尺度参数与训练端不同，且 ROS1 尚没有训练端净空筛选。
- 加入试验结束 stop 服务、零速度保持和下一 trial 的 Gazebo model-state 复位。
- 针对实际出现的 `target_dir=None` 并发异常，在时间戳 `navigation.py` 中加入 `RLock + navigation_epoch` 完整修复；Git 内 ROS1 尚未移植该最终版本。
- 为生成障碍物模板补充与 visual 相同的 collision geometry；Git 内生成 world 已更新，时间戳部署快照的运行 world 尚未重新生成。
- 复核并明确当前 evaluator 默认碰撞半径仍是 `0.15 m`；曾讨论的 `0.02 m` 不是最后文件状态，且与 `0.1 m` raycast 步长不兼容。
- 查明 `rosrun` 权限错误的原因是误用 `chmod -x`；正确命令为 `chmod +x`。

## 13. 后续事项，按优先级排序

### P0：先保护和固化当前状态

1. 当前分支、HEAD 和 upstream 均为 `2-14-7-1` / `5e6540bbd822f73b2ff5ae076c9f561d9fef1086`，ahead/behind 为 `0/0`；不要为拆分已共享历史而改写提交。
2. 当前 8 页论文源文件、`icra.pdf`、Fig. 2 源文件/输出、部分构建辅助文件和本交接文档均有未提交修改；提交前分别审阅，不要把临时构建文件盲目纳入版本管理。
3. 先备份 `/Users/yoloflps/Downloads/ros1 2 18.27.45`；确认消失的 `ros1 2` 是否只是被重命名，不要假设二者相同。
4. 把时间戳副本中最终的 `RLock + navigation_epoch` 修复以可审阅 diff 移植回 Git 管理的 `navrl2/ros1`；在此之前，不得把 Git 内 navigation 当成最终部署版。
5. 两张正文依赖图 `fig_training_initial.png`、`fig_training_final.png` 已由 `ba0a7e8` 纳入版本管理；后续移动或重命名时必须同步更新 `icra.tex`。
6. 给所有正式实验记录绑定 commit、完整配置、checkpoint、随机种子、world/PCD hash 和 CSV 路径。

### P0：先完成 Ubuntu 部署闭环

1. 将时间戳快照的最终 evaluator/navigation 同步到 `/home/wzf/navrl1_ws/src/ros1`。
2. 对照第 10.4 节核验 `rospack find`、SHA-256 和执行权限；evaluator 必须 `chmod +x`。
3. 先启动 navigation，再启动 evaluator；确认 `/rl_navigation/stop` 已注册。
4. 重新生成 `generated_env.world`，确认 visual/collision 数量匹配，再重启 Gazebo。
5. 先跑 3--5 个短 trial，逐一验证碰撞、超时、成功三种终止都能立即零速、复位、发布新目标，且不再出现 `NoneType.clone()`。
6. 再跑正式 50/100/1,000 trial，并保存 CSV、所有终端日志和参数 dump。

### P0：统一论文与代码的指标语义

1. 最终决定表 II 最后一列名称：推荐 `Deadlock Event Recovery Rate (%)`。
2. 决定是否同时显示恢复事件数，例如 `24/53 (45.3%)`。
3. 明确区分代码中的 `post_deadlock_success_rate` 和 event recovery rate。
4. 用自动统计脚本从原始 1,000-trial CSV 重新生成 Table II/III，避免手工填表错误。

### P0：验证后续训练改动

1. 在相同课程一 checkpoint、seed 集合和评估配置下比较：
   - 无教师；
   - 原 `37d1092` 教师；
   - 连续点段分割 + 静态扫掠检查教师。
2. 同时报告碰撞、超时、deadlocked trials、deadlock events、event recovery 和最终成功率。
3. 对大型障碍旁存在小静态障碍的专门场景做单元测试和可视化，检查墙体法向是否仍被污染。

### P0：解决 Isaac/Gazebo 对齐

1. 固定同一组起终点和障碍物种子，逐观测字段对比。
2. 核对 36x4 射线顺序、坐标系、归一化和无命中填充值。
3. 核对 5x10 动态描述符的速度坐标系和尺寸编码。
4. 核对动作从模型到控制器的缩放、频率和 frame。
5. 先修正/确认训练端静态 raster 边界语义，再为 ROS1 起终点加入真正等价的静态/动态 `1.5 m` 净空重采样；否则只是表面上使用相同参数名。
6. 决定 ROS1 是否应严格采用训练端三维 `0.5 m` 成功区；若保留越过目标平面成功规则，论文和表格必须明确两者不同。
7. 统一“20 m x 20 m 工作空间”的含义：当前障碍核心区为 `±10 m`，时间戳起终点却为 `±12.5 m`；确认论文表 II 数据到底使用哪组边界。
8. 选择并标定统一碰撞口径。不要直接使用 `_collision_radius:=0.02`；在 `0.1 m` raycast 下它会基本关闭静态检测。
9. 分别记录 Gazebo contact、occupancy evaluator collision 和 safety-shield intervention，避免把三者合并成一个指标。

### P1：清理部署代码并增加自动测试

1. 时间戳 evaluator 中 `success_xy_radius` / `success_height_tolerance` 被重复读取，应去重。
2. `static_map_alpha` 连续赋值两次，第二次 `1.0` 覆盖第一次 `0.65`；明确预期值后只保留一次。
3. stop service 若初始化超时，目前不会重连；可改为每个 trial 结束按需重建 proxy，或在服务不可用时直接中止评估。
4. raycast、dynamic obstacle service 异常当前会被当成“没有碰撞”；正式评估应至少单独计数并打印，避免感知故障抬高成功率。
5. 为采样器加入 10,000 次属性测试：四边分布、正对边映射、切向范围、抖动范围、高度范围和随机种子可复现；另加边界测试，避免 `±21 m` 因落在 `±20 m` 静态 raster 外而未经真实净空检查。
6. 为 navigation 加入可重复并发测试：在 policy inference/rollout 中触发 stop，确保之后没有非零旧命令和旧 rollout 发布。
7. 为每个生成 world 做静态检查：除明确豁免对象外，`visual` 与 `collision` 的几何数量和尺寸应对应。

### P1：投稿前论文收尾

1. 当前无作者版本为 8 页；补作者、单位、致谢和最终元数据后重新检查分页，必要时再做小幅压缩。
2. 再次编译并检查所有引用、图号、表号、字体和图片分辨率，并确认正文仍无残留 Algorithm 引用或编号。
3. 核对 ICRA 当年投稿模板、页数及补充材料要求，避免以当前占位作者版本直接投稿。
4. 决定是否按师兄建议把 Fig. 4 改成多方法轨迹对比。
5. 清理正文中的 TODO 注释：
   - finalized ablation-study statement；
   - large-obstacle qualitative result reference。
6. 检查未使用 BibTeX 项，例如历史上的 `rvo2008`。

### P1：实飞和视频

1. 确认 Fig. 6 的四个环境均有真实、可追溯的飞行日志。
2. 除速度外，建议视频/补充材料展示：轨迹、与最近障碍距离或 TTC、策略/盾牌干预、deadlock 检测与恢复、实时推理频率。
3. 视频结构建议：方法概览、四类环境、动态交互、大型非凸结构绕行、感知叠加、代表性失败、定量摘要。
4. 若使用 `ros1 real/` Phase 1，必须清楚说明当前动态描述符是否为零，避免将仅靠占据变化的结果表述成显式动态速度预测。

## 14. 本交接文档生成时的验证状态

- 已核对当前分支为 `2-14-7-1`，HEAD 与 upstream 均为 `5e6540bbd822f73b2ff5ae076c9f561d9fef1086`，ahead/behind 为 `0/0`；当前工作树另有论文、Fig. 2、正式 PDF/构建产物和本交接文档的未提交修改。
- 已对照当前 `thesis/icra.tex` 核对 Fig. 1--6、Table I--III 和主要实验文字；当前没有 Algorithm 环境、算法编号或正文算法引用。论文为单文件正文，使用本地 IEEEtran class/bst 和 `thesis/ref.bib`。
- 已检查最近构建日志：TeX Live 2026/pdflatex 成功生成 8 页 `icra.pdf`，未发现 fatal、undefined citation、undefined reference 或 overfull box；仍有 `No author given` 和若干 underfull box。加入作者和单位后可能重新增至 9 页。
- 当前文件校验值：`thesis/icra.tex` 为 `478d4c0712dce77a076810e39df801ed41bcef803c34001ff98e6b4dc5fe6203`，`thesis/icra.pdf` 为 `51275da743aa4561dbe2bf2edee700c000b32993a095a27e7035b595e00f3ddc`。
- 已确认 `fig_training_initial.png` 与 `fig_training_final.png` 均被正文引用并受 Git 跟踪。
- 已核对训练端 opposite-crossing 采样公式、128 次净空筛选、静态/动态碰撞、成功、越界和超时源代码。
- 已核对默认 ROS launch 实际选择 `no_map.yaml`，以及 occupancy map 的 `robot_size`、分辨率、膨胀公式和 `RayCast` 使用 inflated map 的 C++ 实现。
- 已核对时间戳 evaluator 的采样、碰撞、成功、deadlock、stop service 和 trial 循环；Python AST 语法检查通过。
- 已核对时间戳 navigation 的 `RLock + navigation_epoch` 修复；Python AST/tabnanny 检查和隔离的“新目标 -> 重复同目标 -> stop -> 新目标”状态测试在修改时通过。
- 已静态统计：Git 内 `generated_env.world` 为 `107 visual / 107 collision`，时间戳运行 world 为 `71 visual / 1 collision`。
- 已从无人机 collision STL 顶点计算缩放后的包围盒和最大水平径向外廓；这些数值是几何审计结果，不是 Gazebo contact 实测结果。
- 尝试运行 `isaac-training/training/tests/test_wall_teacher.py`，但本机系统 Python 缺少 `torch`，测试未执行：`ModuleNotFoundError: No module named 'torch'`。
- 本轮没有运行 Isaac Sim、Gazebo 或实机 ROS 节点，无法确认 Ubuntu 实际文件哈希、stop 时序、物理接触或 reset 画面。
- 本轮没有重新验证 Table II/III 的原始 1,000-trial 日志，因此表内数字只能称为“当前论文数据”，不能称为本轮复现实验结果。

## 15. 新 Codex 会话建议开场提示

可把下面内容作为新会话第一条消息：

> 请先完整阅读 `/Users/yoloflps/Downloads/study/navrl2/PROJECT_HANDOFF.md`，再核对 `git status`、HEAD、论文主稿和所有 ROS1 路径。论文方法描述以 commit `6e137e952fc075b3804c9addadc074691482df02` 为依据；当前分支和 upstream 基线是 `5e6540bbd822f73b2ff5ae076c9f561d9fef1086`。当前无作者论文为 8 页、正文没有 Algorithm，并统一以 `thesis/icra.pdf` 为论文输出；`thesis/icra.tex`、Fig. 2、PDF/构建产物和交接文档仍有未提交修改。最新可见的仿真部署传输快照是 `/Users/yoloflps/Downloads/ros1 2 18.27.45`，完整 `RLock + navigation_epoch` 修复只在该快照的 `navigation.py`；Ubuntu 真正运行的是 `/home/wzf/navrl1_ws/src/ros1`，必须先核对 `rospack find` 和 SHA-256。不要 reset、clean 或覆盖现有修改，也不要把 Git 内 ROS1、下载副本、Ubuntu 运行树和 `ros1 real/` 混成一个版本。

## 16. 接手时推荐先阅读的文件

1. `PROJECT_HANDOFF.md`
2. `thesis/icra.tex`
3. `isaac-training/training/cfg/train.yaml`
4. `isaac-training/training/cfg/ppo.yaml`
5. `isaac-training/training/scripts/train.py`
6. `isaac-training/training/scripts/eval.py`
7. `isaac-training/training/scripts/env.py`
8. `isaac-training/training/scripts/utils.py`
9. `isaac-training/training/scripts/ppo.py`
10. `isaac-training/training/scripts/wall_follow_teacher.py`
11. `isaac-training/training/tests/test_wall_teacher.py`
12. `ros1/navigation_runner/scripts/deployment_eval.py`
13. `ros1/navigation_runner/scripts/navigation.py`
14. `ros1/uav_simulator/scripts/world_generator.py`
15. `ros1 real/navigation_runner/SU17_PHASE1_DEPLOYMENT.md`
16. `/Users/yoloflps/Downloads/ros1 2 18.27.45/navigation_runner/scripts/deployment_eval.py`
17. `/Users/yoloflps/Downloads/ros1 2 18.27.45/navigation_runner/scripts/navigation.py`
18. `/Users/yoloflps/Downloads/ros1 2 18.27.45/navigation_runner/cfg/mapping/sim/no_map.yaml`
19. `/Users/yoloflps/Downloads/ros1 2 18.27.45/map_manager/include/map_manager/occupancyMap.h`
20. `/Users/yoloflps/Downloads/ros1 2 18.27.45/uav_simulator/scripts/world_generator.py`
21. `/Users/yoloflps/Downloads/ros1 2 18.27.45/uav_simulator/worlds/generated_env/generated_env.world`

最后提醒：本项目最容易出错的地方不是单个公式，而是**版本、指标分母、传感器接口和场景尺度的混用**。任何新的实验结论都应先回答四个问题：使用哪个 commit、哪个 checkpoint、哪份配置、哪套统计定义。

## 17. 本轮对话的决策与修改时间线

这一节专门保留本对话的来龙去脉，避免后来只看到最终代码而不知道为什么这样改。

1. **发现现象**：evaluator 报 `static collision, success=false` 并把轨迹画成红色，但 Gazebo/RViz 里无人机肉眼未接触障碍物，且继续飞到终点。
2. **碰撞半径讨论**：用户希望把 evaluator 的再检查半径从 `0.15 m` 改为 `0.02 m`。进一步审计发现静态 raycast 查询的是按 `robot_size` 膨胀后的 occupancy map，不是原始 Gazebo 几何。
3. **训练/部署对比**：训练端静态碰撞是干净网格射线 `<0.3 m`；部署端是膨胀体素射线 `<=collision_radius`。两者的数据源、膨胀、动态阈值和成功条件都不相同。
4. **确认量化限制**：部署地图/射线步长为 `0.1 m`，所以 `0.02 m` 无法表达真正的 2 cm 静态阈值；最后文件恢复/保留默认 `0.15 m`。
5. **障碍物物理几何**：应用户要求，随机静态和动态 box/cylinder 的生成模板加入与 visual 相同的 collision。其意义仅限于 Gazebo 物理接触/阻挡，不会直接替换 evaluator 的 occupancy 判据。
6. **尺寸统一讨论后撤销**：比较了 Gazebo 无人机 mesh、occupancy `robot_size` 和 shield `robot_size`；用户要求撤销当时的尺寸修改。因此当前仍保留原有多个尺寸口径，没有宣称统一完成。
7. **解释继续飞行根因**：旧 evaluator 只结束自身 trial 循环，没有向独立 navigation 节点发送取消/停止；红轨迹只是结果可视化。
8. **实现 trial-stop**：evaluator 增加 `/rl_navigation/stop` client，navigation 增加同名 service 和零速度保持；下一 trial 通过 `/gazebo/set_model_state` 复位，再发布新目标。
9. **对齐起终点几何**：将部署 evaluator 改为 `opposite_crossing_eval`，四边随机起点、严格正对边目标，目标靠近中心对称点。训练尺度为 `21/18/4 m`，部署快照采用 `12.5/9/2 m`。
10. **澄清实际代码路径**：用户指出真正部署代码是下载目录而非 Git 内 `ros1/`；先后使用过 `ros1 2` 和 `ros1 2 18.27.45`。最后只剩时间戳目录可核验。
11. **修复执行权限误操作**：`rosrun` 报 evaluator 不可执行，原因是用户运行了 `chmod -x`。正确做法是 `chmod +x`。
12. **出现停止竞态**：stop 日志已经打印，但 timer 线程在 rollout 中访问被清空的 `target_dir`，触发 `NoneType.clone()`。
13. **完成并发安全修复**：在时间戳 `navigation.py` 引入锁、epoch、状态快照和发布前校验，阻止取消 trial 的旧命令/rollout 泄漏到新 trial。
14. **最后答复边界**：针对这次异常本身，只更新 `navigation.py` 即可；但完整评估功能仍依赖配套 evaluator，且所有修改必须真正同步到 Ubuntu 并重启进程。
15. **按师兄批注系统修订论文**：调整标题、摘要、Introduction、Related Work、Methodology 和 Experiment Results 的逻辑、引用归属、术语及图表引用；明确论文方法仍以 `6e137e9` 为解释基准。
16. **图表结构定稿**：当前保留 Fig. 1--6 和 Table I--III；删除历史旧 Fig. 4 后重新编号，并将仿真轨迹、走廊轨迹和实飞图分别归入对应实验段落。
17. **删除全部伪代码**：先删除旧 Algorithm 1 并重编号，随后为压缩篇幅删除剩余 Multi-Horizon 伪代码及相关引用；当前正文没有 Algorithm。
18. **统一实验命名与数值语义**：Table II 的消融统一为 `ForesightNav w/o TTC-aware 3D-VO reward`；Table III 的 Timeout Rate 由 `N/A` 改为 `0`；deadlock trial、deadlock event 和 event recovery 的分母保持区分。
19. **方法与实验文字压缩**：合并 Observation 公式，压缩 3D-VO 球堆/预筛选、Multi-Horizon 后续说明、deadlock 指标、Physical Flight Tests 和 Conclusion，同时保留核心公式及定义。
20. **当前分页结果**：最新 `icra.pdf` 为 8 页，引用和文献均解析、无 overfull box；作者/单位仍为空，补齐元数据后必须重新检查分页。`icra_preview.pdf` 已删除，后续只维护 `icra.pdf`。

## 18. 下一位接手者的最短执行清单

若下一项任务仍是 Gazebo 评估，建议严格按顺序执行：

1. 读完第 1.4、7.5--7.9、10.4--10.7 节。
2. 备份时间戳部署快照和 Ubuntu 当前两份脚本。
3. 核对 Ubuntu `rospack find`，再同步 evaluator/navigation 并核对 SHA-256。
4. `chmod +x deployment_eval.py`，重启 navigation/evaluator。
5. 同步正确 `world_generator.py`，重新生成 world，确认 `<visual>` 与 `<collision>` 数量对应，重启 Gazebo。
6. 用 3--5 个 trial 验证 collision、timeout、success 均能 stop -> zero hold -> reset -> next goal。
7. 保存 `/rosparam dump`、world/PCD hash、checkpoint、seed、CSV 和终端日志。
8. 只有在小规模闭环通过后，才跑 50/100/1,000 trial 正式实验。
9. 若目标是与训练端公平比较，先决定并实现相同的起终点净空、成功区、碰撞口径和越界规则。
10. 将最终确认过的部署修复回迁到 Git 管理目录并提交，停止继续制造新的 `ros1 copy` 目录。
