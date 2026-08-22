# Related Work

## Classical, Learning-Based, and Hybrid Navigation

Classical UAV navigation uses graph search, sampling-based planning, and receding-horizon trajectory optimization. Optimization-based planners such as EGO-Planner, FASTER, and ViGO produce interpretable, constraint-aware trajectories but replan online as maps or predicted obstacle states change [1]–[3]. Learning-based methods shift computation to offline experience: CADRL learns an interaction-aware value function [4], Long *et al.* map range and goal observations to continuous controls [5], and NavRL uses PPO for direct 3D UAV velocity control and applies a deployment-time shield inspired by the velocity-obstacle (VO) formulation [6]. Although discounted returns propagate future consequences, scalar reward and return signals provide only indirect and entangled supervision of distinct future collision, deadlock, clearance, and progress outcomes.

Hybrid and safety-augmented methods rely on deployment-time planning, recovery, or action projection [16]–[18], [27], [28]. Other approaches use planner demonstrations or privileged expert trajectories during training [20], [21]. Lee *et al.* specifically use privileged time-of-arrival maps to guide quadrotors around large walls and dead ends [22]. LP-Nav retains NavRL's existing VO shield, but its added VO reward and auxiliary objective are training-only and add no computation to deployment-time action selection. Joint training-time supervision of three-dimensional motion conflicts and semantically separated near-term navigation outcomes remains limited.

## Velocity-Obstacle-Guided Learning

The VO formulation defines velocities that lead to collision under an assumed relative-motion model [7]. Classical VO methods select velocities outside these forbidden regions; ORCA distributes avoidance responsibility and solves for collision-free velocities online [8]. Used as a shield, however, VO geometry filters a nominal policy's command after action generation and therefore does not train proactive avoidance.

VO structure has also shaped reinforcement-learning rewards. DRL-VO uses a two-dimensional directional reward with LiDAR histories and pedestrian kinematics [9], while Han *et al.* include RVO area and expected collision time for differential-drive robots [23]. NavRL instead applies a VO-inspired post-policy shield [6]. LP-Nav derives finite-horizon, time-to-collision-aware risk from a three-dimensional VO model, including obstacle geometry and vertical separation, to supervise short-horizon policy learning while retaining NavRL's shield as a final runtime filter. Neither the 3D VO reward nor the VO-inspired shield provides a formal safety guarantee.

## Predictive Models and Auxiliary Outcome Learning

Predictive navigation methods learn representations or models of future experience. NavRep predicts latent dynamics before policy optimization [10]; RGL combines relational interaction modeling with multi-step crowd lookahead [11]; and BADGR predicts action-conditioned events for online candidate-sequence evaluation [12]. General latent models can introduce prediction error, while online rollout evaluation adds deployment computation. Auxiliary prediction can instead enrich representations while retaining direct action inference: general value functions predict temporally extended signals [24], UNREAL combines reward prediction with auxiliary control and value replay [25], and SPR learns action-conditioned multi-step latent predictions [26].

Compared with existing approaches, ForesightNav combines the direct action inference of learning-based navigation with structured finite-horizon motion-risk modeling and task-oriented predictive supervision. Its TTC-aware 3D-VO reward guides the policy to respond proactively to dynamic motion conflicts, while action-conditioned multi-horizon outcome learning encourages the shared representation to anticipate collision, deadlock, clearance, and progress. These complementary training signals improve navigation around dynamic obstacles and large non-convex structures without requiring online predictive rollouts or adding computation to policy inference.

<!--
Provisional citation map. Renumber globally after the experimental baselines and bibliography are finalized.
[1] Zhou et al., "EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors," RA-L, 2021. https://arxiv.org/abs/2008.08835
[2] Tordesillas et al., "FASTER: Fast and Safe Trajectory Planner for Navigation in Unknown Environments," T-RO, 2022. https://arxiv.org/abs/2001.04420
[3] Xu et al., "ViGO: Vision-Aided UAV Navigation and Dynamic Obstacle Avoidance Using Gradient-Based B-Spline Trajectory Optimization," ICRA, 2023. https://arxiv.org/abs/2209.07003
[4] Chen et al., "Decentralized Non-Communicating Multiagent Collision Avoidance with Deep Reinforcement Learning," ICRA, 2017. https://arxiv.org/abs/1609.07845
[5] Long et al., "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning," ICRA, 2018. https://doi.org/10.1109/ICRA.2018.8461113
[6] Xu et al., "NavRL: Learning Safe Flight in Dynamic Environments," RA-L, 2025. https://arxiv.org/abs/2409.15634
[7] Fiorini and Shiller, "Motion Planning in Dynamic Environments Using Velocity Obstacles," IJRR, 1998. https://doi.org/10.1177/027836499801700706
[8] van den Berg et al., "Reciprocal n-Body Collision Avoidance," ISRR, 2011. https://gamma-web.iacs.umd.edu/ORCA/publications/ORCA.pdf
[9] Xie and Dames, "DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles," T-RO, 2023. https://doi.org/10.1109/TRO.2023.3257549
[10] Dugas et al., "NavRep: Unsupervised Representations for Reinforcement Learning of Robot Navigation in Dynamic Human Environments," ICRA, 2021. https://arxiv.org/abs/2012.04406
[11] Chen et al., "Relational Graph Learning for Crowd Navigation," IROS, 2020. https://arxiv.org/abs/1909.13165
[12] Kahn et al., "BADGR: An Autonomous Self-Supervised Learning-Based Navigation System," RA-L, 2021. https://arxiv.org/abs/2002.05700
[13]–[15] Reserved for the final experimental baselines named in the Introduction.
[16] Faust et al., "PRM-RL: Long-Range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-Based Planning," ICRA, 2018. https://arxiv.org/abs/1710.03937
[17] Kastner et al., "Connecting Deep-Reinforcement-Learning-Based Obstacle Avoidance with Conventional Global Planners Using Waypoint Generators," IROS, 2021. https://arxiv.org/abs/2104.03663
[18] Semnani et al., "Multi-Agent Motion Planning for Dense and Dynamic Environments via Deep Reinforcement Learning," RA-L, 2020. https://arxiv.org/abs/2001.06627
[20] He et al., "Deep Reinforcement Learning Based Local Planner for UAV Obstacle Avoidance Using Demonstration Data," arXiv, 2020. https://arxiv.org/abs/2008.02521
[21] Loquercio et al., "Learning High-Speed Flight in the Wild," Science Robotics, 2021. https://arxiv.org/abs/2110.05113
[22] Lee et al., "Quadrotor Navigation Using Reinforcement Learning with Privileged Information," arXiv, 2025 (revised 2026). https://arxiv.org/abs/2509.08177
[23] Han et al., "Reinforcement Learned Distributed Multi-Robot Navigation with Reciprocal Velocity Obstacle Shaped Rewards," arXiv, 2022. https://arxiv.org/abs/2203.10229
[24] Sutton et al., "Horde: A Scalable Real-Time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction," AAMAS, 2011. https://aamas.csc.liv.ac.uk/Proceedings/aamas2011/papers/A6_R70.pdf
[25] Jaderberg et al., "Reinforcement Learning with Unsupervised Auxiliary Tasks," ICLR, 2017. https://arxiv.org/abs/1611.05397
[26] Schwarzer et al., "Data-Efficient Reinforcement Learning with Self-Predictive Representations," ICLR, 2021. https://arxiv.org/abs/2007.05929
[27] Thananjeyan et al., "Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones," RA-L, 2021. https://arxiv.org/abs/2010.15920
[28] Kochdumper et al., "Provably Safe Reinforcement Learning via Action Projection Using Reachability Analysis and Polynomial Zonotopes," IEEE Open Journal of Control Systems, 2023. https://doi.org/10.1109/OJCSYS.2023.3256305
-->
