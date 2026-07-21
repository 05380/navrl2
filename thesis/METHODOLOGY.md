# Methodology

## A. Finite-Horizon TTC-Aware Three-Dimensional Velocity-Obstacle Modeling

Instantaneous clearance alone cannot distinguish obstacles with similar separation but different relative motion. LP-Nav therefore uses a finite-horizon three-dimensional velocity-obstacle (3D-VO) formulation [7] to quantify dynamic motion conflicts. To represent obstacle geometry without treating all interactions as planar, an obstacle of size `(w_x_i, w_y_i, h_i)` is approximated by an overlapping vertical stack of spherical primitives. Each sphere has radius `b_i = 0.5*sqrt(w_x_i^2 + w_y_i^2)`, adjacent centers are separated by `l_i = max(w_x_i, w_y_i)`, and the stack contains `K_i = ceil(h_i/l_i)` spheres. All spheres inherit the obstacle velocity.

Before collision evaluation, LP-Nav masks vertically irrelevant sphere primitives from the current TTC calculation. For a sphere with current relative height `r_z` and vertical relative velocity `u_z`, its predicted height interval is bounded by `r_z` and `r_z + H*u_z`. The sphere is retained only if this interval intersects the vertical relevance band `[-(b_i + m_h), b_i + m_h]`, where the height margin is chosen such that `m_h >= m_z`. This screening band therefore contains the vertical collision band `[-(b_i + m_z), b_i + m_z]` and does not discard a primitive solely because of a narrower vertical prefilter. For each retained sphere `j`, let `r_j,t = p_o_j,t - p_t` and `u_j,t = v_o_j,t - v_t` denote its relative position and velocity with respect to the UAV. The sphere is inflated using horizontal and vertical safety scales `d_xy_j = b_i + m_xy` and `d_z_j = b_i + m_z`, where `m_xy` and `m_z` approximate the horizontal and vertical extents of the UAV collision envelope. These scales are summarized by `D_j = diag(d_xy_j, d_xy_j, d_z_j)`. Under constant relative velocity, its finite-horizon velocity obstacle is defined as `VO^H_j,t = {v in R^3 | there exists s in (0,H] such that ||D_j^(-1)[r_j,t + s(v_o_j,t - v)]||_2 <= 1}`.

LP-Nav evaluates whether the realized UAV velocity `v_t` belongs to `VO^H_j,t`, rather than explicitly constructing the complete set or selecting and projecting a new velocity. Substituting `v_t` and scaling relative position and velocity by `D_j` yields the boundary-intersection condition `A_j*s^2 + B_j*s + C_j = 0`, where `A_j = dot(u_bar_j, u_bar_j)`, `B_j = 2*dot(r_bar_j, u_bar_j)`, and `C_j = dot(r_bar_j, r_bar_j) - 1`. For a non-overlapping sphere, a valid time to collision exists when the relative motion is nonzero and approaching, the discriminant `Delta_j = B_j^2 - 4*A_j*C_j` is nonnegative, and the first root `T_j = (-B_j - sqrt(Delta_j))/(2*A_j)` lies within `(0, H]`.

A sphere already inside the inflated region is assigned unit risk. A valid in-horizon conflict receives the exponentially decaying TTC-dependent urgency score `rho_j,t = exp(-T_j/tau)`, while motion with no conflict within the horizon receives zero risk. The overall 3D-VO risk is the maximum `rho_j,t` among the retained sphere primitives. This aggregation emphasizes the most imminent conflict, keeps the risk bounded in `[0, 1]`, and prevents its magnitude from scaling linearly with the number of sphere primitives. The scalar TTC risk is computed from simulator states only during training and is not included in the policy observation. Its incorporation into policy learning is described in the following section, and it introduces no additional deployment-time computation beyond NavRL's existing actor-and-shield pipeline. The formulation assumes constant relative velocity within the finite horizon and provides a training-time risk measure rather than a formal safety guarantee.

## B. Reinforcement Learning Formulation

### 1) Problem Formulation

LP-Nav formulates local UAV navigation as a partially observable Markov decision process `M = (S, A, P, R, Omega, O, gamma)`, where `S` is the latent environment state, `A` is the action space, `P` is the transition function, `R` is the reward function, `Omega` is the observation space, `O` is the observation model, and `gamma in [0,1)` is the discount factor. At time `t`, the policy receives a local observation `o_t in Omega` and selects a velocity action according to `pi(a_t | o_t)`. Its objective is

`pi* = arg max_pi E_pi[sum_(t=0)^(T-1) gamma^t r_t]`.

Privileged simulator variables may be used to construct training rewards and auxiliary targets, but they are excluded from policy inference. The policy instead uses onboard-equivalent local sensing. Navigation quantities are represented in a time-varying goal-aligned frame `G_t`, whose origin is the UAV position, whose horizontal `x`-axis points toward the current goal, and whose `z`-axis is aligned with the world vertical; the `y`-axis completes a right-handed frame.

### 2) State and Observation Space

Because the complete state `x_t^env in S` is not observable, the policy input is the local observation

`o_t = (L_t, s_t, D_t),  L_t in R^(36 x 4),  s_t in R^8,  D_t in R^(N_d x 10)`,

where `N_d = 5`. The LiDAR tensor `L_t` contains `N_L = 144` proximity measurements arranged as 36 horizontal directions and four vertical beams. A ray measured at distance `d` is encoded as the clipped proximity `R_L-d`, where `R_L` is the sensing range, so larger values indicate closer geometry. The navigation vector is

`s_t = [r_hat_g^G, d_g^xy, Delta z_g, v_t^G]`,

where `r_hat_g^G` is the three-dimensional unit direction to the goal, `d_g^xy` is the horizontal goal distance, `Delta z_g` is the vertical goal displacement, and `v_t^G` is the UAV velocity, all expressed in `G_t` where applicable.

The matrix `D_t` represents the `N_d` nearest tracked dynamic obstacles within the sensing range. Each row contains the obstacle's normalized relative direction, horizontal distance, vertical displacement, velocity, width indicator, and height indicator. Missing or out-of-range entries are zero padded. The horizontal goal direction defining `G_t` is retained separately for transforming actions into the world frame and is not concatenated into the shared policy feature.

### 3) Action Space

The action space is the bounded three-dimensional velocity set `A = [-v_max,v_max]^3` in `G_t`. Given the shared observation feature, the actor produces positive parameters `alpha_t` and `beta_t` for three independent Beta distributions. A normalized action is sampled and rescaled as

`a_tilde_t ~ Beta(alpha_t, beta_t),  a_t^G = v_max(2*a_tilde_t-1)`.

The resulting command is rotated into the world frame as `a_t^W = R_(G_t->W) a_t^G`. During simulation training, a low-level controller converts it to rotor inputs; at deployment, the world-frame velocity command passes through NavRL's existing shield before publication. Yaw is not part of the learned action.

### 4) Reward Function

The reward retains four task-defining components and groups the remaining low-weight baseline shaping terms into `r_reg_t`:

`r_t = r_progress_t + r_clearance_t + r_trap_t + r_VO_t + r_reg_t + r_term_t`.

**Goal progress.** Let `d_t = ||p_g-p_t||_2` and `Delta d_t = d_(t-1)-d_t`. To permit short retreat maneuvers around extended obstacles, negative progress is attenuated when the forward region is blocked:

`r_progress_t = lambda_g * Delta d_t_tilde`,

where `Delta d_t_tilde = beta_b*Delta d_t` if a forward obstacle is present and `Delta d_t < 0`, and `Delta d_t_tilde = Delta d_t` otherwise, with `0 <= beta_b <= 1`.

**Local clearance.** For safety margin `m`, define the normalized hinge function `h(x) = max(m-x,0)/m`. The clearance contribution is

`r_clearance_t = -lambda_s*(1/N_L)*sum_k h(d_k,t) - lambda_d*(1/N_d)*sum_i h(c_i,t)`,

where `d_k,t` is the distance measured by LiDAR ray `k`, `c_i,t` is the estimated surface clearance of tracked obstacle `i`, and `N_L` is the number of LiDAR rays. This term penalizes only geometry lying within the prescribed margin.

**Trapping and recovery.** Let `c_t^stall` count consecutive transitions with low displacement, low speed, and negligible goal progress, and let `c_t^block` count obstacle-blocked transitions with insufficient progress. The combined contribution is

`r_trap_t = -lambda_stall*q_t + lambda_escape*e_t`,

with `q_t = clip((c_t^stall-W_stall)/W_ramp,0,1)` and `e_t = clip((c_(t-1)^block-c_t^block)/W_block,0,1)`. Thus, the penalty grows only after persistent stalling, whereas a decrease in the blocked-state counter rewards recovery without assuming global route feasibility.

**TTC-aware 3D-VO risk.** The risk from Sec. A enters the reward as

`r_VO_t = -lambda_VO*eta(n)*rho_VO_t,  eta(n) = min(n/N_warm,1)`,

where `n` is the training step. The warmup prevents the analytic risk signal from immediately dominating basic goal-directed learning. The term `r_term_t` collects goal-arrival rewards and penalties for collision, altitude-bound violations, and timeout. Network construction and PPO optimization are described in the following section.
