# Methodology

## A. Finite-Horizon TTC-Aware Three-Dimensional Velocity-Obstacle Modeling

Instantaneous clearance alone cannot distinguish obstacles with similar separation but different relative motion. LP-Nav therefore uses a finite-horizon three-dimensional velocity-obstacle (3D-VO) formulation [7] to quantify dynamic motion conflicts. For a transition from `x_t` to `x_(t+1)` induced by action `a_t`, the model is evaluated from the resulting state and realized UAV velocity at `t+1`. To represent obstacle geometry without treating all interactions as planar, an obstacle of size `(w_x_i, w_y_i, h_i)` is approximated by an overlapping vertical stack of spherical primitives. Each sphere has radius `b_i = 0.5*sqrt(w_x_i^2 + w_y_i^2)`, adjacent centers are separated by `l_i = max(w_x_i, w_y_i)`, and the stack contains `K_i = ceil(h_i/l_i)` spheres. All spheres inherit the obstacle velocity.

Before collision evaluation, LP-Nav masks vertically irrelevant sphere primitives from the current TTC calculation. For a sphere with post-transition relative height `r_(j,t+1)^z` and vertical relative velocity `u_(j,t+1)^z`, its predicted height interval is bounded by `r_(j,t+1)^z` and `r_(j,t+1)^z + H*u_(j,t+1)^z`. The sphere is retained only if this interval intersects the vertical relevance band `[-(b_i + m_h), b_i + m_h]`, where the height margin is chosen such that `m_h >= m_z`. This screening band therefore contains the vertical collision band `[-(b_i + m_z), b_i + m_z]` and does not discard a primitive solely because of a narrower vertical prefilter. For each retained sphere `j`, let `r_(j,t+1) = p^o_(j,t+1) - p_(t+1)` and `u_(j,t+1) = v^o_(j,t+1) - v_(t+1)` denote its relative position and velocity with respect to the UAV. The sphere is inflated using horizontal and vertical safety scales `d_xy_j = b_i + m_xy` and `d_z_j = b_i + m_z`, where `m_xy` and `m_z` approximate the horizontal and vertical extents of the UAV collision envelope. These scales are summarized by `D_j = diag(d_xy_j, d_xy_j, d_z_j)`. Under constant relative velocity, its finite-horizon velocity obstacle is defined as `VO^H_(j,t+1) = {v in R^3 | there exists s in (0,H] such that ||D_j^(-1)[r_(j,t+1) + s(v^o_(j,t+1) - v)]||_2 <= 1}`.

LP-Nav evaluates whether the realized UAV velocity `v_(t+1)` belongs to `VO^H_(j,t+1)`, rather than explicitly constructing the complete set or selecting and projecting a new velocity. Substituting `v_(t+1)` and scaling the post-transition relative position and velocity by `D_j` yields the boundary-intersection condition `A_j*s^2 + B_j*s + C_j = 0`, where `A_j = dot(u_bar_j, u_bar_j)`, `B_j = 2*dot(r_bar_j, u_bar_j)`, and `C_j = dot(r_bar_j, r_bar_j) - 1`. For a non-overlapping sphere, a valid time to collision exists when the relative motion is nonzero and approaching, the discriminant `Delta_j = B_j^2 - 4*A_j*C_j` is nonnegative, and the first root `T_j = (-B_j - sqrt(Delta_j))/(2*A_j)` lies within `(0, H]`.

A sphere already inside the inflated region is assigned unit risk. A valid in-horizon conflict receives the exponentially decaying TTC-dependent urgency score `rho_(j,t+1) = exp(-T_j/tau)`, while motion with no conflict within the horizon receives zero risk. The overall post-transition risk is `rho_VO_(t+1) = max_j rho_(j,t+1)` over the retained sphere primitives. This aggregation emphasizes the most imminent conflict, keeps the risk bounded in `[0, 1]`, and prevents its magnitude from scaling linearly with the number of sphere primitives. The scalar TTC risk is computed from simulator states only during training and is not included in the policy observation. Its incorporation into policy learning is described in the following section, and it introduces no additional deployment-time computation beyond NavRL's existing actor-and-shield pipeline. The formulation assumes constant relative velocity within the finite horizon and provides a training-time risk measure rather than a formal safety guarantee.

## B. Reinforcement Learning Formulation

### 1) Problem Formulation

LP-Nav formulates local UAV navigation as a partially observable Markov decision process `M = (S, A, P, R, Omega, O, gamma)`, where `S` is the latent environment state, `A` is the action space, `P` is the transition function, `R` is the reward function, `Omega` is the observation space, `O` is the observation model, and `gamma in [0,1)` is the discount factor. At time `t`, the policy receives `o_t ~ O(. | x_t)`, samples `a_t ~ pi(. | o_t)`, and induces the transition

`x_(t+1) ~ P(. | x_t, a_t),  r_t = R(x_t, a_t, x_(t+1))`.

Its objective is

`pi* = arg max_pi E_pi[sum_(t=0)^(T-1) gamma^t r_t]`.

Privileged simulator variables may be used to construct training rewards and auxiliary targets, but they are excluded from policy inference. Policy inputs are restricted to local quantities exposed through the deployment-time navigation interface. During simulation training, the corresponding dynamic-obstacle quantities are generated from simulator states. At the beginning of each episode, LP-Nav defines a navigation frame `G` whose horizontal `x`-axis points from the initial UAV position toward the goal, whose `z`-axis is aligned with the world vertical, and whose `y`-axis completes a right-handed frame. The orientation of `G` remains fixed throughout the episode.

### 2) State and Observation Space

Because the complete state `x_t in S` is not observable, the policy input is the local observation

`o_t = (L_t, s_t, D_t),  L_t in R^(36 x 4),  s_t in R^8,  D_t in R^(N_d x 10)`,

where `N_d = 5`. The LiDAR tensor `L_t` contains `N_L = 144` proximity measurements arranged as 36 horizontal directions with `10 deg` angular resolution and four vertical beams at `[-10, 0, 10, 20] deg`. Its horizontal indexing follows the UAV yaw-aligned sensor frame: horizontal beam `k` is cast at azimuth `psi_t + 10k deg`, where `psi_t` is the UAV yaw. At reset, `psi_0` is aligned with the horizontal `x`-axis of `G`; yaw is not part of the learned action. A ray measured at distance `d` is encoded as the clipped proximity `R_L-d`, where `R_L` is the sensing range, so larger values indicate closer geometry. The navigation vector is

`s_t = [r_hat_g^G, d_g^xy, Delta z_g, v_t^G]`,

where `r_hat_g^G` is the three-dimensional unit direction from the current UAV position to the goal expressed in the fixed frame `G`, `d_g^xy` is the horizontal goal distance, `Delta z_g` is the vertical goal displacement, and `v_t^G` is the UAV velocity in `G`. In particular, the lateral component of `r_hat_g^G` retains the signed cross-track direction after the UAV departs from the initial start-to-goal line.

Let `r_(i,t) = p_(i,t)^o-p_t` denote the relative position of dynamic obstacle `i`, and let `d_(i,t)^xy = ||r_(i,t)^xy||_2`. Up to `N_d` obstacles are ordered by increasing horizontal center distance. Candidates outside the sensing range are treated as invalid, and rows not occupied by valid candidates are zero padded. The resulting matrix is `D_t = [d_(1,t), ..., d_(N_d,t)]^T`, with

`d_(i,t) = [r_hat_(i,t)^G, d_(i,t)^xy, Delta z_(i,t), v_(i,t)^(o,G), q_w(w_i), q_h(h_i)] in R^10`,

where `r_hat_(i,t)^G` is the three-dimensional unit relative-position vector expressed in `G`, `Delta z_(i,t)` is the signed vertical displacement, and `v_(i,t)^(o,G)` is the obstacle's absolute velocity rotated into `G`, rather than its velocity relative to the UAV. The width code is `q_w(w_i) = clip(w_i/delta_w-1, 0, 3)`, with `delta_w=0.25 m`. For obstacles with `h_i <= 1.0 m`, the finite height is represented explicitly as `q_h(h_i)=h_i`; taller, vertically spanning obstacles use `Delta z_(i,t)=0` and `q_h(h_i)=0`. Thus, neither size entry should be interpreted as a generally normalized physical dimension. The training implementation obtains these quantities from simulator states, while the deployment interface supplies position, velocity, and size estimates from its dynamic-obstacle detector. Because the dynamic-obstacle encoder flattens the ordered rows rather than applying permutation-invariant pooling, the distance-based ordering is part of the observation definition. The fixed horizontal direction defining `G` is retained separately for transforming actions into the world frame and is not concatenated into the shared policy feature.

### 3) Action Space

The action space is the bounded three-dimensional velocity set `A = [-v_max,v_max]^3` in `G`. Given the shared observation feature, the actor produces positive parameters `alpha_t` and `beta_t` for three independent Beta distributions. A normalized action is sampled and rescaled as

`a_tilde_t ~ Beta(alpha_t, beta_t),  a_t^G = v_max(2*a_tilde_t-1)`.

The resulting command is rotated into the world frame as `a_t^W = R_(G->W) a_t^G`. During simulation training, a low-level controller converts it to rotor inputs; at deployment, the world-frame velocity command passes through NavRL's existing shield before publication. Yaw is not part of the learned action.

### 4) Reward Function

The reward retains four task-defining components and groups the remaining low-weight baseline shaping terms into `r_reg_t`:

`r_t = r_progress_t + r_clearance_t + r_trap_t + r_VO_t + r_reg_t + r_term_t`.

**Goal progress.** Let `d_t = ||p_g-p_t||_2`. For the transition induced by `a_t`, define `Delta d_t = d_t-d_(t+1)`. The nonterminal progress contribution is

`r_progress_t = lambda_g * Delta d_t`.

The complete signed distance change is retained: movement toward the goal receives positive reward, whereas an equal retreat receives an equal penalty. Consequently, a closed forward-backward cycle cannot obtain positive undiscounted progress reward solely from asymmetric shaping.

**Local clearance.** For safety margin `m`, define the margin-scaled hinge penalty `h(x) = max(m-x,0)/m`. Let `mu_(i,t+1) in {0,1}` indicate whether dynamic-obstacle entry `i` is valid and lies within the sensing range after the transition. The clearance contribution is

`r_clearance_t = -lambda_s*(1/N_L)*sum_k h(d_(k,t+1)) - lambda_d*(1/N_d)*sum_i mu_(i,t+1)*h(c_(i,t+1))`,

where `d_(k,t+1)` is the distance measured by LiDAR ray `k`, `c_(i,t+1)` is the estimated surface clearance of tracked obstacle `i`, and `N_L` is the number of LiDAR rays. The implementation equivalently assigns the maximum sensing range `R_L` to invalid dynamic entries, for which `h(R_L)=0`. Division by the fixed capacity `N_d` keeps the reward scale consistent while allowing the aggregate penalty to increase with the number of nearby valid obstacles. Because an estimated surface clearance may be negative during geometric overlap, `h(x)` is not clipped above one.

**Trapping and recovery.** After the transition, let `c_(t+1)^stall` count consecutive steps with low displacement, low speed, and negligible goal progress. Its normalized penalty scale is `q_(t+1) = clip((c_(t+1)^stall-W_stall)/W_ramp,0,1)`. A separate blocked-state counter `c_(t+1)^block` increments while a forward obstacle co-occurs with insufficient progress. A recovery event is armed only when this updated counter first reaches `W_block`, at which point LP-Nav stores the UAV position `p_a`, goal distance `d_a`, and forward clearance `f_a`.

Within a finite recovery window, define the one-shot event

`e_t = I[z_t^esc=1, b_(t+1)=0, ||p_(t+1)^xy-p_a^xy||_2 >= delta_p, ((d_a-d_(t+1)) >= delta_g or (f_(t+1)-f_a) >= delta_f)]`,

where `z_t^esc` indicates that recovery is armed before evaluating the transition outcome and `b_(t+1)` denotes the resulting blocked low-progress condition. The event therefore requires a minimum horizontal displacement together with either recovered goal progress or increased forward clearance. It can trigger at most once per armed event, expires if recovery is not achieved within the prescribed window, and initiates a cooldown after activation. The combined contribution is

`r_trap_t = -lambda_stall*q_(t+1) + lambda_escape*e_t`,

so a transient counter reset alone does not receive a recovery reward. This mechanism encourages demonstrated local recovery without assuming global route feasibility.

**TTC-aware 3D-VO risk.** The risk from Sec. A enters the reward as

`r_VO_t = -lambda_VO*eta(n)*rho_VO_(t+1),  eta(n) = min(n/N_warm,1)`,

where `n` is the training step and `v_(t+1)` is the realized UAV velocity after executing `a_t`. Thus, the active VO term evaluates the transition outcome rather than the commanded velocity directly. The warmup prevents the analytic risk signal from immediately dominating basic goal-directed learning. The term `r_term_t` collects goal-arrival rewards and penalties for collision, altitude-bound violations, and timeout. Network construction and PPO optimization are described in the following section.
