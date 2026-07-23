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

Reward terms and auxiliary-supervision targets may depend on latent variables that are excluded from the policy observation. The policy itself receives only the local navigation quantities defined below. At the beginning of each episode, LP-Nav defines a navigation frame `G` whose horizontal `x`-axis points from the initial UAV position toward the goal, whose `z`-axis is aligned with the world vertical, and whose `y`-axis completes a right-handed frame. The orientation of `G` remains fixed throughout the episode.

### 2) State and Observation Space

Because the complete state `x_t in S` is not observable, the policy input is the local observation

`o_t = (L_t, s_t, D_t),  L_t in R^(36 x 4),  s_t in R^8,  D_t in R^(N_d x 10)`,

where `N_d = 5`. The LiDAR tensor `L_t` contains `N_L = 144` proximity measurements arranged as 36 horizontal directions with angular resolution `Delta psi = 10 deg` and four elevation angles `phi_m in [-10, 0, 10, 20] deg`. Entry `L_t[k,m]`, with `k in {0,...,35}`, corresponds to a ray cast at world-frame azimuth `psi_t + k*Delta psi` and elevation `phi_m`, where `psi_t` is the UAV yaw. The scan is indexed in this yaw-aligned sensor frame and is not rotated or re-indexed into `G`; roll and pitch are excluded from the ray orientation. The yaw is initialized along the horizontal `x`-axis of `G` and regulated independently of the learned translational policy. Under nominal yaw regulation, the sensor and navigation frames therefore remain aligned; a yaw disturbance instead shifts the azimuthal indexing of `L_t` relative to `G`. A ray measured at distance `d` is encoded as the clipped proximity `R_L-d`, where `R_L` is the sensing range, so larger values indicate closer geometry. The navigation vector is

`s_t = [r_hat_g^G, d_g^xy, Delta z_g, v_t^G]`,

where `r_hat_g^G` is the three-dimensional unit direction from the current UAV position to the goal expressed in the fixed frame `G`, `d_g^xy` is the horizontal goal distance, `Delta z_g` is the vertical goal displacement, and `v_t^G` is the UAV velocity in `G`. In particular, the lateral component of `r_hat_g^G` retains the signed cross-track direction after the UAV departs from the initial start-to-goal line.

Let `r_(i,t) = p_(i,t)^o-p_t` denote the relative position of dynamic obstacle `i`, and let `d_(i,t)^xy = ||r_(i,t)^xy||_2`. Up to `N_d` obstacles are ordered by increasing horizontal center distance. Candidates outside the sensing range are treated as invalid, and rows not occupied by valid candidates are zero padded. The resulting matrix is `D_t = [d_(1,t), ..., d_(N_d,t)]^T`, with

`d_(i,t) = [r_hat_(i,t)^G, d_(i,t)^xy, Delta z_(i,t), v_(i,t)^(o,G), q_w(w_i), q_h(h_i)] in R^10`,

where `r_hat_(i,t)^G` is the three-dimensional unit relative-position vector expressed in `G`, `Delta z_(i,t)` is the signed vertical displacement, and `v_(i,t)^(o,G)` is the obstacle's absolute velocity rotated into `G`, rather than its velocity relative to the UAV. The width code is `q_w(w_i) = clip(w_i/delta_w-1, 0, 3)`, with `delta_w=0.25 m`. For obstacles with `h_i <= 1.0 m`, the finite height is represented explicitly as `q_h(h_i)=h_i`; taller, vertically spanning obstacles use `Delta z_(i,t)=0` and `q_h(h_i)=0`. Thus, neither size entry should be interpreted as a generally normalized physical dimension. Because the dynamic-obstacle encoder flattens the ordered rows rather than applying permutation-invariant pooling, the distance-based ordering is part of the observation definition. The fixed horizontal direction defining `G` is retained separately for transforming actions into the world frame and is not concatenated into the shared policy feature.

### 3) Action Space

LP-Nav adopts high-level translational velocity control so that the learned policy is decoupled from platform-specific rotor actuation. At each step, the policy selects `a_t^G = [v_x,t^G, v_y,t^G, v_z,t^G]`, where the three components command longitudinal, lateral, and vertical velocity in the fixed navigation frame `G`. The action space is the axis-aligned bounded set

`A = {a in R^3 | -v_lim <= a_j <= v_lim, j in {x,y,z}}`.

Given the shared observation feature, the actor produces two three-dimensional parameter vectors `alpha_t` and `beta_t`. The implementation parameterizes them as `1 + softplus(.)`, ensuring `alpha_t,j > 1` and `beta_t,j > 1` for each action dimension. These parameters define a factorized distribution over the normalized action:

`pi(a_tilde_t | o_t) = product_(j=1)^3 Beta(a_tilde_t,j; alpha_t,j, beta_t,j),  a_tilde_t in (0,1)^3`.

During policy optimization, sampling from this distribution provides bounded stochastic exploration. For deterministic policy execution, the normalized action is its component-wise mean, `a_tilde_t,j = alpha_t,j/(alpha_t,j+beta_t,j)`. The normalized output is mapped to physical velocity through

`a_t^G = v_lim*(2*a_tilde_t-1)`.

This affine mapping enforces the velocity bounds without clipping samples from an unbounded distribution. In the present setting, `v_lim = 2.0 m/s`; the limit applies independently to each axis, so it bounds individual velocity components rather than the Euclidean speed norm.

Finally, the command is transformed to the world frame using the fixed orientation of `G`,

`a_t^W = R_(G->W)*a_t^G`,

and is tracked by a low-level velocity controller that converts the desired translational velocity into vehicle-level control inputs. Yaw is not produced by the policy; attitude and heading regulation are handled outside the learned action space. This separation allows the policy to concentrate on three-dimensional navigation and collision avoidance while the low-level controller handles vehicle stabilization.

### 4) Reward Function

The five principal shaping components considered in LP-Nav are the TTC-aware 3D-VO risk reward, goal-progress reward, smoothness reward, static-safety reward, and height reward. Their weighted combination is

`r_shape_t = lambda_VO*r_VO_t + lambda_g*r_goal_t + lambda_sm*r_sm_t + lambda_ss*r_ss_t + lambda_h*r_h_t`,

where each `lambda` controls the contribution of the corresponding component. Together, these complementary terms promote efficient goal-directed progress, smooth motion, sufficient static-obstacle clearance, bounded vertical behavior, and anticipatory avoidance of dynamic collision risks.

**TTC-aware 3D-VO risk reward.** The finite-horizon risk derived in Sec. A penalizes realized motion that is predicted to enter an inflated obstacle region. It is defined as

`r_VO_t = -eta(n)*rho_VO_(t+1),  eta(n) = min(n/N_warm,1)`,

where `rho_VO_(t+1)` is the maximum TTC-dependent risk over the retained obstacle primitives and `n` is the optimization step. The warmup factor `eta(n)` gradually introduces anticipatory collision-risk shaping while preserving the bounded risk ordering established by the TTC model.

**Goal-progress reward.** Let `d_t = ||p_g-p_t||_2` denote the UAV-to-goal distance. The progress reward is the signed reduction in this distance over the transition:

`r_goal_t = d_t-d_(t+1)`.

Motion toward the goal is rewarded, whereas retreat produces a penalty of equal magnitude. This symmetric definition prevents a closed forward-backward cycle from obtaining positive undiscounted progress reward.

**Smoothness reward.** Abrupt changes in realized velocity are discouraged using

`r_sm_t = -||v_(t+1)-v_t||_2`.

This term favors temporally consistent motion and reduces oscillatory velocity changes during local obstacle avoidance.

**Static-safety reward.** Let `d_(k,t+1)` be the range returned by LiDAR ray `k`, let `N_L` be the number of rays, and let `m_s` be the prescribed static-obstacle safety margin. The static-safety reward is

`r_ss_t = -(1/N_L)*sum_(k=1)^(N_L) max(m_s-d_(k,t+1),0)/m_s`.

Only returns inside the safety margin contribute to the penalty, while geometry beyond `m_s` has no effect. Averaging over the rays keeps the scale independent of LiDAR resolution and encourages the UAV to maintain clearance from nearby static obstacles.

**Height reward.** Excessive vertical deviation from the goal altitude is penalized outside a tolerance band `epsilon_z`:

`r_h_t = -max(|z_(t+1)-z_g|-epsilon_z,0)^2`.

The tolerance band permits necessary altitude adjustments near obstacles, whereas the quadratic growth outside the band discourages the policy from avoiding difficult interactions by flying unnecessarily high or low. Network construction and policy optimization are described in the following section.

## C. Network Design and Policy Training

The heterogeneous observation is processed by modality-specific encoders. A convolutional encoder maps the LiDAR tensor to a `128`-dimensional static-geometry feature, while an MLP maps the ordered dynamic-obstacle matrix to a `64`-dimensional interaction feature. These embeddings are concatenated with the navigation state and fused into a shared representation `z_t in R^256`. The actor maps `z_t` to the parameters of the bounded Beta policy described in Sec. B.3, whereas the critic estimates the corresponding state value.

To make the shared representation sensitive to short-horizon action consequences, LP-Nav adds a lightweight action-conditioned predictive head during training. From `[z_t,a_tilde_t]`, it predicts future collision and stuck events together with minimum forward clearance and accumulated goal progress over multiple short horizons. These predictions define an auxiliary loss that is optimized jointly with the policy objective:

`L = L_PPO + lambda_aux*L_aux`.

Unlike a full generative world model, this head neither reconstructs observations nor performs recursive latent rollouts. Its role is to encourage the encoder to preserve compact predictive information relevant to navigation safety and progress. The head is omitted during policy execution and therefore adds no online inference cost.

The actor and critic are trained using proximal policy optimization (PPO), combining a clipped policy objective, generalized advantage estimation, a clipped value loss, and entropy regularization. Parallel rollout collection from `1,024` environments provides diverse interaction data and supports efficient minibatch updates. Deterministic evaluations use the mean of the learned Beta policy.

Training follows a stage-wise curriculum in which dynamic-obstacle density is increased between stages. A selected checkpoint from each stage initializes the next stage, where reduced learning rates are used for adaptation to the more difficult configuration. This continuation strategy progressively exposes the policy to denser interactions while retaining navigation behaviors acquired at earlier stages.

## D. Deployment-Time Policy Action Safety Shield

Although the learned policy provides the primary navigation command, function approximation, partial observability, and previously unseen interactions can still produce occasional unsafe actions. LP-Nav therefore retains an optional deployment-time safety shield as a final corrective layer. Given the policy velocity `v_RL`, the shield evaluates the command against finite-horizon velocity-obstacle constraints constructed from the current UAV state and local obstacle estimates. If the command is admissible, it is passed through unchanged. Otherwise, the shield returns a nearby feasible velocity `v_safe` that satisfies the active collision-avoidance and control constraints. Its role can be summarized as

`v_cmd = v_RL` if `v_RL in V_safe`, and `v_cmd = arg min_(v in V_safe) ||v-v_RL||_2` otherwise,

where `V_safe` denotes the instantaneous feasible-velocity set. The shield thus intervenes only when a policy command is predicted to be hazardous and otherwise preserves the behavior selected by the learned policy. It is applied after policy inference and is not involved in policy optimization.

The training-time 3D-VO formulation in Sec. A and the deployment-time shield serve complementary purposes. By penalizing the TTC-dependent risk of realized motion during learning, the policy is encouraged to internalize a reflex-like avoidance response based on both obstacle geometry and relative motion. In particular, a rapidly approaching obstacle produces a shorter TTC and therefore a stronger learning signal than a similarly positioned obstacle with weak closing motion. This encourages the policy to react earlier to fast dynamic obstacles, for which waiting until a geometric clearance threshold is violated may leave insufficient time for correction. Consequently, the policy can generate anticipatory avoidance commands directly from its observation, while the shield remains a final safeguard for residual failures rather than the sole source of collision avoidance.

To separate learned avoidance capability from the contribution of online correction, the experimental evaluation reports two configurations: **LP-Nav without the shield**, which measures the intrinsic navigation and safety performance of the trained policy, and **LP-Nav with the shield**, which measures the additional robustness obtained by correcting residual unsafe actions. Reporting both configurations prevents policy improvements produced by TTC-aware training from being conflated with shield intervention and quantifies the safety-performance trade-off introduced by the deployment layer.
