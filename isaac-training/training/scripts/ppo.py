import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world

class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.n_agents, self.action_dim = action_spec.shape
        self.state_dim = int(observation_spec[("agents", "observation", "state")].shape[-1])
        self.static_latent_dim = 128 + self.state_dim

        
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        auxiliary_cfg = cfg.feature_extractor.get("auxiliary", {})
        self.auxiliary_enabled = bool(auxiliary_cfg.get("enabled", False))
        self.auxiliary_loss_weight = float(auxiliary_cfg.get("loss_weight", 0.0))
        self.auxiliary_future_horizons = []
        for horizon in auxiliary_cfg.get("future_horizons", [1]):
            horizon = int(horizon)
            if horizon >= 1 and horizon not in self.auxiliary_future_horizons:
                self.auxiliary_future_horizons.append(horizon)
        if self.auxiliary_enabled and not self.auxiliary_future_horizons:
            self.auxiliary_future_horizons = [1]
        self.auxiliary_future_collision_weight = float(auxiliary_cfg.get("future_collision_weight", 1.0))
        self.auxiliary_future_stuck_weight = float(auxiliary_cfg.get("future_stuck_weight", 1.0))
        self.auxiliary_future_clearance_weight = float(auxiliary_cfg.get("future_clearance_weight", 1.0))
        self.auxiliary_future_progress_weight = float(auxiliary_cfg.get("future_progress_weight", 1.0))
        self.latent_dynamics_enabled = bool(auxiliary_cfg.get("latent_dynamics_enabled", False))
        self.latent_dynamics_weight = float(auxiliary_cfg.get("latent_dynamics_weight", 0.0))
        self.future_risk_enabled = bool(auxiliary_cfg.get("future_risk_enabled", False))
        self.future_risk_weight = float(auxiliary_cfg.get("future_risk_weight", 0.0))
        self.future_risk_collision_horizons = []
        for horizon in auxiliary_cfg.get("future_risk_collision_horizons", [3, 5]):
            horizon = int(horizon)
            if horizon >= 1 and horizon not in self.future_risk_collision_horizons:
                self.future_risk_collision_horizons.append(horizon)
        self.future_risk_stuck_horizon = max(1, int(auxiliary_cfg.get("future_risk_stuck_horizon", 5)))
        self.future_risk_deadlock_horizon = max(1, int(auxiliary_cfg.get("future_risk_deadlock_horizon", 10)))
        self.future_risk_output_dim = len(self.future_risk_collision_horizons) + 2
        self.auxiliary_output_dim = 4 * len(self.auxiliary_future_horizons)
        if self.auxiliary_enabled and self.auxiliary_loss_weight > 0.0:
            auxiliary_hidden_dim = int(auxiliary_cfg.get("hidden_dim", 128))
            self.auxiliary_predictor = nn.Sequential(
                nn.Linear(256 + self.action_dim, auxiliary_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(auxiliary_hidden_dim),
                nn.Linear(auxiliary_hidden_dim, self.auxiliary_output_dim),
            ).to(self.device)
        else:
            self.auxiliary_predictor = None
        if (
            self.auxiliary_enabled
            and self.future_risk_enabled
            and self.future_risk_weight > 0.0
            and self.future_risk_output_dim > 0
        ):
            future_risk_hidden_dim = int(auxiliary_cfg.get("future_risk_hidden_dim", 128))
            self.future_risk_predictor = nn.Sequential(
                nn.Linear(self.static_latent_dim + self.action_dim, future_risk_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(future_risk_hidden_dim),
                nn.Linear(future_risk_hidden_dim, self.future_risk_output_dim),
            ).to(self.device)
        else:
            self.future_risk_predictor = None
        if self.auxiliary_enabled and self.latent_dynamics_enabled and self.latent_dynamics_weight > 0.0:
            latent_dynamics_hidden_dim = int(auxiliary_cfg.get("latent_dynamics_hidden_dim", 256))
            self.latent_dynamics_state = nn.Sequential(
                nn.Linear(self.static_latent_dim, latent_dynamics_hidden_dim),
                nn.Tanh(),
            ).to(self.device)
            self.latent_dynamics_cell = nn.GRUCell(
                self.static_latent_dim + self.action_dim,
                latent_dynamics_hidden_dim,
            ).to(self.device)
            self.latent_dynamics_head = nn.Sequential(
                nn.Linear(latent_dynamics_hidden_dim, self.static_latent_dim),
                nn.LayerNorm(self.static_latent_dim),
            ).to(self.device)
        else:
            self.latent_dynamics_state = None
            self.latent_dynamics_cell = None
            self.latent_dynamics_head = None

        # Actor etwork
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        feature_extractor_params = list(self.feature_extractor.parameters())
        if self.auxiliary_predictor is not None:
            feature_extractor_params += list(self.auxiliary_predictor.parameters())
        if self.future_risk_predictor is not None:
            feature_extractor_params += list(self.future_risk_predictor.parameters())
        if self.latent_dynamics_cell is not None:
            feature_extractor_params += list(self.latent_dynamics_state.parameters())
            feature_extractor_params += list(self.latent_dynamics_cell.parameters())
            feature_extractor_params += list(self.latent_dynamics_head.parameters())
        self.feature_extractor_optim = torch.optim.Adam(feature_extractor_params, lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)
        if self.auxiliary_predictor is not None:
            self._init_auxiliary_predictor()
        if self.future_risk_predictor is not None:
            self._init_future_risk_predictor()
        if self.latent_dynamics_cell is not None:
            self._init_latent_dynamics()

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def _init_auxiliary_predictor(self):
        for module in self.auxiliary_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
        output_layer = self.auxiliary_predictor[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _init_future_risk_predictor(self):
        for module in self.future_risk_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
        output_layer = self.future_risk_predictor[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def _init_latent_dynamics(self):
        for module in self.latent_dynamics_state.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)
        for name, param in self.latent_dynamics_cell.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, 0.1)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for module in self.latent_dynamics_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.1)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _symlog(x):
        return torch.sign(x) * torch.log1p(x.abs())

    def _auxiliary_input(self, tensordict):
        features = tensordict["_feature"]
        action = tensordict[("agents", "action_normalized")]
        action = action.reshape(*features.shape[:-1], -1)
        return torch.cat([features, action], dim=-1)

    def _static_latent(self, tensordict):
        lidar_feature = tensordict["_cnn_feature"]
        state = tensordict[("agents", "observation", "state")].reshape(*lidar_feature.shape[:-1], -1)
        return torch.cat([lidar_feature, state], dim=-1)

    def _future_risk_input(self, tensordict):
        static_latent = self._static_latent(tensordict)
        action = tensordict[("agents", "action_normalized")]
        action = action.reshape(*static_latent.shape[:-1], -1)
        return torch.cat([static_latent, action], dim=-1)

    def _zero_auxiliary_info(self, tensordict):
        zero = tensordict["_feature"].sum() * 0.0
        return zero, TensorDict({
            "auxiliary_loss": zero.detach(),
            "auxiliary_future_progress_loss": zero.detach(),
            "auxiliary_future_clearance_loss": zero.detach(),
            "auxiliary_future_collision_loss": zero.detach(),
            "auxiliary_future_stuck_loss": zero.detach(),
            "auxiliary_latent_dynamics_loss": zero.detach(),
            "auxiliary_future_risk_loss": zero.detach(),
            "auxiliary_risk_collision_3_loss": zero.detach(),
            "auxiliary_risk_collision_5_loss": zero.detach(),
            "auxiliary_risk_stuck_5_loss": zero.detach(),
            "auxiliary_risk_deadlock_10_loss": zero.detach(),
        }, [])

    def _future_within_horizon(self, signal, done, horizon):
        target = torch.zeros_like(signal, dtype=torch.bool)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        signal_bool = signal >= 0.5
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.zeros_like(signal_bool)
            if offset == 0:
                shifted_signal = signal_bool
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal_bool[:, offset:]
            target = target | (shifted_signal & valid_for_offset)
        return target.float()

    def _multi_step_sum(self, signal, done, horizon):
        target = torch.zeros_like(signal)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.zeros_like(signal)
            if offset == 0:
                shifted_signal = signal
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal[:, offset:]
            target = target + torch.where(valid_for_offset, shifted_signal, torch.zeros_like(shifted_signal))
        return target

    def _multi_step_min(self, signal, done, horizon):
        fill_value = signal.detach().max()
        target = torch.full_like(signal, fill_value)
        valid_for_offset = torch.ones_like(done, dtype=torch.bool)
        done_bool = done.bool()

        for offset in range(horizon):
            shifted_signal = torch.full_like(signal, fill_value)
            if offset == 0:
                shifted_signal = signal
            else:
                prev_done = torch.ones_like(done_bool)
                prev_done[:, :-offset] = done_bool[:, offset - 1:-1]
                valid_for_offset = valid_for_offset & (~prev_done)
                shifted_signal[:, :-offset] = signal[:, offset:]
            target = torch.minimum(target, torch.where(valid_for_offset, shifted_signal, torch.full_like(signal, fill_value)))
        return target

    def _add_multi_step_auxiliary_targets(self, tensordict):
        if (
            (self.auxiliary_predictor is None or not self.auxiliary_future_horizons)
            and self.future_risk_predictor is None
        ):
            return

        next_stats = tensordict["next", "stats"]
        stuck = next_stats["stuck_active"].detach().squeeze(-1)
        done = tensordict["next", "done"].detach().squeeze(-1)

        if self.auxiliary_predictor is not None and self.auxiliary_future_horizons:
            collision = next_stats["collision"].detach().squeeze(-1)
            front_clearance = next_stats["front_clearance"].detach().squeeze(-1)
            goal_progress = next_stats["goal_progress"].detach().squeeze(-1)

            future_collision = []
            future_stuck = []
            future_clearance = []
            future_progress = []
            for horizon in self.auxiliary_future_horizons:
                future_collision.append(self._future_within_horizon(collision, done, horizon))
                future_stuck.append(self._future_within_horizon(stuck, done, horizon))
                future_clearance.append(self._multi_step_min(front_clearance, done, horizon))
                future_progress.append(self._multi_step_sum(goal_progress, done, horizon))

            tensordict.set("_aux_future_collision", torch.stack(future_collision, dim=-1))
            tensordict.set("_aux_future_stuck", torch.stack(future_stuck, dim=-1))
            tensordict.set("_aux_future_clearance", torch.stack(future_clearance, dim=-1))
            tensordict.set("_aux_future_progress", torch.stack(future_progress, dim=-1))

        if self.future_risk_predictor is not None:
            wall_collision = next_stats["wall_collision"].detach().squeeze(-1)
            future_risk_targets = [
                self._future_within_horizon(wall_collision, done, horizon)
                for horizon in self.future_risk_collision_horizons
            ]
            future_risk_targets.append(self._future_within_horizon(stuck, done, self.future_risk_stuck_horizon))
            future_risk_targets.append(self._future_within_horizon(stuck, done, self.future_risk_deadlock_horizon))
            tensordict.set("_aux_future_risk", torch.stack(future_risk_targets, dim=-1))

    def _compute_auxiliary_loss(self, tensordict):
        if (
            self.auxiliary_predictor is None
            and self.future_risk_predictor is None
            and self.latent_dynamics_cell is None
        ):
            return self._zero_auxiliary_info(tensordict)

        zero = tensordict["_feature"].sum() * 0.0
        future_collision_loss = zero
        future_stuck_loss = zero
        future_clearance_loss = zero
        future_progress_loss = zero
        future_risk_loss = zero
        risk_collision_3_loss = zero
        risk_collision_5_loss = zero
        risk_stuck_5_loss = zero
        risk_deadlock_10_loss = zero

        if self.auxiliary_predictor is not None:
            predictions = self.auxiliary_predictor(self._auxiliary_input(tensordict))
            future_predictions = predictions.reshape(
                *predictions.shape[:-1],
                len(self.auxiliary_future_horizons),
                4,
            )
            pred_future_collision = future_predictions[..., 0]
            pred_future_stuck = future_predictions[..., 1]
            pred_future_clearance = future_predictions[..., 2]
            pred_future_progress = future_predictions[..., 3]

            target_future_collision = tensordict["_aux_future_collision"].detach().clamp(0.0, 1.0)
            target_future_stuck = tensordict["_aux_future_stuck"].detach().clamp(0.0, 1.0)
            target_future_clearance = self._symlog(tensordict["_aux_future_clearance"].detach())
            target_future_progress = self._symlog(tensordict["_aux_future_progress"].detach())

            future_collision_loss = F.binary_cross_entropy_with_logits(
                pred_future_collision,
                target_future_collision,
            )
            future_stuck_loss = F.binary_cross_entropy_with_logits(
                pred_future_stuck,
                target_future_stuck,
            )
            future_clearance_loss = F.smooth_l1_loss(pred_future_clearance, target_future_clearance)
            future_progress_loss = F.smooth_l1_loss(pred_future_progress, target_future_progress)

        if self.future_risk_predictor is not None:
            pred_future_risk = self.future_risk_predictor(self._future_risk_input(tensordict))
            target_future_risk = tensordict["_aux_future_risk"].detach().clamp(0.0, 1.0)
            risk_component_losses = F.binary_cross_entropy_with_logits(
                pred_future_risk,
                target_future_risk,
                reduction="none",
            )
            future_risk_loss = risk_component_losses.mean()
            risk_collision_3_loss = risk_component_losses[..., 0].mean()
            if risk_component_losses.shape[-1] > 1:
                risk_collision_5_loss = risk_component_losses[..., 1].mean()
            if risk_component_losses.shape[-1] > 2:
                risk_stuck_5_loss = risk_component_losses[..., 2].mean()
            if risk_component_losses.shape[-1] > 3:
                risk_deadlock_10_loss = risk_component_losses[..., 3].mean()

        latent_dynamics_loss = self._compute_latent_dynamics_loss(tensordict)

        raw_auxiliary_loss = (
            self.auxiliary_future_collision_weight * future_collision_loss
            + self.auxiliary_future_stuck_weight * future_stuck_loss
            + self.auxiliary_future_clearance_weight * future_clearance_loss
            + self.auxiliary_future_progress_weight * future_progress_loss
            + self.future_risk_weight * future_risk_loss
            + self.latent_dynamics_weight * latent_dynamics_loss
        )
        auxiliary_loss = self.auxiliary_loss_weight * raw_auxiliary_loss

        return auxiliary_loss, TensorDict({
            "auxiliary_loss": auxiliary_loss.detach(),
            "auxiliary_future_progress_loss": future_progress_loss.detach(),
            "auxiliary_future_clearance_loss": future_clearance_loss.detach(),
            "auxiliary_future_collision_loss": future_collision_loss.detach(),
            "auxiliary_future_stuck_loss": future_stuck_loss.detach(),
            "auxiliary_latent_dynamics_loss": latent_dynamics_loss.detach(),
            "auxiliary_future_risk_loss": future_risk_loss.detach(),
            "auxiliary_risk_collision_3_loss": risk_collision_3_loss.detach(),
            "auxiliary_risk_collision_5_loss": risk_collision_5_loss.detach(),
            "auxiliary_risk_stuck_5_loss": risk_stuck_5_loss.detach(),
            "auxiliary_risk_deadlock_10_loss": risk_deadlock_10_loss.detach(),
        }, [])

    def _compute_latent_dynamics_loss(self, tensordict):
        if self.latent_dynamics_cell is None:
            return tensordict["_feature"].sum() * 0.0
        try:
            target_next_latent = tensordict["_aux_next_static_latent"].detach()
        except KeyError:
            return tensordict["_feature"].sum() * 0.0

        static_latent = self._static_latent(tensordict)
        action = tensordict[("agents", "action_normalized")].reshape(*static_latent.shape[:-1], -1)
        dynamics_input = torch.cat([static_latent, action], dim=-1)
        flat_static_latent = static_latent.reshape(-1, static_latent.shape[-1])
        hidden = self.latent_dynamics_state(flat_static_latent)
        pred_next_latent = self.latent_dynamics_head(
            self.latent_dynamics_cell(dynamics_input.reshape(-1, dynamics_input.shape[-1]), hidden)
        ).reshape_as(static_latent)
        transition_loss = F.smooth_l1_loss(
            pred_next_latent,
            target_next_latent,
            reduction="none",
        ).mean(dim=-1, keepdim=True)

        done = tensordict["next", "done"].detach().bool()
        valid = (~done).to(dtype=transition_loss.dtype)
        return (transition_loss * valid).sum() / valid.sum().clamp_min(1.0)

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            next_values = self.critic(next_tensordict)["state_value"]
            tensordict.set("_aux_next_static_latent", self._static_latent(next_tensordict).detach())
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)
        self._add_multi_step_auxiliary_targets(tensordict)

        # Training
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        infos = torch.stack(infos).to_tensordict()
        
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}    

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)
        auxiliary_loss, auxiliary_info = self._compute_auxiliary_loss(tensordict)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss + auxiliary_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        info = TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var,
        }, [])
        info.update(auxiliary_info)
        return info
