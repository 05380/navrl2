import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, IndependentNormal, Actor, GAE, make_batch, IndependentBeta, BetaActor, vec_to_world


class LidarHistoryEncoder(nn.Module):
    def __init__(self, history_len: int, frame_feature_dim: int = 64, output_dim: int = 128, num_heads: int = 4):
        super().__init__()
        if frame_feature_dim % num_heads != 0:
            raise ValueError(
                f"lidar_frame_feature_dim={frame_feature_dim} must be divisible by lidar_temporal_heads={num_heads}."
            )
        self.history_len = history_len
        self.frame_encoder = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(frame_feature_dim), nn.LayerNorm(frame_feature_dim), nn.ELU(),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, frame_feature_dim))
        self.temporal_attention = nn.MultiheadAttention(frame_feature_dim, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(frame_feature_dim)
        self.output = nn.Sequential(
            nn.Linear(frame_feature_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, lidar_history: torch.Tensor):
        if lidar_history.shape[-3] != self.history_len:
            raise ValueError(
                f"Expected lidar history length {self.history_len}, got {lidar_history.shape[-3]}."
            )
        batch_shape = lidar_history.shape[:-3]
        history_len, width, height = lidar_history.shape[-3:]
        lidar_history = lidar_history.reshape(-1, history_len, width, height)

        frame_input = lidar_history.reshape(-1, 1, width, height)
        frame_features = self.frame_encoder(frame_input).reshape(-1, history_len, self.temporal_pos.shape[-1])
        frame_features = frame_features + self.temporal_pos

        query = frame_features[:, -1:, :]
        attended, _ = self.temporal_attention(query, frame_features, frame_features, need_weights=False)
        attended = self.temporal_norm(attended.squeeze(1) + query.squeeze(1))
        pooled = frame_features.mean(dim=1)
        lidar_feature = self.output(torch.cat([attended, pooled], dim=-1))
        return lidar_feature.reshape(*batch_shape, -1)


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.n_agents, self.action_dim = action_spec.shape

        
        # Feature extractor for LiDAR
        lidar_history = max(1, int(cfg.feature_extractor.get("lidar_history", 1)))
        if lidar_history == 1:
            feature_extractor_network = nn.Sequential(
                nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
                nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
                nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
                Rearrange("n c w h -> n (c w h)"),
                nn.LazyLinear(128), nn.LayerNorm(128),
            ).to(self.device)
        else:
            feature_extractor_network = LidarHistoryEncoder(
                history_len=lidar_history,
                frame_feature_dim=int(cfg.feature_extractor.get("lidar_frame_feature_dim", 64)),
                output_dim=128,
                num_heads=int(cfg.feature_extractor.get("lidar_temporal_heads", 4)),
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
        self.auxiliary_goal_progress_weight = float(auxiliary_cfg.get("goal_progress_weight", 1.0))
        self.auxiliary_front_clearance_gain_weight = float(auxiliary_cfg.get("front_clearance_gain_weight", 1.0))
        self.auxiliary_stuck_weight = float(auxiliary_cfg.get("stuck_weight", 1.0))
        self.auxiliary_wall_collision_weight = float(auxiliary_cfg.get("wall_collision_weight", 1.0))
        self.auxiliary_collision_weight = float(auxiliary_cfg.get("collision_weight", 1.0))
        if self.auxiliary_enabled and self.auxiliary_loss_weight > 0.0:
            auxiliary_hidden_dim = int(auxiliary_cfg.get("hidden_dim", 128))
            self.auxiliary_predictor = nn.Sequential(
                nn.Linear(256 + self.action_dim, auxiliary_hidden_dim),
                nn.ELU(),
                nn.LayerNorm(auxiliary_hidden_dim),
                nn.Linear(auxiliary_hidden_dim, 5),
            ).to(self.device)
        else:
            self.auxiliary_predictor = None

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

    @staticmethod
    def _symlog(x):
        return torch.sign(x) * torch.log1p(x.abs())

    def _auxiliary_input(self, tensordict):
        features = tensordict["_feature"]
        action = tensordict[("agents", "action_normalized")]
        action = action.reshape(*features.shape[:-1], -1)
        return torch.cat([features, action], dim=-1)

    def _compute_auxiliary_loss(self, tensordict):
        if self.auxiliary_predictor is None:
            zero = tensordict["_feature"].sum() * 0.0
            return zero, TensorDict({
                "auxiliary_loss": zero.detach(),
                "auxiliary_goal_progress_loss": zero.detach(),
                "auxiliary_front_clearance_gain_loss": zero.detach(),
                "auxiliary_stuck_loss": zero.detach(),
                "auxiliary_wall_collision_loss": zero.detach(),
                "auxiliary_collision_loss": zero.detach(),
            }, [])

        predictions = self.auxiliary_predictor(self._auxiliary_input(tensordict))
        pred_goal_progress = predictions[..., 0:1]
        pred_front_clearance_gain = predictions[..., 1:2]
        pred_stuck = predictions[..., 2:3]
        pred_wall_collision = predictions[..., 3:4]
        pred_collision = predictions[..., 4:5]

        next_stats = tensordict["next", "stats"]
        target_goal_progress = self._symlog(next_stats["goal_progress"].detach())
        target_front_clearance_gain = self._symlog(next_stats["front_clearance_gain"].detach())
        target_stuck = next_stats["stuck_active"].detach().clamp(0.0, 1.0)
        target_wall_collision = next_stats["wall_collision"].detach().clamp(0.0, 1.0)
        target_collision = next_stats["collision"].detach().clamp(0.0, 1.0)

        goal_progress_loss = F.smooth_l1_loss(pred_goal_progress, target_goal_progress)
        front_clearance_gain_loss = F.smooth_l1_loss(pred_front_clearance_gain, target_front_clearance_gain)
        stuck_loss = F.binary_cross_entropy_with_logits(pred_stuck, target_stuck)
        wall_collision_loss = F.binary_cross_entropy_with_logits(pred_wall_collision, target_wall_collision)
        collision_loss = F.binary_cross_entropy_with_logits(pred_collision, target_collision)

        raw_auxiliary_loss = (
            self.auxiliary_goal_progress_weight * goal_progress_loss
            + self.auxiliary_front_clearance_gain_weight * front_clearance_gain_loss
            + self.auxiliary_stuck_weight * stuck_loss
            + self.auxiliary_wall_collision_weight * wall_collision_loss
            + self.auxiliary_collision_weight * collision_loss
        )
        auxiliary_loss = self.auxiliary_loss_weight * raw_auxiliary_loss

        return auxiliary_loss, TensorDict({
            "auxiliary_loss": auxiliary_loss.detach(),
            "auxiliary_goal_progress_loss": goal_progress_loss.detach(),
            "auxiliary_front_clearance_gain_loss": front_clearance_gain_loss.detach(),
            "auxiliary_stuck_loss": stuck_loss.detach(),
            "auxiliary_wall_collision_loss": wall_collision_loss.detach(),
            "auxiliary_collision_loss": collision_loss.detach(),
        }, [])

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            next_values = self.critic(next_tensordict)["state_value"]
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
