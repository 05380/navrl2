import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world


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

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
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
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
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

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict
