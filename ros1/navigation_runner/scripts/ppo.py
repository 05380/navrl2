import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world



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
