"""Evaluation entrypoint: load a checkpoint and run deterministic evaluation once."""

import datetime
import os

import hydra
import torch
import wandb
from omni.isaac.kit import SimulationApp
from torchrl.envs.transforms import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType

from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController
from ppo import PPO
from utils import evaluate


FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def print_eval_metrics(eval_info):
    metric_keys = [
        "eval/success_rate",
        "eval/collision_rate",
        "eval/below_bound_rate",
        "eval/above_bound_rate",
        "eval/deadlock_rate",
        "eval/time_limit_rate",
        "eval/stall_rate",
        "eval/stats.return",
        "eval/stats.episode_len",
        "eval/stats.reward_goal_progress",
        "eval/stats.penalty_safety_static",
        "eval/stats.penalty_safety_dynamic",
        "eval/stats.reward_vel",
        "eval/stats.penalty_height",
        "eval/vo_risk_mean",
    ]
    metric_parts = []
    for key in metric_keys:
        value = eval_info.get(key)
        if value is not None:
            metric_parts.append(f"{key}={value:.4f}")
    if metric_parts:
        print("[NavRL]: eval metrics | " + " | ".join(metric_parts))


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    if cfg.get("checkpoint", None) is None:
        raise ValueError("Evaluation requires `checkpoint=...` to be provided.")

    sim_app = SimulationApp(
        {
            "headless": cfg.headless,
            "anti_aliasing": 1,
            "extra_args": [
                "--/persistent/isaac/asset_root/default=http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"
            ],
        }
    )

    checkpoint_path = os.path.expanduser(str(cfg.checkpoint))
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    run = wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}/eval/{checkpoint_name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
        entity=cfg.wandb.entity,
        config=cfg,
        mode=cfg.wandb.mode,
        id=wandb.util.generate_id(),
    )
    run.define_metric("eval/*")

    from env import NavigationEnv

    env = NavigationEnv(cfg)

    controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
    vel_transform = VelController(controller, yaw_control=False)
    transformed_env = TransformedEnv(env, Compose(vel_transform)).eval()
    transformed_env.set_seed(cfg.seed)

    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    policy.eval()
    print("[NavRL]: loaded checkpoint from:", checkpoint_path)

    eval_info = evaluate(
        env=transformed_env,
        policy=policy,
        seed=cfg.seed,
        cfg=cfg,
        exploration_type=ExplorationType.MEAN,
    )
    eval_info["checkpoint"] = checkpoint_path

    run.log(eval_info)
    print_eval_metrics(eval_info)

    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
