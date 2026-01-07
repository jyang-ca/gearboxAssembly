# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Long Trajectory Gear Assembly RL environment.

This script trains the 8-policy structure for gear assembly:
- Policy_Approach (shared)
- Policy_Grasp (shared)
- Policy_Transport_Gear1~4, Carrier, Reducer (6 gear-specific)

Usage:
    # Train full assembly sequence
    python scripts/train_long_trajectory_assembly.py --task Galaxea-LongTrajectoryAssembly-Direct-v0

    # Train specific sub-task
    python scripts/train_long_trajectory_assembly.py --task Galaxea-LongTrajectoryAssembly-Direct-v0 --subtask approach
    python scripts/train_long_trajectory_assembly.py --task Galaxea-LongTrajectoryAssembly-Direct-v0 --subtask grasp
    python scripts/train_long_trajectory_assembly.py --task Galaxea-LongTrajectoryAssembly-Direct-v0 --subtask transport_gear_1
"""

import argparse
import sys
import os
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train Long Trajectory Assembly RL Agent")
parser.add_argument(
    "--task",
    type=str,
    default="Galaxea-LongTrajectoryAssembly-Direct-v0",
    help="Name of the task (environment)."
)
parser.add_argument(
    "--subtask",
    type=str,
    default="full",
    choices=["full", "approach", "grasp", 
             "transport_gear_1", "transport_gear_2", "transport_gear_3", "transport_gear_4",
             "transport_carrier", "transport_reducer"],
    help="Specific sub-task to train."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4096,
    help="Number of parallel environments."
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=1000,
    help="Maximum number of training iterations (Reduced due to high throughput)."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from."
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Name for the experiment (used for logging)."
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations."
)
# Weights & Biases arguments
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="Enable Weights and Biases tracking."
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="LongTrajectoryAssembly",
    help="Weights and Biases project name."
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="Weights and Biases entity (team or username)."
)
parser.add_argument(
    "--wandb_name",
    type=str,
    default=None,
    help="Weights and Biases run name."
)
# Note: --headless is added by AppLauncher.add_app_launcher_args()

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the training code follows after app launch."""

import gymnasium as gym
import torch
from datetime import datetime

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# Import our custom tasks
import Galaxea_Lab_External.tasks


def main():
    """Main training function."""
    print("=" * 60)
    print("Long Trajectory Assembly RL Training")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Subtask: {args_cli.subtask}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print(f"W&B Tracking: {args_cli.track}")
    print("=" * 60)
    
    # Initialize Weights & Biases if tracking is enabled
    if args_cli.track:
        try:
            import wandb
            
            wandb_name = args_cli.wandb_name or f"{args_cli.subtask}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=args_cli.wandb_project,
                entity=args_cli.wandb_entity,
                name=wandb_name,
                config={
                    "task": args_cli.task,
                    "subtask": args_cli.subtask,
                    "num_envs": args_cli.num_envs,
                    "max_iterations": args_cli.max_iterations,
                    "seed": args_cli.seed,
                },
                sync_tensorboard=True,
            )
            print(f"[INFO] W&B initialized: project={args_cli.wandb_project}, run={wandb_name}")
        except ImportError:
            print("[WARN] wandb not installed. Install with: pip install wandb")
            args_cli.track = False
        except Exception as e:
            print(f"[WARN] W&B initialization failed: {e}")
            args_cli.track = False

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # Set training sub-task
    env_cfg.training_subtask = args_cli.subtask

    # Set curriculum starting point based on subtask
    if args_cli.subtask.startswith("transport_gear_"):
        gear_num = int(args_cli.subtask.split("_")[-1])
        env_cfg.curriculum_start_gear_idx = gear_num - 1
    elif args_cli.subtask == "transport_carrier":
        env_cfg.curriculum_start_gear_idx = 4
    elif args_cli.subtask == "transport_reducer":
        env_cfg.curriculum_start_gear_idx = 5

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Import RL-Games runner
    try:
        from rl_games.torch_runner import Runner
        from rl_games.common import env_configurations
        from rl_games.common.algo_observer import AlgoObserver
        
        # Create experiment name
        if args_cli.experiment_name:
            experiment_name = args_cli.experiment_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"LongTrajectoryAssembly_{args_cli.subtask}_{timestamp}"

        # Set up RL-Games configuration
        rl_games_cfg = {
            "params": {
                "seed": args_cli.seed,
                "algo": {"name": "a2c_continuous"},
                "model": {"name": "continuous_a2c_logstd"},
                "network": {
                    "name": "actor_critic",
                    "separate": False,
                    "space": {
                        "continuous": {
                            "mu_activation": "None",
                            "sigma_activation": "None",
                            "mu_init": {"name": "default"},
                            "sigma_init": {"name": "const_initializer", "val": 0},
                            "fixed_sigma": True
                        }
                    },
                    "mlp": {
                        "units": [512, 256, 128],
                        "activation": "elu",
                        "d2rl": False,
                        "initializer": {"name": "default"},
                        "regularizer": {"name": "None"}
                    }
                },
                "config": {
                    "name": args_cli.task,
                    "full_experiment_name": experiment_name,
                    "env_name": "rlgpu",
                    "ppo": True,
                    "mixed_precision": False,
                    "normalize_input": True,
                    "normalize_value": True,
                    "value_bootstrap": True,
                    "reward_shaper": {"scale_value": 1.0},
                    "normalize_advantage": True,
                    "gamma": 0.99,
                    "tau": 0.95,
                    "learning_rate": 3e-4,
                    "lr_schedule": "adaptive",
                    "kl_threshold": 0.008,
                    "max_epochs": args_cli.max_iterations,
                    "save_frequency": 100,
                    "print_stats": True,
                    "grad_norm": 1.0,
                    "entropy_coef": 0.001,
                    "truncate_grads": True,
                    "e_clip": 0.2,
                    "clip_value": True,
                    "horizon_length": 32,
                    "minibatch_size": 16384,
                    "mini_epochs": 8,
                    "critic_coef": 2,
                    "bounds_loss_coef": 0.0001,
                }
            }
        }

        # Load checkpoint if specified
        if args_cli.checkpoint:
            rl_games_cfg["params"]["load_checkpoint"] = True
            rl_games_cfg["params"]["load_path"] = args_cli.checkpoint

        print("Starting RL-Games training...")
        
        # Import Isaac Lab RL-Games integration
        try:
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            
            # Wrap environment for RL-Games
            rl_device = "cuda:0"
            clip_obs = 5.0
            clip_actions = 1.0
            env_wrapped = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
            
            # Register environment
            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
            )
            env_configurations.register("rlgpu", {
                "vecenv_type": "IsaacRlgWrapper",
                "env_creator": lambda **kwargs: env_wrapped
            })
            
            # Set number of actors
            rl_games_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
            rl_games_cfg["params"]["config"]["device"] = rl_device
            rl_games_cfg["params"]["config"]["device_name"] = rl_device
            
            # Create and run RL-Games runner
            runner = Runner(IsaacAlgoObserver())
            runner.load(rl_games_cfg)
            runner.reset()
            runner.run({"train": True, "play": False})
            
            print("[INFO] RL-Games training completed!")
            
        except Exception as e:
            import traceback
            print(f"[WARN] RL-Games integration failed: {e}")
            traceback.print_exc()
            print("[INFO] Running quick environment test (50 steps)...")
            run_simple_training_loop(env, args_cli)

    except ImportError as e:
        print(f"[WARN] RL-Games not found: {e}")
        print("[INFO] Running quick environment test (50 steps)...")
        run_simple_training_loop(env, args_cli)

    # Close environment
    env.close()


def run_simple_training_loop(env, args):
    """Run a simple training loop for testing."""
    print("=" * 60)
    print("Running simple training loop (for testing)")
    print("=" * 60)

    # Import wandb if tracking
    wandb = None
    if args.track:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            pass

    # Reset environment
    obs, info = env.reset()

    num_steps = 50  # Quick test - 50 steps only
    total_reward = 0.0
    episode_count = 0
    
    for step in range(num_steps):
        # Sample random actions
        actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        mean_reward = reward.mean().item()
        total_reward += mean_reward

        # Log to W&B every 10 steps
        if wandb and step % 10 == 0:
            wandb.log({
                "step": step,
                "reward/mean": mean_reward,
                "reward/cumulative": total_reward,
                "episode/terminated": terminated.sum().item(),
                "episode/truncated": truncated.sum().item(),
            })

        # Print progress every 10 steps (quick test)
        if step % 10 == 0:
            print(f"Test Step {step}/{num_steps} - Reward: {mean_reward:.4f}")

        # Reset if done
        if terminated.any() or truncated.any():
            episode_count += 1
            obs, info = env.reset()

    print("Training loop completed!")
    
    # Log final stats to W&B
    if wandb:
        wandb.log({
            "final/total_steps": num_steps,
            "final/total_episodes": episode_count,
            "final/avg_reward": total_reward / num_steps,
        })
        wandb.finish()
        print("[INFO] W&B run finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()

