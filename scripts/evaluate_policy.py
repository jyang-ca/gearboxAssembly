
"""
Evaluation Script for Galaxea Gearbox Assembly Policies
"""

import argparse
import torch
import os
import sys
from datetime import datetime
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate Long Trajectory Assembly Policy")
parser.add_argument("--task", type=str, default="Galaxea-LongTrajectoryAssembly-Direct-v0", help="Task name")
parser.add_argument("--subtask", type=str, default="approach", help="Subtask to evaluate")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint (.pth file)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments for evaluation")
parser.add_argument("--headless", action="store_true", default=True, help="Run without GUI")

# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import DirectRLEnvCfg
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations

# Import the environment configuration logic from local source
# Assuming the script is run from the root directory
sys.path.append(os.getcwd())
from source.Galaxea_Lab_External.Galaxea_Lab_External.tasks.direct.long_trajectory_assembly.long_trajectory_assembly_env import LongTrajectoryAssemblyEnv
from source.Galaxea_Lab_External.Galaxea_Lab_External.tasks.direct.long_trajectory_assembly.long_trajectory_assembly_env_cfg import LongTrajectoryAssemblyEnvCfg

def parse_env_cfg(task_name, device, num_envs, use_fabric):
    cfg = LongTrajectoryAssemblyEnvCfg()
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.sim.use_fabric = use_fabric
    return cfg

def main():
    # 1. Configure Environment
    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=True
    )
    
    # Set the subtask to evaluate
    env_cfg.training_subtask = args_cli.subtask
    
    # Set proper start gear based on subtask
    if args_cli.subtask.startswith("transport_gear_"):
        gear_num = int(args_cli.subtask.split("_")[-1])
        env_cfg.curriculum_start_gear_idx = gear_num - 1
    elif args_cli.subtask == "approach":
        env_cfg.curriculum_start_gear_idx = 0
    
    # Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print(f"\n[INFO] Evaluating Policy: {args_cli.checkpoint}")
    print(f"[INFO] Subtask: {args_cli.subtask}")
    print(f"[INFO] Num Envs: {args_cli.num_envs}\n")

    # 2. Configure RL-Games Player
    rl_games_cfg = {
        "params": {
            "seed": 42,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "fixed_sigma": True,
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                    }
                },
                "mlp": {
                    "units": [512, 256, 128],
                    "activation": "elu",
                    "initializer": {"name": "default"},
                }
            },
            "load_checkpoint": True,
            "load_path": args_cli.checkpoint,
            "config": {
                "name": args_cli.task,
                "env_name": "rlgpu",
                "device": "cuda:0",
                "device_name": "cuda:0",
                "num_actors": args_cli.num_envs,
                "normalize_input": True,
                "normalize_value": True,
            }
        }
    }

    # Register env for RL-Games
    from rl_games.common.env_configurations import register_env_creator
    register_env_creator("rlgpu", lambda **kwargs: env)

    # 3. Create Player (Inference Mode)
    runner = Runner()
    runner.load(rl_games_cfg)
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    
    # 4. Run Evaluation Loop
    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    
    total_steps = 1000  # Evaluate for 1000 steps
    success_count = 0
    
    print("[INFO] Starting simulation loop...")
    
    with torch.inference_mode():
        for step in range(total_steps):
            # Get action from policy
            # RL-Games player.get_action returns dictionary or tensor depending on config/version
            # Usually for a2c_continuous it returns tensor of actions
            
            # Note: player.get_action handles observation normalization if configured
            action = player.get_action(obs, is_deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check success (assuming 'success' key exists in info via wrapper, or we infer from reward/termination)
            # In our vectorization, we don't explicitly output 'success' boolean in info for DirectRLEnv consistently unless added.
            # But high reward usually implies success.
            # Let's monitor the average reward.
            
            if step % 100 == 0:
                avg_rew = reward.mean().item()
                print(f"Step {step}/{total_steps} | Avg Reward: {avg_rew:.4f}")

    print("\n[INFO] Evaluation Complete.")
    print("For visual verification, run with --headless=False on a local machine with display.")
    env.close()

if __name__ == "__main__":
    main()
