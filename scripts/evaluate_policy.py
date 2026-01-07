
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

import imageio.v2 as imageio

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate Long Trajectory Assembly Policy")
parser.add_argument("--task", type=str, default="Galaxea-LongTrajectoryAssembly-Direct-v0", help="Task name")
parser.add_argument("--subtask", type=str, default="approach", help="Subtask to evaluate")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint (.pth file)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments for evaluation")
parser.add_argument(
    "--record_video",
    action="store_true",
    help="Record evaluation as video (disables headless mode and uses rgb_array rendering).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="videos",
    help="Directory to save evaluation videos.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="FPS for recorded evaluation video.",
)

# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# If recording video, make sure we are not in headless mode so rendering works.
if args_cli.record_video:
    args_cli.headless = False
    # Ensure output directory exists
    os.makedirs(args_cli.video_dir, exist_ok=True)

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


class VideoRecorderWrapper(gym.Wrapper):
    """Simple gym wrapper to record RGB frames to an mp4 during evaluation.

    NOTE: This assumes the underlying Isaac Lab env supports `render()` with
    `render_mode='rgb_array'` and returns an HxWx3 uint8 image.
    """

    def __init__(self, env: gym.Env, video_dir: str, subtask: str, num_envs: int, fps: int = 30):
        super().__init__(env)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{subtask}_nenv{num_envs}_{timestamp}.mp4"
        self.video_path = os.path.join(video_dir, filename)
        self.writer = imageio.get_writer(self.video_path, fps=fps)
        self.enabled = True

    def _capture_frame(self):
        if not self.enabled:
            return
        try:
            frame = self.env.render()
            if frame is not None:
                self.writer.append_data(frame)
        except Exception as e:
            # Fail gracefully if rendering is not supported
            print(f"[WARN] Video capture failed, disabling recording: {e}")
            self.enabled = False

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._capture_frame()
        return obs

    def step(self, action):
        result = self.env.step(action)
        self._capture_frame()
        return result

    def close(self):
        try:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.close()
        except Exception:
            pass
        return self.env.close()

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
    
    # Decide render mode: rgb_array for video capture, otherwise default
    render_mode = "rgb_array" if args_cli.record_video else None
    
    # Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Optionally enable video recording
    if args_cli.record_video:
        env = VideoRecorderWrapper(
            env,
            video_dir=args_cli.video_dir,
            subtask=args_cli.subtask,
            num_envs=args_cli.num_envs,
            fps=args_cli.video_fps,
        )
    
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
                },
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
                # minimal reward shaper to satisfy rl_games >=1.7 API
                "reward_shaper": {
                    "scale_value": 1.0,
                    "shift_value": 0.0,
                    "min_val": float("-inf"),
                    "max_val": float("inf"),
                    "log_val": False,
                    "is_torch": True,
                },
            },
        }
    }

    # Register env for RL-Games
    # NOTE: rl_games version in this environment exposes `register` + `configurations`,
    # not `register_env_creator`, so we use that API.
    env_configurations.register(
        "rlgpu",
        {
            "env_creator": lambda **kwargs: env,
            "vecenv_type": "RLGPU",
        },
    )

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
    if args_cli.record_video:
        print(f"[INFO] Evaluation video saved to: {getattr(env, 'video_path', getattr(getattr(env, 'env', None), 'video_path', 'N/A'))}")
    else:
    print("For visual verification, run with --headless=False on a local machine with display.")
    env.close()

if __name__ == "__main__":
    main()
