"""
Evaluation script for Galaxea Gearbox Assembly policies using the SAME
RL-Games + IsaacLab integration as training (train_long_trajectory_assembly.py),
with optional video recording.
"""

import argparse
import os
import sys
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

import imageio.v2 as imageio


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Long Trajectory Assembly Policy (RL-Games compatible)")

    parser.add_argument(
        "--task",
        type=str,
        default="Galaxea-LongTrajectoryAssembly-Direct-v0",
        help="Name of the task (environment).",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="approach",
        choices=[
            "full",
            "approach",
            "grasp",
            "transport_gear_1",
            "transport_gear_2",
            "transport_gear_3",
            "transport_gear_4",
            "transport_carrier",
            "transport_reducer",
        ],
        help="Specific sub-task to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint (.pth file) produced by RL-Games training.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments for evaluation (can be smaller than training).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Number of environment steps to run for evaluation.",
    )

    # Video recording options
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

    # NOTE: AppLauncher adds Isaac/Omniverse/IsaacLab specific CLI flags (including --headless, --device, --disable_fabric, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser


class VideoRecorderWrapper:
    """Minimal video recorder wrapper that hooks into the IsaacLab env used by RL-Games.

    We don't subclass gym.Wrapper here to avoid any version-specific issues; we only
    forward the methods we actually use (reset/step/close/render) and capture frames
    after each env step.
    """

    def __init__(self, env, video_dir: str, task: str, subtask: str, num_envs: int, fps: int = 30):
        self.env = env

        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{task}_{subtask}_nenv{num_envs}_{timestamp}.mp4"
        self.video_path = os.path.join(video_dir, filename)

        self.writer = imageio.get_writer(self.video_path, fps=fps)
        self.enabled = True

    # Basic proxy attributes
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _capture_frame(self):
        if not self.enabled:
            return
        try:
            frame = self.env.render()
            if frame is not None:
                self.writer.append_data(frame)
        except Exception as e:  # noqa: BLE001
            # Fail gracefully if rendering is not supported
            print(f"[WARN] Video capture failed, disabling recording: {e}")
            self.enabled = False

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._capture_frame()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._capture_frame()
        return result

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        try:
            if getattr(self, "writer", None) is not None:
                self.writer.close()
        except Exception:
            pass
        return self.env.close()


def main():
    # -------------------------------------------------------------------------
    # 1. Parse arguments and launch IsaacLab app
    # -------------------------------------------------------------------------
    parser = build_argparser()
    args_cli = parser.parse_args()

    # If recording video, make sure we are not in headless mode so rendering works.
    if args_cli.record_video:
        args_cli.headless = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841  # unused but kept to mirror training script

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  # registers tasks with Gym
    from isaaclab_tasks.utils import parse_env_cfg

    # Ensure our custom tasks are registered, same as training
    import Galaxea_Lab_External.tasks  # noqa: F401

    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.common import vecenv
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # -------------------------------------------------------------------------
    # 2. Configure environment (SAME as training path)
    # -------------------------------------------------------------------------
    # Some IsaacLab builds may not expose `disable_fabric` / `device` on the CLI.
    # Fall back to sensible defaults if they are missing.
    use_fabric_flag = True
    if hasattr(args_cli, "disable_fabric"):
        use_fabric_flag = not args_cli.disable_fabric

    device_str = getattr(args_cli, "device", "cuda:0")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device_str,
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric_flag,
    )

    # Training subtask / curriculum, mirror training script logic
    env_cfg.training_subtask = args_cli.subtask
    if args_cli.subtask.startswith("transport_gear_"):
        gear_num = int(args_cli.subtask.split("_")[-1])
        env_cfg.curriculum_start_gear_idx = gear_num - 1
    elif args_cli.subtask == "transport_carrier":
        env_cfg.curriculum_start_gear_idx = 4
    elif args_cli.subtask == "transport_reducer":
        env_cfg.curriculum_start_gear_idx = 5
    elif args_cli.subtask == "approach":
        env_cfg.curriculum_start_gear_idx = 0

    # Decide render mode: rgb_array for video capture, otherwise default
    render_mode = "rgb_array" if args_cli.record_video else None

    # Base IsaacLab env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Optional video recording on the base env (before RL-Games wrappers)
    if args_cli.record_video:
        env = VideoRecorderWrapper(
            env,
            video_dir=args_cli.video_dir,
            task=args_cli.task,
            subtask=args_cli.subtask,
            num_envs=args_cli.num_envs,
            fps=args_cli.video_fps,
        )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # -------------------------------------------------------------------------
    # 3. RL-Games integration (mirrors train_long_trajectory_assembly.py)
    # -------------------------------------------------------------------------
    rl_device = "cuda:0"
    clip_obs = 5.0
    clip_actions = 1.0

    # Wrap env for RL-Games (this defines per-actor observation/action spaces)
    env_wrapped = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # Register RL-Games vecenv + env configuration
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kwargs: env_wrapped,
        },
    )

    # RL-Games config (copied from training config, but used in play mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"LongTrajectoryAssembly_Eval_{args_cli.subtask}_{timestamp}"

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
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [512, 256, 128],
                    "activation": "elu",
                    "d2rl": False,
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"},
                },
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
                # `max_epochs` is irrelevant for play, but kept for completeness
                "max_epochs": 1,
                "save_frequency": 0,
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
            },
        }
    }

    # Attach device + actor count
    rl_games_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    rl_games_cfg["params"]["config"]["device"] = rl_device
    rl_games_cfg["params"]["config"]["device_name"] = rl_device

    # Tell RL-Games to load our checkpoint
    rl_games_cfg["params"]["load_checkpoint"] = True
    rl_games_cfg["params"]["load_path"] = args_cli.checkpoint

    print(f"\n[INFO] Evaluating Policy: {args_cli.checkpoint}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Subtask: {args_cli.subtask}")
    print(f"[INFO] Num Envs: {args_cli.num_envs}")
    print(f"[INFO] Max Steps: {args_cli.max_steps}\n")

    # -------------------------------------------------------------------------
    # 4. Let RL-Games handle play loop (avoids obs dict/tensor mismatches)
    # -------------------------------------------------------------------------
    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_games_cfg)

    # Run in "play" mode; RL-Games will use the loaded checkpoint and interact
    # with the registered env ("rlgpu") using the same integration path as training.
    print("[INFO] Starting RL-Games play loop...")
    runner.run({"train": False, "play": True})

    print("\n[INFO] Evaluation Complete.")
    if args_cli.record_video:
        # The video path is stored on the video wrapper around the base env
        base_env = env
        video_path = getattr(base_env, "video_path", None)
        print(f"[INFO] Evaluation video saved to: {video_path}")
    else:
        print("For visual verification, re-run with --record_video to save an mp4.")

    env_wrapped.close()


if __name__ == "__main__":
    # Ensure project root is in sys.path (same assumption as other scripts)
    sys.path.append(os.getcwd())
    main()


