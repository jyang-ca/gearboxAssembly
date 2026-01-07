"""
Evaluation + video recording script for RL-Games policies using Isaac Lab cameras.

This script:
- Uses the SAME RL-Games + IsaacLab integration as training (RlGamesVecEnvWrapper + rlgpu)
- Steps the agent manually (like scripts/rl_games/play.py)
- Reads camera tensors from the underlying Isaac Lab env (like deploy_policy.py)
- Saves an mp4 using torchvision.io (if available) or falls back to imageio

Usage example:
    (isaaclab) python scripts/evaluate_policy_camera_rlgames.py \\
        --task Galaxea-LongTrajectoryAssembly-Direct-v0 \\
        --subtask approach \\
        --checkpoint submission/policy_approach.pth \\
        --num_envs 1 \\
        --max_steps 1000 \\
        --enable_cameras \\
        --rendering_mode quality \\
        --save_video \\
        --video_dir videos
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import torch

from isaaclab.app import AppLauncher

try:
    import torchvision.io as tvio
except Exception:  # noqa: BLE001
    tvio = None

import imageio.v2 as imageio


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RL-Games policy with camera-based video recording")

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
        help="Path to the RL-Games checkpoint (.pth) file.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments for evaluation (1 recommended for video).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of environment steps for evaluation.",
    )

    # Video options
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save camera video to mp4 (first env, first available camera).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory to save evaluation videos.",
    )
    parser.add_argument(
        "--camera_priority",
        type=str,
        default="overhead",
        choices=["overhead", "left_hand", "right_hand", "head"],
        help="Preferred camera to record (overhead=full workspace view, falls back if missing).",
    )

    # IsaacLab / AppLauncher args (headless, enable_cameras, rendering_mode, device, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _pick_first_camera(env_unwrapped, priority: str = "overhead") -> Optional[tuple[str, object]]:
    """Pick a camera sensor with preference ordering."""
    priority_list = {
        "overhead": ["overhead_camera", "left_hand_camera", "right_hand_camera", "head_camera"],
        "left_hand": ["left_hand_camera", "overhead_camera", "head_camera", "right_hand_camera"],
        "right_hand": ["right_hand_camera", "overhead_camera", "head_camera", "left_hand_camera"],
        "head": ["head_camera", "overhead_camera", "left_hand_camera", "right_hand_camera"],
    }.get(priority, ["overhead_camera", "left_hand_camera", "head_camera", "right_hand_camera"])

    # Prefer scene.sensors if available
    if hasattr(env_unwrapped, "scene") and hasattr(env_unwrapped.scene, "sensors"):
        sensors = env_unwrapped.scene.sensors
        if isinstance(sensors, dict) and sensors:
            for cand in priority_list:
                if cand in sensors and hasattr(sensors[cand], "data") and "rgb" in getattr(sensors[cand].data, "output", {}):
                    return cand, sensors[cand]
            # fallback to any camera
            for name, sensor in sensors.items():
                if hasattr(sensor, "data") and "rgb" in getattr(sensor.data, "output", {}):
                    return name, sensor

    # Fallback: scan attributes
    for cand in priority_list:
        if hasattr(env_unwrapped, cand):
            sensor = getattr(env_unwrapped, cand)
            if hasattr(sensor, "data") and "rgb" in getattr(sensor.data, "output", {}):
                return cand, sensor
    return None


def main():
    parser = build_argparser()
    # Use parse_known_args to ignore Kit-specific arguments that might be passed
    args_cli, unknown_args = parser.parse_known_args()

    # If saving video, we must have cameras enabled; enforce it
    if args_cli.save_video and not getattr(args_cli, "enable_cameras", False):
        print("[WARN] --save_video requested but --enable_cameras not set; enabling cameras automatically.", flush=True)
        args_cli.enable_cameras = True

    # [OPTIMIZATION] Inject thread limits and extra settings directly into sys.argv 
    # before AppLauncher consumes it. This prevents the "unrecognized arguments" error.
    sys.argv.append("--/carb/threads/workerCount=2")
    sys.argv.append("--/physics/numThreads=2")
    sys.argv.append("--/renderer/multiGpu/enabled=false")
    # Disable DLSS and heavy post-processing to prevent hangs on low-core machines
    sys.argv.append("--/rtx/post/dlss/enabled=false")
    sys.argv.append("--/rtx/post/aa/op=0") # Disable Anti-Aliasing
    
    # Launch IsaacLab app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    # IsaacLab / RL-Games imports (after app launch)
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import Galaxea_Lab_External.tasks  # noqa: F401

    from rl_games.torch_runner import Runner
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # -------------------------------------------------------------------------
    # 1. Configure environment (same path as training, but lighter)
    # -------------------------------------------------------------------------
    device_str = getattr(args_cli, "device", "cuda:0")
    
    # Match training: use_fabric=True (unless explicitly disabled via --disable_fabric)
    disable_fabric = getattr(args_cli, "disable_fabric", False)
    use_fabric_flag = not disable_fabric
    print(f"[INFO] Using use_fabric={use_fabric_flag} (matching training config).", flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device_str,
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric_flag,
    )

    # [FIX] Synchronize enable_cameras with CLI argument
    should_enable_cameras = args_cli.save_video or getattr(args_cli, "enable_cameras", False)
    if hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = should_enable_cameras
        print(f"[INFO] Setting env_cfg.enable_cameras = {should_enable_cameras}", flush=True)
        
        if should_enable_cameras:
            # Keep all cameras (head + hand) enabled as requested.
            if hasattr(env_cfg, "num_rerenders_on_reset"):
                env_cfg.num_rerenders_on_reset = 0
                print(f"[INFO] Set num_rerenders_on_reset to 0 for speed.", flush=True)

    # Subtask / curriculum
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

    # Underlying Isaac env (NO rgb_array render needed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    env_unwrapped = env.unwrapped

    # Find ALL available cameras for multi-camera recording
    all_cameras = {}
    if args_cli.save_video and hasattr(env_unwrapped, "scene") and hasattr(env_unwrapped.scene, "sensors"):
        sensors = env_unwrapped.scene.sensors
        for cam_name in ["overhead_camera", "head_camera", "left_hand_camera", "right_hand_camera"]:
            if cam_name in sensors:
                sensor = sensors[cam_name]
                if hasattr(sensor, "data") and "rgb" in getattr(sensor.data, "output", {}):
                    all_cameras[cam_name] = sensor
                    print(f"[INFO] Found camera '{cam_name}' for recording.", flush=True)
    
    if args_cli.save_video and not all_cameras:
        print("[WARN] No camera sensors with 'rgb' output found; video will NOT be saved.", flush=True)

    # -------------------------------------------------------------------------
    # 2. Wrap env for RL-Games (same integration as training)
    # -------------------------------------------------------------------------
    rl_device = device_str
    clip_obs = 5.0
    clip_actions = 1.0

    rl_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "IsaacRlgWrapper",
            "env_creator": lambda **kwargs: rl_env,
        },
    )

    # RL-Games config (copied from training config, but minimal)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"LongTrajectoryAssembly_EvalCam_{args_cli.subtask}_{timestamp}"

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

    rl_games_cfg["params"]["config"]["num_actors"] = env_unwrapped.num_envs
    rl_games_cfg["params"]["config"]["device"] = rl_device
    rl_games_cfg["params"]["config"]["device_name"] = rl_device
    # We will load the checkpoint manually into the player model to avoid any
    # potential blocking inside rl_games' internal restore logic.
    rl_games_cfg["params"].pop("load_checkpoint", None)
    rl_games_cfg["params"].pop("load_path", None)

    print(f"\n[INFO] Evaluating Policy (camera-based): {args_cli.checkpoint}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Subtask: {args_cli.subtask}")
    print(f"[INFO] Num Envs: {args_cli.num_envs}")
    print(f"[INFO] Max Steps: {args_cli.max_steps}\n")

    # -------------------------------------------------------------------------
    # 3. Create RL-Games player
    # -------------------------------------------------------------------------
    runner = Runner(IsaacAlgoObserver())
    runner.load(rl_games_cfg)
    agent = runner.create_player()

    # Manually load checkpoint into the agent's model
    print(f"[INFO] Manually loading checkpoint: {args_cli.checkpoint}")
    # PyTorch 2.6+ defaults to weights_only=True, which can break older checkpoints.
    # We explicitly set weights_only=False here (trusted checkpoint from this project).
    checkpoint = torch.load(args_cli.checkpoint, map_location=rl_device, weights_only=False)
    if "model" in checkpoint:
        agent.model.load_state_dict(checkpoint["model"], strict=False)
    else:
        agent.model.load_state_dict(checkpoint, strict=False)

    # Also restore running mean/std for observation normalizer if present.
    # Without this, the policy can output near-zero actions and appear frozen.
    rms_keys = [
        "running_mean_std",
        "obs_running_mean_std",
        "obs_running_std",
        "running_std",
    ]
    for k in rms_keys:
        if k in checkpoint and hasattr(agent.model, "running_mean_std"):
            try:
                agent.model.running_mean_std.load_state_dict(checkpoint[k])
                print(f"[INFO] Restored running_mean_std from checkpoint key '{k}'.", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to load running_mean_std from key '{k}': {e}", flush=True)

    print(f"[INFO] Checkpoint loaded successfully.", flush=True)
 
    # -------------------------------------------------------------------------
    # 4. Evaluation loop (manual, like scripts/rl_games/play.py)
    # -------------------------------------------------------------------------
    # [WARM-UP] Step the simulation app a few times before resetting to "wake up" the renderer
    print(f"[INFO] Warming up the simulator...", flush=True)
    for _ in range(10):
        simulation_app.update()
    
    # Reset RL env
    print(f"[INFO] Resetting environment...", flush=True)
    obs = rl_env.reset()
    print(f"[INFO] Environment reset successful.", flush=True)
 
    # Set up multi-camera video recording
    camera_frames = {cam_name: [] for cam_name in all_cameras.keys()}
    camera_video_paths = {}
    fps = 20
    if args_cli.save_video and all_cameras:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = Path(args_cli.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        for cam_name in all_cameras.keys():
            cam_video_path = video_dir / f"rl_eval_{args_cli.subtask}_{cam_name}_{run_ts}.mp4"
            camera_video_paths[cam_name] = cam_video_path
            print(f"[INFO] Will save '{cam_name}' video to: {cam_video_path}", flush=True)
 
    print("[INFO] Starting evaluation loop with manual stepping...", flush=True)
    start_time = time.time()

    # Track end-effector positions for both arms
    ee_trajectory = []  # List of dicts: {step, left_ee_pos, right_ee_pos, left_ee_delta, right_ee_delta}
    prev_left_ee = None
    prev_right_ee = None

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            # RL-Games utility to convert obs to torch on correct device
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=True)

            # Ensure actions have shape (num_envs, action_dim)
            if torch.is_tensor(actions) and actions.dim() == 1:
                actions = actions.unsqueeze(0)

            # [DEBUG] Log action and joint position statistics for first 5 steps
            if step < 5:
                action_norm = torch.norm(actions).item()
                action_mean = actions.mean().item()
                action_std = actions.std().item()
                action_min = actions.min().item()
                action_max = actions.max().item()
                print(f"[ACTION DEBUG] step {step}: norm={action_norm:.4f}, mean={action_mean:.4f}, "
                      f"std={action_std:.4f}, min={action_min:.4f}, max={action_max:.4f}", flush=True)
                
                # Log actual joint positions to verify robot is moving
                joint_pos = env_unwrapped.robot.data.joint_pos[0, :14]  # First env, first 14 joints
                print(f"[JOINT DEBUG] step {step}: pos_norm={torch.norm(joint_pos).item():.4f}, "
                      f"mean={joint_pos.mean().item():.4f}, min={joint_pos.min().item():.4f}, "
                      f"max={joint_pos.max().item():.4f}", flush=True)
            
            # Track end-effector positions for both arms (every step)
            try:
                # Get EE positions from robot data (first env only)
                # Use realsense links as they are at the hand/gripper position
                left_ee_idx = env_unwrapped.robot.find_bodies("left_realsense_link")[0][0]
                right_ee_idx = env_unwrapped.robot.find_bodies("right_realsense_link")[0][0]
                left_ee_pos = env_unwrapped.robot.data.body_pos_w[0, left_ee_idx]
                right_ee_pos = env_unwrapped.robot.data.body_pos_w[0, right_ee_idx]
                
                left_ee_pos_np = left_ee_pos.cpu().numpy()
                right_ee_pos_np = right_ee_pos.cpu().numpy()
                
                # Calculate movement delta from previous step
                left_delta = 0.0 if prev_left_ee is None else float(np.linalg.norm(left_ee_pos_np - prev_left_ee))
                right_delta = 0.0 if prev_right_ee is None else float(np.linalg.norm(right_ee_pos_np - prev_right_ee))
                
                ee_trajectory.append({
                    "step": step,
                    "left_x": float(left_ee_pos_np[0]),
                    "left_y": float(left_ee_pos_np[1]),
                    "left_z": float(left_ee_pos_np[2]),
                    "right_x": float(right_ee_pos_np[0]),
                    "right_y": float(right_ee_pos_np[1]),
                    "right_z": float(right_ee_pos_np[2]),
                    "left_delta": left_delta,
                    "right_delta": right_delta,
                })
                
                prev_left_ee = left_ee_pos_np
                prev_right_ee = right_ee_pos_np
                
                # Log first 5 steps and every 50 steps
                if step < 5 or step % 50 == 0:
                    print(f"[EE POS] step {step}: "
                          f"left=({left_ee_pos_np[0]:.4f}, {left_ee_pos_np[1]:.4f}, {left_ee_pos_np[2]:.4f}) Δ={left_delta:.4f}m, "
                          f"right=({right_ee_pos_np[0]:.4f}, {right_ee_pos_np[1]:.4f}, {right_ee_pos_np[2]:.4f}) Δ={right_delta:.4f}m",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                if step == 0:
                    print(f"[WARN] Failed to get EE positions: {e}", flush=True)

            # Step RL env (which steps underlying Isaac env)
            obs, _, dones, _ = rl_env.step(actions)
            
            # [FIX] Force rendering update after physics step to sync visual state
            # This ensures cameras capture the current physics state, not stale visuals
            simulation_app.update()
 
            # Record frames from ALL cameras if requested
            if args_cli.save_video and all_cameras:
                for cam_name, sensor in all_cameras.items():
                    try:
                        rgb = sensor.data.output["rgb"]  # (num_envs, H, W, 3)
                        if rgb is None:
                            if step == 0:
                                print(f"[WARN] Camera '{cam_name}' returned None for RGB data.", flush=True)
                            continue
                        
                        frame = rgb[0]  # first env
                        # Handle both torch.Tensor (GPU/CPU) and numpy.ndarray
                        if torch.is_tensor(frame):
                            frame_cpu = frame.detach().cpu().to(torch.uint8)
                        else:
                            frame_cpu = torch.from_numpy(frame).to(torch.uint8)
                        
                        camera_frames[cam_name].append(frame_cpu)
                        
                        if step == 0:
                            # Debug stats to verify visibility
                            f_min = frame_cpu.min().item()
                            f_max = frame_cpu.max().item()
                            f_mean = frame_cpu.float().mean().item()
                            print(
                                f"[DEBUG] First frame from '{cam_name}' captured "
                                f"(shape: {frame_cpu.shape}, type: {type(frame)}, "
                                f"min: {f_min}, max: {f_max}, mean: {f_mean:.2f})",
                                flush=True,
                            )
                            # Save first frame as PNG for quick inspection
                            try:
                                debug_dir = Path(args_cli.video_dir)
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                debug_path = debug_dir / f"debug_first_frame_{cam_name}.png"
                                import imageio.v2 as imageio  # local import
                                imageio.imwrite(debug_path, frame_cpu.numpy())
                                print(f"[DEBUG] Saved first frame snapshot to {debug_path}", flush=True)
                            except Exception as e:  # noqa: BLE001
                                print(f"[WARN] Failed to save debug frame: {e}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        if step == 0:
                            print(f"[WARN] Failed to grab camera frame from '{cam_name}': {e}", flush=True)
 
            # Lightweight progress log every step for the first few steps, then every 10
            if step < 5 or (step + 1) % 10 == 0:
                print(f"[DEBUG] step {step + 1}/{args_cli.max_steps} (Time since start: {time.time() - start_time:.2f}s)", flush=True)

            # If any env is done, we just continue evaluation until max_steps
            if len(dones) > 0 and any(dones):
                # Optional: reset internals for done actors (omitted for brevity)
                pass

    elapsed = time.time() - start_time
    print(f"[INFO] Evaluation finished in {elapsed:.2f}s ({args_cli.max_steps/elapsed:.1f} FPS approx.)")

    # -------------------------------------------------------------------------
    # 5. Save end-effector trajectory to CSV
    # -------------------------------------------------------------------------
    if ee_trajectory:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = Path(args_cli.video_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"ee_trajectory_{args_cli.subtask}_{run_ts}.csv"
        
        try:
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ["step", "left_x", "left_y", "left_z", "right_x", "right_y", "right_z", 
                             "left_delta", "right_delta"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ee_trajectory)
            
            # Calculate total movement statistics
            total_left_movement = sum(entry["left_delta"] for entry in ee_trajectory)
            total_right_movement = sum(entry["right_delta"] for entry in ee_trajectory)
            max_left_delta = max(entry["left_delta"] for entry in ee_trajectory)
            max_right_delta = max(entry["right_delta"] for entry in ee_trajectory)
            
            print(f"[INFO] Saved EE trajectory to: {csv_path}", flush=True)
            print(f"[INFO] Total left arm movement: {total_left_movement:.4f}m (max step: {max_left_delta:.4f}m)", flush=True)
            print(f"[INFO] Total right arm movement: {total_right_movement:.4f}m (max step: {max_right_delta:.4f}m)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to write EE trajectory CSV: {e}", flush=True)

    # -------------------------------------------------------------------------
    # 6. Write videos to disk (one per camera)
    # -------------------------------------------------------------------------
    if args_cli.save_video and camera_video_paths:
        for cam_name, video_path in camera_video_paths.items():
            frames = camera_frames.get(cam_name, [])
            if not frames:
                print(f"[WARN] No frames captured from '{cam_name}'; skipping video save.", flush=True)
                continue
            
            try:
                if tvio is not None:
                    # torchvision expects (T, H, W, C) uint8 tensor
                    video_tensor = torch.stack(frames, dim=0)
                    tvio.write_video(str(video_path), video_tensor, fps=fps, video_codec="h264")
                else:
                    # Fallback: imageio, frames as numpy arrays
                    with imageio.get_writer(str(video_path), fps=fps) as writer:
                        for f in frames:
                            writer.append_data(f.cpu().numpy())
                print(f"[INFO] Saved '{cam_name}' video ({len(frames)} frames) to: {video_path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to write video for '{cam_name}': {e}", flush=True)
    elif args_cli.save_video and not all_cameras:
        print("[WARN] Video saving was requested but no cameras were available; nothing was written.", flush=True)

    rl_env.close()
    simulation_app.close()
    print("[INFO] Process finished. Exiting...", flush=True)
    os._exit(0)


if __name__ == "__main__":
    # Ensure project root on path
    sys.path.append(os.getcwd())
    main()


