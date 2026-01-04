#!/usr/bin/env python3
"""
Standardized Policy Deployment Script for Competition

This script provides a unified interface for deploying and evaluating different
policy types in the Isaac Lab environment. It automatically handles:
- Control frequency adaptation based on policy requirements
- Camera observation collection and formatting
- Episode execution and statistics
- Video recording (optional)

Usage:
    python deploy_policy.py --policy_type act --checkpoint act/ckpt/policy_best.ckpt --num_episodes 10

Author: Competition Organizers
Date: 2025-12-10
"""

import argparse
import numpy as np
import torch
import time
import os
import sys
import warnings
import io
import atexit
import json
from pathlib import Path
from typing import Optional

try:
    import torchvision.io as tvio
except Exception:
    tvio = None

# Isaac Lab imports (must be before policy imports)
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Standardized Policy Deployment for Competition")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-Agent-Direct-v0", help="Task name")
parser.add_argument("--policy_type", type=str, required=True, choices=['act', 'diffusion', 'bc', 'replay'], 
                    help="Policy type (act/diffusion/bc/replay)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint or data file (for replay)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--save_video", action="store_true", help="Save episode videos")
parser.add_argument("--temporal_agg", action="store_true", default=True, help="Use temporal aggregation (for ACT)")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks

# Import policy wrapper
sys.path.insert(0, str(Path(__file__).parent))
from policy_wrapper import ACTPolicyWrapper, DiffusionPolicyWrapper, BCPolicyWrapper, DataReplayPolicyWrapper


class _Tee(io.TextIOBase):
    """Write every message to multiple streams (used to mirror stdout/stderr to log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(artifact_dir: Path, run_timestamp: str) -> Path:
    """Mirror stdout/stderr to a timestamped log file next to the video outputs."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / f"video_{run_timestamp}.log"
    log_file = open(log_path, "a", buffering=1)
    atexit.register(log_file.close)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_path


def load_replay_actions(data_path: str):
    """
    Load actions from HDF5 file for replay.
    
    Args:
        data_path: Path to HDF5 file
    
    Returns:
        actions: numpy array of shape (N, 14)
    """
    import h5py
    print(f"\n{'='*80}")
    print("Loading replay data...")
    print(f"Data path: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Read all action components
        left_arm = f['/actions/left_arm_action'][:]  # (N, 6)
        right_arm = f['/actions/right_arm_action'][:]  # (N, 6)
        left_gripper = f['/actions/left_gripper_action'][:]  # (N,)
        right_gripper = f['/actions/right_gripper_action'][:]  # (N,)
        
        # Combine into 14-dim actions: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
        actions = np.concatenate([
            left_arm,  # (N, 6)
            right_arm,  # (N, 6)
            left_gripper[:, np.newaxis],  # (N, 1)
            right_gripper[:, np.newaxis],  # (N, 1)
        ], axis=1)  # (N, 14)
        
        print(f"✓ Loaded {len(actions)} actions")
        print(f"  Action shape: {actions.shape}")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"{'='*80}\n")
        
        return actions


def load_normalization_stats(checkpoint_path: str):
    """
    Load normalization statistics from dataset_stats.pkl file.
    
    Args:
        checkpoint_path: Path to checkpoint file (dataset_stats.pkl should be in same directory)
    
    Returns:
        Dictionary containing qpos_mean, qpos_std, action_mean, action_std (or None if file not found)
    """
    import pickle
    # Fix numpy version compatibility for pickle files saved with numpy 2.x
    if 'numpy._core' not in sys.modules:
        try:
            import numpy.core
            import numpy.core.multiarray
            import numpy.core.umath
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            sys.modules['numpy._core.umath'] = numpy.core.umath
        except (ImportError, AttributeError):
            pass
    
    stats_path = os.path.join(os.path.dirname(checkpoint_path), 'dataset_stats.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print(f"✓ Loaded normalization stats from {stats_path}")
        return stats
    else:
        print(f"⚠ Warning: No dataset_stats.pkl found at {stats_path}")
        return None


def reorder_qpos_env_to_model(qpos_env: torch.Tensor) -> torch.Tensor:
    """
    Reorder qpos from environment format to model format.
    
    Environment format: [Left Arm(6), Right Arm(6), Left Gripper(1), Right Gripper(1)] = 6-6-1-1
    Model format: [Left Arm(6), Left Gripper(1), Right Arm(6), Right Gripper(1)] = 6-1-6-1
    
    Args:
        qpos_env: Joint positions in env format, shape (batch_size, 14)
    
    Returns:
        qpos_model: Joint positions in model format, shape (batch_size, 14)
    """
    # qpos_env: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
    # qpos_model: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    left_arm = qpos_env[:, :6]
    right_arm = qpos_env[:, 6:12]
    left_gripper = qpos_env[:, 12:13]
    right_gripper = qpos_env[:, 13:14]
    
    qpos_model = torch.cat([left_arm, left_gripper, right_arm, right_gripper], dim=-1)
    return qpos_model


def reorder_action_model_to_env(action_model: torch.Tensor) -> torch.Tensor:
    """
    Reorder action from model format to environment format.
    
    Model format: [Left Arm(6), Left Gripper(1), Right Arm(6), Right Gripper(1)] = 6-1-6-1
    Environment format: [Left Arm(6), Right Arm(6), Left Gripper(1), Right Gripper(1)] = 6-6-1-1
    
    Args:
        action_model: Actions in model format, shape (batch_size, 14)
    
    Returns:
        action_env: Actions in env format, shape (batch_size, 14)
    """
    # action_model: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    # action_env: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
    left_arm = action_model[:, :6]
    left_gripper = action_model[:, 6:7]
    right_arm = action_model[:, 7:13]
    right_gripper = action_model[:, 13:14]
    
    action_env = torch.cat([left_arm, right_arm, left_gripper, right_gripper], dim=-1)
    return action_env


def load_policy(policy_type: str, checkpoint_path: str, **kwargs):
    """
    Load policy based on type.
    
    Args:
        policy_type: Type of policy ('act', 'diffusion', 'bc', 'replay')
        checkpoint_path: Path to checkpoint file (or data file for replay)
        **kwargs: Additional arguments for policy
    
    Returns:
        PolicyWrapper instance
    """
    if policy_type == 'act':
        return ACTPolicyWrapper(checkpoint_path, temporal_agg=kwargs.get('temporal_agg', True))
    elif policy_type == 'diffusion':
        return DiffusionPolicyWrapper(checkpoint_path)
    elif policy_type == 'bc':
        return BCPolicyWrapper(checkpoint_path)
    elif policy_type == 'replay':
        return DataReplayPolicyWrapper(checkpoint_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def get_observations(env, policy_wrapper, norm_stats=None):
    """
    Get observations from environment in policy format.
    
    Args:
        env: Isaac Lab environment
        policy_wrapper: Policy wrapper instance
        norm_stats: Dictionary containing qpos_mean and qpos_std for normalization (optional)
    
    Returns:
        qpos: Joint positions (batch_size, state_dim) - NORMALIZED and in model format (6-1-6-1)
        images: Camera images (batch_size, num_cameras, 3, H, W)
    """
    device = policy_wrapper.device
    
    # Get joint positions from environment
    # Use environment's joint indices to get the correct 14 DoF
    # Environment format: (left_arm: 6, right_arm: 6, left_gripper: 1, right_gripper: 1) = 6-6-1-1
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    
    # Concatenate joint positions in environment order: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
    left_arm_pos = robot.data.joint_pos[:, env_unwrapped._left_arm_joint_idx]
    right_arm_pos = robot.data.joint_pos[:, env_unwrapped._right_arm_joint_idx]
    left_gripper_pos = robot.data.joint_pos[:, env_unwrapped._left_gripper_dof_idx]
    right_gripper_pos = robot.data.joint_pos[:, env_unwrapped._right_gripper_dof_idx]
    
    # Concatenate in env format: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
    qpos_env = torch.cat([
        left_arm_pos,
        right_arm_pos,
        left_gripper_pos,
        right_gripper_pos
    ], dim=-1).to(device)  # (num_envs, 14)
    
    # Reorder from env format (6-6-1-1) to model format (6-1-6-1)
    qpos_model = reorder_qpos_env_to_model(qpos_env)
    
    # Normalize qpos if stats are provided
    if norm_stats is not None and 'qpos_mean' in norm_stats and 'qpos_std' in norm_stats:
        qpos_mean = torch.from_numpy(norm_stats['qpos_mean']).float().to(device)
        qpos_std = torch.from_numpy(norm_stats['qpos_std']).float().to(device)
        # Ensure shapes match: qpos_mean and qpos_std might be 1D (14,) or 2D (1, 14)
        if qpos_mean.ndim == 1:
            qpos_mean = qpos_mean.unsqueeze(0)  # (1, 14)
        if qpos_std.ndim == 1:
            qpos_std = qpos_std.unsqueeze(0)  # (1, 14)
        qpos = (qpos_model - qpos_mean) / qpos_std
    else:
        qpos = qpos_model
    
    # Get camera images
    camera_images = []
    camera_name_mapping = {
        'head_rgb': 'head_camera',
        'left_hand_rgb': 'left_hand_camera',
        'right_hand_rgb': 'right_hand_camera'
    }
    
    for cam_name in policy_wrapper.camera_names:
        sensor_name = camera_name_mapping.get(cam_name)
        if sensor_name is None:
            raise ValueError(f"Unknown camera name: {cam_name}")
        
        # Try to get camera from scene.sensors (External Task) or directly from env (Agent Task)
        try:
            # External Task: cameras are registered in scene.sensors
            rgb_data = env_unwrapped.scene[sensor_name].data.output["rgb"]  # (num_envs, 240, 320, 3)
        except KeyError:
            # Agent Task: cameras are direct attributes of the environment
            if hasattr(env_unwrapped, sensor_name):
                camera_obj = getattr(env_unwrapped, sensor_name)
                rgb_data = camera_obj.data.output["rgb"]  # (num_envs, 240, 320, 3)
            else:
                raise ValueError(f"Camera '{sensor_name}' not found in scene.sensors or as environment attribute")
        
        # Convert to tensor and rearrange to (num_envs, C, H, W)
        rgb_tensor = rgb_data.clone()  # (num_envs, 240, 320, 3)
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # (num_envs, 3, 240, 320)
        
        # Normalize to [0, 1] if needed
        # Note: Both HDF5 data and runtime camera output are uint8 [0, 255]
        # Training (utils.py): HDF5 uint8 → always /255.0
        # Inference: Runtime camera uint8 → conditional /255.0 (handles both [0, 255] and [0, 1] cases)
        rgb_tensor = rgb_tensor / 255.0
        
        camera_images.append(rgb_tensor)
    
    # Stack all cameras: (num_envs, num_cameras, 3, H, W)
    images = torch.stack(camera_images, dim=1).to(device)
    
    return qpos, images


def run_episode(env, policy_wrapper, episode_idx: int, save_video: bool = False,
                artifact_dir: Optional[Path] = None, video_basename: Optional[str] = None,
                norm_stats: Optional[dict] = None, max_steps: Optional[int] = None):
    """
    Run one episode with the policy.
    
    Args:
        env: Isaac Lab environment
        policy_wrapper: Policy wrapper instance
        episode_idx: Episode index
        save_video: Whether to save video
        artifact_dir: Directory to save videos
        video_basename: Base name for video files
        norm_stats: Normalization statistics dictionary (for qpos normalization)
        max_steps: Maximum number of steps (if None, use full episode). 
                   If set, only first half of episode will be used (e.g., max_steps=100 -> use 50 steps)
    
    Returns:
        episode_reward: Total episode reward
        episode_length: Episode length in steps
        success: Whether episode succeeded
    """
    # Reset policy if it has reset method
    if hasattr(policy_wrapper, 'reset'):
        policy_wrapper.reset()
    
    # Reset environment
    obs, _ = env.reset()
    
    episode_reward = 0.0
    episode_length = 0
    success = False
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}")
    print(f"{'='*60}")
    
    start_time = time.time()

    frames = []
    if save_video and tvio is None:
        warnings.warn("torchvision.io not available, disabling video capture.")
        save_video = False
    if save_video:
        sim_dt = env.unwrapped.cfg.sim_dt
        decimation = env.unwrapped.cfg.decimation
        fps = max(1, int(round(1.0 / (sim_dt * decimation))))
        video_dir = artifact_dir or (Path(__file__).parent / "videos")
        video_dir.mkdir(exist_ok=True)
        base_name = video_basename or "episode"
        video_path = video_dir / f"{base_name}_ep{episode_idx + 1}.mp4"
    
    # Record observations and actions for analysis
    recorded_data = {
        "qpos": [],  # Normalized, model format (6-1-6-1)
        "actions": [],  # Denormalized, model format (6-1-6-1)
        "image_stats": []  # Image statistics (min, max, mean) for verification
    }
    
    # Limit episode to first 50% of steps (if max_steps is provided)
    # Example: if max_steps=100, only use steps 0-49 (50 steps, first half)
    max_steps_limit = max_steps // 2 if max_steps is not None else None
    
    while True:
        # Check if max_steps limit reached (use only first 50% of episode)
        if max_steps_limit is not None and episode_length >= max_steps_limit:
            break
        
        # Get observations in policy format (normalized and in model format 6-1-6-1)
        qpos, images = get_observations(env, policy_wrapper, norm_stats=norm_stats)
        
        # Predict action (model outputs in model format 6-1-6-1, already denormalized by policy_wrapper)
        with torch.no_grad():
            action_model = policy_wrapper.predict(qpos, images)  # (batch_size, 14) in model format 6-1-6-1
        
        # Record observations and actions (first env only)
        qpos_np = qpos[0].detach().cpu().numpy().tolist()  # Normalized, model format (6-1-6-1)
        action_np = action_model[0].detach().cpu().numpy().tolist()  # Denormalized, model format (6-1-6-1)
        images_float = images[0].float()  # Convert to float for statistics
        image_stats_np = {
            "min": float(images_float.min().item()),
            "max": float(images_float.max().item()),
            "mean": float(images_float.mean().item())
        }
        
        recorded_data["qpos"].append(qpos_np)
        recorded_data["actions"].append(action_np)
        recorded_data["image_stats"].append(image_stats_np)
        
        # Reorder action from model format (6-1-6-1) to environment format (6-6-1-1)
        action_env = reorder_action_model_to_env(action_model)
        
        # Convert action to environment device
        action_env = action_env.to(env.unwrapped.device)
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action_env)
        
        # Handle reward (could be int or tensor)
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        # Check if episode is done
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
        
        # Save video frame (use first env, first camera)
        if save_video:
            frame = images[0, 0].detach().clamp(0, 1).mul(255).to(torch.uint8)
            frame = frame.permute(1, 2, 0).cpu()  # (H, W, C)
            frames.append(frame)
    
    elapsed = time.time() - start_time
    
    # Print episode summary
    print(f"\n{'-'*60}")
    print(f"Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}")
    print(f"  - Total Reward: {episode_reward:.2f}")
    print(f"  - Success: {success}")
    print(f"  - FPS: {episode_length/elapsed:.1f}")
    print(f"{'-'*60}")

    # Write video to disk
    if save_video and len(frames) > 0:
        try:
            video_tensor = torch.stack(frames, dim=0)  # (T, H, W, C)
            tvio.write_video(str(video_path), video_tensor, fps=fps, video_codec="h264")
            print(f"✓ Saved video to: {video_path}")
        except Exception as e:
            warnings.warn(f"Failed to write video: {e}")
    
    # Save recorded data to JSON
    json_dir = artifact_dir or (Path(__file__).parent / "videos")
    json_dir.mkdir(exist_ok=True)
    base_name = video_basename or "episode"
    json_path = json_dir / f"{base_name}_ep{episode_idx + 1}_data.json"
    
    # Add metadata
    recorded_data["metadata"] = {
        "episode_idx": episode_idx + 1,
        "episode_length": episode_length,
        "total_reward": float(episode_reward),
        "success": success,
        "qpos_format": "normalized, model format (6-1-6-1): [Left Arm(6), Left Gripper(1), Right Arm(6), Right Gripper(1)]",
        "action_format": "denormalized, model format (6-1-6-1): [Left Arm(6), Left Gripper(1), Right Arm(6), Right Gripper(1)]"
    }
    
    try:
        with open(json_path, 'w') as f:
            json.dump(recorded_data, f, indent=2)
        print(f"✓ Saved observation and action data to: {json_path}")
    except Exception as e:
        warnings.warn(f"Failed to write JSON: {e}")
    
    return episode_reward, episode_length, success


def run_replay_episode(env, replay_actions, episode_idx):
    """
    Run a single episode using replay actions from recorded data.
    
    Args:
        env: Isaac Lab environment
        replay_actions: numpy array of actions, shape (N, 14)
        episode_idx: Episode index
    
    Returns:
        episode_reward: Total reward
        episode_length: Episode length
        success: Success flag
    """
    print(f"\n{'='*80}")
    print(f"Episode {episode_idx + 1} - Replaying recorded actions")
    print(f"{'='*80}")
    
    # Reset environment
    obs, _ = env.reset()
    
    episode_reward = 0
    episode_length = 0
    success = False
    
    start_time = time.time()
    
    # Run episode with replay actions
    for step_idx in range(len(replay_actions)):
        # Get action from replay data
        action_np = replay_actions[step_idx]  # (14,)
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.unwrapped.device)  # (1, 14)
        
        # Print every 50 steps
        if step_idx % 50 == 0:
            print(f"\n[Replay Step {step_idx}/{len(replay_actions)}]")
            print(f"  Left arm:  {action_np[:6]}")
            print(f"  Right arm: {action_np[6:12]}")
            print(f"  Grippers:  L={action_np[12]:.3f}, R={action_np[13]:.3f}")
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Handle reward
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        # Check if episode is done
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
    
    elapsed = time.time() - start_time
    
    # Print episode summary
    print(f"\n{'-'*60}")
    print(f"Replay Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}/{len(replay_actions)}")
    print(f"  - Total Reward: {episode_reward:.2f}")
    print(f"  - Success: {success}")
    print(f"  - FPS: {episode_length/elapsed:.1f}")
    print(f"{'-'*60}")
    
    return episode_reward, episode_length, success


def main():
    """Main deployment function"""
    
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(__file__).parent / "videos"
    log_path = setup_logging(artifact_dir, run_timestamp)

    print("\n" + "="*60)
    print("STANDARDIZED POLICY DEPLOYMENT")
    print("="*60)
    print(f"Logging to: {log_path}")
    print(f"Policy type: {args_cli.policy_type}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Task: {args_cli.task}")
    print(f"Num episodes: {args_cli.num_episodes}")
    print(f"Num environments: {args_cli.num_envs}")
    print("="*60)
    
    # Special handling for replay mode
    if args_cli.policy_type == 'replay':
        # Load replay actions
        replay_actions = load_replay_actions(args_cli.checkpoint)
        policy_wrapper = None
        norm_stats = None
    else:
        # Load policy
        print("\nLoading policy...")
        policy_wrapper = load_policy(
            args_cli.policy_type,
            args_cli.checkpoint,
            temporal_agg=args_cli.temporal_agg
        )
        replay_actions = None
        
        # Load normalization statistics (for ACT and other policies that need qpos normalization)
        print("\nLoading normalization statistics...")
        norm_stats = load_normalization_stats(args_cli.checkpoint)
        if norm_stats is None:
            print("⚠ Warning: Normalization stats not found. QPOS will not be normalized.")
    
    # Create environment with adapted configuration
    print(f"\nInitializing environment: {args_cli.task}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    
    # Adapt control frequency if policy requires specific frequency
    if policy_wrapper is not None:
        required_freq = policy_wrapper.required_control_frequency
        if required_freq is not None:
            # Calculate required decimation: decimation = 1 / (sim_dt * frequency)
            required_decimation = int(1.0 / (env_cfg.sim.dt * required_freq))
            original_decimation = env_cfg.decimation
            env_cfg.decimation = required_decimation
            
            print(f"\nControl frequency adaptation:")
            print(f"  - Policy requires: {required_freq} Hz")
            print(f"  - Original decimation: {original_decimation} (dt={env_cfg.sim.dt * original_decimation:.3f}s)")
            print(f"  - Adapted decimation: {required_decimation} (dt={env_cfg.sim.dt * required_decimation:.3f}s)")
            print(f"  - Actual frequency: {1.0 / (env_cfg.sim.dt * required_decimation):.1f} Hz")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"✓ Environment created with {args_cli.num_envs} instance(s)")
    
    # Verify actual control frequency
    actual_dt = env.unwrapped.cfg.sim_dt * env.unwrapped.cfg.decimation
    actual_freq = 1.0 / actual_dt
    print(f"\nFinal environment configuration:")
    print(f"  - Control frequency: {actual_freq:.1f} Hz (DT={actual_dt:.3f}s)")
    print(f"  - Simulation dt: {env.unwrapped.cfg.sim_dt}s")
    print(f"  - Decimation: {env.unwrapped.cfg.decimation}")
    
    # Get max_episode_steps from environment (if available) to use only first 50%
    # Set USE_HALF_HORIZON = False to use full episode length
    USE_HALF_HORIZON = False  # Set to False to use full horizon
    
    max_episode_steps = None
    if USE_HALF_HORIZON:
        if hasattr(env, 'spec') and env.spec is not None and hasattr(env.spec, 'max_episode_steps'):
            max_episode_steps = env.spec.max_episode_steps
        elif hasattr(env.unwrapped, 'cfg') and hasattr(env.unwrapped.cfg, 'episode_length_s'):
            # Isaac Lab 환경의 경우 episode_length_s를 사용
            sim_dt = env.unwrapped.cfg.sim_dt
            decimation = env.unwrapped.cfg.decimation
            control_dt = sim_dt * decimation
            max_episode_steps = int(env.unwrapped.cfg.episode_length_s / control_dt)
        
        # Use only first 50% of episode (1~50 if max is 100)
        if max_episode_steps is not None:
            print(f"\nEpisode length limiting: Using only first 50% of steps")
            print(f"  - Max episode steps: {max_episode_steps}")
            print(f"  - Limited to: {max_episode_steps // 2} steps")
    else:
        print(f"\nEpisode length: Using full horizon (no limit)")
    
    # Run episodes
    print(f"\nRunning {args_cli.num_episodes} evaluation episodes...")
    print("="*60)
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    try:
        for episode_idx in range(args_cli.num_episodes):
            if policy_wrapper is not None:
                # Normal policy mode
                reward, length, success = run_episode(
                    env, 
                    policy_wrapper,
                    episode_idx,
                    save_video=args_cli.save_video,
                    artifact_dir=artifact_dir,
                    video_basename=f"video_{run_timestamp}",
                    norm_stats=norm_stats,
                    max_steps=max_episode_steps
                )
            else:
                # Replay mode
                reward, length, success = run_replay_episode(
                    env,
                    replay_actions,
                    episode_idx
                )
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            successes.append(success)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Print final statistics
    if len(episode_rewards) > 0:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"Best reward: {np.max(episode_rewards):.2f}")
        print(f"Worst reward: {np.min(episode_rewards):.2f}")
        print("="*60)
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
