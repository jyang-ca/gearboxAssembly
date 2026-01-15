# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from datetime import datetime
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--no_action", action="store_true", default=False, help="Do not apply actions to the robot.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--use_gt", action="store_true", default=True, help="Use ground truth positions instead of vision detection (default: True).")
parser.add_argument("--use_vision", action="store_true", default=False, help="Use vision-based detection instead of ground truth.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

print(f"args_cli: {args_cli}")
print(f"Python path: {sys.path}")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks
from Galaxea_Lab_External.robots.galaxea_rule_policy import GalaxeaRulePolicy


def main():
    """Rule-based actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, use_action=not args_cli.no_action, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_kwargs = {
            "video_folder": os.path.join("videos", "rule_based_agent"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"{timestamp}",
        }
        print("[INFO] Recording videos during execution.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # sample_every_n_steps = max(int(sample_period / env.step_dt), 1)
    print("env type: ", type(env))

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()
    
    # set camera view for better recording
    if args_cli.video:
        env.unwrapped.sim.set_camera_view(eye=[1.5, 0.0, 1.5], target=[0.5, 0, 1])

    # Initialize Vision System (conditionally based on --use_vision flag)
    vision_system = None
    if args_cli.use_vision:
        # Note: Using absolute path import or ensuring python path logic
        from Galaxea_Lab_External.vision.vision_pose_estimator import VisionPoseEstimator
        vision_system = VisionPoseEstimator(env)
        print("[INFO] Vision System Initialized (REAL Vision Mode: Depth + 2D bbox)")
        
        # Set vision_estimator on environment for policy to use
        env.unwrapped.vision_estimator = vision_system
        
        # Warm-up Loop for Vision (10 steps)
        print("[INFO] Warming up vision system...")
        warmup_steps = 10
        
        # Initialize zero action with current joint positions (State Holding)
        current_joint_pos = env.unwrapped.joint_pos.clone() # (num_envs, num_dof)
        warmup_action = current_joint_pos.clone() # Use current pose as hold command
        
        for _ in range(warmup_steps):
            env.step(warmup_action)
            # Warmup vision to populate history buffer
            vision_system.get_3d_poses('front_camera')
    else:
        print("[INFO] Using Ground Truth (GT) positions (default mode)")
    
    # ---------------------------------------------------------
    # Rule Based Policy Initialization
    # ---------------------------------------------------------
    initial_root_state = env.unwrapped.initial_root_state
    sim_context = env.unwrapped.sim
    scene = env.unwrapped.scene
    obj_dict = env.unwrapped.obj_dict
    
    policy = GalaxeaRulePolicy(sim_context, scene, obj_dict, vision_estimator=vision_system)
    

    # Prepare Policy Plan
    policy.set_initial_root_state(initial_root_state)
    policy.prepare_mounting_plan()
    
    # Get joint indices for mapping
    # We need to access private attributes (using _ convention from the env class)
    left_arm_idx = env.unwrapped._left_arm_joint_idx
    right_arm_idx = env.unwrapped._right_arm_joint_idx
    left_gripper_idx = env.unwrapped._left_gripper_dof_idx
    right_gripper_idx = env.unwrapped._right_gripper_dof_idx
    
    decimation = env.unwrapped.cfg.decimation
    print(f"[INFO] Policy Decimation Sync: {decimation} steps per policy tick")

    step_count = 0
    count = 0
    print(f"[INFO] Interaction Loop starting...")
    score_history = []
    
    try:
        while simulation_app.is_running():
            if args_cli.video:
                if step_count % 100 == 0:
                    print(f"step_count: {step_count}")
            
            if args_cli.video and step_count >= args_cli.video_length:
                print(f"[INFO] Video length reached: {args_cli.video_length} steps. Stopping...")
                break


            # Vision Step: Get Detections (Logging only) - only if using vision
            if vision_system is not None:
                detections = vision_system.get_oracle_detections('front_camera')
                if step_count % 50 == 0 and len(detections) > 0:
                     # print(f"[Vision] Step {step_count}: Detected {len(detections)} objects in Front Camera")
                     pass


            # Get Action from Policy
            policy_action, joint_ids = policy.get_action()
            
            # Action Mapping
            # Initialize action with current position to hold state for uncontrolled parts
            # This is crucial for "partial" control (e.g. only moving left arm)
            full_action = env.unwrapped.joint_pos.clone()
            
            if policy_action is not None and joint_ids is not None:
                # We need to map the 'joint_ids' (simulation indices) to 'action indices' (0..13)
                # The env action vector is constructed as:
                # [Left Arm (6) | Right Arm (6) | Left Grip (1) | Right Grip (1)]
                # See GalaxeaLabAgentEnv._joint_idx logic
                
                # We can deduce the mapping by checking which group the joint_ids belong to.
                
                # 1. Left Arm
                if len(joint_ids) == len(left_arm_idx) and torch.equal(torch.tensor(joint_ids, device=env.unwrapped.device), torch.tensor(left_arm_idx, device=env.unwrapped.device)):
                    full_action[:, :6] = policy_action
                
                # 2. Right Arm
                elif len(joint_ids) == len(right_arm_idx) and torch.equal(torch.tensor(joint_ids, device=env.unwrapped.device), torch.tensor(right_arm_idx, device=env.unwrapped.device)):
                    full_action[:, 6:12] = policy_action
                
                # 3. Combined (Both Arms)
                elif len(joint_ids) == len(left_arm_idx) + len(right_arm_idx):
                     full_action[:, :6] = policy_action[:, :6]
                     full_action[:, 6:12] = policy_action[:, 6:12]
                     
                # 4. Grippers
                # Policy returns single value, but we might check if joint_ids matches specific gripper indices
                elif len(joint_ids) == len(left_gripper_idx): # Left Gripper
                    # Broadcasting 1 value to all left gripper joints (if multiple)
                    # Even if 1, we treat it as broadcast logic
                    val = policy_action.view(-1)[0]
                    # Map to Action Index: 12 (Left Grip)
                    full_action[:, 12] = val
                    
                elif len(joint_ids) == len(right_gripper_idx): # Right Gripper
                    # Map to Action Index: 13 (Right Grip)
                    val = policy_action.view(-1)[0]
                    full_action[:, 13] = val
                
                # 5. Recovery Combined (Arm + Gripper)
                # Sometimes recovery returns Arm + Gripper
                # We can check by length or set intersections, but for now simple length check might be ambiguous if arm=6 and gripper=1 -> 7
                elif len(joint_ids) > 6:
                    # Generic mapping by iterating? Too slow.
                    # Assumption: It's Arm + Gripper of same side.
                    
                    # Check intersection with left arm
                    # Convert to sets for easier check? (But tensors on GPU...)
                    # Let's simple check first element
                    first_id = joint_ids[0]
                    if first_id in left_arm_idx:
                        # Left side
                        full_action[:, :6] = policy_action[:, :6] # Arm
                        # Gripper is the rest
                        val = policy_action[:, 6:].view(-1)[0]
                        full_action[:, 12] = val
                    elif first_id in right_arm_idx:
                         # Right side
                        full_action[:, 6:12] = policy_action[:, :6] # Arm
                        val = policy_action[:, 6:].view(-1)[0]
                        full_action[:, 13] = val
                        
            
            # Apply Action
            obs, reward, terminated, truncated, info = env.step(full_action)
            
            # Manually advance policy time
            # Env advances by decimation * sim_dt
            policy.count += decimation
            
            # Update Policy Score
            current_score = int(reward.item()) if isinstance(reward, torch.Tensor) else int(reward)
            policy.set_score(current_score)
            
            # Capture frame for video recording (RecordVideo wrapper needs this)
            if args_cli.video:
                env.render()
            
            # Score Tracking
            if not score_history or score_history[-1] != current_score:
                score_history.append(current_score)
                history_str = " -> ".join(map(str, score_history))
                print(f"Score History: {history_str}")

            sim_dt = env.unwrapped.sim.get_physics_dt()
            if count % int(1.0 / sim_dt) == 0:
                 # print(f"step: {count}, time: {count * sim_dt}")
                 pass

            # Check if the environment is terminated or truncated
            if terminated.any() or truncated.any():
                print(f"[INFO] Episode terminated/truncated.")
                env.reset()
                policy.count = 0 # Reset policy counter
                # break

            count += 1
            step_count += 1
            
    except KeyboardInterrupt:
        print(f"\n[INFO] KeyboardInterrupt detected. Stopping simulation and saving video...")
    
    finally:
        # Close the environment
        env.close()
        print(f"[INFO] Environment closed.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()