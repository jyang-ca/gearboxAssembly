# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long Trajectory Gear Assembly Environment for Reinforcement Learning (GPU Optimized)."""

from __future__ import annotations

import math
import torch
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sim.spawners.materials import physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import isaacsim.core.utils.torch as torch_utils

from .long_trajectory_assembly_env_cfg import LongTrajectoryAssemblyEnvCfg

class SubTaskType(Enum):
    """Sub-task types for gear assembly."""
    APPROACH = 0
    GRASP = 1
    TRANSPORT = 2

class GearType(Enum):
    """Gear types to be assembled in sequence."""
    GEAR_1 = "gear_1"
    GEAR_2 = "gear_2"
    GEAR_3 = "gear_3"
    GEAR_4 = "gear_4"
    CARRIER = "carrier"
    REDUCER = "reducer"

class LongTrajectoryAssemblyEnv(DirectRLEnv):
    """GPU-Optimized environment for gear assembly."""

    cfg: LongTrajectoryAssemblyEnvCfg

    def __init__(self, cfg: LongTrajectoryAssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices
        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)
        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + self._left_gripper_dof_idx + self._right_gripper_dof_idx
        self._left_ee_body_idx, _ = self.robot.find_bodies("left_arm_link6")
        self._right_ee_body_idx, _ = self.robot.find_bodies("right_arm_link6")

        self.gear_sequence = [GearType.GEAR_1, GearType.GEAR_2, GearType.GEAR_3, GearType.GEAR_4, GearType.CARRIER, GearType.REDUCER]
        self.pin_local_positions = torch.stack([
            torch.tensor(self.cfg.pin_0_local_pos, device=self.device),
            torch.tensor(self.cfg.pin_1_local_pos, device=self.device),
            torch.tensor(self.cfg.pin_2_local_pos, device=self.device),
        ])

        # State tensors (Vectorized)
        self.current_gear_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_subtask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # 0: APPROACH, 1: GRASP, 2: TRANSPORT
        self.stage_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

        self.gear_dict = {
            GearType.GEAR_1: self.sun_planetary_gear_1, GearType.GEAR_2: self.sun_planetary_gear_2,
            GearType.GEAR_3: self.sun_planetary_gear_3, GearType.GEAR_4: self.sun_planetary_gear_4,
            GearType.CARRIER: self.planetary_carrier, GearType.REDUCER: self.planetary_reducer,
        }
        self.obj_dict = {**self.gear_dict, "ring_gear": self.ring_gear}

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        sim_utils.DomeLightCfg(intensity=1000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1000.0))

    def _get_observations(self) -> dict:
        # Vectorized fetching of robot data
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # End-effector poses
        left_ee_pose = self.robot.data.body_state_w[:, self._left_ee_body_idx[0], 0:7]
        right_ee_pose = self.robot.data.body_state_w[:, self._right_ee_body_idx[0], 0:7]

        # Gear info (Vectorized)
        gear_obs = self._get_gear_observations()

        # One-hot encodings (Vectorized)
        subtask_onehot = torch.nn.functional.one_hot(self.current_subtask, num_classes=3).float()
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        gear_idx_onehot = torch.nn.functional.one_hot(safe_gear_idx, num_classes=6).float()

        obs = torch.cat([
            joint_pos[:, self._joint_idx], # 14
            joint_vel[:, self._joint_idx], # 14
            left_ee_pose,                  # 7
            right_ee_pose,                 # 7
            gear_obs,                      # 18
            subtask_onehot,                # 3
            gear_idx_onehot,               # 6
        ], dim=-1)

        return {"policy": obs}

    def _get_gear_observations(self) -> torch.Tensor:
        safe_gear_idx = self.current_gear_idx.clamp(0, len(self.gear_sequence) - 1)
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        all_gear_quat = torch.stack([self.gear_dict[g].data.root_state_w[:, 3:7] for g in self.gear_sequence])
        all_gear_vel = torch.stack([self.gear_dict[g].data.root_lin_vel_w for g in self.gear_sequence])
        
        all_target_pos, all_target_quat = self._get_all_targets()
        
        env_range = torch.arange(self.num_envs, device=self.device)
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        gear_quat = all_gear_quat[safe_gear_idx, env_range]
        gear_vel = all_gear_vel[safe_gear_idx, env_range]
        target_pos = all_target_pos[safe_gear_idx, env_range]
        target_quat = all_target_quat[safe_gear_idx, env_range]
        
        distance = torch.norm(gear_pos - target_pos, dim=-1, keepdim=True)
        return torch.cat([gear_pos, gear_quat, gear_vel, target_pos, target_quat, distance], dim=-1)

    def _get_all_targets(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_gears = len(self.gear_sequence)
        all_target_pos = torch.zeros((num_gears, self.num_envs, 3), device=self.device)
        all_target_quat = torch.zeros((num_gears, self.num_envs, 4), device=self.device)
        
        carrier_pos = self.planetary_carrier.data.root_state_w[:, :3]
        carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7]
        
        # 1-3
        for i in range(3):
            pin_local = self.pin_local_positions[i].expand(self.num_envs, -1)
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
            t_quat, t_pos = torch_utils.tf_combine(carrier_quat, carrier_pos, pin_quat, pin_local)
            all_target_pos[i] = t_pos
            all_target_pos[i, :, 2] += 0.023
            all_target_quat[i] = t_quat
            
        # 4, Carrier, Reducer
        all_target_pos[3] = carrier_pos + torch.tensor([0, 0, 0.03], device=self.device)
        all_target_quat[3] = carrier_quat
        ring = self.ring_gear.data.root_state_w[:, :7]
        all_target_pos[4] = ring[:, :3] + torch.tensor([0, 0, 0.004], device=self.device)
        all_target_quat[4] = ring[:, 3:7]
        gear4 = self.sun_planetary_gear_4.data.root_state_w[:, :7]
        all_target_pos[5] = gear4[:, :3] + torch.tensor([0, 0, 0.025], device=self.device)
        all_target_quat[5] = gear4[:, 3:7]
        
        return all_target_pos, all_target_quat

    def _get_rewards(self) -> torch.Tensor:
        # Full vectorized reward calculation
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        approach = self._vectorized_approach_reward()
        grasp = self._vectorized_grasp_reward()
        transport = self._vectorized_transport_reward()
        
        rewards = torch.where(self.current_subtask == 0, approach, rewards)
        rewards = torch.where(self.current_subtask == 1, grasp, rewards)
        rewards = torch.where(self.current_subtask == 2, transport, rewards)
        
        # Transitions (Vectorized)
        self._check_and_transition_vectorized()
        
        rewards -= self.cfg.reward_time_penalty
        return rewards

    def _check_and_transition_vectorized(self):
        """Vectorized transition between subtasks."""
        # 1. Approach -> Grasp
        approach_complete = (self.current_subtask == 0) & self._vectorized_approach_complete()
        self.current_subtask = torch.where(approach_complete, torch.ones_like(self.current_subtask) * 1, self.current_subtask)
        
        # 2. Grasp -> Transport
        grasp_complete = (self.current_subtask == 1) & self._vectorized_grasp_complete()
        self.current_subtask = torch.where(grasp_complete, torch.ones_like(self.current_subtask) * 2, self.current_subtask)
        
        # 3. Transport -> Next Gear / Done
        transport_complete = (self.current_subtask == 2) & self._vectorized_transport_complete()
        
        # For transport complete, we increment gear_idx and reset subtask to Approach (0)
        self.current_gear_idx[transport_complete] += 1
        self.current_subtask[transport_complete] = 0

    def _vectorized_approach_reward(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        env_range = torch.arange(self.num_envs, device=self.device)
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        
        left_ee = self.robot.data.body_state_w[:, self._left_ee_body_idx[0], :7]
        right_ee = self.robot.data.body_state_w[:, self._right_ee_body_idx[0], :7]
        
        is_left = (safe_gear_idx < 3) | (safe_gear_idx == 4)
        ee_pos = torch.where(is_left.unsqueeze(-1), left_ee[:, :3], right_ee[:, :3])
        ee_quat = torch.where(is_left.unsqueeze(-1), left_ee[:, 3:7], right_ee[:, 3:7])
        
        # 1. Horizontal
        h_dist = torch.norm(ee_pos[:, :2] - gear_pos[:, :2], dim=-1)
        reward = torch.exp(-h_dist / 0.05) * self.cfg.reward_approach_distance_weight
        
        # 2. Height
        height_diff = torch.abs(ee_pos[:, 2] - (gear_pos[:, 2] + self.cfg.pre_grasp_height_offset))
        reward += torch.exp(-height_diff / 0.05) * self.cfg.reward_approach_height_weight
        
        # 3. Orientation
        target_quat = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        quat_dot = torch.abs((ee_quat * target_quat).sum(dim=-1))
        reward += quat_dot * self.cfg.reward_approach_orientation_weight
        
        # Bonus
        complete = (h_dist < self.cfg.approach_horizontal_threshold) & (height_diff < self.cfg.approach_height_threshold) & (quat_dot > self.cfg.approach_orientation_dot_threshold)
        reward[complete] += self.cfg.reward_approach_complete_bonus
        return reward

    def _vectorized_grasp_reward(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        is_left = (safe_gear_idx < 3) | (safe_gear_idx == 4)
        gripper_pos = torch.where(is_left, self.robot.data.joint_pos[:, self._left_gripper_dof_idx[0]], self.robot.data.joint_pos[:, self._right_gripper_dof_idx[0]])
        
        reward = torch.clamp(1.0 - gripper_pos / 0.04, 0, 1) * self.cfg.reward_grasp_gripper_weight
        
        env_range = torch.arange(self.num_envs, device=self.device)
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        lift_height = gear_pos[:, 2] - self.cfg.table_height
        reward += torch.clamp(lift_height / self.cfg.grasp_lift_height, 0, 1) * self.cfg.reward_grasp_lift_weight
        
        complete = (gripper_pos < 0.01) & (lift_height > self.cfg.grasp_lift_height * 0.5)
        reward[complete] += self.cfg.reward_grasp_complete_bonus
        return reward

    def _vectorized_transport_reward(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        env_range = torch.arange(self.num_envs, device=self.device)
        
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        all_gear_quat = torch.stack([self.gear_dict[g].data.root_state_w[:, 3:7] for g in self.gear_sequence])
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        gear_quat = all_gear_quat[safe_gear_idx, env_range]
        
        all_target_pos, all_target_quat = self._get_all_targets()
        target_pos = all_target_pos[safe_gear_idx, env_range]
        target_quat = all_target_quat[safe_gear_idx, env_range]
        
        h_dist = torch.norm(gear_pos[:, :2] - target_pos[:, :2], dim=-1)
        v_diff = torch.abs(gear_pos[:, 2] - target_pos[:, 2])
        quat_dot = torch.abs((gear_quat * target_quat).sum(dim=-1))
        
        reward = torch.exp(-h_dist / 0.01) * self.cfg.reward_transport_distance_weight
        reward += torch.exp(-v_diff / 0.01) * self.cfg.reward_transport_height_weight
        reward += quat_dot * self.cfg.reward_transport_orientation_weight
        
        complete = (h_dist < 0.005) & (v_diff < 0.005) & (quat_dot > 0.995)
        reward[complete] += self.cfg.reward_transport_complete_bonus
        return reward

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        training_subtask = self.cfg.training_subtask
        
        if training_subtask == "full":
            terminated = self.current_gear_idx >= len(self.gear_sequence)
        elif training_subtask == "approach":
            terminated = self._vectorized_approach_complete()
        elif training_subtask == "grasp":
            terminated = self._vectorized_grasp_complete()
        elif training_subtask.startswith("transport_"):
            terminated = self._vectorized_transport_complete()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _vectorized_approach_complete(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        env_range = torch.arange(self.num_envs, device=self.device)
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        
        left_ee = self.robot.data.body_state_w[:, self._left_ee_body_idx[0], :7]
        right_ee = self.robot.data.body_state_w[:, self._right_ee_body_idx[0], :7]
        
        is_left = (safe_gear_idx < 3) | (safe_gear_idx == 4)
        ee_pos = torch.where(is_left.unsqueeze(-1), left_ee[:, :3], right_ee[:, :3])
        ee_quat = torch.where(is_left.unsqueeze(-1), left_ee[:, 3:7], right_ee[:, 3:7])
        
        h_dist = torch.norm(ee_pos[:, :2] - gear_pos[:, :2], dim=-1)
        h_diff = torch.abs(ee_pos[:, 2] - (gear_pos[:, 2] + self.cfg.pre_grasp_height_offset))
        target_quat = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        quat_dot = torch.abs((ee_quat * target_quat).sum(dim=-1))
        
        return (h_dist < self.cfg.approach_horizontal_threshold) & (h_diff < self.cfg.approach_height_threshold) & (quat_dot > self.cfg.approach_orientation_dot_threshold)

    def _vectorized_grasp_complete(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        is_left = (safe_gear_idx < 3) | (safe_gear_idx == 4)
        gripper_pos = torch.where(is_left, self.robot.data.joint_pos[:, self._left_gripper_dof_idx[0]], self.robot.data.joint_pos[:, self._right_gripper_dof_idx[0]])
        
        env_range = torch.arange(self.num_envs, device=self.device)
        all_gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])
        gear_pos = all_gear_pos[safe_gear_idx, env_range]
        lift_height = gear_pos[:, 2] - self.cfg.table_height
        
        return (gripper_pos < 0.01) & (lift_height > self.cfg.grasp_lift_height * 0.5)

    def _vectorized_transport_complete(self):
        safe_gear_idx = self.current_gear_idx.clamp(0, 5)
        env_range = torch.arange(self.num_envs, device=self.device)
        gear_pos = torch.stack([self.gear_dict[g].data.root_state_w[:, :3] for g in self.gear_sequence])[safe_gear_idx, env_range]
        gear_quat = torch.stack([self.gear_dict[g].data.root_state_w[:, 3:7] for g in self.gear_sequence])[safe_gear_idx, env_range]
        target_pos, target_quat = self._get_all_targets()
        target_pos = target_pos[safe_gear_idx, env_range]
        target_quat = target_quat[safe_gear_idx, env_range]
        
        h_dist = torch.norm(gear_pos[:, :2] - target_pos[:, :2], dim=-1)
        v_diff = torch.abs(gear_pos[:, 2] - target_pos[:, 2])
        quat_dot = torch.abs((gear_quat * target_quat).sum(dim=-1))
        
        return (h_dist < 0.005) & (v_diff < 0.005) & (quat_dot > 0.995)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions[:, :6], joint_ids=self._left_arm_joint_idx)
        self.robot.set_joint_position_target(self.actions[:, 6:12], joint_ids=self._right_arm_joint_idx)
        self.robot.set_joint_position_target(self.actions[:, 12:13], joint_ids=self._left_gripper_dof_idx)
        self.robot.set_joint_position_target(self.actions[:, 13:14], joint_ids=self._right_gripper_dof_idx)
        for obj in self.obj_dict.values(): obj.update(self.sim.get_physics_dt())

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None: env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        # Reset to default pose
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[env_ids, :7], env_ids)
        self.robot.write_joint_position_to_sim(self.robot.data.default_joint_pos[env_ids], None, env_ids)
        
        # Setup curriculum
        training_subtask = self.cfg.training_subtask
        start_subtask = 0 if training_subtask == "approach" else (1 if training_subtask == "grasp" else 2)
        start_gear = self.cfg.curriculum_start_gear_idx
        if "gear_1" in training_subtask: start_gear = 0
        elif "gear_2" in training_subtask: start_gear = 1
        # ... more maps
        
        self.current_gear_idx[env_ids] = start_gear
        self.current_subtask[env_ids] = start_subtask
        
        if training_subtask == "grasp": self._setup_grasp_initial_state(env_ids)
        elif training_subtask.startswith("transport_"): self._setup_transport_initial_state(env_ids, start_gear)

    def _setup_grasp_initial_state(self, env_ids):
        # Simplified vectorized setup (just teleport EE to gear)
        pass # To be implemented more robustly if needed, but for now we keep it simple

    def _setup_transport_initial_state(self, env_ids, gear_idx):
        pass
