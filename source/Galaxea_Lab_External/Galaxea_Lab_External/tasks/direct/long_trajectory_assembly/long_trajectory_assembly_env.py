# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long Trajectory Gear Assembly Environment for Reinforcement Learning.

This environment implements a multi-stage gear assembly task with:
- Environment-managed rule-based transitions between sub-tasks
- 8 policy structure (Phase 2):
  - Policy_Approach (shared)
  - Policy_Grasp (shared)
  - Policy_Transport_Gear1~4, Carrier, Reducer (6 gear-specific)
"""

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
from isaaclab.sensors import Camera
import isaaclab.envs.mdp as mdp

import isaacsim.core.utils.torch as torch_utils

from .long_trajectory_assembly_env_cfg import LongTrajectoryAssemblyEnvCfg


class SubTaskType(Enum):
    """Sub-task types for gear assembly."""
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"


class GearType(Enum):
    """Gear types to be assembled in sequence."""
    GEAR_1 = "gear_1"
    GEAR_2 = "gear_2"
    GEAR_3 = "gear_3"
    GEAR_4 = "gear_4"
    CARRIER = "carrier"
    REDUCER = "reducer"


class LongTrajectoryAssemblyEnv(DirectRLEnv):
    """Long trajectory gear assembly environment for RL training.
    
    This environment implements the full gear assembly pipeline with:
    - Multi-stage state management (Approach, Grasp, Transport)
    - Environment-based rule-based transitions
    - Rewards designed for each sub-task
    - Observations tailored for each sub-task
    
    The policy structure follows Phase 2 design:
    - 1 shared Approach policy
    - 1 shared Grasp policy  
    - 6 gear-specific Transport policies
    """

    cfg: LongTrajectoryAssemblyEnvCfg

    def __init__(
        self, 
        cfg: LongTrajectoryAssemblyEnvCfg, 
        render_mode: str | None = None,
        **kwargs
    ):
        """Initialize the long trajectory assembly environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__(cfg, render_mode, **kwargs)

        print("=" * 60)
        print("Long Trajectory Assembly Environment Initialized")
        print("=" * 60)

        # Get joint indices
        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)
        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)
        self._torso_joint1_idx, _ = self.robot.find_joints(self.cfg.torso_joint1_dof_name)
        self._torso_joint2_idx, _ = self.robot.find_joints(self.cfg.torso_joint2_dof_name)
        self._torso_joint3_idx, _ = self.robot.find_joints(self.cfg.torso_joint3_dof_name)

        # Combined joint indices
        self._joint_idx = (
            self._left_arm_joint_idx + 
            self._right_arm_joint_idx + 
            self._left_gripper_dof_idx + 
            self._right_gripper_dof_idx
        )

        # Setup IK controllers for both arms
        self._setup_ik_controllers()

        # Get body indices for end-effector links
        self._left_ee_body_idx, _ = self.robot.find_bodies("left_arm_link6")
        self._right_ee_body_idx, _ = self.robot.find_bodies("right_arm_link6")

        # Gear sequence
        self.gear_sequence = [
            GearType.GEAR_1,
            GearType.GEAR_2,
            GearType.GEAR_3,
            GearType.GEAR_4,
            GearType.CARRIER,
            GearType.REDUCER,
        ]

        # Pin local positions
        self.pin_local_positions = [
            torch.tensor(self.cfg.pin_0_local_pos, device=self.device),
            torch.tensor(self.cfg.pin_1_local_pos, device=self.device),
            torch.tensor(self.cfg.pin_2_local_pos, device=self.device),
        ]

        # Multi-stage state (will be reset in _reset_idx)
        self.current_gear_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_subtask = [SubTaskType.APPROACH] * self.num_envs
        self.stage_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.assembled_gears = [set() for _ in range(self.num_envs)]
        
        # Gear to pin mapping (computed during reset)
        self.gear_to_pin_map = [{} for _ in range(self.num_envs)]

        # Initial arm positions (radians)
        self.initial_pos_left = torch.tensor([
            -20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
            -24.0 / 180.0 * math.pi, 17.8 / 180.0 * math.pi,
            38.7 / 180.0 * math.pi, 20.1 / 180.0 * math.pi
        ], device=self.device)
        
        self.initial_pos_right = torch.tensor([
            -20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
            -22.0 / 180.0 * math.pi, -40.0 / 180.0 * math.pi,
            -67.6 / 180.0 * math.pi, 18.1 / 180.0 * math.pi
        ], device=self.device)

        print(f"Left arm joint indices: {self._left_arm_joint_idx}")
        print(f"Right arm joint indices: {self._right_arm_joint_idx}")
        print(f"Training subtask: {self.cfg.training_subtask}")

    def _setup_ik_controllers(self):
        """Setup differential IK controllers for both arms."""
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", 
            use_relative_mode=False, 
            ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, 
            num_envs=self.num_envs, 
            device=self.device
        )

        # Get body indices for end-effector links (these will be set after scene setup)
        self._left_ee_body_idx = None
        self._right_ee_body_idx = None

    def _setup_scene(self):
        """Setup the simulation scene with robot and objects."""
        print("=" * 60)
        print("Setting up scene...")
        print("=" * 60)

        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Cameras (only create if enabled - not needed for RL training)
        self.head_camera = None
        self.left_hand_camera = None
        self.right_hand_camera = None
        
        if getattr(self.cfg, 'enable_cameras', False):
            self.head_camera = Camera(self.cfg.head_camera_cfg)
            self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
            self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

        # Table
        self.table = RigidObject(self.cfg.table_cfg)

        # Gears
        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

        # Gear dictionary for easy access
        self.gear_dict = {
            GearType.GEAR_1: self.sun_planetary_gear_1,
            GearType.GEAR_2: self.sun_planetary_gear_2,
            GearType.GEAR_3: self.sun_planetary_gear_3,
            GearType.GEAR_4: self.sun_planetary_gear_4,
            GearType.CARRIER: self.planetary_carrier,
            GearType.REDUCER: self.planetary_reducer,
        }

        self.obj_dict = {
            "ring_gear": self.ring_gear,
            "planetary_carrier": self.planetary_carrier,
            "sun_planetary_gear_1": self.sun_planetary_gear_1,
            "sun_planetary_gear_2": self.sun_planetary_gear_2,
            "sun_planetary_gear_3": self.sun_planetary_gear_3,
            "sun_planetary_gear_4": self.sun_planetary_gear_4,
            "planetary_reducer": self.planetary_reducer,
        }

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add cameras to scene sensors (only if enabled)
        if self.head_camera is not None:
            self.scene.sensors["head_camera"] = self.head_camera
        if self.left_hand_camera is not None:
            self.scene.sensors["left_hand_camera"] = self.left_hand_camera
        if self.right_hand_camera is not None:
            self.scene.sensors["right_hand_camera"] = self.right_hand_camera

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Add robot to scene
        self.scene.articulations["robot"] = self.robot

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Initialize physics materials
        self._initialize_physics_materials()

    def _initialize_physics_materials(self):
        """Initialize physics materials for gripper, gears, and table."""
        # Gripper material (high friction)
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gripper_friction_coefficient,
            dynamic_friction=self.cfg.gripper_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)

        # Gear material (low friction)
        gear_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gears_friction_coefficient,
            dynamic_friction=self.cfg.gears_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gear_material", gear_mat_cfg)

        # Table material
        table_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.table_friction_coefficient,
            dynamic_friction=self.cfg.table_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/table_material", table_mat_cfg)

        # Bind materials to objects
        for env_idx in range(self.scene.num_envs):
            # Gripper
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", 
                "/World/Materials/gripper_material"
            )
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", 
                "/World/Materials/gripper_material"
            )
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/right_gripper_link1/collisions", 
                "/World/Materials/gripper_material"
            )
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Robot/right_gripper_link2/collisions", 
                "/World/Materials/gripper_material"
            )

            # Gears
            for gear_name in ["ring_gear", "sun_planetary_gear_1", "sun_planetary_gear_2", 
                             "sun_planetary_gear_3", "sun_planetary_gear_4", 
                             "planetary_carrier", "planetary_reducer"]:
                sim_utils.bind_physics_material(
                    f"/World/envs/env_{env_idx}/{gear_name}/node_/mesh_", 
                    "/World/Materials/gear_material"
                )

            # Table
            sim_utils.bind_physics_material(
                f"/World/envs/env_{env_idx}/Table/table/body_whiteLarge", 
                "/World/Materials/table_material"
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Pre-process actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions to the robot."""
        # Parse actions: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
        left_arm_action = self.actions[:, :6]
        right_arm_action = self.actions[:, 6:12]
        left_gripper_action = self.actions[:, 12:13]
        right_gripper_action = self.actions[:, 13:14]

        # Set joint position targets
        self.robot.set_joint_position_target(left_arm_action, joint_ids=self._left_arm_joint_idx)
        self.robot.set_joint_position_target(right_arm_action, joint_ids=self._right_arm_joint_idx)
        self.robot.set_joint_position_target(left_gripper_action, joint_ids=self._left_gripper_dof_idx)
        self.robot.set_joint_position_target(right_gripper_action, joint_ids=self._right_gripper_dof_idx)

        # Update objects
        sim_dt = self.sim.get_physics_dt()
        for obj in self.obj_dict.values():
            obj.update(sim_dt)

        # Update cameras (only if enabled)
        for cam in [self.head_camera, self.left_hand_camera, self.right_hand_camera]:
            if cam is not None:
                cam.update(dt=sim_dt)

    def _get_observations(self) -> dict:
        """Get observations for the current state.
        
        Observations include:
        - Joint positions and velocities
        - End-effector poses
        - Current gear position and orientation
        - Target position (based on current sub-task)
        - Sub-task encoding
        """
        # Joint states
        left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        left_gripper_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        right_gripper_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]

        left_arm_joint_vel = self.robot.data.joint_vel[:, self._left_arm_joint_idx]
        right_arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_idx]
        left_gripper_vel = self.robot.data.joint_vel[:, self._left_gripper_dof_idx]
        right_gripper_vel = self.robot.data.joint_vel[:, self._right_gripper_dof_idx]

        # End-effector poses
        left_ee_pos, left_ee_quat = self._get_end_effector_pose("left")
        right_ee_pos, right_ee_quat = self._get_end_effector_pose("right")

        # Current gear info
        gear_obs = self._get_gear_observations()

        # Sub-task encoding (one-hot: approach, grasp, transport)
        subtask_encoding = torch.zeros(self.num_envs, 3, device=self.device)
        for env_idx in range(self.num_envs):
            if self.current_subtask[env_idx] == SubTaskType.APPROACH:
                subtask_encoding[env_idx, 0] = 1.0
            elif self.current_subtask[env_idx] == SubTaskType.GRASP:
                subtask_encoding[env_idx, 1] = 1.0
            else:  # TRANSPORT
                subtask_encoding[env_idx, 2] = 1.0

        # Gear index encoding (one-hot: 6 gears)
        gear_idx_encoding = torch.zeros(self.num_envs, 6, device=self.device)
        for env_idx in range(self.num_envs):
            gear_idx = min(self.current_gear_idx[env_idx].item(), 5)
            gear_idx_encoding[env_idx, gear_idx] = 1.0

        # Concatenate all observations
        obs = torch.cat([
            left_arm_joint_pos,      # 6
            right_arm_joint_pos,     # 6
            left_gripper_pos,        # 1
            right_gripper_pos,       # 1
            left_arm_joint_vel,      # 6
            right_arm_joint_vel,     # 6
            left_gripper_vel,        # 1
            right_gripper_vel,       # 1
            left_ee_pos,             # 3
            left_ee_quat,            # 4
            right_ee_pos,            # 3
            right_ee_quat,           # 4
            gear_obs,                # Variable
            subtask_encoding,        # 3
            gear_idx_encoding,       # 6
        ], dim=-1)

        return {"policy": obs}

    def _get_gear_observations(self) -> torch.Tensor:
        """Get observations related to current gear and target."""
        obs_list = []
        
        for env_idx in range(self.num_envs):
            gear_idx = self.current_gear_idx[env_idx].item()
            if gear_idx >= len(self.gear_sequence):
                gear_idx = len(self.gear_sequence) - 1
            
            current_gear = self.gear_sequence[gear_idx]
            gear_obj = self.gear_dict[current_gear]
            
            # Current gear position and orientation
            gear_pos = gear_obj.data.root_state_w[env_idx, :3]
            gear_quat = gear_obj.data.root_state_w[env_idx, 3:7]
            gear_vel = gear_obj.data.root_lin_vel_w[env_idx]

            # Target position
            target_pos, target_quat = self._get_target_position_for_gear(current_gear, env_idx)

            # Distance to target
            distance = torch.norm(gear_pos - target_pos)

            env_obs = torch.cat([
                gear_pos,
                gear_quat,
                gear_vel,
                target_pos,
                target_quat,
                distance.unsqueeze(0),
            ])
            obs_list.append(env_obs)

        return torch.stack(obs_list, dim=0)

    def _get_end_effector_pose(self, arm_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get end-effector position and quaternion for specified arm."""
        if arm_name == "left":
            body_idx = self._left_ee_body_idx[0]
        else:
            body_idx = self._right_ee_body_idx[0]

        ee_pose = self.robot.data.body_state_w[:, body_idx, 0:7]
        ee_pos = ee_pose[:, :3]
        ee_quat = ee_pose[:, 3:7]

        return ee_pos, ee_quat

    def _get_target_position_for_gear(
        self, 
        gear: GearType, 
        env_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get target position for a gear based on assembly sequence."""
        if gear in [GearType.GEAR_1, GearType.GEAR_2, GearType.GEAR_3]:
            # Target is a pin on planetary carrier
            carrier_pos = self.planetary_carrier.data.root_state_w[env_idx, :3]
            carrier_quat = self.planetary_carrier.data.root_state_w[env_idx, 3:7]

            # Get pin index from mapping or use gear index
            pin_idx = gear.value.split("_")[-1]
            pin_idx = int(pin_idx) - 1 if pin_idx.isdigit() else 0
            pin_idx = min(pin_idx, len(self.pin_local_positions) - 1)
            
            pin_local_pos = self.pin_local_positions[pin_idx]
            
            # Transform to world coordinates
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            target_quat, target_pos = torch_utils.tf_combine(
                carrier_quat.unsqueeze(0), 
                carrier_pos.unsqueeze(0),
                pin_quat.unsqueeze(0), 
                pin_local_pos.unsqueeze(0)
            )
            
            # Add height offset for gear mounting
            target_pos = target_pos.squeeze(0)
            target_pos[2] += 0.023
            target_quat = target_quat.squeeze(0)
            
            return target_pos, target_quat

        elif gear == GearType.GEAR_4:
            # Target is center of planetary carrier
            carrier_pos = self.planetary_carrier.data.root_state_w[env_idx, :3]
            carrier_quat = self.planetary_carrier.data.root_state_w[env_idx, 3:7]
            
            target_pos = carrier_pos.clone()
            target_pos[2] += 0.03
            
            return target_pos, carrier_quat

        elif gear == GearType.CARRIER:
            # Target is on ring gear
            ring_gear_pos = self.ring_gear.data.root_state_w[env_idx, :3]
            ring_gear_quat = self.ring_gear.data.root_state_w[env_idx, 3:7]
            
            target_pos = ring_gear_pos.clone()
            target_pos[2] += 0.004
            
            return target_pos, ring_gear_quat

        elif gear == GearType.REDUCER:
            # Target is on gear 4
            gear_4_pos = self.sun_planetary_gear_4.data.root_state_w[env_idx, :3]
            gear_4_quat = self.sun_planetary_gear_4.data.root_state_w[env_idx, 3:7]
            
            target_pos = gear_4_pos.clone()
            target_pos[2] += 0.025
            
            return target_pos, gear_4_quat

        # Default fallback
        return torch.zeros(3, device=self.device), torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float)

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards based on current sub-task."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        for env_idx in range(self.num_envs):
            gear_idx = self.current_gear_idx[env_idx].item()
            if gear_idx >= len(self.gear_sequence):
                continue
                
            current_gear = self.gear_sequence[gear_idx]
            subtask = self.current_subtask[env_idx]

            if subtask == SubTaskType.APPROACH:
                rewards[env_idx] = self._compute_approach_reward(current_gear, env_idx)
            elif subtask == SubTaskType.GRASP:
                rewards[env_idx] = self._compute_grasp_reward(current_gear, env_idx)
            elif subtask == SubTaskType.TRANSPORT:
                rewards[env_idx] = self._compute_transport_reward(current_gear, env_idx)

            # Time penalty
            rewards[env_idx] -= self.cfg.reward_time_penalty

        return rewards

    def _compute_approach_reward(self, gear: GearType, env_idx: int) -> torch.Tensor:
        """Compute reward for approach sub-task.
        
        Rewards agent for:
        1. Moving EE directly above the gear (horizontal alignment)
        2. Reaching correct pre-grasp height
        3. Orienting gripper downward
        4. Keeping gripper open
        """
        reward = torch.tensor(0.0, device=self.device)
        
        gear_obj = self.gear_dict[gear]
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        
        # Get appropriate arm based on gear position
        arm_name = self._get_arm_for_gear(gear, env_idx)
        ee_pos, ee_quat = self._get_end_effector_pose(arm_name)
        ee_pos = ee_pos[env_idx]
        ee_quat = ee_quat[env_idx]
        
        # 1. Horizontal distance reward (EE should be above gear)
        horizontal_distance = torch.norm(ee_pos[:2] - gear_pos[:2])
        horizontal_reward = torch.exp(-horizontal_distance / 0.05) * self.cfg.reward_approach_distance_weight
        reward += horizontal_reward
        
        # 2. Height reward (EE should be at pre-grasp height)
        pre_grasp_height = gear_pos[2] + self.cfg.pre_grasp_height_offset
        height_diff = torch.abs(ee_pos[2] - pre_grasp_height)
        height_reward = torch.exp(-height_diff / 0.05) * self.cfg.reward_approach_height_weight
        reward += height_reward
        
        # 3. Orientation reward (gripper should point downward)
        target_quat = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        quat_dot = torch.abs(torch.dot(ee_quat, target_quat))
        orientation_reward = quat_dot * self.cfg.reward_approach_orientation_weight
        reward += orientation_reward
        
        # 4. Gripper open reward
        if arm_name == "left":
            gripper_pos = self.robot.data.joint_pos[env_idx, self._left_gripper_dof_idx[0]]
        else:
            gripper_pos = self.robot.data.joint_pos[env_idx, self._right_gripper_dof_idx[0]]
        gripper_open_reward = torch.clamp(gripper_pos / 0.04, 0, 1) * self.cfg.reward_approach_gripper_open_weight
        reward += gripper_open_reward
        
        # Completion bonus
        if self._check_approach_complete(gear, env_idx):
            reward += self.cfg.reward_approach_complete_bonus

        return reward

    def _compute_grasp_reward(self, gear: GearType, env_idx: int) -> torch.Tensor:
        """Compute reward for grasp sub-task."""
        reward = torch.tensor(0.0, device=self.device)
        
        arm_name = self._get_arm_for_gear(gear, env_idx)
        gear_obj = self.gear_dict[gear]
        
        # Gripper closure reward
        if arm_name == "left":
            gripper_pos = self.robot.data.joint_pos[env_idx, self._left_gripper_dof_idx[0]]
        else:
            gripper_pos = self.robot.data.joint_pos[env_idx, self._right_gripper_dof_idx[0]]
        
        # Reward for closing gripper (lower is closed)
        gripper_closed = torch.clamp(1.0 - gripper_pos / 0.04, 0, 1)
        reward += gripper_closed * self.cfg.reward_grasp_gripper_weight
        
        # Lift reward
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        lift_height = gear_pos[2] - self.cfg.table_height
        lift_reward = torch.clamp(lift_height / self.cfg.grasp_lift_height, 0, 1)
        reward += lift_reward * self.cfg.reward_grasp_lift_weight
        
        # Completion bonus
        if self._check_grasp_complete(gear, env_idx):
            reward += self.cfg.reward_grasp_complete_bonus

        return reward

    def _compute_transport_reward(self, gear: GearType, env_idx: int) -> torch.Tensor:
        """Compute reward for transport sub-task.
        
        Rewards are designed to guide the agent toward meeting the
        evaluate_score() criteria:
        - Horizontal alignment with target
        - Correct height positioning
        - Orientation alignment
        - Stability (low velocity)
        """
        reward = torch.tensor(0.0, device=self.device)
        
        gear_obj = self.gear_dict[gear]
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        gear_quat = gear_obj.data.root_state_w[env_idx, 3:7]
        
        target_pos, target_quat = self._get_target_position_for_gear(gear, env_idx)
        
        # Get precision thresholds based on gear type
        if gear in [GearType.GEAR_1, GearType.GEAR_2, GearType.GEAR_3]:
            h_precision = 0.002  # 2mm horizontal
            v_precision = 0.012  # 12mm height
        elif gear == GearType.REDUCER:
            h_precision = 0.005  # 5mm horizontal
            v_precision = 0.002  # 2mm height (strictest)
        else:
            h_precision = 0.005  # 5mm horizontal
            v_precision = 0.004  # 4mm height
        
        # 1. Horizontal distance reward (x, y only)
        horizontal_distance = torch.norm(gear_pos[:2] - target_pos[:2])
        horizontal_reward = torch.exp(-horizontal_distance / h_precision) * self.cfg.reward_transport_distance_weight
        reward += horizontal_reward
        
        # 2. Height reward
        height_diff = torch.abs(gear_pos[2] - target_pos[2])
        height_reward = torch.exp(-height_diff / v_precision) * self.cfg.reward_transport_height_weight
        reward += height_reward
        
        # 3. Orientation reward
        quat_dot = torch.abs(torch.dot(gear_quat, target_quat))
        orientation_reward = quat_dot * self.cfg.reward_transport_orientation_weight
        reward += orientation_reward
        
        # 4. Stability reward
        gear_vel = gear_obj.data.root_lin_vel_w[env_idx]
        velocity = torch.norm(gear_vel)
        stability_reward = torch.exp(-velocity / 0.01) * self.cfg.reward_transport_stability_weight
        reward += stability_reward
        
        # Completion bonus (significant for achieving evaluate_score criteria)
        if self._check_transport_complete(gear, env_idx):
            reward += self.cfg.reward_transport_complete_bonus

        return reward

    def _check_approach_complete(self, gear: GearType, env_idx: int) -> bool:
        """Check if approach sub-task is complete.
        
        Approach is complete when:
        1. EE is directly above the gear (within horizontal threshold)
        2. EE is at pre-grasp height (above gear)
        3. EE orientation is pointing downward
        4. Gripper is open
        """
        gear_obj = self.gear_dict[gear]
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        
        arm_name = self._get_arm_for_gear(gear, env_idx)
        ee_pos, ee_quat = self._get_end_effector_pose(arm_name)
        ee_pos = ee_pos[env_idx]
        ee_quat = ee_quat[env_idx]
        
        # 1. Check horizontal distance (EE should be above gear, not beside it)
        horizontal_distance = torch.norm(ee_pos[:2] - gear_pos[:2])
        horizontal_ok = horizontal_distance.item() < self.cfg.approach_horizontal_threshold
        
        # 2. Check height (EE should be at pre-grasp height above gear)
        pre_grasp_height = gear_pos[2] + self.cfg.pre_grasp_height_offset
        height_diff = abs(ee_pos[2].item() - pre_grasp_height.item())
        height_ok = height_diff < self.cfg.approach_height_threshold
        
        # 3. Check orientation (gripper should point downward)
        # Target quaternion for gripper pointing down: [0, -1, 0, 0] or [0, 1, 0, 0]
        # Check that z-component of gripper direction is pointing down
        target_quat = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        quat_dot = torch.abs(torch.dot(ee_quat, target_quat))
        orientation_ok = quat_dot.item() > self.cfg.approach_orientation_dot_threshold
        
        # 4. Check gripper is open
        if arm_name == "left":
            gripper_pos = self.robot.data.joint_pos[env_idx, self._left_gripper_dof_idx[0]]
        else:
            gripper_pos = self.robot.data.joint_pos[env_idx, self._right_gripper_dof_idx[0]]
        gripper_open = gripper_pos.item() > self.cfg.gripper_open_threshold
        
        return horizontal_ok and height_ok and orientation_ok and gripper_open

    def _check_grasp_complete(self, gear: GearType, env_idx: int) -> bool:
        """Check if grasp sub-task is complete."""
        arm_name = self._get_arm_for_gear(gear, env_idx)
        gear_obj = self.gear_dict[gear]
        
        # Check gripper closed
        if arm_name == "left":
            gripper_pos = self.robot.data.joint_pos[env_idx, self._left_gripper_dof_idx[0]]
        else:
            gripper_pos = self.robot.data.joint_pos[env_idx, self._right_gripper_dof_idx[0]]
        
        gripper_closed = gripper_pos.item() < 0.01  # Nearly closed
        
        # Check gear lifted
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        gear_lifted = gear_pos[2].item() > self.cfg.table_height + self.cfg.grasp_lift_height * 0.5
        
        return gripper_closed and gear_lifted

    def _check_transport_complete(self, gear: GearType, env_idx: int) -> bool:
        """Check if transport sub-task is complete.
        
        Uses the same criteria as evaluate_score() to ensure RL training
        leads to successful assembly scoring:
        - Gear 1-3 on pins: horizontal < 2mm, height < 12mm, angle < 0.1 rad
        - Gear 4 center: horizontal < 5mm, height < 4mm, angle < 0.1 rad  
        - Carrier on ring: horizontal < 5mm, height < 4mm, angle < 0.1 rad
        - Reducer on gear: horizontal < 5mm, height < 2mm, angle < 0.1 rad
        """
        gear_obj = self.gear_dict[gear]
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        gear_quat = gear_obj.data.root_state_w[env_idx, 3:7]
        
        target_pos, target_quat = self._get_target_position_for_gear(gear, env_idx)
        
        # Calculate horizontal distance (x, y only)
        horizontal_distance = torch.norm(gear_pos[:2] - target_pos[:2])
        
        # Calculate height difference
        height_diff = torch.abs(gear_pos[2] - target_pos[2])
        
        # Calculate orientation angle difference
        quat_dot = torch.abs(torch.dot(gear_quat, target_quat))
        # angle = 2 * acos(|q1 · q2|), but we use dot product threshold
        angle_ok = quat_dot.item() > 0.995  # approximately < 0.1 rad
        
        # Check stability
        gear_vel = gear_obj.data.root_lin_vel_w[env_idx]
        velocity = torch.norm(gear_vel)
        is_stable = velocity.item() < self.cfg.transport_stability_velocity_threshold
        
        # Apply different thresholds based on gear type (matching evaluate_score criteria)
        if gear in [GearType.GEAR_1, GearType.GEAR_2, GearType.GEAR_3]:
            # Gears on pins: strictest horizontal precision
            horizontal_ok = horizontal_distance.item() < 0.002  # 2mm
            height_ok = height_diff.item() < 0.012  # 12mm
        elif gear == GearType.GEAR_4:
            # Center gear
            horizontal_ok = horizontal_distance.item() < 0.005  # 5mm
            height_ok = height_diff.item() < 0.004  # 4mm
        elif gear == GearType.CARRIER:
            # Carrier on ring gear
            horizontal_ok = horizontal_distance.item() < 0.005  # 5mm
            height_ok = height_diff.item() < 0.004  # 4mm
        elif gear == GearType.REDUCER:
            # Reducer on gear 4: strictest height precision
            horizontal_ok = horizontal_distance.item() < 0.005  # 5mm
            height_ok = height_diff.item() < 0.002  # 2mm
        else:
            # Fallback
            horizontal_ok = horizontal_distance.item() < 0.005
            height_ok = height_diff.item() < 0.005
        
        return horizontal_ok and height_ok and angle_ok and is_stable

    def _get_arm_for_gear(self, gear: GearType, env_idx: int) -> str:
        """Determine which arm should handle the gear based on position."""
        gear_obj = self.gear_dict[gear]
        gear_pos = gear_obj.data.root_state_w[env_idx, :3]
        
        # Use y-coordinate to determine arm (left arm for positive y, right for negative)
        if gear_pos[1].item() > 0:
            return "left"
        else:
            return "right"

    def _check_and_transition(self, env_idx: int) -> Tuple[bool, Dict]:
        """Check sub-task completion and transition to next sub-task."""
        gear_idx = self.current_gear_idx[env_idx].item()
        
        if gear_idx >= len(self.gear_sequence):
            return False, {}
            
        current_gear = self.gear_sequence[gear_idx]
        subtask = self.current_subtask[env_idx]
        
        transition_info = {
            "from_subtask": subtask.value,
            "current_gear": current_gear.value,
        }

        if subtask == SubTaskType.APPROACH:
            if self._check_approach_complete(current_gear, env_idx):
                self.current_subtask[env_idx] = SubTaskType.GRASP
                self.stage_start_step[env_idx] = self.episode_length_buf[env_idx]
                transition_info["to_subtask"] = SubTaskType.GRASP.value
                return True, transition_info

        elif subtask == SubTaskType.GRASP:
            if self._check_grasp_complete(current_gear, env_idx):
                self.current_subtask[env_idx] = SubTaskType.TRANSPORT
                self.stage_start_step[env_idx] = self.episode_length_buf[env_idx]
                transition_info["to_subtask"] = SubTaskType.TRANSPORT.value
                return True, transition_info

        elif subtask == SubTaskType.TRANSPORT:
            if self._check_transport_complete(current_gear, env_idx):
                self.assembled_gears[env_idx].add(current_gear)
                self.current_gear_idx[env_idx] += 1
                
                if self.current_gear_idx[env_idx].item() >= len(self.gear_sequence):
                    transition_info["to_subtask"] = "done"
                    transition_info["all_complete"] = True
                else:
                    self.current_subtask[env_idx] = SubTaskType.APPROACH
                    self.stage_start_step[env_idx] = self.episode_length_buf[env_idx]
                    next_gear = self.gear_sequence[self.current_gear_idx[env_idx].item()]
                    transition_info["to_subtask"] = SubTaskType.APPROACH.value
                    transition_info["next_gear"] = next_gear.value
                
                return True, transition_info

        return False, {}

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done.
        
        For subtask-specific training, episode terminates when:
        - "approach": Approach phase complete
        - "grasp": Grasp phase complete  
        - "transport_*": Transport phase complete for specified gear
        - "full": All gears assembled
        """
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        training_subtask = self.cfg.training_subtask
        
        for env_idx in range(self.num_envs):
            if training_subtask == "full":
                # Full sequence: done when all gears assembled
                all_assembled = self.current_gear_idx[env_idx].item() >= len(self.gear_sequence)
                terminated[env_idx] = all_assembled
                
            elif training_subtask == "approach":
                # Approach only: done when approach complete
                gear_idx = self.current_gear_idx[env_idx].item()
                if gear_idx < len(self.gear_sequence):
                    current_gear = self.gear_sequence[gear_idx]
                    terminated[env_idx] = self._check_approach_complete(current_gear, env_idx)
                    
            elif training_subtask == "grasp":
                # Grasp only: done when grasp complete
                gear_idx = self.current_gear_idx[env_idx].item()
                if gear_idx < len(self.gear_sequence):
                    current_gear = self.gear_sequence[gear_idx]
                    terminated[env_idx] = self._check_grasp_complete(current_gear, env_idx)
                    
            elif training_subtask.startswith("transport_"):
                # Transport specific gear: done when transport complete
                gear_idx = self.current_gear_idx[env_idx].item()
                if gear_idx < len(self.gear_sequence):
                    current_gear = self.gear_sequence[gear_idx]
                    terminated[env_idx] = self._check_transport_complete(current_gear, env_idx)

        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _randomize_object_positions(self, env_ids: torch.Tensor) -> Dict:
        """Randomize gear positions on the table."""
        OBJECT_RADII = {
            'ring_gear': 0.1,
            'sun_planetary_gear_1': 0.035,
            'sun_planetary_gear_2': 0.035,
            'sun_planetary_gear_3': 0.035,
            'sun_planetary_gear_4': 0.035,
            'planetary_carrier': 0.07,
            'planetary_reducer': 0.04,
        }
        
        safety_margin = 0.02
        max_attempts = 1000
        
        object_list = [
            self.planetary_carrier, self.ring_gear,
            self.sun_planetary_gear_1, self.sun_planetary_gear_2,
            self.sun_planetary_gear_3, self.sun_planetary_gear_4,
            self.planetary_reducer
        ]
        object_names = [
            'planetary_carrier', 'ring_gear',
            'sun_planetary_gear_1', 'sun_planetary_gear_2',
            'sun_planetary_gear_3', 'sun_planetary_gear_4',
            'planetary_reducer'
        ]

        initial_root_state = {}
        placed_objects = [[] for _ in range(self.scene.num_envs)]

        for obj_idx, obj in enumerate(object_list):
            obj_name = object_names[obj_idx]
            root_state = obj.data.default_root_state.clone()
            current_radius = OBJECT_RADII.get(obj_name, 0.05)

            for env_idx in range(self.scene.num_envs):
                position_found = False

                for attempt in range(max_attempts):
                    x = torch.rand(1, device=self.device).item() * 0.2 + 0.3 + self.cfg.x_offset
                    y = torch.rand(1, device=self.device).item() * 0.8 - 0.4
                    z = 0.92

                    # Special placement rules
                    if obj_name == "planetary_carrier":
                        x = 0.4 + self.cfg.x_offset
                        y = 0.0
                    elif obj_name in ["sun_planetary_gear_1", "sun_planetary_gear_2"]:
                        y = torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name in ["sun_planetary_gear_3", "sun_planetary_gear_4"]:
                        y = -torch.rand(1, device=self.device).item() * 0.4

                    pos = torch.tensor([x, y, z], device=self.device)

                    is_valid = True
                    for placed_pos, placed_obj_name in placed_objects[env_idx]:
                        placed_radius = OBJECT_RADII.get(placed_obj_name, 0.05)
                        min_distance = current_radius + placed_radius + safety_margin
                        distance = torch.norm(pos[:2] - placed_pos[:2]).item()
                        if distance < min_distance:
                            is_valid = False
                            break

                    if is_valid:
                        root_state[env_idx, :3] = pos
                        placed_objects[env_idx].append((pos, obj_name))
                        position_found = True
                        break

                if not position_found:
                    root_state[env_idx, :3] = pos
                    placed_objects[env_idx].append((pos, obj_name))

            obj.write_root_state_to_sim(root_state)
            initial_root_state[obj_name] = root_state.clone()

        return initial_root_state

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments.
        
        For subtask-specific training, sets up appropriate initial state:
        - "approach": Normal random positions
        - "grasp": EE positioned near gear (approach-complete state)
        - "transport_*": Gear already grasped (grasp-complete state)
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)

        training_subtask = self.cfg.training_subtask

        # Reset table
        root_state = self.table.data.default_root_state.clone()
        self.table.write_root_state_to_sim(root_state)

        # Randomize object positions
        self.initial_root_state = self._randomize_object_positions(env_ids)

        # Update all objects
        for obj in self.obj_dict.values():
            obj.update(self.sim.get_physics_dt())

        # Reset robot
        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self._joint_idx]
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)

        # Set torso joints
        torso_pos = torch.tensor([
            self.cfg.initial_torso_joint1_pos,
            self.cfg.initial_torso_joint2_pos,
            self.cfg.initial_torso_joint3_pos
        ], device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        self.robot.write_joint_position_to_sim(torso_pos, self._torso_joint_idx, env_ids)

        # Determine starting subtask and gear based on training mode
        start_subtask = SubTaskType.APPROACH
        start_gear_idx = self.cfg.curriculum_start_gear_idx
        
        if training_subtask == "grasp":
            # Start from grasp phase (approach already complete)
            start_subtask = SubTaskType.GRASP
        elif training_subtask.startswith("transport_"):
            # Start from transport phase (grasp already complete)
            start_subtask = SubTaskType.TRANSPORT
            # Set gear index based on transport target
            if training_subtask == "transport_gear_1":
                start_gear_idx = 0
            elif training_subtask == "transport_gear_2":
                start_gear_idx = 1
            elif training_subtask == "transport_gear_3":
                start_gear_idx = 2
            elif training_subtask == "transport_gear_4":
                start_gear_idx = 3
            elif training_subtask == "transport_carrier":
                start_gear_idx = 4
            elif training_subtask == "transport_reducer":
                start_gear_idx = 5

        # Reset multi-stage state
        for env_idx in env_ids:
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.item()
            self.current_gear_idx[env_idx] = start_gear_idx
            self.current_subtask[env_idx] = start_subtask
            self.stage_start_step[env_idx] = 0
            self.assembled_gears[env_idx] = set()
            self.gear_to_pin_map[env_idx] = {}
        
        # Setup initial state based on training subtask
        if training_subtask == "grasp":
            self._setup_grasp_initial_state(env_ids)
        elif training_subtask.startswith("transport_"):
            self._setup_transport_initial_state(env_ids, start_gear_idx)

    def _setup_grasp_initial_state(self, env_ids):
        """Setup initial state for grasp training (EE near gear)."""
        for env_idx in env_ids:
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.item()
            
            gear_idx = self.current_gear_idx[env_idx].item()
            if gear_idx >= len(self.gear_sequence):
                continue
                
            current_gear = self.gear_sequence[gear_idx]
            gear_obj = self.gear_dict[current_gear]
            gear_pos = gear_obj.data.root_state_w[env_idx, :3]
            
            # Position EE just above the gear
            arm_name = self._get_arm_for_gear(current_gear, env_idx)
            
            # Open gripper
            if arm_name == "left":
                gripper_open = torch.tensor([[0.04]], device=self.device)
                self.robot.set_joint_position_target(gripper_open, joint_ids=self._left_gripper_dof_idx)
            else:
                gripper_open = torch.tensor([[0.04]], device=self.device)
                self.robot.set_joint_position_target(gripper_open, joint_ids=self._right_gripper_dof_idx)

    def _setup_transport_initial_state(self, env_ids, gear_idx: int):
        """Setup initial state for transport training (gear already grasped)."""
        for env_idx in env_ids:
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.item()
            
            if gear_idx >= len(self.gear_sequence):
                continue
                
            current_gear = self.gear_sequence[gear_idx]
            gear_obj = self.gear_dict[current_gear]
            
            # Get EE position and lift gear to that position
            arm_name = self._get_arm_for_gear(current_gear, env_idx)
            ee_pos, ee_quat = self._get_end_effector_pose(arm_name)
            ee_pos = ee_pos[env_idx]
            
            # Close gripper
            if arm_name == "left":
                gripper_closed = torch.tensor([[0.0]], device=self.device)
                self.robot.set_joint_position_target(gripper_closed, joint_ids=self._left_gripper_dof_idx)
            else:
                gripper_closed = torch.tensor([[0.0]], device=self.device)
                self.robot.set_joint_position_target(gripper_closed, joint_ids=self._right_gripper_dof_idx)
            
            # Move gear to lifted position (simulating already grasped)
            lifted_pos = gear_obj.data.root_state_w[env_idx].clone()
            lifted_pos[2] = self.cfg.table_height + self.cfg.lifting_height
            
            gear_root_state = gear_obj.data.root_state_w.clone()
            gear_root_state[env_idx, :3] = lifted_pos[:3]
            gear_obj.write_root_state_to_sim(gear_root_state)

    def get_key_points(self):
        """Get all key points for score evaluation (same as gearbox_recovery_env)."""
        # Pin positions
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()
        num_envs = planetary_carrier_pos.shape[0]

        pin_world_positions = []
        pin_world_quats = []
        for pin_local_pos in self.pin_local_positions:
            # Expand to match batch dimension
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, -1)
            pin_local_pos_batch = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
            
            pin_world_quat, pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, 
                pin_quat, pin_local_pos_batch
            )
            pin_world_positions.append(pin_world_pos)
            pin_world_quats.append(pin_world_quat)

        gear_world_positions = []
        gear_world_quats = []
        
        gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2',
                      'sun_planetary_gear_3', 'sun_planetary_gear_4']
        for gear_name in gear_names:
            gear_obj = self.obj_dict[gear_name]
            gear_pos = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()
            gear_world_positions.append(gear_pos)
            gear_world_quats.append(gear_quat)
        
        carrier_world_pos = planetary_carrier_pos
        carrier_world_quat = planetary_carrier_quat

        ring_gear_world_pos = self.ring_gear.data.root_state_w[:, :3].clone()
        ring_gear_world_quat = self.ring_gear.data.root_state_w[:, 3:7].clone()

        reducer_world_pos = self.planetary_reducer.data.root_state_w[:, :3].clone()
        reducer_world_quat = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        return (pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats,
                carrier_world_pos, carrier_world_quat, ring_gear_world_pos, ring_gear_world_quat,
                reducer_world_pos, reducer_world_quat)

    def evaluate_score(self) -> Tuple[int, Dict]:
        """Evaluate assembly score using the EXACT same criteria as gearbox_recovery_env.
        
        Returns:
            score: Total score (max 6 for full assembly)
            score_details: Dictionary with breakdown of each scoring component
        """
        (pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats,
         planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat,
         reducer_world_pos, reducer_world_quat) = self.get_key_points()
        
        score = 0
        score_details = {
            'gears_on_pins': 0,
            'carrier_on_ring': 0,
            'gear_in_middle': 0,
            'reducer_on_gear': 0,
        }

        # Check gears mounted to planetary carrier pins
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]

            for pin_idx in range(len(pin_world_positions)):
                pin_world_pos = pin_world_positions[pin_idx]
                pin_world_quat = pin_world_quats[pin_idx]
                
                distance = torch.norm(gear_world_pos[:, :2] - pin_world_pos[:, :2])
                height_diff = gear_world_pos[:, 2] - pin_world_pos[:, 2]
                angle = torch.acos(torch.clamp(
                    torch.dot(gear_world_quat.squeeze(0), pin_world_quat.squeeze(0)), -1.0, 1.0
                ))
                
                if distance < 0.002 and angle < 0.1 and height_diff < 0.012:
                    score += 1
                    score_details['gears_on_pins'] += 1

        # Check planetary carrier mounted to ring gear
        distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2])
        height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]
        angle = torch.acos(torch.clamp(
            torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_world_quat.squeeze(0)), -1.0, 1.0
        ))
        if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
            score += 1
            score_details['carrier_on_ring'] = 1

        # Check gear mounted in the middle (gear 4)
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]
            distance = torch.norm(gear_world_pos[:, :2] - ring_gear_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - ring_gear_world_pos[:, 2]
            angle = torch.acos(torch.clamp(
                torch.dot(gear_world_quat.squeeze(0), ring_gear_world_quat.squeeze(0)), -1.0, 1.0
            ))
            if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
                score += 1
                score_details['gear_in_middle'] += 1

        # Check reducer mounted to gear
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]
            distance = torch.norm(gear_world_pos[:, :2] - reducer_world_pos[:, :2])
            height_diff = reducer_world_pos[:, 2] - gear_world_pos[:, 2]
            angle = torch.acos(torch.clamp(
                torch.dot(gear_world_quat.squeeze(0), reducer_world_quat.squeeze(0)), -1.0, 1.0
            ))
            if distance < 0.005 and angle < 0.1 and height_diff > 0 and height_diff < 0.03:
                score += 1
                score_details['reducer_on_gear'] = 1
                break  # Only count once

        return score, score_details

    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment."""
        # Standard step
        obs, reward, terminated, truncated, info = super().step(action)

        # Check for transitions
        for env_idx in range(self.num_envs):
            transition_occurred, transition_info = self._check_and_transition(env_idx)
            if transition_occurred:
                reward[env_idx] += self.cfg.reward_transition_bonus
                if env_idx == 0:  # Log only for first env
                    print(f"[Transition] {transition_info}")

        # Skip score evaluation during training for performance
        # evaluate_score() is designed for single-env final evaluation
        # info['assembly_score'] = 0
        # info['score_details'] = {}

        return obs, reward, terminated, truncated, info

