# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Long Trajectory Gear Assembly Environment."""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from Galaxea_Lab_External.robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)


@configclass
class LongTrajectoryAssemblyEnvCfg(DirectRLEnvCfg):
    """Configuration for Long Trajectory Gear Assembly Environment.
    
    This environment supports multi-stage assembly tasks with 8 policies:
    - Policy_Approach (shared across all gears)
    - Policy_Grasp (shared across all gears)
    - Policy_Transport_Gear1~4, Carrier, Reducer (6 gear-specific policies)
    
    Transition management is handled via environment-based rule-based transitions.
    """

    # Record data settings
    record_data = False
    record_freq = 5
    
    # Camera settings (disabled by default for RL training)
    enable_cameras = False

    # Environment settings
    sim_dt = 0.01
    decimation = 5
    episode_length_s = 120.0  # Long trajectory: 120 seconds max
    
    # Number of re-renders on reset (for camera sensors)
    num_rerenders_on_reset = 5

    # Action and observation spaces
    # Action: Left arm(6) + Right arm(6) + Left gripper(1) + Right gripper(1) = 14
    action_space = 14
    # Observation space:
    # - Joint pos: 6+6+1+1 = 14
    # - Joint vel: 6+6+1+1 = 14
    # - EE poses: 3+4+3+4 = 14
    # - Gear obs: 3+4+3+3+4+1 = 18
    # - Encodings: 3+6 = 9
    # Total = 69
    observation_space = 69
    state_space = 0

    # Simulation configuration
    # Increase GPU collision stack size to handle many environments (default 2**26)
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt, 
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size=2**31,  # Increased for 8192+ envs (approx 2.1 billion)
        )
    )

    # Robot configuration
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Table configuration
    table_cfg: RigidObjectCfg = TABLE_CFG.replace(
        prim_path="/World/envs/env_.*/Table"
    )

    # Gear configurations with default initial positions
    ring_gear_cfg: RigidObjectCfg = RING_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/ring_gear",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    sun_planetary_gear_1_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4, -0.2, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    sun_planetary_gear_2_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.25, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    sun_planetary_gear_3_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_3",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.15, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    sun_planetary_gear_4_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(
        prim_path="/World/envs/env_.*/sun_planetary_gear_4",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.3, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    planetary_carrier_cfg: RigidObjectCfg = PLANETARY_CARRIER_CFG.replace(
        prim_path="/World/envs/env_.*/planetary_carrier",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.25, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    planetary_reducer_cfg: RigidObjectCfg = PLANETARY_REDUCER_CFG.replace(
        prim_path="/World/envs/env_.*/planetary_reducer",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.1, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Physics material coefficients
    table_friction_coefficient = 0.4
    gears_friction_coefficient = 0.01
    gripper_friction_coefficient = 2.0

    # Camera configurations
    head_camera_cfg: CameraCfg = GALAXEA_HEAD_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam"
    )
    left_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam"
    )
    right_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam"
    )
    
    # Overhead camera for full workspace view (fixed in world, not attached to robot)
    overhead_camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/overhead_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 2.0),  # Above the workspace
            rot=(0.7071, 0.7071, 0.0, 0.0),  # Looking down (90 degree pitch)
            convention="world",
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=4.0, 
        replicate_physics=True
    )

    # Joint names for robot control
    left_arm_joint_dof_name = "left_arm_joint.*"
    right_arm_joint_dof_name = "right_arm_joint.*"
    left_gripper_dof_name = "left_gripper_axis1"
    right_gripper_dof_name = "right_gripper_axis1"
    torso_joint_dof_name = "torso_joint[1-3]"
    torso_joint1_dof_name = "torso_joint1"
    torso_joint2_dof_name = "torso_joint2"
    torso_joint3_dof_name = "torso_joint3"
    torso_joint4_dof_name = "torso_joint4"

    # Initial torso joint positions
    initial_torso_joint1_pos = 0.5
    initial_torso_joint2_pos = -0.8
    initial_torso_joint3_pos = 0.5

    # Table offset
    x_offset = 0.2

    # Assembly precision (1cm as specified in requirements)
    assembly_precision = 0.01  # 1cm

    # Stage timeout configuration (seconds per sub-task)
    stage_timeout_approach = 10.0
    stage_timeout_grasp = 5.0
    stage_timeout_transport = 15.0

    # Reward weights
    reward_approach_distance_weight = 0.1  # Horizontal distance to be above gear
    reward_approach_height_weight = 0.1  # Correct pre-grasp height
    reward_approach_orientation_weight = 0.1  # Gripper pointing downward
    reward_approach_gripper_open_weight = 0.05  # Gripper is open
    reward_approach_complete_bonus = 1.0
    reward_grasp_gripper_weight = 0.1
    reward_grasp_contact_weight = 0.1
    reward_grasp_lift_weight = 0.1
    reward_grasp_complete_bonus = 2.0
    reward_transport_distance_weight = 0.2  # Horizontal alignment reward
    reward_transport_height_weight = 0.2  # Height alignment reward
    reward_transport_orientation_weight = 0.1  # Orientation alignment reward
    reward_transport_stability_weight = 0.1  # Low velocity reward
    reward_transport_complete_bonus = 10.0  # Bonus for meeting evaluate_score criteria
    reward_transition_bonus = 5.0
    reward_time_penalty = 0.001

    # Approach completion thresholds
    approach_distance_threshold = 0.05  # 5cm to gear center (deprecated, use below)
    approach_horizontal_threshold = 0.03  # 3cm - EE must be directly above gear
    approach_height_threshold = 0.02  # 2cm tolerance for pre-grasp height
    approach_orientation_threshold = 0.3  # radians (deprecated)
    approach_orientation_dot_threshold = 0.95  # quaternion dot product threshold (close to 1 = aligned)
    gripper_open_threshold = 0.03  # gripper must be at least this open (rad)
    pre_grasp_height_offset = 0.05  # 5cm above gear for pre-grasp position

    # Grasp completion thresholds
    grasp_gripper_closed_threshold = 0.8  # normalized gripper position
    grasp_contact_force_threshold = 2.0  # Newtons
    grasp_lift_height = 0.1  # 10cm above table

    # Transport completion thresholds
    transport_position_threshold = 0.01  # 1cm precision
    transport_orientation_threshold = 0.1  # radians
    transport_stability_velocity_threshold = 0.01  # m/s

    # Gear assembly sequence
    gear_sequence = [
        "gear_1",  # Sun planetary gear 1
        "gear_2",  # Sun planetary gear 2
        "gear_3",  # Sun planetary gear 3
        "gear_4",  # Sun planetary gear 4 (center)
        "carrier",  # Planetary carrier onto ring gear
        "reducer",  # Planetary reducer onto gear 4
    ]

    # Pin local positions relative to planetary carrier
    pin_0_local_pos = (0.0, -0.054, 0.0)
    pin_1_local_pos = (0.0471, 0.0268, 0.0)
    pin_2_local_pos = (-0.0471, 0.0268, 0.0)

    # TCP (Tool Center Point) offsets
    tcp_offset_x = 0.0079  # 0.3864 - 0.3785
    tcp_offset_z = 0.0909  # 1.1475 - 1.05661

    # Table and grasping heights
    table_height = 0.9
    grasping_height = -0.003
    lifting_height = 0.2

    # Sub-task types for training mode selection
    # "full" - train entire sequence
    # "approach" - train only approach sub-task
    # "grasp" - train only grasp sub-task
    # "transport_gear_1" through "transport_reducer" - train specific transport
    training_subtask = "full"

    # Starting gear index for curriculum learning (0-5)
    curriculum_start_gear_idx = 0

