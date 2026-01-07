# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long Trajectory Gear Assembly Environment for Reinforcement Learning."""

import gymnasium as gym

from . import agents
from .long_trajectory_assembly_env import LongTrajectoryAssemblyEnv
from .long_trajectory_assembly_env_cfg import LongTrajectoryAssemblyEnvCfg

##
# Register environments
##

gym.register(
    id="Galaxea-LongTrajectoryAssembly-Direct-v0",
    entry_point=f"{__name__}.long_trajectory_assembly_env:LongTrajectoryAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_trajectory_assembly_env_cfg:LongTrajectoryAssemblyEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = [
    "LongTrajectoryAssemblyEnv",
    "LongTrajectoryAssemblyEnvCfg",
]

