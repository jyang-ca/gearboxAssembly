# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

# Import environments to trigger gym registration
from .galaxea_lab_external import *  # noqa: F401, F403
from .gearbox_recovery import *  # noqa: F401, F403
from .long_trajectory_assembly import *  # noqa: F401, F403
