---
license: mit
tags:
- reinforcement-learning
- robotics
- isaac-lab
- rtx-5090
- industrial-assembly
datasets:
- simulation
---

# Galaxea Gearbox Assembly R1 Policies

This repository contains the trained Reinforcement Learning (RL) policies for the high-precision gearbox assembly task using the Galaxea R1 robot. These models were trained using **NVIDIA Isaac Lab** on a single **NVIDIA RTX 5090**, achieving state-of-the-art simulation throughput and convergence stability.

## Model Description

The policies are trained to control a 7-DoF robotic arm (Galaxea R1) to assemble a complex planetary gearbox. The task is decomposed into sequential sub-tasks: `Approach` -> `Grasp` -> `Transport` (for each gear).

- **Algorithm**: PPO (Proximal Policy Optimization) via `rl_games`
- **Observation Space**: 69-dim (Joint pos/vel, EE pose, Relative gear targets)
- **Action Space**: 14-dim (Joint position targets + Gripper)
- **Training Framework**: Isaac Lab (DirectRL Mode)

## Performance Metrics

The models were trained with a massive throughput of **~8,200 FPS** (Frames Per Second) using full GPU vectorization.

| Policy | Stage | Avg Reward | Critic Loss | Entropy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Approach** | 1 (Foundation) | ~241.4 | 3.8e-5 | 2.58 | **Converged** |
| **Grasp** | 2 (Manipulation) | ~240.9 | 3.3e-5 | -0.92 | **Converged** |
| **Transport 1** | 3 (Assembly) | ~282.6 | 1.7e-4 | 11.2 | **Robust** |

## Included Files

- `policy_approach.pth`: PyTorch checkpoint for the Approach phase.
- `policy_grasp.pth`: PyTorch checkpoint for the Grasping phase.
- `policy_transport_gear_1.pth`: PyTorch checkpoint for Transporting the first Sun Gear.
- `env_config.py`: The environment configuration used for training (PhysX settings, rewards).
- `agent_config.yaml`: The PPO hyperparameters.

## Usage

These policies are designed to be loaded into the Isaac Lab environment:

```python
# Pseudo-code for loading
from rl_games.torch_runner import Runner

runner = Runner()
runner.load('policy_approach.pth')
# ... run inference ...
```

## Hardware Specification

- **GPU**: NVIDIA GeForce RTX 5090 (32GB)
- **Training Time**: ~3 hours per policy (Optimized from 50+ days)
- **Simultaneous Envs**: 8,192
