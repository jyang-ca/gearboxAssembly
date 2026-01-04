# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequential training script for all 8 subtask policies (Phase 2 design).

This script trains all 8 policies in the following order:
1. Policy_Approach (shared) - learns to approach any gear
2. Policy_Grasp (shared) - learns to grasp any gear  
3. Policy_Transport_Gear1 - learns to transport gear 1 to pin
4. Policy_Transport_Gear2 - learns to transport gear 2 to pin
5. Policy_Transport_Gear3 - learns to transport gear 3 to pin
6. Policy_Transport_Gear4 - learns to transport gear 4 to center
7. Policy_Transport_Carrier - learns to place carrier on ring gear
8. Policy_Transport_Reducer - learns to place reducer on gear 4

Usage:
    # Train all policies sequentially
    python scripts/train_all_subtask_policies.py

    # Train with custom settings
    python scripts/train_all_subtask_policies.py --num_envs 2048 --max_iterations 3000
"""

import argparse
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path


# Define the 8 subtasks in training order
SUBTASK_SEQUENCE = [
    ("approach", "Policy_Approach (Shared)"),
    ("grasp", "Policy_Grasp (Shared)"),
    ("transport_gear_1", "Policy_Transport_Gear1"),
    ("transport_gear_2", "Policy_Transport_Gear2"),
    ("transport_gear_3", "Policy_Transport_Gear3"),
    ("transport_gear_4", "Policy_Transport_Gear4"),
    ("transport_carrier", "Policy_Transport_Carrier"),
    ("transport_reducer", "Policy_Transport_Reducer"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train all 8 subtask policies sequentially (Phase 2 design)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Galaxea-LongTrajectoryAssembly-Direct-v0",
        help="Name of the task (environment)."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="Number of parallel environments for each policy training."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="Maximum training iterations per policy."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/long_trajectory_assembly",
        help="Directory to save policy checkpoints."
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from specific subtask index (0-7). Useful for resuming."
    )
    parser.add_argument(
        "--only_subtask",
        type=str,
        default=None,
        choices=["approach", "grasp", "transport_gear_1", "transport_gear_2",
                 "transport_gear_3", "transport_gear_4", "transport_carrier", "transport_reducer"],
        help="Train only a specific subtask."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (no GUI)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    # Weights & Biases arguments
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="Enable Weights and Biases tracking."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="LongTrajectoryAssembly",
        help="Weights and Biases project name."
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights and Biases entity (team or username)."
    )
    parser.add_argument(
        "--continue-on-failure",
        dest="continue_on_failure",
        action="store_true",
        default=False,
        help="Continue training next policy even if current one fails (for non-interactive mode)."
    )
    return parser.parse_args()


def train_subtask(subtask: str, policy_name: str, args, checkpoint_dir: Path, idx: int):
    """Train a single subtask policy."""
    print("=" * 70)
    print(f"[{idx+1}/8] Training: {policy_name}")
    print(f"  Subtask: {subtask}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Max iterations: {args.max_iterations}")
    print("=" * 70)

    # Create checkpoint directory for this subtask
    subtask_checkpoint_dir = checkpoint_dir / subtask
    subtask_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build command - use official Isaac Lab RL-Games trainer
    cmd = [
        sys.executable,
        "scripts/rl_games/train.py",
        f"--task={args.task}",
        f"--num_envs={args.num_envs}",
        f"--max_iterations={args.max_iterations}",
        f"--seed={args.seed}",
    ]

    if args.headless:
        cmd.append("--headless")
    
    # Add W&B tracking arguments
    if args.track:
        cmd.append("--track")
        cmd.append(f"--wandb-project-name={args.wandb_project}")
        # Sanitize policy name for W&B (remove special characters)
        safe_policy_name = policy_name.replace(" ", "_").replace("(", "").replace(")", "")
        cmd.append(f"--wandb-name={safe_policy_name}")
        if args.wandb_entity:
            cmd.append(f"--wandb-entity={args.wandb_entity}")

    # Run training
    print(f"Running: {' '.join(cmd)}")
    print("-" * 70)

    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print(f"\n✓ {policy_name} training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {policy_name} training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {policy_name} training interrupted by user")
        return False


def main():
    args = parse_args()

    # Create main checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = checkpoint_dir / f"training_log_{timestamp}.txt"

    print("=" * 70)
    print("Long Trajectory Assembly - Sequential Policy Training (Phase 2)")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Number of policies to train: 8")
    print(f"Max iterations per policy: {args.max_iterations}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"W&B Tracking: {args.track}")
    if args.track:
        print(f"W&B Project: {args.wandb_project}")
        print(f"W&B Entity: {args.wandb_entity or '(default)'}")
    print("=" * 70)

    # Determine which subtasks to train
    if args.only_subtask:
        # Train only the specified subtask
        subtasks_to_train = [(s, n) for s, n in SUBTASK_SEQUENCE if s == args.only_subtask]
        start_idx = 0
    else:
        # Train from start_from index
        subtasks_to_train = SUBTASK_SEQUENCE[args.start_from:]
        start_idx = args.start_from

    # Training loop
    results = {}
    for idx, (subtask, policy_name) in enumerate(subtasks_to_train):
        global_idx = start_idx + idx
        success = train_subtask(subtask, policy_name, args, checkpoint_dir, global_idx)
        results[policy_name] = success

        if not success:
            print(f"\nTraining stopped at {policy_name}.")
            if args.continue_on_failure:
                print("Continuing with next policy (--continue-on-failure flag set).")
            else:
                print("Stopping training. Use --continue-on-failure to continue even on failure.")
                break

    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    for policy_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {policy_name}: {status}")
    print("=" * 70)

    # Save results to log
    with open(log_file, 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Checkpoint directory: {checkpoint_dir}\n\n")
        f.write("Results:\n")
        for policy_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            f.write(f"  {policy_name}: {status}\n")

    print(f"\nTraining log saved to: {log_file}")

    # Print next steps
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Trained policies are saved in:", checkpoint_dir)
    print("2. To evaluate the full assembly sequence:")
    print(f"   python scripts/eval_long_trajectory_assembly.py --checkpoint_dir {checkpoint_dir}")
    print("3. To resume training from a specific policy:")
    print(f"   python scripts/train_all_subtask_policies.py --start_from <idx>")
    print("=" * 70)


if __name__ == "__main__":
    main()

