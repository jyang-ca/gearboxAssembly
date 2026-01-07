import argparse
import subprocess
import os
import sys
from pathlib import Path

# Subtasks for the assembly task
SUBTASK_SEQUENCE = [
    "approach",
    "grasp",
    "transport_gear_1",
    "transport_gear_2",
    "transport_gear_3",
    "transport_gear_4",
    "transport_carrier",
    "transport_reducer",
]

def parse_args():
    parser = argparse.ArgumentParser(description="GPU-Optimized Sequential Training for RTX 5090")
    parser.add_argument("--num_envs", type=int, default=8192, help="Parallel envs (RTX 5090 handles this well)")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Iterations per subtask (Optimized for large num_envs)")
    parser.add_argument("--start_from", type=int, default=0, help="Subtask index to start from (0-7)")
    parser.add_argument("--only_subtask", type=str, default=None, choices=SUBTASK_SEQUENCE)
    parser.add_argument("--track", action="store_true", help="Enable W&B tracking")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Custom ICD for Vulkan
    env_vars = os.environ.copy()
    env_vars["VK_ICD_FILENAMES"] = "/root/vulkan_fix/custom_icd.json"
    
    subtasks = [args.only_subtask] if args.only_subtask else SUBTASK_SEQUENCE[args.start_from:]
    
    last_checkpoint = None
    
    for idx, subtask in enumerate(subtasks):
        print(f"\n{'='*80}")
        print(f"STEP {idx+1}/{len(subtasks)}: TRAINING SUBTASK -> {subtask.upper()}")
        print(f"{'='*80}\n")
        
        cmd = [
            "/venv/isaaclab/bin/python",
            "scripts/train_long_trajectory_assembly.py",
            f"--subtask={subtask}",
            f"--num_envs={args.num_envs}",
            f"--max_iterations={args.max_iterations}",
            "--headless"
        ]
        
        if args.track:
            cmd.append("--track=True")
            cmd.append(f"--wandb_name=Optimized_{subtask}")
            
        # Optional: Load last checkpoint for chaining (if applicable)
        # Note: In some curriculum setups, you might want to load the previous stage's weights
        # if last_checkpoint:
        #     cmd.append(f"--checkpoint={last_checkpoint}")
            
        try:
            subprocess.run(cmd, env=env_vars, check=True)
            print(f"\n[SUCCESS] Completed {subtask}")
            
            # Find the last saved checkpoint for this subtask
            # (Isaac Lab usually saves in logs/rl_games/...)
            # We'll need to locate it if we want to chain. 
            # For now, we follow the 8-policy design which uses specialized environments.
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Training failed for {subtask}. Error code: {e.returncode}")
            break

if __name__ == "__main__":
    main()
