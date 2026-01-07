"""
Convert BC checkpoint to RL-Games compatible format

BC model uses 28-dim observations (joint states only)
RL-Games expects 69-dim observations (joint + EE + gear info)

This script:
1. Loads BC checkpoint (28-dim input)
2. Creates RL-Games network (69-dim input) with same architecture
3. Copies BC weights to corresponding layers
4. Initializes new layers randomly (for extra obs dimensions)
5. Saves as RL-Games checkpoint format

Usage:
    python scripts/convert_bc_to_rlgames.py \
        --bc_checkpoint checkpoints/bc_pretrained_approach.pth \
        --output checkpoints/bc_for_rl.pth \
        --obs_dim 69
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


def convert_bc_to_rlgames(bc_path: str, output_path: str, rl_obs_dim: int = 69):
    """
    Convert BC checkpoint to RL-Games format.
    
    Strategy:
    - BC network: [28 -> 512 -> 256 -> 128 -> 14]
    - RL network: [69 -> 512 -> 256 -> 128 -> 14]
    - Copy all weights except first layer input weights
    - For first layer: zero-pad BC weights from 28 to 69 dimensions
    """
    
    print(f"Loading BC checkpoint from: {bc_path}")
    bc_checkpoint = torch.load(bc_path, map_location='cpu', weights_only=False)
    bc_state_dict = bc_checkpoint['model']
    
    print("\n=== BC Model Structure ===")
    for key, tensor in bc_state_dict.items():
        print(f"{key}: {tensor.shape}")
    
    # Create RL-Games compatible state dict
    rl_state_dict = {}
    
    # Copy all layers, adjusting first layer for dimension mismatch
    for key, tensor in bc_state_dict.items():
        if 'network.0.weight' in key:
            # First layer weights: (512, 28) -> (512, 69)
            # Strategy: Zero-pad the extra 41 dimensions
            print(f"\n[INFO] Adapting first layer: {tensor.shape} -> (512, {rl_obs_dim})")
            old_weight = tensor  # (512, 28)
            new_weight = torch.zeros(512, rl_obs_dim)
            new_weight[:, :28] = old_weight  # Copy BC weights for first 28 dims
            # Remaining 41 dims initialized to small random values
            nn.init.xavier_uniform_(new_weight[:, 28:], gain=0.01)
            rl_state_dict[key] = new_weight
        else:
            # Copy all other layers as-is
            rl_state_dict[key] = tensor.clone()
    
    # Add RL-Games specific keys (running_mean_std for observation normalization)
    print("\n[INFO] Adding RL-Games specific components...")
    rl_state_dict['running_mean_std.running_mean'] = torch.zeros(rl_obs_dim)
    rl_state_dict['running_mean_std.running_var'] = torch.ones(rl_obs_dim)
    rl_state_dict['running_mean_std.count'] = torch.tensor(1e-4)
    
    # Value network running_mean_std (for value normalization)
    rl_state_dict['value_mean_std.running_mean'] = torch.zeros(1)
    rl_state_dict['value_mean_std.running_var'] = torch.ones(1)
    rl_state_dict['value_mean_std.count'] = torch.tensor(1e-4)
    
    # Create RL-Games checkpoint format
    rl_checkpoint = {
        'model': rl_state_dict,
        'epoch': 0,  # Will be updated by RL training
        'frame': 0,
        'last_mean_rewards': 500.0,  # Optimistic init (BC should do well)
        'optimizer': None,  # Will be re-initialized by RL-Games
        'env_state': None,
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rl_checkpoint, output_path)
    
    print(f"\n[SUCCESS] Saved RL-Games checkpoint to: {output_path}")
    print(f"  Model keys: {len(rl_state_dict)}")
    print(f"  Observation dimension: {rl_obs_dim}")
    print(f"  First layer adapted: (512, 28) -> (512, {rl_obs_dim})")
    print("\nReady for RL fine-tuning!")


def main():
    parser = argparse.ArgumentParser(description="Convert BC to RL-Games format")
    parser.add_argument("--bc_checkpoint", type=str, required=True, help="Path to BC checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output RL-Games checkpoint path")
    parser.add_argument("--obs_dim", type=int, default=69, help="RL environment observation dimension")
    
    args = parser.parse_args()
    
    convert_bc_to_rlgames(args.bc_checkpoint, args.output, args.obs_dim)


if __name__ == "__main__":
    main()

