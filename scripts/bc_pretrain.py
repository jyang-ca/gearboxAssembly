"""
Behavior Cloning Pre-training from HDF5 Demonstrations

This script:
1. Loads 190 HDF5 demo files
2. Extracts (observation, action) pairs
3. Trains a policy network via supervised learning
4. Saves checkpoint for RL fine-tuning

Usage:
    python scripts/bc_pretrain.py \
        --demo_dir demo_data \
        --num_demos 190 \
        --output_checkpoint checkpoints/bc_pretrained_approach.pth \
        --epochs 100 \
        --batch_size 256
"""

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import os


class DemoDataset(Dataset):
    """Dataset for loading demonstration data from HDF5 files."""
    
    def __init__(self, demo_dir: str, num_demos: int = 190, use_images: bool = False):
        """
        Args:
            demo_dir: Directory containing {1..num_demos}.hdf5 files
            num_demos: Number of demo files to load
            use_images: Whether to include image observations (not recommended for BC)
        """
        self.demo_dir = Path(demo_dir)
        self.num_demos = num_demos
        self.use_images = use_images
        
        # Load all trajectories
        self.observations = []
        self.actions = []
        
        print(f"Loading {num_demos} demonstrations...")
        for i in tqdm(range(1, num_demos + 1)):
            demo_file = self.demo_dir / f"{i}.hdf5"
            if not demo_file.exists():
                print(f"[WARN] File {demo_file} not found, skipping...")
                continue
            
            try:
                with h5py.File(demo_file, 'r') as f:
                    # Extract observations (state-only, no images for BC)
                    left_joint_pos = f['observations/left_arm_joint_pos'][:]
                    left_joint_vel = f['observations/left_arm_joint_vel'][:]
                    left_gripper_pos = f['observations/left_gripper_joint_pos'][:]
                    left_gripper_vel = f['observations/left_gripper_joint_vel'][:]
                    
                    right_joint_pos = f['observations/right_arm_joint_pos'][:]
                    right_joint_vel = f['observations/right_arm_joint_vel'][:]
                    right_gripper_pos = f['observations/right_gripper_joint_pos'][:]
                    right_gripper_vel = f['observations/right_gripper_joint_vel'][:]
                    
                    # Concatenate into observation vector
                    # Shape: (T, 14) for positions + (T, 14) for velocities = (T, 28)
                    obs = np.concatenate([
                        left_joint_pos,   # (T, 6)
                        right_joint_pos,  # (T, 6)
                        left_gripper_pos[:, None],   # (T, 1)
                        right_gripper_pos[:, None],  # (T, 1)
                        left_joint_vel,   # (T, 6)
                        right_joint_vel,  # (T, 6)
                        left_gripper_vel[:, None],   # (T, 1)
                        right_gripper_vel[:, None],  # (T, 1)
                    ], axis=1)  # (T, 28)
                    
                    # Extract actions
                    left_arm_action = f['actions/left_arm_action'][:]
                    right_arm_action = f['actions/right_arm_action'][:]
                    left_gripper_action = f['actions/left_gripper_action'][:]
                    right_gripper_action = f['actions/right_gripper_action'][:]
                    
                    # Concatenate into action vector
                    # Shape: (T, 6) + (T, 6) + (T, 1) + (T, 1) = (T, 14)
                    actions = np.concatenate([
                        left_arm_action,
                        right_arm_action,
                        left_gripper_action[:, None],
                        right_gripper_action[:, None],
                    ], axis=1)  # (T, 14)
                    
                    # Normalize actions to [-1, 1] range (RL training uses this)
                    # Demo actions range: approximately [-2, +3]
                    # Simple clipping for now (can use learned normalization later)
                    actions = np.clip(actions / 2.0, -1.0, 1.0)
                    
                    self.observations.append(obs)
                    self.actions.append(actions)
                    
            except Exception as e:
                print(f"[ERROR] Failed to load {demo_file}: {e}")
                continue
        
        # Concatenate all trajectories
        self.observations = np.concatenate(self.observations, axis=0)  # (N, 28)
        self.actions = np.concatenate(self.actions, axis=0)  # (N, 14)
        
        print(f"Loaded {len(self.observations)} total timesteps from {len(self.observations)//590} demos")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = torch.FloatTensor(self.observations[idx])
        action = torch.FloatTensor(self.actions[idx])
        return obs, action


class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""
    
    def __init__(self, obs_dim: int = 28, action_dim: int = 14, hidden_dims: list = [512, 256, 128]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.network(obs)


def train_bc(args):
    """Main BC training loop."""
    
    # Create dataset and dataloader
    dataset = DemoDataset(args.demo_dir, args.num_demos)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create policy network
    policy = BCPolicy(obs_dim=28, action_dim=14, hidden_dims=[512, 256, 128])
    policy = policy.to(args.device)
    
    # Optimizer and loss
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nStarting BC training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for obs, actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            obs = obs.to(args.device)
            actions = actions.to(args.device)
            
            # Forward pass
            pred_actions = policy(obs)
            loss = criterion(pred_actions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
    
    # Save checkpoint
    output_path = Path(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.epochs,
        'obs_dim': 28,
        'action_dim': 14,
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n[SUCCESS] Saved BC checkpoint to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning Pre-training")
    parser.add_argument("--demo_dir", type=str, default="demo_data", help="Directory with HDF5 demos")
    parser.add_argument("--num_demos", type=int, default=190, help="Number of demos to load")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/bc_pretrained_approach.pth")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()

