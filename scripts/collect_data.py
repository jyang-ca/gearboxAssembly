"""Script to collect data for YOLO training."""

import argparse
import sys
import os
import time
from datetime import datetime
import shutil
import gymnasium as gym
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import json
import numpy as np


def log_debug(msg):
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")
    print(msg)

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for YOLO.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-External-Direct-v0", help="Name of the task.")
parser.add_argument("--no_action", action="store_true", default=False, help="Do not apply actions to the robot.")
parser.add_argument("--dataset_dir", type=str, default="dataset_yolo", help="Directory to save dataset.")
parser.add_argument("--collect_steps", type=int, default=50, help="Number of steps to collect.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE ENABLE CAMERAS
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
log_debug("[DEBUG] App Initialized")


try:
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    import isaacsim.core.utils.torch as torch_utils
    from scipy.spatial.transform import Rotation
    import Galaxea_Lab_External.tasks # Register tasks
    log_debug("[DEBUG] Imports Successful")
except Exception as e:
    import traceback
    log_debug(f"[ERROR] Import Failed: {e}\n{traceback.format_exc()}")
    sys.exit(1)

def project_point_to_image(point_w, cam_pos_w, cam_quat_w, intrinsic_matrix):
    """
    Project world point to image pixel.
    Args:
        point_w: (3,) tensor, world position of point
        cam_pos_w: (3,) tensor, world position of camera
        cam_quat_w: (4,) tensor, world orientation of camera (w, x, y, z)
        intrinsic_matrix: (3, 3) tensor
    Returns:
        pixel: (2,) tensor (u, v) or None if behind camera
    """
    # Import locally to ensure AppLauncher has run
    from isaaclab.utils.math import quat_rotate
    
    # World to Camera transform
    # T_wc = [R | t]
    # T_cw = T_wc^-1 = [R^T | -R^T * t]
    
    # Construct Rotation Matrix from Quaternion (w, x, y, z)
    device = cam_pos_w.device
    q = cam_quat_w.unsqueeze(0) # (1, 4)
    
    r_x = quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]], device=device)).squeeze()
    r_y = quat_rotate(q, torch.tensor([[0.0, 1.0, 0.0]], device=device)).squeeze()
    r_z = quat_rotate(q, torch.tensor([[0.0, 0.0, 1.0]], device=device)).squeeze()
    
    R_wc = torch.stack([r_x, r_y, r_z], dim=1) # (3, 3)
    
    t_wc = cam_pos_w
    
    R_cw = R_wc.T
    t_cw = -torch.matmul(R_cw, t_wc)
    
    # Transform point to camera frame
    point_c = torch.matmul(R_cw, point_w) + t_cw
    
    # Convert OpenGL (X Right, Y Up, -Z Forward) to OpenCV (X Right, Y Down, Z Forward)
    # Rotation 180 around X: y -> -y, z -> -z
    point_c[1] = -point_c[1]
    point_c[2] = -point_c[2]
    
    # Check if point is in front of camera
    if point_c[2] <= 0:
        return None, point_c[2]
        
    # Project to image plane
    # P_img = K * P_c
    point_img_hom = torch.matmul(intrinsic_matrix, point_c)
    
    u = point_img_hom[0] / point_img_hom[2]
    v = point_img_hom[1] / point_img_hom[2]
    
    # log_debug(f"[DEBUG-PROJ] Pc: {point_c.tolist()}, UV: {u.item(), v.item()}")
    
    return torch.stack([u, v]), point_c[2]

def get_bbox_2d(obj_corners_w, cam_pos_w, cam_quat_w, intrinsic_matrix, img_width, img_height):
    """
    Get 2D bounding box from 3D object corners.
    """
    u_min, v_min = float('inf'), float('inf')
    u_max, v_max = float('-inf'), float('-inf')
    
    valid_corners = 0
    depths = []
    
    for corner_w in obj_corners_w:
        uv, depth = project_point_to_image(corner_w, cam_pos_w, cam_quat_w, intrinsic_matrix)
        if uv is not None:
            valid_corners += 1
            depths.append(depth)
            
            u, v = uv[0].item(), uv[1].item()
            u_min = min(u_min, u)
            v_min = min(v_min, v)
            u_max = max(u_max, u)
            v_max = max(v_max, v)
            
    if valid_corners < 4: # Object effectively not visible or behind camera
        return None, None
        
    # Clip to image boundaries
    u_min = max(0, u_min)
    v_min = max(0, v_min)
    u_max = min(img_width, u_max)
    v_max = min(img_height, v_max)
    
    if u_max <= u_min or v_max <= v_min:
        return None, None
        
    # Normalize (0-1) for YOLO
    x_center = ((u_min + u_max) / 2) / img_width
    y_center = ((v_min + v_max) / 2) / img_height
    width = (u_max - u_min) / img_width
    height = (v_max - v_min) / img_height
    
    bbox = (x_center, y_center, width, height)
    avg_depth = sum(depths) / len(depths)
    
    return bbox, avg_depth

def main():
    # Setup dataset directories
    dirs = {
        'images': os.path.join(args_cli.dataset_dir, 'images'),
        'labels': os.path.join(args_cli.dataset_dir, 'labels')
    }
    for d in dirs.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    # Debug directory
    dirs['debug'] = os.path.join(args_cli.dataset_dir, 'debug_images')
    os.makedirs(dirs['debug'], exist_ok=True)

        
    
    # Imports that require App initialization
    from isaaclab.utils.math import quat_rotate
    
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # FORCE Disable Lazy Sensor Update to ensure cameras update in headless
    env_cfg.scene.lazy_sensor_update = False
    
    log_debug(f"[DEBUG] Env Config Parsed: {env_cfg}")
    env = gym.make(args_cli.task, cfg=env_cfg, use_action=not args_cli.no_action)
    log_debug(f"[DEBUG] Env Created: {env}")
    
    # Access base environment
    base_env = env.unwrapped
    
    
    # Reset
    log_debug("[DEBUG] Resetting Env...")
    env.reset()
    log_debug("[DEBUG] Reset Done")
    
    # Warmup Rendering
    print("[INFO] Warming up rendering...")
    log_debug("[INFO] Warming up rendering...")
    for i in range(10):
        try:
            if hasattr(base_env, "sim"):
                 base_env.sim.render()
            elif hasattr(env, "sim"):
                 env.sim.render()
        except Exception as e:
            log_debug(f"[WARN] Warmup Render failed: {e}")


    
    
    # Check Camera Prims
    cameras = {
        'head': base_env.head_camera,
        'left': base_env.left_hand_camera,
        'right': base_env.right_hand_camera
    }
    
    for name, cam in cameras.items():
         log_debug(f"[DEBUG] Camera {name} prim path: {cam.cfg.prim_path}")
         # Force update?
         # cam.update(dt=0.01)
    
    # Define object mapping for classes
    # 0: sun_gear (anonymous)
    # 1: ring_gear
    # 2: reducer
    # 3: planetary_carrier (optional)
    
    object_classes = {
        'sun_planetary_gear_1': 0,
        'sun_planetary_gear_2': 0,
        'sun_planetary_gear_3': 0,
        'sun_planetary_gear_4': 0,
        'ring_gear': 1,
        'planetary_reducer': 2
    }
    
    # Approximate bounding box sizes (half-extents)
    # Ref: galaxea_lab_external_env.py OBJECT_RADII
    object_sizes = {
        'ring_gear': 0.1,
        'sun_planetary_gear_1': 0.035,
        'sun_planetary_gear_2': 0.035,
        'sun_planetary_gear_3': 0.035,
        'sun_planetary_gear_4': 0.035,
        'planetary_reducer': 0.04,
        # Height approximation (z)
        'height': 0.02 
    }
    
    step_count = 0
    
    print("[INFO] Starting Data Collection...")
    log_debug("[INFO] Starting Data Collection...")
    
    while simulation_app.is_running():
        log_debug(f"[DEBUG] Loop Step: {step_count}")
        if step_count >= args_cli.collect_steps:
            print("[INFO] Data collection finished.")
            break
            
        with torch.inference_mode():
            # Random actions to move robot around (potential occlusion)
            actions = 2 * torch.rand(env.action_space.shape, device=base_env.device) - 1
            env.step(actions)
            # Force render for cameras
            try:
                if hasattr(base_env, "sim"):
                     base_env.sim.render()
                elif hasattr(env, "sim"): # Fallback
                     env.sim.render()
            except Exception as e:
                log_debug(f"[WARN] Step Render failed: {e}")
            log_debug(f"[DEBUG] Step {step_count}: Env Stepped")
            
            # Get observations
            # obs_dict = base_env.obs['policy'] # Dict from _get_observations
            


            # We need to access cameras directly for matrices
            log_debug(f"[DEBUG] Has front_camera: {hasattr(base_env, 'front_camera')}")
            if hasattr(base_env, 'front_camera'):
                 log_debug(f"[DEBUG] front_camera type: {type(base_env.front_camera)}")

            cameras = {
                'head': base_env.head_camera,
                # 'left': base_env.left_hand_camera,
                # 'right': base_env.right_hand_camera,
                'front': getattr(base_env, 'front_camera', None)
            }
            # Filter None
            cameras = {k: v for k, v in cameras.items() if v is not None}
            
            # Frame ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            for cam_name, cam in cameras.items():
                # Explicitly update camera data if needed (though step should handle it)
                cam.update(dt=base_env.physics_dt)
                
                # Get camera data
                # rgb = cam.data.output["rgb"] # (N, H, W, 3)
                depth_map = cam.data.output["distance_to_image_plane"] # (N, H, W)
                
                # Assume batch size 1 for data collection
                rgb_img = cam.data.output["rgb"][0].cpu().numpy() # (H, W, 3)
                depth_img = depth_map[0].cpu().numpy() # (H, W)
                
                # Check intrinsic/extrinsic
                # These are (N, 3, 3) and (N, 3)
                intrinsic = cam.data.intrinsic_matrices[0]
                cam_pos_w = cam.data.pos_w[0]
                cam_quat_w = cam.data.quat_w_world[0] # (w, x, y, z)
                
                log_debug(f"[DEBUG] Camera '{cam_name}' Pose: {cam_pos_w.tolist()}, {cam_quat_w.tolist()}")
                
                if cam_name == 'front':
                     # Force correct pose (debugging why config isn't applying)
                     cam_pos_w = torch.tensor([1.5, 0.0, 1.5], device=cam.device)
                     # Quat (w, x, y, z) calculated earlier
                     cam_quat_w = torch.tensor([0.6015, 0.3717, 0.3717, 0.6015], device=cam.device)
                     
                     log_debug(f"[DEBUG] front: Overriding Pose to: {cam_pos_w.tolist()}, {cam_quat_w.tolist()}")
                     log_debug(f"[DEBUG] front: Intrinsic: {intrinsic}")
                
                # Check for invalid quaternion
                if torch.isnan(cam_quat_w).any() or torch.norm(cam_quat_w) < 0.1:
                    log_debug(f"[WARN] Invalid quaternion for {cam_name}: {cam_quat_w}. Skipping.")
                    continue
                
                img_h, img_w, _ = rgb_img.shape
                
                labels = [] # List of "class x y w h"
                
                for obj_name, class_id in object_classes.items():
                    obj = base_env.obj_dict[obj_name]
                    obj_pos_w = obj.data.root_state_w[0, :3]
                    obj_quat_w = obj.data.root_state_w[0, 3:7]
                    
                    if torch.isnan(obj_quat_w).any() or torch.norm(obj_quat_w) < 0.1:
                        # log_debug(f"[WARN] Invalid quaternion for object {obj_name}. Skipping object.")
                        continue
                    # Get object pose
                    # root_state_w: (N, 13) [pos, quat, lin_vel, ang_vel]
                    obj_state = obj.data.root_state_w[0]
                    obj_pos_w = obj_state[:3]
                    obj_quat_w = obj_state[3:7] # (w, x, y, z)
                    
                    if cam_name == 'front':
                         log_debug(f"[DEBUG] front: Obj {obj_name} Pos: {obj_pos_w}")

                    # Get radius/height
                    radius = object_sizes.get(obj_name, 0.04) # Default 0.04
                    height = object_sizes.get('height', 0.02)
                    
                    # 8 corners in local frame
                    dx = torch.tensor([-1, 1], device=base_env.device) * radius
                    dy = torch.tensor([-1, 1], device=base_env.device) * radius
                    dz = torch.tensor([-1, 1], device=base_env.device) * height / 2.0
                    
                    x, y, z = torch.meshgrid(dx, dy, dz, indexing='ij')
                    corners_local = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1) # (8, 3)

                    # Rotate corners
                    # quat_rotate expects (N, 4) and (N, 3)
                    corners_w = quat_rotate(obj_quat_w.unsqueeze(0).repeat(8, 1), corners_local) + obj_pos_w

                    # 2D BBox
                    bbox, obj_dist = get_bbox_2d(corners_w, cam_pos_w, cam_quat_w, intrinsic, img_w, img_h)
                    
                    if bbox is not None:
                        if cam_name == 'front':
                             log_debug(f"[DEBUG] front: Obj {obj_name} BBox: {bbox}")

                        # Occlusion Check
                        # Sample depth at center of bbox
                        cx, cy = bbox[0], bbox[1] # Normalized
                        px, py = int(cx * img_w), int(cy * img_h)
                        
                        px = max(0, min(img_w-1, px))
                        py = max(0, min(img_h-1, py))
                        
                        measured_depth = depth_img[py, px]

                        # Tolerance
                        tolerance = 0.05 # 5cm
                        
                        # Ensure measured_depth is a scalar float
                        if isinstance(measured_depth, torch.Tensor):
                            measured_depth = measured_depth.item()
                        elif isinstance(measured_depth, np.ndarray):
                            measured_depth = float(measured_depth)
                            
                        # Ensure obj_dist is a scalar float
                        if isinstance(obj_dist, torch.Tensor):
                            obj_dist = obj_dist.item()

                        if cam_name == 'front':
                             log_debug(f"[DEBUG] front: Obj {obj_name} Dist: {obj_dist} Measured: {measured_depth}")
                        
                        # Note: If measured_depth is 0.0 (invalid), we treat as VISIBLE unless proven occluded
                        if measured_depth > 0 and measured_depth < (obj_dist - tolerance):
                            # Occluded
                            pass
                        else:
                            # Visible
                            labels.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")

                
                # Filenames
                fname = f"{step_count:06d}_{cam_name}"
                
                # Save Image
                img_path = os.path.join(dirs['images'], fname + ".png")
                
                # Convert to uint8 if needed
                if rgb_img.dtype != np.uint8:
                    rgb_img = (rgb_img * 255).astype(np.uint8)
                    
                img_pil = Image.fromarray(rgb_img)
                img_pil.save(img_path)
                log_debug(f"[DEBUG] Step {step_count}: Saved Image {img_path}")
                
                # Create Debug Image with BBoxes
                debug_img_pil = img_pil.copy()
                draw = ImageDraw.Draw(debug_img_pil)
                
                # Save Label
                lbl_path = os.path.join(dirs['labels'], fname + ".txt")
                with open(lbl_path, "w") as f:
                    for lbl in labels:
                        f.write(f"{lbl}\n")
                        
                        # Draw on debug image (convert normalized center/width to pixels)
                        # cx, cy are center. w, h are width/height.
                        # box: [min_x, min_y, max_x, max_y]
                        # pixel_x = cx * img_w
                        # pixel_y = cy * img_h
                        # pixel_w = w * img_w
                        # pixel_h = h * img_h
                        
                        # Parse label string
                        parts = lbl.split()
                        cls_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        p_min_x = (cx - w/2) * img_w
                        p_min_y = (cy - h/2) * img_h
                        p_max_x = (cx + w/2) * img_w
                        p_max_y = (cy + h/2) * img_h
                        
                        draw.rectangle([p_min_x, p_min_y, p_max_x, p_max_y], outline="green", width=2)
                        
                # Save Debug Image
                debug_img_path = os.path.join(dirs['debug'], fname + "_debug.png")
                debug_img_pil.save(debug_img_path)
                    
        step_count += 1
        if step_count % 10 == 0:
            print(f"Collected {step_count} frames...")

    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log_debug(f"[ERROR] Main Failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)
