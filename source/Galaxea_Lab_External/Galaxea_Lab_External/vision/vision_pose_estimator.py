import torch
import numpy as np
from ultralytics import YOLO
import os

def quat_to_matrix(q):
    """
    Construct Rotation Matrix from Quaternion (w, x, y, z) using torch operations.
    Avoids external dependencies like scipy or pxr-based mathutils inside potential import loops.
    """
    # q: (..., 4) or (4,)
    if q.dim() == 1:
        q = q.unsqueeze(0)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Row-major matrix construction
    #     [1 - 2y^2 - 2z^2, 2xy - 2zw,       2xz + 2yw]
    #     [2xy + 2zw,       1 - 2x^2 - 2z^2, 2yz - 2xw]
    #     [2xz - 2yw,       2yz + 2xw,       1 - 2x^2 - 2y^2]
    
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - x*w)
    
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x**2 + y**2)
    
    # Stack to (N, 3, 3)
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=1)
    
    return R.squeeze()

def project_point_to_image(point_w, cam_pos_w, cam_quat_w, intrinsic_matrix):
    """
    Project world point to image pixel using standard Pinhole model.
    Handles coordinate system conversion (OpenGL -> OpenCV).
    """
    # World to Camera transform
    # T_wc = [R | t] -> T_cw = [R^T | -R^T * t]
    
    R_wc = quat_to_matrix(cam_quat_w)
    t_wc = cam_pos_w
    
    R_cw = R_wc.T
    t_cw = -torch.matmul(R_cw, t_wc)
    
    # Transform point to camera frame
    point_c = torch.matmul(R_cw, point_w) + t_cw
    
    # Convert OpenGL (X Right, Y Up, -Z Forward) to OpenCV (X Right, Y Down, Z Forward)
    # Rotation 180 degrees around X-axis: y -> -y, z -> -z
    point_c[1] = -point_c[1]
    point_c[2] = -point_c[2]
    
    # Check if point is in front of camera (Z > 0 in OpenCV)
    if point_c[2] <= 0:
        return None, point_c[2]
        
    # Project to image plane: P_img = K * P_c
    point_img_hom = torch.matmul(intrinsic_matrix, point_c)
    
    u = point_img_hom[0] / point_img_hom[2]
    v = point_img_hom[1] / point_img_hom[2]
    
    return torch.stack([u, v]), point_c[2]

def get_bbox_2d(obj_corners_w, cam_pos_w, cam_quat_w, intrinsic_matrix, img_width, img_height):
    """
    Compute 2D Bounding Box from 3D object corners in world frame.
    """
    u_min, u_max = float('inf'), float('-inf')
    v_min, v_max = float('inf'), float('-inf')
    valid_corners = 0
    
    for corner_w in obj_corners_w:
        uv, depth = project_point_to_image(corner_w, cam_pos_w, cam_quat_w, intrinsic_matrix)
        if uv is not None:
            valid_corners += 1
            u, v = uv[0].item(), uv[1].item()
            u_min = min(u_min, u)
            v_min = min(v_min, v)
            u_max = max(u_max, u)
            v_max = max(v_max, v)
            
    # Need at least some corners visible 
    if valid_corners < 4: 
        return None
        
    # Clip to image boundaries
    u_min = max(0, u_min)
    v_min = max(0, v_min)
    u_max = min(img_width, u_max)
    v_max = min(img_height, v_max)
    
    if u_max <= u_min or v_max <= v_min:
        return None
        
    # Return pixel coords for usage in agent (not normalized YOLO)
    # can clarify if user wants normalized
    return (u_min, v_min, u_max, v_max)

class VisionPoseEstimator:
    """
    Placeholder/Wrapper for Vision Logic.
    This class can simulate detections (Oracle) or run actual inference (if model provided).
    """
    def __init__(self, env):
        self.env = env
        # Handle wrapped environments (e.g., RecordVideo)
        self.device = getattr(env, 'device', getattr(env.unwrapped, 'device', 'cuda:0'))
        
        # Load YOLO model
        weights_path = "/root/gearboxAssembly/gearbox_training/yolov8n_run2/weights/best.pt"
        if os.path.exists(weights_path):
            print(f"[INFO] VisionPoseEstimator: Loading YOLO model from {weights_path}")
            self.model = YOLO(weights_path)
        else:
            print(f"[WARN] VisionPoseEstimator: YOLO weights not found at {weights_path}. Vision will fail.")
            self.model = None

        self.class_map = {
            0: 'sun_planetary_gear',
            1: 'ring_gear',
            2: 'planetary_reducer',
            3: 'planetary_carrier'
        }
        
        # History Buffer for Smoothing (Improved settings)
        self.pose_history = {} # name -> list of poses (or None)
        self.history_len = 20  # Increased from 10 to 20 for more robust estimation
        self.min_confidence = 10  # Increased from 3 to 10 (50% detection rate required)
        
    def get_oracle_detections(self, camera_name):
        """
        Simulate object detection using Ground Truth state + Projection.
        Returns list of {label, bbox, score}
        """
        # Access scene elements from env
        # Note: robust access usually requires unwrapped or direct attribute access if standard gym
        scene = getattr(self.env.unwrapped, "scene", getattr(self.env, "scene", None))
        if not scene:
             return []

        # Get Camera
        if camera_name not in scene.sensors:
            return []


        
        cam = scene.sensors[camera_name]
        # Ensure data is fresh
        # cam.update(dt=self.env.physics_dt) 
        # (Assuming env.step() updates sensors if lazy_sensor_update=False)

        # Matrices
        intrinsic = cam.data.intrinsic_matrices[0]
        cam_pos_w = cam.data.pos_w[0]
        cam_quat_w = cam.data.quat_w_world[0]
        
        # Override for Front Camera (Updated to match collect_data.py)
        if camera_name == 'front_camera':
             cam_pos_w = torch.tensor([1.2, 0.0, 1.8], device=self.device)
             cam_quat_w = torch.tensor([0.6768, 0.2049, 0.2049, 0.6768], device=self.device)



        img_h, img_w = cam.data.output["rgb"].shape[1:3]
        
        detections = []
        
        # Iterate defined objects (hardcoded list matching collect_data.py)
        # In a real app, this list should be config-driven
        objects = {
            'ring_gear': 0, 'sun_planetary_gear_1': 1, 'sun_planetary_gear_2': 1,
            'sun_planetary_gear_3': 1, 'sun_planetary_gear_4': 1,
            'planetary_carrier': 2, 'planetary_reducer': 3
        }
        
        from isaaclab.utils.math import quat_apply
        
        obj_dict = getattr(self.env.unwrapped, "obj_dict", {})

        for obj_name, cls_id in objects.items():
            obj = None
            if obj_name in obj_dict:
                obj = obj_dict[obj_name]
            elif scene and obj_name in scene.rigid_objects:
                 obj = scene.rigid_objects[obj_name]
                 
            if obj is None:
                continue

            
            obj_pos_w = obj.data.root_pos_w[0]
            obj_quat_w = obj.data.root_quat_w[0]
            
            # Simple BBox approximation (Cube of 5cm)
            # Better: get actual mesh bounds. For now, use same approximation as collect_data
            half_size = 0.025 
            corners_local = torch.tensor([
                [-half_size, -half_size, -half_size], [half_size, -half_size, -half_size],
                [-half_size,  half_size, -half_size], [half_size,  half_size, -half_size],
                [-half_size, -half_size,  half_size], [half_size, -half_size,  half_size],
                [-half_size,  half_size,  half_size], [half_size,  half_size,  half_size]
            ], device=self.device)
            
            # Transform to World
            # corners_w = R * corners_l + p
            corners_w = quat_apply(obj_quat_w.repeat(8, 1), corners_local) + obj_pos_w
            
            bbox = get_bbox_2d(corners_w, cam_pos_w, cam_quat_w, intrinsic, img_w, img_h)
            
            if bbox:


                detections.append({
                    'label': obj_name,
                    'class_id': cls_id,
                    'bbox': bbox, # (u_min, v_min, u_max, v_max)
                    'score': 1.0
                })
                
        return detections
    
    def get_3d_poses_oracle(self, camera_name='front_camera'):
        """
        Get 3D poses of all objects using Ground Truth (Oracle mode).
        This wraps GT data in a vision interface for easy replacement with real vision later.
        
        Args:
            camera_name: Camera to use (currently unused, GT is global)
        
        Returns:
            dict: {
                'object_name': {
                    'position': torch.Tensor([x, y, z]),
                    'orientation': torch.Tensor([w, x, y, z]),
                    'available': bool
                }
            }
        """
        scene = getattr(self.env.unwrapped, "scene", getattr(self.env, "scene", None))
        if not scene:
            return {}
        
        poses = {}
        
        # List of all objects we need to track
        object_names = [
            'ring_gear',
            'planetary_carrier', 
            'planetary_reducer',
            'sun_planetary_gear_1',
            'sun_planetary_gear_2',
            'sun_planetary_gear_3',
            'sun_planetary_gear_4'
        ]
        
        for obj_name in object_names:
            if obj_name in scene:
                obj = scene[obj_name]
                # Get GT state: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, ...]
                state = obj.data.root_state_w[0]  # [0] for single environment
                
                poses[obj_name] = {
                    'position': state[:3].clone(),  # [x, y, z]
                    'orientation': state[3:7].clone(),  # [w, x, y, z]
                    'available': True
                }
            else:
                poses[obj_name] = {
                    'position': None,
                    'orientation': None,
                    'available': False
                }
        
        return poses
    
    def estimate_3d_position_from_bbox(self, bbox, depth_map, intrinsic, cam_pos_w, cam_quat_w):
        """
        Estimate 3D position from 2D bbox and depth map.
        
        Args:
            bbox: tuple (u_min, v_min, u_max, v_max) in pixels
            depth_map: torch.Tensor (H, W) - distance to image plane
            intrinsic: torch.Tensor (3, 3) - camera intrinsic matrix
            cam_pos_w: torch.Tensor (3,) - camera position in world frame
            cam_quat_w: torch.Tensor (4,) - camera quaternion [w, x, y, z]
        
        Returns:
            position_world: torch.Tensor (3,) - 3D position in world frame
        """
        import torch
        
        # 1. Bbox center
        u_center = (bbox[0] + bbox[2]) / 2.0
        v_center = (bbox[1] + bbox[3]) / 2.0
        
        # 2. Get median depth in bbox region with outlier filtering (IMPROVED)
        u_min, v_min, u_max, v_max = [int(x) for x in bbox]
        u_min = max(0, u_min)
        v_min = max(0, v_min)
        u_max = min(depth_map.shape[1], u_max)
        v_max = min(depth_map.shape[0], v_max)
        
        if u_max <= u_min or v_max <= v_min:
            # Invalid bbox, use center point
            u_int, v_int = int(u_center), int(v_center)
            depth = depth_map[v_int, u_int]
        else:
            bbox_region = depth_map[v_min:v_max, u_min:u_max]
            
            # Filter out invalid depths (0 or too far)
            valid_depths = bbox_region[(bbox_region > 0.1) & (bbox_region < 3.0)]
            
            if len(valid_depths) > 10:
                # Use IQR method to remove outliers
                q25 = torch.quantile(valid_depths, 0.25)
                q75 = torch.quantile(valid_depths, 0.75)
                iqr = q75 - q25
                
                # Keep values within 1.5 * IQR range
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                filtered_depths = valid_depths[(valid_depths >= lower_bound) & (valid_depths <= upper_bound)]
                
                if len(filtered_depths) > 0:
                    depth = torch.median(filtered_depths)
                else:
                    depth = torch.median(valid_depths)
            elif len(valid_depths) > 0:
                # Not enough data for IQR, just use median of valid depths
                depth = torch.median(valid_depths)
            else:
                # Fallback to simple median
                depth = torch.median(bbox_region)
        
        # 3. Pixel → Camera Frame (OpenCV convention: X-right, Y-down, Z-forward)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (u_center - cx) * depth / fx
        y_cam = (v_center - cy) * depth / fy
        z_cam = depth
        
        pos_cam = torch.tensor([x_cam, y_cam, z_cam], device=self.device)
        
        # 4. Convert OpenCV to OpenGL coordinate system
        # OpenCV: X-right, Y-down, Z-forward
        # OpenGL: X-right, Y-up, Z-backward
        pos_cam_gl = torch.tensor([pos_cam[0], -pos_cam[1], -pos_cam[2]], device=self.device)
        
        # 5. Camera Frame → World Frame
        from isaaclab.utils.math import quat_rotate
        pos_world = cam_pos_w + quat_rotate(cam_quat_w.unsqueeze(0), pos_cam_gl.unsqueeze(0))[0]
        
        return pos_world
    
    def estimate_orientation_heuristic(self, obj_name, position_world):
        """
        Estimate orientation using heuristic (simplified approach).
        Assumes objects are upright on table.
        
        Args:
            obj_name: Name of object
            position_world: torch.Tensor (3,) - estimated 3D position
        
        Returns:
            orientation: torch.Tensor (4,) - quaternion [w, x, y, z]
        """
        import torch
        
        # Default: upright orientation (identity quaternion)
        # This assumes gears are lying flat on table
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Could add object-specific adjustments here
        # For example, ring_gear might have different default orientation
        
        return orientation
    
    def get_yolo_detections(self, camera_name):
        """
        Run YOLO inference on the camera image.
        """
        if self.model is None:
            return []

        # Get Camera Image
        scene = getattr(self.env.unwrapped, "scene", getattr(self.env, "scene", None))
        if not scene or camera_name not in scene.sensors:
            return []
            
        cam = scene.sensors[camera_name]
        
        # Image is (H, W, 3) tensor, RGB
        image_tensor = cam.data.output["rgb"][0]
        
        # Convert to numpy for YOLO
        # Note: Ultralytics expects BGR or RGB numpy. 
        # Isaac Lab provides float or uint8? usually uint8 0-255 if it's the standard rgb output.
        # Let's check type if needed, but usually it's fine.
        image_np = image_tensor.cpu().numpy()
        
        # Run inference
        results = self.model.predict(image_np, verbose=False, conf=0.5)
        
        detections = []
        
        # Map generic class names to specific instances
        # We need to handle multiple sun gears.
        sun_gears = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist() # [x1, y1, x2, y2]
                
                label_base = self.class_map.get(cls_id, 'unknown')
                
                bbox = tuple(xyxy) # (u_min, v_min, u_max, v_max)
                
                # --- Heuristic Correction ---
                # Fix confusion between ring_gear (Large) and sun_planetary_gear (Small)
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                AREA_THRESHOLD = 4000 # approx 63x63 pixels
                
                if label_base == 'ring_gear' and area < AREA_THRESHOLD:
                    label_base = 'sun_planetary_gear'
                    cls_id = 0
                elif label_base == 'sun_planetary_gear' and area > AREA_THRESHOLD:
                    label_base = 'ring_gear'
                    cls_id = 1
                # ----------------------------

                if label_base == 'sun_planetary_gear':
                    sun_gears.append({'bbox': bbox, 'score': conf})
                else:
                    # ring_gear or planetary_reducer
                    detections.append({
                        'label': label_base,
                        'class_id': cls_id,
                        'bbox': bbox,
                        'score': conf
                    })
        
        # Assign sun gears to specific IDs (1..4)
        # Simple assignment for now
        for i, gear_info in enumerate(sun_gears):
            if i >= 4: break # Max 4 gears
            detections.append({
                'label': f"sun_planetary_gear_{i+1}",
                'class_id': 0,
                'bbox': gear_info['bbox'],
                'score': gear_info['score']
            })
            
        return detections




    def get_3d_poses(self, camera_name='front_camera'):
        """
        Get 3D poses using REAL vision (Depth + 2D bbox).
        """
        import torch
        
        # 1. Get 2D detections (YOLO)
        detections = self.get_yolo_detections(camera_name)
        
        # 2. Get camera data
        scene = getattr(self.env.unwrapped, "scene", getattr(self.env, "scene", None))
        if not scene: # or camera_name not in scene.sensors:
             # Fallback if scene not ready or camera issue
             return {}
             
        # Fallback if camera not in sensors (sometimes happened in debug)
        if camera_name not in scene.sensors:
             return {}

        cam = scene.sensors[camera_name]
        
        # Get depth map and camera parameters
        depth_map = cam.data.output["distance_to_image_plane"][0]
        intrinsic = cam.data.intrinsic_matrices[0]
        cam_pos_w = cam.data.pos_w[0]
        cam_quat_w = cam.data.quat_w_world[0]

        # Override for Front Camera (Consistency with get_oracle_detections)
        if camera_name == 'front_camera':
             cam_pos_w = torch.tensor([1.2, 0.0, 1.8], device=self.device)
             
             # Updated quaternion for eye=[1.2, 0, 1.8], target=[0, 0, 0]
             # [0.6768, 0.2049, 0.2049, 0.6768]
             cam_quat_w = torch.tensor([0.6768, 0.2049, 0.2049, 0.6768], device=self.device)

        
        # 3. Estimate 3D poses from 2D detections + depth
        current_poses = {}
        
        # Process Detections
        for det in detections:
            obj_name = det['label']
            bbox = det['bbox']
            
            try:
                # Estimate position from bbox and depth
                position = self.estimate_3d_position_from_bbox(
                    bbox, depth_map, intrinsic, cam_pos_w, cam_quat_w
                )
                
                # Filter invalid positions (e.g. at origin or below table)
                if torch.norm(position) < 0.1 or position[2] < 0.5:
                     continue

                # Estimate orientation (heuristic for now)
                orientation = self.estimate_orientation_heuristic(obj_name, position)
                
                current_poses[obj_name] = {
                    'position': position,
                    'orientation': orientation
                }
            except Exception as e:
                print(f"[WARN] Failed to estimate pose for {obj_name}: {e}")
                
        # 4. Update History and Smooth
        self.update_history(current_poses)
        smoothed_poses = self.get_smoothed_poses()
        
        # CRITICAL: Do NOT use GT data for planetary_carrier!
        # YOLO model is trained to detect planetary_carrier (class 3)
        # If not detected, it means vision needs more time to accumulate detections
        # The rule_policy will use cached pose or wait for valid detection

        return smoothed_poses

    def update_history(self, current_poses):
        """Update pose history buffer."""
        for name in self.class_map.values(): # Ensure all keys exist
             if name == 'unknown': continue
             # Handle 4 sun gears
             if name == 'sun_planetary_gear':
                  for i in range(1, 5):
                       key = f"sun_planetary_gear_{i}"
                       if key not in self.pose_history: self.pose_history[key] = []
             else:
                  if name not in self.pose_history: self.pose_history[name] = []

        # Iterate current detections and append
        # Note: YOLO might return multiple 'ring_gear' if confused, but here we handled that in get_yolo_detections
        # The keys in current_poses are already unique strings like 'sun_planetary_gear_1'
        
        for name, pose in current_poses.items():
            if name not in self.pose_history:
                self.pose_history[name] = []
            self.pose_history[name].append(pose)
            
        # Manage buffer size 
        # Also append None if object not seen? No, we filter based on recent count.
        # Actually, simpler: just keep list of recent Valid poses. 
        # But we need time window.
        # Let's simple buffer: append (timestamp, pose). 
        # Or just append Pose, and clear if not seen? 
        # Better: Sliding window of last N steps. If object not seen, append None.
        
        for name in self.pose_history.keys():
            if name in current_poses:
                 # Already appended above? No, loop above iterates current_poses
                 pass 
            else:
                 self.pose_history[name].append(None)
            
            # Truncate
            if len(self.pose_history[name]) > self.history_len:
                self.pose_history[name].pop(0)
                
    def get_smoothed_poses(self):
        """Compute median pose from history with outlier removal (IMPROVED)."""
        smoothed = {}
        for name, history in self.pose_history.items():
            valid_poses = [p for p in history if p is not None]
            
            # Confidence check: needs to be seen in at least min_confidence frames
            if len(valid_poses) < self.min_confidence:
                smoothed[name] = {'available': False, 'position': None, 'orientation': None}
                continue
            
            # Compute Median Position with IQR outlier removal
            positions = torch.stack([p['position'] for p in valid_poses])
            
            # Apply IQR-based outlier filtering if we have enough samples
            if len(positions) >= 10:
                # Calculate quartiles for each dimension
                q25 = torch.quantile(positions, 0.25, dim=0)
                q75 = torch.quantile(positions, 0.75, dim=0)
                iqr = q75 - q25
                
                # Define outlier bounds (1.5 * IQR)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                # Filter out outliers (keep only inliers for all dimensions)
                valid_mask = ((positions >= lower_bound) & (positions <= upper_bound)).all(dim=1)
                filtered_positions = positions[valid_mask]
                
                if len(filtered_positions) >= self.min_confidence // 2:
                    # Use filtered positions if we still have enough data
                    median_pos, _ = torch.median(filtered_positions, dim=0)
                else:
                    # Fall back to using all positions if too many were filtered
                    median_pos, _ = torch.median(positions, dim=0)
            else:
                # Not enough samples for outlier detection, use simple median
                median_pos, _ = torch.median(positions, dim=0)
            
            # Orientation: take the most recent valid one (heuristic)
            last_valid_quat = valid_poses[-1]['orientation']
            
            smoothed[name] = {
                'available': True,
                'position': median_pos,
                'orientation': last_valid_quat
            }
        return smoothed
