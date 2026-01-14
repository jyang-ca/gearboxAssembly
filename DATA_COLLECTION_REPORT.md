# Data Collection Debugging & Implementation Report

**Date:** 2026-01-13
**Module:** `Result of collect_data.py` & `galaxea_lab_external` Environment

## 1. Overview
This document summarizes the changes, debugging steps, and solutions implemented to enable robust synthetic data collection for the Gearbox Assembly task. The goal was to generate specific Training Data (RGB Images + YOLO Labels) using Isaac Lab in a headless environment.

## 2. Key Challenges & Solutions

### A. Headless Rendering & Simulation Access
**Problem:**
The script initially failed with `AttributeError: 'OrderEnforcing' object has no attribute 'sim'`.
*   **Context:** `gym.make()` wraps the environment in `OrderEnforcing`, hiding the underlying Isaac Lab `DirectRLEnv` structure.
*   **Result:** `env.sim.render()` calls failed, preventing the simulation from generating visual data.

**Solution:**
*   Implemented a robust accessor for the simulation context:
    ```python
    # collect_data.py
    sim = getattr(env.unwrapped, "sim", getattr(env, "sim", None))
    ```
*   Added `try-except` blocks around the render loop to catch and log specific rendering failures without crashing the entire pipeline.

---

### B. "Invalid Quaternion" & Zero Camera Data
**Problem:**
Cameras (e.g., `head_camera`) returned invalid data (Position: `[0,0,0]`, Quaternion: `[0,0,0,0]`).
*   **Context:** In Isaac Lab's `DirectRLEnv`, defining a `CameraCfg` in the config is not enough. The camera object must be explicitly registered to the `InteractiveScene` to receive physics/rendering updates.
*   **Result:** Generated images were blank or invalid, and object pose estimation relative to cameras failed.

**Solution:**
*   Modified `galaxea_lab_external_env.py` to explicitly register sensors:
    ```python
    self.scene.sensors["head_camera"] = self.head_camera
    self.scene.sensors["front_camera"] = self.front_camera
    ```
*   Set `env_cfg.scene.lazy_sensor_update = False` in `collect_data.py` to enforce sensor updates at every step, which is critical for non-RL data collection loops.

---

### C. Adding a Global "Front" Camera
**Problem:**
Robot-mounted cameras were sometimes obstructed or moved too unpredictably for initial validation. The user requested a static global view.

**Solution:**
1.  **Configuration (`galaxea_lab_external_env_cfg.py`)**:
    *   Added `front_camera_cfg` using `PinholeCameraCfg`.
    *   **Position:** `[1.5, 0.0, 1.5]`
    *   **Target:** `[0.5, 0.0, 1.0]`
2.  **Quaternion Calculation**:
    *   Calculated the Look-At Quaternion to be `[0.6015, 0.3717, 0.3717, 0.6015]` (w, x, y, z).
3.  **Integration**:
    *   Updated `collect_data.py` to iterate over this new `front` camera and save its data alongside others.

---

### D. Empty Labels (Coordinate System Mismatch)
**Problem:**
Images were being saved, but all label files (`.txt`) were empty.
*   **Context:** Debug logs showed that while objects were "Valid", they were detecting as "Behind Camera".
*   **Root Cause:** **Coordinate Convention Mismatch**.
    *   **Isaac Sim / USD (OpenGL):** Forward is **-Z** (Negative Z).
    *   **Standard Vision / OpenCV:** Forward is **+Z** (Positive Z).
*   **Result:** The projection logic filtered out all points with `z < 0`, which meant all points in front of the camera were discarded.

**Solution:**
*   Implemented a coordinate transformation in `collect_data.py` before projection:
    ```python
    # Rotate 180 degrees around X-axis to convert OpenGL to OpenCV
    point_c[1] = -point_c[1] # Flip Y
    point_c[2] = -point_c[2] # Flip Z (now +Z is forward)
    ```
*   Verified that bounding boxes now correctly align with objects in the image.

---

### E. Dependency & Runtime Errors
**Problem:**
Errors such as `ModuleNotFoundError: pxr` and `NameError` for `Rotation`.
*   **Context:** `pxr` (USD) bindings are only available *after* the Isaac Sim application is launched (`AppLauncher`). Importing them at the top level causes crashes.
*   **Context:** `scipy` was missing from the environment.

**Solution:**
*   **Imports:** Moved `isaaclab` and `pxr` related imports **inside** the `main()` function, after `app_launcher.launch()`.
*   **Math:** Replaced `scipy.spatial.transform` dependency with `isaaclab.utils.math.quat_rotate` and native Torch operations to ensure compatibility with the Isaac Lab environment.

## 3. Summary of Files Modified

| File | Change Summary |
| :--- | :--- |
| `scripts/collect_data.py` | Added global camera logic, fixed coordinate systems, robust import handling, bbox debug logging. |
| `galaxea_lab_external_env.py` | Registered cameras to `scene.sensors`, initialized `front_camera`. |
| `galaxea_lab_external_env_cfg.py` | Defined `front_camera` config with calculated pose. |

## 4. How to Run
```bash
./run_collect.sh --collect_steps 50
```
**Output Location:** `dataset_yolo/`
- `images/`: RGB images (e.g., `000001_front.png`)
- `labels/`: YOLO format labels (e.g., `000001_front.txt`)
