# Multi-Vehicle Tracking Using Kalman Filter & Hungarian Association

**Institute:** Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
**Department:** Department of Computer Science and Engineering  
**Course:** CS359 — Artificial Intelligence for Robotics  
**Lab Assignment:** 10  
**Academic Term:** Autumn 2026 / E3S1  

---

## 1. Project Overview & Objective

The objective of this laboratory assignment is to design, implement, and validate a robust, real-time **Multi-Vehicle Tracking System** that processes arbitrary traffic video feeds. Rather than relying on external black-box tracking libraries (e.g., DeepSORT, ByteTrack wrappers), this project implements an **explicit, per-vehicle 2D Constant-Velocity Kalman Filter** combined with a **custom Multi-Object Tracker (MOT) manager** utilizing the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) for global bipartite data association.

### Key Capabilities
- **Universal Video Input:** Ingests any user-specified video format (`.mp4`, `.avi`, `.mov`, `.mkv`), automatically discovering frame dimensions, FPS, and frame count.
- **Deep Vehicle Detection:** Uses a pretrained, lightweight YOLOv8 nano model (`yolov8n.pt`) to detect traffic classes (`car`, `motorcycle`, `bus`, `truck`) without requiring custom model training.
- **Dedicated Kalman Filtering:** Instantiates an independent OpenCV `cv2.KalmanFilter` for every tracked vehicle with state vector $[x, y, v_x, v_y]^T$.
- **Global Data Association:** Solves optimal 1-to-1 matching between predicted track states and newly detected bounding boxes using the Hungarian algorithm, strictly preventing multiple detections from updating the same track.
- **Occlusion Handling & Coasting:** Seamlessly maintains vehicle tracks when detections temporarily drop (e.g., under overpasses or behind trees) by advancing the state in prediction-only mode and recovering the identity once the vehicle reappears.
- **Dual Visual Styling:** Visually distinguishes actively detected vehicles (solid emerald green bounding boxes) from coasted/predicted vehicles (amber corner-bracketed boxes with `[PREDICTED: +Nf]` tag).
- **Motion History Trails:** Renders smooth fading trajectory trails illustrating historical vehicle movement.
- **Video Rendering & Export:** Saves the annotated video stream preserving native resolution and frame rate with safe resource cleanup.

---

## 2. Theoretical Formulation & Kalman Filter Mathematics

### 2.1 State Representation & Kinematics
Each vehicle's planar motion is modeled as a 2D constant-velocity dynamical system in discrete time with time step $\Delta t = 1.0$ frame:

$$\mathbf{x}_k = \begin{bmatrix} x_k \\ y_k \\ v_{x, k} \\ v_{y, k} \end{bmatrix} \in \mathbb{R}^4$$

- $(x_k, y_k)$: Center coordinates of the vehicle bounding box in image pixel space.
- $(v_{x, k}, v_{y, k})$: Horizontal and vertical velocity in pixels per frame.

### 2.2 Measurement Vector
The detector provides the observed bounding box $[x_1, y_1, x_2, y_2]$, from which the center observation is derived:

$$\mathbf{z}_k = \begin{bmatrix} z_{x, k} \\ z_{y, k} \end{bmatrix} = \begin{bmatrix} x_1 + \frac{x_2 - x_1}{2} \\ y_1 + \frac{y_2 - y_1}{2} \end{bmatrix} \in \mathbb{R}^2$$

---

### 2.3 System Matrices

#### 1. State Transition Matrix ($F$)
Projects the kinematic state forward by time step $\Delta t$:

$$F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$x_{k|k-1} = x_{k-1|k-1} + v_{x, k-1|k-1}\Delta t$$
$$y_{k|k-1} = y_{k-1|k-1} + v_{y, k-1|k-1}\Delta t$$

#### 2. Measurement Matrix ($H$)
Maps the 4D state space into the 2D observation space:

$$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

$$\hat{\mathbf{z}}_k = H \mathbf{x}_{k|k-1} = \begin{bmatrix} x_{k|k-1} \\ y_{k|k-1} \end{bmatrix}$$

#### 3. Process Noise Covariance Matrix ($Q$)
Models unmodeled physical perturbations (acceleration, braking, steering maneuvers):

$$Q = \begin{bmatrix} \sigma_p^2 & 0 & 0 & 0 \\ 0 & \sigma_p^2 & 0 & 0 \\ 0 & 0 & \sigma_v^2 & 0 \\ 0 & 0 & 0 & \sigma_v^2 \end{bmatrix}$$

- Configured as $\sigma_p = 1.0$ pixel, $\sigma_v = 2.0$ pixels/frame.

#### 4. Measurement Noise Covariance Matrix ($R$)
Models detector bounding box jitter, scale fluctuations, and edge ambiguity:

$$R = \begin{bmatrix} \sigma_m^2 & 0 \\ 0 & \sigma_m^2 \end{bmatrix}$$

- Configured as $\sigma_m = 4.0$ pixels ($\sigma_m^2 = 16.0$).

#### 5. Estimation Error Covariance Matrix ($P$)
Quantifies confidence in the current state estimate. Initialized at track birth as:

$$P_0 = \begin{bmatrix} 10.0 & 0 & 0 & 0 \\ 0 & 10.0 & 0 & 0 \\ 0 & 0 & 100.0 & 0 \\ 0 & 0 & 0 & 100.0 \end{bmatrix}$$

*(Low initial position uncertainty because it is seeded by a direct measurement; high velocity uncertainty because velocity cannot be inferred from a single snapshot).*

---

### 2.4 Kalman Filter Recursive Algorithm

```
                  ┌─────────────────────────────────────────┐
                  │          Time Update (Predict)          │
                  │  x_{k|k-1} = F * x_{k-1|k-1}            │
                  │  P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q  │
                  └────────────────────┬────────────────────┘
                                       │
                         Is detection associated?
                                      / \
                                Yes /     \ No (Missing detection)
                                   /       \
  ┌─────────────────────────────────────┐  ┌────────────────────────────────────┐
  │     Measurement Update (Correct)    │  │       Coasting (Prediction-Only)   │
  │  y_k = z_k - H * x_{k|k-1}          │  │  x_{k|k} = x_{k|k-1}               │
  │  S_k = H * P_{k|k-1} * H^T + R      │  │  P_{k|k} = P_{k|k-1}               │
  │  K_k = P_{k|k-1} * H^T * S_k^(-1)   │  │  (Uncertainty covariance grows;    │
  │  x_{k|k} = x_{k|k-1} + K_k * y_k    │  │   state continues kinematic path)  │
  │  P_{k|k} = (I - K_k * H) * P_{k|k-1}│  └────────────────────────────────────┘
  └─────────────────────────────────────┘
```

---

## 3. Multi-Vehicle Tracking & Data Association

Data association matches $M$ active predicted tracks with $N$ new frame detections:

1. **Cost Matrix Construction:**  
   For every track $i \in \{1 \dots M\}$ and detection $j \in \{1 \dots N\}$:
   $$C_{i, j} = \|\mathbf{p}_i^{\text{pred}} - \mathbf{c}_j^{\text{det}}\|_2 + \lambda_{\text{IoU}} (1 - \text{IoU}(\mathbf{B}_i^{\text{pred}}, \mathbf{B}_j^{\text{det}})) \cdot D_{\text{max}}$$
2. **Global Optimization (Hungarian Algorithm):**  
   Find the optimal bijective assignment minimizing total assignment cost:
   $$\min_{\pi} \sum_{i} C_{i, \pi(i)}$$
   Solved in $\mathcal{O}(V^3)$ using `scipy.optimize.linear_sum_assignment`.
3. **Gating Distance Filter:**  
   Any matched pair with Euclidean center distance exceeding $D_{\text{max}} = 85.0$ pixels is rejected.
4. **Lifecycle Transitions:**
   - **Matched Tracks:** Kalman Filter is corrected; bounding box smoothed; `time_since_update` reset to 0; state set to `TRACKED`.
   - **Unmatched Active Tracks:** Kalman Filter coasted; `time_since_update` incremented by 1; state set to `COASTED`.
   - **Unmatched Detections:** Instantiates a new `VehicleTrack` with a persistent unique ID.
   - **Pruning:** Tracks with `time_since_update > max_age` (default 20 frames) are permanently deleted.

---

## 4. Project Directory Structure

```
vehicle_tracking_lab/
├── main.py                     # CLI entry point and video processing pipeline
├── detector.py                 # YOLOv8 vehicle detector wrapper
├── kalman_filter.py            # Dedicated 2D Constant-Velocity Kalman Filter
├── tracker.py                  # Multi-vehicle tracker & Hungarian association manager
├── visualization.py            # HUD, bounding boxes, labels, and trajectory rendering
├── generate_sample_video.py    # Synthetic traffic video generator
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── input_videos/               # Input traffic videos (e.g. sample_traffic.mp4, highway_traffic.mp4)
├── output_videos/              # Saved tracked videos and screenshots
└── tests/                      # Automated unit and integration test suite
    ├── __init__.py
    ├── test_kalman_filter.py   # Unit tests for Kalman prediction, correction, coasting
    ├── test_tracker.py         # Unit tests for Hungarian association, 1-to-1 bijection
    └── test_pipeline.py        # Integration tests for video I/O and CLI validation
```

---

## 5. Installation & Setup Guide

### 5.1 System Requirements
- **OS:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+
- **Python:** Python 3.9 through 3.13
- **Hardware:** Standard CPU (runs comfortably at 15–30 FPS on multi-core CPU; GPU optional via CUDA)

### 5.2 Environment Setup (Windows PowerShell)

```powershell
# Navigate to project directory
cd vehicle_tracking_lab

# Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 5.3 Environment Setup (Ubuntu / Linux)

```bash
# Navigate to project directory
cd vehicle_tracking_lab

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 6. How to Run

### 6.1 Quick Start with Included Highway Video

```powershell
python main.py --input input_videos/highway_traffic.mp4 --output output_videos/highway_tracked.mp4
```

### 6.2 Running on Any Custom Video

```powershell
python main.py --input "D:\MyVideos\traffic_crossroad.mp4" --output "output_videos\result.mp4"
```

### 6.3 Command-Line Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-i`, `--input` | `str` | `None` | Path to input traffic video file (.mp4, .avi, etc.) |
| `-o`, `--output` | `str` | `output_videos/tracked_output.mp4` | Path to save annotated video |
| `--conf` | `float` | `0.20` | Detection confidence threshold (0.01 to 0.99; default 0.20 captures distant/shaded vehicles) |
| `--imgsz` | `int` | `0` | YOLO inference size (0 = auto-adaptive: 1280 for 1080p/1440p/4K, 640 for SD) |
| `--max-distance` | `float` | `0.0` | Max Hungarian association distance (0.0 = auto-scaled to video resolution) |
| `--max-age` | `int` | `25` | Max consecutive missed frames before track deletion |
| `--min-hits` | `int` | `2` | Min detections required to confirm a track and count it in total unique vehicles |
| `--model` | `str` | `yolov8n.pt` | YOLO model weights (`yolov8n.pt`, `yolov8s.pt`, etc.) |
| `--device` | `str` | `auto` | Inference device (`cpu`, `cuda`, or `auto`) |
| `--no-agnostic-nms` | flag | `False` | Disable class-agnostic NMS (merges overlapping duplicate vehicle boxes) |
| `--no-show` | flag | `False` | Disable GUI display window (recommended for headless or batch runs) |
| `--no-trajectory`| flag | `False` | Disable rendering of movement trajectory trails |
| `--no-save` | flag | `False` | Do not write video output to disk |
| `--max-frames` | `int` | `0` | Limit frames to process (0 = full video) |

### 6.4 Examples

```powershell
# Run on high-resolution Quad HD real.mp4 with live GUI:
python main.py --input input_videos/real.mp4 --output output_videos/result.mp4

# Run on real.mp4 with custom confidence and maximum frames:
python main.py --input input_videos/real.mp4 --output output_videos/result.mp4 --conf 0.18 --max-frames 200

# High-confidence highway tracking in headless mode:
python main.py --input input_videos/highway_traffic.mp4 --conf 0.35 --no-show

# Generate synthetic traffic test video:
python generate_sample_video.py
```

---

## 7. Automated Test Suite

Run all automated unit and integration tests:

```powershell
python -m pytest -v tests
```

### Test Coverage Summary:
- `test_kalman_initialization`: Validates matrix shapes ($F, H, Q, R, P, x$).
- `test_kalman_prediction`: Validates forward constant-velocity kinematic projection.
- `test_kalman_correction`: Confirms state convergence towards observations.
- `test_kalman_covariance_behavior`: Validates covariance expansion during prediction and reduction upon measurement update.
- `test_kalman_coasting_prediction_only`: Confirms coasting without mathematical divergence.
- `test_track_creation`: Confirms track initialization and ID assignment.
- `test_multi_vehicle_association_and_distinct_ids`: Verifies distinct ID maintenance across parallel tracks.
- `test_one_to_one_matching_guarantee`: Ensures Hungarian assignment prevents double-booking.
- `test_temporary_missing_detection_coasting_and_recovery`: Verifies track survival and re-acquisition after 3 consecutive missed frames.
- `test_track_deletion_after_max_age`: Verifies track deletion after $N > \text{max\_age}$ missed frames.
- `test_empty_detections_handling`: Verifies crash-free handling of frames with zero vehicles.
- `test_invalid_video_path_raises_error`: Validates error handling for missing files.
- `test_synthetic_video_generation_and_reading`: Validates end-to-end video I/O.

---

## 8. Interpreting the Visual Output

1. **Emerald Green Bounding Box (`TRACKED`):** The vehicle was successfully detected by YOLO and its position was filtered and smoothed by the Kalman Filter in the current frame.
2. **Amber Bracketed Bounding Box (`COASTED`):** The vehicle detection was missed (e.g., occlusion or detector miss). The bounding box position is predicted purely by the Kalman Filter kinematics.
3. **Cyan Crosshair:** The exact Kalman-predicted center point $(\hat{x}, \hat{y})$.
4. **Trajectory Trail:** Colored trail displaying the vehicle's historical motion over past frames.
5. **HUD Banner:** Real-time metrics showing Frame Number, Video FPS, Total Active Tracks, Currently Detected count, Coasted count, and Cumulative Vehicles seen.

---

## 9. Common Errors & Troubleshooting

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `FileNotFoundError: Input video file not found` | The path specified in `--input` does not exist. | Verify path spelling or use `input_videos/highway_traffic.mp4`. |
| `NameError: name 'np' is not defined` | Missing NumPy import. | Ensure `import numpy as np` is present in module headers. |
| `pytest: The term 'pytest' is not recognized` | Python Scripts directory is not on Windows system `PATH`. | Run tests using `python -m pytest -v tests`. |
| Low Processing FPS (< 5 FPS) | Running large YOLO model on high-resolution video using CPU. | Use default `yolov8n.pt` (nano) or resize video frames. |
| Track ID switches during crossover | Kinematic Kalman Filter lacks visual appearance features. | Increase `--max-distance` or reduce vehicle velocity disparity; use appearance Re-ID for extreme crossovers. |

---

## 10. Questions & Answers

### Q1: What is a Kalman Filter?
**Ans:** A Kalman Filter is an optimal recursive Bayesian state estimation algorithm that estimates the true hidden state of a dynamic system from a series of noisy measurements observed over time. It operates in two phases: Prediction and Measurement Update (Correction).

### Q2: Why is the Kalman Filter useful for vehicle tracking?
**Ans:** 
1. It filters out detector bounding box jitter and localization noise.
2. It estimates unobserved vehicle velocities ($v_x, v_y$).
3. It predicts vehicle positions during momentary occlusions (tunnels, trees, passing vehicles).
4. It provides predicted positions for data association in the next frame.

### Q3: What is the difference between Prediction and Correction?
**Ans:** 
- **Prediction:** Uses the physical motion model ($F$) and process noise ($Q$) to project the prior state estimate forward in time before any new measurement arrives.
- **Correction:** Fuses the new sensor measurement ($z$) with the predicted estimate using the optimal Kalman Gain ($K$), correcting the state and shrinking estimation uncertainty ($P$).

### Q4: What are the state and measurement vectors in this implementation?
**Ans:**
- **State Vector:** $\mathbf{x} = [x, y, v_x, v_y]^T$ (4D vector representing bounding box center $(x, y)$ and velocities $(v_x, v_y)$).
- **Measurement Vector:** $\mathbf{z} = [z_x, z_y]^T$ (2D vector representing observed center $(x_c, y_c)$ from the YOLO detection).

### Q5: What do the State Transition ($F$) and Measurement ($H$) matrices represent?
**Ans:**
- **State Transition Matrix ($F$):** Encodes the physical constant-velocity kinematic laws: $x_k = x_{k-1} + v_{x, k-1}\Delta t$.
- **Measurement Matrix ($H$):** Maps the 4D state vector to the 2D measurement space by picking out position coordinates: $[x, y]^T = H \mathbf{x}$.

### Q6: What are Process Noise ($Q$) and Measurement Noise ($R$)?
**Ans:**
- **Process Noise ($Q$):** Represents physical uncertainty in the motion model (e.g., sudden braking, vehicle acceleration, steering changes).
- **Measurement Noise ($R$):** Represents detector uncertainty and bounding box localization errors.

### Q7: Why does each vehicle require a separate Kalman Filter?
**Ans:** Each vehicle in a traffic scene operates independently with its own distinct trajectory, speed, lane changes, and kinematics. Sharing a single filter would blend states across vehicles, causing catastrophic tracking failure.

### Q8: What is Data Association, and why use the Hungarian Algorithm?
**Ans:** Data association solves the assignment problem: matching $M$ predicted tracks with $N$ current detections. The Hungarian algorithm solves this as a global bipartite graph minimum-weight matching problem in $\mathcal{O}(V^3)$ time, ensuring that the total association distance across the entire scene is globally minimized and strictly preventing two detections from updating the same track.

### Q9: What happens when a vehicle detection is temporarily missing?
**Ans:** The track enters `COASTED` (prediction-only) state. The Kalman Filter advances the state using the motion model without calling `correct()`. The estimated covariance uncertainty $P$ grows. If a detection matches within `max_distance` before `max_age` frames elapse, the track seamlessly recovers its persistent identity.

### Q10: What is the fundamental difference between Object Detection and Object Tracking?
**Ans:**
- **Object Detection:** An instantaneous per-frame classification and localization task with no temporal awareness or memory between frames.
- **Object Tracking:** A temporal spatio-temporal task that maintains persistent object identities across consecutive video frames, modeling trajectories and velocity over time.

### Q11: What are the known limitations of Kalman Filter-based tracking?
**Ans:**
1. It assumes linear constant-velocity dynamics; sharp erratic turns or rapid stop-and-go maneuvers require non-linear extensions (EKF/UKF).
2. It relies solely on spatial proximity; if two identical vehicles closely cross paths (overlap) during a turn, identity switches can occur without appearance feature re-identification (Re-ID).
