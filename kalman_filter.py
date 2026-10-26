"""
kalman_filter.py
================
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This module implements a dedicated 2D Constant-Velocity Kalman Filter for vehicle
tracking using OpenCV's `cv2.KalmanFilter`.

Mathematical Formulation:
-------------------------
1. State Vector:
   x_k = [x, y, v_x, v_y]^T
   - (x, y): vehicle bounding-box center position in image pixel coordinates.
   - (v_x, v_y): vehicle velocity components in pixels per frame.

2. Measurement Vector:
   z_k = [z_x, z_y]^T
   - (z_x, z_y): vehicle bounding-box center measured by the object detector.

3. State Transition Matrix (F):
   Models constant-velocity motion over time step dt (default dt = 1.0 frame):
   F = [ 1  0  dt  0 ]
       [ 0  1  0  dt ]
       [ 0  0  1   0 ]
       [ 0  0  0   1 ]
   Purpose: Projects the previous state estimate forward in time:
   x_{k|k-1} = F * x_{k-1|k-1}

4. Measurement Matrix (H):
   Maps the 4D state vector to the 2D observation space:
   H = [ 1  0  0  0 ]
       [ 0  1  0  0 ]
   Purpose: Directly extracts the position components [x, y] from the state.

5. Process Noise Covariance Matrix (Q):
   Accounts for unmodeled physical dynamics (acceleration, deceleration, steering):
   Q = G * G^T * (sigma_a)^2
   where G = [dt^2 / 2, dt^2 / 2, dt, dt]^T
   Or diagonal formulation:
   Q = diag([sigma_p^2, sigma_p^2, sigma_v^2, sigma_v^2])
   Purpose: Injects stochastic uncertainty into state prediction so the filter
   remains receptive to real-world velocity changes and trajectory curves.

6. Measurement Noise Covariance Matrix (R):
   Accounts for detector bounding-box noise and localization jitter:
   R = [ sigma_m^2     0     ]
       [    0      sigma_m^2 ]
   Purpose: Determines how much the filter trusts each detector measurement.

7. Error Covariance Matrix (P):
   Represents state estimation uncertainty:
   P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q  (Prediction step)
   K_k = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^(-1)  (Kalman Gain)
   x_{k|k} = x_{k|k-1} + K_k * (z_k - H * x_{k|k-1})  (Correction step)
   P_{k|k} = (I - K_k * H) * P_{k|k-1}  (Correction step)
"""

from typing import Tuple
import cv2
import numpy as np


class VehicleKalmanFilter:
    """
    Dedicated 2D Kalman Filter tracking a single vehicle using a constant-velocity model.
    Encapsulates OpenCV's cv2.KalmanFilter with custom state management and coasting.
    """

    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        dt: float = 1.0,
        process_noise_pos: float = 1.0,
        process_noise_vel: float = 2.0,
        measurement_noise: float = 4.0,
        initial_error_pos: float = 10.0,
        initial_error_vel: float = 100.0,
    ) -> None:
        """
        Initialize the Kalman Filter for a new vehicle track.

        Args:
            initial_x: Initial detected center X coordinate.
            initial_y: Initial detected center Y coordinate.
            dt: Time step between frames (default 1.0 frame).
            process_noise_pos: Standard deviation of position process noise.
            process_noise_vel: Standard deviation of velocity process noise.
            measurement_noise: Standard deviation of detection measurement noise.
            initial_error_pos: Initial position uncertainty variance.
            initial_error_vel: Initial velocity uncertainty variance.
        """
        self.dt = float(dt)

        # 4 dynamic params [x, y, vx, vy], 2 measure params [x, y], 0 control params
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # State transition matrix F (4x4)
        # x_new = x + vx * dt
        # y_new = y + vy * dt
        # vx_new = vx
        # vy_new = vy
        self.kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Measurement matrix H (2x4)
        # Extracts [x, y] from [x, y, vx, vy]
        self.kf.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Process noise covariance Q (4x4)
        self.kf.processNoiseCov = np.array(
            [
                [process_noise_pos**2, 0.0, 0.0, 0.0],
                [0.0, process_noise_pos**2, 0.0, 0.0],
                [0.0, 0.0, process_noise_vel**2, 0.0],
                [0.0, 0.0, 0.0, process_noise_vel**2],
            ],
            dtype=np.float32,
        )

        # Measurement noise covariance R (2x2)
        self.kf.measurementNoiseCov = np.array(
            [
                [measurement_noise**2, 0.0],
                [0.0, measurement_noise**2],
            ],
            dtype=np.float32,
        )

        # Initial estimation error covariance P (4x4)
        # Lower uncertainty for position (since initialized from measurement),
        # higher uncertainty for velocity (unobserved at birth).
        self.kf.errorCovPost = np.array(
            [
                [initial_error_pos, 0.0, 0.0, 0.0],
                [0.0, initial_error_pos, 0.0, 0.0],
                [0.0, 0.0, initial_error_vel, 0.0],
                [0.0, 0.0, 0.0, initial_error_vel],
            ],
            dtype=np.float32,
        )

        # Initial state vector x_0: [x, y, vx=0, vy=0]^T
        self.kf.statePost = np.array(
            [[initial_x], [initial_y], [0.0], [0.0]], dtype=np.float32
        )
        self.kf.statePre = self.kf.statePost.copy()

        # Cache last predicted position
        self.last_predicted: Tuple[float, float] = (initial_x, initial_y)

    def predict(self) -> Tuple[float, float]:
        """
        Advance the motion model by one time step:
        x_{k|k-1} = F * x_{k-1|k-1}
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

        Returns:
            Tuple[float, float]: Predicted center coordinate (x, y).
        """
        prediction = self.kf.predict()
        px = float(prediction[0, 0])
        py = float(prediction[1, 0])
        self.last_predicted = (px, py)
        return self.last_predicted

    def correct(self, measurement_x: float, measurement_y: float) -> Tuple[float, float]:
        """
        Update the filter state and covariance using an observed detection measurement:
        K_k = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^(-1)
        x_{k|k} = x_{k|k-1} + K_k * (z_k - H * x_{k|k-1})
        P_{k|k} = (I - K_k * H) * P_{k|k-1}

        Args:
            measurement_x: Detected center X.
            measurement_y: Detected center Y.

        Returns:
            Tuple[float, float]: Corrected center coordinate (x, y).
        """
        measurement = np.array([[measurement_x], [measurement_y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        cx = float(corrected[0, 0])
        cy = float(corrected[1, 0])
        return cx, cy

    def coast_prediction_only(self) -> Tuple[float, float]:
        """
        Handle missing detection in the current frame (prediction-only / coasting).
        Synchronizes statePost and errorCovPost to the predicted state so future
        predictions continue forward seamlessly while uncertainty covariance grows.

        Returns:
            Tuple[float, float]: Retained predicted center coordinate (x, y).
        """
        # Synchronize OpenCV internal state without a measurement correction
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        px = float(self.kf.statePost[0, 0])
        py = float(self.kf.statePost[1, 0])
        return px, py

    def get_state(self) -> Tuple[float, float, float, float]:
        """
        Return the current state estimate [x, y, vx, vy].
        """
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0]), float(s[2, 0]), float(s[3, 0])

    def get_position(self) -> Tuple[float, float]:
        """
        Return the current estimated (x, y) center position.
        """
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0])

    def get_velocity(self) -> Tuple[float, float]:
        """
        Return the current estimated (vx, vy) velocity vector.
        """
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])

    def get_speed(self) -> float:
        """
        Return the scalar speed magnitude: sqrt(vx^2 + vy^2).
        """
        vx, vy = self.get_velocity()
        return float(np.hypot(vx, vy))

    def get_covariance_trace(self) -> float:
        """
        Return the trace of the position error covariance (measure of uncertainty).
        """
        cov = self.kf.errorCovPost
        return float(cov[0, 0] + cov[1, 1])

    def set_dt(self, dt: float) -> None:
        """
        Update the time step dt and adjust the state transition matrix F accordingly.
        Supports variable frame intervals (e.g., variable FPS video streams).
        """
        self.dt = float(dt)
        self.kf.transitionMatrix[0, 2] = self.dt
        self.kf.transitionMatrix[1, 3] = self.dt

