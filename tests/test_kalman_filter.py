"""
test_kalman_filter.py
=====================
Unit tests for the VehicleKalmanFilter module.
Tests state estimation, constant-velocity motion model, prediction,
measurement correction, covariance updates, and prediction-only coasting.
"""

import sys
import os
import pytest
import numpy as np

# Ensure vehicle_tracking_lab is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kalman_filter import VehicleKalmanFilter


def test_kalman_initialization():
    """Verify state vector and matrix dimensions on initialization."""
    kf = VehicleKalmanFilter(initial_x=100.0, initial_y=200.0, dt=1.0)

    # Initial state should be [100, 200, 0, 0]
    x, y, vx, vy = kf.get_state()
    assert pytest.approx(x, abs=1e-3) == 100.0
    assert pytest.approx(y, abs=1e-3) == 200.0
    assert pytest.approx(vx, abs=1e-3) == 0.0
    assert pytest.approx(vy, abs=1e-3) == 0.0

    # Matrices dimensions
    assert kf.kf.transitionMatrix.shape == (4, 4)
    assert kf.kf.measurementMatrix.shape == (2, 4)
    assert kf.kf.processNoiseCov.shape == (4, 4)
    assert kf.kf.measurementNoiseCov.shape == (2, 2)
    assert kf.kf.errorCovPost.shape == (4, 4)


def test_kalman_prediction():
    """Verify that predicting with non-zero velocity moves position forward linearly."""
    kf = VehicleKalmanFilter(initial_x=50.0, initial_y=80.0, dt=1.0)

    # Manually set state to include a known velocity: vx=4, vy=3
    kf.kf.statePost = np.array([[50.0], [80.0], [4.0], [3.0]], dtype=np.float32)

    # Predict next state: x_new = 50 + 4*1 = 54, y_new = 80 + 3*1 = 83
    px, py = kf.predict()
    assert pytest.approx(px, abs=1e-2) == 54.0
    assert pytest.approx(py, abs=1e-2) == 83.0


def test_kalman_correction():
    """Verify that filter correction pulls state towards the measured observation."""
    kf = VehicleKalmanFilter(initial_x=100.0, initial_y=100.0, dt=1.0)

    # Predict first
    kf.predict()

    # Provide a measurement displaced to (110, 105)
    cx, cy = kf.correct(measurement_x=110.0, measurement_y=105.0)

    # Corrected position should be between predicted (100) and measurement (110, 105)
    assert 100.0 < cx <= 110.0
    assert 100.0 < cy <= 105.0


def test_kalman_covariance_behavior():
    """
    Verify that:
    1. Prediction increases state covariance trace (uncertainty grows due to Q).
    2. Correction decreases state covariance trace (uncertainty shrinks due to Kalman Gain).
    """
    kf = VehicleKalmanFilter(initial_x=100.0, initial_y=100.0, dt=1.0)

    initial_cov = kf.get_covariance_trace()

    # Step 1: Predict -> Uncertainty increases
    kf.predict()
    cov_pre = float(kf.kf.errorCovPre[0, 0] + kf.kf.errorCovPre[1, 1])
    assert cov_pre > initial_cov

    # Step 2: Correct with measurement -> Uncertainty decreases
    kf.correct(102.0, 101.0)
    cov_post = kf.get_covariance_trace()
    assert cov_post < cov_pre


def test_kalman_coasting_prediction_only():
    """
    Verify that when detections are missing, coast_prediction_only() maintains
    linear trajectory without divergence or freeze.
    """
    kf = VehicleKalmanFilter(initial_x=10.0, initial_y=20.0, dt=1.0)

    # Establish initial velocity via two consecutive measurements
    kf.predict()
    kf.correct(15.0, 25.0)
    kf.predict()
    kf.correct(20.0, 30.0)

    # Verify positive estimated velocity
    vx, vy = kf.get_velocity()
    assert vx > 0.0
    assert vy > 0.0

    # Coast for 3 consecutive frames with missing detections
    pos_history = []
    for _ in range(3):
        kf.predict()
        px, py = kf.coast_prediction_only()
        pos_history.append((px, py))

    # Positions should strictly advance forward
    assert pos_history[0][0] < pos_history[1][0] < pos_history[2][0]
    assert pos_history[0][1] < pos_history[1][1] < pos_history[2][1]


def test_kalman_variable_dt():
    """
    Verify that updating dt via set_dt adjusts the state transition matrix F
    and scales the forward constant-velocity projection proportionally.
    """
    kf = VehicleKalmanFilter(initial_x=100.0, initial_y=100.0, dt=1.0)
    kf.kf.statePost = np.array([[100.0], [100.0], [5.0], [10.0]], dtype=np.float32)

    # With dt = 1.0, prediction should be (105, 110)
    px1, py1 = kf.predict()
    assert pytest.approx(px1, abs=1e-2) == 105.0
    assert pytest.approx(py1, abs=1e-2) == 110.0

    # Change dt to 2.5: x_new = 105 + 5.0 * 2.5 = 117.5
    kf.set_dt(2.5)
    assert pytest.approx(kf.kf.transitionMatrix[0, 2], abs=1e-3) == 2.5
    assert pytest.approx(kf.kf.transitionMatrix[1, 3], abs=1e-3) == 2.5

    px2, py2 = kf.predict()
    assert pytest.approx(px2, abs=1e-2) == 117.5
    assert pytest.approx(py2, abs=1e-2) == 135.0

