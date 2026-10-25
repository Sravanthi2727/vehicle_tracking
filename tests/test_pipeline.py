"""
test_pipeline.py
================
Integration and pipeline tests for the Vehicle Tracking application.
Tests:
- All module imports
- CLI argument parsing validation and edge cases
- Video file validation (invalid paths, nonexistent files)
- Synthetic video generation and end-to-end tracking execution
"""

import os
import sys
import pytest
import cv2

# Ensure vehicle_tracking_lab is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import open_video_capture, parse_arguments
from generate_sample_video import generate_traffic_video


def test_module_imports():
    """Verify that all core modules import without syntax or dependency errors."""
    import kalman_filter
    import detector
    import tracker
    import visualization
    import main

    assert hasattr(kalman_filter, "VehicleKalmanFilter")
    assert hasattr(detector, "VehicleDetector")
    assert hasattr(tracker, "MultiVehicleTracker")
    assert hasattr(visualization, "render_frame")
    assert hasattr(main, "run_pipeline")


def test_invalid_video_path_raises_error():
    """Verify that attempting to open a non-existent video path raises FileNotFoundError."""
    non_existent = "non_existent_video_path_12345.mp4"
    with pytest.raises(FileNotFoundError):
        open_video_capture(non_existent)


def test_synthetic_video_generation_and_reading(tmp_path):
    """
    Verify that generate_traffic_video generates a valid readable video file
    with expected resolution and frame count.
    """
    test_video_path = str(tmp_path / "test_traffic.mp4")
    generate_traffic_video(output_path=test_video_path, num_frames=30, fps=25, width=640, height=360)

    assert os.path.isfile(test_video_path)
    assert os.path.getsize(test_video_path) > 1000

    cap = cv2.VideoCapture(test_video_path)
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert frame_count == 30
    assert w == 640
    assert h == 360
