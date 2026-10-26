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


def test_yolo_real_detection_pipeline():
    """
    Verify that the real VehicleDetector model (Ultralytics YOLO) loads weights,
    runs inference on an image array, and returns structured Detection objects.
    """
    from detector import VehicleDetector, Detection
    import numpy as np

    detector = VehicleDetector(model_name="yolov8n.pt", conf_thresh=0.10)

    # Test with a synthetic frame containing a colored rectangle
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (150, 150), (350, 280), (180, 100, 50), -1)

    dets = detector.detect(test_img)
    # The call must return a list of Detection objects without crashing
    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, Detection)
        assert len(d.bbox) == 4
        assert d.width > 0
        assert d.height > 0
        assert 0.0 <= d.confidence <= 1.0


def test_visualization_rendering():
    """
    Verify that render_frame correctly annotates frames for both active and
    coasted vehicle tracks, generating valid BGR image arrays.
    """
    from visualization import render_frame
    from tracker import MultiVehicleTracker, TrackState
    from detector import Detection
    import numpy as np

    frame = np.full((432, 768, 3), 100, dtype=np.uint8)
    tracker = MultiVehicleTracker(max_age=10)

    # Create 1 active track and 1 coasted track
    det1 = Detection(
        bbox=(100.0, 100.0, 150.0, 140.0),
        center=(125.0, 120.0),
        width=50.0,
        height=40.0,
        class_id=2,
        class_name="car",
        confidence=0.88,
    )
    det2 = Detection(
        bbox=(300.0, 200.0, 360.0, 250.0),
        center=(330.0, 225.0),
        width=60.0,
        height=50.0,
        class_id=7,
        class_name="truck",
        confidence=0.92,
    )
    tracker.update([det1, det2])
    # Next frame: only det1 is present; track2 enters COASTED
    active_tracks = tracker.update([det1])

    states = {t.state for t in active_tracks}
    assert TrackState.TRACKED in states
    assert TrackState.COASTED in states

    annotated = render_frame(
        frame=frame,
        tracks=active_tracks,
        frame_num=2,
        total_frames=100,
        fps=15.0,
        total_unique=tracker.total_unique_vehicles,
        draw_trajectories=True,
    )

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (432, 768, 3)
    assert annotated.dtype == np.uint8

