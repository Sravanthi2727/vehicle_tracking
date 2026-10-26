"""
test_tracker.py
===============
Unit tests for the MultiVehicleTracker and VehicleTrack classes.
Tests track creation, multi-vehicle Hungarian association, 1-to-1 matching guarantee,
temporary occlusion / coasting recovery, and track deletion after max_age frames.
"""

import sys
import os
import pytest

# Ensure vehicle_tracking_lab is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import Detection
from tracker import MultiVehicleTracker, TrackState


def make_detection(cx: float, cy: float, w: float = 40.0, h: float = 20.0, class_name: str = "car", conf: float = 0.9) -> Detection:
    """Helper to construct synthetic Detection objects."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return Detection(
        bbox=(x1, y1, x2, y2),
        center=(cx, cy),
        width=w,
        height=h,
        class_id=2,
        class_name=class_name,
        confidence=conf,
    )


def test_track_creation():
    """Verify that an initial detection creates a new track with ID 1."""
    tracker = MultiVehicleTracker(max_age=10, max_distance=50.0)

    det = make_detection(100.0, 150.0)
    active_tracks = tracker.update([det])

    assert len(active_tracks) == 1
    assert active_tracks[0].track_id == 1
    assert active_tracks[0].state == TrackState.TRACKED
    assert active_tracks[0].class_name == "car"
    assert tracker.total_unique_vehicles == 1


def test_multi_vehicle_association_and_distinct_ids():
    """
    Verify that multiple vehicles moving along parallel trajectories
    maintain distinct persistent IDs across frames.
    """
    tracker = MultiVehicleTracker(max_age=10, max_distance=60.0)

    # Frame 1: Two vehicles at x=100 and x=300
    det_v1_f1 = make_detection(100.0, 100.0)
    det_v2_f1 = make_detection(300.0, 100.0)
    tracks_f1 = tracker.update([det_v1_f1, det_v2_f1])

    assert len(tracks_f1) == 2
    id1 = tracks_f1[0].track_id
    id2 = tracks_f1[1].track_id
    assert id1 != id2

    # Frame 2: Both vehicles move forward by 10 pixels in X
    det_v1_f2 = make_detection(110.0, 100.0)
    det_v2_f2 = make_detection(310.0, 100.0)
    tracks_f2 = tracker.update([det_v1_f2, det_v2_f2])

    assert len(tracks_f2) == 2
    f2_ids = {t.track_id for t in tracks_f2}
    assert f2_ids == {id1, id2}


def test_one_to_one_matching_guarantee():
    """
    Verify that two detections cannot be assigned to the same track in one frame.
    One detection must match the existing track, while the second must spawn a new track.
    """
    tracker = MultiVehicleTracker(max_age=10, max_distance=80.0)

    # Initialize 1 track
    tracker.update([make_detection(100.0, 100.0)])
    assert len(tracker.tracks) == 1

    # Send 2 close detections in the next frame
    det_a = make_detection(105.0, 100.0)
    det_b = make_detection(112.0, 100.0)
    tracks = tracker.update([det_a, det_b])

    # Must result in 2 distinct tracks, not 1 track updated twice
    assert len(tracks) == 2
    ids = [t.track_id for t in tracks]
    assert len(set(ids)) == 2


def test_temporary_missing_detection_coasting_and_recovery():
    """
    Verify track behavior during temporary occlusion / missed detection:
    - Track enters COASTED state when detection is missing.
    - Position continues to be predicted by Kalman Filter.
    - When detection reappears within max_distance, track is re-associated and ID is preserved.
    """
    tracker = MultiVehicleTracker(max_age=10, max_distance=80.0)

    # Frame 1 & 2: Establish vehicle motion moving at vx ~ 10 px/frame
    tracker.update([make_detection(50.0, 100.0)])
    tracker.update([make_detection(60.0, 100.0)])
    original_id = tracker.tracks[0].track_id

    # Frame 3, 4, 5: Detection goes missing (e.g. occlusion under bridge)
    for _ in range(3):
        tracks = tracker.update([])
        assert len(tracks) == 1
        assert tracks[0].state == TrackState.COASTED
        assert tracks[0].track_id == original_id
        assert tracks[0].time_since_update > 0

    # Verify that predicted position kept advancing forward during coasting
    predicted_x = tracker.tracks[0].current_center[0]
    assert predicted_x > 60.0

    # Frame 6: Detection reappears near the predicted position (e.g. ~95)
    reappeared_det = make_detection(predicted_x + 2.0, 100.0)
    recovered_tracks = tracker.update([reappeared_det])

    assert len(recovered_tracks) == 1
    assert recovered_tracks[0].track_id == original_id
    assert recovered_tracks[0].state == TrackState.TRACKED
    assert recovered_tracks[0].time_since_update == 0


def test_track_deletion_after_max_age():
    """
    Verify that an unmatched track is pruned/deleted once its missed
    frames exceed max_age.
    """
    max_age = 5
    tracker = MultiVehicleTracker(max_age=max_age, max_distance=50.0)

    # Birth of track
    tracker.update([make_detection(100.0, 100.0)])
    assert len(tracker.tracks) == 1

    # Miss for exactly max_age frames (should still be kept in active tracks)
    for i in range(max_age):
        tracks = tracker.update([])
        assert len(tracks) == 1
        assert tracks[0].time_since_update == i + 1

    # Miss for one more frame (exceeds max_age) -> Track should be pruned
    tracks_after_expiry = tracker.update([])
    assert len(tracks_after_expiry) == 0
    assert len(tracker.tracks) == 0


def test_empty_detections_handling():
    """Verify that passing empty detection lists on an empty tracker does not raise errors."""
    tracker = MultiVehicleTracker()
    tracks = tracker.update([])
    assert len(tracks) == 0
    assert tracker.total_unique_vehicles == 0


def test_cross_category_association_prevention():
    """
    Verify that an incompatible vehicle category (e.g. bicycle, class 1)
    never matches or hijacks an existing four-wheeler track (e.g. truck, class 7)
    even when placed at identical coordinates.
    """
    tracker = MultiVehicleTracker(max_age=10, max_distance=80.0)

    # Frame 1: Truck track initialized
    truck_det = Detection(
        bbox=(90.0, 90.0, 150.0, 150.0),
        center=(120.0, 120.0),
        width=60.0,
        height=60.0,
        class_id=7,
        class_name="truck",
        confidence=0.9,
    )
    tracker.update([truck_det])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].class_name == "truck"
    truck_id = tracker.tracks[0].track_id

    # Frame 2: Only a bicycle detection appears at the same coordinate
    bike_det = Detection(
        bbox=(110.0, 110.0, 130.0, 130.0),
        center=(120.0, 120.0),
        width=20.0,
        height=20.0,
        class_id=1,
        class_name="bicycle",
        confidence=0.85,
    )
    active_tracks = tracker.update([bike_det])

    # Must result in 2 tracks: the truck coasts, while the bicycle spawns a new distinct track
    assert len(active_tracks) == 2
    truck_track = next(t for t in active_tracks if t.track_id == truck_id)
    assert truck_track.state == TrackState.COASTED
    assert truck_track.class_name == "truck"

    new_bike_track = next(t for t in active_tracks if t.track_id != truck_id)
    assert new_bike_track.state == TrackState.TRACKED
    assert new_bike_track.class_name == "bicycle"


def test_corrupt_and_nan_detection_filtering():
    """
    Verify that corrupt detections with NaN coordinates or non-positive dimensions
    are sanitized without crashing the tracker or Hungarian optimization.
    """
    tracker = MultiVehicleTracker(max_age=5, max_distance=50.0)
    import numpy as np

    # Frame 1: Initialize valid track
    tracker.update([make_detection(100.0, 100.0)])
    assert len(tracker.tracks) == 1

    # Frame 2: Pass corrupt detections (NaN bbox, zero width, etc.)
    corrupt_dets = [
        Detection(
            bbox=(float("nan"), 100.0, 120.0, 120.0),
            center=(float("nan"), 110.0),
            width=20.0,
            height=20.0,
            class_id=2,
            class_name="car",
            confidence=0.5,
        ),
        Detection(
            bbox=(100.0, 100.0, 100.0, 120.0),
            center=(100.0, 110.0),
            width=0.0,  # Zero width
            height=20.0,
            class_id=2,
            class_name="car",
            confidence=0.5,
        ),
    ]

    # Tracker must survive gracefully without raising ValueError
    tracks = tracker.update(corrupt_dets)
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.COASTED


def test_class_mismatch_penalty_preference():
    """
    Verify that when two detections are at similar distances from a track,
    the Hungarian algorithm prefers the detection matching the track's class.
    """
    tracker = MultiVehicleTracker(max_age=10, max_distance=60.0)

    # Frame 1: Track born as car at (100, 100)
    tracker.update([make_detection(100.0, 100.0, class_name="car")])
    assert tracker.tracks[0].class_id == 2

    # Frame 2: Two detections:
    # A car at distance 15px (115, 100)
    # A truck at distance 10px (110, 100)
    det_car = Detection(
        bbox=(95.0, 90.0, 135.0, 110.0),
        center=(115.0, 100.0),
        width=40.0,
        height=20.0,
        class_id=2,
        class_name="car",
        confidence=0.9,
    )
    det_truck = Detection(
        bbox=(90.0, 85.0, 130.0, 115.0),
        center=(110.0, 100.0),
        width=40.0,
        height=30.0,
        class_id=7,
        class_name="truck",
        confidence=0.9,
    )

    tracks = tracker.update([det_car, det_truck])
    assert len(tracks) == 2

    # Track 1 should associate with the car detection due to class penalty on truck
    track1 = next(t for t in tracks if t.track_id == 1)
    assert track1.class_name == "car"
    assert pytest.approx(track1.current_center[0], abs=2.0) == 115.0

