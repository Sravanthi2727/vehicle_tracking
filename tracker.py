"""
tracker.py
==========
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This module implements a custom Multi-Object Tracker (MOT) manager for vehicles.
It coordinates per-vehicle Kalman Filters, performs global data association via
the Hungarian algorithm, manages track lifecycles (birth, tentative, confirmed, coasting, death),
and prevents identity collisions.

Important Theoretical Note on Identity Preservation:
---------------------------------------------------
Tracking based on spatial Kalman Filtering and Hungarian association relies on kinematic
continuity. While robust against short frame dropouts and minor occlusions, it cannot
guarantee perfect identity preservation during prolonged full occlusions, crossing paths
at identical velocities, or extreme camera shake without deep visual re-identification
(Re-ID) embeddings. This is an inherent, well-documented characteristic of kinematic MOT.
"""

from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Set, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from detector import Detection
from kalman_filter import VehicleKalmanFilter


class TrackState(Enum):
    """Lifecycle status of a vehicle track."""
    TRACKED = "TRACKED"    # Successfully associated with a detection in the current frame
    COASTED = "COASTED"    # Detection missing; position predicted solely by Kalman Filter


class VehicleTrack:
    """
    Represents an individual vehicle track with its dedicated Kalman Filter,
    trajectory history, lifecycle metrics, and bounding-box geometry.
    """

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        max_trajectory_len: int = 50,
        min_hits: int = 1,
        dt: float = 1.0,
    ) -> None:
        """
        Initialize a new vehicle track from a detection.

        Args:
            track_id: Unique integer identifier.
            detection: Initial vehicle detection.
            max_trajectory_len: Maximum history length for path visualization.
            min_hits: Number of matched hits required to be considered confirmed.
            dt: Time step for motion prediction.
        """
        self.track_id: int = track_id
        self.class_id: int = detection.class_id
        self.class_name: str = detection.class_name
        self.confidence: float = detection.confidence
        self.min_hits: int = min_hits

        # Bounding box geometry (x1, y1, x2, y2)
        self.bbox: Tuple[float, float, float, float] = detection.bbox
        self.width: float = detection.width
        self.height: float = detection.height

        # Initialize dedicated 2D Kalman Filter
        cx, cy = detection.center
        self.kf = VehicleKalmanFilter(initial_x=cx, initial_y=cy, dt=dt)

        # Track history and state
        self.trajectory: Deque[Tuple[float, float]] = deque(maxlen=max_trajectory_len)
        self.trajectory.append((cx, cy))

        self.state: TrackState = TrackState.TRACKED
        self.time_since_update: int = 0  # Number of consecutive frames without detection
        self.hits: int = 1               # Total matched detections
        self.age: int = 1                # Total lifespan in frames

        # Cached prediction for association
        self.predicted_center: Tuple[float, float] = (cx, cy)
        self.predicted_bbox: Tuple[float, float, float, float] = detection.bbox

    @property
    def is_confirmed(self) -> bool:
        """Return True if the track has received enough hits to be confirmed."""
        return self.hits >= self.min_hits

    def predict(self) -> Tuple[float, float]:
        """
        Predict vehicle position in the current frame using the Kalman Filter.

        Returns:
            Tuple[float, float]: Predicted center coordinate (px, py).
        """
        px, py = self.kf.predict()
        self.predicted_center = (px, py)

        # Update predicted bounding box centered at (px, py)
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        self.predicted_bbox = (px - half_w, py - half_h, px + half_w, py + half_h)
        return self.predicted_center

    def update(self, detection: Detection) -> None:
        """
        Correct the Kalman Filter and track state using a newly associated detection.

        Args:
            detection: Associated detection in the current frame.
        """
        cx, cy = detection.center
        corrected_x, corrected_y = self.kf.correct(cx, cy)

        # Smooth dimensions using exponential moving average (alpha = 0.7)
        self.width = 0.7 * detection.width + 0.3 * self.width
        self.height = 0.7 * detection.height + 0.3 * self.height

        # Compute updated bounding box around corrected center
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        self.bbox = (
            corrected_x - half_w,
            corrected_y - half_h,
            corrected_x + half_w,
            corrected_y + half_h,
        )

        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence

        self.trajectory.append((corrected_x, corrected_y))
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        self.state = TrackState.TRACKED

    def mark_missed(self) -> None:
        """
        Handle a frame where no detection matched this track.
        Coasts the Kalman Filter in prediction-only mode.
        """
        # Coast filter state forward in prediction-only mode
        px, py = self.kf.coast_prediction_only()

        # Update bounding box to predicted location
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        self.bbox = (px - half_w, py - half_h, px + half_w, py + half_h)

        self.trajectory.append((px, py))
        self.time_since_update += 1
        self.age += 1
        self.state = TrackState.COASTED

    @property
    def current_center(self) -> Tuple[float, float]:
        """Return the current (x, y) center position."""
        return self.kf.get_position()

    @property
    def velocity(self) -> Tuple[float, float]:
        """Return the current estimated (vx, vy) velocity."""
        return self.kf.get_velocity()

    @property
    def speed(self) -> float:
        """Return the current estimated speed."""
        return self.kf.get_speed()

    def set_dt(self, dt: float) -> None:
        """Update the time step dt in the underlying Kalman Filter."""
        self.dt = float(dt)
        self.kf.set_dt(dt)


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class MultiVehicleTracker:
    """
    Multi-Vehicle Tracking Manager.
    Coordinates prediction, data association, track correction, and lifecycle management.
    """

    # Grouping to prevent impossible associations (e.g. bicycle matching truck)
    TWO_WHEELERS = {1, 3}       # bicycle, motorcycle
    MULTI_WHEELERS = {2, 5, 7}  # car, bus, truck

    def __init__(
        self,
        max_age: int = 25,
        min_hits: int = 1,
        max_distance: float = 85.0,
        iou_weight: float = 0.2,
        class_mismatch_penalty: Optional[float] = None,
        dt: float = 1.0,
    ) -> None:
        """
        Initialize the tracker manager.

        Args:
            max_age: Maximum consecutive missed frames before track deletion.
            min_hits: Minimum detections before track is considered confirmed.
            max_distance: Maximum association center distance gate in pixels.
            iou_weight: Weight given to (1 - IoU) in hybrid distance matching.
            class_mismatch_penalty: Cost penalty added for same-category class mismatch.
            dt: Time step between frames.
        """
        self.max_age: int = int(max_age)
        self.min_hits: int = int(min_hits)
        self.max_distance: float = float(max_distance)
        self.iou_weight: float = float(iou_weight)
        self.class_mismatch_penalty: float = (
            float(class_mismatch_penalty)
            if class_mismatch_penalty is not None
            else 0.4 * self.max_distance
        )
        self.dt: float = float(dt)

        self.tracks: List[VehicleTrack] = []
        self._next_id: int = 1
        self.total_unique_vehicles: int = 0
        self.frame_count: int = 0

    def update(self, detections: List[Detection]) -> List[VehicleTrack]:
        """
        Process the current frame detections and update all active vehicle tracks.

        Algorithm:
        1. Sanitize incoming detections (filter NaN, Inf, non-positive dimensions).
        2. Predict all existing tracks using their Kalman Filters.
        3. Compute association cost matrix with topological category and IoU penalties.
        4. Solve global linear assignment using Hungarian matching.
        5. Apply distance & validity gating to prevent false associations.
        6. Correct matched tracks with their respective detections.
        7. Coast unmatched tracks in prediction-only mode.
        8. Spawn new tracks for unmatched detections.
        9. Remove expired tracks whose missed frames exceed max_age.

        Args:
            detections: List of Detection objects from the vehicle detector.

        Returns:
            List of currently active tracks to visualize.
        """
        self.frame_count += 1

        # Step 0: Sanitize detections to guard against NaN/Inf or corrupt geometry
        valid_detections: List[Detection] = []
        for det in detections:
            if det is None:
                continue
            x1, y1, x2, y2 = det.bbox
            if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                continue
            if det.width <= 0.0 or det.height <= 0.0:
                continue
            cx, cy = det.center
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            valid_detections.append(det)
        detections = valid_detections

        # Step 1: Predict all active tracks
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        matched_track_indices: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Step 2 & 3: Data Association
        if num_tracks > 0 and num_dets > 0:
            # Build cost matrix of size (num_tracks, num_dets)
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

            for t_idx, track in enumerate(self.tracks):
                tx, ty = track.predicted_center
                t_is_2w = track.class_id in self.TWO_WHEELERS
                for d_idx, det in enumerate(detections):
                    d_is_2w = det.class_id in self.TWO_WHEELERS

                    # Disallow cross-category matches (e.g. bicycle cannot match truck)
                    if t_is_2w != d_is_2w:
                        cost_matrix[t_idx, d_idx] = 1e5
                        continue

                    dx, dy = det.center
                    dist = float(np.hypot(tx - dx, ty - dy))

                    # Soft penalty for mismatched class within compatible category
                    class_penalty = (
                        self.class_mismatch_penalty if track.class_id != det.class_id else 0.0
                    )

                    # Optional IoU penalty (1 - IoU)
                    if self.iou_weight > 0.0:
                        iou = compute_iou(track.predicted_bbox, det.bbox)
                        cost = dist + class_penalty + self.iou_weight * (1.0 - iou) * self.max_distance
                    else:
                        cost = dist + class_penalty

                    cost_matrix[t_idx, d_idx] = cost

            # Solve assignment
            if _SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                row_ind, col_ind = self._greedy_assignment(cost_matrix)

            # Step 4: Validate matches against maximum gating distance & category validity
            for t_idx, d_idx in zip(row_ind, col_ind):
                # Reject pairs blocked by category separation
                if cost_matrix[t_idx, d_idx] >= 1e4:
                    continue

                # Check pure Euclidean distance gate
                tx, ty = self.tracks[t_idx].predicted_center
                dx, dy = detections[d_idx].center
                actual_dist = np.hypot(tx - dx, ty - dy)

                if actual_dist <= self.max_distance:
                    matched_track_indices.add(int(t_idx))
                    matched_det_indices.add(int(d_idx))
                    # Step 5: Update matched track
                    self.tracks[t_idx].update(detections[d_idx])
                    if self.tracks[t_idx].hits == self.min_hits and self.min_hits > 1:
                        self.total_unique_vehicles += 1

        # Step 6: Handle unmatched tracks (coasting / prediction-only)
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track.mark_missed()

        # Step 7: Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = VehicleTrack(
                    track_id=self._next_id,
                    detection=det,
                    min_hits=self.min_hits,
                    dt=self.dt,
                )
                self._next_id += 1
                if self.min_hits <= 1:
                    self.total_unique_vehicles += 1
                self.tracks.append(new_track)

        # Step 8: Prune expired tracks
        active_tracks: List[VehicleTrack] = []
        for track in self.tracks:
            # Immediate pruning for unconfirmed transient false positives
            if not track.is_confirmed and track.time_since_update > 1:
                continue
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)

        self.tracks = active_tracks
        return self.tracks

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback greedy bipartite matching if SciPy is unavailable.
        Strictly guarantees 1-to-1 matching (no duplicate associations).
        """
        num_rows, num_cols = cost_matrix.shape
        flat_indices = np.argsort(cost_matrix, axis=None)

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        row_res: List[int] = []
        col_res: List[int] = []

        for idx in flat_indices:
            r = int(idx // num_cols)
            c = int(idx % num_cols)
            if r not in used_rows and c not in used_cols:
                used_rows.add(r)
                used_cols.add(c)
                row_res.append(r)
                col_res.append(c)
                if len(used_rows) == min(num_rows, num_cols):
                    break

        return np.array(row_res, dtype=int), np.array(col_res, dtype=int)

    def get_track_counts(self) -> Tuple[int, int, int]:
        """
        Return counts: (total_active, currently_detected, currently_coasted).
        """
        detected = sum(1 for t in self.tracks if t.state == TrackState.TRACKED)
        coasted = sum(1 for t in self.tracks if t.state == TrackState.COASTED)
        return len(self.tracks), detected, coasted
