"""
visualization.py
================
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This module handles rendering of tracking visualizations:
- Resolution-adaptive bounding boxes (solid active vs. corner-bracketed coasted).
- Persistent vehicle IDs and class labels with high-contrast background pills.
- Kalman Filter predicted positions (crosshair / marker).
- Vehicle movement trajectory history trails.
- Translucent Head-Up Display (HUD) showing frame stats and track metrics.
"""

from typing import List, Tuple
import cv2
import numpy as np

from tracker import TrackState, VehicleTrack


# Color definitions in BGR format
COLOR_ACTIVE_BOX = (50, 220, 50)       # Vibrant green for actively detected tracks
COLOR_ACTIVE_BG = (35, 160, 35)        # Darker green for label background
COLOR_COASTED_BOX = (0, 140, 255)      # Amber/Orange for prediction-only tracks
COLOR_COASTED_BG = (0, 95, 190)        # Dark amber for label background
COLOR_PREDICTED_POINT = (255, 220, 0)  # Bright cyan/yellow for predicted center
COLOR_TRAIL_ACTIVE = (0, 220, 220)     # Cyan trail for active tracks
COLOR_TRAIL_COASTED = (0, 140, 255)    # Orange trail for coasted tracks
COLOR_TEXT = (255, 255, 255)           # White text
COLOR_HUD_BG = (20, 20, 25)            # Dark slate for HUD


def draw_hud(
    frame: np.ndarray,
    frame_num: int,
    total_frames: int,
    fps: float,
    active_count: int,
    detected_count: int,
    coasted_count: int,
    total_unique: int,
) -> None:
    """
    Draw a resolution-adaptive translucent HUD banner at the top of the video frame.
    """
    h, w = frame.shape[:2]
    # Smooth scaling down to 0.55 for standard-definition videos (640p/720p/768p)
    scale = max(0.55, w / 1280.0)
    banner_h = int(round(44 * scale))

    # Create translucent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent bottom border line
    cv2.line(frame, (0, banner_h), (w, banner_h), (80, 80, 90), max(1, int(round(scale))))

    font_scale = 0.55 * scale
    font_thickness = max(1, int(round(scale)))
    y_text = int(round(28 * scale))

    # Left Section: Lab & Frame details
    frame_text = (
        f"Frame: {frame_num}/{total_frames} ({fps:.1f} FPS)"
        if total_frames > 0
        else f"Frame: {frame_num} ({fps:.1f} FPS)"
    )
    cv2.putText(
        frame,
        frame_text,
        (int(round(15 * scale)), y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (230, 230, 230),
        font_thickness + 1,
        cv2.LINE_AA,
    )

    # Middle & Right Section: Tracking Statistics
    stats_text = (
        f"Active: {active_count} | Detected: {detected_count} | "
        f"Coasted: {coasted_count} | Total Seen: {total_unique}"
    )
    (tw, th), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    right_x = max(10, w - tw - int(round(15 * scale)))
    cv2.putText(
        frame,
        stats_text,
        (right_x, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (220, 220, 220),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    """
    Draw a smooth past trajectory trail showing vehicle motion history.
    """
    if len(trajectory) < 2:
        return

    pts = list(trajectory)
    num_pts = len(pts)

    for i in range(1, num_pts):
        pt1 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
        pt2 = (int(round(pts[i][0])), int(round(pts[i][1])))

        # Fade line thickness towards current head
        base_thick = 1 + (i / num_pts) * 2.0 * scale
        thickness = int(np.clip(round(base_thick), 1, int(round(5 * scale))))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Small dot at the oldest point
    start_pt = (int(round(pts[0][0])), int(round(pts[0][1])))
    cv2.circle(frame, start_pt, max(2, int(round(2 * scale))), color, -1, cv2.LINE_AA)


def draw_track(
    frame: np.ndarray,
    track: VehicleTrack,
    draw_trajectories: bool = True,
) -> None:
    """
    Draw an individual vehicle track onto the frame with resolution-adaptive
    bounding box, status pill, predicted center crosshair, and trajectory history.
    """
    h, w = frame.shape[:2]
    scale = max(0.55, w / 1280.0)

    is_active = (track.state == TrackState.TRACKED)

    box_color = COLOR_ACTIVE_BOX if is_active else COLOR_COASTED_BOX
    bg_color = COLOR_ACTIVE_BG if is_active else COLOR_COASTED_BG
    trail_color = COLOR_TRAIL_ACTIVE if is_active else COLOR_TRAIL_COASTED

    # Draw motion trajectory trail
    if draw_trajectories and len(track.trajectory) > 1:
        draw_trajectory(frame, list(track.trajectory), trail_color, scale=scale)

    # Bounding box coordinates clamped to image borders
    x1 = int(np.clip(round(track.bbox[0]), 0, w - 1))
    y1 = int(np.clip(round(track.bbox[1]), 0, h - 1))
    x2 = int(np.clip(round(track.bbox[2]), 0, w - 1))
    y2 = int(np.clip(round(track.bbox[3]), 0, h - 1))

    box_thickness = max(2, int(round(2 * scale)))
    corner_thickness = max(3, int(round(3.5 * scale)))

    # Bounding Box: Solid for active, corner/dashed style for coasted
    if is_active:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness, cv2.LINE_AA)
    else:
        # Distinctive corner bracket styling for coasted/predicted track
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
        corner_len = min(int(round(25 * scale)), max(5, int((x2 - x1) * 0.25)))
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, corner_thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, corner_thickness, cv2.LINE_AA)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, corner_thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, corner_thickness, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, corner_thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, corner_thickness, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, corner_thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, corner_thickness, cv2.LINE_AA)

    # Kalman Filter predicted center marker (crosshair)
    px, py = track.predicted_center
    cpx = int(np.clip(round(px), 0, w - 1))
    cpy = int(np.clip(round(py), 0, h - 1))
    dot_radius = max(3, int(round(3 * scale)))
    marker_size = max(8, int(round(8 * scale)))
    cv2.circle(frame, (cpx, cpy), dot_radius, COLOR_PREDICTED_POINT, -1, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        (cpx, cpy),
        COLOR_PREDICTED_POINT,
        markerType=cv2.MARKER_CROSS,
        markerSize=marker_size,
        thickness=max(1, int(round(scale))),
        line_type=cv2.LINE_AA,
    )

    # Text Label: Vehicle ID + Class + Status
    if is_active:
        label = f"Vehicle {track.track_id}: {track.class_name} ({track.confidence:.2f})"
    else:
        label = f"Vehicle {track.track_id}: {track.class_name} [KF Pred (+{track.time_since_update}f)]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    font_thickness = max(1, int(round(scale)))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    pad_x = int(round(5 * scale))
    pad_y = int(round(4 * scale))
    banner_h = int(round(44 * scale))

    # Position label above bounding box; flip below HUD if near top border
    if y1 - th - (pad_y * 2) > banner_h:
        label_y2 = y1 - pad_y
        label_y1 = label_y2 - th - (pad_y * 2)
    else:
        # Position below HUD banner and within visible bounds
        label_y1 = max(banner_h + 2, y1 + 2)
        label_y2 = label_y1 + th + (pad_y * 2)

    label_x1 = max(0, x1)
    label_x2 = min(w, label_x1 + tw + (pad_x * 2))

    # Background pill for high contrast text readability
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)
    cv2.putText(
        frame,
        label,
        (label_x1 + pad_x, label_y2 - pad_y),
        font,
        font_scale,
        COLOR_TEXT,
        font_thickness,
        cv2.LINE_AA,
    )


def render_frame(
    frame: np.ndarray,
    tracks: List[VehicleTrack],
    frame_num: int,
    total_frames: int,
    fps: float,
    total_unique: int,
    draw_trajectories: bool = True,
) -> np.ndarray:
    """
    Composite all visual tracking overlays onto the video frame.

    Args:
        frame: Original BGR video frame.
        tracks: List of active VehicleTrack objects.
        frame_num: Current frame index (1-indexed).
        total_frames: Total frames in the video (or 0 if unknown).
        fps: Current processing frames per second.
        total_unique: Cumulative number of unique vehicles tracked.
        draw_trajectories: Whether to draw vehicle trajectory history paths.

    Returns:
        Annotated BGR image ready for display and video writing.
    """
    annotated = frame.copy()

    # Draw tracks
    for track in tracks:
        draw_track(annotated, track, draw_trajectories=draw_trajectories)

    # Draw top HUD
    detected = sum(1 for t in tracks if t.state == TrackState.TRACKED)
    coasted = sum(1 for t in tracks if t.state == TrackState.COASTED)
    draw_hud(
        annotated,
        frame_num=frame_num,
        total_frames=total_frames,
        fps=fps,
        active_count=len(tracks),
        detected_count=detected,
        coasted_count=coasted,
        total_unique=total_unique,
    )

    return annotated
