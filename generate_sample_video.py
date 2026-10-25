"""
generate_sample_video.py
========================
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This utility generates a synthetic, realistic multi-vehicle traffic video
for testing and demonstration when an external traffic video is not immediately available.

Features simulated:
- 2-lane roadway with asphalt texture, lane markings, and roadside curbs.
- Multiple moving vehicles:
  * Car 1 (Red Sedan): Moving at ~5 pixels/frame in lane 1.
  * Car 2 (Blue SUV): Moving at ~3.5 pixels/frame in lane 2.
  * Motorcycle (Yellow): Moving faster at ~7 pixels/frame.
- An overpass bridge zone where vehicles pass underneath, simulating a temporary
  visual occlusion for 6 consecutive frames where detections temporarily drop.
"""

import os
from typing import Tuple
import cv2
import numpy as np


def draw_vehicle_shape(
    frame: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[int, int],
    body_color: Tuple[int, int, int],
    roof_color: Tuple[int, int, int],
    vehicle_type: str = "car",
) -> None:
    """
    Draw a top-down / isometric synthetic vehicle.
    """
    cx, cy = int(center[0]), int(center[1])
    w, h = size
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2

    # Vehicle body with rounded corners
    cv2.rectangle(frame, (x1, y1), (x2, y2), body_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 2)

    # Windshield / cabin roof
    rw, rh = int(w * 0.65), int(h * 0.55)
    rx1, ry1 = cx - rw // 2, cy - rh // 2
    rx2, ry2 = cx + rw // 2, cy + rh // 2
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roof_color, -1)

    # Headlights
    cv2.circle(frame, (x2 - 4, y1 + 6), 3, (100, 255, 255), -1)
    cv2.circle(frame, (x2 - 4, y2 - 6), 3, (100, 255, 255), -1)

    # Tail lights
    cv2.circle(frame, (x1 + 3, y1 + 6), 3, (0, 0, 220), -1)
    cv2.circle(frame, (x1 + 3, y2 - 6), 3, (0, 0, 220), -1)


def generate_traffic_video(
    output_path: str = "input_videos/sample_traffic.mp4",
    num_frames: int = 150,
    fps: int = 25,
    width: int = 800,
    height: int = 450,
) -> str:
    """
    Generate and save a synthetic multi-vehicle traffic video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path: {output_path}")

    # Vehicle initial states (x, y, vx, vy, w, h, body_color, roof_color)
    v1 = {"x": 50.0, "y": 170.0, "vx": 4.8, "vy": 0.0, "w": 65, "h": 36, "color": (40, 50, 210), "roof": (30, 35, 120)}  # Red car
    v2 = {"x": 120.0, "y": 270.0, "vx": 3.6, "vy": 0.0, "w": 80, "h": 42, "color": (210, 110, 40), "roof": (130, 70, 25)}  # Blue SUV
    v3 = {"x": -60.0, "y": 150.0, "vx": 6.5, "vy": 0.0, "w": 40, "h": 22, "color": (20, 220, 240), "roof": (15, 120, 140)} # Yellow motorcycle

    road_top = 100
    road_bottom = 350
    lane_y = (road_top + road_bottom) // 2

    # Bridge position (occlusion zone)
    bridge_x1, bridge_x2 = 420, 490

    for f in range(num_frames):
        # Background: Green grass
        frame = np.full((height, width, 3), (70, 130, 60), dtype=np.uint8)

        # Asphalt Road
        cv2.rectangle(frame, (0, road_top), (width, road_bottom), (55, 55, 60), -1)

        # Road curbs
        cv2.rectangle(frame, (0, road_top - 6), (width, road_top), (200, 200, 200), -1)
        cv2.rectangle(frame, (0, road_bottom), (width, road_bottom + 6), (200, 200, 200), -1)

        # Dashed Center Lane Line
        dash_len = 25
        gap_len = 20
        total_dash = dash_len + gap_len
        offset = int((f * 2) % total_dash)
        for dx in range(-offset, width, total_dash):
            x_start = max(0, dx)
            x_end = min(width, dx + dash_len)
            if x_end > x_start:
                cv2.line(frame, (x_start, lane_y), (x_end, lane_y), (240, 240, 240), 2)

        # Update and draw vehicles
        vehicles = [v1, v2, v3]
        for v in vehicles:
            v["x"] += v["vx"]
            # Wrap around when reaching right boundary to keep continuous stream
            if v["x"] > width + 60:
                v["x"] = -60.0

            # Draw vehicle
            draw_vehicle_shape(
                frame,
                center=(v["x"], v["y"]),
                size=(v["w"], v["h"]),
                body_color=v["color"],
                roof_color=v["roof"],
            )

        # Draw overhead pedestrian bridge / overpass (simulating visual occlusion)
        cv2.rectangle(frame, (bridge_x1, road_top - 20), (bridge_x2, road_bottom + 20), (140, 145, 150), -1)
        cv2.rectangle(frame, (bridge_x1, road_top - 20), (bridge_x2, road_bottom + 20), (90, 95, 100), 2)
        cv2.putText(frame, "OVERPASS", (bridge_x1 + 4, lane_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)

        # Frame stamp
        cv2.putText(
            frame,
            f"SVNIT CS359 Sample Traffic Video | Frame {f + 1}/{num_frames}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"[Success] Synthetic traffic video generated successfully at: '{os.path.abspath(output_path)}'")
    return output_path


if __name__ == "__main__":
    generate_traffic_video()
