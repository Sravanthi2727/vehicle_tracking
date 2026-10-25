"""
main.py
=======
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This is the main entry point for the Multi-Vehicle Tracking pipeline.
It orchestrates:
1. Video stream ingestion and validation.
2. Per-frame vehicle detection using YOLOv8 with resolution-adaptive inference.
3. Multi-object tracking with dedicated Kalman Filters and Hungarian data association.
4. Rich visual annotation with active/coasted styling, IDs, and trajectory trails.
5. Real-time display and output video encoding.

Usage:
------
    python main.py --input path/to/traffic.mp4
    python main.py --input input_videos/real.mp4 --output output_videos/result.mp4
    python main.py --input traffic.mp4 --conf 0.20 --imgsz 1280 --max-age 25
"""

import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from detector import VehicleDetector
from tracker import MultiVehicleTracker
from visualization import render_frame


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Vehicle Tracking Using Kalman Filter (CS359 AI for Robotics Lab 10, SVNIT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to input traffic video file (.mp4, .avi, .mov, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_videos/tracked_output.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Vehicle detection confidence threshold (0.01 to 0.99). Default 0.20 detects distant and shaded vehicles.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="YOLO inference image size (0 = auto-scaled: 1280 for 1080p/1440p/4K, 640 for SD)",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=25,
        help="Maximum consecutive missed frames before deleting a vehicle track",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.0,
        help="Max Euclidean association distance in pixels (0.0 = auto-scaled to video resolution)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help="Minimum detection hits required to confirm a track and count it in total vehicles",
    )
    parser.add_argument(
        "--no-agnostic-nms",
        action="store_true",
        help="Disable class-agnostic NMS (which merges overlapping duplicate vehicle bounding boxes)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model weights path or name (e.g. yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference compute device ('cpu', 'cuda', or 'auto')",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display window (recommended for headless or batch runs)",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable rendering of historical movement trajectory trails",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving the output video to disk",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to process (0 = process entire video)",
    )

    args = parser.parse_args()

    # Configuration validations
    if not (0.01 <= args.conf <= 0.99):
        parser.error(f"Confidence threshold --conf must be between 0.01 and 0.99, got {args.conf}")
    if args.max_age < 1:
        parser.error(f"Maximum missed frames --max-age must be >= 1, got {args.max_age}")
    if args.min_hits < 1:
        parser.error(f"Minimum hits --min-hits must be >= 1, got {args.min_hits}")

    return args


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """
    Open and validate the input video stream.

    Args:
        video_path: Path to video file.

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the file cannot be decoded as a valid video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video file not found at: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open or decode video at: '{video_path}'. Ensure the codec is supported.")

    return cap


def setup_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> Optional[cv2.VideoWriter]:
    """
    Initialize cv2.VideoWriter with fallback codecs (mp4v, avc1, XVID).
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    codecs_to_try = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
    ]

    for codec_str, _ in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer

    print(f"[Warning] Failed to initialize VideoWriter with standard codecs for '{output_path}'.")
    return None


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Execute the vehicle tracking pipeline over the input video.
    """
    # Check input video path
    input_path = args.input
    if input_path is None or not os.path.exists(input_path):
        # Check standard fallback locations
        fallbacks = [
            os.path.join("input_videos", "real.mp4"),
            os.path.join("input_videos", "highway_traffic.mp4"),
            os.path.join("input_videos", "sample_traffic.mp4"),
        ]
        found_fallback = None
        for fb in fallbacks:
            if os.path.isfile(fb):
                found_fallback = fb
                break

        if found_fallback:
            input_path = found_fallback
            print(f"[Info] No input specified. Automatically using discovered video: '{found_fallback}'")
        else:
            print("\n[Error] No valid input video specified!")
            print("Please specify an input video using: python main.py --input <path_to_video.mp4>")
            print("To generate a synthetic test video, run: python generate_sample_video.py\n")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("  SVNIT Surat | CS359 AI for Robotics (Lab Assignment 10)")
    print("  Multi-Vehicle Tracking Using Kalman Filter & Hungarian Association")
    print("=" * 70)

    # Open video stream
    try:
        cap = open_video_capture(input_path)
    except Exception as e:
        print(f"[Error] Failed to open input video: {e}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or math.isnan(input_fps):
        input_fps = 25.0

    # Auto-resolve inference resolution (imgsz)
    if args.imgsz > 0:
        effective_imgsz = args.imgsz
        imgsz_mode = "manual"
    else:
        # Scale to 1280 for 1080p, 1440p, 4K videos; 640 for standard definition
        effective_imgsz = 1280 if max(width, height) >= 1920 else 640
        imgsz_mode = "auto-adaptive"

    # Auto-resolve association gating distance (max_distance)
    if args.max_distance > 0.0:
        effective_max_distance = args.max_distance
        dist_mode = "manual"
    else:
        # Scale distance proportionally to image diagonal (base: 85px for ~1000px dimension)
        res_scale = max(1.0, max(width, height) / 1000.0)
        effective_max_distance = 85.0 * res_scale
        dist_mode = "auto-adaptive"

    print(f"[Input Video] Path: {input_path}")
    print(f"[Input Video] Resolution: {width}x{height} | FPS: {input_fps:.2f} | Frames: {total_frames}")
    print(
        f"[Tracking Config] Model: {args.model} | Conf: {args.conf:.2f} | "
        f"ImgSz: {effective_imgsz} ({imgsz_mode}) | Min-Hits: {args.min_hits} | "
        f"Max-Age: {args.max_age} | Max-Dist: {effective_max_distance:.1f}px ({dist_mode})"
    )

    # Initialize Object Detector
    print(f"[Detector] Initializing YOLO vehicle detector (model='{args.model}')...")
    try:
        detector = VehicleDetector(
            model_name=args.model,
            conf_thresh=args.conf,
            imgsz=effective_imgsz,
            device=args.device,
            agnostic_nms=not args.no_agnostic_nms,
        )
    except Exception as e:
        print(f"[Error] Detector initialization failed: {e}")
        cap.release()
        sys.exit(1)

    # Initialize Multi-Vehicle Tracker
    tracker = MultiVehicleTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        max_distance=effective_max_distance,
        dt=1.0,
    )

    # Initialize Video Writer
    writer = None
    if not args.no_save:
        writer = setup_video_writer(args.output, input_fps, width, height)
        if writer:
            print(f"[Output Video] Will save annotated video to: {args.output}")

    window_name = "SVNIT Lab 10: Multi-Vehicle Tracking (Kalman Filter)"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))
        print("[Display] Interactive display enabled. Press 'q' or 'ESC' to exit early.")

    processed_frames = 0
    start_time = time.time()
    last_fps_time = time.time()
    current_fps = 0.0

    try:
        while True:
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                print(f"\n[Info] Reached user-specified maximum frame limit: {args.max_frames}")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                # Video reached end of stream
                break

            processed_frames += 1

            # Step 1: Detect vehicles in frame
            detections = detector.detect(frame)

            # Step 2: Predict and associate with Kalman Filters
            active_tracks = tracker.update(detections)

            # Step 3: Compute current processing FPS
            now = time.time()
            elapsed_chunk = now - last_fps_time
            if elapsed_chunk > 0.4:
                current_fps = 1.0 / max(1e-4, (now - start_time) / processed_frames)
                last_fps_time = now

            # Step 4: Render visualization
            annotated_frame = render_frame(
                frame=frame,
                tracks=active_tracks,
                frame_num=processed_frames,
                total_frames=total_frames,
                fps=current_fps,
                total_unique=tracker.total_unique_vehicles,
                draw_trajectories=not args.no_trajectory,
            )

            # Step 5: Save frame to video
            if writer is not None:
                writer.write(annotated_frame)

            # Step 6: Interactive Display
            if not args.no_show:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    print("\n[User Interrupt] Processing stopped by user keypress.")
                    break

            # Periodic console progress update
            if processed_frames % 15 == 0 or processed_frames == total_frames:
                active, det_count, coast_count = tracker.get_track_counts()
                pct = (processed_frames / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"\rProcessing frame {processed_frames}/{total_frames} ({pct:.1f}%) | "
                    f"Active: {active} (Det: {det_count}, Coast: {coast_count}) | "
                    f"Unique: {tracker.total_unique_vehicles} | Speed: {current_fps:.1f} FPS",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n\n[Warning] Processing interrupted by KeyboardInterrupt (Ctrl+C).")
    finally:
        # Guarantee all hardware and OS resources are safely released
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 70)
    print("  TRACKING COMPLETE — SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Frames Processed : {processed_frames}")
    print(f"Total Processing Time   : {total_time:.2f} seconds")
    print(f"Average Processing Speed: {avg_fps:.2f} FPS")
    print(f"Total Unique Vehicles   : {tracker.total_unique_vehicles}")

    if writer is not None and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Annotated Output Video  : {os.path.abspath(args.output)} ({size_mb:.2f} MB)")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_arguments()
    run_pipeline(args)


if __name__ == "__main__":
    main()
