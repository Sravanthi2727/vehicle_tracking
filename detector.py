"""
detector.py
===========
Multi-Vehicle Tracking Using Kalman Filter
Institute: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
Department: Computer Science and Engineering
Course: CS359 — Artificial Intelligence for Robotics (Lab Assignment 10)

This module provides a pretrained deep-learning vehicle detector using Ultralytics YOLOv8.
It detects traffic participants (cars, motorcycles, buses, trucks, bicycles) from raw
video frames and converts them into structured detections for data association and
Kalman Filtering.

Improvements for High-Resolution & Distant Traffic (e.g. 1440p / 4K):
--------------------------------------------------------------------
1. Configurable `imgsz` (e.g. 1280): Preserves small distant vehicle details.
2. Tunable `conf_thresh` (default 0.20): Captures distant/shaded vehicles.
3. Class-Agnostic NMS (`agnostic_nms=True`): Merges overlapping duplicate boxes.
4. Minimum dimension filter: Discards single-pixel noise.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """
    Structured representation of an individual detected vehicle.
    """
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]               # (center_x, center_y)
    width: float
    height: float
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    """
    Vehicle detector utilizing a pretrained YOLO model (e.g. YOLOv8n, YOLOv8s)
    configured specifically for traffic surveillance classes.
    """

    # COCO 80-class mapping for standard ground vehicles:
    # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
    DEFAULT_VEHICLE_CLASSES: Dict[int, str] = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.20,
        imgsz: Optional[int] = None,
        device: str = "auto",
        vehicle_classes: Optional[Dict[int, str]] = None,
        agnostic_nms: bool = True,
        min_box_size: float = 16.0,
    ) -> None:
        """
        Initialize the YOLO detector.

        Args:
            model_name: Ultralytics model name or path (default 'yolov8n.pt').
            conf_thresh: Minimum confidence threshold (default 0.20).
            imgsz: Inference image size (e.g. 640, 960, 1280, or None for auto).
            device: 'cpu', 'cuda', 'mps', or 'auto'.
            vehicle_classes: Optional dictionary mapping class IDs to class names.
            agnostic_nms: Whether to apply class-agnostic NMS to merge overlapping boxes.
            min_box_size: Minimum width or height in pixels to filter out noise.
        """
        self.model_name = model_name
        self.conf_thresh = float(conf_thresh)
        self.imgsz = imgsz
        self.agnostic_nms = agnostic_nms
        self.min_box_size = float(min_box_size)
        self.vehicle_classes = (
            vehicle_classes if vehicle_classes is not None else self.DEFAULT_VEHICLE_CLASSES
        )

        self._model = None
        self._device = device

    def _ensure_model_loaded(self) -> None:
        """Load YOLO model on first inference call."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                import torch

                if self._device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self._device

                self._model = YOLO(self.model_name)
                self._model.to(device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load YOLO model '{self.model_name}': {e}\n"
                    "Ensure 'ultralytics' and 'torch' are installed properly."
                ) from e

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR video frame and extract vehicle detections.

        Args:
            frame: BGR image array from cv2.VideoCapture.

        Returns:
            List of Detection objects matching target vehicle classes above conf_thresh.
        """
        if frame is None or frame.size == 0:
            return []

        self._ensure_model_loaded()

        target_class_ids = list(self.vehicle_classes.keys())

        # Build inference kwargs
        predict_kwargs = {
            "source": frame,
            "conf": self.conf_thresh,
            "classes": target_class_ids,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
        }
        if self.imgsz is not None and self.imgsz > 0:
            predict_kwargs["imgsz"] = self.imgsz

        results = self._model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes_xyxy = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        cls_ids = first_res.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cid in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)

            # Filter out tiny noise artifacts
            if w < self.min_box_size or h < self.min_box_size:
                continue

            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            cname = self.vehicle_classes.get(cid, "vehicle")

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    width=w,
                    height=h,
                    class_id=int(cid),
                    class_name=cname,
                    confidence=float(conf),
                )
            )

        return detections
