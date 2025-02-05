from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class FaceTracker:
    def __init__(self, model_path: str = "yolov8n.pt", smoothing_window: int = 30):
        self.model = YOLO(model_path)
        self.tracker = None
        self.position_history = deque(maxlen=smoothing_window)
        self.tracking_initialized = False

    def detect_face(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Detect person and return upper body box"""
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        # Get all person detections
        boxes = []
        for box in results.boxes:
            if box.cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                # Focus on upper body
                height = y2 - y1
                y2 = min(h, y1 + height // 2)  # Take upper half
                area = (x2 - x1) * (y2 - y1)
                boxes.append(((x1, y1, x2, y2), area, conf))

        if not boxes:
            # Default box in the center if no detection
            size = min(w, h) // 3
            return (w // 2 - size // 2, h // 3 - size // 2, size, size)

        # Get largest detection with good confidence
        box, _, _ = max(boxes, key=lambda x: x[1] * x[2])  # area * confidence
        x1, y1, x2, y2 = box
        return (x1, y1, x2 - x1, y2 - y1)  # Convert to x, y, width, height format

    def update_tracking(
        self, frame, init_box: Optional[Tuple[int, int, int, int]] = None
    ):
        """Initialize or update tracking"""
        h, w = frame.shape[:2]

        if init_box is not None or not self.tracking_initialized:
            self.tracker = cv2.TrackerMIL.create()
            if init_box is None:
                size = min(w, h) // 3
                init_box = (w // 2 - size // 2, h // 3 - size // 2, size, size)

            success = self.tracker.init(frame, init_box)
            if not success:
                logger.warning("Failed to initialize tracker")
            self.tracking_initialized = True
            self.position_history.clear()
            return w // 2, h // 3

        success, bbox = self.tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cx = x + w // 2
            cy = y + h // 3
        else:
            cx, cy = w // 2, h // 3
            self.tracking_initialized = False

        self.position_history.append((cx, cy))
        return tuple(np.mean(self.position_history, axis=0).astype(int))

    def get_crop_box(
        self,
        center_x: int,
        center_y: int,
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
        margin: float = 0.1,
    ) -> Tuple[int, int, int, int]:
        """Calculate crop box maintaining aspect ratio"""
        # Calculate dimensions
        crop_height = int(frame_height * (1 - margin))
        crop_width = int(crop_height * target_width / target_height)

        # Center the crop box
        x = max(0, min(frame_width - crop_width, center_x - crop_width // 2))
        y = max(0, min(frame_height - crop_height, center_y - crop_height // 3))

        return (
            x,
            y,
            min(x + crop_width, frame_width),
            min(y + crop_height, frame_height),
        )
