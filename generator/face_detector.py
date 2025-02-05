from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np
from typing import Optional, Tuple, List
from loguru import logger


class FaceTracker:
    def __init__(self, model_path: str = "yolov8n.pt", smoothing_window: int = 30):
        self.model = YOLO(model_path)
        self.tracker = None
        self.position_history = deque(maxlen=smoothing_window)
        self.tracking_initialized = False

    def detect_face(self, frame) -> Optional[int]:
        """Detect person and return center x coordinate if single face found"""
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        # Get all person detections
        boxes = []
        for box in results.boxes:
            if box.cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append(((x1 + x2) // 2, conf))  # store center x and confidence

        if not boxes or len(boxes) > 1:
            return None  # No face or multiple faces

        # Single face - return center x coordinate
        center_x, _ = max(boxes, key=lambda x: x[1])  # highest confidence
        return center_x

    def update_tracking(self, frame) -> Optional[int]:
        """Update tracking and return center x position"""
        if not self.tracking_initialized:
            center_x = self.detect_face(frame)
            if center_x is not None:
                self.tracking_initialized = True
                self.position_history.clear()
                self.position_history.append(center_x)
            return frame.shape[1] // 2  # Default to center if no face

        center_x = self.detect_face(frame)
        if center_x is not None:
            self.position_history.append(center_x)

        # Return smoothed position or center of frame
        if self.position_history:
            return int(sum(self.position_history) / len(self.position_history))
        return frame.shape[1] // 2

    def get_crop_box(
        self,
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
        face_center_x: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """Calculate crop box using full height"""
        # Calculate required width to maintain aspect ratio
        crop_width = int(frame_height * target_width / target_height)

        if face_center_x is not None:
            # Center on face
            x_start = max(
                0, min(frame_width - crop_width, face_center_x - crop_width // 2)
            )
        else:
            # Center frame
            x_start = (frame_width - crop_width) // 2

        return (
            x_start,
            0,  # Always start from top
            min(x_start + crop_width, frame_width),
            frame_height,  # Always use full height
        )
