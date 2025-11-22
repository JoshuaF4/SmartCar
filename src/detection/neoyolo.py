"""
NeoYolo obstacle detection integration
Uses Ultralytics YOLO as backend with custom configuration
"""

import logging
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    from unittest.mock import MagicMock
    YOLO = MagicMock
    torch = MagicMock()

import cv2

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    zone: str  # 'danger', 'warning', 'safe'

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class NeoYoloDetector:
    """
    NeoYolo-based obstacle detection
    Wraps Ultralytics YOLO with custom obstacle detection logic
    """

    def __init__(self):
        self.config = config.detection

        # Model settings
        self.model_path = self.config.get('model_path', 'yolov8n.pt')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        self.input_size = self.config.get('input_size', 416)

        # Classes to detect
        self.target_classes = self.config.get('classes', [
            'person', 'car', 'bicycle', 'motorcycle',
            'bus', 'truck', 'traffic_light', 'stop_sign'
        ])

        # Detection zones
        self.danger_zone = self.config.get('danger_zone', {'y_start': 0.6, 'y_end': 1.0})
        self.warning_zone = self.config.get('warning_zone', {'y_start': 0.3, 'y_end': 0.6})

        # YOLO model
        self.model: Optional[YOLO] = None

        # Performance tracking
        self.last_inference_time = 0
        self.detections_count = 0

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_initialized = False

    def initialize(self):
        """Load the YOLO model"""
        try:
            model_path = Path(self.model_path)

            # Use pretrained model if custom model not found
            if not model_path.exists():
                logger.warning(f"Model {self.model_path} not found, using yolov8n.pt")
                self.model_path = 'yolov8n.pt'

            self.model = YOLO(self.model_path)

            # Set device
            if self.device == 'cuda':
                self.model.to('cuda')
                logger.info("Using CUDA for inference")
            else:
                logger.info("Using CPU for inference")

            # Warm up the model
            dummy_input = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model(dummy_input, verbose=False)

            self.is_initialized = True
            logger.info(f"NeoYolo detector initialized with {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform object detection on frame
        Args:
            frame: BGR image from camera
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            logger.warning("Detector not initialized")
            return []

        start_time = time.time()
        detections = []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                imgsz=self.input_size,
                verbose=False
            )

            # Process results
            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box data
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    # Filter by target classes
                    if class_name.lower() not in [c.lower() for c in self.target_classes]:
                        # Check if it's a general obstacle
                        if class_name.lower() not in ['obstacle', 'unknown']:
                            continue

                    # Get bbox coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Calculate center and area
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)

                    # Determine zone based on y position
                    frame_height = frame.shape[0]
                    relative_y = center_y / frame_height

                    if relative_y >= self.danger_zone['y_start']:
                        zone = 'danger'
                    elif relative_y >= self.warning_zone['y_start']:
                        zone = 'warning'
                    else:
                        zone = 'safe'

                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        area=area,
                        zone=zone
                    )
                    detections.append(detection)

            self.last_inference_time = time.time() - start_time
            self.detections_count += len(detections)

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return detections

    def detect_obstacles(self, frame: np.ndarray) -> Dict:
        """
        Detect obstacles and return structured result for navigation
        Returns:
            Dict with 'danger', 'warning', 'safe' zones and their obstacles
        """
        detections = self.detect(frame)

        result = {
            'danger': [],
            'warning': [],
            'safe': [],
            'has_danger': False,
            'has_warning': False,
            'primary_obstacle': None,
            'suggested_action': 'continue'
        }

        for det in detections:
            result[det.zone].append(det)

        result['has_danger'] = len(result['danger']) > 0
        result['has_warning'] = len(result['warning']) > 0

        # Find primary obstacle (largest in danger zone, or warning zone)
        if result['danger']:
            result['primary_obstacle'] = max(result['danger'], key=lambda d: d.area)
            result['suggested_action'] = 'stop'
        elif result['warning']:
            result['primary_obstacle'] = max(result['warning'], key=lambda d: d.area)
            result['suggested_action'] = 'slow'

        # Determine avoidance direction
        if result['primary_obstacle']:
            frame_width = frame.shape[1]
            obs_center_x = result['primary_obstacle'].center[0]

            if obs_center_x < frame_width * 0.4:
                result['avoid_direction'] = 'right'
            elif obs_center_x > frame_width * 0.6:
                result['avoid_direction'] = 'left'
            else:
                result['avoid_direction'] = 'back'
        else:
            result['avoid_direction'] = None

        return result

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on frame"""
        result = frame.copy()

        colors = {
            'danger': (0, 0, 255),    # Red
            'warning': (0, 165, 255), # Orange
            'safe': (0, 255, 0)       # Green
        }

        for det in detections:
            color = colors.get(det.zone, (255, 255, 255))
            x1, y1, x2, y2 = det.bbox

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                result, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )

            # Draw center point
            cv2.circle(result, det.center, 5, color, -1)

        # Draw zone lines
        height = frame.shape[0]
        danger_y = int(height * self.danger_zone['y_start'])
        warning_y = int(height * self.warning_zone['y_start'])

        cv2.line(result, (0, danger_y), (frame.shape[1], danger_y), (0, 0, 255), 1)
        cv2.line(result, (0, warning_y), (frame.shape[1], warning_y), (0, 165, 255), 1)

        return result

    def get_fps(self) -> float:
        """Get inference FPS"""
        if self.last_inference_time > 0:
            return 1.0 / self.last_inference_time
        return 0.0

    def cleanup(self):
        """Cleanup detector resources"""
        self.model = None
        self.is_initialized = False
        logger.info("Detector cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
