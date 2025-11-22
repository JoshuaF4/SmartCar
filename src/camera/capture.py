"""
Camera capture module using PiCamera2
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple, Generator
from threading import Thread, Event
import queue

try:
    from picamera2 import Picamera2
except ImportError:
    # Mock for development
    from unittest.mock import MagicMock
    Picamera2 = MagicMock

import cv2

from ..config import config

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera capture and frame processing"""

    def __init__(self):
        self.config = config.camera

        # Camera settings
        self.width = self.config.get('resolution', {}).get('width', 640)
        self.height = self.config.get('resolution', {}).get('height', 480)
        self.framerate = self.config.get('framerate', 30)
        self.format = self.config.get('format', 'RGB888')
        self.rotation = self.config.get('rotation', 0)

        # Camera instance
        self.camera: Optional[Picamera2] = None

        # Frame buffer
        self._frame_queue = queue.Queue(maxsize=2)
        self._capture_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Current frame
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0

        self.is_initialized = False
        self.is_capturing = False

    def initialize(self):
        """Initialize the camera"""
        try:
            self.camera = Picamera2()

            # Configure camera
            camera_config = self.camera.create_preview_configuration(
                main={
                    "size": (self.width, self.height),
                    "format": self.format
                }
            )
            self.camera.configure(camera_config)

            # Apply rotation if needed
            if self.rotation:
                self.camera.set_controls({"Rotation": self.rotation})

            self.is_initialized = True
            logger.info(f"Camera initialized at {self.width}x{self.height}")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    def start(self):
        """Start camera capture"""
        if not self.is_initialized:
            self.initialize()

        self.camera.start()
        self._stop_event.clear()

        # Start capture thread
        self._capture_thread = Thread(target=self._capture_loop)
        self._capture_thread.daemon = True
        self._capture_thread.start()

        self.is_capturing = True
        logger.info("Camera capture started")

    def stop(self):
        """Stop camera capture"""
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)

        if self.camera:
            self.camera.stop()

        self.is_capturing = False
        logger.info("Camera capture stopped")

    def _capture_loop(self):
        """Background capture loop"""
        while not self._stop_event.is_set():
            try:
                frame = self.camera.capture_array()

                # Convert from RGB to BGR for OpenCV
                if self.format == 'RGB888':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                self.current_frame = frame
                self.frame_count += 1

                # Update queue (non-blocking)
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove old frame and add new
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._frame_queue.put_nowait(frame)

            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest frame"""
        if not self.is_capturing:
            return None

        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return self.current_frame

    def capture_single(self) -> Optional[np.ndarray]:
        """Capture a single frame (for snapshot)"""
        if not self.is_initialized:
            self.initialize()
            self.camera.start()
            time.sleep(0.5)  # Allow auto-exposure

        frame = self.camera.capture_array()

        if self.format == 'RGB888':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        while self.is_capturing:
            frame = self.get_frame()
            if frame is not None:
                yield frame

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Get current frame as JPEG bytes"""
        frame = self.get_frame(timeout=0.5)
        if frame is None:
            return None

        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()

    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize frame to specified dimensions"""
        return cv2.resize(frame, (width, height))

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get frame width and height"""
        return (self.width, self.height)

    def cleanup(self):
        """Cleanup camera resources"""
        self.stop()
        if self.camera:
            self.camera.close()
            self.camera = None
        self.is_initialized = False
        logger.info("Camera cleaned up")

    def __enter__(self):
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
