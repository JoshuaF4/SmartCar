"""
Camera arm controller combining servo control with tracking logic
"""

import logging
import time
from typing import Tuple, Optional, List
from threading import Thread, Event

from ..hardware.servos import ServoController
from ..config import config

logger = logging.getLogger(__name__)


class CameraArm:
    """
    High-level camera arm controller with tracking and scanning capabilities
    """

    def __init__(self, servo_controller: Optional[ServoController] = None):
        self.servo = servo_controller or ServoController()
        self.config = config.hardware.get('servos', {})

        # Tracking state
        self.is_tracking = False
        self._track_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Target tracking
        self.target_position: Optional[Tuple[float, float]] = None
        self.tracking_sensitivity = 0.3
        self.tracking_deadzone = 0.05

        # Scan patterns
        self.scan_positions: List[Tuple[float, float]] = []

    def initialize(self):
        """Initialize the camera arm"""
        if not self.servo.is_initialized:
            self.servo.initialize()
        self.center()
        logger.info("Camera arm initialized")

    def center(self):
        """Move camera to center position"""
        self.servo.center()

    def set_position(self, pan: float, tilt: float):
        """Set camera position directly"""
        self.servo.set_position(pan, tilt)

    def get_position(self) -> Tuple[float, float]:
        """Get current camera position"""
        return self.servo.get_position()

    def look_at_direction(self, direction: str):
        """
        Point camera in a named direction
        Args:
            direction: 'front', 'left', 'right', 'up', 'down'
        """
        if direction == 'front':
            self.center()
        elif direction == 'left':
            self.servo.set_position(150, 45)
        elif direction == 'right':
            self.servo.set_position(30, 45)
        elif direction == 'up':
            self.servo.set_position(90, 10)
        elif direction == 'down':
            self.servo.set_position(90, 80)

    def start_tracking(self, update_callback):
        """
        Start tracking mode
        Args:
            update_callback: Function that returns (x_offset, y_offset) of target
                           Returns None if no target
        """
        self._stop_event.clear()
        self._track_thread = Thread(
            target=self._tracking_loop,
            args=(update_callback,)
        )
        self._track_thread.daemon = True
        self._track_thread.start()
        self.is_tracking = True
        logger.info("Camera tracking started")

    def stop_tracking(self):
        """Stop tracking mode"""
        self._stop_event.set()
        if self._track_thread:
            self._track_thread.join(timeout=1.0)
        self.is_tracking = False
        logger.info("Camera tracking stopped")

    def _tracking_loop(self, update_callback):
        """Background tracking loop"""
        while not self._stop_event.is_set():
            try:
                target = update_callback()

                if target:
                    x_offset, y_offset = target

                    # Apply deadzone
                    if abs(x_offset) < self.tracking_deadzone:
                        x_offset = 0
                    if abs(y_offset) < self.tracking_deadzone:
                        y_offset = 0

                    # Track if outside deadzone
                    if x_offset != 0 or y_offset != 0:
                        self.servo.track_object(
                            x_offset, y_offset,
                            self.tracking_sensitivity
                        )

                time.sleep(0.05)  # 20 Hz update rate

            except Exception as e:
                logger.error(f"Tracking error: {e}")
                time.sleep(0.1)

    def perform_scan(self, pattern: str = 'horizontal') -> List[Tuple[float, float]]:
        """
        Perform a scan pattern and return positions
        Args:
            pattern: 'horizontal', 'vertical', 'grid', 'spiral'
        Returns:
            List of (pan, tilt) positions visited
        """
        positions = []

        if pattern == 'horizontal':
            for pos in self.servo.scan_horizontal(steps=5, delay=0.3):
                positions.append(pos)

        elif pattern == 'vertical':
            # Vertical scan at current pan
            current_pan = self.servo.current_pan
            tilt_range = range(
                int(self.servo.tilt_min),
                int(self.servo.tilt_max) + 1,
                15
            )
            for tilt in tilt_range:
                self.servo.set_position(current_pan, tilt)
                time.sleep(0.3)
                positions.append((current_pan, tilt))

        elif pattern == 'grid':
            for pos in self.servo.scan_area(h_steps=4, v_steps=2, delay=0.2):
                positions.append(pos)

        elif pattern == 'spiral':
            # Spiral pattern from center outward
            center_pan = self.servo.default_pan
            center_tilt = self.servo.default_tilt

            for i in range(1, 4):
                radius = i * 15
                for angle in range(0, 360, 45):
                    import math
                    pan = center_pan + radius * math.cos(math.radians(angle))
                    tilt = center_tilt + radius * 0.5 * math.sin(math.radians(angle))

                    # Clamp to limits
                    pan = max(self.servo.pan_min, min(self.servo.pan_max, pan))
                    tilt = max(self.servo.tilt_min, min(self.servo.tilt_max, tilt))

                    self.servo.set_position(pan, tilt)
                    time.sleep(0.2)
                    positions.append((pan, tilt))

        self.center()
        self.scan_positions = positions
        return positions

    def sweep_for_obstacles(self, callback) -> List[Tuple[float, float, any]]:
        """
        Sweep camera and call callback at each position
        Args:
            callback: Function called at each position, receives (pan, tilt)
                     Should return detection results
        Returns:
            List of (pan, tilt, result) tuples
        """
        results = []

        for pos in self.servo.scan_horizontal(steps=7, delay=0.4):
            pan, tilt = pos
            result = callback(pan, tilt)
            results.append((pan, tilt, result))

        return results

    def point_at_obstacle(self, obstacle_x: float, obstacle_y: float,
                          frame_width: int, frame_height: int):
        """
        Point camera at detected obstacle position in frame
        Args:
            obstacle_x, obstacle_y: Pixel coordinates
            frame_width, frame_height: Frame dimensions
        """
        # Convert to normalized offset (-1 to 1)
        x_offset = (obstacle_x - frame_width / 2) / (frame_width / 2)
        y_offset = (obstacle_y - frame_height / 2) / (frame_height / 2)

        self.servo.track_object(x_offset, y_offset, sensitivity=1.0)

    def cleanup(self):
        """Cleanup camera arm"""
        self.stop_tracking()
        self.servo.cleanup()
        logger.info("Camera arm cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
