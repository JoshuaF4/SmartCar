"""
Servo control for camera arm using PCA9685
"""

import time
import logging
from typing import Tuple

try:
    from adafruit_servokit import ServoKit
except ImportError:
    # Mock for development
    from unittest.mock import MagicMock
    ServoKit = MagicMock

from ..config import config

logger = logging.getLogger(__name__)


class ServoController:
    """Controls camera arm servos via PCA9685"""

    def __init__(self):
        self.config = config.hardware.get('servos', {})

        # Servo channels
        self.pan_channel = self.config.get('camera_pan', 0)
        self.tilt_channel = self.config.get('camera_tilt', 1)

        # Angle limits
        self.pan_min = self.config.get('pan_min', 0)
        self.pan_max = self.config.get('pan_max', 180)
        self.tilt_min = self.config.get('tilt_min', 0)
        self.tilt_max = self.config.get('tilt_max', 90)

        # Default positions
        self.default_pan = self.config.get('default_pan', 90)
        self.default_tilt = self.config.get('default_tilt', 45)

        # Current positions
        self.current_pan = self.default_pan
        self.current_tilt = self.default_tilt

        # ServoKit instance
        self.kit = None
        self.is_initialized = False

    def initialize(self):
        """Initialize PCA9685 servo controller"""
        try:
            i2c_address = self.config.get('i2c_address', 0x40)
            self.kit = ServoKit(channels=16, address=i2c_address)

            # Set pulse width range for standard servos
            self.kit.servo[self.pan_channel].set_pulse_width_range(500, 2500)
            self.kit.servo[self.tilt_channel].set_pulse_width_range(500, 2500)

            # Move to default position
            self.center()

            self.is_initialized = True
            logger.info("Servo controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize servos: {e}")
            raise

    def set_pan(self, angle: float):
        """Set camera pan angle"""
        if not self.is_initialized:
            logger.warning("Servos not initialized")
            return

        angle = max(self.pan_min, min(self.pan_max, angle))
        self.kit.servo[self.pan_channel].angle = angle
        self.current_pan = angle
        logger.debug(f"Pan set to {angle}°")

    def set_tilt(self, angle: float):
        """Set camera tilt angle"""
        if not self.is_initialized:
            logger.warning("Servos not initialized")
            return

        angle = max(self.tilt_min, min(self.tilt_max, angle))
        self.kit.servo[self.tilt_channel].angle = angle
        self.current_tilt = angle
        logger.debug(f"Tilt set to {angle}°")

    def set_position(self, pan: float, tilt: float):
        """Set both pan and tilt angles"""
        self.set_pan(pan)
        self.set_tilt(tilt)

    def center(self):
        """Move camera to center/default position"""
        self.set_position(self.default_pan, self.default_tilt)
        logger.debug("Camera centered")

    def look_left(self, amount: float = 30):
        """Pan camera left by amount"""
        new_pan = self.current_pan + amount
        self.set_pan(new_pan)

    def look_right(self, amount: float = 30):
        """Pan camera right by amount"""
        new_pan = self.current_pan - amount
        self.set_pan(new_pan)

    def look_up(self, amount: float = 15):
        """Tilt camera up by amount"""
        new_tilt = self.current_tilt - amount
        self.set_tilt(new_tilt)

    def look_down(self, amount: float = 15):
        """Tilt camera down by amount"""
        new_tilt = self.current_tilt + amount
        self.set_tilt(new_tilt)

    def scan_horizontal(self, steps: int = 5, delay: float = 0.5):
        """
        Perform horizontal scan for obstacle detection
        Yields (pan_angle, tilt_angle) at each step
        """
        step_size = (self.pan_max - self.pan_min) / steps

        for i in range(steps + 1):
            angle = self.pan_min + (i * step_size)
            self.set_pan(angle)
            time.sleep(delay)
            yield (self.current_pan, self.current_tilt)

        # Return to center
        self.center()

    def scan_area(self, h_steps: int = 3, v_steps: int = 2, delay: float = 0.3):
        """
        Perform 2D area scan
        Yields (pan_angle, tilt_angle) at each position
        """
        h_step_size = (self.pan_max - self.pan_min) / h_steps
        v_step_size = (self.tilt_max - self.tilt_min) / v_steps

        for v in range(v_steps + 1):
            tilt = self.tilt_min + (v * v_step_size)
            self.set_tilt(tilt)

            for h in range(h_steps + 1):
                pan = self.pan_min + (h * h_step_size)
                self.set_pan(pan)
                time.sleep(delay)
                yield (self.current_pan, self.current_tilt)

        # Return to center
        self.center()

    def track_object(self, x_offset: float, y_offset: float,
                     sensitivity: float = 0.5):
        """
        Track an object based on its offset from center
        Args:
            x_offset: -1.0 to 1.0 (negative = left)
            y_offset: -1.0 to 1.0 (negative = up)
            sensitivity: movement multiplier
        """
        pan_adjustment = -x_offset * sensitivity * 10
        tilt_adjustment = y_offset * sensitivity * 5

        new_pan = self.current_pan + pan_adjustment
        new_tilt = self.current_tilt + tilt_adjustment

        self.set_position(new_pan, new_tilt)

    def get_position(self) -> Tuple[float, float]:
        """Get current pan and tilt angles"""
        return (self.current_pan, self.current_tilt)

    def cleanup(self):
        """Return to default position and cleanup"""
        if self.is_initialized:
            self.center()
            self.is_initialized = False
            logger.info("Servo controller cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
