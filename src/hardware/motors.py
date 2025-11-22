"""
Motor control for differential drive robot
Supports L298N motor driver
"""

import time
import logging
from typing import Tuple

try:
    import RPi.GPIO as GPIO
except ImportError:
    # Mock GPIO for development on non-Pi systems
    from unittest.mock import MagicMock
    GPIO = MagicMock()

from ..config import config

logger = logging.getLogger(__name__)


class MotorController:
    """Controls the car's motors using L298N motor driver"""

    def __init__(self):
        self.config = config.hardware.get('motors', {})

        # GPIO pins
        self.left_forward = self.config.get('left_forward', 17)
        self.left_backward = self.config.get('left_backward', 27)
        self.right_forward = self.config.get('right_forward', 22)
        self.right_backward = self.config.get('right_backward', 23)
        self.left_pwm_pin = self.config.get('left_pwm', 12)
        self.right_pwm_pin = self.config.get('right_pwm', 13)

        self.pwm_frequency = self.config.get('pwm_frequency', 1000)

        # PWM objects
        self.left_pwm = None
        self.right_pwm = None

        # Current state
        self.current_speed = 0
        self.is_initialized = False

    def initialize(self):
        """Initialize GPIO pins for motor control"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup direction pins
            for pin in [self.left_forward, self.left_backward,
                       self.right_forward, self.right_backward]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            # Setup PWM pins
            GPIO.setup(self.left_pwm_pin, GPIO.OUT)
            GPIO.setup(self.right_pwm_pin, GPIO.OUT)

            self.left_pwm = GPIO.PWM(self.left_pwm_pin, self.pwm_frequency)
            self.right_pwm = GPIO.PWM(self.right_pwm_pin, self.pwm_frequency)

            self.left_pwm.start(0)
            self.right_pwm.start(0)

            self.is_initialized = True
            logger.info("Motor controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize motors: {e}")
            raise

    def set_speed(self, left_speed: int, right_speed: int):
        """
        Set motor speeds
        Args:
            left_speed: -100 to 100 (negative = backward)
            right_speed: -100 to 100 (negative = backward)
        """
        if not self.is_initialized:
            logger.warning("Motors not initialized")
            return

        # Clamp speeds
        left_speed = max(-100, min(100, left_speed))
        right_speed = max(-100, min(100, right_speed))

        # Left motor
        if left_speed >= 0:
            GPIO.output(self.left_forward, GPIO.HIGH)
            GPIO.output(self.left_backward, GPIO.LOW)
        else:
            GPIO.output(self.left_forward, GPIO.LOW)
            GPIO.output(self.left_backward, GPIO.HIGH)

        # Right motor
        if right_speed >= 0:
            GPIO.output(self.right_forward, GPIO.HIGH)
            GPIO.output(self.right_backward, GPIO.LOW)
        else:
            GPIO.output(self.right_forward, GPIO.LOW)
            GPIO.output(self.right_backward, GPIO.HIGH)

        # Set PWM duty cycle
        self.left_pwm.ChangeDutyCycle(abs(left_speed))
        self.right_pwm.ChangeDutyCycle(abs(right_speed))

        self.current_speed = (left_speed + right_speed) // 2

    def forward(self, speed: int = 50):
        """Move forward at specified speed"""
        self.set_speed(speed, speed)
        logger.debug(f"Moving forward at speed {speed}")

    def backward(self, speed: int = 50):
        """Move backward at specified speed"""
        self.set_speed(-speed, -speed)
        logger.debug(f"Moving backward at speed {speed}")

    def turn_left(self, speed: int = 50):
        """Turn left (pivot)"""
        self.set_speed(-speed, speed)
        logger.debug(f"Turning left at speed {speed}")

    def turn_right(self, speed: int = 50):
        """Turn right (pivot)"""
        self.set_speed(speed, -speed)
        logger.debug(f"Turning right at speed {speed}")

    def curve_left(self, speed: int = 50, ratio: float = 0.5):
        """Curve left while moving forward"""
        left_speed = int(speed * ratio)
        self.set_speed(left_speed, speed)

    def curve_right(self, speed: int = 50, ratio: float = 0.5):
        """Curve right while moving forward"""
        right_speed = int(speed * ratio)
        self.set_speed(speed, right_speed)

    def stop(self):
        """Stop all motors"""
        if self.is_initialized:
            GPIO.output(self.left_forward, GPIO.LOW)
            GPIO.output(self.left_backward, GPIO.LOW)
            GPIO.output(self.right_forward, GPIO.LOW)
            GPIO.output(self.right_backward, GPIO.LOW)
            self.left_pwm.ChangeDutyCycle(0)
            self.right_pwm.ChangeDutyCycle(0)
            self.current_speed = 0
            logger.debug("Motors stopped")

    def emergency_stop(self):
        """Immediate stop for emergencies"""
        self.stop()
        logger.warning("Emergency stop activated")

    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.is_initialized:
            self.stop()
            if self.left_pwm:
                self.left_pwm.stop()
            if self.right_pwm:
                self.right_pwm.stop()
            GPIO.cleanup()
            self.is_initialized = False
            logger.info("Motor controller cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
