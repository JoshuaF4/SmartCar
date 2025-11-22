"""
Sensor management for SmartCar
Supports ultrasonic (HC-SR04) and IR sensors
"""

import time
import logging
from typing import Dict, Optional
from threading import Thread, Event

try:
    import RPi.GPIO as GPIO
except ImportError:
    from unittest.mock import MagicMock
    GPIO = MagicMock()

from ..config import config

logger = logging.getLogger(__name__)


class UltrasonicSensor:
    """HC-SR04 Ultrasonic distance sensor"""

    def __init__(self, trigger_pin: int, echo_pin: int, name: str = "ultrasonic"):
        self.trigger = trigger_pin
        self.echo = echo_pin
        self.name = name
        self.last_distance = float('inf')

    def setup(self):
        """Setup GPIO pins"""
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)
        GPIO.output(self.trigger, GPIO.LOW)

    def measure_distance(self, timeout: float = 0.1) -> float:
        """
        Measure distance in meters
        Returns: distance in meters, or inf if no echo
        """
        try:
            # Send trigger pulse
            GPIO.output(self.trigger, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trigger, GPIO.LOW)

            # Wait for echo start
            start_time = time.time()
            timeout_start = start_time

            while GPIO.input(self.echo) == GPIO.LOW:
                start_time = time.time()
                if start_time - timeout_start > timeout:
                    return float('inf')

            # Wait for echo end
            end_time = start_time
            while GPIO.input(self.echo) == GPIO.HIGH:
                end_time = time.time()
                if end_time - start_time > timeout:
                    return float('inf')

            # Calculate distance
            duration = end_time - start_time
            distance = (duration * 34300) / 2  # Speed of sound = 343 m/s
            distance = distance / 100  # Convert to meters

            self.last_distance = distance
            return distance

        except Exception as e:
            logger.error(f"Error measuring distance on {self.name}: {e}")
            return float('inf')


class SensorManager:
    """Manages all sensors on the car"""

    def __init__(self):
        self.config = config.hardware.get('sensors', {})

        # Ultrasonic sensors
        self.ultrasonic_sensors: Dict[str, UltrasonicSensor] = {}

        # IR sensors
        self.ir_left_pin = self.config.get('ir_left')
        self.ir_right_pin = self.config.get('ir_right')

        # Background monitoring
        self._monitor_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Latest readings
        self.distances: Dict[str, float] = {}
        self.ir_readings: Dict[str, bool] = {}

        self.is_initialized = False

    def initialize(self):
        """Initialize all sensors"""
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup ultrasonic sensors
            for name in ['front', 'left', 'right']:
                sensor_config = self.config.get(name)
                if sensor_config:
                    trigger = sensor_config.get('trigger')
                    echo = sensor_config.get('echo')
                    if trigger and echo:
                        sensor = UltrasonicSensor(trigger, echo, name)
                        sensor.setup()
                        self.ultrasonic_sensors[name] = sensor
                        self.distances[name] = float('inf')
                        logger.debug(f"Initialized {name} ultrasonic sensor")

            # Setup IR sensors
            if self.ir_left_pin:
                GPIO.setup(self.ir_left_pin, GPIO.IN)
                self.ir_readings['left'] = False

            if self.ir_right_pin:
                GPIO.setup(self.ir_right_pin, GPIO.IN)
                self.ir_readings['right'] = False

            self.is_initialized = True
            logger.info(f"Sensor manager initialized with {len(self.ultrasonic_sensors)} ultrasonic sensors")

        except Exception as e:
            logger.error(f"Failed to initialize sensors: {e}")
            raise

    def read_distance(self, sensor_name: str) -> float:
        """Read distance from specific ultrasonic sensor"""
        if sensor_name in self.ultrasonic_sensors:
            distance = self.ultrasonic_sensors[sensor_name].measure_distance()
            self.distances[sensor_name] = distance
            return distance
        return float('inf')

    def read_all_distances(self) -> Dict[str, float]:
        """Read distances from all ultrasonic sensors"""
        for name, sensor in self.ultrasonic_sensors.items():
            self.distances[name] = sensor.measure_distance()
            time.sleep(0.01)  # Small delay between readings
        return self.distances.copy()

    def read_ir_sensors(self) -> Dict[str, bool]:
        """Read IR sensor values (True = line detected)"""
        if self.ir_left_pin:
            self.ir_readings['left'] = GPIO.input(self.ir_left_pin) == GPIO.LOW

        if self.ir_right_pin:
            self.ir_readings['right'] = GPIO.input(self.ir_right_pin) == GPIO.LOW

        return self.ir_readings.copy()

    def get_front_distance(self) -> float:
        """Get front sensor distance"""
        return self.distances.get('front', float('inf'))

    def get_min_distance(self) -> float:
        """Get minimum distance from all sensors"""
        if self.distances:
            return min(self.distances.values())
        return float('inf')

    def is_obstacle_detected(self, threshold: float = 0.3) -> bool:
        """Check if any obstacle is within threshold distance"""
        return self.get_min_distance() < threshold

    def get_obstacle_direction(self, threshold: float = 0.5) -> Optional[str]:
        """
        Get direction of nearest obstacle
        Returns: 'front', 'left', 'right', or None
        """
        self.read_all_distances()

        min_dist = float('inf')
        direction = None

        for name, dist in self.distances.items():
            if dist < threshold and dist < min_dist:
                min_dist = dist
                direction = name

        return direction

    def start_monitoring(self, interval: float = 0.1):
        """Start background sensor monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, args=(interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Sensor monitoring started")

    def stop_monitoring(self):
        """Stop background sensor monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Sensor monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self.read_all_distances()
                self.read_ir_sensors()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Sensor monitoring error: {e}")
                time.sleep(interval)

    def cleanup(self):
        """Cleanup sensor resources"""
        self.stop_monitoring()
        self.is_initialized = False
        logger.info("Sensor manager cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
