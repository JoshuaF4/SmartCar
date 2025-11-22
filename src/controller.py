"""
Main car controller integrating all subsystems
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Tuple
from threading import Thread, Event
from enum import Enum

from .hardware import MotorController, ServoController, SensorManager
from .camera import CameraCapture, CameraArm
from .detection import NeoYoloDetector
from .navigation import AStarPathfinder, ObstacleAvoider
from .config import config

logger = logging.getLogger(__name__)


class CarMode(Enum):
    """Operating modes for the car"""
    IDLE = "idle"
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    PATHFINDING = "pathfinding"
    COMPUTER_ASSISTED = "computer_assisted"


class SmartCarController:
    """
    Main controller for the Smart Car
    Integrates motors, camera, detection, and navigation
    """

    def __init__(self):
        # Hardware components
        self.motors = MotorController()
        self.servos = ServoController()
        self.sensors = SensorManager()

        # Camera system
        self.camera = CameraCapture()
        self.camera_arm = CameraArm(self.servos)

        # Detection and navigation
        self.detector = NeoYoloDetector()
        self.pathfinder = AStarPathfinder()
        self.avoider = ObstacleAvoider(self.sensors)

        # State
        self.mode = CarMode.IDLE
        self.is_running = False
        self._stop_event = Event()

        # Control loop
        self._control_thread: Optional[Thread] = None
        self.control_rate = 20  # Hz

        # Current status
        self.current_speed = 0
        self.current_direction = 'stopped'
        self.last_detection_result = None
        self.current_path = None

        # Callbacks for network communication
        self.on_status_update = None
        self.on_detection = None

        self.is_initialized = False

    def initialize(self):
        """Initialize all subsystems"""
        try:
            logger.info("Initializing Smart Car...")

            self.motors.initialize()
            self.servos.initialize()
            self.sensors.initialize()
            self.camera.initialize()
            self.detector.initialize()

            # Start sensor monitoring
            self.sensors.start_monitoring()

            self.is_initialized = True
            logger.info("Smart Car initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            raise

    def start(self, mode: CarMode = CarMode.AUTONOMOUS):
        """Start the car in specified mode"""
        if not self.is_initialized:
            self.initialize()

        self.mode = mode
        self.is_running = True
        self._stop_event.clear()

        # Start camera
        self.camera.start()

        # Start control loop
        self._control_thread = Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

        logger.info(f"Smart Car started in {mode.value} mode")

    def stop(self):
        """Stop the car"""
        self.is_running = False
        self._stop_event.set()

        # Stop motors immediately
        self.motors.stop()

        # Wait for control thread
        if self._control_thread:
            self._control_thread.join(timeout=2.0)

        self.mode = CarMode.IDLE
        logger.info("Smart Car stopped")

    def _control_loop(self):
        """Main control loop"""
        interval = 1.0 / self.control_rate

        while not self._stop_event.is_set():
            start_time = time.time()

            try:
                if self.mode == CarMode.AUTONOMOUS:
                    self._autonomous_control()
                elif self.mode == CarMode.PATHFINDING:
                    self._pathfinding_control()
                elif self.mode == CarMode.COMPUTER_ASSISTED:
                    self._assisted_control()
                # Manual mode is handled by direct commands

                # Update status
                self._update_status()

            except Exception as e:
                logger.error(f"Control loop error: {e}")

            # Maintain control rate
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _autonomous_control(self):
        """Autonomous navigation with obstacle avoidance"""
        # Get camera frame
        frame = self.camera.get_frame(timeout=0.1)
        if frame is None:
            return

        # Run detection
        detection_result = self.detector.detect_obstacles(frame)
        self.last_detection_result = detection_result

        # Trigger callback
        if self.on_detection and detection_result['has_danger']:
            self.on_detection(detection_result)

        # Get avoidance command
        command = self.avoider.get_avoidance_command(
            detection_result=detection_result
        )

        # Apply motor speeds
        self.motors.set_speed(
            int(command['left_speed']),
            int(command['right_speed'])
        )

        self.current_speed = (command['left_speed'] + command['right_speed']) / 2
        self.current_direction = command['action']

    def _pathfinding_control(self):
        """Follow calculated path while avoiding obstacles"""
        if not self.current_path or len(self.current_path) == 0:
            self.motors.stop()
            return

        # Get next waypoint
        target = self.current_path[0]

        # Simple waypoint following (would need odometry for real use)
        # For now, just move forward and avoid obstacles
        self._autonomous_control()

        # Remove waypoint if reached (simplified)
        # In real implementation, use odometry/localization
        if len(self.current_path) > 1:
            self.current_path.pop(0)

    def _assisted_control(self):
        """Computer-assisted control mode"""
        # Run detection and send to computer
        frame = self.camera.get_frame(timeout=0.1)
        if frame is None:
            return

        detection_result = self.detector.detect_obstacles(frame)
        self.last_detection_result = detection_result

        # In assisted mode, computer sends commands
        # Just do safety checks here
        if self.avoider.should_stop(detection_result=detection_result):
            self.motors.emergency_stop()
            logger.warning("Emergency stop in assisted mode")

    def _update_status(self):
        """Update and broadcast status"""
        status = self.get_status()

        if self.on_status_update:
            self.on_status_update(status)

    def get_status(self) -> Dict:
        """Get current car status"""
        return {
            'mode': self.mode.value,
            'is_running': self.is_running,
            'speed': self.current_speed,
            'direction': self.current_direction,
            'sensor_distances': self.sensors.distances.copy(),
            'has_obstacle': self.avoider.state.value,
            'camera_position': self.servos.get_position(),
            'detection_fps': self.detector.get_fps()
        }

    # Manual control methods
    def manual_forward(self, speed: int = 50):
        """Manual forward command"""
        if self.mode == CarMode.MANUAL:
            self.motors.forward(speed)

    def manual_backward(self, speed: int = 50):
        """Manual backward command"""
        if self.mode == CarMode.MANUAL:
            self.motors.backward(speed)

    def manual_left(self, speed: int = 50):
        """Manual turn left command"""
        if self.mode == CarMode.MANUAL:
            self.motors.turn_left(speed)

    def manual_right(self, speed: int = 50):
        """Manual turn right command"""
        if self.mode == CarMode.MANUAL:
            self.motors.turn_right(speed)

    def manual_stop(self):
        """Manual stop command"""
        self.motors.stop()

    # Camera control
    def move_camera(self, pan: float, tilt: float):
        """Set camera position"""
        self.camera_arm.set_position(pan, tilt)

    def center_camera(self):
        """Center the camera"""
        self.camera_arm.center()

    def scan_environment(self):
        """Perform environment scan"""
        return self.camera_arm.perform_scan('horizontal')

    # Navigation methods
    def set_goal(self, x: float, y: float):
        """Set navigation goal in world coordinates"""
        # Assume we're at origin (would need localization in real use)
        start = (0.0, 0.0)
        goal = (x, y)

        path = self.pathfinder.find_path_world(start, goal)

        if path:
            self.current_path = path
            self.mode = CarMode.PATHFINDING
            logger.info(f"Path found with {len(path)} waypoints")
        else:
            logger.warning("No path found to goal")

    def add_obstacle(self, x: float, y: float, radius: float = 0.2):
        """Add obstacle to pathfinding grid"""
        self.pathfinder.grid_map.add_obstacle_world(x, y, radius)

    # Utility methods
    def get_frame(self):
        """Get current camera frame"""
        return self.camera.get_frame()

    def get_frame_with_detections(self):
        """Get frame with detection boxes drawn"""
        frame = self.camera.get_frame()
        if frame is None:
            return None

        detections = self.detector.detect(frame)
        return self.detector.draw_detections(frame, detections)

    def emergency_stop(self):
        """Emergency stop"""
        self.motors.emergency_stop()
        logger.warning("Emergency stop triggered")

    def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up Smart Car...")

        self.stop()

        self.camera.cleanup()
        self.sensors.cleanup()
        self.servos.cleanup()
        self.motors.cleanup()
        self.detector.cleanup()

        self.is_initialized = False
        logger.info("Smart Car cleaned up")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
