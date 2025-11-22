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
from .navigation import AStarPathfinder, ObstacleAvoider, PathMemory
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
        self.path_memory = PathMemory()

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

        # Position tracking (simplified - would use odometry in real implementation)
        self.current_position = (0.0, 0.0)
        self.current_heading = 0.0
        self._position_update_interval = 0

        # Path recording
        self.record_interval = 5  # Record every N control loops

        # Callbacks for network communication
        self.on_status_update = None
        self.on_detection = None
        self.on_path_recorded = None

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

    def start(self, mode: CarMode = CarMode.AUTONOMOUS, record_path: bool = True):
        """Start the car in specified mode"""
        if not self.is_initialized:
            self.initialize()

        self.mode = mode
        self.is_running = True
        self._stop_event.clear()

        # Start camera
        self.camera.start()

        # Start path recording
        if record_path and mode in [CarMode.AUTONOMOUS, CarMode.PATHFINDING]:
            self.path_memory.start_recording({'mode': mode.value})

        # Start control loop
        self._control_thread = Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

        logger.info(f"Smart Car started in {mode.value} mode")

    def stop(self, path_success: bool = True):
        """Stop the car"""
        self.is_running = False
        self._stop_event.set()

        # Stop motors immediately
        self.motors.stop()

        # Stop path recording and save
        if self.path_memory.is_recording:
            recorded_path = self.path_memory.stop_recording(success=path_success)
            if recorded_path and self.on_path_recorded:
                self.on_path_recorded(recorded_path)

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

        # Update position estimate and record path
        self._update_position(command)
        self._record_path_point(detection_result)

    def _update_position(self, command: Dict):
        """Update estimated position based on motor commands"""
        # Simplified dead reckoning - real implementation would use encoders/IMU
        dt = 1.0 / self.control_rate
        speed = self.current_speed / 100.0  # Normalize

        # Estimate movement
        import math
        dx = speed * math.cos(math.radians(self.current_heading)) * dt * 0.1
        dy = speed * math.sin(math.radians(self.current_heading)) * dt * 0.1

        self.current_position = (
            self.current_position[0] + dx,
            self.current_position[1] + dy
        )

        # Update heading based on turning
        if command['action'] in ['avoid_left', 'turn_left']:
            self.current_heading += 5
        elif command['action'] in ['avoid_right', 'turn_right']:
            self.current_heading -= 5

        self.current_heading = self.current_heading % 360

    def _record_path_point(self, detection_result: Dict):
        """Record current position to path memory"""
        if not self.path_memory.is_recording:
            return

        self._position_update_interval += 1
        if self._position_update_interval < self.record_interval:
            return

        self._position_update_interval = 0

        self.path_memory.record_point(
            x=self.current_position[0],
            y=self.current_position[1],
            heading=self.current_heading,
            speed=self.current_speed,
            obstacle_detected=detection_result.get('has_danger', False),
            sensor_data=self.sensors.distances.copy()
        )

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
            'position': self.current_position,
            'heading': self.current_heading,
            'sensor_distances': self.sensors.distances.copy(),
            'has_obstacle': self.avoider.state.value,
            'camera_position': self.servos.get_position(),
            'detection_fps': self.detector.get_fps(),
            'path_memory_stats': self.path_memory.get_statistics(),
            'is_recording_path': self.path_memory.is_recording
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
    def set_goal(self, x: float, y: float, use_memory: bool = True):
        """Set navigation goal in world coordinates"""
        start = self.current_position
        goal = (x, y)

        path = None

        # Try to use historical path first
        if use_memory:
            path = self.path_memory.get_preferred_route(start, goal)
            if path:
                logger.info(f"Using historical path with {len(path)} waypoints")

        # Fall back to A* pathfinding
        if not path:
            # Add known obstacle hotspots to grid
            avoidance_zones = self.path_memory.get_avoidance_zones()
            for zone_x, zone_y, radius in avoidance_zones:
                self.pathfinder.grid_map.add_obstacle_world(zone_x, zone_y, radius)

            path = self.pathfinder.find_path_world(start, goal)

        if path:
            self.current_path = list(path)
            self.mode = CarMode.PATHFINDING
            logger.info(f"Path found with {len(path)} waypoints")
        else:
            logger.warning("No path found to goal")

    def add_obstacle(self, x: float, y: float, radius: float = 0.2):
        """Add obstacle to pathfinding grid"""
        self.pathfinder.grid_map.add_obstacle_world(x, y, radius)

    # Path memory methods
    def get_path_memory_export(self) -> Dict:
        """Export path memory for transmission to computer"""
        return self.path_memory.export_for_transmission()

    def import_path_data(self, data: Dict):
        """Import path data from computer"""
        self.path_memory.import_from_computer(data)

    def get_obstacle_hotspots(self):
        """Get known obstacle hotspots from path memory"""
        return self.path_memory.get_obstacle_hotspots()

    def get_historical_paths(self, start: Tuple[float, float],
                            goal: Tuple[float, float]):
        """Get successful historical paths between points"""
        return self.path_memory.get_successful_paths(start, goal)

    def reset_position(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        """Reset position tracking"""
        self.current_position = (x, y)
        self.current_heading = heading

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
