"""
Reactive obstacle avoidance for real-time navigation
"""

import logging
import math
from typing import Tuple, Optional, Dict, List
from enum import Enum

from ..hardware.sensors import SensorManager
from ..detection.neoyolo import Detection
from ..config import config

logger = logging.getLogger(__name__)


class AvoidanceState(Enum):
    """Obstacle avoidance state machine states"""
    CLEAR = "clear"
    SLOWING = "slowing"
    AVOIDING = "avoiding"
    STOPPED = "stopped"
    REVERSING = "reversing"


class ObstacleAvoider:
    """
    Reactive obstacle avoidance using sensor fusion
    Combines ultrasonic sensors with camera detections
    """

    def __init__(self, sensor_manager: Optional[SensorManager] = None):
        self.sensors = sensor_manager

        # Safety thresholds
        safety_config = config.safety
        self.min_distance = safety_config.get('min_obstacle_distance', 0.3)
        self.emergency_distance = safety_config.get('emergency_stop_distance', 0.15)

        # Navigation config
        nav_config = config.navigation
        self.max_speed = nav_config.get('max_speed', 0.5)
        self.turn_speed = nav_config.get('turn_speed', 0.3)

        # State
        self.state = AvoidanceState.CLEAR
        self.last_avoidance_direction = None

        # History for decision making
        self.detection_history: List[Dict] = []
        self.max_history = 5

    def process_sensor_data(self) -> Dict:
        """Read and process all sensor data"""
        data = {
            'front': float('inf'),
            'left': float('inf'),
            'right': float('inf'),
            'min_distance': float('inf'),
            'obstacle_direction': None
        }

        if self.sensors and self.sensors.is_initialized:
            distances = self.sensors.read_all_distances()
            data['front'] = distances.get('front', float('inf'))
            data['left'] = distances.get('left', float('inf'))
            data['right'] = distances.get('right', float('inf'))
            data['min_distance'] = min(distances.values()) if distances else float('inf')

            # Determine obstacle direction
            if data['min_distance'] < self.min_distance:
                min_dir = min(distances, key=distances.get)
                data['obstacle_direction'] = min_dir

        return data

    def process_detections(self, detection_result: Dict) -> Dict:
        """
        Process detection results from NeoYolo
        Returns avoidance recommendation
        """
        result = {
            'has_obstacle': False,
            'severity': 'none',
            'direction': None,
            'distance_estimate': float('inf'),
            'recommended_action': 'continue'
        }

        if not detection_result:
            return result

        # Check for obstacles in danger zone
        if detection_result.get('has_danger'):
            result['has_obstacle'] = True
            result['severity'] = 'danger'
            result['recommended_action'] = 'stop'

            if detection_result.get('avoid_direction'):
                result['direction'] = detection_result['avoid_direction']

            # Estimate distance from bounding box size
            primary = detection_result.get('primary_obstacle')
            if primary:
                # Larger box = closer object
                frame_area = 640 * 480  # Assume standard resolution
                relative_size = primary.area / frame_area

                if relative_size > 0.3:
                    result['distance_estimate'] = 0.2
                elif relative_size > 0.1:
                    result['distance_estimate'] = 0.5
                else:
                    result['distance_estimate'] = 1.0

        elif detection_result.get('has_warning'):
            result['has_obstacle'] = True
            result['severity'] = 'warning'
            result['recommended_action'] = 'slow'
            result['direction'] = detection_result.get('avoid_direction')

        return result

    def calculate_avoidance(self, sensor_data: Dict,
                           detection_data: Dict) -> Tuple[float, float, str]:
        """
        Calculate motor speeds for obstacle avoidance
        Uses sensor fusion from ultrasonic and camera

        Returns:
            (left_speed, right_speed, action)
        """
        # Default: full speed ahead
        left_speed = 100.0
        right_speed = 100.0
        action = 'forward'

        # Get distances
        front_dist = sensor_data.get('front', float('inf'))
        left_dist = sensor_data.get('left', float('inf'))
        right_dist = sensor_data.get('right', float('inf'))

        # Combine with camera detection
        detection_dist = detection_data.get('distance_estimate', float('inf'))
        detection_severity = detection_data.get('severity', 'none')
        detection_direction = detection_data.get('direction')

        # Use minimum of sensor and detection distances
        effective_front = min(front_dist, detection_dist)

        # State machine logic
        if effective_front < self.emergency_distance:
            # Emergency stop
            self.state = AvoidanceState.STOPPED
            left_speed = 0
            right_speed = 0
            action = 'emergency_stop'
            logger.warning(f"Emergency stop! Distance: {effective_front:.2f}m")

        elif effective_front < self.min_distance or detection_severity == 'danger':
            # Need to avoid
            self.state = AvoidanceState.AVOIDING

            # Decide avoidance direction
            if detection_direction == 'left' or left_dist < right_dist:
                # Turn right
                left_speed = 50
                right_speed = -30
                action = 'avoid_right'
                self.last_avoidance_direction = 'right'

            elif detection_direction == 'right' or right_dist < left_dist:
                # Turn left
                left_speed = -30
                right_speed = 50
                action = 'avoid_left'
                self.last_avoidance_direction = 'left'

            else:
                # Obstacle straight ahead, reverse slightly then turn
                left_speed = -30
                right_speed = -30
                action = 'reverse'
                self.state = AvoidanceState.REVERSING

        elif effective_front < self.min_distance * 2 or detection_severity == 'warning':
            # Slow down
            self.state = AvoidanceState.SLOWING
            speed_factor = effective_front / (self.min_distance * 2)
            left_speed = int(100 * speed_factor)
            right_speed = int(100 * speed_factor)
            action = 'slow'

        else:
            # Clear path
            self.state = AvoidanceState.CLEAR
            action = 'forward'

        return (left_speed, right_speed, action)

    def get_avoidance_command(self, sensor_data: Dict = None,
                              detection_result: Dict = None) -> Dict:
        """
        Get complete avoidance command with all information

        Returns:
            Dict with motor speeds, action, state info
        """
        # Get sensor data if not provided
        if sensor_data is None:
            sensor_data = self.process_sensor_data()

        # Process detections
        if detection_result:
            detection_data = self.process_detections(detection_result)
        else:
            detection_data = {}

        # Calculate avoidance
        left_speed, right_speed, action = self.calculate_avoidance(
            sensor_data, detection_data
        )

        # Build command
        command = {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'action': action,
            'state': self.state.value,
            'sensor_data': sensor_data,
            'detection_data': detection_data
        }

        # Update history
        self.detection_history.append(command)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        return command

    def should_stop(self, sensor_data: Dict = None,
                    detection_result: Dict = None) -> bool:
        """Quick check if car should stop"""
        if sensor_data is None:
            sensor_data = self.process_sensor_data()

        front_dist = sensor_data.get('front', float('inf'))

        if front_dist < self.emergency_distance:
            return True

        if detection_result and detection_result.get('has_danger'):
            return True

        return False

    def get_safe_direction(self) -> str:
        """Get the safest direction to turn based on sensor data"""
        sensor_data = self.process_sensor_data()

        left_dist = sensor_data.get('left', float('inf'))
        right_dist = sensor_data.get('right', float('inf'))

        if right_dist > left_dist:
            return 'right'
        elif left_dist > right_dist:
            return 'left'
        else:
            return 'back'

    def reset_state(self):
        """Reset avoidance state"""
        self.state = AvoidanceState.CLEAR
        self.last_avoidance_direction = None
        self.detection_history.clear()
