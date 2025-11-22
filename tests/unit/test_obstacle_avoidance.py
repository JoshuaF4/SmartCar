"""
Unit tests for obstacle avoidance module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.navigation.obstacle_avoidance import ObstacleAvoider, AvoidanceState


class TestObstacleAvoider:
    """Tests for ObstacleAvoider class"""

    @pytest.fixture
    def avoider(self):
        """Create avoider with mocked sensor manager"""
        with patch('src.navigation.obstacle_avoidance.config') as mock_config:
            mock_config.safety.get.side_effect = lambda key, default=None: {
                'min_obstacle_distance': 0.3,
                'emergency_stop_distance': 0.15
            }.get(key, default)
            mock_config.navigation.get.side_effect = lambda key, default=None: {
                'max_speed': 0.5,
                'turn_speed': 0.3
            }.get(key, default)

            mock_sensor = MagicMock()
            mock_sensor.is_initialized = True
            mock_sensor.read_all_distances.return_value = {
                'front': 1.0,
                'left': 1.0,
                'right': 1.0
            }

            avoider = ObstacleAvoider(mock_sensor)
        return avoider

    def test_init(self, avoider):
        """Test avoider initialization"""
        assert avoider.state == AvoidanceState.CLEAR
        assert avoider.min_distance == 0.3
        assert avoider.emergency_distance == 0.15

    def test_process_sensor_data(self, avoider):
        """Test processing sensor data"""
        data = avoider.process_sensor_data()

        assert 'front' in data
        assert 'left' in data
        assert 'right' in data
        assert 'min_distance' in data

    def test_process_detections_no_obstacle(self, avoider):
        """Test processing detections with no obstacles"""
        detection_result = {
            'has_danger': False,
            'has_warning': False,
            'primary_obstacle': None,
            'avoid_direction': None
        }

        result = avoider.process_detections(detection_result)

        assert result['has_obstacle'] is False
        assert result['severity'] == 'none'
        assert result['recommended_action'] == 'continue'

    def test_process_detections_danger(self, avoider):
        """Test processing detections with danger"""
        # Create mock detection
        mock_obstacle = MagicMock()
        mock_obstacle.area = 50000  # Large obstacle
        mock_obstacle.center = (320, 400)

        detection_result = {
            'has_danger': True,
            'has_warning': False,
            'primary_obstacle': mock_obstacle,
            'avoid_direction': 'right'
        }

        result = avoider.process_detections(detection_result)

        assert result['has_obstacle'] is True
        assert result['severity'] == 'danger'
        assert result['recommended_action'] == 'stop'
        assert result['direction'] == 'right'

    def test_calculate_avoidance_clear(self, avoider):
        """Test avoidance calculation when clear"""
        sensor_data = {
            'front': 1.0,
            'left': 1.0,
            'right': 1.0,
            'min_distance': 1.0
        }

        detection_data = {
            'distance_estimate': float('inf'),
            'severity': 'none'
        }

        left, right, action = avoider.calculate_avoidance(sensor_data, detection_data)

        assert left == 100.0
        assert right == 100.0
        assert action == 'forward'
        assert avoider.state == AvoidanceState.CLEAR

    def test_calculate_avoidance_emergency_stop(self, avoider):
        """Test emergency stop when too close"""
        sensor_data = {
            'front': 0.1,  # Less than emergency distance
            'left': 1.0,
            'right': 1.0,
            'min_distance': 0.1
        }

        detection_data = {}

        left, right, action = avoider.calculate_avoidance(sensor_data, detection_data)

        assert left == 0
        assert right == 0
        assert action == 'emergency_stop'
        assert avoider.state == AvoidanceState.STOPPED

    def test_calculate_avoidance_turn_right(self, avoider):
        """Test turning right to avoid obstacle on left"""
        sensor_data = {
            'front': 0.25,  # Less than min_distance
            'left': 0.2,    # Closer on left
            'right': 1.0,
            'min_distance': 0.2
        }

        detection_data = {
            'distance_estimate': 0.25,
            'severity': 'danger',
            'direction': 'left'
        }

        left, right, action = avoider.calculate_avoidance(sensor_data, detection_data)

        assert action == 'avoid_right'
        assert left > right  # Turning right means left faster

    def test_calculate_avoidance_turn_left(self, avoider):
        """Test turning left to avoid obstacle on right"""
        sensor_data = {
            'front': 0.25,
            'left': 1.0,
            'right': 0.2,  # Closer on right
            'min_distance': 0.2
        }

        detection_data = {
            'distance_estimate': 0.25,
            'severity': 'danger',
            'direction': 'right'
        }

        left, right, action = avoider.calculate_avoidance(sensor_data, detection_data)

        assert action == 'avoid_left'
        assert right > left  # Turning left means right faster

    def test_calculate_avoidance_slow_down(self, avoider):
        """Test slowing down in warning zone"""
        sensor_data = {
            'front': 0.5,  # Between min and 2*min
            'left': 1.0,
            'right': 1.0,
            'min_distance': 0.5
        }

        detection_data = {
            'distance_estimate': 0.5,
            'severity': 'warning'
        }

        left, right, action = avoider.calculate_avoidance(sensor_data, detection_data)

        assert action == 'slow'
        assert avoider.state == AvoidanceState.SLOWING
        assert left < 100  # Slowed down
        assert right < 100

    def test_get_avoidance_command(self, avoider):
        """Test getting full avoidance command"""
        command = avoider.get_avoidance_command()

        assert 'left_speed' in command
        assert 'right_speed' in command
        assert 'action' in command
        assert 'state' in command
        assert 'sensor_data' in command

    def test_should_stop_emergency(self, avoider):
        """Test should_stop for emergency"""
        sensor_data = {
            'front': 0.1,
            'min_distance': 0.1
        }

        assert avoider.should_stop(sensor_data) is True

    def test_should_stop_detection_danger(self, avoider):
        """Test should_stop with detection danger"""
        detection_result = {'has_danger': True}

        assert avoider.should_stop(detection_result=detection_result) is True

    def test_should_stop_clear(self, avoider):
        """Test should_stop when clear"""
        sensor_data = {
            'front': 1.0,
            'min_distance': 1.0
        }

        assert avoider.should_stop(sensor_data) is False

    def test_get_safe_direction(self, avoider):
        """Test getting safest direction"""
        avoider.sensors.read_all_distances.return_value = {
            'front': 0.5,
            'left': 0.3,
            'right': 1.0
        }

        direction = avoider.get_safe_direction()

        assert direction == 'right'

    def test_reset_state(self, avoider):
        """Test resetting avoider state"""
        avoider.state = AvoidanceState.AVOIDING
        avoider.last_avoidance_direction = 'left'
        avoider.detection_history = [{'test': 1}]

        avoider.reset_state()

        assert avoider.state == AvoidanceState.CLEAR
        assert avoider.last_avoidance_direction is None
        assert len(avoider.detection_history) == 0

    def test_detection_history(self, avoider):
        """Test that detection history is maintained"""
        for _ in range(10):
            avoider.get_avoidance_command()

        # History should be limited
        assert len(avoider.detection_history) <= avoider.max_history
