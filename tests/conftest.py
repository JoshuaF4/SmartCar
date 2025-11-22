"""
Pytest fixtures and configuration for SmartCar tests
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock cv2 before any imports
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = MagicMock()

try:
    import numpy as np
except ImportError:
    np = None


# Mock GPIO for testing on non-Pi systems
@pytest.fixture(autouse=True)
def mock_gpio():
    """Mock RPi.GPIO for all tests"""
    with patch.dict('sys.modules', {'RPi': MagicMock(), 'RPi.GPIO': MagicMock()}):
        yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_frame():
    """Create a sample camera frame for testing"""
    if np is None:
        pytest.skip("numpy not available")
    # Create a 640x480 BGR image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some features
    frame[200:300, 300:400] = [0, 255, 0]  # Green box
    return frame


@pytest.fixture
def sample_detection_result():
    """Sample detection result for testing"""
    return {
        'danger': [],
        'warning': [],
        'safe': [],
        'has_danger': False,
        'has_warning': False,
        'primary_obstacle': None,
        'suggested_action': 'continue',
        'avoid_direction': None
    }


@pytest.fixture
def sample_path_points():
    """Sample path points for testing"""
    return [
        {'x': 0.0, 'y': 0.0, 'timestamp': 1000.0, 'heading': 0.0,
         'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}},
        {'x': 0.1, 'y': 0.0, 'timestamp': 1001.0, 'heading': 0.0,
         'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}},
        {'x': 0.2, 'y': 0.1, 'timestamp': 1002.0, 'heading': 45.0,
         'speed': 45.0, 'obstacle_detected': True, 'sensor_data': {'front': 0.3}},
        {'x': 0.3, 'y': 0.2, 'timestamp': 1003.0, 'heading': 45.0,
         'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}},
        {'x': 0.4, 'y': 0.3, 'timestamp': 1004.0, 'heading': 45.0,
         'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}},
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config_data = {
        'hardware': {
            'motors': {
                'left_forward': 17,
                'left_backward': 27,
                'right_forward': 22,
                'right_backward': 23,
                'left_pwm': 12,
                'right_pwm': 13,
                'pwm_frequency': 1000
            },
            'servos': {
                'i2c_address': 0x40,
                'camera_pan': 0,
                'camera_tilt': 1,
                'pan_min': 0,
                'pan_max': 180,
                'tilt_min': 0,
                'tilt_max': 90,
                'default_pan': 90,
                'default_tilt': 45
            },
            'sensors': {
                'front': {'trigger': 5, 'echo': 6},
                'left': {'trigger': 19, 'echo': 26},
                'right': {'trigger': 20, 'echo': 21}
            }
        },
        'camera': {
            'resolution': {'width': 640, 'height': 480},
            'framerate': 30,
            'format': 'RGB888'
        },
        'detection': {
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'classes': ['person', 'car', 'obstacle']
        },
        'navigation': {
            'grid_resolution': 0.1,
            'robot_radius': 0.15,
            'safety_margin': 0.1,
            'pathfinding': {
                'heuristic_weight': 1.0,
                'diagonal_movement': True,
                'smoothing': True
            }
        },
        'network': {
            'local': {'host': '0.0.0.0', 'websocket_port': 8765, 'http_port': 5000},
            'computer': {'host': '192.168.1.100', 'port': 9000}
        },
        'safety': {
            'min_obstacle_distance': 0.3,
            'emergency_stop_distance': 0.15
        }
    }
    return config_data
