"""
Integration tests for SmartCar controller
"""

import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSmartCarController:
    """Integration tests for SmartCarController"""

    @pytest.fixture
    def mock_components(self, temp_dir):
        """Setup all mocked components"""
        with patch.multiple(
            'src.controller',
            MotorController=MagicMock,
            ServoController=MagicMock,
            SensorManager=MagicMock,
            CameraCapture=MagicMock,
            CameraArm=MagicMock,
            NeoYoloDetector=MagicMock
        ), patch('src.navigation.path_memory.config') as mock_pm_config, \
           patch('src.navigation.pathfinding.config') as mock_pf_config, \
           patch('src.navigation.obstacle_avoidance.config') as mock_oa_config:

            # Configure mocks
            mock_pm_config.navigation.get.return_value = 0.1
            mock_pf_config.navigation.get.return_value = 0.1
            mock_oa_config.safety.get.return_value = 0.3
            mock_oa_config.navigation.get.return_value = 0.5

            from src.controller import SmartCarController, CarMode
            from src.navigation.path_memory import PathMemory

            controller = SmartCarController()

            # Setup mock returns
            controller.motors.is_initialized = True
            controller.servos.is_initialized = True
            controller.sensors.is_initialized = True
            controller.sensors.distances = {'front': 1.0, 'left': 1.0, 'right': 1.0}
            controller.servos.get_position.return_value = (90, 45)
            controller.camera.get_frame.return_value = MagicMock()

            # Use real path memory with temp dir
            controller.path_memory = PathMemory(storage_path=str(temp_dir))

            yield controller, CarMode

    def test_initialization(self, mock_components):
        """Test controller initialization"""
        controller, CarMode = mock_components

        controller.initialize()

        assert controller.is_initialized is True
        controller.motors.initialize.assert_called_once()
        controller.servos.initialize.assert_called_once()
        controller.sensors.initialize.assert_called_once()
        controller.camera.initialize.assert_called_once()

    def test_start_autonomous_mode(self, mock_components):
        """Test starting in autonomous mode"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.start(mode=CarMode.AUTONOMOUS, record_path=True)

        assert controller.is_running is True
        assert controller.mode == CarMode.AUTONOMOUS
        assert controller.path_memory.is_recording is True

        controller.stop()

    def test_start_manual_mode(self, mock_components):
        """Test starting in manual mode"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.start(mode=CarMode.MANUAL, record_path=False)

        assert controller.mode == CarMode.MANUAL
        assert controller.path_memory.is_recording is False

        controller.stop()

    def test_stop_saves_path(self, mock_components):
        """Test that stopping saves the recorded path"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.start(mode=CarMode.AUTONOMOUS, record_path=True)

        # Record some data
        controller.path_memory.record_point(x=0.0, y=0.0)
        controller.path_memory.record_point(x=0.5, y=0.5)

        controller.stop(path_success=True)

        assert controller.path_memory.is_recording is False
        assert len(controller.path_memory.paths) == 1

    def test_get_status(self, mock_components):
        """Test getting controller status"""
        controller, CarMode = mock_components

        controller.initialize()
        status = controller.get_status()

        assert 'mode' in status
        assert 'is_running' in status
        assert 'speed' in status
        assert 'position' in status
        assert 'heading' in status
        assert 'path_memory_stats' in status
        assert 'is_recording_path' in status

    def test_manual_controls(self, mock_components):
        """Test manual control methods"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.start(mode=CarMode.MANUAL)

        controller.manual_forward(50)
        controller.motors.forward.assert_called_with(50)

        controller.manual_backward(40)
        controller.motors.backward.assert_called_with(40)

        controller.manual_left(30)
        controller.motors.turn_left.assert_called_with(30)

        controller.manual_right(30)
        controller.motors.turn_right.assert_called_with(30)

        controller.manual_stop()
        controller.motors.stop.assert_called()

        controller.stop()

    def test_camera_controls(self, mock_components):
        """Test camera control methods"""
        controller, CarMode = mock_components

        controller.initialize()

        controller.move_camera(45, 30)
        controller.center_camera()

    def test_set_goal_with_memory(self, mock_components):
        """Test setting navigation goal using path memory"""
        controller, CarMode = mock_components

        controller.initialize()

        # First record a successful path
        controller.path_memory.start_recording()
        for i in range(5):
            controller.path_memory.record_point(x=i * 0.5, y=i * 0.5)
        controller.path_memory.stop_recording(success=True)

        # Now set goal that should use memory
        controller.set_goal(x=2.0, y=2.0, use_memory=True)

        # Should be in pathfinding mode
        assert controller.mode == CarMode.PATHFINDING
        assert controller.current_path is not None

    def test_set_goal_without_memory(self, mock_components):
        """Test setting goal without using memory"""
        controller, CarMode = mock_components

        controller.initialize()

        controller.set_goal(x=1.0, y=1.0, use_memory=False)

        assert controller.mode == CarMode.PATHFINDING

    def test_path_memory_export(self, mock_components):
        """Test exporting path memory"""
        controller, CarMode = mock_components

        controller.initialize()

        # Record a path
        controller.path_memory.start_recording()
        controller.path_memory.record_point(x=0.0, y=0.0)
        controller.path_memory.record_point(x=1.0, y=1.0)
        controller.path_memory.stop_recording()

        export = controller.get_path_memory_export()

        assert export['total_paths'] == 1
        assert len(export['paths']) == 1

    def test_import_path_data(self, mock_components):
        """Test importing path data from computer"""
        controller, CarMode = mock_components

        controller.initialize()

        data = {
            'optimized_paths': [{
                'path_id': 'imported_001',
                'start_time': 1000.0,
                'end_time': 1005.0,
                'points': [
                    {'x': 0.0, 'y': 0.0, 'timestamp': 1000.0, 'heading': 0.0,
                     'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}}
                ],
                'success': True,
                'total_distance': 0.0,
                'obstacles_encountered': 0,
                'metadata': {}
            }]
        }

        controller.import_path_data(data)

        assert 'imported_001' in controller.path_memory.paths

    def test_obstacle_hotspots(self, mock_components):
        """Test getting obstacle hotspots"""
        controller, CarMode = mock_components

        controller.initialize()

        # Record paths with obstacles
        for _ in range(3):
            controller.path_memory.start_recording()
            controller.path_memory.record_point(x=0.5, y=0.5, obstacle_detected=True)
            controller.path_memory.stop_recording()

        hotspots = controller.get_obstacle_hotspots()

        assert len(hotspots) > 0

    def test_reset_position(self, mock_components):
        """Test resetting position tracking"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.current_position = (5.0, 5.0)
        controller.current_heading = 180.0

        controller.reset_position(x=0.0, y=0.0, heading=0.0)

        assert controller.current_position == (0.0, 0.0)
        assert controller.current_heading == 0.0

    def test_emergency_stop(self, mock_components):
        """Test emergency stop"""
        controller, CarMode = mock_components

        controller.initialize()

        controller.emergency_stop()

        controller.motors.emergency_stop.assert_called_once()

    def test_cleanup(self, mock_components):
        """Test cleanup"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.cleanup()

        assert controller.is_initialized is False
        controller.motors.cleanup.assert_called()
        controller.servos.cleanup.assert_called()

    def test_update_position(self, mock_components):
        """Test position update estimation"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.current_speed = 50

        command = {'action': 'forward'}
        controller._update_position(command)

        # Position should have changed
        assert controller.current_position != (0.0, 0.0)

    def test_record_path_point(self, mock_components):
        """Test recording path points"""
        controller, CarMode = mock_components

        controller.initialize()
        controller.path_memory.start_recording()

        # Record multiple points (need to exceed record_interval)
        for i in range(controller.record_interval + 1):
            controller._position_update_interval = i
            detection_result = {'has_danger': False}
            controller._record_path_point(detection_result)

        assert len(controller.path_memory.current_points) == 1

    def test_context_manager(self, mock_components):
        """Test using controller as context manager"""
        controller, CarMode = mock_components

        # Note: Can't use 'with' here due to mock setup
        controller.__enter__()
        assert controller.is_initialized is True

        controller.__exit__(None, None, None)
        assert controller.is_initialized is False
