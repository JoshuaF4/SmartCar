"""
Unit tests for path memory module
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock cv2 before importing
sys.modules['cv2'] = MagicMock()

# Import directly from module to avoid circular imports
from src.navigation.path_memory import PathMemory, PathPoint, RecordedPath


class TestPathPoint:
    """Tests for PathPoint dataclass"""

    def test_create_path_point(self):
        """Test creating a PathPoint"""
        point = PathPoint(
            x=1.0, y=2.0, timestamp=1000.0,
            heading=45.0, speed=50.0,
            obstacle_detected=False, sensor_data={}
        )

        assert point.x == 1.0
        assert point.y == 2.0
        assert point.heading == 45.0
        assert point.speed == 50.0
        assert point.obstacle_detected is False

    def test_path_point_defaults(self):
        """Test PathPoint default values"""
        point = PathPoint(x=0.0, y=0.0, timestamp=0.0)

        assert point.heading == 0.0
        assert point.speed == 0.0
        assert point.obstacle_detected is False
        assert point.sensor_data == {}


class TestRecordedPath:
    """Tests for RecordedPath dataclass"""

    def test_create_recorded_path(self, sample_path_points):
        """Test creating a RecordedPath"""
        points = [PathPoint(**p) for p in sample_path_points]

        path = RecordedPath(
            path_id="test_path_001",
            start_time=1000.0,
            end_time=1004.0,
            points=points,
            success=True,
            total_distance=0.5
        )

        assert path.path_id == "test_path_001"
        assert len(path.points) == 5
        assert path.success is True
        assert path.total_distance == 0.5

    def test_to_dict(self, sample_path_points):
        """Test converting RecordedPath to dict"""
        points = [PathPoint(**p) for p in sample_path_points]

        path = RecordedPath(
            path_id="test_path",
            start_time=1000.0,
            end_time=1004.0,
            points=points,
            success=True
        )

        data = path.to_dict()

        assert data['path_id'] == "test_path"
        assert len(data['points']) == 5
        assert data['success'] is True

    def test_from_dict(self, sample_path_points):
        """Test creating RecordedPath from dict"""
        data = {
            'path_id': 'test_path',
            'start_time': 1000.0,
            'end_time': 1004.0,
            'points': sample_path_points,
            'success': True,
            'total_distance': 0.5,
            'obstacles_encountered': 1
        }

        path = RecordedPath.from_dict(data)

        assert path.path_id == 'test_path'
        assert len(path.points) == 5
        assert path.success is True
        assert path.total_distance == 0.5


class TestPathMemory:
    """Tests for PathMemory class"""

    @pytest.fixture
    def path_memory(self, temp_dir):
        """Create PathMemory instance with temp storage"""
        with patch('src.navigation.path_memory.config') as mock_config:
            mock_config.navigation.get.return_value = 0.1
            memory = PathMemory(storage_path=str(temp_dir))
        return memory

    def test_init(self, path_memory):
        """Test PathMemory initialization"""
        assert path_memory.is_recording is False
        assert path_memory.current_path is None
        assert len(path_memory.paths) == 0

    def test_start_recording(self, path_memory):
        """Test starting path recording"""
        path_id = path_memory.start_recording({'mode': 'autonomous'})

        assert path_memory.is_recording is True
        assert path_memory.current_path is not None
        assert path_id.startswith('path_')
        assert path_memory.current_path.metadata['mode'] == 'autonomous'

    def test_record_point(self, path_memory):
        """Test recording points"""
        path_memory.start_recording()

        path_memory.record_point(x=0.0, y=0.0, heading=0.0, speed=50.0)
        path_memory.record_point(x=0.1, y=0.0, heading=0.0, speed=50.0)
        path_memory.record_point(x=0.2, y=0.1, heading=45.0, speed=45.0, obstacle_detected=True)

        assert len(path_memory.current_points) == 3
        assert path_memory.current_path.obstacles_encountered == 1

    def test_stop_recording(self, path_memory):
        """Test stopping recording and saving path"""
        path_memory.start_recording()

        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=0.1, y=0.0)
        path_memory.record_point(x=0.2, y=0.1)

        recorded = path_memory.stop_recording(success=True)

        assert path_memory.is_recording is False
        assert recorded is not None
        assert len(recorded.points) == 3
        assert recorded.success is True
        assert recorded.path_id in path_memory.paths

    def test_calculate_distance(self, path_memory):
        """Test path distance calculation"""
        path_memory.start_recording()

        # Create a simple path
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=1.0, y=0.0)
        path_memory.record_point(x=1.0, y=1.0)

        recorded = path_memory.stop_recording()

        # Expected distance: 1.0 + 1.0 = 2.0
        assert abs(recorded.total_distance - 2.0) < 0.01

    def test_save_and_load_path(self, path_memory, temp_dir):
        """Test saving and loading paths"""
        # Record a path
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=0.5, y=0.5)
        recorded = path_memory.stop_recording()

        # Check file exists
        path_file = temp_dir / f"{recorded.path_id}.json"
        assert path_file.exists()

        # Load data and verify
        with open(path_file, 'r') as f:
            data = json.load(f)

        assert data['path_id'] == recorded.path_id
        assert len(data['points']) == 2

    def test_get_paths_near(self, path_memory):
        """Test finding paths near a point"""
        # Record first path
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=0.5, y=0.5)
        path_memory.stop_recording()

        # Record second path (different area)
        path_memory.start_recording()
        path_memory.record_point(x=5.0, y=5.0)
        path_memory.record_point(x=5.5, y=5.5)
        path_memory.stop_recording()

        # Find paths near origin
        nearby = path_memory.get_paths_near(0.0, 0.0, radius=1.0)
        assert len(nearby) == 1

        # Find paths near (5, 5)
        nearby = path_memory.get_paths_near(5.0, 5.0, radius=1.0)
        assert len(nearby) == 1

    def test_get_successful_paths(self, path_memory):
        """Test getting successful paths between points"""
        # Record successful path with clear endpoints
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=0.5, y=0.5)
        path_memory.record_point(x=1.0, y=1.0)
        recorded = path_memory.stop_recording(success=True)

        # Verify the path was stored as successful
        assert path_memory.paths[recorded.path_id].success is True

        # Get successful paths with generous tolerance
        successful = path_memory.get_successful_paths(
            start=(0.0, 0.0),
            goal=(1.0, 1.0),
            tolerance=1.5
        )

        # Should find the successful path
        assert len(successful) >= 1
        assert all(p.success for p in successful)

    def test_get_obstacle_hotspots(self, path_memory):
        """Test identifying obstacle hotspots"""
        # Record multiple paths with obstacles at same location
        for _ in range(3):
            path_memory.start_recording()
            path_memory.record_point(x=0.0, y=0.0)
            path_memory.record_point(x=0.5, y=0.5, obstacle_detected=True)
            path_memory.record_point(x=1.0, y=1.0)
            path_memory.stop_recording()

        # Use min_encounters=1 to catch any hotspots
        hotspots = path_memory.get_obstacle_hotspots(min_encounters=1)

        assert len(hotspots) > 0
        # Check hotspot is near (0.5, 0.5)
        hotspot = hotspots[0]
        assert abs(hotspot[0] - 0.5) < 0.2
        assert abs(hotspot[1] - 0.5) < 0.2
        assert hotspot[2] >= 1  # count

    def test_get_preferred_route(self, path_memory):
        """Test getting preferred route from history"""
        # Record a successful path
        path_memory.start_recording()
        for i in range(5):
            path_memory.record_point(x=i * 0.2, y=i * 0.2)
        path_memory.stop_recording(success=True)

        # Get preferred route
        route = path_memory.get_preferred_route(
            start=(0.0, 0.0),
            goal=(0.8, 0.8)
        )

        assert route is not None
        assert len(route) > 0
        # First point should be near start
        assert abs(route[0][0]) < 0.1
        assert abs(route[0][1]) < 0.1

    def test_get_avoidance_zones(self, path_memory):
        """Test generating avoidance zones"""
        # Record failed path - this should create an avoidance zone
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=0.5, y=0.5, obstacle_detected=True)
        path_memory.stop_recording(success=False)

        zones = path_memory.get_avoidance_zones()

        # Should have at least the failure point as a zone
        assert len(zones) > 0
        # Each zone should be (x, y, radius)
        assert len(zones[0]) == 3

    def test_export_for_transmission(self, path_memory):
        """Test exporting path memory for network transmission"""
        # Record some paths
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=1.0, y=1.0)
        path_memory.stop_recording()

        export = path_memory.export_for_transmission()

        assert 'total_paths' in export
        assert 'paths' in export
        assert 'hotspots' in export
        assert 'avoidance_zones' in export
        assert 'timestamp' in export
        assert export['total_paths'] == 1

    def test_import_from_computer(self, path_memory):
        """Test importing processed data from computer"""
        # Create optimized path data
        data = {
            'optimized_paths': [{
                'path_id': 'optimized_001',
                'start_time': 1000.0,
                'end_time': 1005.0,
                'points': [
                    {'x': 0.0, 'y': 0.0, 'timestamp': 1000.0, 'heading': 0.0,
                     'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}},
                    {'x': 1.0, 'y': 1.0, 'timestamp': 1005.0, 'heading': 45.0,
                     'speed': 50.0, 'obstacle_detected': False, 'sensor_data': {}}
                ],
                'success': True,
                'total_distance': 1.4,
                'obstacles_encountered': 0,
                'metadata': {}
            }],
            'avoidance_zones': [[0.5, 0.5, 0.3]]
        }

        path_memory.import_from_computer(data)

        assert 'optimized_001' in path_memory.paths
        assert path_memory.paths['optimized_001'].metadata['source'] == 'computer'

    def test_get_statistics(self, path_memory):
        """Test getting path memory statistics"""
        # Record a path with some distance
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.record_point(x=1.0, y=0.0)
        path_memory.stop_recording(success=True)

        stats = path_memory.get_statistics()

        assert stats['total_paths'] == 1
        assert stats['total_distance'] > 0
        # Distance should be approximately 1.0 (from 0,0 to 1,0)
        assert abs(stats['total_distance'] - 1.0) < 0.1

    def test_grid_conversion(self, path_memory):
        """Test world to grid coordinate conversion"""
        # Test conversion
        grid_cell = path_memory._to_grid_cell(0.55, 0.73)

        # With 0.1 resolution: 0.55 -> 5, 0.73 -> 7
        assert grid_cell == (5, 7)

    def test_no_recording_when_not_started(self, path_memory):
        """Test that recording doesn't happen when not started"""
        path_memory.record_point(x=1.0, y=1.0)

        assert len(path_memory.current_points) == 0

    def test_clear_old_paths(self, path_memory):
        """Test clearing old paths"""
        # Record a path
        path_memory.start_recording()
        path_memory.record_point(x=0.0, y=0.0)
        path_memory.stop_recording()

        # Manually set old timestamp
        path_id = list(path_memory.paths.keys())[0]
        path_memory.paths[path_id].end_time = 0  # Very old

        # Clear paths older than 1 day
        initial_count = len(path_memory.paths)
        path_memory.clear_old_paths(max_age_days=1)

        # Path should be removed
        assert len(path_memory.paths) < initial_count
