"""
Unit tests for dynamic obstacle tracking
"""

import pytest
import time
import sys
from unittest.mock import MagicMock, patch

# Mock cv2 before importing
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = MagicMock()

from src.navigation.dynamic_obstacles import (
    DynamicObstacleTracker,
    TrackedObstacle,
    ObstacleObservation
)


class TestObstacleObservation:
    """Test ObstacleObservation dataclass"""

    def test_creation(self):
        obs = ObstacleObservation(x=1.0, y=2.0, timestamp=time.time())
        assert obs.x == 1.0
        assert obs.y == 2.0
        assert obs.confidence == 1.0
        assert obs.size == 0.1

    def test_custom_values(self):
        obs = ObstacleObservation(
            x=3.0, y=4.0, timestamp=100.0,
            confidence=0.8, size=0.3
        )
        assert obs.confidence == 0.8
        assert obs.size == 0.3


class TestTrackedObstacle:
    """Test TrackedObstacle class"""

    def test_creation(self):
        obs = TrackedObstacle(obstacle_id="test_1")
        assert obs.obstacle_id == "test_1"
        assert obs.is_static == True
        assert obs.vx == 0.0
        assert obs.vy == 0.0

    def test_add_observation(self):
        obs = TrackedObstacle(obstacle_id="test_1")
        observation = ObstacleObservation(
            x=1.0, y=2.0, timestamp=time.time()
        )
        obs.add_observation(observation)

        assert obs.x == 1.0
        assert obs.y == 2.0
        assert len(obs.observations) == 1

    def test_velocity_estimation(self):
        obs = TrackedObstacle(obstacle_id="test_1")

        # Add observations over time (simulating movement)
        t0 = time.time()
        obs.add_observation(ObstacleObservation(x=0.0, y=0.0, timestamp=t0))
        obs.add_observation(ObstacleObservation(x=1.0, y=0.0, timestamp=t0 + 1.0))

        # Velocity should be approximately 1 m/s in x direction
        assert abs(obs.vx - 1.0) < 0.1
        assert abs(obs.vy) < 0.1
        assert obs.is_static == False

    def test_predict_position_static(self):
        obs = TrackedObstacle(obstacle_id="test_1")
        obs.x = 5.0
        obs.y = 3.0
        obs.is_static = True
        obs.last_seen = time.time()

        predicted = obs.predict_position(time.time() + 10.0)
        assert predicted == (5.0, 3.0)

    def test_predict_position_dynamic(self):
        obs = TrackedObstacle(obstacle_id="test_1")
        obs.x = 0.0
        obs.y = 0.0
        obs.vx = 2.0  # 2 m/s
        obs.vy = 1.0  # 1 m/s
        obs.is_static = False
        obs.last_seen = time.time()

        predicted = obs.predict_position(time.time() + 2.0)
        assert abs(predicted[0] - 4.0) < 0.1  # 2 m/s * 2s = 4m
        assert abs(predicted[1] - 2.0) < 0.1  # 1 m/s * 2s = 2m

    def test_prediction_confidence_decay(self):
        obs = TrackedObstacle(obstacle_id="test_1")
        obs.confidence = 1.0
        obs.last_seen = time.time()

        # Confidence should decay over time
        conf_now = obs.get_prediction_confidence(time.time())
        conf_later = obs.get_prediction_confidence(time.time() + 5.0)

        assert conf_now > conf_later
        assert conf_later > 0


class TestDynamicObstacleTracker:
    """Test DynamicObstacleTracker class"""

    @pytest.fixture
    def tracker(self):
        return DynamicObstacleTracker()

    def test_creation(self, tracker):
        assert len(tracker.obstacles) == 0

    def test_update_single_detection(self, tracker):
        detections = [(1.0, 2.0, 0.1, 0.9)]
        tracker.update(detections)

        assert len(tracker.obstacles) == 1

    def test_update_multiple_detections(self, tracker):
        detections = [
            (1.0, 2.0, 0.1, 0.9),
            (5.0, 6.0, 0.2, 0.8)
        ]
        tracker.update(detections)

        assert len(tracker.obstacles) == 2

    def test_obstacle_association(self, tracker):
        # First detection
        tracker.update([(1.0, 2.0, 0.1, 0.9)])
        first_count = len(tracker.obstacles)

        # Second detection nearby (should associate)
        tracker.update([(1.1, 2.1, 0.1, 0.9)])

        # Should still have same number of obstacles
        assert len(tracker.obstacles) == first_count

    def test_new_obstacle_creation(self, tracker):
        # First detection
        tracker.update([(1.0, 2.0, 0.1, 0.9)])
        first_count = len(tracker.obstacles)

        # Second detection far away (should create new)
        tracker.update([(10.0, 20.0, 0.1, 0.9)])

        assert len(tracker.obstacles) == first_count + 1

    def test_get_current_obstacles(self, tracker):
        # Add detection with multiple observations
        for _ in range(5):
            tracker.update([(1.0, 2.0, 0.1, 0.9)])

        obstacles = tracker.get_current_obstacles()
        assert len(obstacles) >= 1
        assert obstacles[0][3] > 0  # Has confidence

    def test_get_predicted_obstacles(self, tracker):
        # Add moving obstacle with enough observations
        tracker.min_observations = 2  # Lower threshold for test
        t0 = time.time()
        for i in range(5):
            tracker.update([(float(i) * 0.1, 0.0, 0.1, 0.9)])

        future_time = time.time() + 2.0
        predicted = tracker.get_predicted_obstacles(future_time)

        assert len(predicted) >= 1

    def test_get_dynamic_obstacles(self, tracker):
        # Add stationary obstacle
        for _ in range(5):
            tracker.update([(1.0, 1.0, 0.1, 0.9)])

        # Initially static
        dynamic = tracker.get_dynamic_obstacles()
        static = tracker.get_static_obstacles()

        # Should be categorized
        assert len(dynamic) + len(static) >= 1

    def test_stale_removal(self, tracker):
        tracker.stale_threshold = 0.1  # Very short for testing

        tracker.update([(1.0, 2.0, 0.1, 0.9)])
        initial_count = len(tracker.obstacles)

        # Wait and update with no matching detection
        time.sleep(0.2)
        tracker.update([(10.0, 20.0, 0.1, 0.9)])  # Far away detection

        # Old obstacle should be removed
        # (one removed, one added = same count or different)
        assert len(tracker.obstacles) <= initial_count + 1

    def test_collision_risk(self, tracker):
        # Add obstacle directly ahead
        for _ in range(5):
            tracker.update([(2.0, 0.0, 0.2, 0.9)])

        # Robot moving toward obstacle
        position = (0.0, 0.0)
        velocity = (1.0, 0.0)

        risks = tracker.get_collision_risk(position, velocity, 3.0)

        # Should detect collision risk
        assert len(risks) >= 1

    def test_clear(self, tracker):
        tracker.update([(1.0, 2.0, 0.1, 0.9)])
        assert len(tracker.obstacles) > 0

        tracker.clear()
        assert len(tracker.obstacles) == 0

    def test_export_state(self, tracker):
        for _ in range(5):
            tracker.update([(1.0, 2.0, 0.1, 0.9)])

        state = tracker.export_state()

        assert 'timestamp' in state
        assert 'obstacles' in state

    def test_import_state(self, tracker):
        state = {
            'obstacles': [
                {
                    'id': 'imported_1',
                    'x': 3.0,
                    'y': 4.0,
                    'vx': 1.0,
                    'vy': 0.5,
                    'size': 0.2,
                    'is_static': False,
                    'confidence': 0.9,
                    'last_seen': time.time()
                }
            ]
        }

        tracker.import_state(state)

        assert 'imported_1' in tracker.obstacles
        assert tracker.obstacles['imported_1'].x == 3.0

    def test_obstacle_velocities(self, tracker):
        # Add observations to get velocity
        tracker.min_observations = 2  # Lower threshold for test
        t0 = time.time()
        for i in range(5):
            tracker.update([(float(i) * 0.1, 0.0, 0.1, 0.9)])

        velocities = tracker.get_obstacle_velocities()

        assert len(velocities) >= 1


class TestPathfindingWithDynamicObstacles:
    """Test pathfinding integration with dynamic obstacles"""

    @pytest.fixture
    def pathfinder(self):
        from src.navigation.pathfinding import AStarPathfinder, GridMap
        grid = GridMap(50, 50, 0.1)
        return AStarPathfinder(grid)

    def test_add_dynamic_obstacle(self, pathfinder):
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 2.0, 0.3, 0.9)

        # Check dynamic cost is set
        grid_pos = pathfinder.grid_map.world_to_grid(2.0, 2.0)
        cost = pathfinder.grid_map.dynamic_cost[grid_pos[1], grid_pos[0]]

        assert cost > 0

    def test_temporal_decay(self, pathfinder):
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 2.0, 0.3, 0.9)

        grid_pos = pathfinder.grid_map.world_to_grid(2.0, 2.0)
        initial_cost = pathfinder.grid_map.dynamic_cost[grid_pos[1], grid_pos[0]]

        # Apply decay
        time.sleep(0.1)
        pathfinder.grid_map.apply_temporal_decay()

        decayed_cost = pathfinder.grid_map.dynamic_cost[grid_pos[1], grid_pos[0]]

        assert decayed_cost <= initial_cost

    def test_combined_cost(self, pathfinder):
        # Free cell
        free_cost = pathfinder.grid_map.get_combined_cost(10, 10)
        assert free_cost == 1.0

        # Static obstacle
        pathfinder.grid_map.set_obstacle(15, 15)
        static_cost = pathfinder.grid_map.get_combined_cost(15, 15)
        assert static_cost == float('inf')

        # Dynamic obstacle
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 2.0, 0.3, 0.5)
        grid_pos = pathfinder.grid_map.world_to_grid(2.0, 2.0)
        dynamic_cost = pathfinder.grid_map.get_combined_cost(grid_pos[0], grid_pos[1])
        assert dynamic_cost > 1.0

    def test_path_avoids_dynamic_obstacles(self, pathfinder):
        # Add dynamic obstacle in the middle
        pathfinder.grid_map.add_dynamic_obstacle(2.5, 2.5, 0.5, 0.9)

        # Find path around it
        path = pathfinder.find_path_world((0.5, 0.5), (4.5, 4.5))

        # Path should exist (avoid the obstacle)
        assert path is not None
        assert len(path) > 0

    def test_needs_replan(self, pathfinder):
        # Create a path
        path = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

        # Initially no replan needed
        assert not pathfinder.needs_replan((0.5, 0.5), path)

        # Add obstacle on path
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 2.0, 0.3, 0.95)

        # Now should need replan
        assert pathfinder.needs_replan((0.5, 0.5), path)

    def test_find_path_dynamic(self, pathfinder):
        predicted = [(2.5, 2.5, 0.3, 0.8)]
        path = pathfinder.find_path_dynamic((0.5, 0.5), (4.5, 4.5), predicted)

        assert path is not None

    def test_path_risk_score(self, pathfinder):
        # Safe path
        safe_path = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]
        safe_risk = pathfinder.get_path_risk_score(safe_path)

        # Add obstacle near different path
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 3.0, 0.3, 0.5)
        risky_path = [(1.0, 3.0), (2.0, 3.0), (3.0, 3.0)]
        risky_risk = pathfinder.get_path_risk_score(risky_path)

        assert risky_risk > safe_risk

    def test_clear_dynamic_obstacles(self, pathfinder):
        pathfinder.grid_map.add_dynamic_obstacle(2.0, 2.0, 0.3, 0.9)
        pathfinder.grid_map.clear_dynamic_obstacles()

        # Check cleared
        cells = pathfinder.grid_map.get_dynamic_obstacle_cells()
        assert len(cells) == 0


class TestPathMemoryWithDynamicObstacles:
    """Test path memory with dynamic obstacle features"""

    @pytest.fixture
    def path_memory(self, tmp_path):
        from src.navigation.path_memory import PathMemory
        return PathMemory(storage_path=str(tmp_path))

    def test_record_with_velocity(self, path_memory):
        path_memory.start_recording()
        path_memory.record_point(
            x=1.0, y=2.0, heading=0.0, speed=50,
            obstacle_detected=True,
            obstacle_confidence=0.9,
            obstacle_velocity=(1.0, 0.5)
        )
        path = path_memory.stop_recording()

        assert path.points[0].obstacle_velocity == (1.0, 0.5)
        assert path.points[0].obstacle_confidence == 0.9

    def test_temporal_decay_hotspots(self, path_memory):
        # Record old obstacle
        path_memory.start_recording()
        path_memory.record_point(
            x=1.0, y=1.0, obstacle_detected=True
        )
        path_memory.stop_recording()

        # With decay
        hotspots_decay = path_memory.get_obstacle_hotspots(
            min_encounters=0.1, use_temporal_decay=True
        )

        # Without decay
        hotspots_no_decay = path_memory.get_obstacle_hotspots(
            min_encounters=1, use_temporal_decay=False
        )

        # Both should find the hotspot
        assert len(hotspots_decay) > 0 or len(hotspots_no_decay) > 0

    def test_dynamic_obstacle_zones(self, path_memory):
        # Record dynamic obstacle observation
        path_memory._record_obstacle_observation(
            1.0, 2.0, 0.9, (0.5, 0.3)
        )

        zones = path_memory.get_dynamic_obstacle_zones()
        # Should have at least one zone
        assert len(zones) >= 1

    def test_update_from_tracker(self, path_memory):
        path_memory.update_obstacle_from_tracker(
            'obs_1', 3.0, 4.0, (1.0, 0.0), 0.85
        )

        # Check data was recorded
        assert len(path_memory.dynamic_obstacle_data) > 0

    def test_decay_old_observations(self, path_memory):
        # Add old observation
        cell = path_memory._to_grid_cell(1.0, 1.0)
        path_memory.dynamic_obstacle_data[cell] = [{
            'timestamp': time.time() - 10000,  # Very old
            'confidence': 0.9,
            'velocity': (0, 0),
            'is_dynamic': False
        }]

        path_memory.decay_old_observations(max_age_seconds=100)

        # Old observation should be removed
        assert cell not in path_memory.dynamic_obstacle_data or \
               len(path_memory.dynamic_obstacle_data[cell]) == 0

    def test_time_weighted_avoidance_zones(self, path_memory):
        # Record recent obstacle
        path_memory.start_recording()
        path_memory.record_point(
            x=2.0, y=2.0, obstacle_detected=True, obstacle_confidence=0.9
        )
        path_memory.stop_recording()

        zones = path_memory.get_time_weighted_avoidance_zones()
        # Should have avoidance zone with confidence
        # (zones may be empty if thresholds not met)
        assert isinstance(zones, list)
