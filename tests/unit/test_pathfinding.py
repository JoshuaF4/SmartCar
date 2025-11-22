"""
Unit tests for pathfinding module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.navigation.pathfinding import AStarPathfinder, GridMap


class TestGridMap:
    """Tests for GridMap class"""

    def test_init(self):
        """Test GridMap initialization"""
        grid = GridMap(width=100, height=100, resolution=0.1)

        assert grid.width == 100
        assert grid.height == 100
        assert grid.resolution == 0.1
        assert grid.grid.shape == (100, 100)

    def test_set_obstacle(self):
        """Test setting obstacles"""
        grid = GridMap(50, 50, 0.1)

        grid.set_obstacle(10, 10)
        assert grid.grid[10, 10] == 1

        grid.set_obstacle(20, 20)
        assert grid.grid[20, 20] == 1

    def test_clear_obstacle(self):
        """Test clearing obstacles"""
        grid = GridMap(50, 50, 0.1)

        grid.set_obstacle(10, 10)
        grid.clear_obstacle(10, 10)

        assert grid.grid[10, 10] == 0

    def test_is_free(self):
        """Test checking if cell is free"""
        grid = GridMap(50, 50, 0.1)

        assert grid.is_free(10, 10) == True

        grid.set_obstacle(10, 10)
        assert grid.is_free(10, 10) == False

    def test_is_free_out_of_bounds(self):
        """Test is_free for out of bounds"""
        grid = GridMap(50, 50, 0.1)

        assert grid.is_free(-1, 10) is False
        assert grid.is_free(10, 100) is False

    def test_world_to_grid(self):
        """Test world to grid coordinate conversion"""
        grid = GridMap(100, 100, 0.1)

        # 0.55m -> cell 5 (0.55 / 0.1 = 5.5 -> 5)
        x, y = grid.world_to_grid(0.55, 0.73)
        assert x == 5
        assert y == 7

    def test_grid_to_world(self):
        """Test grid to world coordinate conversion"""
        grid = GridMap(100, 100, 0.1)

        # Cell (5, 7) -> (0.55, 0.75) center of cell
        x, y = grid.grid_to_world(5, 7)
        assert abs(x - 0.55) < 0.01
        assert abs(y - 0.75) < 0.01

    def test_add_obstacle_world(self):
        """Test adding obstacle in world coordinates"""
        grid = GridMap(100, 100, 0.1)

        grid.add_obstacle_world(0.5, 0.5, radius=0.2)

        # Check that cells around (5, 5) are marked
        assert grid.grid[5, 5] == 1

    def test_clear(self):
        """Test clearing all obstacles"""
        grid = GridMap(50, 50, 0.1)

        grid.set_obstacle(10, 10)
        grid.set_obstacle(20, 20)
        grid.clear()

        assert grid.grid.sum() == 0

    def test_get_inflated_grid(self):
        """Test getting inflated grid"""
        with patch('src.navigation.pathfinding.config') as mock_config:
            mock_config.navigation.get.return_value = 0.15
            grid = GridMap(50, 50, 0.1)

        grid.set_obstacle(25, 25)
        inflated = grid.get_inflated_grid()

        # Center should still be obstacle
        assert inflated[25, 25] == 1
        # Neighboring cells should also be obstacles due to inflation
        assert inflated[24, 25] == 1
        assert inflated[26, 25] == 1


class TestAStarPathfinder:
    """Tests for A* pathfinding"""

    @pytest.fixture
    def pathfinder(self):
        """Create pathfinder with test grid"""
        with patch('src.navigation.pathfinding.config') as mock_config:
            mock_config.navigation.get.side_effect = lambda key, default=None: {
                'grid_resolution': 0.1,
                'robot_radius': 0.1,
                'safety_margin': 0.05,
                'pathfinding': {
                    'heuristic_weight': 1.0,
                    'diagonal_movement': True,
                    'smoothing': False
                }
            }.get(key, default)

            grid = GridMap(50, 50, 0.1)
            pf = AStarPathfinder(grid)
        return pf

    def test_find_path_simple(self, pathfinder):
        """Test finding simple path"""
        start = (5, 5)
        goal = (10, 10)

        path = pathfinder.find_path(start, goal)

        assert path is not None
        assert len(path) > 0
        assert path[0] == start
        assert path[-1] == goal

    def test_find_path_with_obstacle(self, pathfinder):
        """Test finding path around obstacle"""
        # Add obstacle in direct path
        pathfinder.grid_map.set_obstacle(7, 7)
        pathfinder.grid_map.set_obstacle(8, 7)
        pathfinder.grid_map.set_obstacle(7, 8)
        pathfinder.grid_map.set_obstacle(8, 8)

        start = (5, 5)
        goal = (10, 10)

        path = pathfinder.find_path(start, goal)

        assert path is not None
        # Path should go around obstacle
        assert (7, 7) not in path
        assert (8, 8) not in path

    def test_find_path_blocked(self, pathfinder):
        """Test when goal is blocked"""
        # Block the goal
        goal = (10, 10)
        pathfinder.grid_map.set_obstacle(10, 10)

        start = (5, 5)

        path = pathfinder.find_path(start, goal)

        assert path is None

    def test_find_path_start_blocked(self, pathfinder):
        """Test when start is blocked"""
        start = (5, 5)
        pathfinder.grid_map.set_obstacle(5, 5)

        goal = (10, 10)

        path = pathfinder.find_path(start, goal)

        assert path is None

    def test_find_path_same_start_goal(self, pathfinder):
        """Test when start equals goal"""
        point = (5, 5)

        path = pathfinder.find_path(point, point)

        assert path is not None
        assert len(path) == 1
        assert path[0] == point

    def test_find_path_world(self, pathfinder):
        """Test finding path in world coordinates"""
        start = (0.5, 0.5)
        goal = (1.0, 1.0)

        path = pathfinder.find_path_world(start, goal)

        assert path is not None
        assert len(path) > 0
        # Check path is in world coordinates
        assert all(isinstance(p[0], float) for p in path)

    def test_heuristic(self, pathfinder):
        """Test heuristic function"""
        a = (0, 0)
        b = (3, 4)

        h = pathfinder.heuristic(a, b)

        # Euclidean distance should be 5
        assert abs(h - 5.0) < 0.01

    def test_update_obstacles(self, pathfinder):
        """Test updating obstacles list"""
        obstacles = [
            (0.5, 0.5, 0.1),
            (1.0, 1.0, 0.2)
        ]

        pathfinder.update_obstacles(obstacles)

        # Check obstacles are in grid
        x1, y1 = pathfinder.grid_map.world_to_grid(0.5, 0.5)
        assert pathfinder.grid_map.grid[y1, x1] == 1

    def test_diagonal_movement(self, pathfinder):
        """Test that diagonal movement is used"""
        start = (0, 0)
        goal = (5, 5)

        path = pathfinder.find_path(start, goal)

        assert path is not None
        # With diagonal movement, path should be shorter
        assert len(path) < 11  # Would be 11 without diagonal

    def test_path_optimality(self, pathfinder):
        """Test that path is reasonably optimal"""
        start = (0, 0)
        goal = (10, 0)

        path = pathfinder.find_path(start, goal)

        assert path is not None
        # Straight line path should have ~11 points
        assert len(path) == 11

    def test_line_of_sight(self, pathfinder):
        """Test line of sight checking"""
        grid = pathfinder.grid_map.grid

        # Clear path
        assert pathfinder._line_of_sight((0, 0), (5, 5), grid) is True

        # Add obstacle
        pathfinder.grid_map.set_obstacle(3, 3)
        grid = pathfinder.grid_map.grid

        assert pathfinder._line_of_sight((0, 0), (5, 5), grid) is False
