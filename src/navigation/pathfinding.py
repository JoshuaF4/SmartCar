"""
A* Pathfinding algorithm for local navigation
"""

import logging
import heapq
import math
import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

from ..config import config

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Node:
    """A* search node"""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


class GridMap:
    """2D occupancy grid for pathfinding"""

    def __init__(self, width: int, height: int, resolution: float = 0.1):
        """
        Args:
            width: Grid width in cells
            height: Grid height in cells
            resolution: Meters per cell
        """
        self.width = width
        self.height = height
        self.resolution = resolution

        # 0 = free, 1 = occupied
        self.grid = np.zeros((height, width), dtype=np.uint8)

        # Robot configuration
        nav_config = config.navigation
        self.robot_radius = nav_config.get('robot_radius', 0.15)
        self.safety_margin = nav_config.get('safety_margin', 0.1)

        # Inflation radius in cells
        self.inflation_radius = int(
            (self.robot_radius + self.safety_margin) / resolution
        )

    def set_obstacle(self, x: int, y: int):
        """Mark a cell as obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1

    def clear_obstacle(self, x: int, y: int):
        """Clear an obstacle from cell"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 0

    def is_free(self, x: int, y: int) -> bool:
        """Check if cell is free"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_x / self.resolution)
        grid_y = int(world_y / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = (grid_x + 0.5) * self.resolution
        world_y = (grid_y + 0.5) * self.resolution
        return (world_x, world_y)

    def add_obstacle_world(self, x: float, y: float, radius: float = 0.1):
        """Add obstacle in world coordinates with inflation"""
        center = self.world_to_grid(x, y)
        radius_cells = int(radius / self.resolution) + self.inflation_radius

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    ox, oy = center[0] + dx, center[1] + dy
                    self.set_obstacle(ox, oy)

    def clear(self):
        """Clear all obstacles"""
        self.grid.fill(0)

    def get_inflated_grid(self) -> np.ndarray:
        """Get grid with obstacles inflated by robot radius"""
        from scipy.ndimage import binary_dilation

        structure = np.ones((
            2 * self.inflation_radius + 1,
            2 * self.inflation_radius + 1
        ))

        inflated = binary_dilation(self.grid, structure=structure)
        return inflated.astype(np.uint8)


class AStarPathfinder:
    """A* pathfinding algorithm"""

    def __init__(self, grid_map: Optional[GridMap] = None):
        nav_config = config.navigation.get('pathfinding', {})

        self.heuristic_weight = nav_config.get('heuristic_weight', 1.0)
        self.diagonal_movement = nav_config.get('diagonal_movement', True)
        self.smoothing = nav_config.get('smoothing', True)

        # Grid map
        if grid_map:
            self.grid_map = grid_map
        else:
            # Create default grid
            resolution = config.navigation.get('grid_resolution', 0.1)
            self.grid_map = GridMap(100, 100, resolution)

        # Movement directions
        if self.diagonal_movement:
            self.directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            self.costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        else:
            self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.costs = [1.0, 1.0, 1.0, 1.0]

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate heuristic (Euclidean distance)"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        if self.diagonal_movement:
            return math.sqrt(dx*dx + dy*dy) * self.heuristic_weight
        else:
            return (dx + dy) * self.heuristic_weight

    def find_path(self, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal using A*
        Args:
            start: Starting grid position (x, y)
            goal: Goal grid position (x, y)
        Returns:
            List of positions from start to goal, or None if no path
        """
        # Get inflated grid
        grid = self.grid_map.get_inflated_grid()

        # Check if start or goal is blocked
        if grid[start[1], start[0]] == 1:
            logger.warning("Start position is blocked")
            return None

        if grid[goal[1], goal[0]] == 1:
            logger.warning("Goal position is blocked")
            return None

        # Initialize
        open_set = []
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {start: 0}

        start_node = Node(
            f_score=self.heuristic(start, goal),
            position=start,
            g_score=0
        )
        heapq.heappush(open_set, start_node)

        came_from = {}

        while open_set:
            current = heapq.heappop(open_set)

            if current.position == goal:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current.position)
                if self.smoothing:
                    path = self._smooth_path(path, grid)
                return path

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            # Explore neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current.position[0] + dx, current.position[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < self.grid_map.width and
                        0 <= neighbor[1] < self.grid_map.height):
                    continue

                # Check if blocked
                if grid[neighbor[1], neighbor[0]] == 1:
                    continue

                if neighbor in closed_set:
                    continue

                # Calculate g score
                tentative_g = current.g_score + self.costs[i]

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)

                    neighbor_node = Node(
                        f_score=f_score,
                        position=neighbor,
                        g_score=tentative_g
                    )
                    heapq.heappush(open_set, neighbor_node)
                    came_from[neighbor] = current.position

        logger.warning("No path found")
        return None

    def _reconstruct_path(self, came_from: dict,
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dict"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _smooth_path(self, path: List[Tuple[int, int]],
                     grid: np.ndarray) -> List[Tuple[int, int]]:
        """Smooth path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        current_idx = 0

        while current_idx < len(path) - 1:
            # Try to skip waypoints
            for next_idx in range(len(path) - 1, current_idx, -1):
                if self._line_of_sight(path[current_idx], path[next_idx], grid):
                    smoothed.append(path[next_idx])
                    current_idx = next_idx
                    break
            else:
                current_idx += 1
                if current_idx < len(path):
                    smoothed.append(path[current_idx])

        return smoothed

    def _line_of_sight(self, start: Tuple[int, int],
                       end: Tuple[int, int],
                       grid: np.ndarray) -> bool:
        """Check if there's a clear line of sight between two points"""
        x0, y0 = start
        x1, y1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x = x0
        y = y0

        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1

        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if grid[y, x] == 1:
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True

    def find_path_world(self, start: Tuple[float, float],
                        goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Find path using world coordinates
        Returns path in world coordinates
        """
        start_grid = self.grid_map.world_to_grid(*start)
        goal_grid = self.grid_map.world_to_grid(*goal)

        path_grid = self.find_path(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert to world coordinates
        path_world = [self.grid_map.grid_to_world(*p) for p in path_grid]
        return path_world

    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """
        Update grid with new obstacles
        Args:
            obstacles: List of (x, y, radius) in world coordinates
        """
        self.grid_map.clear()
        for x, y, radius in obstacles:
            self.grid_map.add_obstacle_world(x, y, radius)
