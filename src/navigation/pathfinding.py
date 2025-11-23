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
    """2D occupancy grid for pathfinding with dynamic obstacle support"""

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

        # 0 = free, 1 = occupied (static obstacles)
        self.grid = np.zeros((height, width), dtype=np.uint8)

        # Dynamic obstacle cost layer (0.0 to 1.0 confidence)
        self.dynamic_cost = np.zeros((height, width), dtype=np.float32)

        # Obstacle timestamps for decay
        self.obstacle_timestamps = np.zeros((height, width), dtype=np.float64)

        # Robot configuration
        nav_config = config.navigation
        self.robot_radius = nav_config.get('robot_radius', 0.15)
        self.safety_margin = nav_config.get('safety_margin', 0.1)

        # Inflation radius in cells
        self.inflation_radius = int(
            (self.robot_radius + self.safety_margin) / resolution
        )

        # Dynamic obstacle settings
        self.obstacle_decay_rate = nav_config.get('obstacle_decay_rate', 0.5)
        self.min_confidence_threshold = 0.1

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
        self.dynamic_cost.fill(0)
        self.obstacle_timestamps.fill(0)

    def add_dynamic_obstacle(self, x: float, y: float, radius: float = 0.1,
                             confidence: float = 1.0):
        """
        Add a dynamic obstacle with confidence level
        Args:
            x, y: World coordinates
            radius: Obstacle radius in meters
            confidence: Detection confidence (0.0 to 1.0)
        """
        import time
        center = self.world_to_grid(x, y)
        radius_cells = int(radius / self.resolution) + self.inflation_radius
        current_time = time.time()

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    ox, oy = center[0] + dx, center[1] + dy
                    if 0 <= ox < self.width and 0 <= oy < self.height:
                        # Update dynamic cost with max confidence
                        self.dynamic_cost[oy, ox] = max(
                            self.dynamic_cost[oy, ox],
                            confidence
                        )
                        self.obstacle_timestamps[oy, ox] = current_time

    def update_dynamic_obstacles(self, obstacles: List[Tuple[float, float, float, float]]):
        """
        Update grid with multiple dynamic obstacles
        Args:
            obstacles: List of (x, y, radius, confidence)
        """
        for x, y, radius, confidence in obstacles:
            self.add_dynamic_obstacle(x, y, radius, confidence)

    def apply_temporal_decay(self):
        """Apply temporal decay to dynamic obstacle costs"""
        import time
        current_time = time.time()

        # Calculate decay for each cell
        time_diff = current_time - self.obstacle_timestamps
        decay = np.exp(-self.obstacle_decay_rate * time_diff)

        # Apply decay where timestamps exist
        mask = self.obstacle_timestamps > 0
        self.dynamic_cost[mask] *= decay[mask]

        # Clear very low confidence areas
        self.dynamic_cost[self.dynamic_cost < self.min_confidence_threshold] = 0

    def get_combined_cost(self, x: int, y: int) -> float:
        """
        Get combined cost of static and dynamic obstacles
        Returns: 0.0 (free) to inf (blocked)
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')

        # Static obstacle is impassable
        if self.grid[y, x] == 1:
            return float('inf')

        # Dynamic obstacle adds cost based on confidence
        dynamic = self.dynamic_cost[y, x]
        if dynamic > 0.8:  # High confidence = treat as blocked
            return float('inf')

        # Return cost multiplier (1.0 = free, higher = avoid)
        return 1.0 + dynamic * 10.0  # Scale dynamic cost

    def clear_dynamic_obstacles(self):
        """Clear only dynamic obstacles, keep static"""
        self.dynamic_cost.fill(0)
        self.obstacle_timestamps.fill(0)

    def get_dynamic_obstacle_cells(self) -> List[Tuple[int, int, float]]:
        """
        Get all cells with dynamic obstacles
        Returns: List of (x, y, confidence)
        """
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.dynamic_cost[y, x] > self.min_confidence_threshold:
                    cells.append((x, y, self.dynamic_cost[y, x]))
        return cells

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
    """A* pathfinding algorithm with dynamic obstacle support"""

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

        # Dynamic replanning settings
        self.use_dynamic_costs = nav_config.get('use_dynamic_costs', True)
        self.replan_threshold = nav_config.get('replan_threshold', 0.3)
        self._last_path = None
        self._last_goal = None

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

                # Calculate g score with dynamic cost
                base_cost = self.costs[i]
                if self.use_dynamic_costs:
                    dynamic_multiplier = self.grid_map.get_combined_cost(neighbor[0], neighbor[1])
                    if dynamic_multiplier == float('inf'):
                        continue  # Skip blocked cells
                    base_cost *= dynamic_multiplier

                tentative_g = current.g_score + base_cost

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
        Update grid with new static obstacles
        Args:
            obstacles: List of (x, y, radius) in world coordinates
        """
        self.grid_map.clear()
        for x, y, radius in obstacles:
            self.grid_map.add_obstacle_world(x, y, radius)

    def update_dynamic_obstacles(self, obstacles: List[Tuple[float, float, float, float]]):
        """
        Update grid with dynamic obstacles (with confidence)
        Args:
            obstacles: List of (x, y, radius, confidence) in world coordinates
        """
        # Apply decay to existing dynamic obstacles
        self.grid_map.apply_temporal_decay()

        # Add new dynamic obstacles
        self.grid_map.update_dynamic_obstacles(obstacles)

    def needs_replan(self, current_position: Tuple[float, float],
                    current_path: List[Tuple[float, float]]) -> bool:
        """
        Check if replanning is needed due to dynamic obstacles
        Returns True if path is blocked or significantly impacted
        """
        if not current_path:
            return True

        # Check if any waypoint on current path is now blocked
        for waypoint in current_path:
            grid_pos = self.grid_map.world_to_grid(*waypoint)
            cost = self.grid_map.get_combined_cost(grid_pos[0], grid_pos[1])

            if cost == float('inf'):
                logger.info("Replan needed: waypoint blocked by dynamic obstacle")
                return True

            if cost > 1.0 + self.replan_threshold * 10:
                logger.info("Replan needed: path cost increased significantly")
                return True

        return False

    def find_path_dynamic(self, start: Tuple[float, float],
                          goal: Tuple[float, float],
                          predicted_obstacles: List[Tuple[float, float, float, float]] = None
                          ) -> Optional[List[Tuple[float, float]]]:
        """
        Find path considering dynamic/predicted obstacles
        Args:
            start: Start position (world coordinates)
            goal: Goal position (world coordinates)
            predicted_obstacles: List of (x, y, radius, confidence) for predicted positions
        Returns:
            Path in world coordinates or None
        """
        # Apply temporal decay
        self.grid_map.apply_temporal_decay()

        # Add predicted obstacles if provided
        if predicted_obstacles:
            self.grid_map.update_dynamic_obstacles(predicted_obstacles)

        # Find path with dynamic costs
        path = self.find_path_world(start, goal)

        # Store for replan checking
        self._last_path = path
        self._last_goal = goal

        return path

    def get_path_risk_score(self, path: List[Tuple[float, float]]) -> float:
        """
        Calculate risk score for a path based on dynamic obstacles
        Higher score = more risky
        """
        if not path:
            return float('inf')

        total_risk = 0.0
        for waypoint in path:
            grid_pos = self.grid_map.world_to_grid(*waypoint)
            cost = self.grid_map.get_combined_cost(grid_pos[0], grid_pos[1])

            if cost == float('inf'):
                return float('inf')

            total_risk += cost - 1.0  # Subtract base cost

        return total_risk / len(path)  # Average risk per waypoint

    def clear_and_update(self, static_obstacles: List[Tuple[float, float, float]],
                        dynamic_obstacles: List[Tuple[float, float, float, float]]):
        """
        Clear grid and update with both static and dynamic obstacles
        Args:
            static_obstacles: List of (x, y, radius)
            dynamic_obstacles: List of (x, y, radius, confidence)
        """
        # Clear static grid only
        self.grid_map.grid.fill(0)

        # Add static obstacles
        for x, y, radius in static_obstacles:
            self.grid_map.add_obstacle_world(x, y, radius)

        # Update dynamic obstacles
        self.grid_map.update_dynamic_obstacles(dynamic_obstacles)
