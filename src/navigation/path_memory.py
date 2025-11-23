"""
Path memory and mapping system for learning from previous routes
"""

import json
import logging
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class PathPoint:
    """A single point in a recorded path"""
    x: float
    y: float
    timestamp: float
    heading: float = 0.0  # degrees
    speed: float = 0.0
    obstacle_detected: bool = False
    sensor_data: Dict = field(default_factory=dict)
    obstacle_confidence: float = 1.0  # Confidence in obstacle detection
    obstacle_velocity: Tuple[float, float] = (0.0, 0.0)  # Observed obstacle velocity


@dataclass
class RecordedPath:
    """A complete recorded path/route"""
    path_id: str
    start_time: float
    end_time: float
    points: List[PathPoint]
    success: bool = True  # Did the path complete successfully?
    total_distance: float = 0.0
    obstacles_encountered: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'path_id': self.path_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'points': [asdict(p) for p in self.points],
            'success': self.success,
            'total_distance': self.total_distance,
            'obstacles_encountered': self.obstacles_encountered,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RecordedPath':
        points = [PathPoint(**p) for p in data.get('points', [])]
        return cls(
            path_id=data['path_id'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            points=points,
            success=data.get('success', True),
            total_distance=data.get('total_distance', 0.0),
            obstacles_encountered=data.get('obstacles_encountered', 0),
            metadata=data.get('metadata', {})
        )


class PathMemory:
    """
    Manages path recording, storage, and retrieval
    Enables learning from previous routes
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            project_root = Path(__file__).parent.parent.parent
            storage_path = project_root / "data" / "paths"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Current recording
        self.is_recording = False
        self.current_path: Optional[RecordedPath] = None
        self.current_points: List[PathPoint] = []

        # Loaded paths
        self.paths: Dict[str, RecordedPath] = {}

        # Spatial index for quick lookup
        self.spatial_grid: Dict[Tuple[int, int], List[str]] = {}
        self.grid_resolution = config.navigation.get('grid_resolution', 0.1)

        # Statistics
        self.total_paths = 0
        self.total_distance = 0.0

        # Dynamic obstacle tracking
        self.obstacle_decay_rate = config.navigation.get('obstacle_decay_rate', 0.1)
        self.obstacle_half_life = config.navigation.get('obstacle_half_life', 300.0)  # 5 minutes
        self.dynamic_obstacle_data: Dict[Tuple[int, int], List[Dict]] = {}

        # Load existing paths
        self._load_paths()

    def _load_paths(self):
        """Load all saved paths from storage"""
        try:
            for path_file in self.storage_path.glob("*.json"):
                with open(path_file, 'r') as f:
                    data = json.load(f)
                    recorded_path = RecordedPath.from_dict(data)
                    self.paths[recorded_path.path_id] = recorded_path
                    self._index_path(recorded_path)

            self.total_paths = len(self.paths)
            logger.info(f"Loaded {self.total_paths} paths from storage")

        except Exception as e:
            logger.error(f"Error loading paths: {e}")

    def _index_path(self, path: RecordedPath):
        """Add path to spatial index for quick lookup"""
        for point in path.points:
            grid_cell = self._to_grid_cell(point.x, point.y)
            if grid_cell not in self.spatial_grid:
                self.spatial_grid[grid_cell] = []
            if path.path_id not in self.spatial_grid[grid_cell]:
                self.spatial_grid[grid_cell].append(path.path_id)

    def _to_grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell"""
        return (
            int(x / self.grid_resolution),
            int(y / self.grid_resolution)
        )

    def start_recording(self, metadata: Dict = None) -> str:
        """Start recording a new path"""
        path_id = f"path_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"

        self.current_path = RecordedPath(
            path_id=path_id,
            start_time=time.time(),
            end_time=0,
            points=[],
            metadata=metadata or {}
        )
        self.current_points = []
        self.is_recording = True

        logger.info(f"Started recording path: {path_id}")
        return path_id

    def record_point(self, x: float, y: float, heading: float = 0.0,
                     speed: float = 0.0, obstacle_detected: bool = False,
                     sensor_data: Dict = None, obstacle_confidence: float = 1.0,
                     obstacle_velocity: Tuple[float, float] = (0.0, 0.0)):
        """Record a point in the current path"""
        if not self.is_recording:
            return

        point = PathPoint(
            x=x,
            y=y,
            timestamp=time.time(),
            heading=heading,
            speed=speed,
            obstacle_detected=obstacle_detected,
            sensor_data=sensor_data or {},
            obstacle_confidence=obstacle_confidence,
            obstacle_velocity=obstacle_velocity
        )

        self.current_points.append(point)

        if obstacle_detected:
            self.current_path.obstacles_encountered += 1
            # Track dynamic obstacle data
            self._record_obstacle_observation(x, y, obstacle_confidence, obstacle_velocity)

    def _record_obstacle_observation(self, x: float, y: float,
                                     confidence: float, velocity: Tuple[float, float]):
        """Record obstacle observation for dynamic tracking"""
        cell = self._to_grid_cell(x, y)

        if cell not in self.dynamic_obstacle_data:
            self.dynamic_obstacle_data[cell] = []

        self.dynamic_obstacle_data[cell].append({
            'timestamp': time.time(),
            'confidence': confidence,
            'velocity': velocity,
            'is_dynamic': abs(velocity[0]) > 0.01 or abs(velocity[1]) > 0.01
        })

    def stop_recording(self, success: bool = True) -> Optional[RecordedPath]:
        """Stop recording and save the path"""
        if not self.is_recording or not self.current_path:
            return None

        self.current_path.end_time = time.time()
        self.current_path.points = self.current_points
        self.current_path.success = success
        self.current_path.total_distance = self._calculate_distance(self.current_points)

        # Save path
        self._save_path(self.current_path)

        # Add to memory
        self.paths[self.current_path.path_id] = self.current_path
        self._index_path(self.current_path)

        self.total_paths += 1
        self.total_distance += self.current_path.total_distance

        logger.info(f"Recorded path {self.current_path.path_id}: "
                   f"{len(self.current_points)} points, "
                   f"{self.current_path.total_distance:.2f}m")

        result = self.current_path
        self.current_path = None
        self.current_points = []
        self.is_recording = False

        return result

    def _calculate_distance(self, points: List[PathPoint]) -> float:
        """Calculate total distance of path"""
        if len(points) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total += math.sqrt(dx*dx + dy*dy)

        return total

    def _save_path(self, path: RecordedPath):
        """Save path to storage"""
        try:
            file_path = self.storage_path / f"{path.path_id}.json"
            with open(file_path, 'w') as f:
                json.dump(path.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving path: {e}")

    def get_paths_near(self, x: float, y: float,
                       radius: float = 1.0) -> List[RecordedPath]:
        """Get all paths that pass near a point"""
        nearby_paths = set()
        center_cell = self._to_grid_cell(x, y)
        cell_radius = int(radius / self.grid_resolution) + 1

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.spatial_grid:
                    nearby_paths.update(self.spatial_grid[cell])

        return [self.paths[pid] for pid in nearby_paths if pid in self.paths]

    def get_successful_paths(self, start: Tuple[float, float],
                             goal: Tuple[float, float],
                             tolerance: float = 0.5) -> List[RecordedPath]:
        """Get successful paths between similar start and goal"""
        matching = []

        for path in self.paths.values():
            if not path.success or len(path.points) < 2:
                continue

            # Check if start matches
            path_start = (path.points[0].x, path.points[0].y)
            start_dist = math.sqrt(
                (path_start[0] - start[0])**2 +
                (path_start[1] - start[1])**2
            )

            # Check if end matches
            path_end = (path.points[-1].x, path.points[-1].y)
            end_dist = math.sqrt(
                (path_end[0] - goal[0])**2 +
                (path_end[1] - goal[1])**2
            )

            if start_dist <= tolerance and end_dist <= tolerance:
                matching.append(path)

        # Sort by distance (prefer shorter paths)
        matching.sort(key=lambda p: p.total_distance)

        return matching

    def get_obstacle_hotspots(self, min_encounters: int = 3,
                              use_temporal_decay: bool = True) -> List[Tuple[float, float, float]]:
        """
        Get locations where obstacles are frequently encountered
        With temporal decay, recent observations are weighted higher
        Returns: List of (x, y, weighted_count)
        """
        current_time = time.time()
        obstacle_scores: Dict[Tuple[int, int], float] = {}

        for path in self.paths.values():
            for point in path.points:
                if point.obstacle_detected:
                    cell = self._to_grid_cell(point.x, point.y)

                    if use_temporal_decay:
                        # Apply temporal decay based on observation age
                        age = current_time - point.timestamp
                        decay = math.exp(-age * math.log(2) / self.obstacle_half_life)
                        weight = point.obstacle_confidence * decay
                    else:
                        weight = 1.0

                    obstacle_scores[cell] = obstacle_scores.get(cell, 0) + weight

        # Filter by minimum score and convert back to world coords
        hotspots = []
        for cell, score in obstacle_scores.items():
            if score >= min_encounters:
                x = (cell[0] + 0.5) * self.grid_resolution
                y = (cell[1] + 0.5) * self.grid_resolution
                hotspots.append((x, y, score))

        # Sort by score
        hotspots.sort(key=lambda h: h[2], reverse=True)

        return hotspots

    def get_dynamic_obstacle_zones(self) -> List[Tuple[float, float, float, Tuple[float, float]]]:
        """
        Get zones where dynamic (moving) obstacles have been detected
        Returns: List of (x, y, confidence, avg_velocity)
        """
        current_time = time.time()
        dynamic_zones = []

        for cell, observations in self.dynamic_obstacle_data.items():
            # Filter to dynamic observations only
            dynamic_obs = [o for o in observations if o.get('is_dynamic', False)]

            if not dynamic_obs:
                continue

            # Calculate time-weighted confidence and average velocity
            total_weight = 0
            weighted_vx = 0
            weighted_vy = 0

            for obs in dynamic_obs:
                age = current_time - obs['timestamp']
                weight = obs['confidence'] * math.exp(-age * math.log(2) / self.obstacle_half_life)

                total_weight += weight
                weighted_vx += obs['velocity'][0] * weight
                weighted_vy += obs['velocity'][1] * weight

            if total_weight > 0.1:  # Minimum confidence threshold
                x = (cell[0] + 0.5) * self.grid_resolution
                y = (cell[1] + 0.5) * self.grid_resolution
                avg_velocity = (weighted_vx / total_weight, weighted_vy / total_weight)

                dynamic_zones.append((x, y, total_weight, avg_velocity))

        return dynamic_zones

    def decay_old_observations(self, max_age_seconds: float = None):
        """
        Remove obstacle observations older than max_age
        This helps keep memory usage bounded
        """
        if max_age_seconds is None:
            max_age_seconds = self.obstacle_half_life * 10  # Keep 10 half-lives

        current_time = time.time()
        cutoff = current_time - max_age_seconds

        for cell in list(self.dynamic_obstacle_data.keys()):
            # Filter out old observations
            self.dynamic_obstacle_data[cell] = [
                obs for obs in self.dynamic_obstacle_data[cell]
                if obs['timestamp'] > cutoff
            ]

            # Remove empty cells
            if not self.dynamic_obstacle_data[cell]:
                del self.dynamic_obstacle_data[cell]

    def get_time_weighted_avoidance_zones(self) -> List[Tuple[float, float, float, float]]:
        """
        Get avoidance zones with time-weighted confidence scores
        Returns: List of (x, y, radius, confidence)
        """
        zones = []
        current_time = time.time()

        # Add time-weighted hotspots
        hotspots = self.get_obstacle_hotspots(min_encounters=1, use_temporal_decay=True)
        for x, y, score in hotspots:
            if score > 0.5:  # Minimum confidence threshold
                radius = min(0.5, 0.1 + score * 0.05)
                zones.append((x, y, radius, min(score, 1.0)))

        # Add failure points with decay
        for path in self.paths.values():
            if not path.success and path.points:
                last = path.points[-1]
                age = current_time - last.timestamp
                decay = math.exp(-age * math.log(2) / self.obstacle_half_life)
                if decay > 0.1:  # Only include if still relevant
                    zones.append((last.x, last.y, 0.3, decay))

        return zones

    def update_obstacle_from_tracker(self, obstacle_id: str, x: float, y: float,
                                     velocity: Tuple[float, float], confidence: float):
        """
        Update obstacle data from dynamic obstacle tracker
        Called in real-time as obstacles are tracked
        """
        cell = self._to_grid_cell(x, y)

        if cell not in self.dynamic_obstacle_data:
            self.dynamic_obstacle_data[cell] = []

        self.dynamic_obstacle_data[cell].append({
            'timestamp': time.time(),
            'confidence': confidence,
            'velocity': velocity,
            'is_dynamic': abs(velocity[0]) > 0.01 or abs(velocity[1]) > 0.01,
            'obstacle_id': obstacle_id
        })

    def get_preferred_route(self, start: Tuple[float, float],
                            goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Get the best known route based on historical data
        Returns simplified path as list of (x, y) waypoints
        """
        # Find successful paths
        successful = self.get_successful_paths(start, goal)

        if not successful:
            return None

        # Use the best (shortest successful) path
        best_path = successful[0]

        # Simplify to waypoints
        waypoints = []
        for i, point in enumerate(best_path.points):
            # Keep every Nth point or important points
            if i == 0 or i == len(best_path.points) - 1:
                waypoints.append((point.x, point.y))
            elif i % 5 == 0:  # Every 5th point
                waypoints.append((point.x, point.y))
            elif point.obstacle_detected:
                waypoints.append((point.x, point.y))

        return waypoints

    def get_avoidance_zones(self) -> List[Tuple[float, float, float]]:
        """
        Get zones to avoid based on failed paths and obstacles
        Returns: List of (x, y, radius)
        """
        zones = []

        # Add hotspots as avoidance zones
        hotspots = self.get_obstacle_hotspots(min_encounters=2)
        for x, y, count in hotspots:
            # Larger radius for more frequent obstacles
            radius = min(0.5, 0.1 + count * 0.05)
            zones.append((x, y, radius))

        # Add failure points from unsuccessful paths
        for path in self.paths.values():
            if not path.success and path.points:
                # Last point of failed path
                last = path.points[-1]
                zones.append((last.x, last.y, 0.3))

        return zones

    def export_for_transmission(self) -> Dict:
        """
        Export path memory for transmission to computer
        Returns condensed format for network transfer
        """
        return {
            'total_paths': self.total_paths,
            'total_distance': self.total_distance,
            'paths': [p.to_dict() for p in self.paths.values()],
            'hotspots': self.get_obstacle_hotspots(),
            'avoidance_zones': self.get_avoidance_zones(),
            'timestamp': time.time()
        }

    def import_from_computer(self, data: Dict):
        """
        Import processed path data from computer
        Computer may have optimized routes or merged data from multiple cars
        """
        # Import optimized paths
        if 'optimized_paths' in data:
            for path_data in data['optimized_paths']:
                path = RecordedPath.from_dict(path_data)
                path.metadata['source'] = 'computer'
                self.paths[path.path_id] = path
                self._index_path(path)
                self._save_path(path)

        # Import updated avoidance zones
        if 'avoidance_zones' in data:
            # Store for use by pathfinder
            self._computer_avoidance_zones = data['avoidance_zones']

        logger.info(f"Imported data from computer")

    def get_statistics(self) -> Dict:
        """Get path memory statistics"""
        successful = sum(1 for p in self.paths.values() if p.success)

        return {
            'total_paths': self.total_paths,
            'successful_paths': successful,
            'failed_paths': self.total_paths - successful,
            'total_distance': self.total_distance,
            'hotspots_count': len(self.get_obstacle_hotspots()),
            'storage_size_mb': sum(
                f.stat().st_size for f in self.storage_path.glob("*.json")
            ) / (1024 * 1024)
        }

    def clear_old_paths(self, max_age_days: int = 30):
        """Remove paths older than specified days"""
        cutoff = time.time() - (max_age_days * 24 * 3600)
        removed = 0

        for path_id in list(self.paths.keys()):
            path = self.paths[path_id]
            if path.end_time < cutoff:
                # Remove file
                file_path = self.storage_path / f"{path_id}.json"
                if file_path.exists():
                    file_path.unlink()

                # Remove from memory
                del self.paths[path_id]
                removed += 1

        if removed > 0:
            # Rebuild spatial index
            self.spatial_grid = {}
            for path in self.paths.values():
                self._index_path(path)

            logger.info(f"Removed {removed} old paths")
