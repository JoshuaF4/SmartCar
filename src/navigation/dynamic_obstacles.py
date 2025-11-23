"""
Dynamic obstacle tracking and prediction for moving objects
"""

import time
import math
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class ObstacleObservation:
    """Single observation of an obstacle"""
    x: float
    y: float
    timestamp: float
    confidence: float = 1.0
    size: float = 0.1  # Estimated radius


@dataclass
class TrackedObstacle:
    """A tracked obstacle with history and velocity estimation"""
    obstacle_id: str
    observations: deque = field(default_factory=lambda: deque(maxlen=50))

    # Current state
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0  # Velocity x
    vy: float = 0.0  # Velocity y
    size: float = 0.1

    # Tracking metadata
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_static: bool = True
    confidence: float = 1.0

    def add_observation(self, obs: ObstacleObservation):
        """Add new observation and update state"""
        self.observations.append(obs)
        self.last_seen = obs.timestamp

        if len(self.observations) == 1:
            self.first_seen = obs.timestamp
            self.x = obs.x
            self.y = obs.y
            self.size = obs.size
        else:
            self._update_velocity()
            self.x = obs.x
            self.y = obs.y
            self.size = max(self.size, obs.size)

        self.confidence = obs.confidence

    def _update_velocity(self):
        """Estimate velocity from recent observations"""
        if len(self.observations) < 2:
            return

        # Use last few observations for velocity estimation
        recent = list(self.observations)[-5:]
        if len(recent) < 2:
            return

        # Simple linear regression for velocity
        dt = recent[-1].timestamp - recent[0].timestamp
        if dt < 0.01:  # Avoid division by zero
            return

        dx = recent[-1].x - recent[0].x
        dy = recent[-1].y - recent[0].y

        self.vx = dx / dt
        self.vy = dy / dt

        # Determine if static (velocity below threshold)
        speed = math.sqrt(self.vx**2 + self.vy**2)
        self.is_static = speed < 0.05  # 5cm/s threshold

    def predict_position(self, future_time: float) -> Tuple[float, float]:
        """Predict position at future time"""
        if self.is_static:
            return (self.x, self.y)

        dt = future_time - self.last_seen
        predicted_x = self.x + self.vx * dt
        predicted_y = self.y + self.vy * dt

        return (predicted_x, predicted_y)

    def get_prediction_confidence(self, future_time: float) -> float:
        """Get confidence in prediction (decreases with time)"""
        dt = future_time - self.last_seen
        # Exponential decay of confidence
        decay_rate = 0.5  # Halve confidence every 2 seconds
        return self.confidence * math.exp(-decay_rate * dt)


class DynamicObstacleTracker:
    """
    Tracks multiple dynamic obstacles over time
    Provides velocity estimation and position prediction
    """

    def __init__(self):
        self.obstacles: Dict[str, TrackedObstacle] = {}

        # Configuration
        nav_config = config.navigation
        self.association_threshold = nav_config.get('obstacle_association_threshold', 0.5)
        self.stale_threshold = nav_config.get('obstacle_stale_threshold', 5.0)
        self.min_observations = nav_config.get('min_obstacle_observations', 3)

        # Obstacle ID counter
        self._next_id = 0

        # Decay parameters
        self.decay_rate = nav_config.get('obstacle_decay_rate', 0.1)

    def update(self, detections: List[Tuple[float, float, float, float]]):
        """
        Update tracker with new detections
        Args:
            detections: List of (x, y, size, confidence)
        """
        current_time = time.time()

        # Associate detections with existing obstacles
        unmatched_detections = []
        matched_obstacle_ids = set()

        for det in detections:
            x, y, size, confidence = det
            obs = ObstacleObservation(
                x=x, y=y,
                timestamp=current_time,
                confidence=confidence,
                size=size
            )

            # Find best matching obstacle
            best_match = None
            best_distance = float('inf')

            for obs_id, obstacle in self.obstacles.items():
                if obs_id in matched_obstacle_ids:
                    continue

                # Predict where obstacle should be
                predicted = obstacle.predict_position(current_time)
                dist = math.sqrt(
                    (predicted[0] - x)**2 +
                    (predicted[1] - y)**2
                )

                if dist < self.association_threshold and dist < best_distance:
                    best_match = obs_id
                    best_distance = dist

            if best_match:
                self.obstacles[best_match].add_observation(obs)
                matched_obstacle_ids.add(best_match)
            else:
                unmatched_detections.append(obs)

        # Create new obstacles for unmatched detections
        for obs in unmatched_detections:
            obs_id = f"obs_{self._next_id}"
            self._next_id += 1

            obstacle = TrackedObstacle(obstacle_id=obs_id)
            obstacle.add_observation(obs)
            self.obstacles[obs_id] = obstacle

        # Remove stale obstacles
        self._remove_stale(current_time)

    def _remove_stale(self, current_time: float):
        """Remove obstacles not seen recently"""
        stale_ids = []
        for obs_id, obstacle in self.obstacles.items():
            if current_time - obstacle.last_seen > self.stale_threshold:
                stale_ids.append(obs_id)

        for obs_id in stale_ids:
            del self.obstacles[obs_id]

    def get_current_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """
        Get current obstacle positions
        Returns: List of (x, y, size, confidence)
        """
        current_time = time.time()
        result = []

        for obstacle in self.obstacles.values():
            if len(obstacle.observations) >= self.min_observations:
                # Apply temporal decay to confidence
                age = current_time - obstacle.last_seen
                decayed_confidence = obstacle.confidence * math.exp(-self.decay_rate * age)

                result.append((
                    obstacle.x,
                    obstacle.y,
                    obstacle.size,
                    decayed_confidence
                ))

        return result

    def get_predicted_obstacles(self, prediction_time: float) -> List[Tuple[float, float, float, float]]:
        """
        Get predicted obstacle positions at future time
        Args:
            prediction_time: Absolute timestamp for prediction
        Returns: List of (x, y, size, confidence)
        """
        result = []

        for obstacle in self.obstacles.values():
            if len(obstacle.observations) >= self.min_observations:
                predicted_pos = obstacle.predict_position(prediction_time)
                confidence = obstacle.get_prediction_confidence(prediction_time)

                result.append((
                    predicted_pos[0],
                    predicted_pos[1],
                    obstacle.size,
                    confidence
                ))

        return result

    def get_dynamic_obstacles(self) -> List[TrackedObstacle]:
        """Get all obstacles identified as moving"""
        return [
            obs for obs in self.obstacles.values()
            if not obs.is_static and len(obs.observations) >= self.min_observations
        ]

    def get_static_obstacles(self) -> List[TrackedObstacle]:
        """Get all obstacles identified as static"""
        return [
            obs for obs in self.obstacles.values()
            if obs.is_static and len(obs.observations) >= self.min_observations
        ]

    def get_obstacle_velocities(self) -> Dict[str, Tuple[float, float]]:
        """Get velocity estimates for all tracked obstacles"""
        return {
            obs_id: (obs.vx, obs.vy)
            for obs_id, obs in self.obstacles.items()
            if len(obs.observations) >= self.min_observations
        }

    def clear(self):
        """Clear all tracked obstacles"""
        self.obstacles.clear()

    def get_collision_risk(self, position: Tuple[float, float],
                           velocity: Tuple[float, float],
                           time_horizon: float = 2.0) -> List[Tuple[str, float, float]]:
        """
        Calculate collision risk with dynamic obstacles
        Args:
            position: Current robot position (x, y)
            velocity: Current robot velocity (vx, vy)
            time_horizon: Time to look ahead in seconds
        Returns:
            List of (obstacle_id, collision_time, min_distance)
        """
        risks = []
        current_time = time.time()

        for obs_id, obstacle in self.obstacles.items():
            if len(obstacle.observations) < self.min_observations:
                continue

            # Check for collision at multiple time points
            min_distance = float('inf')
            collision_time = None

            for t in np.linspace(0, time_horizon, 20):
                future_time = current_time + t

                # Robot position at time t
                robot_x = position[0] + velocity[0] * t
                robot_y = position[1] + velocity[1] * t

                # Obstacle position at time t
                obs_pos = obstacle.predict_position(future_time)

                distance = math.sqrt(
                    (robot_x - obs_pos[0])**2 +
                    (robot_y - obs_pos[1])**2
                )

                if distance < min_distance:
                    min_distance = distance
                    collision_time = t

            # Check if within collision threshold
            collision_threshold = obstacle.size + 0.2  # Robot radius
            if min_distance < collision_threshold * 2:
                risks.append((obs_id, collision_time, min_distance))

        # Sort by collision time
        risks.sort(key=lambda x: x[1])
        return risks

    def export_state(self) -> Dict:
        """Export tracker state for transmission"""
        return {
            'timestamp': time.time(),
            'obstacles': [
                {
                    'id': obs.obstacle_id,
                    'x': obs.x,
                    'y': obs.y,
                    'vx': obs.vx,
                    'vy': obs.vy,
                    'size': obs.size,
                    'is_static': obs.is_static,
                    'confidence': obs.confidence,
                    'first_seen': obs.first_seen,
                    'last_seen': obs.last_seen
                }
                for obs in self.obstacles.values()
                if len(obs.observations) >= self.min_observations
            ]
        }

    def import_state(self, data: Dict):
        """Import obstacle data from external source"""
        for obs_data in data.get('obstacles', []):
            obs_id = obs_data['id']

            if obs_id not in self.obstacles:
                obstacle = TrackedObstacle(obstacle_id=obs_id)
                self.obstacles[obs_id] = obstacle
            else:
                obstacle = self.obstacles[obs_id]

            # Update obstacle state
            obstacle.x = obs_data['x']
            obstacle.y = obs_data['y']
            obstacle.vx = obs_data.get('vx', 0)
            obstacle.vy = obs_data.get('vy', 0)
            obstacle.size = obs_data.get('size', 0.1)
            obstacle.is_static = obs_data.get('is_static', True)
            obstacle.confidence = obs_data.get('confidence', 1.0)
            obstacle.last_seen = obs_data.get('last_seen', time.time())
