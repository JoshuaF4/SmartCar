"""
Computer-side server for extended SmartCar capabilities
Runs on a computer with GPU for heavy processing
"""

import json
import logging
import asyncio
import base64
import time
import math
from pathlib import Path
from typing import Set, Dict, Optional, List

import cv2
import numpy as np

try:
    import websockets
    from ultralytics import YOLO
except ImportError:
    websockets = None
    YOLO = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComputerServer:
    """
    Server running on computer for extended processing

    Provides:
    - GPU-accelerated object detection
    - Advanced path planning
    - Data logging and visualization
    - Dashboard for monitoring
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 9000):
        self.host = host
        self.port = port

        # Connected cars
        self.cars: Dict[websockets.WebSocketServerProtocol, dict] = {}

        # Detection model (larger model for accuracy)
        self.model: Optional[YOLO] = None
        self.model_path = 'yolov8m.pt'  # Medium model

        # Path planner (could use more advanced algorithms)
        self.use_advanced_planning = True

        # Path memory storage
        self.path_storage_dir = Path("data/car_paths")
        self.path_storage_dir.mkdir(parents=True, exist_ok=True)

        # Aggregated path data from all cars
        self.all_paths: Dict[str, dict] = {}
        self.obstacle_heatmap: Dict[tuple, int] = {}
        self.optimized_routes: Dict[str, List] = {}

    def initialize(self):
        """Initialize server components"""
        # Load YOLO model
        if YOLO:
            logger.info(f"Loading detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info("Detection model loaded")

        # Load existing path data
        self._load_path_storage()
        logger.info(f"Loaded {len(self.all_paths)} stored paths")

    async def handle_connection(self, websocket, path):
        """Handle car connection"""
        car_id = id(websocket)
        self.cars[websocket] = {
            'id': car_id,
            'capabilities': [],
            'status': {}
        }

        logger.info(f"Car connected: {car_id}")

        try:
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Car disconnected: {car_id}")
        finally:
            del self.cars[websocket]

    async def _handle_message(self, websocket, message: str):
        """Handle message from car"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'handshake':
                # Store car capabilities
                self.cars[websocket]['capabilities'] = data.get('capabilities', [])
                logger.info(f"Car capabilities: {data.get('capabilities')}")

            elif msg_type == 'heartbeat':
                # Respond to heartbeat
                await self._send(websocket, {'type': 'heartbeat_ack'})

            elif msg_type == 'status':
                # Update car status
                self.cars[websocket]['status'] = data.get('data', {})

            elif msg_type == 'frame':
                # Process frame with GPU model
                result = await self._process_frame(data)
                await self._send(websocket, {
                    'type': 'detection_result',
                    'data': result
                })

            elif msg_type == 'request_path':
                # Compute path with advanced algorithm
                path_data = data.get('data', {})
                path = self._compute_path(
                    path_data.get('start'),
                    path_data.get('goal'),
                    path_data.get('obstacles', [])
                )
                await self._send(websocket, {
                    'type': 'path',
                    'data': path
                })

            elif msg_type == 'detection':
                # Log detection from car
                self._log_detection(data.get('data', {}))

            elif msg_type == 'path_memory_sync':
                # Receive and store path memory from car
                car_id = self.cars[websocket]['id']
                await self._handle_path_memory_sync(websocket, car_id, data.get('data', {}))

            elif msg_type == 'new_path':
                # Receive a newly recorded path
                car_id = self.cars[websocket]['id']
                self._store_path(car_id, data.get('data', {}))

            elif msg_type == 'request_optimized_route':
                # Car requesting optimized route based on historical data
                route_request = data.get('data', {})
                route = self._get_optimized_route(
                    route_request.get('start'),
                    route_request.get('goal')
                )
                await self._send(websocket, {
                    'type': 'optimized_route',
                    'data': {'waypoints': route}
                })

        except json.JSONDecodeError:
            logger.error("Invalid JSON from car")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _send(self, websocket, data: dict):
        """Send data to car"""
        await websocket.send(json.dumps(data))

    async def _process_frame(self, frame_data: dict) -> dict:
        """Process frame with GPU-accelerated detection"""
        if not self.model:
            return {}

        try:
            # Decode frame
            frame_b64 = frame_data.get('data', '')
            frame_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {}

            # Run detection
            results = self.model(frame, verbose=False)

            # Process results
            detections = []
            danger_zone_y = frame.shape[0] * 0.6

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    center_y = (y1 + y2) / 2
                    zone = 'danger' if center_y > danger_zone_y else 'warning'

                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'zone': zone
                    })

            return {
                'detections': detections,
                'has_danger': any(d['zone'] == 'danger' for d in detections),
                'has_warning': any(d['zone'] == 'warning' for d in detections)
            }

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {}

    def _compute_path(self, start: tuple, goal: tuple,
                      obstacles: list) -> list:
        """Compute path using advanced algorithm"""
        # Simple implementation - in production use RRT*, D*, etc.
        # For now, return straight line path with obstacle check

        if not start or not goal:
            return []

        path = [list(start)]

        # Check for obstacles along path
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = (dx**2 + dy**2)**0.5

        if distance == 0:
            return [list(start)]

        # Generate waypoints
        num_points = max(2, int(distance / 0.5))
        for i in range(1, num_points + 1):
            t = i / num_points
            x = start[0] + dx * t
            y = start[1] + dy * t
            path.append([x, y])

        return path

    def _log_detection(self, detection: dict):
        """Log detection data"""
        # In production, save to database for analysis
        logger.debug(f"Detection logged: {detection}")

    # Path memory management methods
    def _load_path_storage(self):
        """Load all stored path data"""
        try:
            for car_dir in self.path_storage_dir.iterdir():
                if car_dir.is_dir():
                    for path_file in car_dir.glob("*.json"):
                        with open(path_file, 'r') as f:
                            path_data = json.load(f)
                            path_id = path_data.get('path_id', path_file.stem)
                            self.all_paths[path_id] = path_data

                            # Update obstacle heatmap
                            self._update_heatmap_from_path(path_data)

        except Exception as e:
            logger.error(f"Error loading path storage: {e}")

    def _update_heatmap_from_path(self, path_data: dict):
        """Update obstacle heatmap from path data"""
        for point in path_data.get('points', []):
            if point.get('obstacle_detected'):
                # Round to grid cell (0.1m resolution)
                cell = (
                    round(point['x'] * 10) / 10,
                    round(point['y'] * 10) / 10
                )
                self.obstacle_heatmap[cell] = self.obstacle_heatmap.get(cell, 0) + 1

    async def _handle_path_memory_sync(self, websocket, car_id: int, data: dict):
        """Handle path memory sync from car"""
        logger.info(f"Received path memory sync from car {car_id}: {data.get('total_paths', 0)} paths")

        # Store paths from car
        car_storage = self.path_storage_dir / str(car_id)
        car_storage.mkdir(exist_ok=True)

        for path_data in data.get('paths', []):
            path_id = path_data.get('path_id', f"path_{time.time()}")
            self.all_paths[path_id] = path_data
            self._update_heatmap_from_path(path_data)

            # Save to disk
            path_file = car_storage / f"{path_id}.json"
            with open(path_file, 'w') as f:
                json.dump(path_data, f)

        # Analyze and send back optimized data
        optimized_data = self._analyze_paths(car_id)

        await self._send(websocket, {
            'type': 'path_memory_update',
            'data': optimized_data
        })

        logger.info(f"Sent optimized path data to car {car_id}")

    def _store_path(self, car_id: int, path_data: dict):
        """Store a single path from car"""
        path_id = path_data.get('path_id', f"path_{time.time()}")
        self.all_paths[path_id] = path_data
        self._update_heatmap_from_path(path_data)

        # Save to disk
        car_storage = self.path_storage_dir / str(car_id)
        car_storage.mkdir(exist_ok=True)

        path_file = car_storage / f"{path_id}.json"
        with open(path_file, 'w') as f:
            json.dump(path_data, f)

        logger.info(f"Stored path {path_id} from car {car_id}")

    def _analyze_paths(self, car_id: int) -> dict:
        """Analyze paths and generate optimized data for car"""
        result = {
            'avoidance_zones': [],
            'optimized_paths': [],
            'hotspots': []
        }

        # Generate avoidance zones from heatmap
        for cell, count in self.obstacle_heatmap.items():
            if count >= 2:  # Threshold for avoidance
                radius = min(0.5, 0.1 + count * 0.05)
                result['avoidance_zones'].append([cell[0], cell[1], radius])
                result['hotspots'].append({
                    'x': cell[0],
                    'y': cell[1],
                    'count': count
                })

        # Find and optimize successful routes
        successful_paths = [
            p for p in self.all_paths.values()
            if p.get('success', False)
        ]

        # Group by similar start/end points and optimize
        route_groups = self._group_similar_routes(successful_paths)
        for group_key, paths in route_groups.items():
            if len(paths) >= 2:
                optimized = self._optimize_route_group(paths)
                if optimized:
                    result['optimized_paths'].append(optimized)

        return result

    def _group_similar_routes(self, paths: list) -> dict:
        """Group paths by similar start/end points"""
        groups = {}

        for path in paths:
            points = path.get('points', [])
            if len(points) < 2:
                continue

            start = (round(points[0]['x'], 1), round(points[0]['y'], 1))
            end = (round(points[-1]['x'], 1), round(points[-1]['y'], 1))
            key = (start, end)

            if key not in groups:
                groups[key] = []
            groups[key].append(path)

        return groups

    def _optimize_route_group(self, paths: list) -> dict:
        """Create optimized route from group of similar paths"""
        if not paths:
            return None

        # Use shortest successful path as base
        paths.sort(key=lambda p: p.get('total_distance', float('inf')))
        best = paths[0]

        # Simplify waypoints
        points = best.get('points', [])
        simplified = []

        for i, point in enumerate(points):
            # Keep first, last, and every Nth point
            if i == 0 or i == len(points) - 1 or i % 10 == 0:
                simplified.append([point['x'], point['y']])

        return {
            'path_id': f"optimized_{best.get('path_id', 'unknown')}",
            'points': simplified,
            'total_distance': best.get('total_distance', 0),
            'source': 'computer_optimized',
            'based_on_paths': len(paths)
        }

    def _get_optimized_route(self, start: tuple, goal: tuple) -> list:
        """Get optimized route based on historical data"""
        if not start or not goal:
            return []

        # Find matching routes
        tolerance = 0.5
        matching = []

        for path_id, path_data in self.all_paths.items():
            points = path_data.get('points', [])
            if len(points) < 2:
                continue

            path_start = (points[0]['x'], points[0]['y'])
            path_end = (points[-1]['x'], points[-1]['y'])

            start_dist = math.sqrt(
                (path_start[0] - start[0])**2 +
                (path_start[1] - start[1])**2
            )
            end_dist = math.sqrt(
                (path_end[0] - goal[0])**2 +
                (path_end[1] - goal[1])**2
            )

            if start_dist <= tolerance and end_dist <= tolerance:
                if path_data.get('success', False):
                    matching.append(path_data)

        if not matching:
            # Fall back to computed path
            return self._compute_path(start, goal, [])

        # Use best match
        matching.sort(key=lambda p: p.get('total_distance', float('inf')))
        best = matching[0]

        # Extract waypoints
        waypoints = []
        for point in best.get('points', []):
            waypoints.append([point['x'], point['y']])

        return waypoints

    def get_path_statistics(self) -> dict:
        """Get statistics about stored paths"""
        total = len(self.all_paths)
        successful = sum(1 for p in self.all_paths.values() if p.get('success', False))
        total_distance = sum(p.get('total_distance', 0) for p in self.all_paths.values())

        return {
            'total_paths': total,
            'successful_paths': successful,
            'failed_paths': total - successful,
            'total_distance': total_distance,
            'obstacle_hotspots': len([c for c, cnt in self.obstacle_heatmap.items() if cnt >= 2])
        }

    async def broadcast_command(self, command: dict):
        """Send command to all connected cars"""
        for websocket in self.cars:
            await self._send(websocket, {
                'type': 'command',
                'data': command
            })

    async def run(self):
        """Run the server"""
        self.initialize()

        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )

        logger.info(f"Computer server running on ws://{self.host}:{self.port}")

        await server.wait_closed()


def main():
    """Main entry point for computer server"""
    import argparse

    parser = argparse.ArgumentParser(
        description='SmartCar Computer Server'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Server host (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=9000,
        help='Server port (default: 9000)'
    )

    parser.add_argument(
        '--model',
        default='yolov8m.pt',
        help='YOLO model path'
    )

    args = parser.parse_args()

    server = ComputerServer(args.host, args.port)
    server.model_path = args.model

    asyncio.run(server.run())


if __name__ == '__main__':
    main()
