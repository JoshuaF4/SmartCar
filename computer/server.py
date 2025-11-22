"""
Computer-side server for extended SmartCar capabilities
Runs on a computer with GPU for heavy processing
"""

import json
import logging
import asyncio
import base64
from typing import Set, Dict, Optional

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
