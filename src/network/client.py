"""
Client for connecting to computer for extended capabilities
Runs on the Raspberry Pi, connects to computer server
"""

import json
import logging
import asyncio
import time
import base64
from typing import Optional, Callable
from threading import Thread, Event

try:
    import websockets
except ImportError:
    websockets = None

import cv2
import numpy as np

from ..config import config

logger = logging.getLogger(__name__)


class ComputerClient:
    """
    Client that connects Pi to computer for extended processing

    Extended capabilities when connected:
    - Advanced model inference on computer GPU
    - Path planning with more compute
    - Data logging and analysis
    - Remote monitoring dashboard
    """

    def __init__(self, car_controller):
        self.car = car_controller
        self.config = config.network.get('computer', {})

        self.host = self.config.get('host', '192.168.1.100')
        self.port = self.config.get('port', 9000)
        self.reconnect_interval = self.config.get('reconnect_interval', 5)

        self.websocket = None
        self.is_connected = False
        self._stop_event = Event()
        self._client_thread: Optional[Thread] = None

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_command: Optional[Callable] = None

        # Extended capabilities flags
        self.use_remote_detection = False
        self.use_remote_planning = False

    def start(self):
        """Start client connection"""
        self._stop_event.clear()
        self._client_thread = Thread(target=self._connection_loop)
        self._client_thread.daemon = True
        self._client_thread.start()
        logger.info(f"Computer client started, connecting to {self.host}:{self.port}")

    def stop(self):
        """Stop client connection"""
        self._stop_event.set()
        if self._client_thread:
            self._client_thread.join(timeout=2.0)
        self.is_connected = False
        logger.info("Computer client stopped")

    def _connection_loop(self):
        """Main connection loop with reconnection"""
        while not self._stop_event.is_set():
            try:
                asyncio.run(self._connect_and_run())
            except Exception as e:
                logger.error(f"Connection error: {e}")

            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {self.reconnect_interval}s...")
                time.sleep(self.reconnect_interval)

    async def _connect_and_run(self):
        """Connect to computer and handle messages"""
        uri = f"ws://{self.host}:{self.port}"

        try:
            async with websockets.connect(uri) as ws:
                self.websocket = ws
                self.is_connected = True
                logger.info(f"Connected to computer at {uri}")

                if self.on_connected:
                    self.on_connected()

                # Send initial handshake
                await self._send({
                    'type': 'handshake',
                    'device': 'smartcar',
                    'capabilities': [
                        'camera',
                        'motors',
                        'sensors',
                        'detection',
                        'path_memory'
                    ]
                })

                # Send initial path memory sync
                await self._sync_path_memory()

                # Main message loop
                while not self._stop_event.is_set():
                    try:
                        message = await asyncio.wait_for(
                            ws.recv(),
                            timeout=1.0
                        )
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await self._send({'type': 'heartbeat'})

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.is_connected = False
            self.websocket = None
            if self.on_disconnected:
                self.on_disconnected()

    async def _handle_message(self, message: str):
        """Handle message from computer"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'command':
                # Execute command on car
                result = self._execute_command(data.get('data', {}))
                await self._send({
                    'type': 'command_result',
                    'data': result
                })

            elif msg_type == 'request_frame':
                # Send camera frame for remote processing
                await self._send_frame()

            elif msg_type == 'detection_result':
                # Receive detection results from computer
                self._apply_detection_result(data.get('data', {}))

            elif msg_type == 'path':
                # Receive computed path from computer
                self._apply_path(data.get('data', []))

            elif msg_type == 'config_update':
                # Update configuration
                self._update_config(data.get('data', {}))

            elif msg_type == 'enable_remote':
                # Enable remote processing
                feature = data.get('feature')
                if feature == 'detection':
                    self.use_remote_detection = True
                elif feature == 'planning':
                    self.use_remote_planning = True

            elif msg_type == 'disable_remote':
                # Disable remote processing
                feature = data.get('feature')
                if feature == 'detection':
                    self.use_remote_detection = False
                elif feature == 'planning':
                    self.use_remote_planning = False

            elif msg_type == 'request_path_memory':
                # Computer requesting path memory sync
                await self._sync_path_memory()

            elif msg_type == 'path_memory_update':
                # Receive processed/optimized path data from computer
                self._import_path_data(data.get('data', {}))

            elif msg_type == 'optimized_route':
                # Computer sends optimized route based on historical data
                route = data.get('data', {})
                if route.get('waypoints'):
                    self.car.current_path = route['waypoints']
                    logger.info(f"Received optimized route with {len(route['waypoints'])} waypoints")

            if self.on_command:
                self.on_command(data)

        except json.JSONDecodeError:
            logger.error("Invalid JSON from computer")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    def _execute_command(self, command_data: dict) -> dict:
        """Execute command from computer"""
        command = command_data.get('command')
        result = {'success': True}

        try:
            if command == 'forward':
                self.car.manual_forward(command_data.get('speed', 50))
            elif command == 'backward':
                self.car.manual_backward(command_data.get('speed', 50))
            elif command == 'left':
                self.car.manual_left(command_data.get('speed', 50))
            elif command == 'right':
                self.car.manual_right(command_data.get('speed', 50))
            elif command == 'stop':
                self.car.manual_stop()
            elif command == 'set_speed':
                left = command_data.get('left', 0)
                right = command_data.get('right', 0)
                self.car.motors.set_speed(left, right)
            elif command == 'camera':
                pan = command_data.get('pan', 90)
                tilt = command_data.get('tilt', 45)
                self.car.move_camera(pan, tilt)
            else:
                result['success'] = False
                result['error'] = f'Unknown command: {command}'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)

        return result

    async def _send(self, data: dict):
        """Send data to computer"""
        if self.websocket:
            await self.websocket.send(json.dumps(data))

    async def _send_frame(self):
        """Send camera frame to computer"""
        frame = self.car.get_frame()
        if frame is not None:
            # Compress and encode
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode()

            await self._send({
                'type': 'frame',
                'data': frame_b64,
                'width': frame.shape[1],
                'height': frame.shape[0]
            })

    def _apply_detection_result(self, result: dict):
        """Apply detection result from computer"""
        # Use computer's detection result for navigation
        if self.use_remote_detection:
            self.car.last_detection_result = result
            logger.debug("Applied remote detection result")

    def _apply_path(self, path: list):
        """Apply computed path from computer"""
        if self.use_remote_planning:
            self.car.current_path = path
            logger.info(f"Applied remote path with {len(path)} waypoints")

    def _update_config(self, config_update: dict):
        """Update car configuration from computer"""
        for key, value in config_update.items():
            config.set(key, value)
        logger.info("Configuration updated from computer")

    def _import_path_data(self, data: dict):
        """Import processed path data from computer"""
        if hasattr(self.car, 'import_path_data'):
            self.car.import_path_data(data)
            logger.info("Imported path memory update from computer")

    async def _sync_path_memory(self):
        """Send path memory to computer for processing/storage"""
        try:
            path_export = self.car.get_path_memory_export()
            await self._send({
                'type': 'path_memory_sync',
                'data': path_export
            })
            logger.info(f"Synced path memory: {path_export.get('total_paths', 0)} paths")
        except Exception as e:
            logger.error(f"Failed to sync path memory: {e}")

    def send_status(self):
        """Send current status to computer"""
        if self.is_connected:
            asyncio.run(self._send({
                'type': 'status',
                'data': self.car.get_status()
            }))

    def sync_path_memory(self):
        """Manually trigger path memory sync to computer"""
        if self.is_connected:
            asyncio.run(self._sync_path_memory())

    def send_recorded_path(self, path_data: dict):
        """Send a newly recorded path to computer"""
        if self.is_connected:
            asyncio.run(self._send({
                'type': 'new_path',
                'data': path_data
            }))

    def send_detection(self, detection_result: dict):
        """Send detection result to computer"""
        if self.is_connected:
            asyncio.run(self._send({
                'type': 'detection',
                'data': detection_result
            }))

    def request_path(self, start: tuple, goal: tuple, obstacles: list):
        """Request path computation from computer"""
        if self.is_connected:
            asyncio.run(self._send({
                'type': 'request_path',
                'data': {
                    'start': start,
                    'goal': goal,
                    'obstacles': obstacles
                }
            }))
