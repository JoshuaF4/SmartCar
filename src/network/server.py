"""
WebSocket and HTTP server for remote control and monitoring
Runs on the Raspberry Pi
"""

import json
import logging
import asyncio
import base64
from typing import Optional, Set
from threading import Thread

try:
    import websockets
    from flask import Flask, Response, jsonify, request
    from flask_socketio import SocketIO, emit
except ImportError:
    websockets = None
    Flask = None
    SocketIO = None

import cv2

from ..config import config

logger = logging.getLogger(__name__)


class CarServer:
    """
    Server for remote control and monitoring of the car
    Provides WebSocket for real-time control and HTTP for video streaming
    """

    def __init__(self, car_controller):
        self.car = car_controller
        self.config = config.network.get('local', {})

        self.host = self.config.get('host', '0.0.0.0')
        self.ws_port = self.config.get('websocket_port', 8765)
        self.http_port = self.config.get('http_port', 5000)

        # Connected clients
        self.ws_clients: Set = set()

        # Flask app for HTTP
        self.app = Flask(__name__) if Flask else None
        self.socketio = SocketIO(self.app, cors_allowed_origins="*") if SocketIO else None

        # Server threads
        self._ws_thread: Optional[Thread] = None
        self._http_thread: Optional[Thread] = None

        self.is_running = False

        if self.app:
            self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/status')
        def status():
            return jsonify(self.car.get_status())

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_video(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/snapshot')
        def snapshot():
            frame = self.car.get_frame_with_detections()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                return Response(buffer.tobytes(), mimetype='image/jpeg')
            return 'No frame available', 404

        @self.app.route('/control', methods=['POST'])
        def control():
            data = request.json
            return jsonify(self._handle_command(data))

        # SocketIO events
        if self.socketio:
            @self.socketio.on('connect')
            def handle_connect():
                logger.info(f"Client connected")
                emit('status', self.car.get_status())

            @self.socketio.on('command')
            def handle_command(data):
                result = self._handle_command(data)
                emit('response', result)

            @self.socketio.on('set_mode')
            def handle_set_mode(data):
                mode = data.get('mode')
                if mode:
                    from ..controller import CarMode
                    self.car.mode = CarMode(mode)
                    emit('status', self.car.get_status())

    def _generate_video(self):
        """Generate video frames for streaming"""
        streaming_config = config.network.get('streaming', {})
        quality = streaming_config.get('quality', 80)

        while self.is_running:
            frame = self.car.get_frame_with_detections()
            if frame is not None:
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

    def _handle_command(self, data: dict) -> dict:
        """Handle control command"""
        command = data.get('command')
        result = {'success': True, 'command': command}

        try:
            if command == 'forward':
                speed = data.get('speed', 50)
                self.car.manual_forward(speed)

            elif command == 'backward':
                speed = data.get('speed', 50)
                self.car.manual_backward(speed)

            elif command == 'left':
                speed = data.get('speed', 50)
                self.car.manual_left(speed)

            elif command == 'right':
                speed = data.get('speed', 50)
                self.car.manual_right(speed)

            elif command == 'stop':
                self.car.manual_stop()

            elif command == 'emergency_stop':
                self.car.emergency_stop()

            elif command == 'camera_move':
                pan = data.get('pan', 90)
                tilt = data.get('tilt', 45)
                self.car.move_camera(pan, tilt)

            elif command == 'camera_center':
                self.car.center_camera()

            elif command == 'scan':
                self.car.scan_environment()

            elif command == 'set_goal':
                x = data.get('x', 0)
                y = data.get('y', 0)
                self.car.set_goal(x, y)

            else:
                result['success'] = False
                result['error'] = f'Unknown command: {command}'

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            logger.error(f"Command error: {e}")

        return result

    async def _websocket_handler(self, websocket, path):
        """Handle WebSocket connection"""
        self.ws_clients.add(websocket)
        logger.info(f"WebSocket client connected from {websocket.remote_address}")

        try:
            # Send initial status
            await websocket.send(json.dumps({
                'type': 'status',
                'data': self.car.get_status()
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type', 'command')

                    if msg_type == 'command':
                        result = self._handle_command(data)
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'data': result
                        }))

                    elif msg_type == 'get_status':
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'data': self.car.get_status()
                        }))

                    elif msg_type == 'get_frame':
                        jpeg = self.car.camera.get_jpeg()
                        if jpeg:
                            await websocket.send(json.dumps({
                                'type': 'frame',
                                'data': base64.b64encode(jpeg).decode()
                            }))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket client disconnected")
        finally:
            self.ws_clients.remove(websocket)

    async def _broadcast(self, message: dict):
        """Broadcast message to all WebSocket clients"""
        if self.ws_clients:
            msg = json.dumps(message)
            await asyncio.gather(
                *[client.send(msg) for client in self.ws_clients],
                return_exceptions=True
            )

    def _run_websocket_server(self):
        """Run WebSocket server in thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        start_server = websockets.serve(
            self._websocket_handler,
            self.host,
            self.ws_port
        )

        loop.run_until_complete(start_server)
        logger.info(f"WebSocket server running on ws://{self.host}:{self.ws_port}")
        loop.run_forever()

    def _run_http_server(self):
        """Run HTTP/SocketIO server in thread"""
        if self.socketio:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.http_port,
                debug=False,
                use_reloader=False
            )
        elif self.app:
            self.app.run(
                host=self.host,
                port=self.http_port,
                debug=False,
                threaded=True
            )

    def start(self):
        """Start all servers"""
        self.is_running = True

        # Start WebSocket server
        if websockets:
            self._ws_thread = Thread(target=self._run_websocket_server)
            self._ws_thread.daemon = True
            self._ws_thread.start()

        # Start HTTP server
        if self.app:
            self._http_thread = Thread(target=self._run_http_server)
            self._http_thread.daemon = True
            self._http_thread.start()
            logger.info(f"HTTP server running on http://{self.host}:{self.http_port}")

        # Setup car callbacks
        self.car.on_status_update = self._on_status_update

        logger.info("Car server started")

    def stop(self):
        """Stop all servers"""
        self.is_running = False
        logger.info("Car server stopped")

    def _on_status_update(self, status: dict):
        """Called when car status updates"""
        # Would broadcast to all clients
        pass

    def send_alert(self, alert_type: str, message: str):
        """Send alert to all connected clients"""
        # Implementation would broadcast via WebSocket
        pass
