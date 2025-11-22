# SmartCar - Raspberry Pi Car with NeoYolo Obstacle Detection

A comprehensive autonomous car system for Raspberry Pi with camera arm, NeoYolo-based obstacle detection, A* pathfinding, and extended computer connectivity capabilities.

## Features

- **NeoYolo Obstacle Detection**: Real-time object detection using YOLO-based neural networks
- **Camera Arm Control**: Pan/tilt servo-controlled camera for environment scanning
- **A* Pathfinding**: Grid-based pathfinding with obstacle avoidance
- **Reactive Obstacle Avoidance**: Sensor fusion from ultrasonic sensors and camera
- **Path Memory & Learning**: Records routes, learns from previous paths, identifies obstacle hotspots
- **Network Control**: WebSocket and HTTP server for remote control
- **Computer Connectivity**: Extended processing capabilities when connected to a computer with GPU
- **Path Transmission**: Syncs path memory to computer for analysis, optimization, and storage

## Hardware Requirements

### Raspberry Pi
- Raspberry Pi 4 (recommended) or Pi 3
- Pi Camera Module v2
- L298N Motor Driver
- 2x DC Motors
- PCA9685 Servo Controller
- 2x Servo Motors (pan/tilt)
- 3x HC-SR04 Ultrasonic Sensors
- 2x IR Sensors (optional, for line following)
- Battery pack (7.4V+ recommended)

### Computer (for extended capabilities)
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM

## Installation

### On Raspberry Pi

```bash
# Clone the repository
git clone https://github.com/your-repo/SmartCar.git
cd SmartCar

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Pi-specific packages (run on Pi only)
pip install RPi.GPIO picamera2 adafruit-circuitpython-servokit
```

### On Computer (for extended capabilities)

```bash
# Clone the repository
git clone https://github.com/your-repo/SmartCar.git
cd SmartCar

# Install dependencies
pip install torch torchvision ultralytics websockets opencv-python numpy
```

## Configuration

Edit `config/settings.yaml` to match your hardware setup:

- GPIO pin assignments
- Servo channels and limits
- Camera settings
- Network addresses
- Safety thresholds

## Usage

### Basic Autonomous Mode

```bash
# On Raspberry Pi
python -m src.main --mode autonomous
```

### With Network Server

```bash
# Starts HTTP server on port 5000 and WebSocket on 8765
python -m src.main --mode autonomous
```

Access the video feed at: `http://<pi-ip>:5000/video_feed`

### Manual Control Mode

```bash
python -m src.main --mode manual
```

### With Computer Connection

```bash
# On Computer - Start the server
python computer/server.py --port 9000

# On Raspberry Pi - Connect to computer
python -m src.main --mode assisted --connect --computer-host 192.168.1.100
```

### Command Line Options

```
-m, --mode          Operating mode: autonomous, manual, assisted, pathfinding
--no-server         Disable network server
-c, --connect       Connect to computer for extended capabilities
--computer-host     Computer server hostname/IP
--computer-port     Computer server port
-v, --verbose       Enable verbose logging
--config            Path to custom config file
```

## API Endpoints

### HTTP Endpoints

- `GET /status` - Get car status
- `GET /video_feed` - MJPEG video stream
- `GET /snapshot` - Single frame with detections
- `POST /control` - Send control command

### WebSocket Commands

```json
{
  "type": "command",
  "command": "forward",
  "speed": 50
}
```

Commands: `forward`, `backward`, `left`, `right`, `stop`, `emergency_stop`, `camera_move`, `camera_center`, `scan`, `set_goal`

## Architecture

```
SmartCar/
├── config/
│   └── settings.yaml         # Configuration
├── src/
│   ├── hardware/
│   │   ├── motors.py         # Motor control (L298N)
│   │   ├── servos.py         # Servo control (PCA9685)
│   │   └── sensors.py        # Ultrasonic & IR sensors
│   ├── camera/
│   │   ├── capture.py        # PiCamera2 capture
│   │   └── arm.py            # Camera arm control
│   ├── detection/
│   │   └── neoyolo.py        # YOLO obstacle detection
│   ├── navigation/
│   │   ├── pathfinding.py    # A* pathfinding
│   │   ├── obstacle_avoidance.py  # Reactive avoidance
│   │   └── path_memory.py    # Route learning & memory
│   ├── network/
│   │   ├── server.py         # Control server (on Pi)
│   │   └── client.py         # Computer connection
│   ├── controller.py         # Main car controller
│   ├── config.py             # Configuration manager
│   └── main.py               # Entry point
├── computer/
│   └── server.py             # Computer-side server
├── data/
│   └── paths/                # Stored path memory
├── models/                   # YOLO models
└── requirements.txt
```

## Path Memory & Learning

The SmartCar learns from previous routes to improve navigation over time:

### How It Works

1. **Recording**: During autonomous operation, the car records:
   - Position coordinates (x, y)
   - Heading and speed
   - Obstacle detections
   - Sensor readings at each point

2. **Local Storage**: Paths are stored locally on the Pi in `data/paths/`

3. **Computer Sync**: When connected to a computer:
   - Path memory is transmitted for analysis
   - Computer aggregates data from multiple cars
   - Optimized routes are computed and sent back
   - Obstacle heatmaps identify dangerous areas

### Features

- **Historical Routes**: Uses successful past routes for navigation
- **Obstacle Hotspots**: Identifies frequently encountered obstacle locations
- **Avoidance Zones**: Automatically avoids areas with high failure rates
- **Route Optimization**: Computer analyzes and simplifies routes
- **Multi-Car Learning**: Computer aggregates learning from multiple cars

### Usage

```python
# The car automatically records paths during autonomous mode
car.start(mode=CarMode.AUTONOMOUS, record_path=True)

# Navigate using historical paths
car.set_goal(x=2.0, y=3.0, use_memory=True)

# Export path memory for analysis
path_data = car.get_path_memory_export()

# Get obstacle hotspots
hotspots = car.get_obstacle_hotspots()
```

### Computer-Side Storage

The computer server stores paths from all connected cars and provides:
- Centralized path database
- Cross-car learning
- Advanced route optimization
- Obstacle heatmap generation

## Extended Capabilities (Computer Connection)

When connected to a computer, the SmartCar gains:

1. **GPU-Accelerated Detection**: Use larger YOLO models (YOLOv8m/l) on computer GPU
2. **Advanced Path Planning**: Complex algorithms (RRT*, D*) with more compute
3. **Data Logging**: Store and analyze detection/navigation data
4. **Remote Dashboard**: Full monitoring and control interface

## Detection Zones

The detection system uses three zones:

- **Danger Zone** (bottom 40%): Immediate threat, triggers stop
- **Warning Zone** (middle 30%): Upcoming obstacle, triggers slow down
- **Safe Zone** (top 30%): Distant objects, continue normally

## Safety Features

- Emergency stop when obstacles < 15cm
- Automatic slow down in warning zone
- Manual override capability
- Battery voltage monitoring
- Maximum runtime limit

## Customization

### Custom Detection Classes

Edit `config/settings.yaml`:

```yaml
detection:
  classes:
    - person
    - car
    - custom_object
```

### Custom Pathfinding

Modify `src/navigation/pathfinding.py` to implement different algorithms (RRT, D*, etc.)

## Troubleshooting

### Camera not working
- Ensure Pi Camera is enabled: `sudo raspi-config`
- Check cable connection
- Verify with: `libcamera-hello`

### Motors not responding
- Check GPIO pin assignments
- Verify L298N connections
- Check battery voltage

### Detection slow
- Use smaller model (yolov8n.pt)
- Reduce input resolution
- Connect to computer for GPU acceleration

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests with tests and documentation.
