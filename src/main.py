"""
Main entry point for SmartCar
"""

import argparse
import logging
import signal
import sys
import time

from .controller import SmartCarController, CarMode
from .network.server import CarServer
from .network.client import ComputerClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartCar:
    """
    Main application class for SmartCar
    """

    def __init__(self):
        self.controller = SmartCarController()
        self.server = CarServer(self.controller)
        self.client = ComputerClient(self.controller)

        self.is_running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)

    def start(self, mode: str = 'autonomous', enable_server: bool = True,
              connect_computer: bool = False):
        """
        Start the SmartCar

        Args:
            mode: Operating mode ('autonomous', 'manual', 'assisted')
            enable_server: Start the control server
            connect_computer: Connect to computer for extended capabilities
        """
        try:
            logger.info("Starting SmartCar...")

            # Initialize controller
            self.controller.initialize()

            # Start server if enabled
            if enable_server:
                self.server.start()

            # Connect to computer if enabled
            if connect_computer:
                self.client.start()

            # Start controller in specified mode
            car_mode = CarMode(mode)
            self.controller.start(car_mode)

            self.is_running = True
            logger.info(f"SmartCar running in {mode} mode")

            # Keep running
            while self.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the SmartCar"""
        self.is_running = False

        self.client.stop()
        self.server.stop()
        self.controller.cleanup()

        logger.info("SmartCar stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SmartCar - Raspberry Pi Car with NeoYolo Obstacle Detection'
    )

    parser.add_argument(
        '-m', '--mode',
        choices=['autonomous', 'manual', 'assisted', 'pathfinding'],
        default='autonomous',
        help='Operating mode (default: autonomous)'
    )

    parser.add_argument(
        '--no-server',
        action='store_true',
        help='Disable network server'
    )

    parser.add_argument(
        '-c', '--connect',
        action='store_true',
        help='Connect to computer for extended capabilities'
    )

    parser.add_argument(
        '--computer-host',
        type=str,
        help='Computer server host address'
    )

    parser.add_argument(
        '--computer-port',
        type=int,
        help='Computer server port'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load custom config if specified
    if args.config:
        from .config import config
        config.load_config(args.config)

    # Update computer connection settings
    if args.computer_host:
        from .config import config
        config.set('network.computer.host', args.computer_host)

    if args.computer_port:
        from .config import config
        config.set('network.computer.port', args.computer_port)

    # Create and start SmartCar
    car = SmartCar()
    car.start(
        mode=args.mode,
        enable_server=not args.no_server,
        connect_computer=args.connect
    )


if __name__ == '__main__':
    main()
