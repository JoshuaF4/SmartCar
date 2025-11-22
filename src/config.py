"""
Configuration management for SmartCar
"""

import os
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Configuration manager that loads settings from YAML file"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "settings.yaml"

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('hardware.motors.left_forward')
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            config = config.setdefault(k, {})

        config[keys[-1]] = value

    @property
    def hardware(self):
        return self._config.get('hardware', {})

    @property
    def camera(self):
        return self._config.get('camera', {})

    @property
    def detection(self):
        return self._config.get('detection', {})

    @property
    def navigation(self):
        return self._config.get('navigation', {})

    @property
    def network(self):
        return self._config.get('network', {})

    @property
    def safety(self):
        return self._config.get('safety', {})


# Global config instance
config = Config()
