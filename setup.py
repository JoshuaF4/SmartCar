"""
Setup script for SmartCar
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="smartcar",
    version="1.0.0",
    author="SmartCar Team",
    description="Raspberry Pi Car with NeoYolo Obstacle Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "websockets>=11.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "pi": [
            "RPi.GPIO>=0.7.1",
            "picamera2>=0.3.12",
            "adafruit-circuitpython-servokit>=1.3.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smartcar=src.main:main",
            "smartcar-server=computer.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
