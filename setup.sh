#!/bin/bash
# ============================================
# Ergonomic Posture Monitor - Pi 5 Setup
# Run this once: bash setup.sh
# ============================================

echo "========================================"
echo "  Ergonomic Posture Monitor Setup"
echo "  Pi 5 + Camera Module 2 + Buzzer"
echo "========================================"

# Update system
echo "[1/5] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "[2/5] Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    python3-libcamera \
    python3-numpy \
    libcap-dev \
    libatlas-base-dev \
    libopenblas-dev

# Create virtual environment WITH system packages access
# (needed because picamera2 is installed system-wide)
echo "[3/5] Creating Python virtual environment..."
cd ~/ergonomic
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install Python packages
echo "[4/5] Installing Python packages..."
pip install --upgrade pip
pip install mediapipe opencv-python-headless gpiozero

# Test imports
echo "[5/5] Testing imports..."
python3 -c "
import cv2; print(f'  OpenCV: {cv2.__version__}')
import mediapipe; print(f'  MediaPipe: {mediapipe.__version__}')
import numpy; print(f'  NumPy: {numpy.__version__}')
try:
    from picamera2 import Picamera2; print('  picamera2: OK')
except: print('  picamera2: NOT FOUND (will use OpenCV fallback)')
try:
    from gpiozero import Buzzer; print('  gpiozero: OK')
except: print('  gpiozero: NOT FOUND (buzzer disabled)')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  To run:  cd ~/ergonomic"
echo "           source venv/bin/activate"
echo "           python3 posture_monitor.py"
echo "========================================"
