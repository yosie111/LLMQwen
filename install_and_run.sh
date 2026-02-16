#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════╗"
echo "║         Qwen Chat UI - Setup & Launch        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed!"
    echo "Install with: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi
python3 --version
echo ""

# Create venv
echo "[2/4] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi
echo ""

# Activate and install
echo "[3/4] Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

echo "Dependencies installed!"
echo ""

# Launch
echo "[4/4] Launching..."
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Model downloads on first run (~1GB)         ║"
echo "║  Open: http://localhost:5000                 ║"
echo "║  Stop: Ctrl+C                               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

python app.py
