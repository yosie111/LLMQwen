@echo off
chcp 65001 >nul
title Qwen Chat - Setup & Launch
color 0A

echo ╔══════════════════════════════════════════════╗
echo ║         Qwen Chat UI - Installer             ║
echo ║         התקנת ממשק צ'אט Qwen                 ║
echo ╚══════════════════════════════════════════════╝
echo.

:: Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
python --version
echo.

:: Create virtual environment
echo [2/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: Activate and install
echo [3/4] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat

pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Try running: pip install torch transformers flask accelerate
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

:: Launch
echo [4/4] Launching Qwen Chat UI...
echo.
echo ╔══════════════════════════════════════════════╗
echo ║  The model will download on first run (~1GB)  ║
echo ║  המודל יורד בהפעלה הראשונה (~1GB)             ║
echo ║                                              ║
echo ║  Open in browser / פתח בדפדפן:               ║
echo ║  http://localhost:5000                        ║
echo ║                                              ║
echo ║  Press Ctrl+C to stop / לעצירה לחץ Ctrl+C    ║
echo ╚══════════════════════════════════════════════╝
echo.

python app.py

pause
