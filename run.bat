@echo off
chcp 65001 >nul
title Qwen Chat UI
color 0B

echo ╔══════════════════════════════════════════════╗
echo ║           Qwen Chat UI - Launch              ║
echo ║           הפעלת ממשק צ'אט Qwen               ║
echo ╚══════════════════════════════════════════════╝
echo.

:: Check venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run install_and_run.bat first.
    pause
    exit /b 1
)

:: Activate
call .venv\Scripts\activate.bat

echo Starting server...
echo Open in browser: http://localhost:5000
echo Press Ctrl+C to stop
echo.

python app.py

pause
