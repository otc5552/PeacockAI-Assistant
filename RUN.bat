@echo off
title PeacockAgent Setup
color 0A

echo.
echo  ===================================
echo   🦚 PeacockAgent - Setup
echo  ===================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/2] Installing dependencies...
pip install -r requirements.txt --quiet

echo [2/2] Launching PeacockAgent...
echo.
python main.py

pause
