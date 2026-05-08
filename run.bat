@echo off
echo ============================================================
echo  Fabric AI - Starting Server
echo ============================================================

cd /d "%~dp0\backend"

echo.
echo Server starting at: http://localhost:8000
echo Press Ctrl+C to stop.
echo.

set PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=python

"%PYTHON%" main.py
