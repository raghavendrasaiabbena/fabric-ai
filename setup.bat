@echo off
echo ============================================================
echo  Fabric AI - Setup
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/4] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Activating environment and installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency install failed.
    pause
    exit /b 1
)

echo.
echo [3/4] Creating required directories...
if not exist uploads mkdir uploads
if not exist enhanced mkdir enhanced
if not exist models mkdir models

echo.
echo [4/4] Downloading AI super-resolution models...
python backend\download_models.py

echo.
echo ============================================================
echo  Setup complete!
echo  Run:  run.bat
echo ============================================================
pause
