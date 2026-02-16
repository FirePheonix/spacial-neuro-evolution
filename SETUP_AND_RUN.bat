@echo off
echo ========================================
echo Spatiotemporal AV Navigator Setup
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] Pip upgraded
echo.

echo Step 4: Installing dependencies...
echo This may take 2-3 minutes...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the demo:
echo   1. Make sure you're in the virtual environment
echo   2. cd examples
echo   3. python 01_basic_traffic_modeling.py
echo.
echo Or just run: RUN_DEMO.bat
echo.
pause
