@echo off
echo ========================================
echo Running Spatiotemporal AV Navigator Demo
echo ========================================
echo.

if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run SETUP_AND_RUN.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

cd examples

echo Starting demo...
echo This will take 2-3 minutes...
echo.
python 01_basic_traffic_modeling.py

echo.
echo ========================================
echo Demo Complete!
echo ========================================
echo.
echo Check the visualizations\ folder for generated images!
echo.
pause
