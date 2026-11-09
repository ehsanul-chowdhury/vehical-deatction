@echo off
REM Start the detector. Any arguments are passed straight to detect.py, so
REM   run.bat --source 0 --quality 60
REM behaves exactly like calling detect.py by hand.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found.
    echo         Run setup.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "models\EEE_project_yolov9s.onnx" (
    echo [ERROR] Missing models\EEE_project_yolov9s.onnx
    echo         Model weights are not stored in git. Copy the .onnx file
    echo         into the models\ folder, then run this again.
    echo.
    pause
    exit /b 1
)

echo Starting detector. The page address is printed below.
echo Press ctrl-c in this window to stop.
echo.

.venv\Scripts\python.exe detect.py %*

echo.
pause
exit /b 0
