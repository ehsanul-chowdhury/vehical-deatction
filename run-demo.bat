@echo off
REM Replay the recorded clip through the full live pipeline, as if it were an
REM IP camera. Useful when no real camera is to hand.
REM
REM The two processes have a required start order: the detector must be
REM listening before the publisher dials in. Getting that wrong by hand
REM leaves rtsp_server.py printing "waiting for a detector", so this script
REM starts them in the right order with a pause in between.
setlocal
cd /d "%~dp0"

set "CLIP=demo\dhaka\EEE_project_demo.mp4"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)
if not exist "models\EEE_project_yolov9s.onnx" (
    echo [ERROR] Missing models\EEE_project_yolov9s.onnx - copy it into models\
    echo.
    pause
    exit /b 1
)
if not exist "%CLIP%" (
    echo [ERROR] Missing %CLIP%
    echo         The rendered clip is not stored in git. Copy it in, or use
    echo         run.bat with a real camera or webcam instead.
    echo.
    pause
    exit /b 1
)

REM Tell OpenCV to listen for an incoming RTSP session instead of dialling
REM out. The quotes keep cmd from treating the pipe as a command separator.
REM "start" passes this environment on to the detector window.
set "OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_flags;listen|rtsp_transport;tcp"

echo Starting the detector in a second window ...
start "EEE detector" cmd /k ".venv\Scripts\python.exe detect.py --source rtsp://127.0.0.1:8554/live"

echo Waiting for it to listen on port 8554 ...
timeout /t 6 /nobreak >nul

echo Publishing %CLIP% to rtsp://127.0.0.1:8554/live
echo.
echo   Open http://localhost:8000 in a browser.
echo   Press ctrl-c here to stop the publisher; close the other window to
echo   stop the detector.
echo.

.venv\Scripts\python.exe demo\rtsp_server.py --source "%CLIP%"

echo.
pause
exit /b 0
