@echo off
REM One-time setup on Windows. Safe to run again; it reuses an existing .venv.
setlocal
cd /d "%~dp0"

echo ============================================
echo   EEE_project - vehicle detection setup
echo ============================================
echo.

REM -- 1. Pick an interpreter ---------------------------------------------
REM onnxruntime is roughly 2x faster than the OpenCV DNN fallback, but it
REM publishes no wheels above Python 3.12. Prefer a version it supports and
REM only fall back to whatever "python" happens to be on PATH.
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
)
if not defined PY (
    py -3.10 --version >nul 2>&1 && set "PY=py -3.10"
)
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] No Python installation found.
    echo.
    echo   Install Python 3.12 from https://www.python.org/downloads/
    echo   and tick "Add python.exe to PATH" in the installer.
    goto :fail
)

for /f "delims=" %%I in ('%PY% --version 2^>^&1') do set "PYVER=%%I"
echo   interpreter : %PYVER%
echo.

REM -- 2. Virtual environment --------------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo   venv        : reusing existing .venv
) else (
    echo   venv        : creating .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        goto :fail
    )
)
set "VPY=.venv\Scripts\python.exe"
echo.

REM -- 3. Required packages ----------------------------------------------
echo Installing required packages ^(opencv, numpy^) ...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Could not install the required packages.
    echo         Check your internet connection and run setup.bat again.
    goto :fail
)
echo   done.
echo.

REM -- 4. Optional speedup -----------------------------------------------
REM Failure here is not fatal: detector.py falls back to the OpenCV DNN
REM backend, which works everywhere but is about half the speed.
echo Installing onnxruntime ^(optional, about 2x faster^) ...
"%VPY%" -m pip install onnxruntime --quiet >nul 2>&1
if errorlevel 1 (
    echo   [skipped] No onnxruntime wheel for this Python version.
    echo             The slower OpenCV DNN backend will be used instead.
) else (
    echo   done.
)
echo.

REM -- 5. Optional demo dependency ---------------------------------------
REM PyAV bundles ffmpeg and is only needed by run-demo.bat, which replays
REM the recorded clip as a fake IP camera.
echo Installing PyAV ^(optional, needed only by run-demo.bat^) ...
"%VPY%" -m pip install av --quiet >nul 2>&1
if errorlevel 1 (
    echo   [skipped] PyAV unavailable; run-demo.bat will not work.
) else (
    echo   done.
)
echo.

REM -- 6. Check the large files that git does not carry ------------------
echo ============================================
echo   Checking files not stored in git
echo ============================================
set "READY=1"
if exist "models\EEE_project_yolov9s.onnx" (
    echo   [ok]      models\EEE_project_yolov9s.onnx
) else (
    echo   [MISSING] models\EEE_project_yolov9s.onnx
    echo             Model weights are too large for git. Copy the .onnx
    echo             file into the models\ folder before running.
    set "READY=0"
)
if exist "demo\dhaka\EEE_project_demo.mp4" (
    echo   [ok]      demo\dhaka\EEE_project_demo.mp4
) else (
    echo   [absent]  demo\dhaka\EEE_project_demo.mp4
    echo             Only needed by run-demo.bat. Live camera and webcam
    echo             sources work without it.
)
echo.

echo ============================================
if "%READY%"=="1" (
    echo   Setup complete.
) else (
    echo   Setup finished, but the model file is missing.
)
echo ============================================
echo.
echo   run.bat                     pick a source interactively
echo   run.bat --source 0          laptop webcam
echo   run.bat --source rtsp://... IP camera
echo   run-demo.bat                replay the recorded clip
echo.
echo   Then open http://localhost:8000 in a browser.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
