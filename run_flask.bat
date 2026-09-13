@echo off
cd /d "%~dp0radioAdv"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] venv not found: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Starting Flask backend: http://127.0.0.1:5000
echo Close this window to stop the server.
echo.
"%~dp0.venv\Scripts\python.exe" app.py
echo.
echo Flask stopped.
pause
