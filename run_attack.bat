@echo off
cd /d "%~dp0radioAdv"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] venv not found: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Running FGSM white-box attack...
"%~dp0.venv\Scripts\python.exe" attack.py --config parameters/attack/fgsm_vtcnn2_white_attack_win.yaml --vGPU 0
echo.
if errorlevel 1 (
    echo [ERROR] attack failed, exit code %errorlevel%
) else (
    echo [OK] attack finished.
)
pause
