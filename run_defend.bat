@echo off
cd /d "%~dp0radioAdv"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] venv not found: %~dp0.venv\Scripts\python.exe
    pause
    exit /b 1
)
echo Running FGSM adversarial training...
"%~dp0.venv\Scripts\python.exe" defend.py --config parameters/defend/fgsm_adv_train_win.yaml --vGPU 0
echo.
if errorlevel 1 (
    echo [ERROR] training failed, exit code %errorlevel%
) else (
    echo [OK] training finished.
)
pause
