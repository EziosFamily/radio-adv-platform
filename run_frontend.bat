@echo off
cd /d "%~dp0173_vue\173_vue"
if exist "node_modules\vite\bin\vite.js" (
    echo Starting Vue frontend: http://127.0.0.1:5173
    call npm run dev -- --host 127.0.0.1 --port 5173
) else (
    echo [ERROR] node_modules not found. Run npm install in 173_vue\173_vue
    pause
    exit /b 1
)
pause
