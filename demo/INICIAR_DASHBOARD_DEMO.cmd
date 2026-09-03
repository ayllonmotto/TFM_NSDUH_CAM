@echo off
setlocal
title TFM NSDUH 2024 - Dashboard Demo

echo.
echo ============================================================
echo TFM NSDUH 2024 - DASHBOARD DEMO
echo ============================================================
echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL no esta disponible en este equipo.
    echo.
    pause
    exit /b 1
)

echo Iniciando dashboard_demo en WSL...
echo URL: http://127.0.0.1:5000/dashboard
echo.

wsl.exe bash -lc "cd ~/BD/TFM_NSDUH/demo && ./iniciar_dashboard_demo.sh"

echo.
echo dashboard_demo ha finalizado.
pause

endlocal
