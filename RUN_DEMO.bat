@echo off
title RunwayMC - Interactive Demo
color 0A

echo.
echo  ============================================================
echo    RunwayMC - Interactive Demo Launcher
echo  ============================================================
echo.

REM Set up PATH so tools are found
set PATH=C:\Users\Admin\AppData\Roaming\Python\Python314\Scripts;C:\msys64\ucrt64\bin;F:\git\Git\cmd;%PATH%

REM Run the demo
cd /d "%~dp0"
python run_demo.py

echo.
echo  ============================================================
echo    Demo finished! Press any key to close this window.
echo  ============================================================
pause >nul
