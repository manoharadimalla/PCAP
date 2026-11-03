@echo off
title RunwayMC - Interactive Terminal Demo
color 0A

echo.
echo  ============================================================
echo    RunwayMC - Interactive Demo Launcher
echo  ============================================================
echo.

REM Optional path override for another computer:
REM set PATH=C:\path\to\python;C:\path\to\mingw\bin;%PATH%
set PATH=C:\Users\Admin\AppData\Roaming\Python\Python314\Scripts;C:\msys64\ucrt64\bin;F:\git\Git\cmd;%PATH%

cd /d "%~dp0"
python run_demo.py

echo.
echo  ============================================================
echo    Demo finished! Press any key to close this window.
echo  ============================================================
pause >nul
