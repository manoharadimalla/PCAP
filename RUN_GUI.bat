@echo off
title RunwayMC - Desktop GUI
color 0B

echo.
echo  ============================================================
echo    RunwayMC - Native Desktop GUI Launcher (PySide6)
echo  ============================================================
echo.

set MPLCONFIGDIR=%TEMP%\matplotlib

cd /d "%~dp0"
C:\Python314\python.exe -m gui

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] An error occurred while launching the GUI.
    echo.
    pause
)

