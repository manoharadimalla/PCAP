@echo off
title RunwayMC - Desktop GUI
color 0B

echo.
echo  ============================================================
echo    RunwayMC - Native Desktop GUI Launcher (PySide6)
echo  ============================================================
echo.

REM Set up environment PATH
set PATH=C:\Users\Admin\AppData\Roaming\Python\Python314\Scripts;C:\msys64\ucrt64\bin;F:\git\Git\cmd;%PATH%
set MPLCONFIGDIR=%TEMP%\matplotlib

cd /d "%~dp0"
python -m gui

echo.
