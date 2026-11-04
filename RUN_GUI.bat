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

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m gui
) else (
    if exist "C:\Python314\python.exe" (
        C:\Python314\python.exe -m gui
    ) else (
        echo.
        echo  [ERROR] Python is not found in PATH or at C:\Python314\python.exe.
        echo.
        pause
        exit /b 1
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] An error occurred while launching the GUI.
    echo.
    pause
)

