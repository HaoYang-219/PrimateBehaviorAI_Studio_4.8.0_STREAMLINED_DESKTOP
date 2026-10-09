@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo [1/5] Checking Python 3.11 for Windows build...
where python >nul 2>nul
if errorlevel 1 goto :no_python
python -c "import sys,struct; assert sys.version_info[:2] == (3,11) and struct.calcsize('P')*8 == 64, 'Need Windows Python 3.11 x64'"
if errorlevel 1 goto :no_python
if not exist .buildvenv\Scripts\python.exe (
    python -m venv .buildvenv
    if errorlevel 1 goto :error
)
set "BYPY=.buildvenv\Scripts\python.exe"
echo [2/5] Installing dependencies in PRIVATE BUILD venv (not on end user PC)...
"%BYPY%" -m pip --disable-pip-version-check install -r requirements.txt -r requirements-desktop-build.txt
if errorlevel 1 goto :error
echo [3/5] Running frozen-runtime preflight...
"%BYPY%" -c "from primate_ai.self_test import run_self_test; run_self_test()"
if errorlevel 1 goto :error
echo [4/5] Packaging onedir Windows EXE...
"%BYPY%" -m PyInstaller --noconfirm --clean --windowed --name PrimateBehaviorAI --icon assets\app_icon.ico --add-data "assets;assets" --add-data "models;models" --collect-all PySide6 --collect-all cv2 --collect-submodules sklearn --hidden-import pandas --hidden-import matplotlib --hidden-import openpyxl main.py
if errorlevel 1 goto :error
if not exist dist\PrimateBehaviorAI\PrimateBehaviorAI.exe goto :error
echo [5/5] Checking frozen executable exists...
echo Frozen EXE ready: dist\PrimateBehaviorAI\PrimateBehaviorAI.exe
exit /b 0
:no_python
echo This file is for the SOFTWARE BUILDER, not the end user.
echo Please use Python 3.11 x64 on a Windows build machine, or GitHub Actions.
exit /b 1
:error
echo BUILD FAILED; check error output.
exit /b 1
