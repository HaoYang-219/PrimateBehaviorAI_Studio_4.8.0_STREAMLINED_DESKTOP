@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "RUNTIME=%LOCALAPPDATA%\PrimateBehaviorAI\runtime_py38"
set "RPY=%RUNTIME%\Scripts\python.exe"
if not exist "%RPY%" (
    echo Shared runtime is not initialized yet.
    echo Run START_APP.bat once first.
    pause
    exit /b 1
)
"%RPY%" "%CD%\runtime_bootstrap.py" --ensure --requirements "%CD%\requirements.txt"
if errorlevel 1 (
    pause
    exit /b 1
)
"%RPY%" "%CD%\main.py"
echo.
echo App exited. Press any key to close this window.
pause >nul
