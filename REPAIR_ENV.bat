@echo off
setlocal EnableExtensions
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "RUNTIME=%LOCALAPPDATA%\PrimateBehaviorAI\runtime_py38"
echo This will remove ONLY the shared Python runtime:
echo   %RUNTIME%
echo.
echo It will NOT delete videos, analysis results, database, models, Anaconda, or system Python.
echo The next START_APP.bat launch will rebuild the runtime once.
echo.
choice /C YN /N /M "Continue? [Y/N]: "
if errorlevel 2 exit /b 0
if exist "%RUNTIME%" rmdir /s /q "%RUNTIME%"
echo Shared runtime removed.
echo Run START_APP.bat to rebuild it.
pause
