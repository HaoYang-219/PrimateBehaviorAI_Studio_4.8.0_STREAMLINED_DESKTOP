@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo PrimateBehaviorAI now uses a shared runtime.
echo You normally do NOT need to run INSTALL.bat.
echo.
echo START_APP.bat automatically initializes the runtime on first launch.
echo Running this file will perform the same one-time initialization.
echo.
call START_APP.bat
