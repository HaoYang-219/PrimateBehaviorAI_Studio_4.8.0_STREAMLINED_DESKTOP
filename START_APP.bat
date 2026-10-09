@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "APPHOME=%LOCALAPPDATA%\PrimateBehaviorAI"
set "RUNTIME=%APPHOME%\runtime_py38"
set "RPY=%RUNTIME%\Scripts\python.exe"
set "RPYW=%RUNTIME%\Scripts\pythonw.exe"
set "BASEPY="

rem Fast path: existing shared runtime. No dependency installation when requirements are unchanged.
if exist "%RPY%" (
    "%RPY%" "%CD%\runtime_bootstrap.py" --ensure --requirements "%CD%\requirements.txt"
    if errorlevel 1 goto :runtime_error
    start "" "%RPYW%" "%CD%\main.py"
    exit /b 0
)

echo ============================================================
echo  PrimateBehaviorAI Studio 4.8.0
 echo First launch: creating shared runtime once.
echo  Future versions reuse the same runtime automatically.
echo ============================================================
echo.

rem Optional explicit Python override.
if defined PRIMATE_PYTHON if exist "%PRIMATE_PYTHON%" (
    "%PRIMATE_PYTHON%" -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2]==(3,8) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "BASEPY=%PRIMATE_PYTHON%"
)

rem Prefer active Conda only when it is exactly Python 3.8 x64.
if not defined BASEPY if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2]==(3,8) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "BASEPY=%CONDA_PREFIX%\python.exe"
)

rem Scan PATH.
if not defined BASEPY (
    for /f "delims=" %%P in ('where python.exe 2^>nul') do (
        if not defined BASEPY (
            "%%P" -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2]==(3,8) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
            if !errorlevel! equ 0 set "BASEPY=%%P"
        )
    )
)

if not defined BASEPY goto :no_python

"%BASEPY%" "%CD%\runtime_bootstrap.py" --ensure --requirements "%CD%\requirements.txt"
if errorlevel 1 goto :runtime_error

if not exist "%RPYW%" goto :runtime_error
start "" "%RPYW%" "%CD%\main.py"
exit /b 0

:no_python
echo.
echo ERROR: First launch needs 64-bit Python 3.8.x to create the shared runtime.
echo Your Python 3.8.5 can be used. If it is D:\anaconda3\python.exe, run:
echo   set PRIMATE_PYTHON=D:\anaconda3\python.exe
echo   START_APP.bat
echo.
pause
exit /b 1

:runtime_error
echo.
echo ERROR: Shared runtime initialization or validation failed.
echo Run DIAGNOSE.bat for details.
pause
exit /b 1
