@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "APPHOME=%LOCALAPPDATA%\PrimateBehaviorAI"
set "RUNTIME=%APPHOME%\runtime_py38"
set "RPY=%RUNTIME%\Scripts\python.exe"
set "OUT=%CD%\diagnostics.txt"
(
  echo PrimateBehaviorAI Studio 4.8.0 diagnostics
  echo ==========================================
  echo Date: %DATE% %TIME%
  echo Code folder: %CD%
  echo App home: %APPHOME%
  echo Shared runtime: %RUNTIME%
  echo.
  echo [PATH Python]
  where python.exe 2^>nul
  python -c "import sys,struct,platform; print(sys.executable); print(sys.version); print('Bits:',struct.calcsize('P')*8); print('Machine:',platform.machine())" 2^>^&1
  echo.
  echo [Shared runtime]
  if exist "%RPY%" (
    "%RPY%" "%CD%\runtime_bootstrap.py" --status --requirements "%CD%\requirements.txt" 2^>^&1
    "%RPY%" -m pip --version 2^>^&1
    "%RPY%" -c "import PySide6,cv2,numpy,pandas,sklearn,matplotlib; print('PySide6',PySide6.__version__); print('cv2',cv2.__version__); print('numpy',numpy.__version__); print('pandas',pandas.__version__); print('sklearn',sklearn.__version__); print('matplotlib',matplotlib.__version__)" 2^>^&1
  ) else (
    echo Shared runtime does not exist yet.
  )
  echo.
  echo [Persistent data]
  dir "%APPHOME%\data" 2^>^&1
  echo.
  echo [Persistent models]
  dir "%APPHOME%\models" 2^>^&1
) > "%OUT%" 2>&1

type "%OUT%"
echo.
echo Diagnostic report saved to:
echo   %OUT%
pause
