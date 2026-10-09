@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call BUILD_DESKTOP.bat
if errorlevel 1 exit /b 1
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  where ISCC.exe >nul 2>nul
  if errorlevel 1 (
    echo Inno Setup 6 not found. Install it on BUILD machine only.
    exit /b 1
  )
  set "ISCC=ISCC.exe"
)
"%ISCC%" installer\PrimateBehaviorAI.iss
if errorlevel 1 exit /b 1
echo End users install installer_output\PrimateBehaviorAI_Setup_4.8.0_x64.exe
exit /b 0
