#define MyAppName "PrimateBehaviorAI Studio"
#define MyAppVersion "4.8.0"
#define MyAppExeName "PrimateBehaviorAI.exe"
[Setup]
AppId={{DBE92F36-8C5A-4F87-81F5-4D729B09E79E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=PrimateBehaviorAI Research
DefaultDirName={localappdata}\Programs\PrimateBehaviorAI
DefaultGroupName=PrimateBehaviorAI
PrivilegesRequired=lowest
OutputDir=..\installer_output
OutputBaseFilename=PrimateBehaviorAI_Setup_4.8.0_x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=..\assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
CloseApplications=yes
RestartApplications=no
[Files]
Source: "..\dist\PrimateBehaviorAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\PrimateBehaviorAI"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\PrimateBehaviorAI"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; Flags: unchecked
[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 PrimateBehaviorAI"; Flags: nowait postinstall skipifsilent
