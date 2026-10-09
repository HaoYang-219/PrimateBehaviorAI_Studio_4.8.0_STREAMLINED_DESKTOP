# Windows 安装包构建与发布

推荐完整图文步骤（文字版）：`docs/GITHUB_SETUP_CN.md`。

自动构建入口：`.github/workflows/windows-installer.yml`，使用 GitHub Actions 的 Windows Runner、Python 3.11 x64、PyInstaller onedir 及 Inno Setup 6。

成果：Artifacts 中下载 `PrimateBehaviorAI-4.8.0-Windows-Installer`，解压获得 `PrimateBehaviorAI_Setup_4.8.0_x64.exe`。

本地方式：Windows 电脑安装 Python 3.11 x64、Inno Setup 6 后，执行 `BUILD_INSTALLER.bat`。依赖仅安装在私有 `.buildvenv` 中，不会要求实验电脑安装 Python。终端实机必须验证驱动、媒体编解码、升级、卸载和录像质量；源码检查不替代安装验证。
