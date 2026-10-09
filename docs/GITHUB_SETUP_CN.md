# 获取免 Python 安装的 Windows Setup.exe（图省事版）

## 路线 A：推荐使用 GitHub Desktop（无需命令行）

1. 将 `PrimateBehaviorAI_Studio_4.8.0_STREAMLINED_DESKTOP.zip` **解压到一个本地文件夹**，不要只把 ZIP 文件上传 GitHub；确保文件夹里有 `main.py`、`requirements.txt`、`installer`、`.github/workflows/windows-installer.yml`。
2. 在 GitHub 网站创建一个 **Private（私有）** 仓库，比如 `PrimateBehaviorAI-Desktop`，不上传任何真实猴子视频和个人信息。
3. 安装并登录 [GitHub Desktop](https://desktop.github.com/)；选择 **File → Add local repository**，指定上述解压文件夹；如果提示不是 Git 仓库，选择 **Create a repository here**，然后提交所有代码（Commit）。
4. 点击 **Publish repository**，务必确认 **Keep this code private**。如果 GitHub 网站先创建了同名仓库，请按 GitHub Desktop 的提示选择发布/同步，避免将两个不相关仓库硬合并。
5. 浏览器打开该仓库 → **Actions**。如果要求启用工作流，先启用；选择左侧 **Windows desktop installer** → 右侧 **Run workflow** → 选 `main` → 再点击绿色 **Run workflow**。
6. 等待流程变为绿色成功（会下载依赖并编译，可能需要数分钟到十几分钟或更久）。进入此次运行页面 → 页面下方 **Artifacts** → 下载 `PrimateBehaviorAI-4.8.0-Windows-Installer`。
7. 解压下载的 Artifacts ZIP，得到 **`PrimateBehaviorAI_Setup_4.8.0_x64.exe`**。拷贝到没有安装 Python 的 Windows 10/11 x64 电脑，双击安装。可创建桌面快捷方式。

如果 GitHub Actions 构建失败，打开失败步骤右边的日志，将最下方红色报错发给开发者，不要直接把失败的产物当作可安装软件。

> **注意**：这是在你创建和提交代码后由 GitHub 的 Windows 云构建机生成安装包的方案，并不需要你本机有 Python。安装包会包含第三方组件，因此可能明显大于源码 ZIP。Windows Defender/SmartScreen 对尚未代码签名的自建安装包可能提示未知发布者，商业正式发行时应使用正规代码签名与安全验证。

## 路线 B：如果你本机有 Windows 构建环境

1. 安装 Python **3.11 x64**、Inno Setup 6；确保系统联网。
2. 在解压的工程根目录双击 `BUILD_INSTALLER.bat`，让脚本自动创建隔离构建环境、安装打包依赖、执行自检和构建。
3. 构建成功后，在 `installer_output` 找到 `PrimateBehaviorAI_Setup_4.8.0_x64.exe`。

## 安装前的必要验收

- 在没有 Python 的干净 Windows 上安装、启动、卸载，安装不会要求终端用户下载 Python 包。
- 用真实 1 / 2 / 3 路摄像头检查：设备编号、自动画质、联合快速预检、实时画面比例、长时间采集、正常停止与放弃。
- 复核 Session 时间轴、ROI 标定、单视频/前后/多视角分析、复核标注及 Excel 导出。
- 检验 125% / 150% Windows 缩放与 1366×768 小屏幕；目前未经过这些 Windows UI 实机测试，不应直接保证兼容。
- 发布前核查 Qt/PySide6、OpenCV、视频编解码、模型和数据许可证，以及高校职务成果权属。
