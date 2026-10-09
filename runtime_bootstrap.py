from __future__ import print_function
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

PRODUCT = 'PrimateBehaviorAI'
RUNTIME_NAME = 'runtime_py38'


def app_home():
    local = os.environ.get('LOCALAPPDATA', '').strip()
    if local:
        return Path(local) / PRODUCT
    if os.name == 'nt':
        return Path.home() / 'AppData' / 'Local' / PRODUCT
    xdg = os.environ.get('XDG_DATA_HOME', '').strip()
    return (Path(xdg) if xdg else Path.home() / '.local' / 'share') / PRODUCT


def runtime_dir():
    override = os.environ.get('PRIMATE_RUNTIME', '').strip()
    return Path(override) if override else app_home() / RUNTIME_NAME


def runtime_python(rt):
    return rt / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def marker_path(rt):
    return rt / '.primate_runtime.json'


def load_marker(rt):
    p = marker_path(rt)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}



def parse_pinned_requirements(req):
    out = {}
    for raw in Path(req).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '==' not in line:
            continue
        name, ver = line.split('==', 1)
        out[name.strip()] = ver.strip()
    return out


def requirements_satisfied(rt, req):
    """Return True when the existing shared runtime already has all pinned versions.

    This prevents a code-only release (or a changed comment in requirements.txt)
    from triggering a needless reinstall.
    """
    pins = parse_pinned_requirements(req)
    if not pins:
        return False
    py = runtime_python(rt)
    code = (
        "import sys\n"
        "import importlib.metadata as m\n"
        "pins=%r\n"
        "ok=True\n"
        "for k,want in pins.items():\n"
        "    try:\n"
        "        v=m.version(k)\n"
        "    except Exception:\n"
        "        ok=False\n"
        "        continue\n"
        "    if v != want:\n"
        "        ok=False\n"
        "sys.exit(0 if ok else 1)\n"
    ) % pins
    try:
        return subprocess.call([str(py), '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False

def save_marker(rt, req_hash):
    p = marker_path(rt)
    p.write_text(json.dumps({
        'runtime_schema': 1,
        'python': '3.8.x x64',
        'requirements_sha256': req_hash,
    }, indent=2), encoding='utf-8')


def require_base_python38_x64():
    if sys.version_info[:2] != (3, 8) or struct.calcsize('P') * 8 != 64:
        raise RuntimeError('首次初始化需要 64 位 Python 3.8.x。当前为 Python %s (%s-bit)。' % (
            sys.version.split()[0], struct.calcsize('P') * 8))


def run(cmd):
    print('> ' + ' '.join(str(x) for x in cmd))
    subprocess.check_call([str(x) for x in cmd])


def create_runtime(rt):
    require_base_python38_x64()
    rt.parent.mkdir(parents=True, exist_ok=True)
    print('[1/3] 创建共享运行环境：%s' % rt)
    run([sys.executable, '-m', 'venv', str(rt)])


def install_requirements(rt, req):
    py = runtime_python(rt)
    if not py.exists():
        raise RuntimeError('共享 Runtime 中没有找到 Python：%s' % py)
    print('[2/3] 安装/同步固定依赖（不会升级 pip）...')
    run([
        str(py), '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
        '--prefer-binary', '--index-url', 'https://pypi.org/simple', '-r', str(req)
    ])


def self_test(rt):
    py = runtime_python(rt)
    print('[3/3] Runtime 自检...')
    code = (
        "import sys,struct,PySide6,cv2,numpy,pandas,sklearn,matplotlib;"
        "assert sys.version_info[:2]==(3,8);"
        "assert struct.calcsize('P')*8==64;"
        "print('Runtime OK:',sys.version.split()[0],PySide6.__version__,cv2.__version__)"
    )
    run([str(py), '-c', code])


def ensure(req):
    req = Path(req).resolve()
    if not req.exists():
        raise RuntimeError('requirements.txt 不存在：%s' % req)
    rt = runtime_dir()
    py = runtime_python(rt)
    req_hash = sha256_file(req)
    marker = load_marker(rt)

    if not py.exists():
        create_runtime(rt)
        install_requirements(rt, req)
        self_test(rt)
        save_marker(rt, req_hash)
        print('共享 Runtime 初始化完成。')
        return 0

    if marker.get('requirements_sha256') != req_hash:
        if requirements_satisfied(rt, req):
            print('依赖指纹变化，但现有 Runtime 的固定依赖版本完全匹配；跳过重新安装。')
            self_test(rt)
            save_marker(rt, req_hash)
            return 0
        print('检测到真实依赖变化，只更新一次共享 Runtime。')
        install_requirements(rt, req)
        self_test(rt)
        save_marker(rt, req_hash)
        print('共享 Runtime 已同步。')
        return 0

    print('共享 Runtime 已存在且依赖未变化，跳过安装。')
    return 0


def status(req):
    req = Path(req).resolve()
    rt = runtime_dir()
    py = runtime_python(rt)
    print('App home :', app_home())
    print('Runtime  :', rt)
    print('Python   :', py)
    print('Exists   :', py.exists())
    if req.exists():
        print('Req hash :', sha256_file(req))
    print('Marker   :', load_marker(rt))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensure', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--requirements', default='requirements.txt')
    args = ap.parse_args()
    try:
        if args.status:
            return status(args.requirements)
        return ensure(args.requirements)
    except subprocess.CalledProcessError as e:
        print('ERROR: 命令执行失败，退出码 %s。' % e.returncode)
        return int(e.returncode or 1)
    except Exception as e:
        print('ERROR:', e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
