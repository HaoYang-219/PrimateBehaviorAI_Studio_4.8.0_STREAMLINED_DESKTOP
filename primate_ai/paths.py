from __future__ import annotations
import os
import shutil
from pathlib import Path

PRODUCT_DIR = 'PrimateBehaviorAI'


def _local_appdata():
    value = os.environ.get('LOCALAPPDATA', '').strip()
    if value:
        return Path(value)
    if os.name == 'nt':
        return Path.home() / 'AppData' / 'Local'
    value = os.environ.get('XDG_DATA_HOME', '').strip()
    if value:
        return Path(value)
    return Path.home() / '.local' / 'share'


APP_HOME = _local_appdata() / PRODUCT_DIR
RUNTIME_DIR = APP_HOME / 'runtime_py38'
DATA_DIR = APP_HOME / 'data'
MODELS_DIR = APP_HOME / 'models'
PROJECTS_DIR = DATA_DIR / 'projects'
CAPTURES_DIR = DATA_DIR / 'captures'
DB_PATH = DATA_DIR / 'studio.db'
LEGACY_HOME = Path.home() / PRODUCT_DIR


def ensure_app_dirs():
    APP_HOME.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def migrate_legacy_database():
    """Copy the old per-user database once; result paths inside it remain unchanged."""
    ensure_app_dirs()
    legacy_db = LEGACY_HOME / 'studio.db'
    if not DB_PATH.exists() and legacy_db.exists():
        try:
            shutil.copy2(str(legacy_db), str(DB_PATH))
            return True
        except Exception:
            return False
    return False
