"""User-owned experiment templates, stored outside the installed application.

Device numbers are intentionally never saved: Windows may renumber cameras after
reconnection. Template parameters do not control the stimulation hardware.
"""
from __future__ import annotations
import json
import os
import re
import tempfile
from pathlib import Path
from .paths import DATA_DIR

TEMPLATE_DIR = DATA_DIR / "templates"
TEMPLATE_FIELDS = (
    "protocol_code", "session_name", "operator", "resolution", "fps", "quality_schema",
    "condition", "carrier_frequency", "modulation_frequency",
    "signal_strength", "signal_unit", "waveform", "antenna_distance",
    "custom_parameters"
)


def safe_name(name):
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(name).strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")[:72]
    if not cleaned or cleaned in (".", ".."):
        raise ValueError("请填写有效的模板名称。")
    return cleaned


def list_templates():
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.json") if p.is_file())


def save_template(name, values):
    """Atomic template save. Existing templates are replaced after validation."""
    name = safe_name(name)
    if not isinstance(values, dict):
        raise ValueError("模板内容应为字典。")
    payload = {"schema": 1, "name": name,
               "settings": {k: values[k] for k in TEMPLATE_FIELDS if k in values}}
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATE_DIR / (name + '.json')
    fd, temp = tempfile.mkstemp(prefix='.template-', suffix='.tmp', dir=str(TEMPLATE_DIR))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, str(path))
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def load_template(name):
    path = TEMPLATE_DIR / (safe_name(name) + '.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not isinstance(data.get('settings'), dict):
        raise ValueError('模板结构不兼容。')
    return {k: data['settings'][k] for k in TEMPLATE_FIELDS if k in data['settings']}
