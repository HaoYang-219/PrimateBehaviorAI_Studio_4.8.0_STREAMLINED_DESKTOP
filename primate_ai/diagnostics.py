"""Privacy-conscious diagnostics export: no experiment videos or animal metadata."""
from __future__ import annotations
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def build_diagnostic_archive(destination):
    import cv2
    import numpy
    import pandas
    import sklearn
    import matplotlib
    import PySide6
    info = {
        'product': 'PrimateBehaviorAI Studio', 'version': '4.8.0',
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'platform': platform.platform(), 'architecture': platform.machine(),
        'python': sys.version.split()[0], 'frozen': bool(getattr(sys, 'frozen', False)),
        'opencv': cv2.__version__, 'numpy': numpy.__version__,
        'pandas': pandas.__version__, 'scikit_learn': sklearn.__version__,
        'matplotlib': matplotlib.__version__, 'pyside6': PySide6.__version__,
        'note': 'No videos, experiment IDs, session paths or ROI data included.'
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(destination), 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('system_info.json', json.dumps(info, ensure_ascii=False, indent=2))
        z.writestr('README.txt', 'This archive has system environment information only. No experimental video or personal records.\n')
    return destination
