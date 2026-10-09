"""Non-hardware smoke tests for the frozen Windows desktop distribution."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path


def run_self_test():
    import cv2
    import numpy
    import pandas
    import sklearn
    import matplotlib
    import openpyxl
    from PySide6.QtCore import QLibraryInfo
    from .session_templates import safe_name
    from .behavior_review import ReviewStore, export_workbook
    from .capture.multicam import SharedClock
    assert safe_name('常用 40Hz') == '常用_40Hz'
    assert SharedClock().t0 is None
    with tempfile.TemporaryDirectory() as d:
        temp = Path(d) / 'check.json'
        temp.write_text(json.dumps({'value':1}), encoding='utf-8')
        assert json.loads(temp.read_text(encoding='utf-8'))['value'] == 1
        store=ReviewStore(d)
        store.add('smoke',0.5,0.6,review='confirmed')
        assert len(ReviewStore(d).events)==1
        assert export_workbook(d).exists()
    info = {
        'result': 'PASS', 'opencv': cv2.__version__, 'numpy': numpy.__version__,
        'pandas': pandas.__version__, 'sklearn': sklearn.__version__,
        'openpyxl':openpyxl.__version__,'matplotlib': matplotlib.__version__, 'qt_prefix': bool(QLibraryInfo.path(QLibraryInfo.PrefixPath)),
    }
    import sys
    if sys.stdout is not None:
        print(json.dumps(info, ensure_ascii=False))
    return info
