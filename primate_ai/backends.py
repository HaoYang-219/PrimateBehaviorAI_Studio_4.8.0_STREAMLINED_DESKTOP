from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BackendStatus:
    name: str
    available: bool
    detail: str


def probe_backends(model_dir):
    """Report only backends that are truly wired into the 4.0 analysis path."""
    return [
        BackendStatus('动态多边形高精度跟踪', True,
                      '当前已启用：金字塔光流 + 前后向一致性 + RANSAC + 漂移抑制；左右眼随头部运动'),
        BackendStatus('SAM2 视频分割后端', False,
                      '接口预留，当前主分析链路尚未接入；不会显示为已启用或偷偷替代当前跟踪器'),
    ]
