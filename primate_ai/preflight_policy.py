"""Conservative, deterministic combined-camera preflight policy (no Qt)."""
from __future__ import annotations


def preflight_candidates(width, height, fps, automatic=True):
    """Fast automatic path: verify preferred setting, at most two safe fallbacks.

    Do not treat supported per-camera modes as proof of concurrent USB bandwidth.
    Fixed/manual mode never silently falls back to lower quality.
    """
    preferred = (int(width), int(height), int(fps))
    if not automatic:
        return [preferred]
    options = [preferred, (1280, 720, 30), (640, 480, 30)]
    unique = []
    for mode in options:
        if mode not in unique:
            unique.append(mode)
    return unique
