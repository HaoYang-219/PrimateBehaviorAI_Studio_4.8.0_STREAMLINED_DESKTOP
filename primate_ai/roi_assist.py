# coding: utf-8
"""Image-quality suggestions for choosing an ROI annotation reference frame.

No learned pose model is used. The user must confirm body/eye/limb ROI manually.
"""
from __future__ import annotations
import cv2
import numpy as np


def image_reference_quality(bgr):
    """Prefer sufficiently bright, sharp images without overexposure."""
    if bgr is None or getattr(bgr, 'size', 0) == 0:
        return -1.0
    h,w=bgr.shape[:2]
    small=cv2.resize(bgr, (min(w,640),max(1,int(h*min(w,640)/float(max(w,1))))), interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    brightness=float(np.mean(gray))
    detail=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    clipped=float(np.mean((gray<8)|(gray>247)))
    light_factor=max(0.0,1.0-abs(brightness-125.0)/125.0)
    return float(np.log1p(max(0.,detail))*light_factor*(1.0-clipped))


def suggest_reference_time(read_at, duration_s):
    duration=float(max(0.,duration_s))
    if duration<=0:return None
    target=[min(duration*.07,2.), min(duration*.2,5.),min(duration*.35,12.),
            min(duration*.52,25.),min(duration*.72,45.),min(duration*.9,75.)]
    best_t=None;best_q=-1.0
    for t in sorted(set(max(0.,min(t,duration-.03)) for t in target)):
        frame=read_at(t)
        quality=image_reference_quality(frame)
        if quality>best_q:
            best_t,best_q=t,quality
    return best_t
