from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal


CAMERA_LABELS = {
    'front': '前视角',
    'left': '左侧视角',
    'right': '右侧视角',
    'front_left': '前左斜视',
    'front_right': '前右斜视',
    'custom': '自定义',
}




def _backend_candidates():
    if os.name == 'nt':
        return [
            ('DirectShow', cv2.CAP_DSHOW),
            ('MediaFoundation', cv2.CAP_MSMF),
            ('Auto', cv2.CAP_ANY),
        ]
    return [('Auto', cv2.CAP_ANY)]


def _configure_cap(cap, width, height, fps, fourcc='MJPG'):
    """Apply a multi-camera-friendly capture mode. MJPEG is preferred to reduce USB bandwidth."""
    try:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _decode_fourcc(value):
    try:
        value = int(value)
        text = ''.join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
        text = ''.join(ch if 32 <= ord(ch) < 127 else '?' for ch in text)
        return text.strip('\x00') or '未知'
    except Exception:
        return '未知'


def _open_configured_camera(device_index, width, height, fps, fourcc='MJPG'):
    """Open one camera using Windows-friendly backend fallbacks."""
    for backend_name, backend in _backend_candidates():
        cap = cv2.VideoCapture(int(device_index), backend)
        if not cap.isOpened():
            cap.release()
            continue
        _configure_cap(cap, width, height, fps, fourcc=fourcc)
        return cap, backend_name
    return None, None


def _read_until_frame(cap, attempts=30, sleep_s=0.05):
    for _ in range(max(1, int(attempts))):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, 'size', 0) > 0:
            return True, frame
        time.sleep(float(sleep_s))
    return False, None




def _open_ready_camera(device_index, width, height, fps, fourcc='MJPG', attempts=36, retries_per_backend=2, require_mode=True):
    """Open a camera and verify that it can actually deliver frames.

    DirectShow can report isOpened()==True while the device is not yet stream-ready.
    This helper retries each backend and only returns a handle after a real frame
    has been received. It also allows fallback to Media Foundation/Auto when
    DirectShow opens but cannot deliver frames.
    """
    last_backend = None
    for backend_name, backend in _backend_candidates():
        last_backend = backend_name
        for retry in range(max(1, int(retries_per_backend))):
            cap = cv2.VideoCapture(int(device_index), backend)
            if not cap.isOpened():
                cap.release()
                time.sleep(0.08)
                continue
            _configure_cap(cap, width, height, fps, fourcc=fourcc)
            ok, frame = _read_until_frame(cap, attempts=attempts, sleep_s=0.04)
            if ok and frame is not None and getattr(frame, 'size', 0) > 0:
                fh, fw = frame.shape[:2]
                if (not require_mode) or (abs(int(fw) - int(width)) <= 8 and abs(int(fh) - int(height)) <= 8):
                    return cap, backend_name, frame
            cap.release()
            time.sleep(0.20 + 0.15 * retry)
    return None, last_backend, None

def _stream_metrics(cap, first_frame, target_fps, duration_s):
    start = time.perf_counter()
    frames = 0
    failures = 0
    sample = first_frame.copy() if first_frame is not None else None
    times = []
    while time.perf_counter() - start < float(duration_s):
        got, frame = cap.read()
        now = time.perf_counter()
        if got and frame is not None and getattr(frame, 'size', 0) > 0:
            frames += 1
            times.append(now)
            sample = frame.copy()
        else:
            failures += 1
            time.sleep(0.003)
    elapsed = max(1e-6, time.perf_counter() - start)
    observed = frames / elapsed
    intervals = np.diff(np.asarray(times, dtype=np.float64)) if len(times) >= 2 else np.asarray([], dtype=np.float64)
    median_ms = float(np.median(intervals) * 1000.0) if intervals.size else 0.0
    p95_ms = float(np.percentile(intervals, 95) * 1000.0) if intervals.size else 0.0
    mean_ms = float(np.mean(intervals) * 1000.0) if intervals.size else 0.0
    jitter_ms = float(np.std(intervals) * 1000.0) if intervals.size else 0.0
    total_attempts = max(1, frames + failures)
    failure_ratio = failures / float(total_attempts)
    ratio = observed / max(1e-6, float(target_fps))

    # Three-grade judgement: do not call a mildly slow camera a hard failure.
    if frames >= 12 and ratio >= 0.85 and failure_ratio <= 0.03 and (p95_ms <= (1000.0 / max(1.0, target_fps)) * 1.8 or p95_ms == 0.0):
        grade = 'stable'
        grade_cn = '稳定'
    elif frames >= 8 and ratio >= 0.65 and failure_ratio <= 0.12 and (p95_ms <= (1000.0 / max(1.0, target_fps)) * 3.0 or p95_ms == 0.0):
        grade = 'degraded'
        grade_cn = '可用但降级'
    else:
        grade = 'poor'
        grade_cn = '不建议采集'

    h, w = (sample.shape[:2] if sample is not None else (0, 0))
    return {
        'ok': grade != 'poor',
        'grade': grade,
        'grade_cn': grade_cn,
        'width': int(w), 'height': int(h),
        'requested_fps': float(target_fps),
        'observed_fps': float(observed),
        'fps_ratio': float(ratio),
        'frames': int(frames),
        'read_failures': int(failures),
        'failure_ratio': float(failure_ratio),
        'mean_interval_ms': mean_ms,
        'median_interval_ms': median_ms,
        'p95_interval_ms': p95_ms,
        'jitter_ms': jitter_ms,
        'sample_frame': sample,
    }


def test_single_camera_config(spec, duration_s=2.5):
    """Quantified single-camera preflight at the exact requested mode."""
    idx = int(spec['device_index'])
    width = int(spec['width'])
    height = int(spec['height'])
    fps = float(spec['fps'])
    requested_fourcc = str(spec.get('fourcc', 'MJPG'))
    cap, backend, first = _open_ready_camera(
        idx, width, height, fps, fourcc=requested_fourcc, attempts=40, retries_per_backend=2, require_mode=True
    )
    if cap is None or first is None:
        return {'ok': False, 'grade': 'poor', 'grade_cn': '不建议采集', 'device_index': idx,
                'backend': backend or '', 'requested_fps': float(fps), 'width': 0, 'height': 0,
                'reason': '无法以请求模式 %dx%d @ %.1f FPS 输出有效画面；驱动可能退回了更低分辨率。' % (width, height, fps)}
    try:
        metrics = _stream_metrics(cap, first, fps, duration_s)
        metrics.update({
            'device_index': idx,
            'backend': backend or '',
            'reported_fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            'fourcc': _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            'requested_fourcc': requested_fourcc,
            'requested_width': int(width), 'requested_height': int(height),
        })
        if metrics['grade'] == 'stable':
            metrics['reason'] = ''
        elif metrics['grade'] == 'degraded':
            metrics['reason'] = '可以采集，但实际帧率或帧间隔稳定性低于目标；建议优先使用推荐模式。'
        else:
            metrics['reason'] = '能出画面，但持续读取性能不足，不建议用于正式实验。'
        return metrics
    finally:
        cap.release()


def recommend_camera_mode(spec):
    """Try conservative modes and return the highest-quality stable/usable recommendation.

    This never changes the experiment settings automatically; it only reports a recommendation.
    """
    idx = int(spec['device_index'])
    target_fps = int(spec['fps'])
    requested = (int(spec['width']), int(spec['height']), target_fps)
    candidates = [requested]
    # Keep temporal resolution when possible: 720p at the same FPS before lowering FPS.
    for mode in [(1280, 720, target_fps), (1920, 1080, 30), (1280, 720, 30), (640, 480, 30), (640, 480, 15)]:
        if mode not in candidates:
            candidates.append(mode)
    best_degraded = None
    tried = []
    for w, h, fps in candidates:
        test_spec = dict(spec)
        test_spec.update({'width': w, 'height': h, 'fps': fps, 'fourcc': 'MJPG'})
        result = test_single_camera_config(test_spec, duration_s=1.6)
        summary = {'width': w, 'height': h, 'fps': fps, 'grade': result.get('grade'),
                   'observed_fps': result.get('observed_fps', 0.0), 'fourcc': result.get('fourcc', '')}
        tried.append(summary)
        if result.get('grade') == 'stable':
            return {'found': True, 'device_index': idx, 'recommended': summary, 'tried': tried}
        if result.get('grade') == 'degraded' and best_degraded is None:
            best_degraded = summary
    if best_degraded is not None:
        return {'found': True, 'device_index': idx, 'recommended': best_degraded, 'tried': tried}
    return {'found': False, 'device_index': idx, 'recommended': None, 'tried': tried}


def test_camera_group(camera_specs, duration_s=3.0, startup_attempts=44, retries_per_backend=2):
    """Open all selected cameras simultaneously and quantify concurrent streaming."""
    specs = list(camera_specs)
    caps = []
    results = {}
    if not specs:
        return {'ok': False, 'results': {}, 'reason': '没有启用摄像头'}
    try:
        first_frames = {}
        for spec in specs:
            idx = int(spec['device_index'])
            cap, backend, first = _open_ready_camera(
                idx, spec['width'], spec['height'], spec['fps'], fourcc=spec.get('fourcc', 'MJPG'),
                attempts=startup_attempts, retries_per_backend=retries_per_backend, require_mode=True
            )
            if cap is None or first is None:
                results[str(idx)] = {'ok': False, 'grade': 'poor', 'grade_cn': '不建议采集',
                                     'device_index': idx, 'backend': backend or '',
                                     'reason': '联合开启时无法以请求模式 %dx%d @ %s FPS 输出画面' % (
                                         int(spec['width']), int(spec['height']), str(spec['fps']))}
                return {'ok': False, 'results': results, 'reason': '联合开启失败或有摄像头无法保持请求分辨率'}
            caps.append((spec, cap, backend))
            first_frames[idx] = first.copy()

        lock = threading.Lock()
        warm = {}
        def warm_one(spec, cap, backend):
            idx = int(spec['device_index'])
            first = first_frames.get(idx)
            ok, frame = _read_until_frame(cap, attempts=20, sleep_s=0.03)
            if (not ok or frame is None) and first is not None:
                ok, frame = True, first
            with lock:
                warm[idx] = (ok, frame.copy() if ok and frame is not None else None, backend)
        threads = []
        for spec, cap, backend in caps:
            t = threading.Thread(target=warm_one, args=(spec, cap, backend), daemon=True)
            threads.append(t); t.start()
        for t in threads:
            t.join(timeout=6.0)

        if len(warm) != len(caps) or any(not warm.get(int(spec['device_index']), (False, None, ''))[0] for spec, _, _ in caps):
            for spec, cap, backend in caps:
                idx = int(spec['device_index'])
                ok, frame, bname = warm.get(idx, (False, None, backend))
                results[str(idx)] = {'ok': bool(ok), 'grade': 'stable' if ok else 'poor',
                                     'grade_cn': '稳定' if ok else '不建议采集', 'device_index': idx,
                                     'backend': bname or '', 'reason': '' if ok else '单独可用，但联合开启后无法稳定得到首帧',
                                     'sample_frame': frame}
            return {'ok': False, 'results': results,
                    'reason': '摄像头单独可能可用，但联合开启后有通道无法出帧；优先检查USB Hub/控制器带宽或供电。'}

        monitor = {}
        def monitor_one(spec, cap, backend, first_frame):
            idx = int(spec['device_index'])
            metrics = _stream_metrics(cap, first_frame, float(spec['fps']), duration_s)
            metrics.update({'device_index': idx, 'backend': backend or '',
                            'reported_fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                            'fourcc': _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
                            'requested_fourcc': str(spec.get('fourcc', 'MJPG')),
                            'requested_width': int(spec['width']), 'requested_height': int(spec['height'])})
            if metrics['grade'] == 'stable':
                metrics['reason'] = ''
            elif metrics['grade'] == 'degraded':
                metrics['reason'] = '联合采集可用但性能降级。'
            else:
                metrics['reason'] = '联合采集性能不足。'
            with lock:
                monitor[idx] = metrics
        threads = []
        for spec, cap, backend in caps:
            idx = int(spec['device_index'])
            first = warm.get(idx, (False, None, backend))[1]
            t = threading.Thread(target=monitor_one, args=(spec, cap, backend, first), daemon=True)
            threads.append(t); t.start()
        for t in threads:
            t.join(timeout=float(duration_s) + 6.0)
        for spec, _, backend in caps:
            idx = int(spec['device_index'])
            if idx not in monitor:
                monitor[idx] = {'ok': False, 'grade': 'poor', 'grade_cn': '不建议采集',
                                'device_index': idx, 'backend': backend or '', 'reason': '联合稳定性测试超时'}
        all_usable = all(monitor[int(spec['device_index'])].get('grade') != 'poor' for spec, _, _ in caps)
        has_degraded = any(monitor[int(spec['device_index'])].get('grade') == 'degraded' for spec, _, _ in caps)
        return {'ok': bool(all_usable), 'warning': bool(all_usable and has_degraded),
                'results': {str(k): v for k, v in monitor.items()},
                'reason': '' if all_usable else '联合采集不稳定；若逐路测试均通过，优先判断为USB带宽/供电/控制器争用。'}
    finally:
        for _, cap, _ in caps:
            try:
                cap.release()
            except Exception:
                pass


class SharedClock(object):
    def __init__(self):
        self.event = threading.Event()
        self.abort = threading.Event()
        self.lock = threading.Lock()
        self.t0 = None

    def arm(self, delay_s=0.35):
        with self.lock:
            self.t0 = time.perf_counter() + float(delay_s)
        self.event.set()

    def stop(self):
        self.abort.set()
        self.event.set()


class CameraWorker(QThread):
    frame = Signal(int, object)
    status = Signal(int, object)
    timing = Signal(int, object)
    alert = Signal(int, object)
    ready = Signal(int)
    failed = Signal(int, str)
    finished_ok = Signal(int, object)

    def __init__(
        self,
        slot_id,
        device_index,
        view_name,
        output_video,
        output_timestamps,
        width,
        height,
        fps,
        shared_clock,
        record=True,
        preview_every=2,
        parent=None,
    ):
        super().__init__(parent)
        self.slot_id = int(slot_id)
        self.device_index = int(device_index)
        self.view_name = str(view_name)
        self.output_video = Path(output_video)
        self.output_timestamps = Path(output_timestamps)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.clock = shared_clock
        self.record = bool(record)
        self.preview_every = max(1, int(preview_every))
        self._stop_local = threading.Event()
        self._cap = None
        self._writer = None

    def request_stop(self):
        self._stop_local.set()

    def _open_capture(self):
        return _open_configured_camera(self.device_index, self.width, self.height, self.fps)

    @staticmethod
    def _warmup_read(cap, attempts=30, sleep_s=0.05):
        return _read_until_frame(cap, attempts=attempts, sleep_s=sleep_s)

    def _open_writer(self, actual_w, actual_h, writer_fps):
        if not self.record:
            return None
        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(
            str(self.output_video),
            fourcc,
            float(writer_fps),
            (int(actual_w), int(actual_h)),
        )
        if not writer.isOpened():
            return None
        return writer

    @staticmethod
    def _motion_score(prev_gray, frame):
        small_w = 320
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return 0.0, None
        scale = small_w / float(w)
        small = cv2.resize(frame, (small_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is None or prev_gray.shape != gray.shape:
            return 0.0, gray
        diff = cv2.absdiff(gray, prev_gray)
        score = float(np.mean(diff))
        return score, gray

    def run(self):
        # Formal recording uses a stronger open path than simple enumeration:
        # validate real frames, retry, and fall back across Windows backends.
        cap, backend_name, warm_frame = _open_ready_camera(
            self.device_index, self.width, self.height, self.fps,
            fourcc='MJPG', attempts=42, retries_per_backend=2
        )
        self._cap = cap
        if cap is None or warm_frame is None:
            self.failed.emit(
                self.slot_id,
                '设备 %d 在正式采集启动阶段无法稳定输出画面。程序已尝试 DirectShow / MediaFoundation / Auto '
                '并进行了重复预热。若预检刚刚通过，通常是 Windows 摄像头句柄释放/重新打开竞争或 USB 总线瞬时争用。'
                '请稍候后重试，或更换USB接口/降低采集模式。' % self.device_index
            )
            return
        actual_h, actual_w = warm_frame.shape[:2]
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
        if actual_fps < 5 or actual_fps > 240:
            actual_fps = self.fps
        writer_fps = actual_fps

        writer = self._open_writer(actual_w, actual_h, writer_fps)
        self._writer = writer
        if self.record and writer is None:
            cap.release()
            self.failed.emit(self.slot_id, '摄像头已打开，但无法创建录像文件：%s' % self.output_video)
            return

        self.ready.emit(self.slot_id)
        self.clock.event.wait(timeout=15.0)
        if self.clock.abort.is_set() or self.clock.t0 is None:
            if writer is not None:
                writer.release()
            cap.release()
            self.finished_ok.emit(self.slot_id, {
                'frames': 0, 'written_frames': 0, 'frame_deficit': 0,
                'gap_events': 0, 'gap_equivalent_frames': 0,
                'read_failures_total': 0, 'video': str(self.output_video),
                'timestamps': str(self.output_timestamps), 'aborted_before_start': True,
                'actual_width': actual_w, 'actual_height': actual_h,
                'actual_fps_reported': actual_fps,
                'capture_backend': backend_name,
            })
            return

        while time.perf_counter() < self.clock.t0:
            if self.clock.abort.is_set() or self._stop_local.is_set():
                break
            time.sleep(0.001)

        self.output_timestamps.parent.mkdir(parents=True, exist_ok=True)
        ts_file = open(str(self.output_timestamps), 'w', newline='', encoding='utf-8-sig')
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow([
            'frame_index', 'session_time_s', 'monotonic_s', 'wall_time_utc',
            'interframe_ms', 'motion_score', 'gap_events_total',
            'gap_equivalent_frames_total', 'read_failures_total'
        ])

        frame_idx = 0
        gap_events = 0
        gap_equivalent_total = 0
        prev_gray = None
        motion_hist = deque(maxlen=max(30, int(self.fps * 4)))
        interval_hist = deque(maxlen=max(90, int(self.fps * 6)))
        last_alert_t = -99.0
        total_read_failures = 0
        consecutive_read_failures = 0
        last_status_t = -1.0
        last_frame_mono = None
        last_ts_flush = time.perf_counter()
        first_frame_session_t = None
        last_session_t = 0.0
        nominal_period_s = 1.0 / max(1.0, float(self.fps))

        try:
            while not self.clock.abort.is_set() and not self._stop_local.is_set():
                ok, frame = cap.read()
                now_mono = time.perf_counter()
                if not ok or frame is None:
                    consecutive_read_failures += 1
                    total_read_failures += 1
                    if consecutive_read_failures > 30:
                        self.failed.emit(self.slot_id, '摄像头连续读取失败，录制已停止。')
                        break
                    time.sleep(0.003)
                    continue
                consecutive_read_failures = 0
                session_t = max(0.0, now_mono - float(self.clock.t0))
                last_session_t = session_t
                if first_frame_session_t is None:
                    first_frame_session_t = session_t

                interframe_ms = 0.0
                if last_frame_mono is not None:
                    interval_s = max(0.0, now_mono - last_frame_mono)
                    interframe_ms = interval_s * 1000.0
                    interval_hist.append(interframe_ms)
                    # A timing gap is only counted when the interval is clearly larger
                    # than one nominal frame period. This is a host-arrival gap estimate,
                    # not a hardware camera frame counter.
                    if interval_s > nominal_period_s * 1.5:
                        equiv = max(0, int(round(interval_s / nominal_period_s)) - 1)
                        if equiv > 0:
                            gap_events += 1
                            gap_equivalent_total += equiv
                last_frame_mono = now_mono

                if writer is not None:
                    writer.write(frame)

                motion, prev_gray = self._motion_score(prev_gray, frame)
                motion_hist.append(motion)

                wall = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
                ts_writer.writerow([
                    frame_idx,
                    '%.6f' % session_t,
                    '%.6f' % now_mono,
                    wall,
                    '%.3f' % interframe_ms,
                    '%.4f' % motion,
                    gap_events,
                    gap_equivalent_total,
                    total_read_failures,
                ])

                # Flush metadata periodically, so an interrupted Session retains
                # as much of its acquisition clock as practical.
                if now_mono - last_ts_flush >= 5.0:
                    ts_file.flush()
                    last_ts_flush = now_mono

                # Emit a lightweight per-frame timestamp for multi-camera software
                # alignment QC. No image is copied for this signal.
                self.timing.emit(self.slot_id, {
                    'frame_index': int(frame_idx),
                    'session_time_s': float(session_t),
                    'monotonic_s': float(now_mono),
                })

                # Robust, intentionally conservative real-time motion alert.
                # It is only a review marker, never a medical label.
                if len(motion_hist) >= max(20, int(self.fps * 1.5)):
                    arr = np.asarray(motion_hist, dtype=np.float32)
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med))) + 1e-6
                    robust_z = (motion - med) / (1.4826 * mad)
                    if robust_z >= 7.0 and motion >= med + 3.0 and session_t - last_alert_t >= 0.65:
                        last_alert_t = session_t
                        self.alert.emit(self.slot_id, {
                            'time_s': session_t,
                            'type': '快速运动候选',
                            'motion_score': motion,
                            'robust_z': robust_z,
                        })

                if frame_idx % self.preview_every == 0:
                    self.frame.emit(self.slot_id, frame.copy())

                if session_t - last_status_t >= 0.5:
                    written_frames = frame_idx + 1
                    effective_elapsed = max(nominal_period_s, session_t - (first_frame_session_t or 0.0) + nominal_period_s)
                    capture_fps = written_frames / effective_elapsed
                    theoretical_frames = max(1, int(round(effective_elapsed * self.fps)))
                    frame_deficit = max(0, theoretical_frames - written_frames)
                    if interval_hist:
                        interval_arr = np.asarray(interval_hist, dtype=np.float64)
                        p95_interval_ms = float(np.percentile(interval_arr, 95))
                        jitter_ms = float(np.std(interval_arr))
                    else:
                        p95_interval_ms = 0.0
                        jitter_ms = 0.0
                    state = '静止'
                    if len(motion_hist) >= 10:
                        med = float(np.median(np.asarray(motion_hist, dtype=np.float32)))
                        if motion > max(5.5, med * 2.0):
                            state = '快速运动候选'
                        elif motion > max(2.2, med * 1.2):
                            state = '活动'
                    self.status.emit(self.slot_id, {
                        'fps': capture_fps,
                        'device_fps': actual_fps,
                        'frames': written_frames,
                        'written_frames': written_frames,
                        'theoretical_frames': theoretical_frames,
                        'frame_deficit': frame_deficit,
                        'gap_events': gap_events,
                        'gap_equivalent_frames': gap_equivalent_total,
                        'read_failures_total': total_read_failures,
                        'p95_interval_ms': p95_interval_ms,
                        'jitter_ms': jitter_ms,
                        'session_time_s': session_t,
                        'motion': motion,
                        'state': state,
                        'width': actual_w,
                        'height': actual_h,
                    })
                    last_status_t = session_t

                frame_idx += 1
        finally:
            try:
                ts_file.flush()
                ts_file.close()
            except Exception:
                pass
            if writer is not None:
                writer.release()
            cap.release()

        written_frames = int(frame_idx)
        if first_frame_session_t is None:
            effective_elapsed = 0.0
        else:
            effective_elapsed = max(0.0, last_session_t - first_frame_session_t + nominal_period_s)
        theoretical_frames = int(round(effective_elapsed * self.fps)) if effective_elapsed > 0 else 0
        frame_deficit = max(0, theoretical_frames - written_frames)
        if interval_hist:
            interval_arr = np.asarray(interval_hist, dtype=np.float64)
            p95_interval_ms = float(np.percentile(interval_arr, 95))
            jitter_ms = float(np.std(interval_arr))
            max_interval_ms = float(np.max(interval_arr))
        else:
            p95_interval_ms = 0.0
            jitter_ms = 0.0
            max_interval_ms = 0.0
        effective_fps = written_frames / max(nominal_period_s, effective_elapsed) if written_frames else 0.0
        self.finished_ok.emit(self.slot_id, {
            'frames': written_frames,
            'written_frames': written_frames,
            'theoretical_frames': theoretical_frames,
            'frame_deficit': frame_deficit,
            'gap_events': int(gap_events),
            'gap_equivalent_frames': int(gap_equivalent_total),
            'read_failures_total': int(total_read_failures),
            'effective_fps': float(effective_fps),
            'p95_interval_ms': float(p95_interval_ms),
            'jitter_ms': float(jitter_ms),
            'max_interval_ms': float(max_interval_ms),
            'video': str(self.output_video),
            'timestamps': str(self.output_timestamps),
            'actual_width': actual_w,
            'actual_height': actual_h,
            'actual_fps_reported': actual_fps,
            'capture_backend': backend_name,
            'quality_note': 'frame_deficit/gap_equivalent_frames are host-side timing quality indicators; exact sensor-level dropped frames require camera/driver frame counters.',
        })


class LivePreviewWorker(QThread):
    frame = Signal(int, object)
    status = Signal(int, object)
    opened = Signal(int, object)
    failed = Signal(int, str)
    finished_preview = Signal(int)

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = dict(spec)
        self.slot_id = int(self.spec.get('slot_id', 0))
        self._stop_local = threading.Event()

    def request_stop(self):
        self._stop_local.set()

    def run(self):
        idx = int(self.spec['device_index'])
        cap, backend, first = _open_ready_camera(
            idx, int(self.spec['width']), int(self.spec['height']), float(self.spec['fps']),
            fourcc=str(self.spec.get('fourcc', 'MJPG')), attempts=36, retries_per_backend=2
        )
        if cap is None or first is None:
            self.failed.emit(self.slot_id, '设备 %d 无法进入持续实时预览。' % idx)
            self.finished_preview.emit(self.slot_id)
            return
        try:
            h, w = first.shape[:2]
            reported = float(cap.get(cv2.CAP_PROP_FPS) or self.spec['fps'])
            self.opened.emit(self.slot_id, {
                'device_index': idx, 'backend': backend or '', 'width': int(w), 'height': int(h),
                'reported_fps': reported, 'fourcc': _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
            })
            frame_count = 0
            t0 = time.perf_counter()
            last_status = t0
            current = first
            while not self._stop_local.is_set():
                if current is None:
                    ok, current = cap.read()
                    if not ok or current is None:
                        time.sleep(0.005)
                        continue
                frame_count += 1
                if frame_count % 2 == 0:
                    self.frame.emit(self.slot_id, current.copy())
                now = time.perf_counter()
                if now - last_status >= 0.5:
                    fps = frame_count / max(1e-6, now - t0)
                    self.status.emit(self.slot_id, {
                        'fps': fps, 'dropped': 0, 'state': '实时预览',
                        'width': current.shape[1], 'height': current.shape[0],
                    })
                    last_status = now
                ok, nxt = cap.read()
                current = nxt if ok and nxt is not None else None
        finally:
            cap.release()
            self.finished_preview.emit(self.slot_id)


class LivePreviewController(QObject):
    frame = Signal(int, object)
    status = Signal(int, object)
    log = Signal(str)
    failed = Signal(str)
    all_opened = Signal()
    stopped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers = []
        self._running = False
        self._next_index = 0
        self._opened_slots = set()
        self._finished_slots = set()

    @property
    def running(self):
        return self._running

    def start(self, specs):
        if self._running:
            return
        specs = [dict(x) for x in specs]
        if not specs:
            return
        self.workers = []
        self._opened_slots = set()
        self._finished_slots = set()
        self._next_index = 0
        self._running = True
        for spec in specs:
            w = LivePreviewWorker(spec)
            w.frame.connect(self.frame)
            w.status.connect(self.status)
            w.opened.connect(self._on_opened)
            w.failed.connect(self._on_failed)
            w.finished_preview.connect(self._on_finished)
            self.workers.append(w)
        self.log.emit('正在逐路打开实时预览，确认移动摄像头时画面可实时变化…')
        self._start_next()

    def _start_next(self):
        if not self._running:
            return
        if self._next_index >= len(self.workers):
            self.all_opened.emit()
            self.log.emit('所有启用摄像头实时预览已就绪。')
            return
        worker = self.workers[self._next_index]
        self._next_index += 1
        worker.start()

    def _on_opened(self, slot, info):
        self._opened_slots.add(int(slot))
        self.log.emit('摄像头 %d 实时预览已打开：%dx%d / %s。' % (
            int(slot) + 1, int(info.get('width', 0)), int(info.get('height', 0)), info.get('backend', '')
        ))
        self._start_next()

    def _on_failed(self, slot, message):
        self.failure_reason = '第%d路：%s' % (int(slot) + 1, message)
        self.log.emit('摄像头 %d 实时预览失败：%s' % (int(slot) + 1, message))
        self.failed.emit('第 %d 路实时预览失败：%s' % (int(slot) + 1, message))
        self.stop(wait=False)

    def _on_finished(self, slot):
        self._finished_slots.add(int(slot))
        if self.workers and len(self._finished_slots) >= len([w for w in self.workers if w.isRunning() or w.isFinished()]):
            # Final stopped signal is also emitted by stop(wait=True) below; this
            # branch mainly covers workers ending naturally.
            pass

    def stop(self, wait=True, timeout_ms=5000):
        if not self.workers:
            was_running = self._running
            self._running = False
            if was_running:
                self.stopped.emit()
            return
        for w in self.workers:
            w.request_stop()
        if wait:
            deadline = time.time() + timeout_ms / 1000.0
            for w in self.workers:
                if w.isRunning():
                    remain = max(0.0, deadline - time.time())
                    w.wait(int(remain * 1000))
        self._running = False
        self.workers = []
        self.stopped.emit()


class MultiCameraController(QObject):
    frame = Signal(int, object)
    status = Signal(int, object)
    alert = Signal(int, object)
    sync_quality = Signal(object)
    log = Signal(str)
    all_ready = Signal()
    failed = Signal(str)
    stopped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clock = None
        self.workers = []
        self.ready_slots = set()
        self.finished_slots = set()
        self.results = {}
        self.session_dir = None
        self.metadata_dir = None
        self.session_meta = None
        self._running = False
        self._next_worker_index = 0
        self._discard_on_stop = False
        self.failure_reason = None
        self.active_slots = []
        self.timestamp_buffers = {}
        self.sync_samples = []
        self._baseline_offsets = None
        self._offset_history = []
        self._last_sync_emit_mono = 0.0

    @property
    def running(self):
        return self._running

    def session_time_s(self):
        """Current shared Session time in seconds, based on the same monotonic clock as frame timestamps."""
        if self.clock is None or self.clock.t0 is None:
            return None
        return max(0.0, time.perf_counter() - float(self.clock.t0))

    def start(self, session_dir, camera_specs, session_meta):
        if self._running:
            raise RuntimeError('摄像头采集已经在运行。')
        if not (1 <= len(camera_specs) <= 3):
            raise ValueError('请选择 1–3 路已启用摄像头。')
        indices = [int(x['device_index']) for x in camera_specs]
        if len(set(indices)) != len(indices):
            raise ValueError('已启用摄像头不能使用重复的设备编号。')
        self.expected_count = len(camera_specs)

        self.clock = SharedClock()
        self.workers = []
        self.ready_slots = set()
        self.finished_slots = set()
        self.results = {}
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.session_dir / '_metadata'
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.session_meta = dict(session_meta)
        self.session_meta['created_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        self.session_meta['storage_format'] = 'AVI / MJPEG'
        self.session_meta['camera_specs'] = camera_specs
        self.session_meta['session_state'] = 'recording_or_initializing'
        self.session_meta['note_recovery'] = 'If interrupted, inspect recorded AVI files and _metadata/timestamps_*.csv; session.json may remain in an unfinished state.'
        (self.session_dir / 'session.json').write_text(
            json.dumps(self.session_meta, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        self._running = True
        self.active_slots = sorted(int(spec.get('slot_id', i)) for i, spec in enumerate(camera_specs))
        self.timestamp_buffers = {slot: deque(maxlen=240) for slot in self.active_slots}
        self.sync_samples = []
        self._baseline_offsets = None
        self._offset_history = []
        self._last_sync_emit_mono = 0.0

        for seq, spec in enumerate(camera_specs):
            slot = int(spec.get('slot_id', seq))
            stem = 'cam%02d_%s' % (slot + 1, spec['view_key'])
            worker = CameraWorker(
                slot_id=slot,
                device_index=spec['device_index'],
                view_name=spec['view_key'],
                output_video=self.session_dir / (stem + '.avi'),
                output_timestamps=self.metadata_dir / ('timestamps_%s.csv' % stem),
                width=spec['width'],
                height=spec['height'],
                fps=spec['fps'],
                shared_clock=self.clock,
                record=True,
                preview_every=2,
            )
            worker.frame.connect(self.frame)
            worker.status.connect(self.status)
            worker.timing.connect(self._on_timing)
            worker.alert.connect(self.alert)
            worker.ready.connect(self._on_ready)
            worker.failed.connect(self._on_failed)
            worker.finished_ok.connect(self._on_finished)
            self.workers.append(worker)

        self._next_worker_index = 0
        self._discard_on_stop = False
        self.failure_reason = None
        self.log.emit('正在逐路打开 %d 路摄像头并准备同步时钟…' % len(camera_specs))
        if self.workers:
            self.workers[0].start()
            self._next_worker_index = 1

    def _on_ready(self, slot):
        self.ready_slots.add(int(slot))
        self.log.emit('摄像头 %d 已就绪。' % (int(slot) + 1))
        expected = getattr(self, 'expected_count', len(self.workers))
        if len(self.ready_slots) < expected and self._next_worker_index < len(self.workers):
            worker = self.workers[self._next_worker_index]
            self._next_worker_index += 1
            worker.start()
            return
        if len(self.ready_slots) == expected and self.clock is not None and not self.clock.abort.is_set():
            self.clock.arm(delay_s=0.55)
            self.session_meta['sync_start_wall_utc'] = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
            self.all_ready.emit()
            self.log.emit('%d 路摄像头已同步开始采集。' % expected)

    def _on_timing(self, slot, info):
        slot = int(slot)
        if slot not in self.timestamp_buffers:
            return
        try:
            ts = float(info.get('session_time_s', 0.0))
        except Exception:
            return
        self.timestamp_buffers[slot].append(ts)
        if len(self.active_slots) <= 1:
            return
        if any(not self.timestamp_buffers.get(s) for s in self.active_slots):
            return
        now = time.perf_counter()
        if now - self._last_sync_emit_mono < 0.25:
            return
        self._last_sync_emit_mono = now

        # Use a common target time that every camera has already reached, then
        # choose the nearest received frame timestamp from each stream. This is
        # much more meaningful than comparing asynchronously emitted UI status times.
        target_t = min(self.timestamp_buffers[s][-1] for s in self.active_slots)
        nearest = {}
        for s in self.active_slots:
            buf = self.timestamp_buffers[s]
            nearest[s] = min(buf, key=lambda x: abs(x - target_t))
        vals = [nearest[s] for s in self.active_slots]
        spread_ms = (max(vals) - min(vals)) * 1000.0

        ref = self.active_slots[0]
        offsets = {s: (nearest[s] - nearest[ref]) * 1000.0 for s in self.active_slots}
        self._offset_history.append(offsets)
        if self._baseline_offsets is None and len(self._offset_history) >= 10:
            baseline = {}
            recent = self._offset_history[:10]
            for s in self.active_slots:
                baseline[s] = float(np.median([x.get(s, 0.0) for x in recent]))
            self._baseline_offsets = baseline
        drift_ms = 0.0
        if self._baseline_offsets is not None:
            drift_ms = max(abs(offsets[s] - self._baseline_offsets.get(s, 0.0)) for s in self.active_slots)

        spreads = [float(x['current_spread_ms']) for x in self.sync_samples] + [float(spread_ms)]
        arr = np.asarray(spreads, dtype=np.float64)
        mean_ms = float(np.mean(arr)) if arr.size else 0.0
        p95_ms = float(np.percentile(arr, 95)) if arr.size else 0.0
        max_ms = float(np.max(arr)) if arr.size else 0.0

        # Grade relative to nominal frame period. This is host-arrival software
        # synchronization quality, not sensor exposure synchronization.
        fps = 30.0
        try:
            if self.session_meta:
                fps = float(self.session_meta.get('requested_fps', 30.0) or 30.0)
        except Exception:
            fps = 30.0
        frame_ms = 1000.0 / max(1.0, fps)
        basis = p95_ms if len(spreads) >= 5 else spread_ms
        if basis <= frame_ms * 0.5:
            grade, grade_cn = 'excellent', '很好'
        elif basis <= frame_ms:
            grade, grade_cn = 'good', '良好'
        elif basis <= frame_ms * 2.0:
            grade, grade_cn = 'usable', '可用'
        else:
            grade, grade_cn = 'poor', '偏大'

        sample = {
            'session_time_s': float(target_t),
            'current_spread_ms': float(spread_ms),
            'mean_spread_ms': mean_ms,
            'p95_spread_ms': p95_ms,
            'max_spread_ms': max_ms,
            'drift_ms': float(drift_ms),
            'grade': grade,
            'grade_cn': grade_cn,
            'reference_slot': int(ref),
            'offsets_ms': {str(k): float(v) for k, v in offsets.items()},
            'note': '基于主机接收帧时间戳的最近邻软件同步质量，不等同于相机传感器曝光同步。',
        }
        self.sync_samples.append(sample)
        self.sync_quality.emit(sample)

    def _on_failed(self, slot, message):
        self.failure_reason = '第%d路：%s' % (int(slot) + 1, message)
        self.log.emit('摄像头 %d：%s' % (int(slot) + 1, message))
        if self.clock is not None:
            self.clock.stop()
        self._running = False
        self.failed.emit('第 %d 路摄像头失败：%s' % (int(slot) + 1, message))

    def _on_finished(self, slot, result):
        self.finished_slots.add(int(slot))
        self.results[str(int(slot))] = result
        if len(self.finished_slots) == len(self.workers):
            self._finalize_session()

    def stop(self, discard=False):
        if not self._running and not self.workers:
            return
        self._discard_on_stop = bool(discard)
        if self.clock is not None:
            self.clock.stop()
        for w in self.workers:
            w.request_stop()
            if not w.isRunning() and not w.isFinished():
                self.finished_slots.add(int(w.slot_id))
        self.log.emit('正在停止录像%s…' % ('并准备删除本次数据' if self._discard_on_stop else '并写入索引文件'))
        if len(self.finished_slots) >= len(self.workers):
            self._finalize_session()

    def _write_capture_quality_csv(self):
        if not self.session_dir:
            return
        path = (getattr(self, 'metadata_dir', None) or (self.session_dir / '_metadata')) / 'capture_quality.csv'
        fields = [
            'camera_slot', 'written_frames', 'theoretical_frames', 'frame_deficit',
            'effective_fps', 'gap_events', 'gap_equivalent_frames', 'read_failures_total',
            'p95_interval_ms', 'jitter_ms', 'max_interval_ms', 'actual_width', 'actual_height',
            'actual_fps_reported', 'capture_backend'
        ]
        with open(str(path), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for slot_text in sorted(self.results.keys(), key=lambda x: int(x)):
                r = self.results.get(slot_text, {}) or {}
                row = {'camera_slot': int(slot_text) + 1}
                for key in fields[1:]:
                    row[key] = r.get(key, '')
                writer.writerow(row)

    def _write_sync_quality_csv(self):
        if not self.session_dir or not self.sync_samples:
            return
        path = (getattr(self, 'metadata_dir', None) or (self.session_dir / '_metadata')) / 'sync_quality.csv'
        fields = [
            'session_time_s', 'current_spread_ms', 'mean_spread_ms', 'p95_spread_ms',
            'max_spread_ms', 'drift_ms', 'grade_cn', 'offsets_ms'
        ]
        with open(str(path), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in self.sync_samples:
                writer.writerow({
                    'session_time_s': '%.6f' % float(s.get('session_time_s', 0.0)),
                    'current_spread_ms': '%.3f' % float(s.get('current_spread_ms', 0.0)),
                    'mean_spread_ms': '%.3f' % float(s.get('mean_spread_ms', 0.0)),
                    'p95_spread_ms': '%.3f' % float(s.get('p95_spread_ms', 0.0)),
                    'max_spread_ms': '%.3f' % float(s.get('max_spread_ms', 0.0)),
                    'drift_ms': '%.3f' % float(s.get('drift_ms', 0.0)),
                    'grade_cn': s.get('grade_cn', ''),
                    'offsets_ms': json.dumps(s.get('offsets_ms', {}), ensure_ascii=False),
                })

    def _finalize_session(self):
        self._running = False
        meta = dict(self.session_meta or {})
        meta['camera_results'] = self.results
        meta['finished_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        meta['session_dir'] = str(self.session_dir) if self.session_dir else ''
        meta['discarded'] = bool(self._discard_on_stop)
        meta['session_state'] = 'failed' if self.failure_reason else 'finished'
        if self.failure_reason:
            meta['capture_failure_reason'] = self.failure_reason
        if self.sync_samples:
            last = dict(self.sync_samples[-1])
            meta['software_sync_quality'] = {
                'mean_spread_ms': last.get('mean_spread_ms', 0.0),
                'p95_spread_ms': last.get('p95_spread_ms', 0.0),
                'max_spread_ms': last.get('max_spread_ms', 0.0),
                'drift_ms': last.get('drift_ms', 0.0),
                'grade': last.get('grade', ''),
                'grade_cn': last.get('grade_cn', ''),
                'note': last.get('note', ''),
            }
        if self.session_dir and self._discard_on_stop:
            try:
                shutil.rmtree(str(self.session_dir), ignore_errors=False)
                self.log.emit('本次采集目录已删除。')
            except Exception as e:
                meta['discard_delete_error'] = str(e)
                self.log.emit('删除本次采集目录失败：%s' % e)
        elif self.session_dir:
            try:
                self._write_capture_quality_csv()
                self._write_sync_quality_csv()
                (self.session_dir / 'session.json').write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
                )
            except Exception as e:
                self.log.emit('写入采集质量文件失败：%s' % e)
        self.stopped.emit(meta)

    def wait_for_stop(self, timeout_ms=8000):
        deadline = time.time() + timeout_ms / 1000.0
        for w in list(self.workers):
            remain = max(0.0, deadline - time.time())
            w.wait(int(remain * 1000))


def probe_camera_indices(max_index=8, warmup_attempts=12):
    """Probe readable camera indices. Only devices that return a real frame are listed."""
    found = []
    if os.name == 'nt':
        backends = [('DirectShow', cv2.CAP_DSHOW), ('MediaFoundation', cv2.CAP_MSMF)]
    else:
        backends = [('Auto', cv2.CAP_ANY)]

    for idx in range(int(max_index)):
        device_info = None
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            try:
                if not cap.isOpened():
                    continue
                ok = False
                frame = None
                for _ in range(max(1, int(warmup_attempts))):
                    ok, frame = cap.read()
                    if ok and frame is not None and getattr(frame, 'size', 0) > 0:
                        break
                    time.sleep(0.03)
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                device_info = {
                    'index': idx, 'width': int(w), 'height': int(h),
                    'fps': fps, 'backend': backend_name,
                }
                break
            finally:
                cap.release()
        if device_info is not None:
            found.append(device_info)
    return found


def storage_free_gb(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(p))
    return usage.free / (1024.0 ** 3)
