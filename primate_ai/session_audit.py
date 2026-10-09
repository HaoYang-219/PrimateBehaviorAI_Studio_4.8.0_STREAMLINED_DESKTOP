# coding: utf-8
"""Post-acquisition acceptance checks for 1-3 camera experiment sessions.

This audits host files and timestamps; it does NOT claim camera hardware frame counters
or exposure-level synchronization. No GUI or Qt required for testing.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.acceptance-', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, str(path))
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _timestamp_summary(file_path):
    """Stream timestamps so an hour-long session does not require a DataFrame."""
    info = {'rows': 0, 'first_s': None, 'last_s': None, 'nonmonotonic': 0,
            'index_gaps': 0, 'read_error': '', 'start_index': None, 'end_index': None}
    try:
        with Path(file_path).open('r', newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            if not {'frame_index', 'session_time_s'}.issubset(set(reader.fieldnames or [])):
                raise ValueError('时间戳文件缺少帧号或 Session 时间列')
            prev_t, prev_idx = None, None
            for row in reader:
                idx = int(row['frame_index'])
                ts = float(row['session_time_s'])
                if idx < 0 or not math.isfinite(ts) or ts < 0:
                    raise ValueError('发现无效帧号或时间戳')
                if prev_idx is not None:
                    if idx <= prev_idx or ts < prev_t:
                        info['nonmonotonic'] += 1
                    if idx > prev_idx + 1:
                        info['index_gaps'] += idx - prev_idx - 1
                else:
                    info['first_s'] = ts
                    info['start_index'] = idx
                prev_t, prev_idx = ts, idx
                info['last_s'] = ts
                info['end_index'] = idx
                info['rows'] += 1
    except (OSError, KeyError, ValueError, OverflowError) as exc:
        info['read_error'] = str(exc)
    return info


def _probe_video(path):
    """Validate that a recording is decodable at its beginning and near its end."""
    import cv2
    path = Path(path)
    result = {'filename': path.name, 'exists': path.is_file(), 'bytes': 0,
              'opened': False, 'start_readable': False, 'end_readable': False,
              'width': 0, 'height': 0, 'fps_header': 0.0, 'frame_count_header': 0,
              'error': ''}
    if not path.is_file():
        result['error'] = '缺少视频文件'
        return result
    result['bytes'] = path.stat().st_size
    if result['bytes'] <= 4096:
        result['error'] = '视频文件为空或明显过小'
        return result
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result['error'] = '视频解码器无法打开文件'
            return result
        result['opened'] = True
        result['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        result['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        result['fps_header'] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        result['frame_count_header'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        result['start_readable'] = bool(ok and frame is not None and frame.size > 0)
        n = result['frame_count_header']
        if n > 3:
            # On some AVI codecs accurate random seeks are unavailable. Try nearby frames.
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 3))
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    result['end_readable'] = True
        else:
            result['end_readable'] = result['start_readable']
        if not result['start_readable']:
            result['error'] = '文件可打开，但首帧不能解码'
    finally:
        cap.release()
    return result


def assess_session(session_dir, save=True):
    """Return a conservative, readable QC result. Never relabel jitter as real lost sensor frames."""
    root = Path(session_dir)
    manifest_file = root / 'session.json'
    if not manifest_file.exists():
        raise FileNotFoundError('找不到 session.json：%s' % root)
    meta = json.loads(manifest_file.read_text(encoding='utf-8'))
    specs = meta.get('camera_specs') or []
    if not specs:
        raise ValueError('session.json 中没有摄像头清单')
    results = []
    warnings = []
    errors = []
    for spec in specs:
        slot = int(spec.get('slot_id', 0)) + 1
        view = str(spec.get('view_key') or 'custom')
        prefix = 'cam%02d_%s' % (slot, view)
        # Prefer actual paths recorded by the acquisition controller.
        worker_results = meta.get('camera_results') or {}
        recorded = worker_results.get(str(slot-1), worker_results.get(str(slot), {}))
        if not isinstance(recorded, dict):
            recorded = {}
        video = Path(recorded.get('video') or root / (prefix + '.avi'))
        if not video.is_absolute():
            video = root / video
        stamp = Path(recorded.get('timestamps') or root / '_metadata' / ('timestamps_' + prefix + '.csv'))
        if not stamp.is_absolute():
            stamp = root / stamp
        if not stamp.is_file():
            legacy = root / ('timestamps_' + prefix + '.csv')
            if legacy.is_file():
                stamp = legacy
        movie = _probe_video(video)
        timing = _timestamp_summary(stamp)
        reasons = []
        level = '通过'
        if not movie['start_readable'] or not timing['rows']:
            level = '未通过'
            reasons.append(movie['error'] or timing['read_error'] or '缺少可用视频或时间戳')
        if timing['read_error'] or timing['nonmonotonic']:
            level = '未通过'
            reasons.append('时间戳格式损坏或序列不单调')
        if not movie['end_readable'] and movie['start_readable']:
            if level != '未通过':
                level = '需复核'
            reasons.append('视频末端无法通过随机读取验证；需人工查看片尾')
        header_count = movie['frame_count_header']
        timestamp_count = timing['rows']
        if header_count and timestamp_count:
            delta = abs(header_count - timestamp_count)
            if delta > max(3, int(timestamp_count * .02)):
                level = '需复核' if level == '通过' else level
                reasons.append('视频头帧数与时间戳记录相差 %d 帧（可能存在编码/索引差异）' % delta)
        requested_w, requested_h = int(spec.get('width') or 0), int(spec.get('height') or 0)
        if movie['opened'] and requested_w and requested_h and (movie['width'], movie['height']) != (requested_w, requested_h):
            level = '需复核' if level == '通过' else level
            reasons.append('视频实际分辨率与请求值不一致')
        if level == '未通过':
            errors.append('第%d路：%s' % (slot, '；'.join(reasons)))
        elif level == '需复核':
            warnings.append('第%d路：%s' % (slot, '；'.join(reasons)))
        results.append({'slot': slot, 'view': view, 'status': level,
                        'video': movie, 'timestamps': timing, 'notes': reasons})
    sync = meta.get('software_sync_quality') or {}
    if len(specs) > 1:
        try:
            p95 = float(sync.get('p95_spread_ms'))
            fps = float(meta.get('requested_fps') or specs[0].get('fps') or 30.)
            if math.isfinite(p95) and p95 > 2000. / max(fps, 1.):
                warnings.append('软件同步 P95 %.1f ms 超过两帧时间；不建议未经校正直接融合短促行为事件' % p95)
        except (TypeError, ValueError):
            warnings.append('未找到有效的多路主机侧同步统计；请确认同步时间日志')
    state = str(meta.get('session_state') or '')
    if state not in ('completed', 'stopped', 'finished'):
        warnings.append('Session 状态为 %s：可能是异常中断或未完成收尾，请检查录像末尾' % (state or '未知'))
    overall = '未通过' if errors else ('需复核' if warnings else '通过')
    summary = {'schema_version': 1, 'product': 'PrimateBehaviorAI',
               'generated_utc': datetime.now(timezone.utc).isoformat(),
               'session_dir': str(root), 'session_state': state,
               'camera_count': len(results), 'overall_status': overall,
               'cameras': results, 'sync_host_quality': sync,
               'warnings': warnings, 'errors': errors,
               'limitations': '仅检查主机侧视频可解码性与采集时间戳，不代表传感器曝光同步、真实硬件掉帧计数或行为检测准确率。'}
    if save:
        _atomic_text(root / '_metadata' / 'capture_acceptance.json', json.dumps(summary, ensure_ascii=False, indent=2))
        _atomic_text(root / '采集验收报告.html', render_audit_html(summary))
    return summary


def render_audit_html(result):
    esc = lambda x: html.escape(str(x))
    rows = []
    for camera in result['cameras']:
        video = camera['video']
        log = camera['timestamps']
        rows.append('<tr><td>第%d路</td><td>%s</td><td>%s</td><td>%dx%d</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            camera['slot'], esc(camera['view']), esc(camera['status']), video['width'], video['height'],
            esc(video['frame_count_header']), esc(log['rows']), esc('；'.join(camera['notes']) or '—')))
    issues = result['errors'] + result['warnings']
    issue_markup = '<p>本次检查未发现文件级异常。</p>' if not issues else '<ul>' + ''.join('<li>'+esc(x)+'</li>' for x in issues) + '</ul>'
    sync = result.get('sync_host_quality') or {}
    p95 = sync.get('p95_spread_ms')
    sync_markup = ('主机接收帧时间戳偏差 P95：%.1f ms（不代表硬件曝光同步）' % float(p95)) if p95 is not None else '无有效同步统计或仅单路采集'
    return '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>采集验收报告</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:35px auto;padding:0 22px;color:#1c2a33;background:#fff}h1{font-size:25px}h2{font-size:18px;margin-top:30px}.subtitle{color:#617585}.status{font-weight:700;border-left:4px solid #365e72;padding:12px;background:#f4f7f8}table{border-collapse:collapse;width:100%%;font-size:14px}th,td{border:1px solid #dbe2e5;padding:9px;text-align:left}th{background:#f3f5f7}.fine{font-size:12px;color:#72828a}</style></head><body>
<h1>实验采集验收报告</h1><p class="subtitle">Session：%s</p><p class="status">检查结果：%s　｜　摄像头 %d 路　｜　录制状态：%s</p>
<h2>每路视频与时间戳</h2><table><tr><th>设备</th><th>视角</th><th>结果</th><th>实际分辨率</th><th>视频头帧数</th><th>时间戳行数</th><th>说明</th></tr>%s</table>
<h2>同步与异常提示</h2><p>%s</p>%s
<p class="fine">%s</p><p class="fine">文件级验收是快速检查，不能代替完整逐帧解码、人工审片或外部硬件同步验证。</p>
</body></html>''' % (esc(Path(result['session_dir']).name), esc(result['overall_status']), result['camera_count'], esc(result['session_state']), ''.join(rows), esc(sync_markup), issue_markup, esc(result['limitations']))
