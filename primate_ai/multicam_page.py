from __future__ import annotations

import re
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl, QSettings
from PySide6.QtGui import QImage, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QFrame, QFormLayout, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QCheckBox, QDoubleSpinBox, QGridLayout,
    QTabWidget, QScrollArea, QSizePolicy, QInputDialog
)

from .capture.multicam import (
    MultiCameraController, LivePreviewController, probe_camera_indices, storage_free_gb,
    test_single_camera_config, test_camera_group, recommend_camera_mode,
)
from .paths import CAPTURES_DIR
from .session_templates import list_templates, save_template, load_template
from .session_audit import assess_session
from .preflight_policy import preflight_candidates


VIEW_OPTIONS = [
    ('front', '前视角'),
    ('left', '左侧视角'),
    ('right', '右侧视角'),
    ('front_left', '前左斜视'),
    ('front_right', '前右斜视'),
    ('custom', '自定义'),
]


class ExperimentSessionLogger(object):
    """Simple, append-friendly Session logger centered on continuous stage labels.

    User-facing output is intentionally compact: session_record.csv + session.json.
    Technical timing/QC artifacts are stored under _metadata/.
    """
    RECORD_FIELDS = [
        'record_id','record_type','stage_type','stage_name','condition',
        'start_session_time_s','end_session_time_s','duration_s',
        'carrier_frequency_hz','modulation_frequency_hz','signal_strength','signal_unit',
        'waveform','antenna_distance_cm','custom_parameters','note','operator',
        'start_local_time','end_local_time','start_utc_time','end_utc_time'
    ]
    EVENT_FIELDS = [
        'session_time_s','monotonic_s','local_time','utc_time','source','event_type',
        'action','record_id','parameters_json','note'
    ]
    WINDOW_FIELDS = [
        'source_id','condition','window_type','start_s','end_s','duration_s','note','parameters_json'
    ]

    def __init__(self, session_dir, operator=''):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.session_dir / '_metadata'
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.operator = str(operator or '')
        self.active_stage = None
        self.stage_blocks = []
        self.records = []
        self._stage_counter = 0
        self._note_counter = 0
        self._ensure_csv(self.session_dir / 'session_record.csv', self.RECORD_FIELDS)
        self._ensure_csv(self.meta_dir / 'experiment_events.csv', self.EVENT_FIELDS)

    def _ensure_csv(self, path, fields):
        path = Path(path)
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), 'w', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    @staticmethod
    def _stamp(session_time_s):
        return {
            'session_time_s': float(max(0.0, session_time_s)),
            'monotonic_s': float(time.perf_counter()),
            'local_time': datetime.now().astimezone().isoformat(timespec='milliseconds'),
            'utc_time': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        }

    @staticmethod
    def _fmt(value):
        return ('%.6f' % value) if isinstance(value, float) else value

    def _append_csv(self, path, fields, row):
        path = Path(path)
        exists = path.exists() and path.stat().st_size > 0
        with open(str(path), 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({k: self._fmt(row.get(k, '')) for k in fields})
            f.flush()

    def _event(self, session_time_s, source, event_type, action='mark', record_id='', parameters=None, note=''):
        row = self._stamp(session_time_s)
        row.update({
            'source': str(source), 'event_type': str(event_type), 'action': str(action),
            'record_id': str(record_id or ''),
            'parameters_json': json.dumps(parameters or {}, ensure_ascii=False, sort_keys=True),
            'note': str(note or ''),
        })
        self._append_csv(self.meta_dir / 'experiment_events.csv', self.EVENT_FIELDS, row)
        return row

    @staticmethod
    def _param_value(params, key, default=''):
        value = (params or {}).get(key, default)
        return '' if value is None else value

    def start_stage(self, session_time_s, stage_type, stage_name, params=None, note=''):
        ended = None
        if self.active_stage is not None:
            ended = self.end_stage(session_time_s, reason='switch_stage')
        self._stage_counter += 1
        sid = 'P%02d' % self._stage_counter
        st = self._stamp(session_time_s)
        self.active_stage = {
            'stage_id': sid,
            'stage_type': str(stage_type),
            'stage_name': str(stage_name),
            'params': dict(params or {}),
            'note': str(note or ''),
            'start': st,
        }
        self._event(session_time_s, 'stage', stage_name, 'start', sid, params, note)
        self._rewrite_record_sorted()  # crash-safe in-progress stage checkpoint
        return ended, dict(self.active_stage)

    def end_stage(self, session_time_s, reason='manual'):
        if self.active_stage is None:
            return None
        a = self.active_stage
        p = dict(a.get('params') or {})
        en = self._stamp(session_time_s)
        dur = max(0.0, en['session_time_s'] - a['start']['session_time_s'])
        row = {
            'record_id': a['stage_id'], 'record_type': 'stage',
            'stage_type': a['stage_type'], 'stage_name': a['stage_name'],
            'condition': self._param_value(p, 'condition', ''),
            'start_session_time_s': a['start']['session_time_s'],
            'end_session_time_s': en['session_time_s'], 'duration_s': dur,
            'carrier_frequency_hz': self._param_value(p, 'carrier_frequency_hz', ''),
            'modulation_frequency_hz': self._param_value(p, 'modulation_frequency_hz', ''),
            'signal_strength': self._param_value(p, 'signal_strength', ''),
            'signal_unit': self._param_value(p, 'signal_unit', ''),
            'waveform': self._param_value(p, 'waveform', ''),
            'antenna_distance_cm': self._param_value(p, 'antenna_distance_cm', ''),
            'custom_parameters': self._param_value(p, 'custom_parameters', ''),
            'note': a['note'], 'operator': self.operator,
            'start_local_time': a['start']['local_time'], 'end_local_time': en['local_time'],
            'start_utc_time': a['start']['utc_time'], 'end_utc_time': en['utc_time'],
        }
        self.stage_blocks.append(row)
        self.records.append(row)
        self._rewrite_record_sorted()
        self._event(en['session_time_s'], 'stage', a['stage_name'], 'end', a['stage_id'], {'duration_s': dur, 'reason': reason}, a['note'])
        self.active_stage = None
        return row

    def add_note(self, session_time_s, note):
        self._note_counter += 1
        rid = 'N%03d' % self._note_counter
        st = self._stamp(session_time_s)
        row = {
            'record_id': rid, 'record_type': 'note', 'stage_type': '', 'stage_name': '', 'condition': '',
            'start_session_time_s': st['session_time_s'], 'end_session_time_s': st['session_time_s'], 'duration_s': 0.0,
            'carrier_frequency_hz': '', 'modulation_frequency_hz': '', 'signal_strength': '', 'signal_unit': '',
            'waveform': '', 'antenna_distance_cm': '', 'custom_parameters': '',
            'note': str(note or ''), 'operator': self.operator,
            'start_local_time': st['local_time'], 'end_local_time': st['local_time'],
            'start_utc_time': st['utc_time'], 'end_utc_time': st['utc_time'],
        }
        self.records.append(row)
        self._rewrite_record_sorted()
        self._event(session_time_s, 'note', 'manual_note', 'mark', rid, {}, note)
        return row

    def add_ai_candidate(self, session_time_s, camera_slot, event_type, motion_score=0.0, robust_z=0.0):
        self._event(session_time_s, 'ai_candidate', event_type, 'mark', '', {
            'camera_slot': int(camera_slot), 'motion_score': float(motion_score), 'robust_z': float(robust_z)
        }, '')

    def _rewrite_record_sorted(self):
        path = self.session_dir / 'session_record.csv'
        rows=list(self.records)
        if self.active_stage is not None:
            a=self.active_stage; p=a.get('params',{}); st=a['start']
            rows.append({
                'record_id':a['stage_id'],'record_type':'stage','stage_type':a['stage_type'],
                'stage_name':a['stage_name'],'condition':p.get('condition',''),
                'start_session_time_s':st['session_time_s'], 'end_session_time_s':'',
                'duration_s':'','carrier_frequency_hz':p.get('carrier_frequency_hz',''),
                'modulation_frequency_hz':p.get('modulation_frequency_hz',''),
                'signal_strength':p.get('signal_strength',''),'signal_unit':p.get('signal_unit',''),
                'waveform':p.get('waveform',''),'antenna_distance_cm':p.get('antenna_distance_cm',''),
                'custom_parameters':p.get('custom_parameters',''),
                'note':a.get('note',''),'operator':self.operator,
                'start_local_time':st['local_time'],'end_local_time':'',
                'start_utc_time':st['utc_time'],'end_utc_time':''})
        rows.sort(key=lambda r: (float(r.get('start_session_time_s',0.0) or 0.0),str(r.get('record_id',''))))
        # Atomic file replacement: unexpected shutdown cannot truncate the old record.
        temp=path.with_suffix('.csv.tmp')
        with open(str(temp),'w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=self.RECORD_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:self._fmt(row.get(k,'')) for k in self.RECORD_FIELDS})
            f.flush()
        temp.replace(path)

    def _write_analysis_windows(self):
        path = self.meta_dir / 'analysis_windows.csv'
        with open(str(path), 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=self.WINDOW_FIELDS)
            writer.writeheader()
            for r in self.stage_blocks:
                params = {
                    'condition': r.get('condition',''),
                    'carrier_frequency_hz': r.get('carrier_frequency_hz',''),
                    'modulation_frequency_hz': r.get('modulation_frequency_hz',''),
                    'signal_strength': r.get('signal_strength',''),
                    'signal_unit': r.get('signal_unit',''),
                    'waveform': r.get('waveform',''),
                    'antenna_distance_cm': r.get('antenna_distance_cm',''),
                    'custom_parameters': r.get('custom_parameters',''),
                }
                writer.writerow({
                    'source_id': r.get('record_id',''),
                    'condition': r.get('condition','') or r.get('stage_name',''),
                    'window_type': r.get('stage_type',''),
                    'start_s': self._fmt(float(r.get('start_session_time_s',0.0))),
                    'end_s': self._fmt(float(r.get('end_session_time_s',0.0))),
                    'duration_s': self._fmt(float(r.get('duration_s',0.0))),
                    'note': r.get('note',''),
                    'parameters_json': json.dumps(params, ensure_ascii=False, sort_keys=True),
                })
        return len(self.stage_blocks)

    def finalize(self, session_end_s, reason='session_stop'):
        if self.active_stage is not None:
            self.end_stage(session_end_s, reason)
        self._rewrite_record_sorted()
        n = self._write_analysis_windows()
        return {
            'stage_count': len(self.stage_blocks),
            'analysis_window_count': n,
            'user_files': ['session_record.csv', 'session.json'],
            'technical_metadata_dir': '_metadata',
            'note': '原始视频连续保存；实验阶段通过时间戳虚拟切分，默认不物理切视频。'
        }


class ProbeWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self,parent=None,max_index=6):
        super().__init__(parent)
        self.max_index=int(max_index)

    def run(self):
        try:
            self.done.emit(probe_camera_indices(self.max_index,warmup_attempts=8))
        except Exception as e:
            self.failed.emit(str(e))


class PreflightWorker(QThread):
    progress = Signal(str)
    frame = Signal(int, object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, specs, parent=None, detailed=False, automatic=False):
        super().__init__(parent)
        self.specs = [dict(x) for x in specs]
        self.detailed = bool(detailed)
        self.automatic = bool(automatic)

    def _send_frames(self, group, specs):
        by_device = {int(x['device_index']): int(x['slot_id']) for x in specs}
        for key, metrics in (group.get('results') or {}).items():
            frame = metrics.get('sample_frame')
            if frame is not None and int(key) in by_device:
                self.frame.emit(by_device[int(key)], frame)

    def run(self):
        try:
            if not self.detailed:
                # One combined test, never N expensive single-camera tests on success.
                current=self.specs[0]
                candidate_modes=preflight_candidates(current['width'],current['height'],current['fps'],self.automatic)
                last = None
                for w,h,fps in candidate_modes:
                    specs = [dict(x,width=w,height=h,fps=fps) for x in self.specs]
                    self.progress.emit('快速联合检查：%d路 · %d×%d @%d FPS' % (len(specs),w,h,fps))
                    group = test_camera_group(specs,duration_s=1.7,startup_attempts=18,retries_per_backend=1)
                    self._send_frames(group,specs)
                    last = {'ok':bool(group.get('ok')), 'stage':'group','single':{},'group':group,
                            'reason':group.get('reason',''), 'quick':True, 'mode':(w,h,fps),
                            'automatic':self.automatic}
                    if group.get('ok'):
                        self.done.emit(last)
                        return
                self.done.emit(last or {'ok':False,'stage':'group','single':{},'group':{},'reason':'无可用采集模式'})
                return

            single = {}
            for i,spec in enumerate(self.specs):
                slot=int(spec.get('slot_id',i));idx=int(spec['device_index'])
                self.progress.emit('深度诊断：逐路检查摄像头 %d（设备 %d）…' % (slot+1,idx))
                res=test_single_camera_config(spec,duration_s=2.5)
                if res.get('grade') != 'stable':
                    res['recommendation']=recommend_camera_mode(spec)
                single[str(slot)]=res
                frame=res.get('sample_frame')
                if frame is not None:self.frame.emit(slot,frame)
                if res.get('grade')=='poor':
                    self.done.emit({'ok':False,'stage':'single','single':single,'group':None,
                                    'reason':'第 %d 路当前模式不建议正式采集。' % (slot+1),'quick':False})
                    return
            self.progress.emit('深度诊断：所有通道联合稳定性测试…')
            group=test_camera_group(self.specs,duration_s=3.0)
            self._send_frames(group,self.specs)
            self.done.emit({'ok':bool(group.get('ok')),'stage':'group','single':single,'group':group,
                            'reason':group.get('reason',''),'quick':False})
        except Exception as exc:
            self.failed.emit(str(exc))


class AspectVideoLabel(QLabel):
    """Keeps original 4:3/16:9 frame proportions and redraws after resize."""
    def __init__(self, parent=None):
        super().__init__('等待摄像头',parent)
        self._source = None
        self.frame_ratio = 4.0/3.0
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160,120)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred)

    def set_bgr_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2]
        self.frame_ratio=float(w)/max(1,h)
        self._source=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888).copy()
        self._redraw()

    def _redraw(self):
        if self._source is None:return
        pix=QPixmap.fromImage(self._source)
        self.setPixmap(pix.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._redraw()


class CameraPreviewCard(QFrame):
    def __init__(self, slot, default_device, default_view, parent=None):
        super().__init__(parent)
        self.slot = slot
        self.setObjectName('card')
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel('摄像头 %d' % (slot + 1))
        title.setStyleSheet('font-size:16px;font-weight:700;')
        self.enabled = QCheckBox('启用')
        self.enabled.setChecked(slot == 0)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.enabled)
        layout.addLayout(top)

        self.preview = AspectVideoLabel()
        self.preview.setStyleSheet('background:#03080C;border:1px solid #28475C;border-radius:10px;color:#6F8998;')
        layout.addWidget(self.preview, 1)

        row = QHBoxLayout()
        self.device = QComboBox()
        for i in range(8):
            self.device.addItem('设备 %d' % i, i)
        self.device.setCurrentIndex(default_device)
        self.view = QComboBox()
        for key, text in VIEW_OPTIONS:
            self.view.addItem(text, key)
        for i in range(self.view.count()):
            if self.view.itemData(i) == default_view:
                self.view.setCurrentIndex(i)
                break
        row.addWidget(QLabel('设备'))
        row.addWidget(self.device, 1)
        row.addWidget(QLabel('视角'))
        row.addWidget(self.view, 1)
        layout.addLayout(row)

        self.status = QLabel('FPS -- · --×-- · 待机')
        self.status.setStyleSheet('font-weight:700;color:#B9CBD5;')
        layout.addWidget(self.status)
        self.qc_status = QLabel('写入 -- · 目标帧差 -- · P95 -- ms')
        self.qc_status.setObjectName('muted')
        layout.addWidget(self.qc_status)
        self.preflight = QLabel('采集预检：未执行')
        self.preflight.setObjectName('muted')
        self.preflight.setWordWrap(True)
        layout.addWidget(self.preflight)

    def populate_devices(self, found):
        previous = self.device.currentData()
        self.device.clear()
        if not found:
            self.device.addItem('未检测到可读取设备', None)
            return
        for info in found:
            text = '设备 %d · %dx%d · %.1f FPS · %s' % (
                int(info.get('index', -1)), int(info.get('width', 0)), int(info.get('height', 0)),
                float(info.get('fps', 0.0)), info.get('backend', '')
            )
            self.device.addItem(text, int(info.get('index', -1)))
        target = None
        for i in range(self.device.count()):
            if self.device.itemData(i) == previous:
                target = i
                break
        if target is not None:
            self.device.setCurrentIndex(target)

    def set_preflight(self, text, ok=None):
        self.preflight.setText('采集预检：' + str(text))
        if ok is True:
            self.preflight.setStyleSheet('color:#43D18B;font-weight:700;')
        elif ok == 'warning':
            self.preflight.setStyleSheet('color:#F0B84B;font-weight:700;')
        elif ok is False:
            self.preflight.setStyleSheet('color:#FF7070;font-weight:700;')
        else:
            self.preflight.setStyleSheet('')

    def selected_device(self):
        value = self.device.currentData()
        return None if value is None else int(value)

    def set_frame(self, bgr):
        if bgr is not None:
            self.preview.set_bgr_frame(bgr)

    def set_status(self, info):
        self.status.setText('%.1f FPS · %.0f×%.0f · %s' % (
            float(info.get('fps', 0.0)), float(info.get('width', 0)), float(info.get('height', 0)), info.get('state', '未知')))
        self.qc_status.setText('写入 %d · 目标帧差 %d · P95 %.1f ms' % (
            int(info.get('written_frames', info.get('frames', 0))), int(info.get('frame_deficit', 0)), float(info.get('p95_interval_ms', 0.0))))


class SessionAuditWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, session_dir, parent=None):
        super().__init__(parent)
        self.session_dir = Path(session_dir)

    def run(self):
        try:
            self.completed.emit(assess_session(self.session_dir, save=True))
        except Exception as e:
            self.failed.emit(str(e))


class MultiCamPage(QWidget):
    def __init__(self, qss='', parent=None):
        super().__init__(parent)
        if qss:
            self.setStyleSheet(qss)
        self.controller = MultiCameraController(self)
        self.preview_controller = LivePreviewController(self)
        self.probe_worker = None
        self.preflight_worker = None
        self.audit_worker = None
        self.preflight_signature = None
        self.latest_status = {}
        self.recording_started = False
        self.current_session_dir = None
        self.experiment_logger = None
        self.active_slots = set()
        self._pending_start = None
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._tick)
        self._elapsed_s = 0.0
        self.qt_settings=QSettings('PrimateBehaviorAI','Studio')
        self._last_disk_warn=0.0
        self._disk_emergency_stop=False
        self._build()
        self._wire()

    def _build(self):
        # Top: always-visible camera views. Bottom: two workflow tabs.
        # This keeps live video visible while preventing experiment controls from being squeezed.
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        head = QHBoxLayout()
        a = QVBoxLayout()
        title = QLabel('新建实验 · 视频采集')
        title.setObjectName('title')
        sub = QLabel('连接摄像头 → 快速准备 → 查看实时画面 → 开始实验；设备异常时再做深度诊断。')
        sub.setObjectName('subtitle')
        sub.setWordWrap(True)
        a.addWidget(title)
        a.addWidget(sub)
        head.addLayout(a, 1)
        badge = QLabel('4.8.0 · DESKTOP')
        badge.setStyleSheet('background:#102B37;color:#27D7C6;border-radius:8px;padding:7px 10px;font-weight:700;')
        head.addWidget(badge)
        root.addLayout(head)

        camera_bar = QHBoxLayout()
        camera_bar.addWidget(QLabel('本次使用'))
        self.camera_count = QComboBox()
        for n in (1,2,3):
            self.camera_count.addItem('%d 路摄像头'%n,n)
        self.camera_count.setCurrentIndex(0)
        camera_bar.addWidget(self.camera_count)
        self.camera_choice_tip=QLabel('先检测设备；检测结果会自动推荐启用数量，也可以手动切换。')
        self.camera_choice_tip.setObjectName('muted');self.camera_choice_tip.setWordWrap(True)
        camera_bar.addWidget(self.camera_choice_tip,1)
        root.addLayout(camera_bar)

        cams = QHBoxLayout()
        cams.setSpacing(10)
        self.cards = [
            CameraPreviewCard(0, 0, 'front'),
            CameraPreviewCard(1, 1, 'left'),
            CameraPreviewCard(2, 2, 'right'),
        ]
        for card in self.cards:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            card.preview.setMinimumSize(160,120)
            cams.addWidget(card, 1)
        self.camera_strip = cams
        root.addLayout(cams)
        self._apply_camera_count()

        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.setDocumentMode(True)
        self.workflow_tabs.setStyleSheet('QTabBar::tab{padding:10px 22px;font-weight:700;} QTabBar::tab:selected{color:#27D7C6;}')
        root.addWidget(self.workflow_tabs, 1)

        # -------------------- Tab 1: acquisition preparation --------------------
        prep_scroll = QScrollArea()
        prep_scroll.setWidgetResizable(True)
        prep_scroll.setFrameShape(QFrame.NoFrame)
        prep = QWidget()
        pl = QVBoxLayout(prep)
        pl.setContentsMargins(8, 10, 8, 10)
        pl.setSpacing(10)
        prep_scroll.setWidget(prep)
        self.workflow_tabs.addTab(prep_scroll, '① 采集准备')

        control = QFrame()
        control.setObjectName('card')
        cl = QVBoxLayout(control)
        ct = QLabel('Session 与采集参数')
        ct.setObjectName('cardtitle')
        cl.addWidget(ct)

        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(8)

        self.monkey_id = QLineEdit('M01')
        self.protocol_code = QLineEdit('EMF40Hz')
        self.session_name = QLineEdit('EMF_session')
        self.operator = QLineEdit('')
        self.resolution = QComboBox()
        self.resolution.addItems(['自动适配 · 优先30 FPS','1920×1080','1280×720','640×480'])
        self._auto_selected_mode=(1280,720,30)
        self.fps = QComboBox()
        self.fps.addItems(['30 FPS', '60 FPS'])
        self.advanced_toggle=QCheckBox('高级摄像设置 / 完整诊断')
        self.advanced_toggle.setChecked(False)
        self.session_duration_mode = QLabel('手动结束（不限时）')
        self.session_duration_mode.setObjectName('muted')

        f1 = QFormLayout(); f1.addRow('猴编号', self.monkey_id); f1.addRow('实验方案/批次', self.protocol_code); f1.addRow('Session 名称', self.session_name); f1.addRow('操作者', self.operator)
        f2 = QFormLayout(); f2.addRow('视频质量', self.resolution); f2.addRow('Session 时长', self.session_duration_mode)
        self.advanced_fps_row=QWidget(); af=QHBoxLayout(self.advanced_fps_row);af.setContentsMargins(0,0,0,0);af.addWidget(QLabel('目标帧率'));af.addWidget(self.fps,1)
        f2.addRow('',self.advanced_toggle);f2.addRow('',self.advanced_fps_row)
        self.advanced_fps_row.hide()
        w1 = QWidget(); w1.setLayout(f1); w2 = QWidget(); w2.setLayout(f2)
        grid.addWidget(w1, 0, 0)
        grid.addWidget(w2, 0, 1)

        storage_box = QGroupBox('保存与质量状态')
        sf = QFormLayout(storage_box)
        self.save_root = QLineEdit(str(self.qt_settings.value('save_root',str(CAPTURES_DIR))))
        self.save_root.setReadOnly(True)
        self.save_root.setToolTip('当前 Session 的保存根目录。点击“选择目录…”修改。')
        self.save_browse_btn = QPushButton('选择目录…')
        save_row = QHBoxLayout(); save_row.setContentsMargins(0,0,0,0)
        save_row.addWidget(self.save_root, 1); save_row.addWidget(self.save_browse_btn)
        save_box = QWidget(); save_box.setLayout(save_row)
        self.disk = QLabel('-- GB')
        self.sync = QLabel('待机')
        self.sync_detail = QLabel('尚未开始同步质量评估')
        self.sync_detail.setWordWrap(True); self.sync_detail.setObjectName('muted')
        self.elapsed = QLabel('00:00.0')
        sf.addRow('保存根目录', save_box); sf.addRow('磁盘剩余', self.disk); sf.addRow('软件同步状态', self.sync); sf.addRow('同步质量（主机）', self.sync_detail); sf.addRow('采集时长', self.elapsed)
        grid.addWidget(storage_box, 0, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 2)
        cl.addLayout(grid)

        # Named templates save experimental settings only (never Windows camera indices).
        tpl = QGroupBox('常用实验方案 · 参数模板')
        tp = QHBoxLayout(tpl)
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(220)
        self.template_load_btn = QPushButton('读取模板')
        self.template_save_btn = QPushButton('保存当前方案…')
        tp.addWidget(QLabel('方案'))
        tp.addWidget(self.template_combo, 1)
        tp.addWidget(self.template_load_btn)
        tp.addWidget(self.template_save_btn)
        cl.addWidget(tpl)
        self._refresh_templates()

        steps = QGroupBox('采集流程')
        bg = QGridLayout(steps)
        self.probe_btn = QPushButton('① 检测设备')
        self.preflight_btn = QPushButton('② 快速准备'); self.preflight_btn.setEnabled(False)
        self.deep_check_btn = QPushButton('完整硬件诊断');self.deep_check_btn.setEnabled(False);self.deep_check_btn.hide()
        self.full_scan_btn=QPushButton('搜索更多摄像头');self.full_scan_btn.hide()
        self.preview_btn = QPushButton('③ 实时预览'); self.preview_btn.setEnabled(False)
        self.start_btn = QPushButton('④ ● 开始同步采集'); self.start_btn.setObjectName('primary'); self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton('■ 停止采集'); self.stop_btn.setEnabled(False)
        self.discard_btn = QPushButton('× 放弃本次采集'); self.discard_btn.setEnabled(False)
        self.open_btn = QPushButton('打开最近采集目录'); self.open_btn.setEnabled(False)
        self.audit_btn = QPushButton('采集质量验收…')
        bg.addWidget(self.probe_btn,0,0); bg.addWidget(self.preflight_btn,0,1); bg.addWidget(self.preview_btn,0,2); bg.addWidget(self.start_btn,0,3)
        bg.addWidget(self.deep_check_btn,2,0,1,2)
        bg.addWidget(self.full_scan_btn,2,2,1,2)
        bg.addWidget(self.stop_btn,1,0); bg.addWidget(self.discard_btn,1,1); bg.addWidget(self.open_btn,1,2); bg.addWidget(self.audit_btn,1,3)
        for c in range(4): bg.setColumnStretch(c,1)
        cl.addWidget(steps)

        result_box = QGroupBox('步骤结果')
        rg = QGridLayout(result_box)
        self.step_probe = QLabel('① 检测设备：未执行')
        self.step_preflight = QLabel('② 采集预检：未执行')
        self.step_preview = QLabel('③ 实时预览：未执行')
        self.step_record = QLabel('④ 同步采集：未开始')
        for i, lab in enumerate([self.step_probe, self.step_preflight, self.step_preview, self.step_record]):
            lab.setWordWrap(True); lab.setObjectName('muted'); rg.addWidget(lab, i//2, i%2)
        rg.setColumnStretch(0,1); rg.setColumnStretch(1,1)
        cl.addWidget(result_box)
        self.acceptance_status = QLabel('采集验收：录制结束后自动检查视频、帧索引和时间戳；也可手动检查旧 Session。')
        self.acceptance_status.setObjectName('muted')
        self.acceptance_status.setWordWrap(True)
        cl.addWidget(self.acceptance_status)
        pl.addWidget(control)

        prep_note = QLabel('正常只需“检测设备 → 快速准备 → 开始录像”。设备异常才展开高级设置；录像中不会自动降低画质。')
        prep_note.setObjectName('muted'); prep_note.setWordWrap(True); pl.addWidget(prep_note); pl.addStretch()

        # -------------------- Tab 2: simple stage + protocol workflow --------------------
        exp_scroll = QScrollArea()
        exp_scroll.setWidgetResizable(True)
        exp_scroll.setFrameShape(QFrame.NoFrame)
        exp = QWidget()
        el = QVBoxLayout(exp)
        el.setContentsMargins(8, 10, 8, 10)
        el.setSpacing(10)
        exp_scroll.setWidget(exp)
        self.workflow_tabs.addTab(exp_scroll, '② 阶段与参数')

        intro = QLabel('这里只做两件事：① 记录本次刺激参数；② 实验过程中切换 Baseline / 刺激期 / Recovery 等阶段。三路视频始终连续录像，不会因为切换阶段而重新开关摄像头。')
        intro.setObjectName('muted'); intro.setWordWrap(True); el.addWidget(intro)

        # Common protocol parameters: filled once, and snapshotted whenever a stage starts.
        param_box = QGroupBox('实验参数 · 填一次即可，需要变化时再修改')
        pg = QGridLayout(param_box)
        pg.setHorizontalSpacing(14); pg.setVerticalSpacing(8)
        self.exp_condition = QComboBox()
        for key,text in [('Exposure','实刺激 / Exposure'),('Sham','假刺激 / Sham'),('A','盲法 A'),('B','盲法 B'),('Other','其他')]: self.exp_condition.addItem(text,key)
        self.carrier_frequency = QDoubleSpinBox(); self.carrier_frequency.setRange(0,10000000); self.carrier_frequency.setDecimals(3); self.carrier_frequency.setSpecialValueText('未记录'); self.carrier_frequency.setSuffix(' Hz')
        self.modulation_frequency = QDoubleSpinBox(); self.modulation_frequency.setRange(0,100000); self.modulation_frequency.setDecimals(3); self.modulation_frequency.setSpecialValueText('未记录'); self.modulation_frequency.setSuffix(' Hz')
        self.signal_strength = QLineEdit('')
        self.signal_unit = QComboBox(); self.signal_unit.addItems(['mT','µT','V/m','W/m²','W','设备档位','其他'])
        self.exp_waveform = QComboBox(); self.exp_waveform.addItems(['未记录','正弦','方波/脉冲','调制波','其他'])
        self.antenna_distance = QDoubleSpinBox(); self.antenna_distance.setRange(0,10000); self.antenna_distance.setDecimals(1); self.antenna_distance.setSpecialValueText('未记录'); self.antenna_distance.setSuffix(' cm')
        self.custom_parameters = QLineEdit(''); self.custom_parameters.setPlaceholderText('可直接写其他参数，例如：AM 80%、天线型号A、方向水平……')
        pg.addWidget(QLabel('实验条件'),0,0); pg.addWidget(self.exp_condition,0,1)
        pg.addWidget(QLabel('载波频率'),0,2); pg.addWidget(self.carrier_frequency,0,3)
        pg.addWidget(QLabel('调制频率'),1,0); pg.addWidget(self.modulation_frequency,1,1)
        pg.addWidget(QLabel('信号强度'),1,2); pg.addWidget(self.signal_strength,1,3); pg.addWidget(self.signal_unit,1,4)
        pg.addWidget(QLabel('波形'),2,0); pg.addWidget(self.exp_waveform,2,1)
        pg.addWidget(QLabel('天线距离'),2,2); pg.addWidget(self.antenna_distance,2,3)
        pg.addWidget(QLabel('其他参数 / 备注'),3,0); pg.addWidget(self.custom_parameters,3,1,1,4)
        param_tip = QLabel('参数不会单独控制刺激设备；在“开始 / 切换阶段”时自动快照保存。如果下一次刺激参数不同，先改这里，再切换到“刺激期”。')
        param_tip.setObjectName('muted'); param_tip.setWordWrap(True); pg.addWidget(param_tip,4,0,1,5)
        el.addWidget(param_box)

        stage_box = QGroupBox('实验阶段 · 只需切换当前阶段')
        sl = QGridLayout(stage_box)
        sl.setHorizontalSpacing(14); sl.setVerticalSpacing(9)
        self.stage_type = QComboBox()
        for key,text in [('baseline','Baseline / 刺激前'),('stimulation','刺激期'),('recovery','Recovery / 刺激后'),('adaptation','适应/稳定期'),('observation','观察期'),('other','其他阶段')]: self.stage_type.addItem(text,key)
        self.stage_note = QLineEdit(''); self.stage_note.setPlaceholderText('可选，例如：第1轮、动物躁动、设备调整后……')
        self.stage_switch_btn = QPushButton('▶ 开始 / 切换到该阶段'); self.stage_switch_btn.setObjectName('primary')
        self.stage_end_btn = QPushButton('■ 结束当前阶段')
        sw = QHBoxLayout(); sw.addWidget(self.stage_switch_btn,2); sw.addWidget(self.stage_end_btn,1)
        sww = QWidget(); sww.setLayout(sw)
        self.stage_status = QLabel('当前阶段：未开始'); self.stage_status.setObjectName('muted'); self.stage_status.setWordWrap(True)
        sl.addWidget(QLabel('下一阶段'),0,0); sl.addWidget(self.stage_type,0,1)
        sl.addWidget(QLabel('阶段备注'),1,0); sl.addWidget(self.stage_note,1,1)
        sl.addWidget(sww,2,0,1,2); sl.addWidget(self.stage_status,3,0,1,2)
        el.addWidget(stage_box)

        note_box = QGroupBox('临时备注 · 可选')
        nl = QHBoxLayout(note_box)
        self.quick_note = QLineEdit(''); self.quick_note.setPlaceholderText('例如：右上肢明显抽动、人员进入、摄像头短暂遮挡……')
        self.note_add_btn = QPushButton('+ 记录当前时间备注')
        nl.addWidget(self.quick_note,1); nl.addWidget(self.note_add_btn)
        el.addWidget(note_box)

        timeline_card = QFrame(); timeline_card.setObjectName('card'); tl = QVBoxLayout(timeline_card)
        tt = QLabel('实验时间线'); tt.setObjectName('cardtitle'); tl.addWidget(tt)
        note = QLabel('这里只显示阶段和人工备注，避免实时 AI 候选把操作界面刷得很乱。AI 候选仍保存在技术元数据中，实验结束后统一分析。')
        note.setObjectName('muted'); note.setWordWrap(True); tl.addWidget(note)
        self.timeline = QTableWidget(0, 4); self.timeline.setHorizontalHeaderLabels(['Session时间','记录','参数 / 备注','持续']); self.timeline.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.timeline.setMinimumHeight(210); tl.addWidget(self.timeline)
        el.addWidget(timeline_card)

        split_note = QLabel('成熟系统建议：原始视频不要按阶段物理切开。每路只保存一条连续原始视频，阶段边界用时间戳记录；分析时按阶段“虚拟切片”。这样同步更稳定、原始数据不丢失，也便于以后重新定义时间窗。需要分享某一阶段时，再从分析结果中导出片段。')
        split_note.setObjectName('muted'); split_note.setWordWrap(True); el.addWidget(split_note)

        log_card = QFrame(); log_card.setObjectName('card'); ll = QVBoxLayout(log_card)
        lt = QLabel('采集状态'); lt.setObjectName('cardtitle'); ll.addWidget(lt)
        self.log = QLabel('等待开始。'); self.log.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.log.setWordWrap(True); self.log.setObjectName('muted'); self.log.setMinimumHeight(62); ll.addWidget(self.log)
        el.addWidget(log_card); el.addStretch()

        self._refresh_disk()
        self._set_experiment_controls(False)
    def _refresh_templates(self, select_name=None):
        if not hasattr(self, 'template_combo'):
            return
        names = list_templates()
        current = select_name or self.template_combo.currentText()
        self.template_combo.clear()
        self.template_combo.addItems(names)
        if current in names:
            self.template_combo.setCurrentText(current)

    def _template_values(self):
        return {
            'protocol_code': self.protocol_code.text(),
            'session_name': self.session_name.text(),
            'operator': self.operator.text(),
            'quality_schema': 2,
            'resolution': self.resolution.currentIndex(),
            'fps': self.fps.currentIndex(),
            'condition': self.exp_condition.currentData(),
            'carrier_frequency': self.carrier_frequency.value(),
            'modulation_frequency': self.modulation_frequency.value(),
            'signal_strength': self.signal_strength.text(),
            'signal_unit': self.signal_unit.currentText(),
            'waveform': self.exp_waveform.currentText(),
            'antenna_distance': self.antenna_distance.value(),
            'custom_parameters': self.custom_parameters.text(),
        }

    def _save_template(self):
        name, ok = QInputDialog.getText(self, '保存实验方案', '方案名称：', text=self.protocol_code.text().strip() or '常用方案')
        if not ok:
            return
        try:
            path = save_template(name, self._template_values())
            self._refresh_templates(path.stem)
            self._set_step_result(1, '已保存实验模板：%s（不保存设备编号）' % path.stem, 'ok')
        except Exception as e:
            QMessageBox.warning(self, '保存模板失败', str(e))

    def _load_template(self):
        name = self.template_combo.currentText().strip()
        if not name:
            QMessageBox.information(self, '暂无模板', '先填写一次参数，再点“保存当前方案”。')
            return
        if self.controller.running:
            QMessageBox.warning(self, '正在采集', '正式录像期间不能切换实验模板。')
            return
        try:
            values = load_template(name)
            for key, target in [('protocol_code', self.protocol_code),
                                ('session_name', self.session_name), ('operator', self.operator),
                                ('signal_strength', self.signal_strength),
                                ('custom_parameters', self.custom_parameters)]:
                if key in values:
                    target.setText(str(values[key]))
            for key, combo in [('resolution', self.resolution), ('fps', self.fps)]:
                if key in values:
                    i = int(values[key])
                    # 4.7 templates: 0=1080p,1=720p,2=480p.
                    # 4.8 templates: 0=Auto,1=1080p,2=720p,3=480p.
                    if key=='resolution' and int(values.get('quality_schema',1))<2:
                        i = {0:1,1:2,2:3}.get(i,i)
                    if 0 <= i < combo.count():
                        combo.setCurrentIndex(i)
            for key, combo, use_data in [('condition', self.exp_condition, True),
                                         ('signal_unit', self.signal_unit, False),
                                         ('waveform', self.exp_waveform, False)]:
                if key in values:
                    i = combo.findData(values[key]) if use_data else combo.findText(str(values[key]))
                    if i >= 0:
                        combo.setCurrentIndex(i)
            for key, control in [('carrier_frequency', self.carrier_frequency),
                                 ('modulation_frequency', self.modulation_frequency),
                                 ('antenna_distance', self.antenna_distance)]:
                if key in values:
                    control.setValue(float(values[key]))
            self._invalidate_preflight()
            self._set_step_result(1, '已读取方案：%s；摄像头请重新检测并预检。' % name, 'ok')
        except Exception as e:
            QMessageBox.warning(self, '读取模板失败', str(e))

    def _toggle_advanced(self,show):
        self.advanced_fps_row.setVisible(bool(show))
        self.deep_check_btn.setVisible(bool(show))
        self.full_scan_btn.setVisible(bool(show))

    def _adjust_preview_size(self):
        active=[c for c in self.cards if c.isVisible()]
        if not active:return
        usable=max(200,self.width()-len(active)*32-48)
        per=usable/len(active)
        for card in active:
            ratio=card.preview.frame_ratio or 4.0/3.0
            cap=min({1:475,2:365,3:305}.get(len(active),305),max(175,int(self.height()*0.36)))
            expected=int(max(150,min(cap,per/ratio)))
            if card.preview.height()!=expected:
                card.preview.setFixedHeight(expected)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'cards'):
            QTimer.singleShot(0,self._adjust_preview_size)

    def _set_config_enabled(self, enabled):
        widgets = [self.camera_count, self.monkey_id, self.protocol_code, self.session_name, self.operator, self.resolution, self.fps, self.save_browse_btn, self.template_combo, self.template_load_btn, self.template_save_btn]
        for card in self.cards:
            widgets.extend([card.enabled, card.device, card.view])
        for w in widgets:
            w.setEnabled(bool(enabled))

    def _apply_camera_count(self, *args):
        # Unlike fixed three-card layouts, inactive views disappear entirely.
        # Mapping is set only while acquisition is stopped.
        n=int(self.camera_count.currentData() or 1)
        for i,card in enumerate(self.cards):
            show=i<n
            card.setVisible(show)
            card.enabled.blockSignals(True)
            card.enabled.setChecked(show)
            card.enabled.blockSignals(False)
            card.enabled.setVisible(False)   # source count is the single control
        if hasattr(self,'camera_choice_tip'):
            self.camera_choice_tip.setText('已选择 %d 路；请给每一路指定正确设备和拍摄视角。'%n)
        if hasattr(self,'start_btn'):
            self._invalidate_preflight()
        QTimer.singleShot(0,self._adjust_preview_size)

    def _wire(self):
        self.camera_count.currentIndexChanged.connect(self._apply_camera_count)
        self.save_browse_btn.clicked.connect(self._choose_save_root)
        self.template_save_btn.clicked.connect(self._save_template)
        self.template_load_btn.clicked.connect(self._load_template)
        self.deep_check_btn.clicked.connect(lambda:self._preflight(detailed=True))
        self.full_scan_btn.clicked.connect(lambda:self._probe(full=True))
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.probe_btn.clicked.connect(self._probe)
        self.preflight_btn.clicked.connect(lambda:self._preflight(detailed=False))
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.discard_btn.clicked.connect(self._discard)
        self.open_btn.clicked.connect(self._open_session)
        self.audit_btn.clicked.connect(self._audit_session_manual)
        self.stage_switch_btn.clicked.connect(self._switch_stage)
        self.stage_end_btn.clicked.connect(self._end_stage)
        self.note_add_btn.clicked.connect(self._add_note)
        self.preview_controller.frame.connect(self._on_frame)
        self.preview_controller.status.connect(self._on_preview_status)
        self.preview_controller.log.connect(self._append_log)
        self.preview_controller.failed.connect(self._on_preview_failed)
        self.preview_controller.all_opened.connect(self._on_preview_all_opened)
        self.preview_controller.stopped.connect(self._on_preview_stopped)
        self.controller.frame.connect(self._on_frame)
        self.controller.status.connect(self._on_status)
        self.controller.sync_quality.connect(self._on_sync_quality)
        self.controller.alert.connect(self._on_alert)
        self.controller.log.connect(self._append_log)
        self.controller.all_ready.connect(self._on_all_ready)
        self.controller.failed.connect(self._on_failed)
        self.controller.stopped.connect(self._on_stopped)
        self.resolution.currentIndexChanged.connect(self._invalidate_preflight)
        self.fps.currentIndexChanged.connect(self._invalidate_preflight)
        for card in self.cards:
            card.enabled.stateChanged.connect(self._invalidate_preflight)
            card.device.currentIndexChanged.connect(self._invalidate_preflight)

    def _append_log(self, text):
        old = self.log.text()
        lines = (old + '\n' + text).strip().splitlines()[-9:]
        self.log.setText('\n'.join(lines))

    def _set_step_result(self, step, text, state='idle'):
        lab = {1:self.step_probe, 2:self.step_preflight, 3:self.step_preview, 4:self.step_record}.get(int(step))
        if lab is None:
            return
        prefix = {1:'① 检测设备：', 2:'② 采集预检：', 3:'③ 实时预览：', 4:'④ 同步采集：'}[int(step)]
        lab.setText(prefix + str(text))
        color = {'ok':'#43D18B','warn':'#F0B84B','bad':'#FF7070','run':'#62A8FF','idle':'#8299A8'}.get(state,'#8299A8')
        weight = '700' if state in ('ok','warn','bad','run') else '400'
        lab.setStyleSheet('color:%s;font-weight:%s;' % (color, weight))

    def _choose_save_root(self):
        current = Path(self.save_root.text().strip() or CAPTURES_DIR)
        start = current if current.exists() else current.parent
        if not start.exists():
            start = Path.home()
        p = QFileDialog.getExistingDirectory(
            self, '选择采集保存根目录', str(start),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        if p:
            self.save_root.setText(str(Path(p)))
            self.qt_settings.setValue('save_root',str(Path(p)))
            self._refresh_disk()
            self.preflight_signature = None

    def _refresh_disk(self):
        try:
            gb = storage_free_gb(self.save_root.text().strip() or CAPTURES_DIR)
            self.disk.setText('%.1f GB' % gb)
        except Exception:
            self.disk.setText('无法读取')

    def _active_specs(self):
        active_cards = [c for c in self.cards if c.enabled.isChecked()]
        w, h, fps = self._settings()
        specs = []
        for card in active_cards:
            idx = card.selected_device()
            if idx is None:
                continue
            specs.append({
                'slot_id': int(card.slot),
                'device_index': int(idx),
                'view_key': str(card.view.currentData()),
                'view_name': card.view.currentText(),
                'width': int(w), 'height': int(h), 'fps': int(fps), 'fourcc': 'MJPG',
            })
        return specs

    def _current_preflight_signature(self):
        specs = self._active_specs()
        return tuple(sorted((int(x['slot_id']), int(x['device_index']), int(x['width']), int(x['height']), int(x['fps'])) for x in specs))

    def _invalidate_preflight(self, *args):
        self.preflight_signature = None
        if hasattr(self, 'preview_controller') and self.preview_controller.running:
            self.preview_controller.stop(wait=True)
        if hasattr(self, 'start_btn') and not self.controller.running:
            self.start_btn.setEnabled(False)
        if hasattr(self, 'preview_btn'):
            self.preview_btn.setEnabled(False)
            self.preview_btn.setText('③ 实时预览')
        for card in self.cards:
            card.set_preflight('未执行', None)
        if hasattr(self, 'step_preflight'):
            self._set_step_result(2, '配置已变化，请重新预检', 'idle')
            self._set_step_result(3, '等待预检通过', 'idle')
            self._set_step_result(4, '等待预检与实时预览', 'idle')

    def _camera_cache_key(self):
        indices=sorted(int(s['device_index']) for s in self._active_specs())
        return 'last_good_mode_'+'_'.join(str(x) for x in indices)

    def _preflight(self,detailed=False):
        if self.controller.running:
            return
        if self.preview_controller.running:
            self.preview_controller.stop(wait=True)
            self.preview_btn.setText('③ 实时预览')
        if self.resolution.currentIndex()==0 and not detailed:
            raw=str(self.qt_settings.value(self._camera_cache_key(),'')).strip()
            try:
                values=tuple(int(x) for x in raw.split(','))
                if len(values)==3 and values[:2] in ((1920,1080),(1280,720),(640,480)) and values[2] in (30,60):
                    self._auto_selected_mode=values
                else:
                    self._auto_selected_mode=(1280,720,30)
            except (ValueError,TypeError):
                self._auto_selected_mode=(1280,720,30)
        specs = self._active_specs()
        if not specs:
            QMessageBox.warning(self, '未启用摄像头' , '请至少启用 1 路摄像头，并先点击“检测设备”。')
            return
        indices = [int(x['device_index']) for x in specs]
        if len(indices) != len(set(indices)):
            QMessageBox.warning(self, '设备重复', '已启用通道不能选择相同设备。')
            return
        for card in self.cards:
            if card.slot in [int(x['slot_id']) for x in specs]:
                card.set_preflight('正在测试…', None)
            else:
                card.set_preflight('未启用', None)
        self.preflight_signature = None
        self.preflight_btn.setEnabled(False)
        self.deep_check_btn.setEnabled(False)
        self.probe_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self._set_config_enabled(False)
        self.deep_check_btn.setEnabled(False)
        self._set_step_result(2, '正在快速联合检测…' if not detailed else '正在完整诊断…', 'run')
        self._set_step_result(3, '等待预检完成', 'idle')
        self._set_step_result(4, '等待预检完成', 'idle')
        self._append_log('检查中：%s。' % ('逐路加联合诊断' if detailed else '选中相机联合短测试'))
        self.preflight_worker = PreflightWorker(specs, self, detailed=detailed, automatic=self.resolution.currentIndex()==0)
        self.preflight_worker.progress.connect(self._append_log)
        self.preflight_worker.frame.connect(self._on_frame)
        self.preflight_worker.done.connect(self._preflight_done)
        self.preflight_worker.failed.connect(self._preflight_failed)
        self.preflight_worker.start()

    @staticmethod
    def _metric_text(info):
        req_w = int(info.get('requested_width', 0) or 0)
        req_h = int(info.get('requested_height', 0) or 0)
        act_w = int(info.get('width', 0) or 0)
        act_h = int(info.get('height', 0) or 0)
        mode_text = ('目标 %dx%d / 实际 %dx%d' % (req_w, req_h, act_w, act_h)) if req_w and req_h else ('实际 %dx%d' % (act_w, act_h))
        return ('%s | 实测 %.1f/目标 %.1f FPS | P95 %.1f ms | 抖动 %.1f ms | 失败 %d | %s | %s' % (
            info.get('grade_cn', '未知'), float(info.get('observed_fps', 0.0)), float(info.get('requested_fps', 0.0)),
            float(info.get('p95_interval_ms', 0.0)), float(info.get('jitter_ms', 0.0)), int(info.get('read_failures', 0)),
            mode_text, info.get('fourcc', '未知')
        ))

    def _preflight_done(self, result):
        self.preflight_btn.setEnabled(True)
        self.deep_check_btn.setEnabled(True)
        self.probe_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self._set_config_enabled(True)
        specs = self._active_specs()
        slot_by_device = {int(x['device_index']): int(x['slot_id']) for x in specs}
        single = result.get('single') or {}
        warnings = []
        for slot_text, info in single.items():
            try:
                slot = int(slot_text)
                grade = info.get('grade', 'poor')
                state = True if grade == 'stable' else ('warning' if grade == 'degraded' else False)
                text = '逐路 ' + self._metric_text(info)
                rec = info.get('recommendation') or {}
                recommended = rec.get('recommended')
                if recommended and grade != 'stable':
                    text += '；推荐 %dx%d @ %d FPS %s' % (
                        int(recommended['width']), int(recommended['height']), int(recommended['fps']), recommended.get('fourcc', 'MJPG'))
                self.cards[slot].set_preflight(text, state)
                if grade == 'degraded':
                    warnings.append('第%d路逐路测试为“可用但降级”' % (slot + 1))
            except Exception:
                pass

        group = result.get('group') or {}
        for idx_text, info in (group.get('results') or {}).items():
            try:
                idx = int(idx_text); slot = slot_by_device.get(idx)
                if slot is not None:
                    grade = info.get('grade', 'poor')
                    state = True if grade == 'stable' else ('warning' if grade == 'degraded' else False)
                    self.cards[slot].set_preflight('联合 %s · %dx%d · %.1f FPS' % (info.get('grade_cn','未知'),int(info.get('width',0)),int(info.get('height',0)),float(info.get('observed_fps',0))), state)
                    if grade == 'degraded':
                        warnings.append('第%d路联合测试为“可用但降级”' % (slot + 1))
            except Exception:
                pass

        details = []
        source = (group.get('results') or {}) if group else {}
        for idx_text, info in source.items():
            details.append('设备 %s：%s' % (idx_text, self._metric_text(info)))
        if details:
            self._append_log('联合测试结果：' + '；'.join(details))

        if result.get('ok'):
            mode=result.get('mode')
            if mode and self.resolution.currentIndex()==0:
                self._auto_selected_mode=tuple(mode)
                key=self._camera_cache_key()
                self.qt_settings.setValue(key,'%d,%d,%d'%tuple(mode))
                if tuple(mode)!=(1280,720,30):
                    self._append_log('自动选定当前可用模式：%d×%d @%d FPS；正式录像前不会再自动变更。'%tuple(mode))
            self.preflight_signature = self._current_preflight_signature()
            self.start_btn.setEnabled(True)
            self.preview_btn.setEnabled(True)
            if warnings:
                msg = '采集预检可用，但存在性能降级：\n\n' + '\n'.join(sorted(set(warnings))) + \
                      '\n\n建议优先按推荐模式调整。程序接下来会进入实时预览，用真实连续画面再次确认三路相机。'
                self._set_step_result(2, '可用，但存在降级通道', 'warn')
                self._append_log(msg.replace('\n', ' '))
                self._append_log('注意：联合模式有性能降级，建议在高级设置中处理。')
            else:
                w,h,fps=self._settings()
                self._set_step_result(2, '通过 · %dx%d @ %d FPS · %d路' % (w,h,fps,len(specs)), 'ok')
                self._append_log('快速联合预检通过；已进入真实预览。' if result.get('quick') else '完整联合预检通过。')
            QTimer.singleShot(650, lambda: self._start_live_preview(auto=True))
            return

        self.preflight_signature = None
        reason = result.get('reason', '采集预检失败。')
        if result.get('stage') == 'group' and single and all(x.get('grade') != 'poor' for x in single.values()):
            reason += '\n\n逐路均可用，但联合开启失败：高度疑似 USB Hub / USB 控制器共享带宽、供电或驱动争用。建议分散到不同物理USB口，或降低分辨率后重新预检。'
        self._set_step_result(2, '未通过：' + reason.split('\n')[0], 'bad')
        self._append_log('采集预检失败：' + reason.replace('\n', ' '))
        QMessageBox.warning(self, '当前视频质量无法稳定采集', reason+'\n\n可改用自动模式；若仍失败，请在“高级摄像设置”中进行完整硬件诊断。')

    def _preflight_failed(self, msg):
        self.preflight_signature = None
        self.preflight_btn.setEnabled(True)
        self.deep_check_btn.setEnabled(True)
        self.probe_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self._set_config_enabled(True)
        self._set_step_result(2, '失败：' + str(msg).split('\n')[0], 'bad')
        QMessageBox.warning(self, '采集预检失败', msg)

    def _probe(self,full=False):
        if self.controller.running:
            return
        if self.preview_controller.running:
            self.preview_controller.stop(wait=True)
        self.preflight_signature = None
        self.start_btn.setEnabled(False)
        self.probe_btn.setEnabled(False)
        self._set_step_result(1, '正在扫描并验证真实视频流…', 'run')
        self._append_log('正在扫描设备 0–%d，并验证是否能读取真实画面…' % (11 if full else 5))
        self.probe_worker = ProbeWorker(self,max_index=12 if full else 6)
        self.probe_worker.done.connect(self._probe_done)
        self.probe_worker.failed.connect(self._probe_failed)
        self.probe_worker.start()

    def _probe_done(self, found):
        self.probe_btn.setEnabled(True)
        if not found:
            self.preflight_btn.setEnabled(False)
            self._set_step_result(1, '未发现可读取摄像头', 'bad')
            self._append_log('未检测到可读取摄像头。')
            QMessageBox.warning(self, '未发现摄像头', '未检测到可读取的摄像头。请检查 Windows 相机权限、USB 连接和设备占用。')
            return
        desc = []
        for x in found:
            desc.append('设备 %d：%dx%d / %.1f FPS / %s' % (
                x['index'], x['width'], x['height'], x['fps'], x.get('backend', '')
            ))
        # Only list devices that really returned frames. Assign them to cards sequentially.
        self.preflight_signature = None
        self.preview_btn.setEnabled(False)
        self.preview_btn.setText('③ 实时预览')
        for card in self.cards:
            card.populate_devices(found)
            card.enabled.setChecked(False)
            card.set_preflight('未执行', None)
        recommended=min(3,len(found))
        self.camera_count.blockSignals(True)
        self.camera_count.setCurrentIndex(recommended-1)
        self.camera_count.blockSignals(False)
        for i, info in enumerate(found[:3]):
            card = self.cards[i]
            for j in range(card.device.count()):
                if card.device.itemData(j) == int(info['index']):
                    card.device.setCurrentIndex(j)
                    break
            card.enabled.setChecked(True)
        self._apply_camera_count()
        summary = '发现 %d 路可读取设备：%s' % (len(found), '；'.join('设备%d %dx%d %.1fFPS' % (x['index'], x['width'], x['height'], x['fps']) for x in found))
        self._set_step_result(1, summary, 'ok')
        self.preflight_btn.setEnabled(True)
        self.deep_check_btn.setEnabled(True)
        self._set_step_result(2, '等待采集预检', 'idle')
        self._set_step_result(3, '等待采集预检', 'idle')
        self._set_step_result(4, '等待采集预检', 'idle')
        self._append_log('检测到 %d 路可读取视频设备：\n%s' % (len(found), '\n'.join(desc)))

    def _probe_failed(self, msg):
        self.probe_btn.setEnabled(True)
        self.preflight_btn.setEnabled(False)
        self._set_step_result(1, '检测失败：' + str(msg).split('\n')[0], 'bad')
        QMessageBox.warning(self, '检测失败', msg)

    def _toggle_preview(self):
        if self.controller.running:
            return
        if self.preview_controller.running:
            self.preview_controller.stop(wait=True)
            self.preview_btn.setText('③ 实时预览')
            return
        self._start_live_preview(auto=False)

    def _start_live_preview(self, auto=False):
        if self.controller.running or self.preview_controller.running:
            return
        if self.preflight_signature != self._current_preflight_signature():
            if not auto:
                QMessageBox.warning(self, '请先预检', '实时预览前请先完成当前配置的采集预检。')
            return
        specs = self._active_specs()
        if not specs:
            return
        self.preview_btn.setEnabled(True)
        self.preview_btn.setText('停止实时预览')
        self.start_btn.setEnabled(True)
        self._append_log('进入持续实时预览：这里显示的是实时视频流，不是预检截图。')
        self.preview_controller.start(specs)

    def _on_preview_all_opened(self):
        self.preview_btn.setText('停止实时预览')
        self.sync.setText('实时预览中（尚未录像）')
        self._set_step_result(3, '已打开 %d 路持续实时画面，请移动镜头确认画面变化' % len(self._active_specs()), 'ok')
        self._set_step_result(4, '已可开始同步采集', 'idle')

    def _on_preview_status(self, slot, info):
        if 0 <= int(slot) < len(self.cards):
            self.cards[int(slot)].set_status(info)

    def _on_preview_failed(self, message):
        self.preflight_signature = None
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setText('③ 实时预览')
        self.sync.setText('实时预览失败')
        self._set_step_result(3, '失败：' + str(message).split('\n')[0], 'bad')
        self._set_step_result(4, '不可开始，请重新预检', 'bad')
        QMessageBox.warning(self, '实时预览失败', message + '\n\n预检后持续预览仍失败，说明该配置暂时不够稳定，请重新预检或调整USB接口/采集模式。')

    def _on_preview_stopped(self):
        self.preview_btn.setText('③ 实时预览')
        if not self.controller.running and self.preflight_signature == self._current_preflight_signature():
            self.preview_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.sync.setText('预览已停止，等待采集')
            self._set_step_result(3, '已停止；预检结果仍有效', 'idle')
            self._set_step_result(4, '已可开始同步采集', 'idle')
        if self._pending_start is not None:
            # Give DirectShow/UVC drivers a short, explicit release interval before
            # reopening for formal recording. This avoids the common "preflight
            # passed, recording reopen immediately fails" race.
            QTimer.singleShot(800, self._start_pending_recording)

    def _settings(self):
        choice=self.resolution.currentIndex()
        if choice == 0:
            return tuple(self._auto_selected_mode)
        w,h={1:(1920,1080),2:(1280,720),3:(640,480)}.get(choice,(1280,720))
        fps = 30 if self.fps.currentIndex()==0 else 60
        return w,h,fps

    @staticmethod
    def _safe_name(text):
        text = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', text.strip())
        return text[:50] or 'session'

    def _start(self):
        if self.controller.running:
            return
        active_cards = [c for c in self.cards if c.enabled.isChecked()]
        if not active_cards:
            QMessageBox.warning(self, '未启用摄像头', '请至少启用 1 路摄像头。建议先点击“检测设备”。')
            return
        specs = self._active_specs()
        indices = [int(x['device_index']) for x in specs]
        if len(specs) != len(active_cards):
            QMessageBox.warning(self, '设备无效', '已启用通道中存在无效设备。请重新检测设备。')
            return
        if len(set(indices)) != len(indices):
            QMessageBox.warning(self, '设备重复', '已启用的摄像头不能选择相同设备。')
            return
        if self.preflight_signature != self._current_preflight_signature():
            QMessageBox.warning(
                self, '请先进行采集预检',
                '正式采集前请点击“② 快速准备”，验证全部启用摄像头能够同时稳定出帧。'
            )
            return
        if any('可用但降级' in c.preflight.text() for c in active_cards):
            if QMessageBox.question(
                self, '存在降级通道',
                '预检中至少有一路为“可用但降级”。对于短促抽搐实验，稳定帧率非常重要。\n\n'
                '更建议先按推荐模式调整并重新预检。是否仍使用当前设置开始采集？'
            ) != QMessageBox.Yes:
                return
        self.active_slots = set(int(x['slot_id']) for x in specs)
        w, h, fps = self._settings()
        if self.resolution.currentIndex()==0 and (w < 1280 or h < 720 or fps < 30):
            if QMessageBox.question(self,'确认采集画质',
                    '联合验证的稳定模式为 %d×%d @%d FPS。\n\n该规格可能不适合细微眼部或短促震颤分析。仍以此画质开始正式录像吗？'%(w,h,fps)) != QMessageBox.Yes:
                return
        root = Path(self.save_root.text().strip() or CAPTURES_DIR)
        try:
            free_gb = storage_free_gb(root)
        except Exception as e:
            QMessageBox.warning(self, '保存目录不可用', str(e))
            return
        if free_gb < 10:
            if QMessageBox.question(self, '磁盘空间较少', '当前磁盘仅剩 %.1f GB。三路 1080p MJPEG 录像可能占用较大空间，仍然开始吗？' % free_gb) != QMessageBox.Yes:
                return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        monkey = self._safe_name(self.monkey_id.text())
        sess = self._safe_name(self.session_name.text())
        protocol = self._safe_name(self.protocol_code.text())
        # Never overwrite a previous experiment even when two starts share a second.
        base_dir = root / ('%s_%s_%s_%s' % (stamp, monkey, protocol, sess))
        try:
            root.mkdir(parents=True, exist_ok=True)
            session_dir = None
            for candidate_number in range(1000):
                candidate = base_dir if candidate_number == 0 else root / (base_dir.name + '_%03d' % candidate_number)
                try:
                    candidate.mkdir(exist_ok=False)
                    session_dir = candidate
                    break
                except FileExistsError:
                    continue
            if session_dir is None:
                raise OSError('同名 Session 数量过多，请修改 Session 名称后重试。')
        except OSError as e:
            QMessageBox.warning(self, '无法创建实验文件夹', str(e))
            return
        meta = {
            'product': 'PrimateBehaviorAI Studio',
            'version': '4.8.0',
            'active_camera_count': len(specs),
            'monkey_id': self.monkey_id.text().strip(),
            'protocol_code': self.protocol_code.text().strip(),
            'operator': self.operator.text().strip(),
            'session_design': 'continuous_recording_with_stage_timestamps',
            'session_name': self.session_name.text().strip(),
            'requested_width': w,
            'requested_height': h,
            'requested_fps': fps,
            'planned_duration_s': 0,
            'session_stop_mode': 'manual_unlimited',
            'sync_mode': 'software_monotonic_timestamp_nearest_frame_qc',
            'note': '连续原始录像；实验阶段使用时间戳标记并在分析时虚拟切片。参数只用于记录，不直接控制刺激硬件；同步质量为主机接收帧时间戳指标，不等同于传感器曝光同步。',
        }
        self.timeline.setRowCount(0)
        self._disk_emergency_stop=False
        self.acceptance_status.setText('采集验收：录制中，结束后自动检测。')
        self.latest_status = {}
        self.current_session_dir = session_dir
        self._pending_start = (session_dir, specs, meta)
        self._set_step_result(4, '正在释放预览并逐路打开摄像头…', 'run')
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.probe_btn.setEnabled(False)
        self.preflight_btn.setEnabled(False)
        self._set_config_enabled(False)
        if self.preview_controller.running:
            self.sync.setText('正在关闭实时预览并释放摄像头句柄…')
            self._append_log('正式录像前先安全关闭实时预览，等待 Windows 释放摄像头句柄。')
            self.preview_controller.stop(wait=True)
        else:
            QTimer.singleShot(250, self._start_pending_recording)

    def _start_pending_recording(self):
        if self._pending_start is None or self.controller.running:
            return
        session_dir, specs, meta = self._pending_start
        self._pending_start = None
        try:
            self.controller.start(session_dir, specs, meta)
            self.experiment_logger = ExperimentSessionLogger(session_dir, operator=self.operator.text().strip())
        except Exception as e:
            self._set_config_enabled(True)
            self.probe_btn.setEnabled(True)
            self.preflight_btn.setEnabled(True)
            self.preview_btn.setEnabled(self.preflight_signature == self._current_preflight_signature())
            QMessageBox.critical(self, '无法开始采集', str(e))
            return
        self.stop_btn.setEnabled(True)
        self.discard_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.sync.setText('正在逐路重新打开 %d 路摄像头…' % len(specs))
        self.sync_detail.setText('等待各路首批时间戳…')
        self._elapsed_s = 0.0
        self.elapsed.setText('00:00.0')

    def _on_all_ready(self):
        self.recording_started = True
        self._elapsed_s = 0.0
        self.elapsed_timer.start()
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.discard_btn.setEnabled(True)
        self.sync.setText('软件同步采集中' if len(self.active_slots) > 1 else '单路采集中')
        self._set_step_result(4, '已开始 %d 路采集，Session 时间轴运行中' % len(self.active_slots), 'ok')
        if len(self.active_slots) <= 1:
            self.sync_detail.setText('单路采集，无需跨机同步质量评估。')
        self._set_experiment_controls(True)
        if self.experiment_logger is not None:
            self.experiment_logger._event(0.0,'system','Session开始','start',parameters={'camera_count':len(self.active_slots)})
        self._add_timeline_row(0.0,'Session开始','%d 路同步采集' % len(self.active_slots),'')
        if hasattr(self, 'workflow_tabs'):
            self.workflow_tabs.setCurrentIndex(1)

    def _stop(self):
        self._finalize_experiment_metadata('session_stop')
        self.elapsed_timer.stop()
        self.stop_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.sync.setText('正在停止采集并安全关闭文件…')
        self.controller.stop(discard=False)

    def _discard(self):
        if not self.controller.running:
            return
        if QMessageBox.question(
            self, '放弃本次采集',
            '确定停止并删除本次采集吗？\n\n本次 Session 的视频、时间戳和索引文件都会删除，无法恢复。'
        ) != QMessageBox.Yes:
            return
        self._set_experiment_controls(False)
        self.elapsed_timer.stop()
        self.stop_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.sync.setText('正在停止并删除本次采集…')
        self.controller.stop(discard=True)

    def _on_frame(self, slot, frame):
        if 0 <= int(slot) < len(self.cards):
            self.cards[int(slot)].set_frame(frame)
            if int(slot)==0:
                self._preview_resize_counter=getattr(self,'_preview_resize_counter',0)+1
                if self._preview_resize_counter%45==1:self._adjust_preview_size()

    def _on_status(self, slot, info):
        slot = int(slot)
        self.latest_status[slot] = info
        if 0 <= slot < len(self.cards):
            self.cards[slot].set_status(info)
        if len(self.active_slots) == 1:
            self.sync.setText('单路采集 | 无需跨机同步')

    def _on_sync_quality(self, info):
        if len(self.active_slots) <= 1:
            return
        current = float(info.get('current_spread_ms', 0.0))
        mean = float(info.get('mean_spread_ms', 0.0))
        p95 = float(info.get('p95_spread_ms', 0.0))
        maximum = float(info.get('max_spread_ms', 0.0))
        drift = float(info.get('drift_ms', 0.0))
        grade = str(info.get('grade_cn', ''))
        self.sync.setText('%d路软件同步采集中 | %s' % (len(self.active_slots), grade or '评估中'))
        self.sync_detail.setText(
            '当前 %.1f ms | 平均 %.1f ms | P95 %.1f ms | 最大 %.1f ms | 漂移 %.1f ms' %
            (current, mean, p95, maximum, drift)
        )
        if grade == '很好':
            self.sync_detail.setStyleSheet('color:#43D18B;font-weight:700;')
        elif grade == '良好':
            self.sync_detail.setStyleSheet('color:#43D18B;')
        elif grade == '可用':
            self.sync_detail.setStyleSheet('color:#F0B84B;font-weight:700;')
        else:
            self.sync_detail.setStyleSheet('color:#FF7070;font-weight:700;')

    def _session_time(self):
        try:
            value = self.controller.session_time_s()
            if value is not None: return float(value)
        except Exception: pass
        if self.latest_status:
            try: return max(float(x.get('session_time_s',0.0)) for x in self.latest_status.values())
            except Exception: pass
        return float(self._elapsed_s)

    @staticmethod
    def _format_session_time(seconds):
        seconds=max(0.0,float(seconds)); mins=int(seconds//60); secs=seconds-mins*60
        return '%02d:%05.2f' % (mins,secs)

    def _add_timeline_row(self, session_time_s, record, detail='', duration=''):
        row=self.timeline.rowCount(); self.timeline.insertRow(row)
        for c,v in enumerate([self._format_session_time(session_time_s),record,detail,duration]):
            self.timeline.setItem(row,c,QTableWidgetItem(str(v)))
        self.timeline.scrollToBottom()

    def _protocol_params(self):
        carrier=float(self.carrier_frequency.value())
        modulation=float(self.modulation_frequency.value())
        distance=float(self.antenna_distance.value())
        return {
            'condition': str(self.exp_condition.currentData()),
            'condition_name': self.exp_condition.currentText(),
            'carrier_frequency_hz': '' if carrier <= 0 else carrier,
            'modulation_frequency_hz': '' if modulation <= 0 else modulation,
            'signal_strength': self.signal_strength.text().strip(),
            'signal_unit': self.signal_unit.currentText(),
            'waveform': self.exp_waveform.currentText(),
            'antenna_distance_cm': '' if distance <= 0 else distance,
            'custom_parameters': self.custom_parameters.text().strip(),
        }

    def _set_protocol_enabled(self, enabled):
        for w in [self.exp_condition,self.carrier_frequency,self.modulation_frequency,self.signal_strength,
                  self.signal_unit,self.exp_waveform,self.antenna_distance,self.custom_parameters]:
            w.setEnabled(bool(enabled))

    def _set_experiment_controls(self, recording):
        recording=bool(recording)
        active=bool(self.experiment_logger is not None and self.experiment_logger.active_stage is not None)
        self.stage_switch_btn.setEnabled(recording)
        self.stage_end_btn.setEnabled(recording and active)
        self.stage_type.setEnabled(recording)
        self.stage_note.setEnabled(recording)
        # Protocol parameters may be prepared before recording and edited between stages.
        # A running stage already owns the snapshot taken at its start.
        self._set_protocol_enabled(True)
        self.quick_note.setEnabled(recording)
        self.note_add_btn.setEnabled(recording)

    def _stage_detail(self, params):
        parts=[]
        cond=params.get('condition_name','')
        if cond: parts.append(cond)
        c=params.get('carrier_frequency_hz','')
        m=params.get('modulation_frequency_hz','')
        if c not in ('',None): parts.append('载波 %.3g Hz' % float(c))
        if m not in ('',None): parts.append('调制 %.3g Hz' % float(m))
        if params.get('signal_strength'): parts.append('%s %s' % (params.get('signal_strength'),params.get('signal_unit','')))
        if params.get('antenna_distance_cm') not in ('',None): parts.append('距离 %.1f cm' % float(params.get('antenna_distance_cm')))
        if params.get('waveform') and params.get('waveform')!='未记录': parts.append(str(params.get('waveform')))
        return ' | '.join(parts)

    def _switch_stage(self):
        if not self.recording_started or self.experiment_logger is None: return
        t=self._session_time(); key=str(self.stage_type.currentData()); name=self.stage_type.currentText(); note=self.stage_note.text().strip()
        params=self._protocol_params()
        try:
            ended, active=self.experiment_logger.start_stage(t,key,name,params,note)
        except Exception as e:
            QMessageBox.warning(self,'无法切换阶段',str(e)); return
        if ended is not None:
            self._add_timeline_row(t, str(ended.get('stage_name','阶段'))+' 结束', '', '%.1f s' % float(ended.get('duration_s',0.0)))
        detail=self._stage_detail(params)
        if note: detail=(detail+' | ' if detail else '')+note
        self._add_timeline_row(t,name+' 开始',detail,'')
        self.stage_status.setText('当前阶段：%s（%s 开始）' % (name,self._format_session_time(t)))
        self.stage_note.clear()
        self._set_experiment_controls(True)

    def _end_stage(self):
        if not self.recording_started or self.experiment_logger is None or self.experiment_logger.active_stage is None: return
        t=self._session_time(); name=self.experiment_logger.active_stage.get('stage_name','阶段')
        row=self.experiment_logger.end_stage(t,'manual'); dur=float(row.get('duration_s',0.0)) if row else 0.0
        self.stage_status.setText('当前阶段：未开始（上一阶段 %s 已结束）' % name)
        self._add_timeline_row(t,name+' 结束','', '%.1f s' % dur)
        self._set_experiment_controls(True)

    def _add_note(self):
        if not self.recording_started or self.experiment_logger is None: return
        text=self.quick_note.text().strip()
        if not text:
            QMessageBox.information(self,'备注为空','先输入一条简短备注再记录。'); return
        t=self._session_time(); self.experiment_logger.add_note(t,text); self._add_timeline_row(t,'人工备注',text,''); self.quick_note.clear()

    def _finalize_experiment_metadata(self, reason):
        self._set_experiment_controls(False)
        if self.experiment_logger is None: return
        t=self._session_time()
        try:
            summary=self.experiment_logger.finalize(t,reason)
            if self.controller.session_meta is not None:
                self.controller.session_meta['experiment_summary']=summary; self.controller.session_meta['experiment_end_session_time_s']=float(t)
        except Exception as e: self._append_log('实验事件元数据收尾失败：%s' % e)

    def _on_alert(self, slot, info):
        # Keep live AI candidates out of the operator timeline to reduce visual clutter.
        # They remain available in technical metadata for later review.
        t=float(info.get('time_s',0.0)); motion=float(info.get('motion_score',0.0))
        if self.experiment_logger is not None:
            self.experiment_logger.add_ai_candidate(t,int(slot)+1,info.get('type','快速运动候选'),motion,float(info.get('robust_z',0.0)))

    def _on_failed(self, message):
        self._finalize_experiment_metadata('capture_failure')
        self.elapsed_timer.stop()
        self._set_experiment_controls(False)
        self.preflight_signature = None
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self._pending_start = None
        self.probe_btn.setEnabled(True)
        self.preflight_btn.setEnabled(True)
        self._set_config_enabled(True)
        self.sync.setText('失败')
        self.sync_detail.setText('同步质量评估中止。')
        self._set_step_result(4, '采集失败：' + str(message).split('\n')[0], 'bad')
        try:
            self.controller.stop(discard=False)
        except Exception:
            pass
        QMessageBox.critical(self, '摄像头采集失败', message + '\n\n已经打开的摄像头会被停止；已采集到的数据与事件记录会尽量安全保留。')

    def _on_stopped(self, meta):
        self.elapsed_timer.stop()
        self._set_experiment_controls(False)
        self.recording_started = False
        self.stage_status.setText('当前阶段：未开始')
        self.preflight_signature = None
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.probe_btn.setEnabled(True)
        self.preflight_btn.setEnabled(True)
        self._set_config_enabled(True)
        discarded = bool((meta or {}).get('discarded'))
        if discarded:
            delete_error = (meta or {}).get('discard_delete_error')
            self.open_btn.setEnabled(False)
            self.sync.setText('已停止（本次数据已放弃）' if not delete_error else '已停止（删除不完整）')
            self.acceptance_status.setText('采集验收：本次 Session 已放弃，不生成质量报告。')
            self._set_step_result(4, '已停止并放弃本次 Session' if not delete_error else '已停止，但数据删除不完整', 'warn' if delete_error else 'idle')
            self.sync_detail.setText('本次采集已放弃，质量文件随 Session 一并删除。')
            old_dir = self.current_session_dir
            self.current_session_dir = None
            if delete_error:
                QMessageBox.warning(self, '已停止，但删除失败', '录像已经停止，但本次目录未能完全删除：\n%s\n\n目录：%s' % (delete_error, old_dir))
            else:
                QMessageBox.information(self, '已放弃本次采集', '录像已经安全停止，本次 Session 数据已删除。')
        else:
            self.open_btn.setEnabled(bool(self.current_session_dir))
            self.sync.setText('已停止（数据已保存）')
            self._set_step_result(4, '已停止，%d 路数据与实验时间线已保存' % len(self.active_slots), 'ok')
            sq = (meta or {}).get('software_sync_quality', {}) or {}
            if sq:
                self.sync_detail.setText('最终：平均 %.1f ms | P95 %.1f ms | 最大 %.1f ms | 漂移 %.1f ms | %s' % (
                    float(sq.get('mean_spread_ms', 0.0)), float(sq.get('p95_spread_ms', 0.0)),
                    float(sq.get('max_spread_ms', 0.0)), float(sq.get('drift_ms', 0.0)), sq.get('grade_cn', '')
                ))
            QMessageBox.information(
                self, '采集已停止',
                '%d 路连续原始视频、session_record.csv 和 session.json 已保存。\n技术时间戳与质量文件已整理到 _metadata 文件夹。\n\n目录：\n%s\n\n后续分析会直接按阶段时间戳进行虚拟切片，无需手工切视频。' % (len(self.active_slots), self.current_session_dir)
            )
        for card in self.cards:
            if card.enabled.isChecked():
                card.set_preflight('下次采集前请重新预检', None)
        self.experiment_logger = None
        self._refresh_disk()
        if not discarded and self.current_session_dir:
            self._launch_audit(self.current_session_dir, auto=True)

    def _audit_session_manual(self):
        if self.controller.running or self.recording_started:
            QMessageBox.information(self, '正在录像', '请先停止录制，再执行文件级采集验收。')
            return
        source = self.current_session_dir
        if not source or not (Path(source) / 'session.json').exists():
            path = QFileDialog.getExistingDirectory(self, '选择需要验收的实验 Session',
                    str(self.save_root.text().strip() or CAPTURES_DIR))
            if not path:
                return
            source = Path(path)
        self._launch_audit(source)

    def _launch_audit(self, session_dir, auto=False):
        if self.audit_worker is not None and self.audit_worker.isRunning():
            return
        self.audit_btn.setEnabled(False)
        self.acceptance_status.setText('采集验收：正在检查录像可读性和逐帧时间戳…')
        self.audit_worker = SessionAuditWorker(session_dir, self)
        self.audit_worker.completed.connect(lambda result: self._audit_completed(result, auto))
        self.audit_worker.failed.connect(self._audit_failed)
        self.audit_worker.finished.connect(lambda: self.audit_btn.setEnabled(True))
        self.audit_worker.start()

    def _audit_completed(self, result, auto):
        outcome = result.get('overall_status', '未知')
        self.acceptance_status.setText('采集验收：%s（%d路） · 已生成采集验收报告.html' % (
            outcome, int(result.get('camera_count', 0))))
        report = Path(result['session_dir']) / '采集验收报告.html'
        if not auto or outcome != '通过':
            msg = '检查结果：%s\n报告：%s' % (outcome, report)
            if result.get('errors') or result.get('warnings'):
                details = (result.get('errors', []) + result.get('warnings', []))[:4]
                msg += '\n\n' + '\n'.join(str(x) for x in details)
            QMessageBox.warning(self, '实验采集验收', msg) if outcome != '通过' else QMessageBox.information(self, '实验采集验收', msg)
        if not auto and report.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report.resolve())))

    def _audit_failed(self, message):
        self.acceptance_status.setText('采集验收：未完成（%s）' % message)
        QMessageBox.warning(self, '采集验收失败', message)

    def _tick(self):
        if not self.recording_started:
            return
        now=time.monotonic()
        if now-self._last_disk_warn>5.0:
            self._last_disk_warn=now
            try:
                left=storage_free_gb(self.save_root.text().strip() or CAPTURES_DIR)
                self.disk.setText('%.1f GB' % left)
                if left < 3.0:
                    self.disk.setStyleSheet('color:#FF7070;font-weight:700;')
                if left < 1.0 and not self._disk_emergency_stop:
                    # Stop all channels safely before exhausting free space.
                    self._disk_emergency_stop = True
                    self._append_log('磁盘不足 1 GB：已启动安全停止，以保护已经写入的数据。')
                    self._set_step_result(4, '磁盘空间不足，安全停止采集并保留数据', 'bad')
                    self._stop()
                    QTimer.singleShot(0, lambda: QMessageBox.warning(self, '磁盘空间不足', '存储盘剩余空间不足 1 GB，程序已安全停止并保留本次录像。请更换保存盘后重新开始。'))
                    return
                if left < 3.0 and int(now) % 20 < 5:
                    self._append_log('磁盘空间警告：仅剩 %.1f GB。' % left)
            except Exception: pass
        if self.latest_status:
            self._elapsed_s = max(float(x.get('session_time_s', 0.0)) for x in self.latest_status.values())
        mins = int(self._elapsed_s // 60)
        secs = self._elapsed_s - mins * 60
        self.elapsed.setText('%02d:%04.1f' % (mins, secs))
        if self.experiment_logger is not None and self.experiment_logger.active_stage is not None:
            a=self.experiment_logger.active_stage; st=float(a.get('start',{}).get('session_time_s',self._elapsed_s)); dur=max(0.0,self._elapsed_s-st)
            self.stage_status.setText('当前阶段：%s | 已进行 %.1f s' % (a.get('stage_name','阶段'),dur))
            self.stage_status.setStyleSheet('color:#43D18B;font-weight:700;')

    def _open_session(self):
        if self.current_session_dir and Path(self.current_session_dir).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.current_session_dir).resolve())))

    def shutdown(self):
        if self.preflight_worker is not None and self.preflight_worker.isRunning():
            self.preflight_worker.wait(7000)
        if self.preview_controller.running:
            self.preview_controller.stop(wait=True)
        if self.controller.running:
            self._finalize_experiment_metadata('app_shutdown')
            self.controller.stop()
            self.controller.wait_for_stop(8000)
