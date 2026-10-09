from __future__ import annotations
import os, sys, time, traceback, json
from pathlib import Path
import pandas as pd
from PySide6.QtCore import Qt,QThread,Signal,QUrl,QTimer
from PySide6.QtGui import QIcon,QDesktopServices,QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QLineEdit,QFileDialog,
    QStackedWidget,QFrame,QButtonGroup,QRadioButton,QSlider,QCheckBox,QComboBox,QProgressBar,QPlainTextEdit,QTableWidget,
    QTableWidgetItem,QHeaderView,QMessageBox,QAbstractItemView,QGroupBox,QFormLayout,QSpinBox,QDoubleSpinBox,QSplitter,QScrollArea)
try:
    from PySide6.QtMultimedia import QMediaPlayer,QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_MEDIA=True
except Exception:HAVE_MEDIA=False
from .models import AnalysisConfig
from .analysis_service import run_analysis
from .database import Database
from .calibration import CalibrationDialog
from .backends import probe_backends
from .paths import APP_HOME, DATA_DIR, MODELS_DIR, DB_PATH, CAPTURES_DIR, ensure_app_dirs, migrate_legacy_database
from .multicam_page import MultiCamPage
from .diagnostics import build_diagnostic_archive

APP='PrimateBehaviorAI · 动物行为实验平台'; VERSION='4.8.0 · Desktop Edition'
ROOT=Path(__file__).resolve().parents[1]
ensure_app_dirs()
migrate_legacy_database()
WORK=DATA_DIR
DB=Database(DB_PATH)

EVENT_CN={'twitch-like':'抽搐样短促运动','tremor-like':'震颤样重复运动','convulsion-like':'全身同步异常运动','sway-like':'身体摆动/姿势不稳','eye-state-change-like':'眼部状态变化','hypoactivity-like':'低活动/困倦相关','unknown-novelty':'未知新颖行为'}
REGION_CN={'whole_body':'全身','head':'头部','left_eye':'左眼','right_eye':'右眼','torso':'躯干','left_upper_limb':'左上肢','right_upper_limb':'右上肢','left_lower_limb':'左下肢','right_lower_limb':'右下肢','unknown':'未知'}

QSS='''
*{font-family:"Microsoft YaHei UI";font-size:13px;} QMainWindow,QWidget{background:#071018;color:#EAF2F7;}
QFrame#sidebar{background:#0A151F;border-right:1px solid #1C3444;} QFrame#card{background:#0E1C27;border:1px solid #203B4D;border-radius:14px;}
QLabel#title{font-size:26px;font-weight:700;} QLabel#subtitle{color:#8299A8;} QLabel#cardtitle{font-size:16px;font-weight:700;} QLabel#muted{color:#8299A8;}
QPushButton{background:#102535;border:1px solid #28475C;border-radius:9px;padding:10px 15px;color:#EAF2F7;} QPushButton:hover{background:#18394D;}
QPushButton#primary{background:#27D7C6;color:#04211E;border:0;font-weight:700;font-size:15px;padding:13px 20px;} QPushButton#primary:hover{background:#45E2D4;}
QPushButton#side{background:transparent;border:0;text-align:left;padding:12px 16px;color:#9AB0BE;} QPushButton#side:checked{background:#123044;color:white;border-left:3px solid #27D7C6;}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#08141D;border:1px solid #27465A;border-radius:8px;padding:9px;color:white;}
QPlainTextEdit{background:#050B10;border:1px solid #1E3442;border-radius:10px;color:#B9CBD5;padding:8px;}
QTableWidget{background:#08131C;alternate-background-color:#0B1923;border:1px solid #203B4D;border-radius:10px;gridline-color:#17303F;selection-background-color:#16445A;}
QHeaderView::section{background:#102534;color:#AFC1CC;padding:8px;border:0;border-right:1px solid #203B4D;} QProgressBar{background:#10202C;border:0;border-radius:5px;height:10px;} QProgressBar::chunk{background:#27D7C6;border-radius:5px;}
QSlider::groove:horizontal{height:5px;background:#17303E;border-radius:2px;} QSlider::handle:horizontal{background:#27D7C6;width:14px;margin:-5px 0;border-radius:7px;}
QGroupBox{border:1px solid #203B4D;border-radius:10px;margin-top:12px;padding-top:14px;} QGroupBox::title{subcontrol-origin:margin;left:12px;color:#AFC1CC;}
'''

def path_button(parent, label, line, video=False):
    row=QHBoxLayout();lab=QLabel(label);lab.setFixedWidth(95);row.addWidget(lab);row.addWidget(line,1);btn=QPushButton('选择文件' if video else '选择目录');row.addWidget(btn)
    if video:btn.clicked.connect(lambda: choose_video(parent,line))
    else:btn.clicked.connect(lambda: choose_dir(parent,line))
    return row

def choose_video(parent,line):
    p,_=QFileDialog.getOpenFileName(parent,'选择视频','','视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)')
    if p:line.setText(p)
def choose_dir(parent,line):
    p=QFileDialog.getExistingDirectory(parent,'选择目录');
    if p:line.setText(p)
def open_path(p):
    p=Path(p)
    if p.exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))

class Worker(QThread):
    log=Signal(str);done=Signal(int,str);failed=Signal(int,str)
    def __init__(self,pid,cfg):super().__init__();self.pid=pid;self.cfg=cfg
    def run(self):
        try:
            DB.set_project_status(self.pid,'running');_,events=run_analysis(self.cfg,self.log.emit)
            DB.replace_events(self.pid,events.to_dict('records'));DB.set_project_status(self.pid,'done');self.done.emit(self.pid,self.cfg.output_dir)
        except Exception as e:
            DB.set_project_status(self.pid,'failed');self.failed.emit(self.pid,f'{e}\n{traceback.format_exc()}')

class MetricCard(QFrame):
    def __init__(self,value,title,accent='#27D7C6'):
        super().__init__();self.setObjectName('card');l=QVBoxLayout(self);v=QLabel(value);v.setStyleSheet(f'font-size:26px;font-weight:700;color:{accent};');t=QLabel(title);t.setObjectName('muted');l.addWidget(v);l.addWidget(t)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle(APP);self.resize(1540,940);self.setMinimumSize(980,650);self.setStyleSheet(QSS)
        ico=ROOT/'assets'/'app_icon.ico'
        if ico.exists():self.setWindowIcon(QIcon(str(ico)))
        self.worker=None;self.current_project=None;self.roi_path='';self._build();self.refresh_history();self.refresh_results_selector();self.refresh_backends()
    def _build(self):
        root=QWidget();self.setCentralWidget(root);h=QHBoxLayout(root);h.setContentsMargins(0,0,0,0);h.setSpacing(0)
        side=QFrame();side.setObjectName('sidebar');side.setFixedWidth(220);sl=QVBoxLayout(side);sl.setContentsMargins(16,18,16,18)
        brand=QLabel('PrimateBehaviorAI');brand.setStyleSheet('font-size:18px;font-weight:700;color:#27D7C6;');sl.addWidget(brand);cn=QLabel('灵长类智能行为分析');cn.setStyleSheet('font-size:13px;color:#B6C8D2;');sl.addWidget(cn);sl.addSpacing(24)
        self.nav=[]
        for i,txt in enumerate(['新建实验','分析数据','查看结果','历史实验','设置','使用帮助']):
            b=QPushButton(txt);b.setObjectName('side');b.setCheckable(True);b.clicked.connect(lambda checked=False,x=i:self.switch(x));sl.addWidget(b);self.nav.append(b)
        sl.addStretch();st=QLabel('● 本地计算\n视频默认不上传云端\n'+VERSION);st.setObjectName('muted');sl.addWidget(st);h.addWidget(side)
        self.pages=QStackedWidget();h.addWidget(self.pages,1)
        self.multicam_page=MultiCamPage(QSS,self);self.pages.addWidget(self.multicam_page);self.pages.addWidget(self.build_new());self.pages.addWidget(self.build_results());self.pages.addWidget(self.build_history());self.pages.addWidget(self.build_system());self.pages.addWidget(self.build_help());self.switch(0)
    def switch(self,i):
        self.pages.setCurrentIndex(i)
        for j,b in enumerate(self.nav):b.setChecked(i==j)
        if i==3:self.refresh_history()
        if i==2:
            self.refresh_results_selector()
            if self.current_project:self.load_result_project(self.current_project)
        if i==4:self.refresh_backends()
    def page_shell(self,title,sub):
        w=QWidget();l=QVBoxLayout(w);l.setContentsMargins(28,22,28,22);head=QHBoxLayout();a=QVBoxLayout();t=QLabel(title);t.setObjectName('title');s=QLabel(sub);s.setObjectName('subtitle');a.addWidget(t);a.addWidget(s);head.addLayout(a);head.addStretch();badge=QLabel(VERSION);badge.setStyleSheet('background:#102B37;color:#27D7C6;border-radius:8px;padding:8px 12px;font-weight:700;');head.addWidget(badge);l.addLayout(head);l.addSpacing(14);return w,l
    def card(self,title,sub=''):
        f=QFrame();f.setObjectName('card');l=QVBoxLayout(f);l.setContentsMargins(18,16,18,16);t=QLabel(title);t.setObjectName('cardtitle');l.addWidget(t)
        if sub:s=QLabel(sub);s.setObjectName('muted');s.setWordWrap(True);l.addWidget(s)
        return f,l
    def build_new(self):
        w,l=self.page_shell('分析数据','选择实验文件夹或视频文件；软件自动判断分析方式，不需要先了解不同算法模式。')
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.NoFrame);inner=QWidget();il=QVBoxLayout(inner);il.setSpacing(10);scroll.setWidget(inner);l.addWidget(scroll,1)

        c,cl=self.card('① 选择数据','采集好的 Session 选“实验文件夹”；外部导入的普通视频选“视频文件”。')
        mode=QHBoxLayout()
        self.rb_session=QRadioButton('Session');self.rb_single=QRadioButton('single');self.rb_pair=QRadioButton('before_after')
        self.rb_session.setChecked(True)
        self.hidden_mode_row=QWidget();hidden_layout=QHBoxLayout(self.hidden_mode_row)
        for rb in (self.rb_session,self.rb_single,self.rb_pair):hidden_layout.addWidget(rb)
        self.hidden_mode_row.hide();cl.addWidget(self.hidden_mode_row)
        self.pick_session_btn=QPushButton('选择实验文件夹…');self.pick_session_btn.setObjectName('primary')
        self.pick_session_btn.clicked.connect(self.choose_analysis_session)
        self.pick_video_btn=QPushButton('选择单个视频…');self.pick_video_btn.clicked.connect(self.choose_analysis_video)
        mode.addWidget(self.pick_session_btn);mode.addWidget(self.pick_video_btn);mode.addStretch()
        cl.addLayout(mode)
        self.import_mode_summary=QLabel('当前：实验文件夹 · 自动读取 1–3 路视频和阶段记录')
        self.import_mode_summary.setObjectName('muted');cl.addWidget(self.import_mode_summary)
        self.session_input=QWidget();sil=QVBoxLayout(self.session_input);sil.setContentsMargins(0,6,0,0)
        sr=QHBoxLayout();self.analysis_session_dir=QLineEdit();self.analysis_session_dir.setReadOnly(True);self.analysis_session_dir.setPlaceholderText('选择由“实验 Session”页面采集生成的 Session 目录');sb=QPushButton('选择 Session…');sb.clicked.connect(self.choose_analysis_session);self.scan_session_btn=QPushButton('扫描 Session');self.scan_session_btn.clicked.connect(self.scan_analysis_session);sr.addWidget(self.analysis_session_dir,1);sr.addWidget(sb);sr.addWidget(self.scan_session_btn);sil.addLayout(sr)
        self.session_scan_status=QLabel('选择文件夹后将自动读取视频，无需手动扫描。');self.session_scan_status.setObjectName('muted');self.session_scan_status.setWordWrap(True);sil.addWidget(self.session_scan_status)
        self.session_cam_table=QTableWidget(0,7);self.session_cam_table.setHorizontalHeaderLabels(['启用','摄像头','视角','视频文件','ROI 状态','标定','选择 ROI']);self.session_cam_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents);self.session_cam_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.Stretch);self.session_cam_table.verticalHeader().setVisible(False);self.session_cam_table.setMinimumHeight(155);self.session_cam_table.setMaximumHeight(245);sil.addWidget(self.session_cam_table)
        tips=QLabel('每个视角必须使用自己的 ROI/Mask；不同摄像头的透视和遮挡不同，不能共用同一套 Mask。多路模式当前采用“逐视角分析 + Session 时间轴候选融合”，不会伪装成 3D 重建。');tips.setObjectName('muted');tips.setWordWrap(True);sil.addWidget(tips);cl.addWidget(self.session_input)
        self.session_camera_items=[]

        self.legacy_input=QWidget();llg=QVBoxLayout(self.legacy_input);llg.setContentsMargins(0,6,0,0)
        self.before=QLineEdit();self.after=QLineEdit();self.single=QLineEdit();self.pair_rows=QWidget();pr=QVBoxLayout(self.pair_rows);pr.setContentsMargins(0,0,0,0);pr.addLayout(path_button(self,'照射前视频',self.before,True));pr.addLayout(path_button(self,'照射后视频',self.after,True));self.single_row=QWidget();ssr=QVBoxLayout(self.single_row);ssr.setContentsMargins(0,0,0,0);ssr.addLayout(path_button(self,'输入视频',self.single,True));self.compare_before_after=QCheckBox('与另一段视频做前后对比')
        self.compare_before_after.toggled.connect(self._toggle_compare_mode)
        llg.addWidget(self.compare_before_after);llg.addWidget(self.pair_rows);llg.addWidget(self.single_row);cl.addWidget(self.legacy_input);self.legacy_input.hide();self.pair_rows.hide();self.single_row.hide();il.addWidget(c)

        self.legacy_roi_card,self.legacy_roi_layout=self.card('视频 ROI 标定','历史单路视频使用一套 ROI。Session 多路模式的 ROI 请直接在上方摄像头表中逐路完成。');rr=QHBoxLayout();self.roi=QLineEdit();rr.addWidget(self.roi,1);sel=QPushButton('选择已有标定');sel.clicked.connect(self.choose_roi);auto=QPushButton('开始智能标定');auto.setObjectName('primary');auto.clicked.connect(self.calibrate);rr.addWidget(sel);rr.addWidget(auto);self.legacy_roi_layout.addLayout(rr);self.legacy_roi_card.hide();il.addWidget(self.legacy_roi_card)

        c3,c3l=self.card('分析策略','效果优先。Session 多路分析会分别保留每路完整结果，再按统一 Session 时间轴融合同类候选事件；不会把不同视角的连续数值终点盲目求平均。');form=QFormLayout();self.quality=QComboBox();self.quality.addItems(['极致精度 · 900（推荐）','高精度 · 600','标准精度 · 420']);self.backend=QComboBox();self.backend.addItems(['自动选择已接入后端','动态多边形跟踪（当前）']);self.th=QDoubleSpinBox();self.th.setDecimals(3);self.th.setRange(.970,.999);self.th.setSingleStep(.001);self.th.setValue(.995);self.dashboard=QCheckBox('为每一路生成 Dashboard 复核视频');self.dashboard.setChecked(True);form.addRow('空间分析精度',self.quality);form.addRow('跟踪后端',self.backend);form.addRow('基线稀有度阈值',self.th);form.addRow('',self.dashboard);c3l.addLayout(form)
        self.analysis_advanced_card=c3
        self.analysis_advanced_card.hide()
        self.analysis_advanced_toggle=QCheckBox('显示高级分析设置（默认使用高精度设置）')
        il.addWidget(self.analysis_advanced_toggle)
        self.analysis_advanced_toggle.toggled.connect(self.analysis_advanced_card.setVisible)
        il.addWidget(c3)

        c4,c4l=self.card('输出与任务','Session 多路结果顶层只保留总报告、事件表和摘要；逐视角完整结果统一收进 details 文件夹，避免结果目录过于杂乱。');self.name=QLineEdit('实验_'+time.strftime('%Y%m%d_%H%M'));self.out=QLineEdit(str(WORK/'projects'/time.strftime('%Y%m%d_%H%M%S')));c4l.addLayout(path_button(self,'结果目录',self.out,False));c4l.addWidget(QLabel('任务名称'));c4l.addWidget(self.name);self.analysis_input_status=QLabel('输入状态：等待选择实验 Session。');self.analysis_input_status.setObjectName('muted');self.analysis_input_status.setWordWrap(True);c4l.addWidget(self.analysis_input_status);self.runbtn=QPushButton('开始智能分析');self.runbtn.setObjectName('primary');self.runbtn.clicked.connect(self.start_analysis);c4l.addWidget(self.runbtn);self.progress=QProgressBar();self.progress.setRange(0,0);self.progress.hide();c4l.addWidget(self.progress);self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(170);c4l.addWidget(self.log);il.addWidget(c4);il.addStretch()

        self.rb_session.toggled.connect(self._analysis_mode_ui);self.rb_single.toggled.connect(self._analysis_mode_ui);self.rb_pair.toggled.connect(self._analysis_mode_ui);self._analysis_mode_ui()
        return w

    def _analysis_mode_ui(self, *args):
        is_session=self.rb_session.isChecked();is_single=self.rb_single.isChecked();is_pair=self.rb_pair.isChecked()
        self.session_input.setVisible(is_session);self.legacy_input.setVisible(not is_session);self.legacy_roi_card.setVisible(not is_session);self.single_row.setVisible(is_single);self.pair_rows.setVisible(is_pair);self.compare_before_after.setVisible(not is_session)
        if is_session:
            self.import_mode_summary.setText('当前：实验文件夹 · 自动识别 1–3 路视频与阶段')
            self.analysis_input_status.setText('请选择一次实验的 Session 文件夹。')
        elif is_single:
            self.import_mode_summary.setText('当前：单视频分析' )
            self.analysis_input_status.setText('单个视频已就绪时，请确认 ROI 标定。')
        else:
            self.import_mode_summary.setText('当前：前后视频对比')
            self.analysis_input_status.setText('请选择照射前后两段视频。')

    def choose_analysis_video(self):
        p,_=QFileDialog.getOpenFileName(self,'选择待分析视频','','视频 (*.avi *.mp4 *.mov *.mkv)')
        if p:
            self.single.setText(p);self.before.setText(p)
            self.compare_before_after.setChecked(False)
            self.rb_single.setChecked(True)
            self.name.setText(Path(p).stem+'_分析')

    def _toggle_compare_mode(self, enabled):
        if enabled:
            if self.single.text().strip():self.before.setText(self.single.text().strip())
            self.rb_pair.setChecked(True)
        else:
            if not self.rb_session.isChecked():self.rb_single.setChecked(True)

    def _mode_ui(self,single):
        # Backward-compatible helper retained for older signal paths.
        self._analysis_mode_ui()

    def choose_analysis_session(self):
        start=str(CAPTURES_DIR if 'CAPTURES_DIR' in globals() else WORK)
        p=QFileDialog.getExistingDirectory(self,'选择实验 Session 目录',start)
        if p:
            self.rb_session.setChecked(True)
            self.analysis_session_dir.setText(p);self.name.setText(Path(p).name+'_分析');self.session_scan_status.setText('扫描结果：目录已选择，等待扫描。');self.session_scan_status.setStyleSheet('color:#8299A8;');self.scan_analysis_session()

    @staticmethod
    def _view_cn(view):
        m={'front':'前视角','left':'左侧视角','right':'右侧视角','front_left':'前左斜视','front_right':'前右斜视','custom':'自定义'}
        return m.get(str(view),str(view))

    def scan_analysis_session(self):
        d=Path(self.analysis_session_dir.text().strip())
        self.session_cam_table.setRowCount(0);self.session_camera_items=[]
        if not d.exists():
            self.session_scan_status.setText('扫描结果：Session 目录不存在。');self.session_scan_status.setStyleSheet('color:#FF7070;font-weight:700;');return
        specs=[]
        meta_path=d/'session.json'
        if meta_path.exists():
            try:
                meta=json.loads(meta_path.read_text(encoding='utf-8-sig'));specs=meta.get('camera_specs') or []
            except Exception: specs=[]
        candidates=[]
        for vp in sorted(list(d.glob('cam*.avi'))+list(d.glob('cam*.mp4'))+list(d.glob('cam*.mov'))+list(d.glob('cam*.mkv'))):
            stem=vp.stem;camera_id=stem.split('_')[0];view='_'.join(stem.split('_')[1:]) or 'unknown';slot=None
            try:slot=max(0,int(camera_id.replace('cam',''))-1)
            except Exception:pass
            for sp in specs:
                if slot is not None and int(sp.get('slot_id',-99))==slot:
                    view=str(sp.get('view_key') or sp.get('view_name') or view);break
            candidates.append({'camera_id':camera_id,'view':view,'video_path':str(vp),'roi_path':''})
        # Deduplicate by video path and cap at the product-supported 3 views.
        uniq=[];seen=set()
        for x in candidates:
            if x['video_path'] in seen:continue
            seen.add(x['video_path']);uniq.append(x)
        candidates=uniq[:3]
        if not candidates:
            self.session_scan_status.setText('扫描结果：没有找到 cam*.avi/mp4/mov/mkv 视频。');self.session_scan_status.setStyleSheet('color:#FF7070;font-weight:700;');self.analysis_input_status.setText('输入状态：Session 中未发现可分析视频。');return
        for r,item in enumerate(candidates):
            default_roi=d/('roi_%s.json'%item['camera_id'])
            if default_roi.exists():item['roi_path']=str(default_roi)
            item['enabled_widget']=QCheckBox();item['enabled_widget'].setChecked(True);item['enabled_widget'].stateChanged.connect(self._refresh_session_analysis_status)
            self.session_camera_items.append(item);self.session_cam_table.insertRow(r)
            holder=QWidget();hl=QHBoxLayout(holder);hl.setContentsMargins(0,0,0,0);hl.setAlignment(Qt.AlignCenter);hl.addWidget(item['enabled_widget']);self.session_cam_table.setCellWidget(r,0,holder)
            self.session_cam_table.setItem(r,1,QTableWidgetItem(item['camera_id']));self.session_cam_table.setItem(r,2,QTableWidgetItem(self._view_cn(item['view'])));self.session_cam_table.setItem(r,3,QTableWidgetItem(Path(item['video_path']).name))
            roi_item=QTableWidgetItem('已标定' if item['roi_path'] else '未标定');self.session_cam_table.setItem(r,4,roi_item)
            b1=QPushButton('标定');b1.clicked.connect(lambda checked=False,row=r:self.calibrate_session_camera(row));self.session_cam_table.setCellWidget(r,5,b1)
            b2=QPushButton('选择…');b2.clicked.connect(lambda checked=False,row=r:self.choose_session_roi(row));self.session_cam_table.setCellWidget(r,6,b2)
        self.session_scan_status.setText('扫描结果：发现 %d 路 Session 视频。请逐路确认 ROI/Mask 后开始分析。'%len(candidates));self.session_scan_status.setStyleSheet('color:#43D18B;font-weight:700;')
        self._refresh_session_analysis_status()

    def _refresh_session_analysis_status(self):
        enabled=[x for x in self.session_camera_items if x.get('enabled_widget') and x['enabled_widget'].isChecked()]
        ready=[x for x in enabled if x.get('roi_path') and Path(x['roi_path']).exists()]
        self.analysis_input_status.setText('输入状态：已启用 %d 路，已完成 ROI %d/%d。%s'%(len(enabled),len(ready),len(enabled),'可以开始分析。' if enabled and len(ready)==len(enabled) else '请完成每一路 ROI。'))
        self.analysis_input_status.setStyleSheet('color:%s;font-weight:700;'%('#43D18B' if enabled and len(ready)==len(enabled) else '#F0B84B'))

    def calibrate_session_camera(self,row):
        if row<0 or row>=len(self.session_camera_items):return
        item=self.session_camera_items[row];video=item['video_path'];out=str(Path(self.analysis_session_dir.text())/('roi_%s.json'%item['camera_id']))
        d=CalibrationDialog(video,out,self)
        def saved(path,row=row):
            self.session_camera_items[row]['roi_path']=str(path);self.session_cam_table.item(row,4).setText('已标定');self._refresh_session_analysis_status()
        d.saved.connect(saved);d.exec()

    def choose_session_roi(self,row):
        if row<0 or row>=len(self.session_camera_items):return
        p,_=QFileDialog.getOpenFileName(self,'选择 %s 的 ROI 标定'%self.session_camera_items[row]['camera_id'],'','JSON (*.json)')
        if p:self.session_camera_items[row]['roi_path']=p;self.session_cam_table.item(row,4).setText('已选择');self._refresh_session_analysis_status()

    def choose_roi(self):
        p,_=QFileDialog.getOpenFileName(self,'选择区域配置','','JSON (*.json)');
        if p:self.roi.setText(p)
    def calibration_video(self):
        if self.rb_single.isChecked():return self.single.text().strip()
        return self.before.text().strip() or self.after.text().strip()
    def calibrate(self):
        v=self.calibration_video()
        if not v or not Path(v).exists():QMessageBox.warning(self,'缺少视频','请先选择用于标定的视频。');return
        out=str(Path(v).with_name('primate_rois_v5.json'));d=CalibrationDialog(v,out,self);d.saved.connect(self.roi.setText);d.exec()
    def make_cfg(self):
        q={0:900,1:600,2:420}[self.quality.currentIndex()]
        common=dict(output_dir=self.out.text().strip(),analysis_width=q,event_threshold=float(self.th.value()),make_dashboard=self.dashboard.isChecked(),tracking_backend=self.backend.currentText(),project_name=self.name.text().strip() or '未命名实验')
        if self.rb_session.isChecked():
            cams=[]
            for x in self.session_camera_items:
                if x.get('enabled_widget') and x['enabled_widget'].isChecked():
                    cams.append({'camera_id':x['camera_id'],'view':x['view'],'video_path':x['video_path'],'roi_path':x.get('roi_path','')})
            return AnalysisConfig(mode='session_multicam',session_dir=self.analysis_session_dir.text().strip(),camera_inputs=cams,**common)
        if self.rb_single.isChecked():
            return AnalysisConfig(mode='single',single_path=self.single.text().strip(),roi_path=self.roi.text().strip(),**common)
        return AnalysisConfig(mode='before_after',before_path=self.before.text().strip(),after_path=self.after.text().strip(),roi_path=self.roi.text().strip(),**common)
    def start_analysis(self):
        if self.worker and self.worker.isRunning():return
        cfg=self.make_cfg()
        try:cfg.validate()
        except Exception as e:QMessageBox.warning(self,'参数不完整',str(e));return
        db_single=cfg.session_dir if cfg.mode=='session_multicam' else cfg.single_path
        db_roi=('MULTI_ROI' if cfg.mode=='session_multicam' else cfg.roi_path)
        pid=DB.create_project(cfg.project_name,cfg.mode,cfg.before_path,cfg.after_path,db_single,db_roi,cfg.output_dir);self.current_project=pid
        self.log.clear();self.log.appendPlainText('[系统] 分析任务已创建：'+('Session 多视角' if cfg.mode=='session_multicam' else cfg.mode));self.runbtn.setEnabled(False);self.progress.show();self.worker=Worker(pid,cfg);self.worker.log.connect(self.log.appendPlainText);self.worker.done.connect(self.analysis_done);self.worker.failed.connect(self.analysis_failed);self.worker.start()
    def analysis_done(self,pid,out):
        self.runbtn.setEnabled(True);self.progress.hide();self.log.appendPlainText('[完成] 结果已生成：'+out);self.current_project=pid;QMessageBox.information(self,'分析完成','分析已完成，正在打开“查看结果”。');self.switch(2)
    def analysis_failed(self,pid,msg):
        self.runbtn.setEnabled(True);self.progress.hide();self.log.appendPlainText(msg);QMessageBox.critical(self,'分析失败',msg.splitlines()[0])

    def build_results(self):
        w,l=self.page_shell('查看结果','选择一次分析结果，打开报告或对 AI 候选逐条复核。')
        choose=QHBoxLayout();choose.addWidget(QLabel('分析结果'))
        self.result_selector=QComboBox();self.result_selector.setMinimumWidth(250)
        self.result_selector.currentIndexChanged.connect(self.result_selector_changed)
        choose.addWidget(self.result_selector,1)
        self.res_title=QLabel('尚未选择分析任务');self.res_title.setObjectName('muted');choose.addWidget(self.res_title)
        l.addLayout(choose)
        top=QHBoxLayout()
        self.open_report=QPushButton('打开中文报告');self.open_report.setObjectName('primary');self.open_report.clicked.connect(lambda:self._open_result('report.html'))
        self.review_studio=QPushButton('同步视频与事件复核');self.review_studio.clicked.connect(self.open_review_studio)
        self.open_dashboard=QPushButton('复核视频文件');self.open_dashboard.clicked.connect(self._open_dashboard_result)
        self.open_folder=QPushButton('结果所在目录');self.open_folder.clicked.connect(lambda:self._open_result(''))
        for b in (self.open_report,self.review_studio,self.open_dashboard,self.open_folder):top.addWidget(b)
        top.addStretch();l.addLayout(top)
        self.result_summary=QLabel('分析完成后，可在这里直接选取历史结果。')
        self.result_summary.setObjectName('muted');self.result_summary.setWordWrap(True);l.addWidget(self.result_summary)
        tabs=QTabWidget();tabs.setDocumentMode(True)
        event_page=QWidget();rl=QVBoxLayout(event_page)
        self.event_table=QTableWidget(0,8)
        self.event_table.setHorizontalHeaderLabels(['事件','开始(s)','峰值','分数','区域','候选解释','视角支持','专家结论'])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(5,QHeaderView.Stretch)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.itemSelectionChanged.connect(self.event_selected)
        rl.addWidget(self.event_table,1)
        rev=QHBoxLayout()
        for txt,val,col in [('确认异常','confirmed','#2BCB9C'),('正常行为','normal','#62A8FF'),('无法判断','uncertain','#FFD166')]:
            b=QPushButton(txt);b.setStyleSheet('background:%s;color:#041018;font-weight:700;'%col)
            b.clicked.connect(lambda checked=False,v=val:self.review_selected(v));rev.addWidget(b)
        rev.addStretch();rl.addLayout(rev)
        tabs.addTab(event_page,'AI 候选与人工结论')
        playback=QWidget();ll=QVBoxLayout(playback)
        if HAVE_MEDIA:
            self.video=QVideoWidget();ll.addWidget(self.video,1)
            self.player=QMediaPlayer();self.audio=QAudioOutput();self.audio.setVolume(0)
            self.player.setAudioOutput(self.audio);self.player.setVideoOutput(self.video)
            ctrl=QHBoxLayout();pb=QPushButton('播放 / 暂停');pb.clicked.connect(self.toggle_play)
            ctrl.addWidget(pb);self.pos=QSlider(Qt.Horizontal);self.pos.sliderMoved.connect(lambda v:self.player.setPosition(v))
            ctrl.addWidget(self.pos,1);ll.addLayout(ctrl)
            self.player.durationChanged.connect(lambda d:self.pos.setMaximum(d));self.player.positionChanged.connect(self.pos.setValue)
        else:
            ll.addWidget(QLabel('当前运行环境没有多媒体组件，请使用“同步视频与事件复核”打开原始视频。'))
        tabs.addTab(playback,'视频预览')
        l.addWidget(tabs,1)
        return w
    def refresh_results_selector(self):
        if not hasattr(self,'result_selector'):return
        current=self.current_project
        self.result_selector.blockSignals(True);self.result_selector.clear()
        for p in DB.list_projects():
            if p['status']=='done':
                self.result_selector.addItem(p['name']+' · '+time.strftime('%m-%d %H:%M',time.localtime(p['created_at'])),int(p['id']))
        if current is None and self.result_selector.count():current=self.result_selector.itemData(0)
        for i in range(self.result_selector.count()):
            if self.result_selector.itemData(i)==current:self.result_selector.setCurrentIndex(i);break
        self.result_selector.blockSignals(False)
        if current is not None:self.current_project=int(current)

    def result_selector_changed(self,idx):
        if idx < 0:return
        pid=self.result_selector.itemData(idx)
        if pid is not None:
            self.current_project=int(pid)
            self.load_result_project(self.current_project)

    def toggle_play(self):
        if not HAVE_MEDIA:return
        from PySide6.QtMultimedia import QMediaPlayer
        self.player.pause() if self.player.playbackState()==QMediaPlayer.PlayingState else self.player.play()
    def load_result_project(self,pid):
        if not pid:return
        ps=[r for r in DB.list_projects() if r['id']==pid]
        if not ps:return
        p=ps[0];self.res_title.setText(p['name']);self.event_table.setRowCount(0);events=DB.list_events(pid)
        self.result_summary.setText('候选事件 %d 条 · %s · 请选择事件复核，AI 候选不等同于诊断结论。'%(len(events),'分析完成' if p['status']=='done' else p['status']))
        support={}
        expert_reviews={}
        review_path=Path(p['output_dir'])/'review_annotations.csv'
        if review_path.exists():
            try:
                from .behavior_review import _read_csv
                expert_reviews={str(x.get('event_id','')):x.get('review','')
                                for x in _read_csv(review_path) if x.get('origin')=='AI'}
            except Exception:pass
        if p['mode']=='session_multicam':
            f=Path(p['output_dir'])/'events.csv'
            if f.exists():
                try:
                    df=pd.read_csv(f)
                    for _,x in df.iterrows():
                        support[int(x.get('event_id',0))]='%d路 · %s' % (int(x.get('support_count',1)),str(x.get('support_views','')))
                except Exception:pass
            self.open_report.setText('打开多视角总报告');self.open_dashboard.setText('打开多视角复核视频')
        else:
            self.open_report.setText('打开综合报告');self.open_dashboard.setText('打开复核视频')
        for r,e in enumerate(events):
            self.event_table.insertRow(r);vals=[e['event_index'],f"{e['start_s']:.2f}",f"{e['peak_s']:.2f}",f"{e['score']:.3f}",REGION_CN.get(e['region'],e['region']),EVENT_CN.get(e['label'],e['label']),support.get(int(e['event_index']),'—'),expert_reviews.get(str(e['event_index']),e['review'])]
            for c,v in enumerate(vals):it=QTableWidgetItem(str(v));it.setData(Qt.UserRole,e['id']);self.event_table.setItem(r,c,it)
        if HAVE_MEDIA:
            v=Path(p['output_dir'])/'annotated_dashboard.mp4'
            if not v.exists():
                candidates=sorted(Path(p['output_dir']).glob('details/cam*/annotated_dashboard.mp4')) or sorted(Path(p['output_dir']).glob('cam*/annotated_dashboard.mp4'));v=candidates[0] if candidates else v
            if v.exists():self.player.setSource(QUrl.fromLocalFile(str(v.resolve())))

    def event_selected(self):
        if not HAVE_MEDIA or not self.current_project:return
        rows=self.event_table.selectionModel().selectedRows()
        if rows:
            try:s=float(self.event_table.item(rows[0].row(),1).text());self.player.setPosition(max(0,int((s-1)*1000)))
            except:pass
    def review_selected(self,val):
        rows=self.event_table.selectionModel().selectedRows()
        if not rows:return
        row=rows[0].row();eid=self.event_table.item(row,0).data(Qt.UserRole)
        DB.review_event(int(eid),val);self.event_table.item(row,7).setText(val)
        try:
            from .behavior_review import ReviewStore
            ps=[r for r in DB.list_projects() if r['id']==self.current_project]
            if ps:
                store=ReviewStore(ps[0]['output_dir'])
                event_index=str(self.event_table.item(row,0).text())
                for i,e in enumerate(store.events):
                    if e['origin']=='AI' and str(e['event_id'])==event_index:
                        store.update(i,review=val)
                        break
        except Exception as e:
            QMessageBox.warning(self,'复核保存提示','数据库已更新，但人工复核文件保存失败：'+str(e))
    def open_review_studio(self):
        if not self.current_project:
            QMessageBox.information(self,'未选择任务','请先在历史任务中选择一个已完成的分析结果。')
            return
        projects=[r for r in DB.list_projects() if r['id']==self.current_project]
        if not projects:return
        from .review_ui import ReviewDialog
        self.review_dialog=ReviewDialog(projects[0]['output_dir'],self)
        self.review_dialog.exec()
        self.load_result_project(self.current_project)

    def _open_dashboard_result(self):
        if not self.current_project:return
        ps=[r for r in DB.list_projects() if r['id']==self.current_project]
        if not ps:return
        root=Path(ps[0]['output_dir']);direct=root/'annotated_dashboard.mp4'
        if direct.exists():open_path(direct);return
        candidates=sorted(root.glob('details/cam*/annotated_dashboard.mp4')) or sorted(root.glob('cam*/annotated_dashboard.mp4'))
        if len(candidates)==1:open_path(candidates[0]);return
        if len(candidates)>1:
            # Multiple camera dashboards: open the result folder rather than arbitrarily choosing one view.
            open_path(root);QMessageBox.information(self,'多路复核视频','本任务包含 %d 路 Dashboard，已打开结果目录。请进入 details/cam01、cam02、cam03 分别查看。'%len(candidates));return
        QMessageBox.information(self,'暂无复核视频','当前任务没有生成 Dashboard 视频。')

    def _open_result(self,name):
        if not self.current_project:return
        ps=[r for r in DB.list_projects() if r['id']==self.current_project]
        if not ps:return
        p=Path(ps[0]['output_dir']);open_path(p/name if name else p)
    def build_history(self):
        w,l=self.page_shell('历史实验','查找过去的分析结果，双击直接打开；需要时批量导出对比。')
        tools=QHBoxLayout();self.history_search=QLineEdit();self.history_search.setPlaceholderText('搜索实验名 / 任务状态…');self.history_search.textChanged.connect(self.refresh_history);tools.addWidget(self.history_search,1)
        tools.addStretch()
        self.batch_export_button=QPushButton('导出所选实验比较 Excel')
        self.batch_export_button.clicked.connect(self.export_selected_batch)
        tools.addWidget(self.batch_export_button);l.addLayout(tools)
        self.hist=QTableWidget(0,4);self.hist.setHorizontalHeaderLabels(['实验名称','状态','创建时间','分析类型']);self.hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.hist.setSelectionBehavior(QAbstractItemView.SelectRows);self.hist.doubleClicked.connect(self.history_open);l.addWidget(self.hist,1);return w
    def refresh_history(self):
        if not hasattr(self,'hist'):return
        rows=DB.list_projects();self.hist.setRowCount(0)
        query=self.history_search.text().strip().lower() if hasattr(self,'history_search') else ''
        idx=0
        for r in rows:
            if query and query not in (r['name']+' '+r['status']+' '+r['mode']).lower():continue
            self.hist.insertRow(idx);mode_cn={'session_multicam':'实验Session','before_after':'前后对比','single':'单视频'}.get(r['mode'],r['mode'])
            vals=[r['name'],{'done':'已完成','running':'分析中','failed':'失败'}.get(r['status'],r['status']),time.strftime('%Y-%m-%d %H:%M',time.localtime(r['created_at'])),mode_cn]
            for c,v in enumerate(vals):it=QTableWidgetItem(str(v));it.setData(Qt.UserRole,r['id']);self.hist.setItem(idx,c,it)
            idx+=1
    def export_selected_batch(self):
        rows=self.hist.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self,'请先选择','请先在历史任务表中选择至少一个已完成的分析任务；可按 Ctrl 多选。')
            return
        ids=set(int(self.hist.item(r.row(),0).data(Qt.UserRole)) for r in rows)
        roots=[r['output_dir'] for r in DB.list_projects() if r['id'] in ids and r['status']=='done']
        if not roots:
            QMessageBox.warning(self,'尚无完成结果','所选任务尚未完成分析。')
            return
        path,_=QFileDialog.getSaveFileName(self,'导出跨 Session 比较','实验批次比较.xlsx','Excel (*.xlsx)')
        if not path:return
        try:
            from .behavior_review import export_batch_workbook
            final=export_batch_workbook(roots,path)
            QMessageBox.information(self,'已完成','批次比较已导出：\n'+str(final))
        except Exception as exc:
            QMessageBox.critical(self,'导出失败',str(exc))

    def history_open(self):
        rows=self.hist.selectionModel().selectedRows()
        if rows:self.current_project=int(self.hist.item(rows[0].row(),0).data(Qt.UserRole));self.switch(2)
    def build_system(self):
        w,l=self.page_shell('设置','常用设置与故障诊断；普通实验无需调整 AI 后端。')
        card,c=self.card('应用与本地数据','软件采用本地处理；视频与标注默认不会上传云端。')
        for label,value in [('应用版本','4.8.0'),('工作数据目录',str(APP_HOME)),('默认分析输出',str(WORK/'projects')),('模型目录',str(MODELS_DIR))]:
            info=QLabel(label+'：'+value);info.setWordWrap(True);c.addWidget(info)
        l.addWidget(card)
        c2,c2l=self.card('故障诊断','遇到摄像头打不开、安装问题等，可导出不包含原始实验视频的技术信息。')
        diagnostic_btn=QPushButton('导出诊断文件…');diagnostic_btn.clicked.connect(self.export_diagnostics)
        c2l.addWidget(diagnostic_btn)
        self.show_backends=QCheckBox('显示高级模型与后端详情（技术人员使用）');c2l.addWidget(self.show_backends)
        self.backend_detail=QFrame();self.backend_box=QVBoxLayout(self.backend_detail)
        self.backend_detail.hide();c2l.addWidget(self.backend_detail)
        self.show_backends.toggled.connect(self.backend_detail.setVisible)
        l.addWidget(c2);l.addStretch();return w
    def export_diagnostics(self):
        name='PrimateBehaviorAI_Diagnostics.zip'
        path,_=QFileDialog.getSaveFileName(self,'导出环境诊断包',str(Path.home()/name),'ZIP压缩包 (*.zip)')
        if not path:return
        try:
            result=build_diagnostic_archive(path)
            QMessageBox.information(self,'诊断包已生成','已保存：\n%s\n\n只包含系统环境与依赖版本，不含视频、ROI或实验参数。' % result)
        except Exception as e:
            QMessageBox.warning(self,'诊断失败',str(e))

    def refresh_backends(self):
        if not hasattr(self,'backend_box'):return
        while self.backend_box.count():
            x=self.backend_box.takeAt(0);w=x.widget();w.deleteLater() if w else None
        for b in probe_backends(MODELS_DIR):
            f=QFrame();r=QHBoxLayout(f);dot=QLabel('●');dot.setStyleSheet('color:'+('#5BE7A9' if b.available else '#FFD166')+';font-size:18px;');r.addWidget(dot);a=QVBoxLayout();n=QLabel(b.name);n.setStyleSheet('font-weight:700;');d=QLabel(b.detail);d.setObjectName('muted');a.addWidget(n);a.addWidget(d);r.addLayout(a,1);self.backend_box.addWidget(f)
    def build_help(self):
        w,l=self.page_shell('使用帮助','常用操作只需要下面四步；技术参数和诊断仅在遇到问题时使用。')
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.NoFrame)
        content=QWidget();cl=QVBoxLayout(content);scroll.setWidget(content);l.addWidget(scroll,1)
        tips=[
            ('1 · 开始实验','选择使用的 1–3 路摄像头，点击“检测设备”和“快速准备”。查看画面是否实时变化后开始录像。'),
            ('2 · 记录实验阶段','录像不停，在“阶段与参数”中填写载波、调制、信号强度、天线距离，切换 Baseline、刺激期、恢复期。'),
            ('3 · 分析数据','点击“分析数据”，选择实验 Session 文件夹。程序自动找到各路视频；分别标定可见体区，开始分析。'),
            ('4 · 复核与导出','在“查看结果”选择实验，查看中文报告或同步视频。确认候选事件，并导出 Excel。'),
        ]
        for name,desc in tips:
            card,card_layout=self.card(name,desc);cl.addWidget(card)
        extra,el=self.card('常见问题与科学使用边界')
        faq=QLabel('多路启动失败：优先降低分辨率、检查 USB Hub 与摄像头占用；必要时打开完整诊断。\n'
                   '画面不清晰：优先确认照明、快门与镜头；细微眼部/震颤分析需要足够的空间和时间分辨率。\n'
                   '采集同步：普通 USB 摄像头记录主机侧时间戳，不等于硬件曝光同步。\n'
                   '异常标记：抽搐样、困倦样是视频筛查候选，须由研究人员复核，不应视为医学诊断。')
        faq.setWordWrap(True);el.addWidget(faq);cl.addWidget(extra);cl.addStretch()
        return w
    def closeEvent(self,e):
        capturing=hasattr(self,'multicam_page') and self.multicam_page.controller.running
        analyzing=self.worker and self.worker.isRunning()
        if capturing or analyzing:
            parts=[]
            if capturing:parts.append('1–3 路摄像头仍在采集')
            if analyzing:parts.append('分析任务仍在运行')
            msg='，'.join(parts)+'。关闭程序会中断当前任务，确定退出吗？'
            if QMessageBox.question(self,'退出程序',msg)!=QMessageBox.Yes:
                e.ignore();return
        if capturing:
            self.multicam_page.shutdown()
        e.accept()

def run():
    app=QApplication(sys.argv);app.setApplicationName(APP);app.setStyle('Fusion');win=MainWindow();win.show();sys.exit(app.exec())
