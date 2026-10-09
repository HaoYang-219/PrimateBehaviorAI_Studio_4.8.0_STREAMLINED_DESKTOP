# coding: utf-8
"""Synchronized video review and expert event coding dialog (1–3 sources)."""
from __future__ import annotations
import time
from pathlib import Path

import cv2
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtGui import QImage,QPixmap,QKeySequence,QShortcut,QPainter,QColor,QPen
from PySide6.QtWidgets import (QWidget,QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,
    QPushButton,QTableWidget,QTableWidgetItem,QHeaderView,QSlider,QComboBox,
    QDoubleSpinBox,QLineEdit,QFrame,QMessageBox,QFileDialog,QSplitter,QInputDialog)
from .behavior_review import (ReviewStore, CameraTimeIndex, camera_sources,
                              read_codebook,save_codebook,export_workbook,stage_windows)
from .paths import DATA_DIR


class StageTimeline(QWidget):
    seek_requested=Signal(float)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.windows=[];self.duration=1.;self.position=0.
        self.setMinimumHeight(37);self.setMaximumHeight(43)
        self.setToolTip('实验阶段：点击阶段条可直接定位到该阶段开始。')

    def set_data(self,windows,duration):
        self.windows=list(windows);self.duration=max(1.,float(duration));self.update()

    def set_position(self,position):
        self.position=float(position);self.update()

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect=self.rect().adjusted(4,4,-4,-4)
        painter.setPen(Qt.NoPen);painter.setBrush(QColor('#1A303E'))
        painter.drawRoundedRect(rect,6,6)
        if not self.windows:
            painter.setPen(QColor('#9BAFBB'));painter.drawText(rect,Qt.AlignCenter,'无阶段记录 · 可直接拖动时间轴复核')
        for w in self.windows:
            a=max(0.,min(self.duration,float(w['start_s'])));b=max(a,min(self.duration,float(w['end_s'])))
            x1=rect.left()+int(rect.width()*a/self.duration)
            x2=rect.left()+int(rect.width()*b/self.duration)
            rw=max(1,x2-x1)
            name=(w['name']+' '+w['condition']).strip()
            label=name.lower()
            color='#387C8B' if any(k in label for k in ('baseline','基线','刺激前')) else ('#A56B39' if any(k in label for k in ('刺激','exposure','stimulation')) else '#396B60')
            painter.setBrush(QColor(color));painter.drawRect(x1,rect.top(),rw,rect.height())
            if rw>55:
                painter.setPen(QColor('#FFFFFF'));painter.drawText(x1+4,rect.center().y()+5,name[:14]);painter.setPen(Qt.NoPen)
        x=rect.left()+int(rect.width()*max(0.,min(self.position,self.duration))/self.duration)
        painter.setPen(QPen(QColor('#FFFFFF'),2));painter.drawLine(x,rect.top(),x,rect.bottom())
        painter.end()

    def mousePressEvent(self,event):
        if event.button()!=Qt.LeftButton:return
        x=float(event.position().x()) if hasattr(event,'position') else float(event.x())
        t=max(0.,min(self.duration,(x-4)/max(1.,self.width()-8)*self.duration))
        # Simple click navigates to that point, rather than modifying phase boundaries.
        self.seek_requested.emit(t)



class ReviewDialog(QDialog):
    """Playback timeline uses Session seconds, not just nominal video timestamps.

    Source camera recordings are read-only. Every edit is persisted atomically
    in review_annotations.csv, never written back into AI raw predictions.
    """
    def __init__(self,result_dir,parent=None):
        super().__init__(parent)
        self.root=Path(result_dir)
        self.setWindowTitle('同步视频与专家复核 · '+self.root.name)
        self.resize(1420,860);self.setMinimumSize(960,640)
        self.store=ReviewStore(self.root)
        self.codebook=read_codebook(DATA_DIR)
        self.sources=camera_sources(self.root)
        self.players=[]
        self.duration=0.
        self.session_t=0.
        self.playing=False;self.speed=1.
        self._wall_tick=None
        self._rendered_t=None
        self._loaded_labels=[]
        self._build()
        self._open_cameras()
        self._reload_events()
        self.seek(0.)

    def _build(self):
        root=QVBoxLayout(self);root.setContentsMargins(14,12,14,12)
        banner=QHBoxLayout()
        title=QLabel('三路同步回放与事件复核');title.setStyleSheet('font-size:19px;font-weight:bold;')
        banner.addWidget(title);banner.addStretch()
        self.sync_hint=QLabel('逐帧时间戳优先；回放用于人工复核，非硬件帧同步')
        banner.addWidget(self.sync_hint);root.addLayout(banner)
        self.split=QSplitter(Qt.Vertical)
        video=QFrame();v=QVBoxLayout(video);self.video_row=QHBoxLayout()
        self.video_panels=[]
        for j in range(3):
            panel=QFrame();panel.setStyleSheet('QFrame{border:1px solid #28475C;border-radius:7px;}')
            box=QVBoxLayout(panel);name=QLabel('视角 %d'%(j+1));name.setAlignment(Qt.AlignCenter)
            img=QLabel('无视频');img.setMinimumSize(220,180);img.setAlignment(Qt.AlignCenter)
            img.setStyleSheet('background:#03080C;color:#ABC1CE;')
            box.addWidget(name);box.addWidget(img,1)
            self.video_row.addWidget(panel,1)
            self.video_panels.append((panel,name,img))
        v.addLayout(self.video_row,1)
        toolbar=QHBoxLayout()
        self.play_btn=QPushButton('▶ 播放');self.play_btn.clicked.connect(self.toggle_play)
        step_back=QPushButton('◀ 1帧');step_back.clicked.connect(lambda:self.step(-1))
        step_next=QPushButton('1帧 ▶');step_next.clicked.connect(lambda:self.step(1))
        self.speed_box=QComboBox()
        for s in [0.25,0.5,1.,2.]:self.speed_box.addItem('%gx'%s,s)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentIndexChanged.connect(lambda:self._set_speed())
        self.clock=QLabel('00:00.000')
        toolbar.addWidget(self.play_btn);toolbar.addWidget(step_back);toolbar.addWidget(step_next)
        toolbar.addWidget(self.speed_box);toolbar.addStretch();toolbar.addWidget(self.clock)
        v.addLayout(toolbar)
        self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,300000)
        self.slider.sliderPressed.connect(lambda:self._pause())
        self.slider.sliderReleased.connect(lambda:self.seek(self.slider.value()/1000.))
        self.slider.valueChanged.connect(self._slider_moved)
        v.addWidget(self.slider)
        self.stage_bar=StageTimeline()
        self.stage_bar.seek_requested.connect(self.seek)
        v.addWidget(self.stage_bar)
        self.split.addWidget(video)
        lower=QFrame();ll=QVBoxLayout(lower)
        row=QHBoxLayout();row.addWidget(QLabel('候选与人工事件'))
        row.addStretch()
        count=QLabel('快捷键：空格播放/暂停，←/→逐帧，1确认，2正常，3存疑')
        count.setStyleSheet('color:#90A8B5;');row.addWidget(count);ll.addLayout(row)
        self.table=QTableWidget(0,7);self.table.setHorizontalHeaderLabels(['时间(s)','行为','区间(s)','区域','摄像头','来源','专家状态'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selected)
        ll.addWidget(self.table,1)
        edit=QHBoxLayout()
        self.behavior=QComboBox();self.behavior.setEditable(True)
        self.behavior.addItems([x['label'] for x in self.codebook])
        self.start=QDoubleSpinBox();self.end=QDoubleSpinBox()
        for sb in [self.start,self.end]:sb.setRange(0,864000.);sb.setDecimals(3);sb.setSuffix(' s')
        self.note=QLineEdit();self.note.setPlaceholderText('人工备注 / 判断依据')
        edit.addWidget(QLabel('行为'));edit.addWidget(self.behavior,2)
        edit.addWidget(QLabel('起'));edit.addWidget(self.start,1)
        edit.addWidget(QLabel('止'));edit.addWidget(self.end,1)
        edit.addWidget(self.note,2)
        ll.addLayout(edit)
        actions=QHBoxLayout()
        for title,state in [('✓ 确认','confirmed'),('× 正常行为/误报','normal'),('? 无法判断','uncertain')]:
            b=QPushButton(title);b.clicked.connect(lambda _=False,s=state:self._review(s));actions.addWidget(b)
        add=QPushButton('＋ 人工补标');add.clicked.connect(self._add_manual);actions.addWidget(add)
        b=QPushButton('管理行为词典');b.clicked.connect(self._edit_codebook);actions.addWidget(b)
        b=QPushButton('导出科研 Excel');b.clicked.connect(self._export);actions.addWidget(b)
        ll.addLayout(actions)
        hint=QLabel('AI检测为候选事件；只有人工确认的记录纳入“人工确认统计”。原始视频和 AI 原始 events.csv 保持不变。')
        hint.setWordWrap(True);hint.setStyleSheet('color:#9CB4C4;');ll.addWidget(hint)
        self.split.addWidget(lower);self.split.setSizes([510,340]);root.addWidget(self.split,1)
        self.timer=QTimer(self);self.timer.setInterval(80);self.timer.timeout.connect(self._tick)
        QShortcut(QKeySequence('Space'),self).activated.connect(self.toggle_play)
        QShortcut(QKeySequence('Left'),self).activated.connect(lambda:self.step(-1))
        QShortcut(QKeySequence('Right'),self).activated.connect(lambda:self.step(1))
        for k,s in [('1','confirmed'),('2','normal'),('3','uncertain')]:
            QShortcut(QKeySequence(k),self).activated.connect(lambda state=s:self._review(state))

    def _open_cameras(self):
        for i,item in enumerate(self.sources[:3]):
            fn=Path(item['video_path']);cap=cv2.VideoCapture(str(fn))
            if not cap.isOpened():continue
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.
            frames=cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            idx=CameraTimeIndex(item.get('session_dir',''),item['camera_id'],fps)
            length=idx.times[-1] if idx.times else frames/max(fps,1)
            self.duration=max(self.duration,length)
            self.players.append({'capture':cap,'index':idx,'fps':fps,'last_frame':-1,
                                 'panel':self.video_panels[i], 'source':item})
            panel,lab,img=self.video_panels[i]
            lab.setText('%s · %s'%(item.get('camera_id',''),item.get('view','')))
        for i,(panel,lab,img) in enumerate(self.video_panels):panel.setVisible(i<len(self.players))
        if not self.players:
            self.duration=300.
            self.sync_hint.setText('未找到可关联的原视频；仍可复核事件表，建议重新选择 Session 分析结果。')
        elif any(not p['index'].times for p in self.players):
            self.sync_hint.setText('部分视频缺少逐帧采集时标，已使用名义 FPS 回退，精细跨视角同步需谨慎。')
        else:self.sync_hint.setText('%d 路原视频 · 逐帧 Session 时标映射'%len(self.players))
        self.slider.setMaximum(int(min(self.duration,864000.)*1000))
        self.stage_bar.set_data(stage_windows(self.root),self.duration)

    def _draw(self):
        for p in self.players:
            cap=p['capture'];frame_id=p['index'].frame_at(self.session_t)
            if frame_id==p['last_frame']:continue
            if frame_id!=p['last_frame']+1:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            ok,bgr=cap.read()
            if not ok:continue
            p['last_frame']=frame_id
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2];q=QImage(rgb.data,w,h,3*w,QImage.Format_RGB888).copy()
            label=p['panel'][2]
            label.setPixmap(QPixmap.fromImage(q).scaled(label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        m=int(self.session_t)//60;s=self.session_t-60*m
        self.clock.setText('%02d:%06.3f / %.1f s'%(m,s,self.duration))
        self.stage_bar.set_position(self.session_t)
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True);self.slider.setValue(int(self.session_t*1000));self.slider.blockSignals(False)

    def _slider_moved(self,v):
        if self.slider.isSliderDown():self.seek(v/1000.)

    def seek(self,t):
        self.session_t=min(max(0.,float(t)),self.duration)
        self._draw()

    def _set_speed(self):self.speed=float(self.speed_box.currentData() or 1.)

    def _pause(self):
        self.playing=False;self.timer.stop();self.play_btn.setText('▶ 播放')

    def toggle_play(self):
        if self.playing:self._pause();return
        self.playing=True;self.play_btn.setText('Ⅱ 暂停');self._wall_tick=time.monotonic();self.timer.start()

    def _tick(self):
        now=time.monotonic();elapsed=max(0.,min(.4,now-self._wall_tick));self._wall_tick=now
        self.seek(self.session_t+elapsed*self.speed)
        if self.session_t>=self.duration:self._pause()

    def step(self,direction):
        self._pause()
        delta=1./(self.players[0]['fps'] if self.players else 30.)
        self.seek(self.session_t+int(direction)*delta)

    def _reload_events(self,selected_id=None):
        self._loaded_labels=[x['event_id'] for x in self.store.events]
        self.table.blockSignals(True);self.table.setRowCount(0)
        target=None
        for n,r in enumerate(self.store.events):
            self.table.insertRow(n)
            vals=['%.3f'%float(r['peak_s']),r['label'],
                  '%.3f–%.3f'%(float(r['start_s']),float(r['end_s'])),
                  r['region'],r['support_cameras'],r['origin'],r['review']]
            for c,s in enumerate(vals):self.table.setItem(n,c,QTableWidgetItem(str(s)))
            if selected_id==r['event_id']:target=n
        self.table.blockSignals(False)
        if target is not None:self.table.selectRow(target)

    def _current_index(self):
        sel=self.table.selectionModel().selectedRows()
        return sel[0].row() if sel else None

    def _on_selected(self):
        idx=self._current_index()
        if idx is None or idx>=len(self.store.events):return
        e=self.store.events[idx];self._pause()
        self.seek(max(0.,float(e['peak_s'])-.3))
        self.behavior.setCurrentText(e['label']);self.start.setValue(float(e['start_s']))
        self.end.setValue(float(e['end_s']));self.note.setText(e['note'])

    def _review(self,state):
        idx=self._current_index()
        if idx is None:return
        a=self.start.value();b=self.end.value()
        if b<a:
            QMessageBox.warning(self,'时间区间错误','结束时间不能早于开始时间。');return
        eid=self.store.events[idx]['event_id']
        self.store.update(idx,label=self.behavior.currentText(),start_s=a,end_s=b,
                          peak_s=min(b,max(a,self.store.events[idx]['peak_s'])),
                          note=self.note.text(),review=state)
        self._reload_events(eid)

    def _add_manual(self):
        label=self.behavior.currentText().strip()
        if not label:return
        a=self.start.value();b=self.end.value()
        # New event uses current playback instant if the default range was left at zero.
        if a==0 and b==0:a=b=self.session_t
        if b<a:
            QMessageBox.warning(self,'时间区间错误','结束时间不能早于开始时间。');return
        event_id=self.store.add(label,a,b,note=self.note.text(),review='confirmed')
        self._reload_events(event_id)

    def _edit_codebook(self):
        help_text = '每行一种行为，格式：行为名称|point 或 行为名称|state\npoint=瞬时事件，state=持续状态'
        initial = "\n".join(x['label']+'|'+x['kind'] for x in self.codebook)
        value,ok = QInputDialog.getMultiLineText(self,'编辑行为词典',help_text,initial)
        if not ok:return
        entries=[]
        for line in value.splitlines():
            words=line.split('|');name=words[0].strip()
            if name:entries.append({'label':name,'kind':words[1].strip() if len(words)>1 else 'point'})
        if not entries:return
        save_codebook(DATA_DIR,entries);self.codebook=read_codebook(DATA_DIR)
        self.behavior.clear();self.behavior.addItems([x['label'] for x in self.codebook])

    def _export(self):
        try:
            target=export_workbook(self.root)
            QMessageBox.information(self,'已导出','科研汇总表已保存：\n'+str(target))
        except Exception as e:
            QMessageBox.critical(self,'导出失败',str(e))

    def closeEvent(self,event):
        self._pause()
        for p in self.players:p['capture'].release()
        super().closeEvent(event)
