from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np
from PySide6.QtCore import Qt, Signal, QPointF
from .roi_assist import suggest_reference_time
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (QDialog,QHBoxLayout,QVBoxLayout,QListWidget,QListWidgetItem,
    QPushButton,QLabel,QSlider,QMessageBox,QWidget,QSplitter,QComboBox,QFormLayout)

REGIONS=[
 ('whole_body','全身'),('head','头部'),('left_eye','左眼'),('right_eye','右眼'),('torso','躯干'),
 ('left_upper_limb','左上肢'),('right_upper_limb','右上肢'),
 ('left_lower_limb','左下肢'),('right_lower_limb','右下肢')]
COLORS=['#27D7C6','#62A8FF','#FFD166','#F5B7B1','#B28DFF','#5BE7A9','#FF8B8B','#7BDFF2','#F7A072']
VIEWPOINTS=[
 ('right_profile','右侧位（猴子右侧朝向相机）'),
 ('left_profile','左侧位（猴子左侧朝向相机）'),
 ('oblique','斜侧位'),
 ('frontal','正面/近正面')]
VISIBILITY=[('visible','可见'),('partial','部分可见'),('invisible','不可见')]
VIS_CN=dict(VISIBILITY)

class VideoCanvas(QLabel):
    pointAdded=Signal(float,float); undoRequested=Signal(); finishRequested=Signal()
    def __init__(self):
        super().__init__(); self.setMinimumSize(760,480); self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background:#05090D;border:1px solid #203444;border-radius:12px;')
        self.frame=None; self.frame_rgb=None; self.active=[]; self.active_region='whole_body'
        self.scale=1.; self.offx=self.offy=0.; self.setMouseTracking(True)
    def set_frame(self,bgr):
        self.frame=bgr.copy(); self.frame_rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); self.update_pixmap()
    def resizeEvent(self,e): super().resizeEvent(e); self.update_pixmap()
    def mousePressEvent(self,e):
        if self.frame is None:return
        if e.button()==Qt.RightButton:self.undoRequested.emit();return
        if e.button()!=Qt.LeftButton:return
        p=e.position(); x=(p.x()-self.offx)/max(self.scale,1e-9); y=(p.y()-self.offy)/max(self.scale,1e-9)
        h,w=self.frame.shape[:2]
        if 0<=x<w and 0<=y<h:self.pointAdded.emit(float(x),float(y))
    def mouseDoubleClickEvent(self,e):
        if e.button()==Qt.LeftButton:self.finishRequested.emit()
    def update_pixmap(self):
        if self.frame_rgb is None:return
        h,w=self.frame_rgb.shape[:2]; cw=max(10,self.width()-10); ch=max(10,self.height()-10)
        self.scale=min(cw/float(w),ch/float(h)); dw=max(1,int(w*self.scale)); dh=max(1,int(h*self.scale))
        self.offx=(self.width()-dw)/2.; self.offy=(self.height()-dh)/2.
        img=cv2.resize(self.frame_rgb,(dw,dh),interpolation=cv2.INTER_AREA)
        q=QImage(img.data,dw,dh,img.strides[0],QImage.Format_RGB888).copy(); pm=QPixmap.fromImage(q)
        painter=QPainter(pm); painter.setRenderHint(QPainter.Antialiasing)
        idx=next((i for i,(key,_) in enumerate(REGIONS) if key==self.active_region),0)
        key,cn=REGIONS[idx]; pts=self.active
        if pts:
            col=QColor(COLORS[idx]); painter.setPen(QPen(col,3)); qpts=[QPointF(x*self.scale,y*self.scale) for x,y in pts]
            if len(qpts)>=3:painter.drawPolygon(QPolygonF(qpts))
            elif len(qpts)==2:painter.drawLine(qpts[0],qpts[1])
            for pt in qpts:painter.setBrush(col); painter.drawEllipse(pt,4,4)
            if qpts:painter.drawText(qpts[0]+QPointF(6,-7),cn)
        painter.end(); canvas=QPixmap(self.size()); canvas.fill(QColor('#05090D'))
        p=QPainter(canvas); p.drawPixmap(int(self.offx),int(self.offy),pm); p.end(); self.setPixmap(canvas)

class CalibrationDialog(QDialog):
    saved=Signal(str)
    def __init__(self, video_path, out_path, parent=None):
        super().__init__(parent); self.video_path=video_path; self.out_path=out_path
        self.setWindowTitle('实验区域标定 · 视角感知'); self.resize(1380,840)
        self.cap=cv2.VideoCapture(video_path)
        if not self.cap.isOpened():raise RuntimeError('无法打开视频')
        self.fps=float(self.cap.get(cv2.CAP_PROP_FPS) or 30); self.frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.duration=self.frames/max(self.fps,1e-6)
        self.w=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.h=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ref_time=None; self.polys={}; self.confirmed=set(); self.active=[]; self.region_idx=0
        self.visibility={k:'visible' for k,_ in REGIONS}; self.viewpoint='oblique'; self._syncing_visibility=False
        self._build(); self.seek(min(1.0,self.duration/3.0)); self._select(0)
    def _build(self):
        self.setStyleSheet('''QDialog{background:#071018;color:#EAF2F7;font-family:"Microsoft YaHei UI";} QLabel{color:#EAF2F7;} QPushButton{background:#123044;color:#EAF2F7;border:1px solid #29485C;border-radius:8px;padding:9px 14px;} QPushButton:hover{background:#19435C;} QPushButton#primary{background:#27D7C6;color:#06201D;border:0;font-weight:700;} QListWidget{background:#0D1A24;border:1px solid #203444;border-radius:10px;padding:5px;color:#DCE8EE;} QListWidget::item{padding:10px;border-radius:6px;} QListWidget::item:selected{background:#17384B;color:white;} QComboBox{background:#0D1A24;border:1px solid #29485C;border-radius:7px;padding:7px;color:white;} QSlider::groove:horizontal{height:5px;background:#18303E;border-radius:2px;} QSlider::handle:horizontal{background:#27D7C6;width:14px;margin:-5px 0;border-radius:7px;}''')
        root=QHBoxLayout(self); split=QSplitter(); root.addWidget(split)
        left=QWidget(); ll=QVBoxLayout(left); title=QLabel('区域、视角与可见性'); title.setStyleSheet('font-size:18px;font-weight:700;'); ll.addWidget(title)
        hint=QLabel('先选择视频真实视角，再逐区标定。\n不可见区域不会参与异常判定；部分可见区域会降低权重。\n画布始终只显示当前区域，互不遮挡。')
        hint.setWordWrap(True); hint.setStyleSheet('color:#8FA7B6;'); ll.addWidget(hint)
        form=QFormLayout(); self.view_combo=QComboBox()
        for key,cn in VIEWPOINTS:self.view_combo.addItem(cn,key)
        self.view_combo.setCurrentIndex(2); self.view_combo.currentIndexChanged.connect(self.view_changed); form.addRow('相机视角',self.view_combo); ll.addLayout(form)
        self.list=QListWidget()
        for _,cn in REGIONS:self.list.addItem(QListWidgetItem(cn+'   ·   未标定   ·   可见'))
        self.list.currentRowChanged.connect(self._select); ll.addWidget(self.list,1)
        visrow=QHBoxLayout(); visrow.addWidget(QLabel('当前区域可见性')); self.vis_combo=QComboBox()
        for key,cn in VISIBILITY:self.vis_combo.addItem(cn,key)
        self.vis_combo.currentIndexChanged.connect(self.visibility_changed); visrow.addWidget(self.vis_combo,1); ll.addLayout(visrow)
        applyv=QPushButton('按当前视角设置推荐可见性'); applyv.clicked.connect(self.apply_view_defaults); ll.addWidget(applyv)
        auto=QPushButton('按当前视角生成初始区域'); auto.clicked.connect(self.auto_propose); ll.addWidget(auto)
        confirm=QPushButton('确认当前区域'); confirm.setObjectName('primary'); confirm.clicked.connect(self.confirm_current); ll.addWidget(confirm)
        clear=QPushButton('重新绘制当前区域'); clear.clicked.connect(self.clear_current); ll.addWidget(clear)
        preview=QPushButton('查看全部区域预览'); preview.clicked.connect(self.preview_all); ll.addWidget(preview)
        clearall=QPushButton('重新选择参考帧'); clearall.clicked.connect(self.clear_all); ll.addWidget(clearall)
        save=QPushButton('保存全部标定'); save.clicked.connect(self.save); ll.addWidget(save)
        split.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); top=QHBoxLayout(); self.current=QLabel(); self.current.setStyleSheet('font-size:16px;font-weight:700;'); top.addWidget(self.current); top.addStretch(); self.time=QLabel(); self.time.setStyleSheet('color:#8FA7B6;'); top.addWidget(self.time); rl.addLayout(top)
        self.canvas=VideoCanvas(); self.canvas.pointAdded.connect(self.add_point); self.canvas.undoRequested.connect(self.undo); self.canvas.finishRequested.connect(self.finish); rl.addWidget(self.canvas,1)
        row=QHBoxLayout(); prev=QPushButton('上一帧'); prev.clicked.connect(lambda:self.step(-1)); row.addWidget(prev); nxt=QPushButton('下一帧'); nxt.clicked.connect(lambda:self.step(1)); row.addWidget(nxt)
        self.slider=QSlider(Qt.Horizontal); self.slider.setRange(0,max(1,self.frames-1)); self.slider.valueChanged.connect(self.slider_change); row.addWidget(self.slider,1); rl.addLayout(row)
        recommend=QPushButton('推荐清晰参考帧（人工确认后标定）'); recommend.clicked.connect(self.suggest_reference_frame); rl.addWidget(recommend)
        foot=QLabel('可见/部分可见：按真实轮廓画 Mask；不可见：无需精确绘制，确认后系统自动建立占位 Mask，并在分析中完全屏蔽该区域。')
        foot.setWordWrap(True); foot.setStyleSheet('color:#78909E;'); rl.addWidget(foot)
        split.addWidget(right); split.setSizes([330,1050])
    def _state_text(self,key,cn):
        if key in self.confirmed: state='已确认'
        elif key in self.polys: state='待确认'
        else: state='未标定'
        return '%s   ·   %s   ·   %s' % (cn,state,VIS_CN.get(self.visibility.get(key,'visible'),'可见'))
    def _refresh_item(self,idx):
        key,cn=REGIONS[idx]; self.list.item(idx).setText(self._state_text(key,cn))
    def _select(self,idx):
        if idx<0:return
        self.region_idx=idx; key,cn=REGIONS[idx]; self.active=list(self.polys.get(key,[]))
        self.canvas.active_region=key; self.canvas.active=self.active
        self._syncing_visibility=True
        v=self.visibility.get(key,'visible'); vi=next((i for i in range(self.vis_combo.count()) if self.vis_combo.itemData(i)==v),0); self.vis_combo.setCurrentIndex(vi)
        self._syncing_visibility=False
        state='已确认' if key in self.confirmed else ('待确认' if key in self.polys else '未标定')
        self.current.setText('当前区域：%s  ·  %s  ·  %s' % (cn,state,VIS_CN.get(v,'可见'))); self.canvas.update_pixmap()
    def view_changed(self): self.viewpoint=str(self.view_combo.currentData() or 'oblique')
    def visibility_changed(self):
        if self._syncing_visibility:return
        key,cn=REGIONS[self.region_idx]; self.visibility[key]=str(self.vis_combo.currentData() or 'visible'); self._refresh_item(self.region_idx); self._select(self.region_idx)
    def apply_view_defaults(self):
        self.viewpoint=str(self.view_combo.currentData() or 'oblique')
        v={k:'visible' for k,_ in REGIONS}
        if self.viewpoint=='right_profile':
            v['left_eye']='invisible'; v['left_upper_limb']='partial'; v['left_lower_limb']='partial'
        elif self.viewpoint=='left_profile':
            v['right_eye']='invisible'; v['right_upper_limb']='partial'; v['right_lower_limb']='partial'
        elif self.viewpoint=='oblique':
            v['left_eye']='partial'; v['right_eye']='partial'
        self.visibility.update(v)
        for i in range(len(REGIONS)):self._refresh_item(i)
        self._select(self.region_idx)
    def read_at(self,t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC,max(0,min(t,self.duration))*1000); ok,f=self.cap.read(); return f if ok else None
    def seek(self,t):
        if self.ref_time is not None and (self.polys or self.active):t=self.ref_time
        f=self.read_at(t)
        if f is None:return
        self.t=t; self.canvas.set_frame(f); self.slider.blockSignals(True); self.slider.setValue(int(round(t*self.fps))); self.slider.blockSignals(False)
        self.time.setText('%06.2f 秒  ·  %.2f FPS  ·  %d×%d' % (t,self.fps,self.w,self.h))
    def suggest_reference_frame(self):
        if self.polys or self.active or self.ref_time is not None:
            QMessageBox.information(self, '请先重新选择参考帧',
                '当前已经开始区域标定。为了保持全部 ROI 使用相同参考帧，请先点击“重新选择参考帧”，然后再使用推荐功能。')
            return
        t = suggest_reference_time(self.read_at, self.duration)
        if t is None:
            QMessageBox.warning(self, '无法推荐', '没有找到有效的可读取参考帧。')
            return
        self.seek(t)
        QMessageBox.information(self, '清晰参考帧',
            '已按清晰度与曝光指标选择 %.1f 秒附近的帧。请确认猴子头部、眼睛和四肢可见性，再逐区标定。' % t)

    def slider_change(self,v):self.seek(v/self.fps)
    def step(self,d):self.seek(getattr(self,'t',0)+d/self.fps)
    def add_point(self,x,y):
        if self.visibility.get(REGIONS[self.region_idx][0])=='invisible':return
        if self.ref_time is None:self.ref_time=getattr(self,'t',0)
        self.active.append((x,y)); self.canvas.active=self.active; self.canvas.update_pixmap()
    def undo(self):
        if self.active:self.active.pop(); self.canvas.update_pixmap()
    def finish(self):self.confirm_current()
    def _placeholder_poly(self,key):
        # Invisible regions only need a safe placeholder. They are gated to zero during analysis.
        if key in ('left_eye','right_eye') and 'head' in self.polys and len(self.polys['head'])>=3:
            a=np.asarray(self.polys['head'],dtype=float); x1,y1=a.min(0); x2,y2=a.max(0); cx=(x1+x2)/2.; cy=(y1+y2)/2.; s=max(3.,min(x2-x1,y2-y1)*.05)
        elif 'whole_body' in self.polys and len(self.polys['whole_body'])>=3:
            a=np.asarray(self.polys['whole_body'],dtype=float); x1,y1=a.min(0); x2,y2=a.max(0); cx=(x1+x2)/2.; cy=(y1+y2)/2.; s=max(4.,min(x2-x1,y2-y1)*.03)
        else:
            cx=self.w/2.; cy=self.h/2.; s=5.
        return [(cx-s,cy-s),(cx+s,cy-s),(cx+s,cy+s),(cx-s,cy+s)]
    def confirm_current(self):
        key,cn=REGIONS[self.region_idx]; vis=self.visibility.get(key,'visible')
        if vis=='invisible' and len(self.active)<3:
            self.active=self._placeholder_poly(key); self.canvas.active=self.active
        if len(self.active)<3:
            QMessageBox.warning(self,'尚未完成','当前区域至少需要 3 个顶点；若该区域确实看不见，请先把可见性设为“不可见”。');return
        if self.ref_time is None:self.ref_time=getattr(self,'t',0)
        self.polys[key]=list(self.active); self.confirmed.add(key); self._refresh_item(self.region_idx)
        self.current.setText('当前区域：%s  ·  已确认  ·  %s' % (cn,VIS_CN.get(vis,'可见')))
        next_idx=None
        for j in range(self.region_idx+1,len(REGIONS)):
            if REGIONS[j][0] not in self.confirmed:next_idx=j;break
        if next_idx is None:
            for j in range(0,self.region_idx):
                if REGIONS[j][0] not in self.confirmed:next_idx=j;break
        if next_idx is not None:self.list.setCurrentRow(next_idx)
        self.canvas.update_pixmap()
    def clear_current(self):
        key,cn=REGIONS[self.region_idx]; self.polys.pop(key,None); self.confirmed.discard(key); self.active=[]; self.canvas.active=[]; self._refresh_item(self.region_idx); self.current.setText('当前区域：'+cn+'  ·  未标定'); self.canvas.update_pixmap()
    def clear_all(self):
        self.polys={}; self.confirmed=set(); self.active=[]; self.ref_time=None
        for i in range(len(REGIONS)):self._refresh_item(i)
        self.canvas.active=[]; self.seek(getattr(self,'t',0))
    def auto_propose(self):
        self.apply_view_defaults()
        if 'whole_body' in self.polys:
            a=np.asarray(self.polys['whole_body']); x1,y1=a.min(0); x2,y2=a.max(0)
        else:x1,y1,x2,y2=.20*self.w,.07*self.h,.72*self.w,.98*self.h
        W,H=x2-x1,y2-y1
        boxes={
          'whole_body':(x1,y1,x2,y2),'head':(x1+.28*W,y1,x1+.72*W,y1+.34*H),
          'left_eye':(x1+.355*W,y1+.095*H,x1+.485*W,y1+.205*H),'right_eye':(x1+.515*W,y1+.095*H,x1+.655*W,y1+.205*H),
          'torso':(x1+.20*W,y1+.29*H,x1+.69*W,y1+.73*H),'left_upper_limb':(x1+.04*W,y1+.30*H,x1+.43*W,y1+.70*H),
          'right_upper_limb':(x1+.49*W,y1+.30*H,x1+.88*W,y1+.76*H),'left_lower_limb':(x1+.14*W,y1+.61*H,x1+.48*W,y2),
          'right_lower_limb':(x1+.47*W,y1+.61*H,x1+.82*W,y2)}
        for k,b in boxes.items():
            xa,ya,xb,yb=b; self.polys[k]=[(xa,ya),(xb,ya),(xb,yb),(xa,yb)] if self.visibility.get(k)!='invisible' else self._placeholder_poly(k)
        if self.ref_time is None:self.ref_time=getattr(self,'t',0)
        self.confirmed=set()
        for i in range(len(REGIONS)):self._refresh_item(i)
        self._select(self.region_idx)
        QMessageBox.information(self,'已生成初始区域','已按照当前视角生成 9 个区域建议，并设置推荐可见性。\n请按真实画面逐个确认；侧面不可见的眼睛/肢体不会参与异常判定。')
    def preview_all(self):
        if self.canvas.frame is None:return
        rgb=cv2.cvtColor(self.canvas.frame,cv2.COLOR_BGR2RGB).copy()
        for i,(key,cn) in enumerate(REGIONS):
            pts=self.polys.get(key,[]); vis=self.visibility.get(key,'visible')
            if len(pts)<3 or vis=='invisible':continue
            arr=np.asarray(pts,np.int32).reshape((-1,1,2)); color=((55+37*i)%255,(180+29*i)%255,(230-17*i)%255)
            cv2.polylines(rgb,[arr],True,color,3,cv2.LINE_AA); x,y=map(int,pts[0]); cv2.putText(rgb,'%s[%s]'%(key,VIS_CN.get(vis,'')),(x+6,max(18,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2,cv2.LINE_AA)
        h,w=rgb.shape[:2]; sc=min(1180./w,720./h,1.0); dw,dh=max(1,int(w*sc)),max(1,int(h*sc)); show=cv2.resize(rgb,(dw,dh),interpolation=cv2.INTER_AREA)
        q=QImage(show.data,dw,dh,show.strides[0],QImage.Format_RGB888).copy(); dlg=QDialog(self); dlg.setWindowTitle('可见区域叠加预览'); dlg.resize(dw+40,dh+90)
        lay=QVBoxLayout(dlg); lab=QLabel(); lab.setAlignment(Qt.AlignCenter); lab.setPixmap(QPixmap.fromImage(q)); lay.addWidget(lab,1)
        tip=QLabel('不可见区域不会显示，也不会进入后续异常判定。'); tip.setStyleSheet('color:#8FA7B6;'); lay.addWidget(tip); dlg.exec()
    def save(self):
        # Invisible regions may use an automatic placeholder; visible/partial regions must be explicitly confirmed.
        for i,(k,cn) in enumerate(REGIONS):
            if self.visibility.get(k)=='invisible' and len(self.polys.get(k,[]))<3:
                self.polys[k]=self._placeholder_poly(k); self.confirmed.add(k); self._refresh_item(i)
        miss=[cn for k,cn in REGIONS if self.visibility.get(k)!='invisible' and len(self.polys.get(k,[]))<3]
        if miss:QMessageBox.warning(self,'仍有未标定区域','请完成：'+'、'.join(miss));return
        unconfirmed=[cn for k,cn in REGIONS if self.visibility.get(k)!='invisible' and k not in self.confirmed]
        if unconfirmed:QMessageBox.warning(self,'仍有待确认区域','请逐个确认：'+'、'.join(unconfirmed));return
        regions={}
        rois={}
        for k,_ in REGIONS:
            pts=self.polys.get(k) or self._placeholder_poly(k); self.polys[k]=pts
            regions[k]={'polygon':[[float(x),float(y)] for x,y in pts],'visibility':self.visibility.get(k,'visible')}
            a=np.asarray(pts,dtype=np.float32); x1,y1=a.min(axis=0); x2,y2=a.max(axis=0); rois[k]=[float(x1),float(y1),float(x2),float(y2)]
        ref=float(self.ref_time if self.ref_time is not None else getattr(self,'t',0))
        if ref > 5.0:
            ans=QMessageBox.question(self,'参考帧较晚','当前参考帧位于 %.1f 秒。动态跟踪将从该参考帧开始，前面的片段不会进入分析。\n正式实验建议在视频前 5 秒内选择清晰参考帧。\n仍然保存吗？' % ref)
            if ans != QMessageBox.Yes:return
        data={'roi_format_version':5,'image_width':self.w,'image_height':self.h,'reference_time_s':ref,
              'viewpoint':self.viewpoint,'regions':regions,'rois':rois,
              'tracking':{'enabled':True,'mode':'accuracy','eye_parent':'head','eyes':['left_eye','right_eye'],
                          'visibility_gating':True,'viewpoint_aware':True}}
        Path(self.out_path).parent.mkdir(parents=True,exist_ok=True); Path(self.out_path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); self.saved.emit(self.out_path); self.accept()
    def closeEvent(self,e): self.cap.release(); super().closeEvent(e)
