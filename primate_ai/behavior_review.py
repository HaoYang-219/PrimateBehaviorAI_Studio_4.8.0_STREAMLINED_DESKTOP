# coding: utf-8
"""Human review and compact research export for 1-3 synchronized camera sessions.

No PySide6 dependency: validators/export are testable headlessly. Labels are
screening annotations and never establish a medical diagnosis.
"""
from __future__ import annotations
import bisect
from array import array
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CODEBOOK = [
    {"label":"抽搐样短促运动","kind":"point"},
    {"label":"震颤样重复运动","kind":"state"},
    {"label":"身体摆动","kind":"state"},
    {"label":"自主肢体运动","kind":"state"},
    {"label":"整体活动","kind":"state"},
    {"label":"静止/休息","kind":"state"},
    {"label":"困倦样低活动","kind":"state"},
    {"label":"头部活动","kind":"state"},
    {"label":"眼部状态变化","kind":"point"},
    {"label":"理毛/抓挠","kind":"state"},
    {"label":"无法辨认","kind":"point"},
]
FIELDS = ['event_id','origin','label','start_s','end_s','peak_s','review','region','support_cameras','note','updated_utc']
REVIEW_OPTIONS = ('unreviewed','confirmed','normal','uncertain')


def _read_csv(path):
    path=Path(path)
    if not path.is_file(): return []
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def _atomic_csv(path, fields, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.review_',suffix='.csv',dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            wr.writeheader();wr.writerows(rows);f.flush();os.fsync(f.fileno())
        os.replace(name,str(path))
    finally:
        if os.path.exists(name):os.unlink(name)


def _numeric(x, default=0.):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError): return default


def _validate_event(e):
    e={k:e.get(k,'') for k in FIELDS}
    e['event_id']=str(e['event_id'])
    e['origin']=str(e['origin'] or 'manual')
    e['label']=str(e['label'] or '无法辨认')[:120]
    e['review']=e['review'] if e['review'] in REVIEW_OPTIONS else 'unreviewed'
    a=max(0.,_numeric(e['start_s']));b=max(a,_numeric(e['end_s'],a))
    e['start_s']=round(a,4);e['end_s']=round(b,4)
    e['peak_s']=round(min(b,max(a,_numeric(e['peak_s'],a))),4)
    e['note']=str(e.get('note',''))[:3000]
    e['updated_utc']=str(e['updated_utc'] or datetime.now(timezone.utc).isoformat())
    return e


class ReviewStore:
    def __init__(self,output_dir):
        self.root=Path(output_dir)
        self.path=self.root/'review_annotations.csv'
        self.events=self._load()

    def _load(self):
        if self.path.exists():
            return [_validate_event(x) for x in _read_csv(self.path)]
        events=[]
        for i,r in enumerate(_read_csv(self.root/'events.csv'),1):
            a=_numeric(r.get('start_s'));b=max(a,_numeric(r.get('end_s'),a))
            events.append(_validate_event({
                'event_id':str(r.get('event_id') or i),'origin':'AI',
                'label':str(r.get('candidate_label') or r.get('label') or '未知候选'),
                'start_s':a,'end_s':b,'peak_s':_numeric(r.get('peak_s'),a),
                'review':'unreviewed','region':r.get('primary_region',r.get('region','')),
                'support_cameras':r.get('support_cameras',''),
            }))
        return events

    def save(self):
        self.events=sorted([_validate_event(e) for e in self.events],key=lambda x:(x['start_s'],str(x['event_id'])))
        _atomic_csv(self.path,FIELDS,self.events)

    def update(self,index,**changes):
        if index<0 or index>=len(self.events):raise IndexError(index)
        row=dict(self.events[index]);row.update(changes)
        row['updated_utc']=datetime.now(timezone.utc).isoformat()
        self.events[index]=_validate_event(row);self.save()

    def add(self,label,start_s,end_s,note='',review='confirmed'):
        used=set(str(x['event_id']) for x in self.events)
        n=1
        while 'M%04d'%n in used:n+=1
        self.events.append(_validate_event({'event_id':'M%04d'%n,'origin':'manual',
                  'label':label,'start_s':start_s,'end_s':end_s,'peak_s':start_s,
                  'review':review,'note':note}))
        self.save()
        return 'M%04d'%n


def codebook_path(root):return Path(root)/'behavior_codebook.json'


def read_codebook(root):
    p=codebook_path(root)
    if not p.is_file():return list(DEFAULT_CODEBOOK)
    try:
        data=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(data,list):raise ValueError('Not list')
        entries=[]
        for x in data:
            if isinstance(x,dict) and str(x.get('label','')).strip():
                entries.append({'label':str(x['label']).strip()[:120],
                                'kind':'state' if x.get('kind')=='state' else 'point'})
        return entries or list(DEFAULT_CODEBOOK)
    except (OSError,ValueError):return list(DEFAULT_CODEBOOK)


def save_codebook(root,entries):
    p=codebook_path(root);p.parent.mkdir(parents=True,exist_ok=True)
    clean=[];seen=set()
    for x in entries:
        label=str(x.get('label','')).strip()[:120]
        if label and label not in seen:
            clean.append({'label':label,'kind':'state' if x.get('kind')=='state' else 'point'});seen.add(label)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)


def camera_sources(out_dir):
    """Return camera video sources associated with one analysis output."""
    out=Path(out_dir);m=out/'details'/'analysis_manifest.json'
    if m.exists():
        try:
            d=json.loads(m.read_text(encoding='utf-8'))
            root=Path(d.get('session_dir',''))
            cams=[]
            for n,x in enumerate(d.get('camera_inputs',[])):
                f=Path(str(x.get('video_path','')))
                if f.is_file():cams.append({'camera_id':x.get('camera_id','cam%d'%(n+1)),
                   'view':x.get('view',''), 'video_path':str(f), 'session_dir':str(root)})
            if cams:return cams[:3]
        except (OSError,ValueError,TypeError):pass
    # Single-video and old paired projects have no camera manifest.
    return []


class CameraTimeIndex:
    """Camera capture timestamps indexed by frame number, with FPS fallback."""
    def __init__(self,session_dir,camera_id,fps):
        self.fps=max(1.,float(fps or 30.))
        self.frames=array('I');self.times=array('d')
        root=Path(session_dir)
        candidates=list(root.glob('timestamps_%s*.csv'%camera_id))
        candidates+=list((root/'_metadata').glob('timestamps_%s*.csv'%camera_id))
        for fn in candidates:
            try:
                frames=array('I');times=array('d')
                with fn.open('r',encoding='utf-8-sig',newline='') as f:
                    for row in csv.DictReader(f):
                        if not row.get('frame_index') or not row.get('session_time_s'):continue
                        index=int(float(row['frame_index']))
                        tick=float(row['session_time_s'])
                        if index<0 or not math.isfinite(tick):raise ValueError('Invalid timestamp')
                        if frames and (index<=frames[-1] or tick<times[-1]):raise ValueError('Non-monotonic timestamps')
                        frames.append(index);times.append(tick)
                if len(frames)>=2:
                    self.frames=frames;self.times=times;break
            except (OSError,ValueError,KeyError,OverflowError):continue

    def frame_at(self,t):
        t=max(0.,float(t))
        if not self.times:return max(0,int(round(t*self.fps)))
        i=bisect.bisect_left(self.times,t)
        if i==0:return self.frames[0]
        if i==len(self.times):return self.frames[-1]
        return self.frames[i] if abs(self.times[i]-t)<abs(t-self.times[i-1]) else self.frames[i-1]

    def time_at(self,frame):
        frame=int(frame)
        if not self.frames:return frame/self.fps
        i=bisect.bisect_left(self.frames,frame)
        if i==0:return self.times[0]
        if i==len(self.frames):return self.times[-1]
        return self.times[i] if abs(self.frames[i]-frame)<abs(frame-self.frames[i-1]) else self.times[i-1]


def stage_windows(out_dir):
    """Return non-overlapping annotated observation stages, in Session seconds."""
    m=Path(out_dir)/'details'/'analysis_manifest.json'
    if not m.exists():return []
    try:root=Path(json.loads(m.read_text(encoding='utf-8')).get('session_dir',''))
    except (OSError,ValueError,TypeError):return []
    candidates=[root/'_metadata'/'analysis_windows.csv',root/'analysis_windows.csv']
    rows=[]
    for source in candidates:
        if source.exists():
            for x in _read_csv(source):
                a=_numeric(x.get('start_s'),-1.);b=_numeric(x.get('end_s'),-1.)
                if 0<=a<b:
                    rows.append({'start_s':a,'end_s':b,'condition':str(x.get('condition','')),
                                 'name':str(x.get('window_type',x.get('window_name','观察阶段')))})
            break
    return sorted(rows,key=lambda x:x['start_s'])


def export_workbook(out_dir, target=None):
    """One concise workbook; detailed per-frame CSV remains in details/."""
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.utils import get_column_letter
    root=Path(out_dir);target=Path(target) if target else root/'科研汇总.xlsx'
    book=Workbook();book.remove(book.active)
    manifest=root/'details'/'analysis_manifest.json'
    metadata={}
    if manifest.exists():
        try:metadata=json.loads(manifest.read_text(encoding='utf-8'))
        except (ValueError,OSError):pass
    def make(title,rows):
        sheet=book.create_sheet(title)
        if not rows:sheet.append(['无数据']);return
        cols=list(dict.fromkeys(k for r in rows for k in r.keys()))
        sheet.append(cols)
        def safe_cell(val):
            if isinstance(val,(dict,list)):val=str(val)
            if isinstance(val,str) and val[:1] in ('=','+','-','@'):
                return "'"+val
            return val
        for row in rows:sheet.append([safe_cell(row.get(k,'')) for k in cols])
        sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
        for cell in sheet[1]:
            cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='183947');cell.alignment=Alignment(wrap_text=True)
        for i,c in enumerate(cols,1):sheet.column_dimensions[get_column_letter(i)].width=min(38,max(13,len(c)*1.7))
    make('实验信息',[{'项目':root.name,'数据目录':str(root),'摄像头数量':metadata.get('camera_count',''),
           '基线范围':str(metadata.get('baseline_ranges_s','')),'说明':'视频自动检测为候选事件，需人工复核'}])
    make('阶段统计',_read_csv(root/'details'/'stage_summary.csv'))
    make('核心指标',_read_csv(root/'summary.csv'))
    store=ReviewStore(root)
    make('事件与人工复核',store.events)
    # Only manually confirmed events count toward reviewed event summaries.
    grouped={}
    for e in store.events:
        if e['review']!='confirmed':continue
        g=grouped.setdefault(e['label'],{'行为':'', '确认次数':0,'确认持续秒数':0.})
        g['行为']=e['label'];g['确认次数']+=1
        g['确认持续秒数']+=max(0.,float(e['end_s'])-float(e['start_s']))
    make('人工确认统计',list(grouped.values()))
    target.parent.mkdir(parents=True,exist_ok=True);book.save(str(target))
    return target


def export_batch_workbook(result_dirs, target):
    """Simple between-session comparison; no pooling of incomparable camera views."""
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill
    from openpyxl.utils import get_column_letter
    folders=[Path(x) for x in result_dirs if Path(x).is_dir()]
    book=Workbook();overview=book.active;overview.title='实验列表'
    overview.append(['实验','摄像头','候选事件','人工确认','阶段数','结果路径'])
    st=book.create_sheet('按阶段各视角统计');st.append(['实验','摄像头','阶段','条件','有效覆盖(s)','平均活动量','低活动比例','候选次数'])
    metrics=book.create_sheet('核心指标');metrics.append(['实验','摄像头','指标','值','原始记录'])
    labels=book.create_sheet('人工确认行为');labels.append(['实验','行为','开始(s)','结束(s)','持续(s)','说明'])
    for root in folders:
        metadata={}
        m=root/'details'/'analysis_manifest.json'
        if m.exists():
            try:metadata=json.loads(m.read_text(encoding='utf-8'))
            except (ValueError,OSError):pass
        events=_read_csv(root/'events.csv')
        reviewed=ReviewStore(root).events
        confirmed=[e for e in reviewed if e.get('review')=='confirmed']
        phases=_read_csv(root/'details'/'stage_summary.csv')
        overview.append([root.name,metadata.get('camera_count',''),len(events),len(confirmed),len(phases),str(root)])
        for r in phases:
            st.append([root.name,r.get('camera_id',''),r.get('stage',''),r.get('condition',''),
                       _numeric(r.get('covered_s')),_numeric(r.get('mean_activity_proxy')),
                       _numeric(r.get('low_motion_fraction')),_numeric(r.get('candidate_count'))])
        for r in _read_csv(root/'summary.csv'):
            metrics.append([root.name,r.get('camera_id',''),r.get('指标',r.get('metric','')),
                            r.get('数值',r.get('value','')),json.dumps(r,ensure_ascii=False)[:1200]])
        for r in confirmed:
            a=float(r['start_s']);b=float(r['end_s'])
            labels.append([root.name,r['label'],a,b,round(max(0.,b-a),3),r.get('note','')])
    for sheet in book:
        sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
        for c in sheet[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='183947')
        for c in range(1,sheet.max_column+1):sheet.column_dimensions[get_column_letter(c)].width=24 if c<6 else 43
    path=Path(target);path.parent.mkdir(parents=True,exist_ok=True);book.save(str(path));return path
