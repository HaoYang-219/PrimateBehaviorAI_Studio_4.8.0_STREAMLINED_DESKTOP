from __future__ import annotations
import contextlib
import io
import sys
import json
import html
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
from .models import AnalysisConfig
from .engine import analyze as engine


class CallbackWriter(io.TextIOBase):
    def __init__(self, cb):
        self.cb=cb; self.buf=''
    def write(self,s):
        if not s:return 0
        self.buf+=str(s)
        while '\n' in self.buf:
            line,self.buf=self.buf.split('\n',1)
            if self.cb is print:
                sys.__stdout__.write(line+'\n'); sys.__stdout__.flush()
            elif self.cb:
                self.cb(line)
        return len(s)
    def flush(self):
        if self.buf and self.cb is print:
            sys.__stdout__.write(self.buf+'\n'); sys.__stdout__.flush()
        elif self.buf and self.cb:
            self.cb(self.buf)
        self.buf=''


def _read_baseline_ranges(session_dir: Path):
    # 4.3.4+: stage windows are stored as technical metadata.
    candidates=[session_dir/'_metadata'/'analysis_windows.csv', session_dir/'analysis_windows.csv']
    for path in candidates:
        if not path.exists(): continue
        try: df=pd.read_csv(path)
        except Exception: continue
        ranges=[]
        for _,r in df.iterrows():
            wt=str(r.get('window_type','')).strip().lower()
            cond=str(r.get('condition','')).strip().lower()
            if wt!='baseline' and 'baseline' not in cond: continue
            try: a=float(r.get('start_s')); b=float(r.get('end_s'))
            except Exception: continue
            if np.isfinite(a) and np.isfinite(b) and b>a: ranges.append((a,b))
        if ranges: return ranges
    # Backward compatibility with 4.3.0-4.3.3.
    path=session_dir/'phase_blocks.csv'
    if not path.exists(): return []
    try: df=pd.read_csv(path)
    except Exception: return []
    ranges=[]
    for _,r in df.iterrows():
        ptype=str(r.get('phase_type','')).strip().lower(); pname=str(r.get('phase_name','')).strip().lower()
        if ptype!='baseline' and 'baseline' not in pname: continue
        try: a=float(r.get('start_session_time_s')); b=float(r.get('end_session_time_s'))
        except Exception: continue
        if np.isfinite(a) and np.isfinite(b) and b>a: ranges.append((a,b))
    return ranges


def _frame_session_clock(session_dir: Path, video_path: str, frame_ids, fps: float):
    """Map decoded frame numbers to captured host timestamps.

    Logged session_time_s is a host acquisition timestamp, not hardware exposure
    time. Fall back to nominal FPS only for legacy sessions without log files.
    """
    ids=np.asarray(frame_ids,dtype=float)
    stem=Path(video_path).stem
    for path in [session_dir/'_metadata'/('timestamps_%s.csv' % stem), session_dir/('timestamps_%s.csv' % stem)]:
        if not path.exists(): continue
        try:
            df=pd.read_csv(path,usecols=['frame_index','session_time_s'])
            df=df.replace([np.inf,-np.inf],np.nan).dropna().sort_values('frame_index').drop_duplicates('frame_index')
            if len(df)<2: continue
            frames=df.frame_index.to_numpy(float)
            ticks=df.session_time_s.to_numpy(float)
            # Invalid / nonmonotonic logs must not silently corrupt event times.
            if np.any(np.diff(frames)<=0) or np.any(np.diff(ticks)<0): continue
            dt=1.0/max(float(fps),1.)
            result=np.interp(ids,frames,ticks)
            result=np.where(ids<frames[0],ticks[0]+(ids-frames[0])*dt,result)
            result=np.where(ids>frames[-1],ticks[-1]+(ids-frames[-1])*dt,result)
            return result,'captured_frame_timestamps'
        except (OSError,ValueError,KeyError,TypeError):
            continue
    return ids/max(float(fps),1.0),'nominal_fps_fallback'


def _video_to_session(video_seconds, frame_ids, mapped_times, fps):
    """Convert detector event time to session clock using per-frame index mapping."""
    ids=np.asarray(frame_ids,dtype=float)
    ts=np.asarray(mapped_times,dtype=float)
    q=np.asarray(video_seconds,dtype=float)*max(float(fps),1.)
    if len(ids)<2: return q/max(float(fps),1.)
    return np.interp(q,ids,ts,left=ts[0],right=ts[-1])


def _subset_reference(frame_df: pd.DataFrame, ranges, video_offset_s: float, fps: float):
    if not ranges or not len(frame_df): return None
    session_t=frame_df['time_s'].to_numpy(dtype=float)+float(video_offset_s)
    mask=np.zeros(len(frame_df),dtype=bool)
    for a,b in ranges:
        mask |= (session_t>=float(a)) & (session_t<=float(b))
    bframe=frame_df.loc[mask].copy().reset_index(drop=True)
    # Require enough continuous material to make the 10 s long-window reference meaningful.
    if len(bframe) < max(60,int(round(float(fps)*8.0))): return None
    return bframe


def _subset_reference_at_times(frame_df, ranges, session_t, fps):
    if not ranges:return None
    valid=np.zeros(len(frame_df),dtype=bool)
    for a,b in ranges: valid |= (session_t>=a)&(session_t<b)
    bframe=frame_df.loc[valid].copy().reset_index(drop=True)
    return bframe if len(bframe)>=max(60,int(round(float(fps)*8.))) else None


def _label_window(t, windows: pd.DataFrame):
    if windows is None or not len(windows): return ''
    for _,r in windows.iterrows():
        try:
            a=float(r.get('start_session_time_s',r.get('start_s'))); b=float(r.get('end_session_time_s',r.get('end_s')))
        except Exception: continue
        if a <= t <= b:
            wt=str(r.get('window_name',r.get('phase_name',r.get('window_type',''))))
            cond=str(r.get('condition','')).strip()
            return (cond+' · '+wt) if cond and cond not in wt else wt
    return ''

def _fuse_events(all_events: pd.DataFrame, tolerance_s: float=.20):
    """Associate short event candidates without transitive chain merging.

    Each fused group contains at most one candidate per camera, and all peaks
    must lie within tolerance of its first event. Different stages are never
    merged. These are review candidates, not verified behavioral labels.
    """
    standard=['event_id','start_s','end_s','duration_s','peak_s','peak_score','baseline_rarity','review_priority','candidate_label','primary_region','confidence','behavior_state',
              'support_count','support_cameras','support_views','analysis_window','evidence_level']
    if all_events is None or not len(all_events): return pd.DataFrame(columns=standard)
    rows=[]
    def flush(cluster,label):
        if not cluster:return
        cg=pd.DataFrame(cluster)
        def numeric_max(name):
            if name not in cg:return 0.0
            vals=pd.to_numeric(cg[name],errors='coerce')
            return float(vals.max()) if vals.notna().any() else 0.0
        def mode(name,default=''):
            if name not in cg:return default
            m=cg[name].dropna().astype(str).mode()
            return str(m.iloc[0]) if len(m) else default
        peak_idx=pd.to_numeric(cg.peak_score,errors='coerce').fillna(-1).idxmax()
        start=float(pd.to_numeric(cg.session_start_s).min())
        end=float(pd.to_numeric(cg.session_end_s).max())
        count=int(cg.camera_id.astype(str).nunique())
        rows.append({'event_id':len(rows)+1,'start_s':start,'end_s':end,
            'duration_s':max(0.,end-start),'peak_s':float(cg.loc[peak_idx,'session_peak_s']),
            'peak_score':numeric_max('peak_score'),'baseline_rarity':numeric_max('baseline_rarity'),
            'review_priority':numeric_max('review_priority'),'candidate_label':str(label),
            'primary_region':mode('primary_region','unknown'),'confidence':numeric_max('confidence'),
            'behavior_state':mode('behavior_state'),'support_count':count,
            'support_cameras':','.join(sorted(cg.camera_id.astype(str).unique())),
            'support_views':','.join(sorted(cg.view.astype(str).unique())),
            'analysis_window':mode('analysis_window'),
            'evidence_level':'多视角候选' if count>=2 else '单视角待复核'})
    for label,g in all_events.groupby('candidate_label',dropna=False):
        g=g.sort_values('session_peak_s')
        cluster=[]; anchor=None; cameras=set(); phase=None; regions=set()
        for _,r in g.iterrows():
            t=float(r.session_peak_s); cam=str(r.camera_id)
            window=str(r.get('analysis_window',''))
            region=str(r.get('primary_region','unknown'))
            known=region not in ('unknown','whole_body','nan','')
            region_ok=(not known or not regions or region in regions or str(label) not in
                       ('twitch-like','eye-state-change-like','sway-like'))
            # Do not bridge separate actions, stage boundaries or different limbs.
            fit=(bool(cluster) and t-anchor<=float(tolerance_s) and
                 cam not in cameras and window==phase and region_ok)
            if not fit:
                flush(cluster,label); cluster=[]; cameras=set(); regions=set(); anchor=t; phase=window
            cluster.append(r.to_dict()); cameras.add(cam)
            if known: regions.add(region)
        flush(cluster,label)
    return pd.DataFrame(rows,columns=standard).sort_values('peak_s').reset_index(drop=True)


def _per_stage_summary(camera_id, view, frames, session_times, camera_events, windows, baseline_ranges=None):
    """One row per observation stage, using only that camera's valid time coverage."""
    if windows is None or not len(windows):return []
    activity=pd.to_numeric(frames.get('whole_body_motion',pd.Series(np.nan,index=frames.index)),errors='coerce').to_numpy(float)
    quality=pd.to_numeric(frames.get('frame_quality',pd.Series(np.nan,index=frames.index)),errors='coerce').to_numpy(float)
    t=np.asarray(session_times,dtype=float)
    ref_mask=np.zeros(len(t),dtype=bool)
    for ra,rb in (baseline_ranges or []): ref_mask|=(t>=float(ra))&(t<float(rb))
    ref_valid=ref_mask & np.isfinite(activity) & np.isfinite(quality) & (quality>=.55)
    if not ref_valid.any(): ref_valid=np.isfinite(activity) & np.isfinite(quality) & (quality>=.55)
    threshold=float(np.nanquantile(activity[ref_valid],.20)) if ref_valid.any() else 0.
    rows=[]
    for _,w in windows.iterrows():
        try: a=float(w.start_s); b=float(w.end_s)
        except (ValueError,TypeError,AttributeError): continue
        if not np.isfinite(a) or not np.isfinite(b) or b<=a: continue
        sel=(t>=a)&(t<b)
        count=int(sel.sum())
        if not count: continue
        valid=sel & np.isfinite(activity) & np.isfinite(quality) & (quality>=.55)
        # A generic low-motion fraction, not a diagnosis of drowsiness.
        x=activity[valid]
        if len(x):
            low=float(np.mean(x<=threshold)); mean=float(np.mean(x))
        else: low=float('nan');mean=float('nan')
        ev=camera_events
        if ev is not None and len(ev) and 'session_peak_s' in ev:
            peaks=pd.to_numeric(ev.session_peak_s,errors='coerce')
            ev=ev[(peaks>=a)&(peaks<b)]
            tw=int(ev.candidate_label.astype(str).eq('twitch-like').sum())
            trem=int(ev.candidate_label.astype(str).eq('tremor-like').sum())
            candidates=len(ev)
        else: tw=trem=candidates=0
        # Effective sampled time is a camera-coverage estimate, capped by stage length.
        observed=float(np.nanmax(t[sel])-np.nanmin(t[sel])) if count>1 else 0.
        observed=min(float(b-a),max(0.,observed))
        rows.append({'camera_id':camera_id,'view':view,'stage_id':str(w.get('source_id','')),
            'stage':str(w.get('window_type','')),'condition':str(w.get('condition','')),
            'start_s':a,'end_s':b,'stage_duration_s':b-a,'covered_s':round(observed,3),
            'frames':count,'valid_quality_ratio':round(float(valid.sum())/count,4),
            'mean_activity_proxy':round(mean,5),'low_motion_fraction':round(low,4),
            'candidate_count':int(candidates),'twitch_candidate_count':tw,
            'tremor_candidate_count':trem,
            'candidate_per_min':round(60.*candidates/max(observed,1.e-6),3) if observed>=1. else float('nan')})
    return rows


def _write_multicam_report(out: Path, cfg: AnalysisConfig, camera_rows, fused: pd.DataFrame, baseline_ranges, windows, stage_rows):
    cam_html=''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
        html.escape(str(x['camera_id'])), html.escape(str(x['view'])), html.escape(Path(x['video_path']).name), html.escape(str(x['baseline_mode']))) for x in camera_rows)
    event_html='<p>未发现达到当前阈值的候选事件。</p>'
    if len(fused):
        event_html='<table><tr><th>时间(s)</th><th>事件</th><th>区域</th><th>视角支持</th><th>支持路数</th><th>窗口</th><th>分数</th></tr>'
        for _,r in fused.head(300).iterrows():
            event_html += '<tr><td>%.2f</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%.3f</td></tr>' % (
                float(r.peak_s),html.escape(str(r.candidate_label)),html.escape(str(r.primary_region)),html.escape(str(r.support_views)),int(r.support_count),html.escape(str(r.analysis_window)),float(r.peak_score))
        event_html+='</table>'
    stage_html='<p>当前 Session 没有可用的已结束阶段或有效逐帧数据。</p>'
    if stage_rows:
        stage_html='<table><tr><th>视角</th><th>阶段</th><th>有效帧比例</th><th>活动量指标</th><th>低运动比例</th><th>候选次数</th><th>候选/分钟</th></tr>'
        for r in stage_rows:
            stage_html += '<tr><td>%s</td><td>%s</td><td>%.1f%%</td><td>%.4f</td><td>%.1f%%</td><td>%d</td><td>%s</td></tr>' % (
                html.escape(str(r['view'])),html.escape(str(r['condition'])),
                100*float(r['valid_quality_ratio']),float(r['mean_activity_proxy']),100*float(r['low_motion_fraction']),int(r['candidate_count']),
                ('%.2f'%float(r['candidate_per_min'])) if np.isfinite(r['candidate_per_min']) else '—')
        stage_html+='</table>'
    # Show observation integrity before presenting AI candidates as review evidence.
    capture_qc_path = Path(cfg.session_dir) / '_metadata' / 'capture_acceptance.json'
    capture_qc_html = '<p>此 Session 尚未生成采集验收报告。建议先在采集页面执行“采集质量验收”。</p>'
    if capture_qc_path.exists():
        try:
            audit = json.loads(capture_qc_path.read_text(encoding='utf-8'))
            value = html.escape(str(audit.get('overall_status', '未知')))
            caveats = (audit.get('warnings') or []) + (audit.get('errors') or [])
            evidence = ''.join('<li>%s</li>' % html.escape(str(x)) for x in caveats[:8])
            capture_qc_html = '<p>采集文件级验收：<b>%s</b>。%s</p>' % (value, ('<ul>'+evidence+'</ul>') if evidence else '未发现文件级异常。')
        except (OSError, ValueError, TypeError):
            capture_qc_html = '<p>采集验收元数据无法读取；请检查原始视频。</p>'

    # Compact descriptive within-camera comparison. Values are NOT averaged over views.
    trend_html = '<p>未记录足够的实验阶段供比较。</p>'
    if stage_rows:
        positive = [float(r['mean_activity_proxy']) for r in stage_rows if np.isfinite(float(r['mean_activity_proxy']))]
        ceiling = max(positive) if positive else 0.0
        if ceiling > 0:
            chunks = []
            for r in stage_rows:
                value = float(r['mean_activity_proxy'])
                if not np.isfinite(value):
                    continue
                amount = min(100., max(0., value / ceiling * 100.))
                chunks.append('<div style="margin:8px 0"><div>%s · %s：%.3f</div><div style="height:9px;background:#1a3644;border-radius:4px"><div style="height:9px;width:%.1f%%;background:#35b9ad;border-radius:4px"></div></div></div>' % (
                    html.escape(str(r['view'])), html.escape(str(r['condition'])), value, amount))
            if chunks:
                trend_html = ''.join(chunks)
    time_note= '；'.join('%s：%s' % (html.escape(str(r['camera_id'])), '逐帧时间戳' if r.get('clock_mode')=='captured_frame_timestamps' else '标称帧率估算') for r in camera_rows)
    br='；'.join('%.1f–%.1f s' % (a,b) for a,b in baseline_ranges) if baseline_ranges else '未找到已结束的 Baseline 阶段；各路使用自身视频进行探索性参考'
    doc='''<!doctype html><html><head><meta charset="utf-8"><title>多视角 Session 分析报告</title><style>
body{font-family:"Microsoft YaHei",Arial;background:#071018;color:#eaf2f7;margin:0;padding:28px}h1{margin:0 0 8px}.sub{color:#8ca2b0;margin-bottom:18px}.card{background:#0e1c27;border:1px solid #203b4d;border-radius:14px;padding:18px;margin:14px 0}.warn{background:#2a2513;border:1px solid #6b5920;padding:12px;border-radius:10px}table{border-collapse:collapse;width:100%%}th,td{border-bottom:1px solid #27465a;padding:8px;text-align:left}th{color:#9fb5c2}.tag{display:inline-block;background:#123044;padding:5px 9px;border-radius:7px;margin:3px}</style></head><body>
<h1>实验 Session · 1–3 路多视角行为分析</h1><div class="sub">PrimateBehaviorAI 4.8.0 · 连续原始视频 + 阶段时间戳虚拟切片 + 多视角候选融合</div>
<div class="warn"><b>说明：</b>当前多视角模式会对每一路分别进行 ROI/行为分析，再按 Session 时间轴把同类候选事件进行时间邻近融合。它不是 3D 重建，也不会把不同视角的数值终点盲目平均。多视角共同支持可提高复核优先级，但仍需专家确认。</div>
<div class="card"><h2>Session</h2><p>目录：%s</p><p>Baseline参考：%s</p><p>分析摄像头：%d 路</p></div>
<div class="card"><h2>原始采集质量</h2>%s</div>
<div class="card"><h2>阶段活动量趋势（相对标尺）</h2><p>条形长度仅辅助比较；不同摄像头画面尺度不同，不作跨视角定量比较。</p>%s</div>
<div class="card"><h2>按阶段量化（各视角独立）</h2><p>活动量为无单位代理指标；低运动比例以本机位 Baseline 的低活动阈值为参考（无 Baseline 时使用自身片段）。候选事件需人工复核。</p>%s</div>
<div class="card"><h2>时间基准</h2><p>%s</p></div>
<div class="card"><h2>摄像头与参考策略</h2><table><tr><th>摄像头</th><th>视角</th><th>视频</th><th>参考</th></tr>%s</table></div>
<div class="card"><h2>多视角候选事件</h2><p><span class="tag">1路支持：单视角候选</span><span class="tag">2–3路支持：优先复核</span></p>%s</div>
</body></html>''' % (html.escape(str(cfg.session_dir)),html.escape(br),len(camera_rows),capture_qc_html,trend_html,stage_html,time_note,cam_html,event_html)
    (out/'report.html').write_text(doc,encoding='utf-8')


def _run_session_multicam(cfg: AnalysisConfig):
    out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    details=out/'details'; details.mkdir(parents=True,exist_ok=True)
    session_dir=Path(cfg.session_dir)
    baseline_ranges=_read_baseline_ranges(session_dir)
    windows_path=session_dir/'_metadata'/'analysis_windows.csv'
    if not windows_path.exists(): windows_path=session_dir/'analysis_windows.csv'
    try: analysis_windows=pd.read_csv(windows_path) if windows_path.exists() else pd.DataFrame()
    except Exception: analysis_windows=pd.DataFrame()
    all_events=[]; camera_rows=[]; endpoint_rows=[]; stage_rows=[]
    for n,item in enumerate(cfg.camera_inputs,1):
        camera_id=str(item.get('camera_id') or ('cam%02d'%n)); view=str(item.get('view') or 'unknown')
        video=str(item['video_path']); roi=str(item['roi_path'])
        cam_out=details/camera_id; cam_out.mkdir(parents=True,exist_ok=True)
        print('[多视角] %s / %s：提取原始 FPS 多区域特征...' % (camera_id,view))
        info=engine.get_video_info(video); rois=engine.load_rois(roi,info.width,info.height)
        _,frame_df,_=engine.extract_frame_features(video,rois,analysis_width=cfg.analysis_width,progress=True,roi_path=roi,tracking_quality='accuracy')
        win=engine.make_multiscale_windows(frame_df,info.fps)
        session_t,clock_mode=_frame_session_clock(session_dir,video,frame_df.frame.to_numpy(float),info.fps)
        offset=float(session_t[0]-float(frame_df.time_s.iloc[0]))
        bframe=_subset_reference(frame_df,baseline_ranges,offset,info.fps) if clock_mode!='captured_frame_timestamps' else _subset_reference_at_times(frame_df,baseline_ranges,session_t,info.fps)
        if bframe is not None:
            bwin=engine.make_multiscale_windows(bframe,info.fps); self_ref=False; exploratory=False; baseline_mode='Session Baseline'
            ref_frame,ref_win=bframe,bwin
        else:
            ref_frame,ref_win=frame_df,win; self_ref=True; exploratory=True; baseline_mode='Self-reference exploratory'
        explain,events,states,s10,s30,minute,summary=engine.analyze_feature_tables(
            frame_df,win,ref_frame,ref_win,event_threshold=cfg.event_threshold,self_reference=self_ref,session='session',exploratory=exploratory)
        rsum=engine.make_region_summary(ref_win,win)
        engine.write_analysis_outputs(cam_out,frame_df,win,explain,events,states,s10,s30,minute,summary,rsum)
        engine.export_candidate_clips(video,events,cam_out/'candidate_clips')
        if cfg.make_dashboard:
            print('[多视角] %s：生成复核 Dashboard...' % camera_id)
            engine.annotate_video(video,explain,events,str(cam_out/'annotated_dashboard.mp4'),frame_df)
        ev=events.copy()
        if len(ev):
            ev['camera_id']=camera_id; ev['view']=view; ev['video_offset_s']=float(offset)
            ev['session_start_s']=_video_to_session(ev.start_s.astype(float),frame_df.frame,session_t,info.fps)
            ev['session_end_s']=_video_to_session(ev.end_s.astype(float),frame_df.frame,session_t,info.fps)
            ev['session_peak_s']=_video_to_session(ev.peak_s.astype(float),frame_df.frame,session_t,info.fps)
            ev['analysis_window']=[_label_window(float(t),analysis_windows) for t in ev.session_peak_s]
            all_events.append(ev)
        stage_rows.extend(_per_stage_summary(camera_id,view,frame_df,session_t,ev if len(ev) else pd.DataFrame(),analysis_windows,baseline_ranges))
        core=cam_out/'core_endpoints_cn.csv'
        if core.exists():
            cdf=pd.read_csv(core)
            cdf.insert(0,'view',view); cdf.insert(0,'camera_id',camera_id); endpoint_rows.append(cdf)
        camera_rows.append({'camera_id':camera_id,'view':view,'video_path':video,'roi_path':roi,'video_offset_s':offset,'clock_mode':clock_mode,'baseline_mode':baseline_mode})
    raw=pd.concat(all_events,ignore_index=True) if all_events else pd.DataFrame()
    if len(raw): raw.to_csv(details/'multicam_events_raw.csv',index=False,encoding='utf-8-sig')
    fused=_fuse_events(raw,.20)
    fused.to_csv(out/'events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(camera_rows).to_csv(details/'cameras.csv',index=False,encoding='utf-8-sig')
    if endpoint_rows: pd.concat(endpoint_rows,ignore_index=True).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    if stage_rows: pd.DataFrame(stage_rows).to_csv(details/'stage_summary.csv',index=False,encoding='utf-8-sig')
    manifest={'session_dir':str(session_dir),'camera_count':len(camera_rows),'baseline_ranges_s':baseline_ranges,'fusion_tolerance_s':.20,'fusion_method':'nontransitive_one_per_camera_stage_gated','camera_inputs':camera_rows,'note':'Continuous raw videos; stage timestamps define virtual analysis windows. Per-camera analysis followed by time-near same-label candidate fusion; not 3D reconstruction.'}
    (details/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    _write_multicam_report(out,cfg,camera_rows,fused,baseline_ranges,analysis_windows,stage_rows)
    # Return shape compatible with Worker/DB.
    return {'events':fused,'summary':pd.DataFrame([{'camera_count':len(camera_rows)}])}, fused


def run_analysis(cfg: AnalysisConfig, log: Optional[Callable[[str],None]]=None):
    cfg.validate(); out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    writer=CallbackWriter(log)
    with contextlib.redirect_stdout(writer),contextlib.redirect_stderr(writer):
        if cfg.mode=='session_multicam':
            res,events=_run_session_multicam(cfg)
            writer.flush(); return res,events
        if cfg.mode=='single':
            res=engine.run_one(cfg.single_path,out,cfg.roi_path,cfg.analysis_width,baseline_pack=None,
                               event_threshold=cfg.event_threshold,make_annotated=cfg.make_dashboard)
        else:
            bdir=out/'baseline_reference'; bdir.mkdir(parents=True,exist_ok=True)
            binfo=engine.get_video_info(cfg.before_path); brois=engine.load_rois(cfg.roi_path,binfo.width,binfo.height)
            print('[基线] 提取照射前原始 FPS 多区域特征...')
            _,bframe,_=engine.extract_frame_features(cfg.before_path,brois,analysis_width=cfg.analysis_width,
                progress=True,roi_path=cfg.roi_path,tracking_quality='accuracy')
            bwin=engine.make_multiscale_windows(bframe,binfo.fps)
            bexplain,bevents,bstates,bs10,bs30,bminute,bsummary=engine.analyze_feature_tables(
                bframe,bwin,bframe,bwin,event_threshold=cfg.event_threshold,self_reference=True,session='before',exploratory=False)
            engine.write_analysis_outputs(bdir,bframe,bwin,bexplain,bevents,bstates,bs10,bs30,bminute,bsummary,
                                          engine.make_region_summary(bwin,bwin))
            print('[照射后] 在行为状态条件下与个体基线比较...')
            res=engine.run_one(cfg.after_path,out,cfg.roi_path,cfg.analysis_width,baseline_pack=(bframe,bwin),
                               event_threshold=cfg.event_threshold,make_annotated=cfg.make_dashboard)
            comp=engine.make_before_after_comparison(bsummary,res['summary'])
            comp.to_csv(out/'before_after_comparison.csv',index=False)
            cmap={'activity_mean':'整体活动量','immobility_ratio':'静止时间占比','low_arousal_burden_ratio':'低觉醒/困倦样负荷','arousal_proxy_index':'觉醒代理指数','prolonged_immobility_max_s':'最长连续静止(s)','head_drop_ratio':'低头/头位下降占比','eye_closure_like_ratio':'眼睑闭合样占比','twitch_like_per_min':'抽搐样事件(次/min)','tremor_like_burden_ratio':'震颤样负荷','convulsion_like_per_min':'全身同步异常运动(次/min)','sway_like_per_min':'身体摆动(次/min)','upper_limb_asymmetry':'上肢活动不对称','lower_limb_asymmetry':'下肢活动不对称','postural_sway_proxy':'姿势摆动代理量'}
            cq=comp[comp.metric.isin(cmap)].copy(); cq['指标']=cq.metric.map(cmap); cq.to_csv(out/'before_after_core_cn.csv',index=False,encoding='utf-8-sig')
            s10=pd.read_csv(out/'statistics_10s.csv'); s30=pd.read_csv(out/'statistics_30s.csv'); minute=pd.read_csv(out/'minute_summary.csv')
            engine.save_html_report(out/'report.html',res['summary'],res['events'],s10,s30,minute,comp)
    writer.flush()
    events_path=out/'events.csv'; events=pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    return res,events
