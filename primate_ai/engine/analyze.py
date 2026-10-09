#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PrimateBehaviorAI Studio 4.2 EMF research analysis engine.

The engine is designed for visible behavioral phenotyping, not clinical diagnosis.
It preserves source FPS (30 FPS recommended), tracks anatomical ROIs, learns
behavior states from the baseline session, detects state-conditioned novelty,
and runs event-specific detectors at multiple time scales.
"""
from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from .tracking import DynamicROITracker, poly_bbox
except ImportError:
    from tracking import DynamicROITracker, poly_bbox

warnings.filterwarnings("ignore")
EPS = 1e-8

REGIONS = [
    "whole_body", "head", "left_eye", "right_eye", "torso",
    "left_upper_limb", "right_upper_limb",
    "left_lower_limb", "right_lower_limb",
]
BODY_REGIONS = ["head", "torso", "left_upper_limb", "right_upper_limb",
                "left_lower_limb", "right_lower_limb"]
LIMB_REGIONS = ["left_upper_limb", "right_upper_limb", "left_lower_limb", "right_lower_limb"]
EYE_REGIONS = ["left_eye", "right_eye"]
EVENT_TYPES = ["twitch-like", "tremor-like", "convulsion-like", "sway-like",
               "eye-state-change-like", "hypoactivity-like", "unknown-novelty"]

VISIBILITY_WEIGHT = {"visible":1.0, "partial":0.58, "invisible":0.0}
VIEWPOINT_CN = {"right_profile":"右侧位", "left_profile":"左侧位", "oblique":"斜侧位", "frontal":"正面/近正面", "unknown":"未指定"}
EVENT_CN = {"twitch-like":"抽搐样短促运动", "tremor-like":"震颤样重复运动", "convulsion-like":"全身同步异常运动",
            "sway-like":"身体摆动/姿势不稳", "eye-state-change-like":"眼部状态变化", "hypoactivity-like":"低活动/困倦相关",
            "unknown-novelty":"未知新颖行为"}
REGION_CN = {"whole_body":"全身", "head":"头部", "left_eye":"左眼", "right_eye":"右眼", "torso":"躯干",
             "left_upper_limb":"左上肢", "right_upper_limb":"右上肢", "left_lower_limb":"左下肢", "right_lower_limb":"右下肢", "unknown":"未知"}


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frames: int
    duration: float


def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: %s" % path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(path, fps, width, height, frames, frames / max(fps, EPS))


def clamp_box(box: Iterable[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1,y1,x2,y2 = [int(round(v)) for v in box]
    x1=max(0,min(w-2,x1)); y1=max(0,min(h-2,y1))
    x2=max(x1+2,min(w,x2)); y2=max(y1+2,min(h,y2))
    return x1,y1,x2,y2


def _box_poly(box):
    x1,y1,x2,y2=box
    return np.asarray([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)


def _split_legacy_eyes(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split an old broad Eyes ROI into two conservative eye masks.

    This is only backward compatibility. A new 4.2 calibration with left/right
    eyes drawn separately is strongly preferred.
    """
    x1,y1=poly.min(axis=0); x2,y2=poly.max(axis=0)
    w=x2-x1; h=y2-y1
    top=y1+.16*h; bot=y2-.16*h
    left=np.asarray([[x1+.03*w,top],[x1+.47*w,top],[x1+.47*w,bot],[x1+.03*w,bot]],np.float32)
    right=np.asarray([[x1+.53*w,top],[x1+.97*w,top],[x1+.97*w,bot],[x1+.53*w,bot]],np.float32)
    return left,right


def default_chair_polygons(w: int, h: int) -> Dict[str, np.ndarray]:
    f={
        "whole_body":(.20,.08,.72,.98), "head":(.35,.08,.58,.43),
        "left_eye":(.392,.17,.455,.255), "right_eye":(.465,.17,.535,.255),
        "torso":(.31,.32,.57,.77), "left_upper_limb":(.25,.33,.43,.70),
        "right_upper_limb":(.43,.35,.58,.92), "left_lower_limb":(.28,.58,.45,.98),
        "right_lower_limb":(.42,.58,.58,.98),
    }
    out={}
    for k,b in f.items():
        out[k]=_box_poly((b[0]*w,b[1]*h,b[2]*w,b[3]*h))
    return out


def _load_raw_roi_data(path: Optional[str]):
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_roi_context(path: Optional[str]):
    """Read camera viewpoint and per-region visibility priors.

    Old ROI files default to all-visible and unknown viewpoint. Visibility is an
    acquisition/annotation prior and is deliberately independent of image-motion
    confidence so a hidden eye/limb can never create an event by itself.
    """
    data=_load_raw_roi_data(path) if path else {}
    viewpoint=str(data.get("viewpoint","unknown") or "unknown")
    regions=data.get("regions",{}) or {}
    vis={}
    for r in REGIONS:
        item=regions.get(r,{}) if isinstance(regions,dict) else {}
        label=str(item.get("visibility","visible")) if isinstance(item,dict) else "visible"
        if label not in VISIBILITY_WEIGHT:label="visible"
        vis[r]=float(VISIBILITY_WEIGHT[label])
    return viewpoint,vis

def load_roi_polygons(path: Optional[str], w: int, h: int):
    """Load 4.0 polygons and transparently support 3.x eye/ROI formats."""
    if not path:
        return default_chair_polygons(w,h),0.0,False
    data=_load_raw_roi_data(path)
    src_w=max(1,int(data.get("image_width",w))); src_h=max(1,int(data.get("image_height",h)))
    sx=float(w)/src_w; sy=float(h)/src_h
    regions=data.get("regions",{}) or {}; legacy=data.get("rois",{}) or {}

    def item_poly(name):
        item=regions.get(name,{}) or {}
        pts=item.get("polygon") if isinstance(item,dict) else None
        if pts and len(pts)>=3:
            a=np.asarray(pts,dtype=np.float32)
            if float(np.max(a))<=1.0:
                a[:,0]*=w; a[:,1]*=h
            else:
                a[:,0]*=sx; a[:,1]*=sy
            return a
        b=legacy.get(name)
        if isinstance(b,list) and len(b)>=4:
            vals=np.asarray(b[:4],dtype=np.float32)
            if float(np.max(vals))<=1.0:
                vals[[0,2]]*=w; vals[[1,3]]*=h
            else:
                vals[[0,2]]*=sx; vals[[1,3]]*=sy
            return _box_poly(vals)
        return None

    out={}
    for r in [x for x in REGIONS if x not in EYE_REGIONS]:
        p=item_poly(r)
        if p is None:
            raise ValueError("ROI file is incomplete. Missing region: %s" % r)
        out[r]=p
    le=item_poly("left_eye"); re=item_poly("right_eye")
    if le is None or re is None:
        old=item_poly("eyes")
        if old is None:
            raise ValueError("ROI file is incomplete. Missing left_eye/right_eye (or legacy eyes).")
        le,re=_split_legacy_eyes(old)
    out["left_eye"]=le; out["right_eye"]=re
    enabled=bool((data.get("tracking",{}) or {}).get("enabled",True))
    ref=float(data.get("reference_time_s",data.get("frame_time_s",0.0)) or 0.0)
    return out,ref,enabled


def load_rois(path: Optional[str], w: int, h: int) -> Dict[str, Tuple[int,int,int,int]]:
    polys,_,_=load_roi_polygons(path,w,h)
    return {r:poly_bbox(p,w,h) for r,p in polys.items()}


def motion_from_diff(diff: np.ndarray) -> Dict[str,float]:
    if diff.size==0 or min(diff.shape[:2])<4:
        return dict(motion=0.,motion_peak=0.,active=0.,motion_cx=.5,motion_cy=.5,concentration=0.)
    motion=float(cv2.mean(diff)[0]/255.0); peak=float(np.max(diff)/255.0)
    _,mask=cv2.threshold(diff,12,255,cv2.THRESH_BINARY)
    active=float(cv2.countNonZero(mask)/max(1,diff.size))
    mm=cv2.moments(diff)
    if abs(mm["m00"])>EPS:
        cx=float(mm["m10"]/mm["m00"]/max(1,diff.shape[1]-1)); cy=float(mm["m01"]/mm["m00"]/max(1,diff.shape[0]-1))
    else: cx=cy=.5
    hh,ww=diff.shape[:2]; max_cell=0.
    for gy in range(3):
        for gx in range(3):
            y0,y1=gy*hh//3,(gy+1)*hh//3; x0,x1=gx*ww//3,(gx+1)*ww//3
            max_cell=max(max_cell,float(cv2.mean(diff[y0:y1,x0:x1])[0]/255.0))
    return dict(motion=motion,motion_peak=peak,active=active,motion_cx=cx,motion_cy=cy,
                concentration=float(max_cell/(motion+EPS)))


def eye_features(gray: np.ndarray) -> Dict[str,float]:
    if gray.size==0 or min(gray.shape[:2])<6:
        return dict(contrast=0.,sharp=0.,dark_ratio=0.,dark_height=0.,visibility=0.,brightness=0.)
    h,w=gray.shape[:2]
    if w>160:
        s=160.0/w; e=cv2.resize(gray,(160,max(12,int(round(h*s)))),interpolation=cv2.INTER_AREA)
    else: e=gray
    mean,std=cv2.meanStdDev(e); mu=float(mean[0,0]); sd=float(std[0,0])
    gx=cv2.Sobel(e,cv2.CV_16S,1,0,ksize=3); gy=cv2.Sobel(e,cv2.CV_16S,0,1,ksize=3)
    sharp=float((cv2.mean(cv2.convertScaleAbs(gx))[0]+cv2.mean(cv2.convertScaleAbs(gy))[0])/160.0)
    contrast=float(sd/64.0); thr=max(5.0,mu-.65*sd); dark=(e<thr).astype(np.uint8)
    dark_ratio=float(dark.mean()); row_frac=dark.mean(axis=1)
    rows=np.where(row_frac>max(.05,float(np.percentile(row_frac,65))))[0]
    dark_height=float((rows[-1]-rows[0]+1)/max(1,len(row_frac))) if len(rows)>=2 else 0.
    # Conservative visibility: tiny/flat/blurred crops are treated as unavailable.
    vis=float(np.clip(.45*min(1.,contrast)+.45*min(1.,sharp)+.10*min(1.,dark_ratio*8),0,1))
    return dict(contrast=contrast,sharp=sharp,dark_ratio=dark_ratio,dark_height=dark_height,
                visibility=vis,brightness=mu/255.0)


def _poly_crop_gray(frame: np.ndarray, poly: np.ndarray, target_w: int) -> np.ndarray:
    h,w=frame.shape[:2]; x1,y1,x2,y2=poly_bbox(poly,w,h); roi=frame[y1:y2,x1:x2]
    if roi.size==0: return np.zeros((8,8),np.uint8)
    local=poly.copy(); local[:,0]-=x1; local[:,1]-=y1
    mask=np.zeros(roi.shape[:2],np.uint8); cv2.fillPoly(mask,[np.round(local).astype(np.int32)],255)
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # Use local median outside the polygon to prevent background from driving motion.
    med=int(np.median(gray[mask>0])) if np.any(mask>0) else 0
    gray=np.where(mask>0,gray,med).astype(np.uint8)
    rh,rw=gray.shape[:2]
    if target_w>0 and rw>target_w:
        s=target_w/float(rw); gray=cv2.resize(gray,(target_w,max(8,int(round(rh*s)))),interpolation=cv2.INTER_AREA)
    return gray


def _crop_quality(gray: np.ndarray) -> Tuple[float,float]:
    if gray.size<64: return 0.,0.
    sharp=float(cv2.Laplacian(gray,cv2.CV_32F).var())
    brightness=float(gray.mean()/255.0)
    sharp_q=float(np.clip(sharp/220.0,0,1))
    bright_q=float(np.clip(1.0-abs(brightness-.5)/.5,0,1))
    return sharp_q,bright_q


def extract_frame_features(video: str, rois_orig: Dict[str,Tuple[int,int,int,int]],
                           analysis_width: int=600, progress: bool=True,
                           roi_path: Optional[str]=None, tracking_quality: str="accuracy"):
    """Extract source-FPS features with viewpoint/visibility-aware quality gating."""
    info=get_video_info(video); cap=cv2.VideoCapture(video)
    polygons,ref_time,tracking_enabled=load_roi_polygons(roi_path,info.width,info.height)
    viewpoint,visibility_priors=load_roi_context(roi_path)
    tracker=None; start_frame=0; pending=None
    if tracking_enabled and polygons:
        start_frame=max(0,min(info.frames-1,int(round(ref_time*info.fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame); ok,first=cap.read()
        if not ok: raise RuntimeError("Could not read ROI reference frame")
        tracker=DynamicROITracker(first,polygons,quality=tracking_quality,visibility_priors=visibility_priors); pending=first
    else:
        polygons={r:_box_poly(rois_orig[r]) for r in REGIONS}; start_frame=0
    widths={
        "whole_body":max(320,int(analysis_width)), "head":max(260,int(analysis_width*.62)),
        "left_eye":max(180,int(analysis_width*.36)), "right_eye":max(180,int(analysis_width*.36)),
        "torso":max(260,int(analysis_width*.62)), "left_upper_limb":max(280,int(analysis_width*.64)),
        "right_upper_limb":max(280,int(analysis_width*.64)), "left_lower_limb":max(280,int(analysis_width*.64)),
        "right_lower_limb":max(280,int(analysis_width*.64)),
    }
    rows=[]; prev={}; frame_idx=start_frame; processed=0
    while True:
        if pending is not None:
            frame=pending; pending=None; track_results={}; current_polys={r:p.copy() for r,p in polygons.items()}
        else:
            ok,frame=cap.read()
            if not ok: break
            if tracker is not None:
                track_results=tracker.update(frame); current_polys=tracker.current_polygons()
            else:
                track_results={}; current_polys=polygons
        row={"frame":frame_idx,"time_s":frame_idx/info.fps}
        gray_rois={r:_poly_crop_gray(frame,current_polys[r],widths[r]) for r in REGIONS}
        for r in REGIONS:
            prior=float(visibility_priors.get(r,1.0)); g=gray_rois[r]
            if r not in prev or prior<=.01:
                m=dict(motion=0.,motion_peak=0.,active=0.,motion_cx=.5,motion_cy=.5,concentration=0.)
            else:
                pg=prev[r]
                if pg.shape!=g.shape: pg=cv2.resize(pg,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_AREA)
                m=motion_from_diff(cv2.absdiff(pg,g))
            for k,v in m.items(): row["%s_%s"%(r,k)]=float(v)*prior if k in ("motion","motion_peak","active") else v
            b=poly_bbox(current_polys[r],info.width,info.height); x1,y1,x2,y2=b
            row["%s_x1"%r]=x1; row["%s_y1"%r]=y1; row["%s_x2"%r]=x2; row["%s_y2"%r]=y2
            if r in track_results:
                tr=track_results[r]; bw=max(2,x2-x1); bh=max(2,y2-y1)
                row["%s_track_conf"%r]=float(tr.confidence)
                row["%s_track_dx"%r]=float(tr.dx/bw)*prior; row["%s_track_dy"%r]=float(tr.dy/bh)*prior
                row["%s_track_speed"%r]=float(np.hypot(tr.dx/bw,tr.dy/bh))*prior; row["%s_track_scale"%r]=float(tr.scale)
            else:
                row["%s_track_conf"%r]=(1.0 if tracker is None else .5)*prior
                row["%s_track_dx"%r]=0.; row["%s_track_dy"%r]=0.; row["%s_track_speed"%r]=0.; row["%s_track_scale"%r]=1.
            row["%s_visibility_prior"%r]=prior
            row["%s_visibility"%r]=float(np.clip(prior*row["%s_track_conf"%r]/max(prior,EPS),0,1)) if prior>.01 else 0.
        for er in EYE_REGIONS:
            prior=float(visibility_priors.get(er,1.0)); ef=eye_features(gray_rois[er])
            for k,v in ef.items(): row["%s_%s"%(er,k)]=v
            if er in prev and prior>.01:
                pe=prev[er]
                if pe.shape!=gray_rois[er].shape: pe=cv2.resize(pe,(gray_rois[er].shape[1],gray_rois[er].shape[0]),interpolation=cv2.INTER_AREA)
                row["%s_change"%er]=float(cv2.mean(cv2.absdiff(pe,gray_rois[er]))[0]/255.0)*prior
            else: row["%s_change"%er]=0.
            row["%s_visibility"%er]=float(np.clip(row["%s_visibility"%er]*row["head_track_conf"]*prior,0,1)) if prior>.01 else 0.
        # Relative head height is view-robust enough for within-animal baseline comparison.
        wb_h=max(4.,float(row["whole_body_y2"]-row["whole_body_y1"])); hc=.5*(row["head_y1"]+row["head_y2"]); tc=.5*(row["torso_y1"]+row["torso_y2"])
        row["head_drop_index"]=float((hc-tc)/wb_h)
        sharp_q,bright_q=_crop_quality(gray_rois["whole_body"])
        visible_body=[r for r in ["head","torso"]+LIMB_REGIONS if visibility_priors.get(r,1.0)>.05]
        body_confs=[row["%s_track_conf"%r] for r in visible_body] or [row["whole_body_track_conf"]]
        row["tracking_quality"]=float(np.median(body_confs)); row["frame_sharpness_quality"]=sharp_q; row["frame_brightness_quality"]=bright_q
        row["frame_quality"]=float(np.clip(.70*row["tracking_quality"]+.20*sharp_q+.10*bright_q,0,1))
        rows.append(row); prev=gray_rois; processed+=1; frame_idx+=1
        if progress and processed%max(1,int(info.fps*10))==0:
            print("  processed %6.1fs / %6.1fs"%(processed/info.fps,max(0.,info.duration-start_frame/info.fps)),flush=True)
    cap.release()
    if not rows: raise RuntimeError("No frames decoded")
    df=pd.DataFrame(rows)
    for r in BODY_REGIONS:
        prior=float(visibility_priors.get(r,1.0))
        internal=np.maximum(0.,df["%s_motion"%r].to_numpy()-0.45*df["whole_body_motion"].to_numpy())
        trajectory=np.maximum(0.,df["%s_track_speed"%r].to_numpy()-0.35*df["whole_body_track_speed"].to_numpy())
        df["%s_residual"%r]=(internal+.75*trajectory)
    for er in EYE_REGIONS:
        prior=float(visibility_priors.get(er,1.0)); df["%s_residual"%er]=np.maximum(0.,df["%s_motion"%er].to_numpy()-.55*df["head_motion"].to_numpy())
    df.attrs["coverage_start_s"]=float(df.time_s.iloc[0]); df.attrs["coverage_end_s"]=float(df.time_s.iloc[-1]); df.attrs["viewpoint"]=viewpoint; df.attrs["visibility_priors"]=visibility_priors
    return info,df,{r:poly_bbox(polygons[r],info.width,info.height) for r in REGIONS}

def band_power(signal: np.ndarray, fps: float, lo: float, hi: float):
    x=np.asarray(signal,dtype=np.float32)
    if x.size<8 or np.std(x)<1e-8: return 0.,0.
    x=x-x.mean(); spec=np.abs(np.fft.rfft(x*np.hanning(len(x)).astype(np.float32)))**2
    freq=np.fft.rfftfreq(len(x),d=1.0/fps); valid=freq>0; total=float(spec[valid].sum())+EPS
    band=(freq>=lo)&(freq<=hi)
    if not np.any(band): return 0.,0.
    idx=np.where(band)[0]; return float(spec[band].sum()/total),float(freq[idx[np.argmax(spec[idx])]])


def sign_reversal_rate(x: np.ndarray) -> float:
    x=np.asarray(x,dtype=np.float32)
    if len(x)<4:return 0.
    d=np.diff(x); s=np.sign(d); valid=(np.abs(d[1:])>1e-8)&(np.abs(d[:-1])>1e-8)
    if not np.any(valid):return 0.
    return float(np.mean((s[1:]*s[:-1]<0)[valid]))


def _window(df: pd.DataFrame, center: int, length: int):
    # Intentionally do NOT shift truncated edge windows. Coverage then stays <=1.
    half=length//2; a=max(0,center-half); b=min(len(df),center-half+length)
    return df.iloc[a:b],float(max(0,b-a))/max(1,length)


def make_multiscale_windows(df: pd.DataFrame, fps: float, short_s: float=.40,
                            mid_s: float=2.0, long_s: float=10.0, hop_s: float=.20) -> pd.DataFrame:
    n=len(df); hop=max(1,int(round(hop_s*fps))); ns=max(6,int(round(short_s*fps)))
    nm=max(ns,int(round(mid_s*fps))); nl=max(nm,int(round(long_s*fps))); rows=[]; all_local=BODY_REGIONS+EYE_REGIONS
    for center in range(0,n,hop):
        sw,scov=_window(df,center,ns); mw,mcov=_window(df,center,nm); lw,lcov=_window(df,center,nl)
        row={"center_s":float(df.iloc[center].time_s),"short_start_s":float(sw.iloc[0].time_s),"short_end_s":float(sw.iloc[-1].time_s),
             "mid_start_s":float(mw.iloc[0].time_s),"mid_end_s":float(mw.iloc[-1].time_s),"long_start_s":float(lw.iloc[0].time_s),"long_end_s":float(lw.iloc[-1].time_s),
             "short_coverage":min(1.,scov),"mid_coverage":min(1.,mcov),"long_coverage":min(1.,lcov)}
        row["S_whole_body_motion_mean"]=float(sw["whole_body_motion"].mean()) if len(sw) else 0.; row["S_whole_body_motion_max"]=float(sw["whole_body_motion"].max()) if len(sw) else 0.
        for r in all_local:
            sig=sw["%s_residual"%r].to_numpy(dtype=np.float32); row["S_%s_res_mean"%r]=float(sig.mean()) if len(sig) else 0.; row["S_%s_res_max"%r]=float(sig.max()) if len(sig) else 0.
            if len(sig)>=6:
                vel=np.diff(sig)*fps; acc=np.diff(vel)*fps; jerk=np.diff(acc)*fps if len(acc)>=2 else np.zeros(1); row["S_%s_acc"%r]=float(np.mean(np.abs(acc))) if len(acc) else 0.; row["S_%s_jerk"%r]=float(np.mean(np.abs(jerk))) if len(jerk) else 0.; row["S_%s_return"%r]=float(max(0.,sig.max()-.5*(sig[0]+sig[-1])))
            else: row["S_%s_acc"%r]=row["S_%s_jerk"%r]=row["S_%s_return"%r]=0.
            dx=sw.get("%s_track_dx"%r,pd.Series(np.zeros(len(sw)))).to_numpy(dtype=np.float32); row["S_%s_reversal"%r]=sign_reversal_rate(dx); row["S_%s_locality"%r]=float((sig.mean()+EPS)/(sw["whole_body_motion"].mean()+sig.mean()+EPS))
        for r in all_local:
            sig=mw["%s_residual"%r].to_numpy(dtype=np.float32); hi=min(12.0,max(2.1,fps/2.0-.2)); bp,peak=band_power(sig,fps,2.0,hi); lbp,lpeak=band_power(sig,fps,.3,2.0)
            row["M_%s_hf_power"%r]=bp; row["M_%s_hf_peak_hz"%r]=peak; row["M_%s_lf_power"%r]=lbp; row["M_%s_lf_peak_hz"%r]=lpeak; row["M_%s_reversal"%r]=sign_reversal_rate(sig); row["M_%s_activity"%r]=float(sig.mean()) if len(sig) else 0.
            if "%s_visibility"%r in mw: row["M_%s_visibility"%r]=float(mw["%s_visibility"%r].median())
            else: row["M_%s_visibility"%r]=1.0
        for r in ["whole_body"]+BODY_REGIONS:
            arr=lw["%s_motion"%r].to_numpy(dtype=np.float32); row["L_%s_activity"%r]=float(arr.mean()) if len(arr) else 0.
        row["L_head_drop_index"]=float(lw["head_drop_index"].median()) if "head_drop_index" in lw else 0.
        row["M_tracking_quality"]=float(mw.tracking_quality.mean()); row["M_frame_quality"]=float(mw.frame_quality.mean()); row["L_tracking_quality"]=float(lw.tracking_quality.mean()); row["L_frame_quality"]=float(lw.frame_quality.mean())
        for er in EYE_REGIONS:
            row["M_%s_change"%er]=float(mw["%s_change"%er].mean()); row["M_%s_dark_height"%er]=float(mw["%s_dark_height"%er].mean()); row["M_%s_dark_ratio"%er]=float(mw["%s_dark_ratio"%er].mean())
        rows.append(row)
    out=pd.DataFrame(rows); out.attrs.update(getattr(df,'attrs',{})); return out

def _numeric_feature_columns(df: pd.DataFrame):
    ignore={"center_s","short_start_s","short_end_s","mid_start_s","mid_end_s","long_start_s","long_end_s",
            "short_coverage","mid_coverage","long_coverage"}
    return [c for c in df.columns if c not in ignore and np.issubdtype(df[c].dtype,np.number)]


def _midrank_percentile(value: float, ref: np.ndarray) -> float:
    a=np.asarray(ref,dtype=float); a=a[np.isfinite(a)]
    if len(a)==0 or not np.isfinite(value):return .5
    less=float(np.sum(a<value)); equal=float(np.sum(np.isclose(a,value,rtol=1e-6,atol=1e-9)))
    return float((less+.5*equal)/len(a))


def _percentile_array(values: np.ndarray, ref: np.ndarray):
    return np.asarray([_midrank_percentile(float(v),ref) for v in values],dtype=np.float32)


def _two_sided_percentile(value: float, ref: np.ndarray) -> float:
    p=_midrank_percentile(value,ref); return float(min(1.,2.*abs(p-.5)))


def _choose_state_count(X: np.ndarray) -> int:
    if len(X)<50 or np.nanstd(X)<1e-6:return 1
    max_k=min(5,max(2,len(X)//60)); best_k=2; best=-1e9
    sample=X if len(X)<=500 else X[np.linspace(0,len(X)-1,500).astype(int)]
    for k in range(2,max_k+1):
        try:
            km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(sample)
            if len(set(km.labels_))<2:continue
            s=float(silhouette_score(sample,km.labels_));
            if s>best:best=s;best_k=k
        except Exception:pass
    return best_k if best>0.08 else 1


def _state_name(row: pd.Series, baseline: pd.DataFrame):
    wb=float(row.get("L_whole_body_activity",0)); q30=float(baseline["L_whole_body_activity"].quantile(.30)); q75=float(baseline["L_whole_body_activity"].quantile(.75))
    if wb<=q30:return "Low activity / rest","低活动/静止"
    upper=float(row.get("L_left_upper_limb_activity",0)+row.get("L_right_upper_limb_activity",0))
    lower=float(row.get("L_left_lower_limb_activity",0)+row.get("L_right_lower_limb_activity",0))
    head=float(row.get("L_head_activity",0)); torso=float(row.get("L_torso_activity",0))
    if wb>=q75 and torso>=max(upper/2,lower/2):return "Whole-body active","全身活动"
    if upper>max(lower,head*1.3):return "Upper-limb active","上肢活动"
    if lower>max(upper,head*1.3):return "Lower-limb active","下肢活动"
    if head>max(upper/2,lower/2):return "Head-oriented activity","头部/朝向活动"
    return "Mixed activity","混合活动"


def fit_behavior_states(train: pd.DataFrame, target: pd.DataFrame):
    cols=["L_whole_body_activity","L_head_activity","L_torso_activity","L_left_upper_limb_activity",
          "L_right_upper_limb_activity","L_left_lower_limb_activity","L_right_lower_limb_activity"]
    X=train[cols].fillna(0).to_numpy(np.float32); Q=target[cols].fillna(0).to_numpy(np.float32)
    scaler=StandardScaler().fit(X); Xs=scaler.transform(X); Qs=scaler.transform(Q); k=_choose_state_count(Xs)
    if k==1:
        tr=np.zeros(len(train),dtype=int); tq=np.zeros(len(target),dtype=int); centers=np.zeros((1,len(cols)),np.float32)
    else:
        km=KMeans(n_clusters=k,n_init=20,random_state=0).fit(Xs); tr=km.labels_.astype(int); tq=km.predict(Qs).astype(int); centers=km.cluster_centers_
    names={}; rows=[]
    for s in sorted(set(tr.tolist())):
        subset=train.iloc[np.where(tr==s)[0]]; proto=subset.median(numeric_only=True); en,cn=_state_name(proto,train); names[s]=(en,cn)
        rows.append(dict(state_id=s,state_name=en,state_name_cn=cn,windows=int(np.sum(tr==s))))
    return tr,tq,names,pd.DataFrame(rows),dict(columns=cols,scaler=scaler,centers=centers)


def _representation(train: pd.DataFrame,target: pd.DataFrame):
    cols=_numeric_feature_columns(train); X0=train[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    Q0=target[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)
    sc=StandardScaler().fit(X0); Xs=sc.transform(X0); Qs=sc.transform(Q0)
    dim=max(2,min(18,Xs.shape[1],max(2,Xs.shape[0]-1))); pca=PCA(n_components=dim,random_state=0).fit(Xs)
    return pca.transform(Xs),pca.transform(Qs)


def state_conditioned_scores(train: pd.DataFrame,target: pd.DataFrame,train_state: np.ndarray,target_state: np.ndarray,self_reference=False):
    X,Q=_representation(train,target); state_dev=np.zeros(len(target),np.float32); novelty=np.zeros(len(target),np.float32)
    # Global novelty reference (leave-self-out distances).
    kg=max(2,min(7,len(X)-1)) if len(X)>2 else 1; ng=NearestNeighbors(n_neighbors=min(len(X),kg+1)).fit(X)
    d0=ng.kneighbors(X)[0]; ref_global=d0[:,1:].mean(1) if d0.shape[1]>1 else d0[:,0]
    if self_reference and len(target)==len(train): q_global=ref_global
    else:
        qd=NearestNeighbors(n_neighbors=min(len(X),max(1,kg))).fit(X).kneighbors(Q)[0]; q_global=qd.mean(1)
    novelty=_percentile_array(q_global,ref_global)
    for s in sorted(set(target_state.tolist())):
        ti=np.where(target_state==s)[0]; bi=np.where(train_state==s)[0]
        if len(bi)<4:
            state_dev[ti]=novelty[ti]; continue
        B=X[bi]; k=max(2,min(7,len(B)-1)); nn=NearestNeighbors(n_neighbors=min(len(B),k+1)).fit(B)
        bd=nn.kneighbors(B)[0]; bref=bd[:,1:].mean(1) if bd.shape[1]>1 else bd[:,0]
        if self_reference and len(target)==len(train) and np.array_equal(ti,bi): qd=bref
        else:
            qd=NearestNeighbors(n_neighbors=min(len(B),max(1,k))).fit(B).kneighbors(Q[ti])[0].mean(1)
        state_dev[ti]=_percentile_array(qd,bref)
    return state_dev,novelty


def _p(row, col, train):
    if col not in train:return .5
    return _midrank_percentile(float(row.get(col,0.)),train[col].to_numpy(dtype=float))


def _event_raw_table(train: pd.DataFrame, target: pd.DataFrame):
    """Per-window event morphology with region-visibility gating."""
    rows=[]
    for _,r in target.iterrows():
        d={}; best=(0.,"unknown")
        for rg in BODY_REGIONS:
            if float(r.get("M_%s_visibility"%rg,1.0))<.30:continue
            raw=.30*_p(r,"S_%s_jerk"%rg,train)+.25*_p(r,"S_%s_return"%rg,train)+.20*_p(r,"S_%s_res_max"%rg,train)+.15*_p(r,"S_%s_reversal"%rg,train)+.10*_p(r,"S_%s_locality"%rg,train)
            if raw>best[0]:best=(raw,rg)
        d["twitch_pattern_score"],d["twitch_region"]=best
        best=(0.,"unknown")
        for rg in BODY_REGIONS:
            if float(r.get("M_%s_visibility"%rg,1.0))<.30:continue
            raw=.50*_p(r,"M_%s_hf_power"%rg,train)+.25*_p(r,"M_%s_reversal"%rg,train)+.25*_p(r,"M_%s_activity"%rg,train)
            if raw>best[0]:best=(raw,rg)
        d["tremor_pattern_score"],d["tremor_region"]=best
        visible=[rg for rg in BODY_REGIONS if float(r.get("M_%s_visibility"%rg,1.0))>=.30]
        ps=np.asarray([_p(r,"S_%s_res_max"%rg,train) for rg in visible],dtype=float) if visible else np.zeros(0)
        bodyp=_p(r,"S_whole_body_motion_max",train)
        if len(ps)>=3:
            sync=float(np.mean(ps>.92)); top=np.sort(ps)[-min(4,len(ps)):]
            conv=float(.45*np.mean(top)+.35*sync+.20*bodyp) if sync>=.50 and bodyp>=.90 else float(.30*np.mean(top)+.20*sync+.15*bodyp)
        else:conv=0.
        d["convulsion_pattern_score"]=min(1.,conv); d["convulsion_region"]="whole_body"
        if float(r.get("M_torso_visibility",1.0))>=.30:
            d["sway_pattern_score"]=float(.55*_p(r,"M_torso_lf_power",train)+.20*_p(r,"M_torso_reversal",train)+.15*_p(r,"M_torso_activity",train)+.10*_p(r,"M_head_lf_power",train))
        else:d["sway_pattern_score"]=0.
        d["sway_region"]="torso"
        ebest=(0.,"unknown")
        for er in EYE_REGIONS:
            vis=float(r.get("M_%s_visibility"%er,0.))
            if vis<.35:continue
            pchg=_p(r,"M_%s_change"%er,train); ph=_two_sided_percentile(float(r.get("M_%s_dark_height"%er,0.)),train["M_%s_dark_height"%er].to_numpy(float)); pr=_two_sided_percentile(float(r.get("M_%s_dark_ratio"%er,0.)),train["M_%s_dark_ratio"%er].to_numpy(float)); raw=.55*pchg+.25*ph+.20*pr
            if raw>ebest[0]:ebest=(raw,er)
        d["eye_pattern_score"],d["eye_region"]=ebest
        pa=_p(r,"L_whole_body_activity",train); d["hypoactivity_pattern_score"]=float(1.-pa); d["hypoactivity_region"]="whole_body"; rows.append(d)
    return pd.DataFrame(rows,index=target.index)

def build_explain(train: pd.DataFrame,target: pd.DataFrame,self_reference=False):
    train_state,target_state,names,state_table,_=fit_behavior_states(train,target)
    state_dev,novelty=state_conditioned_scores(train,target,train_state,target_state,self_reference=self_reference)
    raw_train=_event_raw_table(train,train); raw_target=_event_raw_table(train,target)
    out=target[["center_s","short_start_s","short_end_s","mid_start_s","mid_end_s","long_start_s","long_end_s",
                "short_coverage","mid_coverage","long_coverage"]].copy()
    out["behavior_state_id"]=target_state
    out["behavior_state"]=[names.get(int(s),("Mixed activity","混合活动"))[0] for s in target_state]
    out["behavior_state_cn"]=[names.get(int(s),("Mixed activity","混合活动"))[1] for s in target_state]
    out["state_deviation_percentile"]=state_dev; out["state_novelty_percentile"]=novelty
    out["quality_score"]=np.clip(.65*target["M_tracking_quality"].to_numpy(float)+.35*target["M_frame_quality"].to_numpy(float),0,1)
    out["quality_status"]=np.where(out.quality_score>=.55,"OK","UNAVAILABLE")
    mapping=[("twitch","short_coverage"),("tremor","mid_coverage"),("convulsion","short_coverage"),("sway","mid_coverage"),("eye","mid_coverage"),("hypoactivity","long_coverage")]
    for key,covcol in mapping:
        pcol=key+"_pattern_score"; rcol=key+"_region"
        out[pcol]=raw_target[pcol].to_numpy(float); out[rcol]=raw_target[rcol].astype(str).to_numpy()
        ref=raw_train[pcol].to_numpy(float); vals=raw_target[pcol].to_numpy(float)
        rarity=_percentile_array(vals,ref)
        out[key+"_baseline_rarity"]=rarity
        # Event score emphasizes actual pattern while retaining baseline rarity.
        out[key+"_score"]=np.clip(.65*vals+.35*rarity,0,1)
        badcov=out[covcol].to_numpy(float)<.85; out.loc[badcov,key+"_score"]=0.; out.loc[badcov,key+"_baseline_rarity"]=0.
    # Eye-specific quality gate; body event quality uses overall QC.
    eyevis=[]
    for i,r in target.iterrows():
        rg=str(out.loc[i,"eye_region"]); eyevis.append(float(r.get("M_%s_visibility"%rg,0.)) if rg in EYE_REGIONS else 0.)
    out["eye_visibility_at_event"]=eyevis
    out.loc[out.eye_visibility_at_event<.35,["eye_score","eye_baseline_rarity"]]=0.
    bad=out.quality_score<.55
    for key,_ in mapping:
        out.loc[bad,[key+"_score",key+"_baseline_rarity"]]=0.
    # Unknown novelty is intentionally separate from named event detectors.
    specific=np.max(np.column_stack([out[k+"_score"].to_numpy(float) for k,_ in mapping]),axis=1)
    out["unknown_score"]=np.where((np.maximum(state_dev,novelty)>.99)&(specific<.80)&(out.long_coverage.to_numpy(float)>=.90),np.maximum(state_dev,novelty),0.)
    out["unknown_region"]="unknown"
    # Pick current explanation but DO NOT call ordinary motion HIGH.
    labels=[];regions=[];patterns=[];rarities=[];att=[];conf=[]
    label_map={"twitch":"twitch-like","tremor":"tremor-like","convulsion":"convulsion-like","sway":"sway-like",
               "eye":"eye-state-change-like","hypoactivity":"hypoactivity-like"}
    for i in range(len(out)):
        candidates=[]
        for key,_ in mapping:
            score=float(out.iloc[i][key+"_score"]); rarity=float(out.iloc[i][key+"_baseline_rarity"]); pattern=float(out.iloc[i][key+"_pattern_score"])
            # Whole-body convulsion is deliberately held to a stricter standard than focal events.
            # Gross voluntary repositioning can move many overlapping ROIs at once.
            if key=="convulsion" and not (rarity >= (.990 if self_reference else .995) and pattern >= .90):
                score=0.
            candidates.append((score,rarity,pattern,label_map[key],str(out.iloc[i][key+"_region"])))
        candidates.append((float(out.iloc[i].unknown_score),float(max(state_dev[i],novelty[i])),float(out.iloc[i].unknown_score),"unknown-novelty","unknown"))
        best=max(candidates,key=lambda x:x[0])
        # Only expose event label if evidence is genuinely strong; otherwise EXPECTED.
        review=.55*best[1]+.45*best[2]
        label_rarity_cut=.97 if self_reference else .99
        label_pattern_cut=.80 if self_reference else .72
        if best[1]>=label_rarity_cut and best[2]>=label_pattern_cut and float(out.iloc[i].quality_score)>=.55:
            labels.append(best[3]);regions.append(best[4]);patterns.append(best[2]);rarities.append(best[1]);att.append(review);conf.append(best[0])
        else:
            labels.append("normal/expected");regions.append("");patterns.append(0.);rarities.append(0.);att.append(0.);conf.append(0.)
    out["candidate_label"]=labels; out["primary_region"]=regions; out["pattern_score"]=patterns
    out["baseline_rarity"]=rarities; out["review_priority"]=att; out["attention_score"]=att
    out["anomaly_score"]=att  # backward-compatible display column, now event attention not generic motion anomaly
    out["explanation_confidence"]=conf
    return out,state_table


def _event_window(row,label):
    if label in ("twitch-like","convulsion-like"):
        return float(row.short_start_s),float(row.short_end_s)
    if label in ("tremor-like","sway-like","eye-state-change-like","unknown-novelty"):
        return float(row.mid_start_s),float(row.mid_end_s)
    return float(row.long_start_s),float(row.long_end_s)


def merge_events(explain: pd.DataFrame, score_threshold: float=.99, gap_s: float=.35,
                 min_duration_s: float=.10, pattern_min: float=.72) -> pd.DataFrame:
    cand=explain[(explain.candidate_label!="normal/expected") & (explain.baseline_rarity>=score_threshold) &
                 (explain.pattern_score>=pattern_min) & (explain.quality_score>=.55)].copy()
    cols=["event_id","start_s","end_s","duration_s","peak_s","peak_score","baseline_rarity",
          "review_priority","candidate_label","primary_region","confidence","behavior_state"]
    if cand.empty:return pd.DataFrame(columns=cols)
    events=[]
    for _,r in cand.sort_values("center_s").iterrows():
        st,en=_event_window(r,r.candidate_label); key=(str(r.candidate_label),str(r.primary_region))
        if events and events[-1]["key"]==key and st<=events[-1]["end_s"]+gap_s:
            e=events[-1]; e["end_s"]=max(e["end_s"],en)
            if float(r.review_priority)>e["review_priority"]:
                e.update(peak_s=float(r.center_s),peak_score=float(r.pattern_score),baseline_rarity=float(r.baseline_rarity),
                         review_priority=float(r.review_priority),confidence=float(r.explanation_confidence),behavior_state=str(r.behavior_state))
        else:
            events.append(dict(key=key,start_s=st,end_s=en,peak_s=float(r.center_s),peak_score=float(r.pattern_score),
                               baseline_rarity=float(r.baseline_rarity),review_priority=float(r.review_priority),
                               candidate_label=key[0],primary_region=key[1],confidence=float(r.explanation_confidence),
                               behavior_state=str(r.behavior_state)))
    rows=[]
    for i,e in enumerate(events,1):
        dur=max(0.,e["end_s"]-e["start_s"])
        if dur<min_duration_s:continue
        e.pop("key",None); e["event_id"]=i; e["duration_s"]=dur; rows.append(e)
    return pd.DataFrame(rows,columns=cols)


def _immobility_threshold(reference_frame: pd.DataFrame):
    arr=reference_frame.whole_body_motion.to_numpy(float); return float(np.percentile(arr,25)) if len(arr) else 0.


def _longest_true_run(mask: np.ndarray, fps: float):
    best=0; cur=0; bouts3=0; in3=False
    for v in np.asarray(mask,dtype=bool):
        if v:
            cur+=1
            if cur>=int(round(3.0*fps)) and not in3:bouts3+=1; in3=True
            best=max(best,cur)
        else:cur=0; in3=False
    return float(best/max(fps,EPS)),int(bouts3)


def _behavioral_endpoints(frame_df: pd.DataFrame, reference_frame: pd.DataFrame):
    if len(frame_df)==0:return {}
    fps=1.0/max(EPS,float(np.median(np.diff(frame_df.time_s.to_numpy(float))))) if len(frame_df)>2 else 30.0
    imm_th=_immobility_threshold(reference_frame); low=frame_df.whole_body_motion.to_numpy(float)<=imm_th
    longest,bouts3=_longest_true_run(low,fps)
    # Head-drop is always interpreted relative to the individual's baseline and only when the head is visible.
    refhd=reference_frame.head_drop_index.to_numpy(float) if "head_drop_index" in reference_frame else np.zeros(len(reference_frame)); hd_th=float(np.quantile(refhd,.85)) if len(refhd) else 0.
    head_vis=frame_df.get("head_visibility",pd.Series(np.ones(len(frame_df)))).to_numpy(float)>=.30; head_drop=(frame_df.get("head_drop_index",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)>=hd_th)&head_vis
    eye_available=np.zeros(len(frame_df),dtype=bool); eye_closed=np.zeros(len(frame_df),dtype=bool)
    for er in EYE_REGIONS:
        vis=frame_df.get(er+"_visibility",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)>=.35
        refvis=reference_frame.get(er+"_visibility",pd.Series(np.zeros(len(reference_frame)))).to_numpy(float)>=.35
        vals=reference_frame.get(er+"_dark_height",pd.Series(np.zeros(len(reference_frame)))).to_numpy(float); usable=vals[refvis]
        if len(usable)>=10:
            th=float(np.quantile(usable,.15)); cur=frame_df.get(er+"_dark_height",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)
            eye_available|=vis; eye_closed|=(vis&(cur<=th))
    # Low-arousal/"drowsiness-like" burden is a conservative multi-signal proxy, not a diagnosis.
    n_avail=np.ones(len(frame_df),dtype=float)+head_vis.astype(float)+eye_available.astype(float)
    positives=low.astype(float)+head_drop.astype(float)+eye_closed.astype(float)
    low_arousal=(positives>=np.maximum(2.,np.ceil(.67*n_avail)))
    # Eye-closure-like ratio only uses frames where at least one annotated eye is actually visible.
    eye_ratio=float(np.mean(eye_closed[eye_available])) if np.any(eye_available) else np.nan
    # Left/right asymmetry only compares regions that are visible on both sides.
    def asym(a,b):
        av=frame_df.get(a+"_visibility",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)>=.70; bv=frame_df.get(b+"_visibility",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)>=.70; ok=av&bv
        if not np.any(ok):return np.nan
        x=frame_df.get(a+"_residual",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)[ok]; y=frame_df.get(b+"_residual",pd.Series(np.zeros(len(frame_df)))).to_numpy(float)[ok]
        return float(abs(np.mean(x)-np.mean(y))/(np.mean(x)+np.mean(y)+EPS))
    return dict(low_arousal_burden_ratio=float(np.mean(low_arousal)), arousal_proxy_index=float(100.*(1.-np.mean(low_arousal))),
                prolonged_immobility_max_s=longest, immobility_bouts_ge_3s=bouts3,
                head_drop_ratio=float(np.mean(head_drop[head_vis])) if np.any(head_vis) else np.nan,
                eye_closure_like_ratio=eye_ratio, upper_limb_asymmetry=asym("left_upper_limb","right_upper_limb"),
                lower_limb_asymmetry=asym("left_lower_limb","right_lower_limb"),
                postural_sway_proxy=float(frame_df.get("torso_track_speed",pd.Series(np.zeros(len(frame_df)))).mean()))

def make_interval_statistics(frame_df: pd.DataFrame, explain: pd.DataFrame, events: pd.DataFrame,
                             reference_frame: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    if len(frame_df)==0:return pd.DataFrame()
    start=float(frame_df.time_s.min()); end=float(frame_df.time_s.max()); imm_th=_immobility_threshold(reference_frame); rows=[]; a=math.floor(start/bin_s)*bin_s
    while a<end+EPS:
        b=min(a+bin_s,end+1e-6); f=frame_df[(frame_df.time_s>=a)&(frame_df.time_s<b)]; x=explain[(explain.center_s>=a)&(explain.center_s<b)]
        if len(f)==0:a+=bin_s;continue
        ep=_behavioral_endpoints(f,reference_frame)
        row=dict(start_s=a,end_s=b,duration_s=max(0.,b-a),activity_mean=float(f.whole_body_motion.mean()),immobility_ratio=float(np.mean(f.whole_body_motion.to_numpy(float)<=imm_th)),
                 tracking_quality=float(f.tracking_quality.mean()),frame_quality=float(f.frame_quality.mean()),left_eye_visibility=float(f.left_eye_visibility.mean()),right_eye_visibility=float(f.right_eye_visibility.mean()),
                 attention_mean=float(x.attention_score.mean()) if len(x) else 0.,state_deviation_p95=float(x.state_deviation_percentile.quantile(.95)) if len(x) else 0.,dominant_state=str(x.behavior_state_cn.mode().iloc[0]) if len(x) and len(x.behavior_state_cn.mode()) else "")
        row.update(ep)
        for et in EVENT_TYPES:
            ee=events[(events.candidate_label==et)&(events.start_s<b)&(events.end_s>=a)] if len(events) else events; row[et.replace("-","_")+"_count"]=int(len(ee)); row[et.replace("-","_")+"_duration_s"]=float(sum(max(0.,min(b,float(q.end_s))-max(a,float(q.start_s))) for _,q in ee.iterrows())) if len(ee) else 0.
        rows.append(row); a+=bin_s
    return pd.DataFrame(rows)

def make_session_summary(session: str, frame_df: pd.DataFrame, explain: pd.DataFrame, events: pd.DataFrame,
                         reference_frame: pd.DataFrame):
    start=float(frame_df.time_s.min()); end=float(frame_df.time_s.max()); dur=max(EPS,end-start); imm_th=_immobility_threshold(reference_frame); ep=_behavioral_endpoints(frame_df,reference_frame)
    viewpoint=str(getattr(frame_df,'attrs',{}).get('viewpoint','unknown'))
    row=dict(session=session,duration_s=dur,coverage_start_s=start,coverage_end_s=end,camera_viewpoint=viewpoint,camera_viewpoint_cn=VIEWPOINT_CN.get(viewpoint,'未指定'),
             activity_mean=float(frame_df.whole_body_motion.mean()),immobility_ratio=float(np.mean(frame_df.whole_body_motion.to_numpy(float)<=imm_th)),
             valid_quality_ratio=float(np.mean(explain.quality_score.to_numpy(float)>=.55)),mean_tracking_quality=float(frame_df.tracking_quality.mean()),
             left_eye_visible_ratio=float(np.mean(frame_df.left_eye_visibility.to_numpy(float)>=.35)),right_eye_visible_ratio=float(np.mean(frame_df.right_eye_visibility.to_numpy(float)>=.35)),
             mean_attention=float(explain.attention_score.mean()),high_attention_ratio=float(np.mean(explain.attention_score.to_numpy(float)>=.8)),
             dominant_behavior_state=str(explain.behavior_state_cn.mode().iloc[0]) if len(explain.behavior_state_cn.mode()) else "")
    row.update(ep)
    if len(explain):
        vc=explain.behavior_state_cn.value_counts(normalize=True)
        for state_name,ratio in vc.items(): row['state_%s_ratio'%str(state_name).replace('/','_').replace(' ','_')]=float(ratio)
    for et in EVENT_TYPES:
        ee=events[events.candidate_label==et] if len(events) else events; prefix=et.replace("-","_"); row[prefix+"_count"]=int(len(ee)); row[prefix+"_per_min"]=float(len(ee)*60.0/dur); total=float(ee.duration_s.sum()) if len(ee) else 0.; row[prefix+"_duration_s"]=total; row[prefix+"_burden_ratio"]=float(total/dur); row[prefix+"_latency_s"]=float(ee.start_s.min()-start) if len(ee) else np.nan
    return pd.DataFrame([row])

def make_region_summary(train: pd.DataFrame,target: pd.DataFrame):
    rows=[]
    for rg in REGIONS:
        cols=[c for c in target.columns if rg in c and c in train.columns and np.issubdtype(target[c].dtype,np.number)]
        for c in cols:
            a=float(train[c].median()); b=float(target[c].median()); pct=float(100*(b-a)/(abs(a)+EPS)) if abs(a)>1e-7 else np.nan
            rows.append(dict(region=rg,feature=c,reference_median=a,target_median=b,change_percent=pct))
    return pd.DataFrame(rows)


def make_before_after_comparison(before_summary: pd.DataFrame, after_summary: pd.DataFrame):
    if before_summary.empty or after_summary.empty:return pd.DataFrame()
    b=before_summary.iloc[0]; a=after_summary.iloc[0]; rows=[]
    skip={"session","coverage_start_s","coverage_end_s"}
    common=[c for c in before_summary.columns if c in after_summary.columns and c not in skip]
    for c in common:
        try:
            bv=float(b[c]); av=float(a[c]); delta=av-bv; pct=100*delta/(abs(bv)+EPS) if abs(bv)>1e-8 else np.nan
            rows.append(dict(metric=c,before=bv,after=av,delta=delta,change_percent=pct))
        except Exception:pass
    return pd.DataFrame(rows)


def save_curve(explain: pd.DataFrame,events: pd.DataFrame,out: Path):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,5.2)); ax=fig.add_subplot(111)
    ax.plot(explain.center_s,explain.state_deviation_percentile,label="State deviation",linewidth=1.1)
    for k,label in [("twitch_score","Twitch"),("tremor_score","Tremor"),("convulsion_score","Convulsion"),("eye_score","Eye")]:
        ax.plot(explain.center_s,explain[k],label=label,linewidth=.9,alpha=.8)
    for _,e in events.iterrows():ax.axvspan(e.start_s,e.end_s,alpha=.10)
    ax.axhline(.99,linestyle="--",linewidth=.8); ax.set_ylim(0,1.02); ax.set_xlabel("Time (s)"); ax.set_ylabel("Score / baseline percentile")
    ax.set_title("State-aware multi-scale behavior timeline"); ax.legend(ncol=5,fontsize=8); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)


def _pretty_label(s):
    return {"normal/expected":"EXPECTED","twitch-like":"Twitch-like","tremor-like":"Tremor-like",
            "convulsion-like":"Convulsion-like","sway-like":"Sway-like","eye-state-change-like":"Eye-state change",
            "hypoactivity-like":"Hypoactivity","unknown-novelty":"Unknown novelty"}.get(str(s),str(s))


def _pretty_region(s):
    return {"whole_body":"Whole body","head":"Head","left_eye":"Left eye","right_eye":"Right eye","torso":"Torso",
            "left_upper_limb":"Left upper limb","right_upper_limb":"Right upper limb","left_lower_limb":"Left lower limb",
            "right_lower_limb":"Right lower limb","unknown":"Unknown","":""}.get(str(s),str(s))


def _nearest(df: pd.DataFrame,t: float):
    arr=df.center_s.to_numpy(float); return df.iloc[int(np.argmin(np.abs(arr-t)))]


def _nearest_frame(df: pd.DataFrame,t: float):
    arr=df.time_s.to_numpy(float); idx=int(np.searchsorted(arr,t)); idx=max(0,min(len(arr)-1,idx));
    if idx>0 and abs(arr[idx-1]-t)<abs(arr[idx]-t):idx-=1
    return df.iloc[idx]


def annotate_video(video: str, explain: pd.DataFrame,events: pd.DataFrame,dst: str,frame_df: pd.DataFrame):
    info=get_video_info(video); cap=cv2.VideoCapture(video); wr=cv2.VideoWriter(dst,cv2.VideoWriter_fourcc(*"mp4v"),info.fps,(info.width,info.height))
    start=float(explain.center_s.min()); end=float(explain.center_s.max()); duration=max(info.duration,EPS); i=0
    while True:
        ok,frame=cap.read();
        if not ok:break
        t=i/info.fps; i+=1
        if t<start or t>end:
            cv2.putText(frame,"PrimateBehaviorAI 4.0 | outside calibrated coverage",(24,42),cv2.FONT_HERSHEY_SIMPLEX,.7,(210,210,210),2,cv2.LINE_AA); wr.write(frame); continue
        r=_nearest(explain,t); f=_nearest_frame(frame_df,t); label=str(r.candidate_label); attention=float(r.attention_score); qc=float(r.quality_score)
        # translucent panels
        overlay=frame.copy(); cv2.rectangle(overlay,(18,18),(520,255),(10,22,32),-1); cv2.rectangle(overlay,(info.width-350,18),(info.width-18,255),(10,22,32),-1); cv2.addWeighted(overlay,.68,frame,.32,0,frame)
        cv2.putText(frame,"PrimateBehaviorAI 4.0",(34,50),cv2.FONT_HERSHEY_SIMPLEX,.82,(214,240,234),2,cv2.LINE_AA)
        cv2.putText(frame,"STATE: %s"%str(r.behavior_state),(34,88),cv2.FONT_HERSHEY_SIMPLEX,.55,(190,210,220),1,cv2.LINE_AA)
        status="UNAVAILABLE" if qc<.55 else (_pretty_label(label) if label!="normal/expected" else "EXPECTED")
        col=(90,90,245) if label!="normal/expected" and qc>=.55 else ((85,190,230) if qc<.55 else (100,220,140))
        cv2.putText(frame,status,(34,132),cv2.FONT_HERSHEY_SIMPLEX,.92,col,2,cv2.LINE_AA)
        cv2.putText(frame,"Review priority: %.3f"%attention,(34,166),cv2.FONT_HERSHEY_SIMPLEX,.52,(190,210,220),1,cv2.LINE_AA)
        cv2.putText(frame,"QC: %.2f   Baseline rarity: %.3f"%(qc,float(r.baseline_rarity)),(34,198),cv2.FONT_HERSHEY_SIMPLEX,.48,(160,185,200),1,cv2.LINE_AA)
        cv2.putText(frame,"Region: %s"%_pretty_region(str(r.primary_region)),(34,228),cv2.FONT_HERSHEY_SIMPLEX,.48,(160,185,200),1,cv2.LINE_AA)
        x=info.width-330; y=52
        for k,nm in [("twitch_score","Twitch"),("tremor_score","Tremor"),("convulsion_score","Convulsion"),("sway_score","Sway"),("eye_score","Eye")]:
            v=float(r[k]); cv2.putText(frame,"%-11s %.2f"%(nm,v),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(205,220,230),1,cv2.LINE_AA); y+=32
        # Dynamic primary ROI box only when an event is active.
        rg=str(r.primary_region)
        if rg in REGIONS and label!="normal/expected":
            x1=int(f.get(rg+"_x1",0));y1=int(f.get(rg+"_y1",0));x2=int(f.get(rg+"_x2",0));y2=int(f.get(rg+"_y2",0));cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
        # Timeline
        yy=info.height-46; x1=30; x2=info.width-30; cv2.line(frame,(x1,yy),(x2,yy),(80,100,115),3)
        for _,e in events.iterrows():
            a=x1+int((float(e.start_s)/duration)*(x2-x1)); b=x1+int((float(e.end_s)/duration)*(x2-x1)); cv2.line(frame,(a,yy),(b,yy),(80,120,245),6)
        cur=x1+int((t/duration)*(x2-x1));cv2.circle(frame,(cur,yy),6,(230,240,245),-1)
        wr.write(frame)
    cap.release();wr.release()


def ffmpeg_clip(src,start,end,dst):
    if shutil.which("ffmpeg") is None:return False
    cmd=["ffmpeg","-y","-loglevel","error","-ss","%.3f"%max(0,start),"-i",src,"-t","%.3f"%max(.1,end-max(0,start)),"-an","-c:v","copy","-avoid_negative_ts","make_zero",dst]
    try:subprocess.run(cmd,check=True,timeout=90);return True
    except Exception:return False


def export_candidate_clips(video,events,out_dir,pad=.7):
    out_dir.mkdir(parents=True,exist_ok=True)
    for _,e in events.iterrows():
        ffmpeg_clip(video,float(e.start_s)-pad,float(e.end_s)+pad,str(out_dir/("event_%03d_%0.2fs_%s.mp4"%(int(e.event_id),float(e.peak_s),str(e.primary_region)))))


def _fmt(v,percent=False):
    try:
        if pd.isna(v):return "不可用"
        x=float(v); return ("%.1f%%"%(100*x)) if percent else ("%.3g"%x)
    except Exception:return html.escape(str(v))


def _event_table_cn(events: pd.DataFrame):
    if events is None or events.empty:return "<p class='muted'>未检出达到复核阈值的候选事件。</p>"
    e=events.copy(); e["候选类型"]=e.candidate_label.map(EVENT_CN).fillna(e.candidate_label); e["部位"]=e.primary_region.map(REGION_CN).fillna(e.primary_region)
    e=e.rename(columns={"start_s":"开始时间(s)","end_s":"结束时间(s)","duration_s":"持续(s)","peak_s":"峰值时间(s)","peak_score":"形态分数","baseline_rarity":"基线稀有度","review_priority":"复核优先级","behavior_state":"行为状态"})
    cols=[c for c in ["候选类型","部位","开始时间(s)","结束时间(s)","持续(s)","峰值时间(s)","形态分数","基线稀有度","复核优先级","行为状态"] if c in e]
    return e[cols].to_html(index=False,float_format=lambda x:"%.3g"%x,classes="tbl")


def _core_comparison_cn(comparison):
    if comparison is None or comparison.empty:return ""
    names={
      "activity_mean":"整体活动量","immobility_ratio":"静止时间占比","low_arousal_burden_ratio":"低觉醒/困倦样负荷","arousal_proxy_index":"觉醒代理指数(0-100)",
      "prolonged_immobility_max_s":"最长连续静止(s)","head_drop_ratio":"低头/头位下降占比","eye_closure_like_ratio":"眼睑闭合样占比","mean_tracking_quality":"平均跟踪质量",
      "twitch_like_per_min":"抽搐样事件(次/min)","tremor_like_burden_ratio":"震颤样负荷","convulsion_like_per_min":"全身同步异常运动(次/min)","sway_like_per_min":"身体摆动(次/min)",
      "upper_limb_asymmetry":"上肢活动不对称","lower_limb_asymmetry":"下肢活动不对称","postural_sway_proxy":"姿势摆动代理量","unknown_novelty_per_min":"未知新颖行为(次/min)"}
    q=comparison[comparison.metric.isin(names)].copy()
    if q.empty:return ""
    q["指标"]=q.metric.map(names); q=q.rename(columns={"before":"照射前","after":"照射后","delta":"变化量","change_percent":"变化百分比(%)"})
    return q[["指标","照射前","照射后","变化量","变化百分比(%)"]].to_html(index=False,float_format=lambda x:"%.4g"%x,classes="tbl")


def _trend_table_cn(df):
    if df is None or df.empty:return "<p class='muted'>无可用统计。</p>"
    cols={"start_s":"开始(s)","end_s":"结束(s)","activity_mean":"活动量","immobility_ratio":"静止占比","low_arousal_burden_ratio":"困倦样负荷","arousal_proxy_index":"觉醒代理指数",
          "twitch_like_count":"抽搐样","tremor_like_count":"震颤样","convulsion_like_count":"全身同步异常","sway_like_count":"摆动","eye_state_change_like_count":"眼部变化","dominant_state":"主要行为状态","tracking_quality":"跟踪质量"}
    q=df[[c for c in cols if c in df]].rename(columns=cols)
    return q.to_html(index=False,float_format=lambda x:"%.3g"%x,classes="tbl")


def save_html_report(path: Path, session_summary: pd.DataFrame, events: pd.DataFrame,
                     stats10: pd.DataFrame, stats30: pd.DataFrame, minute: pd.DataFrame,
                     comparison: Optional[pd.DataFrame]=None):
    s=session_summary.iloc[0] if len(session_summary) else pd.Series(dtype=object)
    view=html.escape(str(s.get('camera_viewpoint_cn','未指定'))); valid=_fmt(s.get('valid_quality_ratio',np.nan),True); tq=_fmt(s.get('mean_tracking_quality',np.nan))
    activity=_fmt(s.get('activity_mean',np.nan)); imm=_fmt(s.get('immobility_ratio',np.nan),True); arousal=_fmt(s.get('arousal_proxy_index',np.nan)); drowsy=_fmt(s.get('low_arousal_burden_ratio',np.nan),True)
    head=_fmt(s.get('head_drop_ratio',np.nan),True); eye=_fmt(s.get('eye_closure_like_ratio',np.nan),True); longest=_fmt(s.get('prolonged_immobility_max_s',np.nan))
    tw=_fmt(s.get('twitch_like_per_min',0)); tr=_fmt(s.get('tremor_like_burden_ratio',0),True); cv=_fmt(s.get('convulsion_like_per_min',0)); sw=_fmt(s.get('sway_like_per_min',0)); ua=_fmt(s.get('upper_limb_asymmetry',np.nan)); la=_fmt(s.get('lower_limb_asymmetry',np.nan)); ps=_fmt(s.get('postural_sway_proxy',np.nan))
    comparison_html=_core_comparison_cn(comparison); compare_block=("<div class='card'><h2>照射前 vs 照射后核心指标</h2>"+comparison_html+"</div>") if comparison_html else ""
    events_html=_event_table_cn(events); minute_html=_trend_table_cn(minute); t30=_trend_table_cn(stats30); t10=_trend_table_cn(stats10)
    state=html.escape(str(s.get('dominant_behavior_state','')))
    state_rows=[]
    for k,v in s.items():
        if str(k).startswith('state_') and str(k).endswith('_ratio'):
            nm=str(k)[6:-6].replace('_','/'); state_rows.append({'行为状态':nm,'时间占比':float(v)})
    state_html=pd.DataFrame(state_rows).sort_values('时间占比',ascending=False).to_html(index=False,float_format=lambda x:'%.1f%%'%(100*x),classes='tbl') if state_rows else "<p class='muted'>无可用状态分布。</p>"
    left_eye_vis=_fmt(s.get('left_eye_visible_ratio',np.nan),True); right_eye_vis=_fmt(s.get('right_eye_visible_ratio',np.nan),True)
    # A concise automatic interpretation, intentionally descriptive rather than causal.
    note=[]
    if comparison is not None and len(comparison):
        cm=comparison.set_index('metric')
        def delta(metric):
            try:return float(cm.loc[metric,'delta'])
            except Exception:return np.nan
        if np.isfinite(delta('low_arousal_burden_ratio')) and delta('low_arousal_burden_ratio')>0.03:note.append('照射后低觉醒/困倦样负荷较照射前增加。')
        if np.isfinite(delta('twitch_like_per_min')) and delta('twitch_like_per_min')>0:note.append('照射后抽搐样短促运动候选频率增加。')
        if np.isfinite(delta('activity_mean')) and delta('activity_mean')<0:note.append('照射后整体活动量下降。')
        if not note:note.append('核心指标未显示一致方向的大幅变化；仍建议结合 Sham 对照和专家复核。')
    else:note.append('当前为单视频探索结果，只描述本段视频内的行为表型，不能用于判断暴露效应。')
    interpretation=''.join('<li>'+html.escape(x)+'</li>' for x in note)
    doc="""<!doctype html><html><head><meta charset='utf-8'><title>灵长类电磁暴露行为表型报告</title><style>
    body{font-family:'Microsoft YaHei UI','Microsoft YaHei',Arial,sans-serif;background:#071018;color:#eaf2f7;margin:28px;line-height:1.65}h1{color:#36d6c6;margin-bottom:4px}h2{margin-top:8px}.sub{color:#8da5b3}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.metric{background:#102532;border:1px solid #274252;border-radius:12px;padding:14px}.metric b{font-size:22px;color:#7fe8dc;display:block}.card{background:#0e1c27;border:1px solid #203b4d;border-radius:14px;padding:18px;margin:16px 0;overflow:auto}.tbl{border-collapse:collapse;width:100%%;font-size:12px}.tbl th,.tbl td{border-bottom:1px solid #29404e;padding:7px;text-align:left}.tbl th{color:#7fe8dc;background:#102532;position:sticky;top:0}.muted,.note{color:#9fb4c1}.warn{background:#342714;border:1px solid #725522;border-radius:10px;padding:12px;color:#f3d99b}.ok{color:#74e8b0}.tag{display:inline-block;padding:4px 9px;background:#163544;border-radius:999px;margin-right:6px;color:#9eece4}</style></head><body>
    <h1>灵长类电磁暴露行为表型分析报告</h1><div class='sub'>PrimateBehaviorAI 4.8.0 · 状态感知 / 视角感知 / 多时间尺度 · 科研筛查版</div>
    <div class='warn'><b>解释边界：</b>本报告量化的是视频中可见的行为变化。抽搐样、震颤样、困倦样等均为候选表型，不等同于医学诊断；Before/After 差异也不能单独证明由电磁暴露导致，正式研究应结合 Sham 对照、重复 session 和专家复核。</div>
    <div class='card'><h2>数据质量与视角</h2><div class='grid'><div class='metric'><span>相机视角</span><b>%(view)s</b></div><div class='metric'><span>有效分析时间占比</span><b>%(valid)s</b></div><div class='metric'><span>平均跟踪质量</span><b>%(tq)s</b></div><div class='metric'><span>主要行为状态</span><b style='font-size:16px'>%(state)s</b></div><div class='metric'><span>左眼有效可见率</span><b>%(left_eye_vis)s</b></div><div class='metric'><span>右眼有效可见率</span><b>%(right_eye_vis)s</b></div></div></div>
    <div class='card'><h2>一、觉醒 / 困倦维度</h2><div class='grid'><div class='metric'><span>觉醒代理指数</span><b>%(arousal)s</b></div><div class='metric'><span>低觉醒/困倦样负荷</span><b>%(drowsy)s</b></div><div class='metric'><span>静止时间占比</span><b>%(imm)s</b></div><div class='metric'><span>最长连续静止</span><b>%(longest)s s</b></div><div class='metric'><span>低头/头位下降</span><b>%(head)s</b></div><div class='metric'><span>眼睑闭合样代理</span><b>%(eye)s</b></div></div><p class='note'>“困倦样负荷”要求低活动、头位下降、可见眼闭合样信号中的多项同时出现；侧面不可见的眼睛不会参与计算。</p></div>
    <div class='card'><h2>二、神经运动异常维度</h2><div class='grid'><div class='metric'><span>抽搐样事件</span><b>%(tw)s /min</b></div><div class='metric'><span>震颤样负荷</span><b>%(tr)s</b></div><div class='metric'><span>全身同步异常运动</span><b>%(cv)s /min</b></div><div class='metric'><span>身体摆动事件</span><b>%(sw)s /min</b></div></div></div>
    <div class='card'><h2>三、姿态稳定与左右不对称</h2><div class='grid'><div class='metric'><span>姿势摆动代理量</span><b>%(ps)s</b></div><div class='metric'><span>上肢活动不对称</span><b>%(ua)s</b></div><div class='metric'><span>下肢活动不对称</span><b>%(la)s</b></div></div><p class='note'>只有左右两侧都达到可见性要求时才计算左右不对称；侧位视频不会因为远侧肢体不可见而制造假异常。</p></div>
    <div class='card'><h2>四、整体活动</h2><div class='grid'><div class='metric'><span>整体活动量</span><b>%(activity)s</b></div><div class='metric'><span>静止时间占比</span><b>%(imm)s</b></div></div></div>
    %(compare_block)s
    <div class='card'><h2>五、自发行为状态分布</h2>%(state_html)s<p class='note'>当前版本自动量化低活动、全身活动、上肢活动、下肢活动、头部/朝向活动和混合活动。抓挠、梳理、刻板行为等需要积累人工标签后训练监督分类器，因此本版不把它们伪装成已可靠识别的标签。</p></div>
    <div class='card'><h2>六、时间演化与自动描述</h2><ul>%(interpretation)s</ul><h3>每分钟趋势</h3>%(minute)s<h3>30 秒统计</h3>%(t30)s<h3>10 秒统计</h3>%(t10)s</div>
    <div class='card'><h2>候选事件清单（建议专家逐条复核）</h2>%(events)s</div>
    <div class='card'><h2>建议的正式实验终点</h2><p><span class='tag'>低觉醒/困倦样负荷</span><span class='tag'>活动/静止</span><span class='tag'>抽搐样 /min</span><span class='tag'>震颤样负荷</span><span class='tag'>全身同步异常运动</span><span class='tag'>身体摆动/姿势稳定</span><span class='tag'>行为状态占比</span><span class='tag'>首次变化潜伏期</span></p><p class='note'>不建议把普通单目侧面视频中的瞳孔大小作为主要终点；非常细微/高频肌束颤动也超出 30 FPS 视频的可靠观测范围。</p></div>
    </body></html>""" % dict(view=view,valid=valid,tq=tq,state=state,arousal=arousal,drowsy=drowsy,imm=imm,longest=longest,head=head,eye=eye,tw=tw,tr=tr,cv=cv,sw=sw,ps=ps,ua=ua,la=la,activity=activity,compare_block=compare_block,interpretation=interpretation,minute=minute_html,t30=t30,t10=t10,events=events_html,state_html=state_html,left_eye_vis=left_eye_vis,right_eye_vis=right_eye_vis)
    path.write_text(doc,encoding="utf-8")

def analyze_feature_tables(frame_df: pd.DataFrame,windows: pd.DataFrame,reference_frame: pd.DataFrame,
                           reference_windows: pd.DataFrame,event_threshold=.99,self_reference=False,session="post",exploratory=False):
    explain,states=build_explain(reference_windows,windows,self_reference=self_reference)
    # In single-video exploratory mode the same clip contributes to its own reference.
    # Use a high pattern requirement but a slightly lower rarity cutoff so a true focal
    # transient is not mathematically diluted by being part of its own baseline.
    eff_threshold=min(float(event_threshold),.975) if (self_reference and exploratory) else float(event_threshold)
    pattern_min=.85 if (self_reference and exploratory) else .72
    events=merge_events(explain,score_threshold=eff_threshold,pattern_min=pattern_min)
    stats10=make_interval_statistics(frame_df,explain,events,reference_frame,10.0)
    stats30=make_interval_statistics(frame_df,explain,events,reference_frame,30.0)
    minute=make_interval_statistics(frame_df,explain,events,reference_frame,60.0)
    summary=make_session_summary(session,frame_df,explain,events,reference_frame)
    return explain,events,states,stats10,stats30,minute,summary


def write_analysis_outputs(out: Path,frame_df: pd.DataFrame,windows: pd.DataFrame,explain: pd.DataFrame,
                           events: pd.DataFrame,states: pd.DataFrame,stats10: pd.DataFrame,
                           stats30: pd.DataFrame,minute: pd.DataFrame,summary: pd.DataFrame,
                           region_summary: Optional[pd.DataFrame]=None,comparison: Optional[pd.DataFrame]=None):
    out.mkdir(parents=True,exist_ok=True)
    frame_df.to_csv(out/"frame_features.csv",index=False);windows.to_csv(out/"multiscale_features.csv",index=False)
    explain.to_csv(out/"behavior_timeline.csv",index=False); explain.to_csv(out/"anomaly_timeseries.csv",index=False)
    events.to_csv(out/"events.csv",index=False);states.to_csv(out/"behavior_states.csv",index=False)
    stats10.to_csv(out/"statistics_10s.csv",index=False);stats30.to_csv(out/"statistics_30s.csv",index=False)
    minute.to_csv(out/"minute_summary.csv",index=False);summary.to_csv(out/"session_summary.csv",index=False)
    # Human-readable core endpoints for the EMF experiment workflow.
    if len(summary):
        sr=summary.iloc[0]
        core=[
          ("activity_mean","整体活动量","a.u."),("immobility_ratio","静止时间占比","ratio"),("low_arousal_burden_ratio","低觉醒/困倦样负荷","ratio"),
          ("arousal_proxy_index","觉醒代理指数","0-100"),("prolonged_immobility_max_s","最长连续静止","s"),("head_drop_ratio","低头/头位下降占比","ratio"),
          ("eye_closure_like_ratio","眼睑闭合样代理占比","ratio"),("twitch_like_per_min","抽搐样事件频率","events/min"),("tremor_like_burden_ratio","震颤样负荷","ratio"),
          ("convulsion_like_per_min","全身同步异常运动频率","events/min"),("sway_like_per_min","身体摆动事件频率","events/min"),
          ("postural_sway_proxy","姿势摆动代理量","a.u."),("upper_limb_asymmetry","上肢活动不对称","ratio"),("lower_limb_asymmetry","下肢活动不对称","ratio"),
          ("valid_quality_ratio","有效分析时间占比","ratio"),("mean_tracking_quality","平均跟踪质量","0-1")
        ]
        pd.DataFrame([dict(metric=k,指标=cn,value=sr.get(k,np.nan),单位=unit) for k,cn,unit in core]).to_csv(out/"core_endpoints_cn.csv",index=False,encoding="utf-8-sig")
    if region_summary is not None:region_summary.to_csv(out/"region_summary.csv",index=False)
    # Compact event burden table for downstream statistics.
    erows=[]
    duration=float(summary.iloc[0].get("duration_s",1.0)) if len(summary) else 1.0
    for et in EVENT_TYPES:
        ee=events[events.candidate_label==et] if len(events) else events
        erows.append(dict(event_type=et,count=int(len(ee)),per_min=float(len(ee)*60.0/max(duration,EPS)),
                          total_duration_s=float(ee.duration_s.sum()) if len(ee) else 0.,
                          burden_ratio=float((ee.duration_s.sum() if len(ee) else 0.)/max(duration,EPS)),
                          mean_pattern_score=float(ee.peak_score.mean()) if len(ee) else 0.,
                          mean_baseline_rarity=float(ee.baseline_rarity.mean()) if len(ee) else 0.))
    pd.DataFrame(erows).to_csv(out/"event_type_summary.csv",index=False)
    if len(explain):
        occ=explain.groupby(["behavior_state_id","behavior_state","behavior_state_cn"]).size().reset_index(name="windows")
        occ["ratio"]=occ.windows/float(len(explain));occ.to_csv(out/"state_occupancy.csv",index=False)
    if comparison is not None:
        comparison.to_csv(out/"before_after_comparison.csv",index=False)
        cmap={"activity_mean":"整体活动量","immobility_ratio":"静止时间占比","low_arousal_burden_ratio":"低觉醒/困倦样负荷","arousal_proxy_index":"觉醒代理指数","prolonged_immobility_max_s":"最长连续静止(s)","head_drop_ratio":"低头/头位下降占比","eye_closure_like_ratio":"眼睑闭合样占比","twitch_like_per_min":"抽搐样事件(次/min)","tremor_like_burden_ratio":"震颤样负荷","convulsion_like_per_min":"全身同步异常运动(次/min)","sway_like_per_min":"身体摆动(次/min)","upper_limb_asymmetry":"上肢活动不对称","lower_limb_asymmetry":"下肢活动不对称","postural_sway_proxy":"姿势摆动代理量"}
        cq=comparison[comparison.metric.isin(cmap)].copy(); cq["指标"]=cq.metric.map(cmap); cq.to_csv(out/"before_after_core_cn.csv",index=False,encoding="utf-8-sig")
    save_curve(explain,events,out/"anomaly_curve.png")
    save_html_report(out/"report.html",summary,events,stats10,stats30,minute,comparison)


def run_one(video: str,out_dir: Path,rois_path: Optional[str],analysis_width: int,
            baseline_pack=None,event_threshold: float=.99,make_annotated: bool=True):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);info=get_video_info(video);rois=load_rois(rois_path,info.width,info.height)
    print("[分析] 提取原始 FPS、视角感知的多区域行为特征...")
    _,frame_df,_=extract_frame_features(video,rois,analysis_width=analysis_width,progress=True,roi_path=rois_path,tracking_quality="accuracy")
    windows=make_multiscale_windows(frame_df,info.fps)
    if baseline_pack is None:
        ref_frame,ref_win=frame_df,windows; self_ref=True; session="single"
    else:
        ref_frame,ref_win=baseline_pack; self_ref=False; session="post"
    explain,events,states,s10,s30,minute,summary=analyze_feature_tables(frame_df,windows,ref_frame,ref_win,event_threshold,self_ref,session,exploratory=(baseline_pack is None))
    rsum=make_region_summary(ref_win,windows);write_analysis_outputs(out,frame_df,windows,explain,events,states,s10,s30,minute,summary,rsum)
    export_candidate_clips(video,events,out/"candidate_clips")
    if make_annotated:
        print("[分析] 生成原分辨率复核 Dashboard...")
        annotate_video(video,explain,events,str(out/"annotated_dashboard.mp4"),frame_df)
    return dict(info=info,frame_df=frame_df,windows=windows,explain=explain,events=events,summary=summary)
