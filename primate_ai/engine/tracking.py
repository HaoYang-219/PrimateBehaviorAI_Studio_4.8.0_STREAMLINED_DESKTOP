#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
import cv2
import numpy as np

REGIONS=["whole_body","head","left_eye","right_eye","torso","left_upper_limb","right_upper_limb","left_lower_limb","right_lower_limb"]
TRACK_INDEPENDENT=["whole_body","head","torso","left_upper_limb","right_upper_limb","left_lower_limb","right_lower_limb"]
EYE_REGIONS=["left_eye","right_eye"]

def _clip_poly(poly,w,h):
    p=poly.astype(np.float32).copy(); p[:,0]=np.clip(p[:,0],0,max(0,w-1)); p[:,1]=np.clip(p[:,1],0,max(0,h-1)); return p

def poly_bbox(poly,w,h):
    x1=int(np.floor(poly[:,0].min())); y1=int(np.floor(poly[:,1].min())); x2=int(np.ceil(poly[:,0].max()))+1; y2=int(np.ceil(poly[:,1].max()))+1
    x1=max(0,min(w-2,x1)); y1=max(0,min(h-2,y1)); x2=max(x1+2,min(w,x2)); y2=max(y1+2,min(h,y2)); return x1,y1,x2,y2

def transform_poly(poly,A):
    ones=np.ones((len(poly),1),dtype=np.float32); xy1=np.concatenate([poly.astype(np.float32),ones],axis=1); return (xy1@A.T).astype(np.float32)

def mask_for_poly(shape,poly):
    m=np.zeros(shape[:2],np.uint8); cv2.fillPoly(m,[np.round(poly).astype(np.int32)],255); return m

@dataclass
class TrackResult:
    polygon: np.ndarray
    box: Tuple[int,int,int,int]
    confidence: float
    dx: float
    dy: float
    scale: float

class DynamicROITracker:
    """Accuracy-oriented ROI tracker with visibility gating.

    Regions marked invisible in calibration are deliberately not tracked and return
    confidence 0. Partial regions are tracked but their confidence is down-weighted.
    Eyes follow head motion to avoid eyelid-texture drift.
    """
    def __init__(self,frame_bgr,polygons,quality="accuracy",visibility_priors=None):
        self.h,self.w=frame_bgr.shape[:2]; self.prev_gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
        self.polys={k:_clip_poly(np.asarray(v,dtype=np.float32),self.w,self.h) for k,v in polygons.items()}
        missing=[r for r in REGIONS if r not in self.polys]
        if missing:raise ValueError("Tracker polygons missing: "+", ".join(missing))
        self.visibility={r:float(np.clip((visibility_priors or {}).get(r,1.0),0,1)) for r in REGIONS}
        self.initial_size={}
        for r,p in self.polys.items():
            b=poly_bbox(p,self.w,self.h); self.initial_size[r]=(max(2,b[2]-b[0]),max(2,b[3]-b[1]))
        self.quality=quality; self.points={}; self.templates={}
        self.last_affine={k:np.array([[1,0,0],[0,1,0]],dtype=np.float32) for k in REGIONS}; self.frame_index=0
        self.max_corners=120 if quality=="accuracy" else 72; self.win=(27,27) if quality=="accuracy" else (21,21); self.levels=3 if quality=="accuracy" else 2
        for r in TRACK_INDEPENDENT:
            if self.visibility.get(r,1.0)<=.05:
                self.points[r]=None; continue
            self.points[r]=self._detect_points(self.prev_gray,self.polys[r]); self.templates[r]=self._crop_gray(self.prev_gray,self.polys[r])
    def _detect_points(self,gray,poly):
        x1,y1,x2,y2=poly_bbox(poly,self.w,self.h); pad=max(4,int(.05*max(x2-x1,y2-y1)))
        x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(self.w,x2+pad); y2=min(self.h,y2+pad); crop=gray[y1:y2,x1:x2]
        local=poly.copy(); local[:,0]-=x1; local[:,1]-=y1; mask=np.zeros(crop.shape[:2],np.uint8); cv2.fillPoly(mask,[np.round(local).astype(np.int32)],255)
        pts=cv2.goodFeaturesToTrack(crop,maxCorners=self.max_corners,qualityLevel=.006 if self.quality=="accuracy" else .015,minDistance=4,mask=mask,blockSize=5,useHarrisDetector=False)
        if pts is not None:pts=pts.astype(np.float32); pts[:,:,0]+=x1; pts[:,:,1]+=y1
        return pts
    def _crop_gray(self,gray,poly):
        x1,y1,x2,y2=poly_bbox(poly,self.w,self.h); return gray[y1:y2,x1:x2].copy()
    def _template_fallback(self,gray,region,poly):
        tpl=self.templates.get(region); ident=np.array([[1,0,0],[0,1,0]],np.float32)
        if tpl is None or tpl.size<64:return poly,0.0,ident
        x1,y1,x2,y2=poly_bbox(poly,self.w,self.h); bw,bh=x2-x1,y2-y1; pad_x=max(12,int(.35*bw)); pad_y=max(12,int(.35*bh))
        sx1=max(0,x1-pad_x); sy1=max(0,y1-pad_y); sx2=min(self.w,x2+pad_x); sy2=min(self.h,y2+pad_y); search=gray[sy1:sy2,sx1:sx2]
        if search.shape[0]<tpl.shape[0] or search.shape[1]<tpl.shape[1]:return poly,0.0,ident
        try:res=cv2.matchTemplate(search,tpl,cv2.TM_CCOEFF_NORMED); _,score,_,loc=cv2.minMaxLoc(res)
        except Exception:return poly,0.0,ident
        dx=float(sx1+loc[0]-x1); dy=float(sy1+loc[1]-y1); A=np.array([[1,0,dx],[0,1,dy]],dtype=np.float32); return transform_poly(poly,A),float(max(0.,score)),A
    def _track_region(self,gray,region):
        prev_poly=self.polys[region]; p0=self.points.get(region); A=None; conf=0.; next_pts=None
        if p0 is not None and len(p0)>=6:
            x1,y1,x2,y2=poly_bbox(prev_poly,self.w,self.h); bw,bh=x2-x1,y2-y1; px=max(20,int(.38*bw)); py=max(20,int(.38*bh))
            sx1=max(0,x1-px); sy1=max(0,y1-py); sx2=min(self.w,x2+px); sy2=min(self.h,y2+py); prev_crop=self.prev_gray[sy1:sy2,sx1:sx2]; curr_crop=gray[sy1:sy2,sx1:sx2]
            q0=p0.reshape(-1,2).copy(); inside=(q0[:,0]>=sx1)&(q0[:,0]<sx2)&(q0[:,1]>=sy1)&(q0[:,1]<sy2); q0=q0[inside]
            if len(q0)>=6 and prev_crop.size and curr_crop.size:
                local0=q0.copy(); local0[:,0]-=sx1; local0[:,1]-=sy1; local0=local0.reshape(-1,1,2).astype(np.float32)
                p1,st1,_=cv2.calcOpticalFlowPyrLK(prev_crop,curr_crop,local0,None,winSize=self.win,maxLevel=self.levels,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                if p1 is not None:
                    p0b,st2,_=cv2.calcOpticalFlowPyrLK(curr_crop,prev_crop,p1,None,winSize=self.win,maxLevel=self.levels,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                    good=(st1.reshape(-1)>0)&(st2.reshape(-1)>0); fb=np.linalg.norm(local0.reshape(-1,2)-p0b.reshape(-1,2),axis=1); good&=fb<(1.8 if self.quality=="accuracy" else 2.8)
                    a0=local0.reshape(-1,2)[good]; a1=p1.reshape(-1,2)[good]
                    if len(a0)>=6:
                        a0[:,0]+=sx1; a0[:,1]+=sy1; a1[:,0]+=sx1; a1[:,1]+=sy1
                        AA,inl=cv2.estimateAffinePartial2D(a0,a1,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=1000,confidence=.995,refineIters=10)
                        if AA is not None:
                            aa,bb=float(AA[0,0]),float(AA[0,1]); scale=float((aa*aa+bb*bb)**.5); trans=float(np.hypot(AA[0,2],AA[1,2]))
                            if .82<=scale<=1.22 and trans<=.45*max(bw,bh):
                                A=AA.astype(np.float32); inlier_ratio=float(inl.mean()) if inl is not None else .5; conf=float(np.clip(.5*min(1.,len(a0)/30.)+.5*inlier_ratio,0,1)); next_pts=a1.reshape(-1,1,2).astype(np.float32)
        if A is None:
            if region not in ("whole_body","torso") and self.frame_index%8==0:new_poly,tconf,A=self._template_fallback(gray,region,prev_poly); conf=.45*tconf
            else:A=np.array([[1,0,0],[0,1,0]],dtype=np.float32); new_poly=prev_poly.copy(); conf=.12
        else:new_poly=transform_poly(prev_poly,A)
        new_poly=_clip_poly(new_poly,self.w,self.h); nb=poly_bbox(new_poly,self.w,self.h); iw,ih=self.initial_size[region]; rw=(nb[2]-nb[0])/max(1.,iw); rh=(nb[3]-nb[1])/max(1.,ih)
        if not (.72<=rw<=1.38 and .72<=rh<=1.38):
            oldc=prev_poly.mean(axis=0); newc=new_poly.mean(axis=0); d=newc-oldc; new_poly=_clip_poly(prev_poly+d,self.w,self.h); A=np.array([[1,0,float(d[0])],[0,1,float(d[1])]],dtype=np.float32); conf=min(conf,.45)
        conf*=self.visibility.get(region,1.0)
        return new_poly,conf,A,next_pts
    def update(self,frame_bgr):
        gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY); results={}; affines={}; tracked_pts={}; old={k:v.copy() for k,v in self.polys.items()}
        order=["head","whole_body","torso","left_upper_limb","right_upper_limb","left_lower_limb","right_lower_limb"]
        for r in order:
            if self.visibility.get(r,1.0)<=.05:
                p=self.polys[r].copy(); results[r]=TrackResult(p,poly_bbox(p,self.w,self.h),0.,0.,0.,1.); affines[r]=np.array([[1,0,0],[0,1,0]],np.float32); tracked_pts[r]=None; continue
            new_poly,conf,A,next_pts=self._track_region(gray,r); old_c=old[r].mean(axis=0); new_c=new_poly.mean(axis=0); a,b=float(A[0,0]),float(A[0,1]); scale=float((a*a+b*b)**.5)
            self.polys[r]=new_poly; affines[r]=A; self.last_affine[r]=A; tracked_pts[r]=next_pts; results[r]=TrackResult(new_poly,poly_bbox(new_poly,self.w,self.h),conf,float(new_c[0]-old_c[0]),float(new_c[1]-old_c[1]),scale)
        headA=affines.get("head",np.array([[1,0,0],[0,1,0]],np.float32)); hres=results["head"]
        for r in EYE_REGIONS:
            eold=old[r]; enew=_clip_poly(transform_poly(eold,headA),self.w,self.h); ec=eold.mean(axis=0); nc=enew.mean(axis=0); self.polys[r]=enew; self.last_affine[r]=headA
            prior=self.visibility.get(r,1.0); results[r]=TrackResult(enew,poly_bbox(enew,self.w,self.h),hres.confidence*prior,float(nc[0]-ec[0]),float(nc[1]-ec[1]),hres.scale)
        self.prev_gray=gray; self.frame_index+=1; refresh_every=15 if self.quality=="accuracy" else 20
        for r in TRACK_INDEPENDENT:
            if self.visibility.get(r,1.0)<=.05:continue
            pts=tracked_pts.get(r); need=(self.frame_index%refresh_every==0) or pts is None or (pts is not None and len(pts)<14)
            if results[r].confidence<.30 and self.frame_index%4==0:need=True
            if need:self.points[r]=self._detect_points(gray,self.polys[r])
            elif pts is not None:self.points[r]=pts
            if self.frame_index%24==0 and results[r].confidence>.45:self.templates[r]=self._crop_gray(gray,self.polys[r])
        return results
    def current_boxes(self):return {r:poly_bbox(self.polys[r],self.w,self.h) for r in REGIONS}
    def current_polygons(self):return {r:self.polys[r].copy() for r in REGIONS}
