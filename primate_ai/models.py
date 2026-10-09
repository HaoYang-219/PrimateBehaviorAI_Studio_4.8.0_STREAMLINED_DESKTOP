from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class AnalysisConfig:
    mode: str='before_after'
    before_path: str=''
    after_path: str=''
    single_path: str=''
    roi_path: str=''
    output_dir: str='results'
    analysis_width: int=900
    event_threshold: float=.995
    make_dashboard: bool=True
    tracking_backend: str='auto'
    project_name: str='未命名实验'
    session_dir: str=''
    camera_inputs: list=field(default_factory=list)

    @staticmethod
    def _validate_roi_file(path):
        if not path or not Path(path).exists():
            raise ValueError('请为每一路待分析视频完成身体区域标定。')
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
        except Exception as e:
            raise ValueError('ROI 标定文件无法读取，请重新标定或选择正确的 JSON 文件。') from e
        legacy=data.get('rois',{}) or {}; regions=data.get('regions',{}) or {}
        required=['whole_body','head','torso','left_upper_limb','right_upper_limb','left_lower_limb','right_lower_limb']
        missing=[]
        def has_region(r):
            b=legacy.get(r); item=regions.get(r,{}) or {}; pts=item.get('polygon') if isinstance(item,dict) else None
            return (isinstance(b,list) and len(b)>=4) or (isinstance(pts,list) and len(pts)>=3)
        for r in required:
            if not has_region(r): missing.append(r)
        eyes_ok=(has_region('left_eye') and has_region('right_eye')) or has_region('eyes')
        if not eyes_ok: missing.append('left_eye/right_eye')
        if missing:
            raise ValueError('ROI 标定文件不完整，缺少区域：'+', '.join(missing)+'。请重新标定。')

    def validate(self):
        if self.mode=='session_multicam':
            if not self.session_dir or not Path(self.session_dir).exists():
                raise ValueError('请选择有效的实验 Session 目录。')
            if not (1 <= len(self.camera_inputs or []) <= 3):
                raise ValueError('Session 分析需要选择 1–3 路摄像头视频。')
            seen=set()
            for item in self.camera_inputs:
                vp=str(item.get('video_path','')).strip(); rp=str(item.get('roi_path','')).strip()
                if not vp or not Path(vp).exists():
                    raise ValueError('存在无效的摄像头视频，请重新扫描 Session。')
                if vp in seen:
                    raise ValueError('同一视频不能重复加入多路分析。')
                seen.add(vp)
                self._validate_roi_file(rp)
        elif self.mode=='single':
            if not self.single_path or not Path(self.single_path).exists():
                raise ValueError('请选择有效的单视频文件。')
            self._validate_roi_file(self.roi_path)
        else:
            if not self.before_path or not Path(self.before_path).exists():
                raise ValueError('请选择有效的照射前视频。')
            if not self.after_path or not Path(self.after_path).exists():
                raise ValueError('请选择有效的照射后视频。')
            self._validate_roi_file(self.roi_path)
        Path(self.output_dir).mkdir(parents=True,exist_ok=True)
