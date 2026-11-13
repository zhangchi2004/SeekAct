"""EPIC full-video camera mining with frame-indexed VITRA hand observations.

Full-source camera motion determines temporal boundaries at native FPS. VITRA boundaries are never candidates.
Metric scale is fitted per output clip using overlapping body-frame camera
translations from VITRA, with explicit agreement gates and retained evidence.
No hand samples are extrapolated or filled across absent observations.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .dense_actions import ActionConfig, make_actions
from .long_camera import LongCameraConfig, dense_timeline, mine_long_camera

SEED=20260920
CALIBRATION={"window_context_s":15.,"min_pairs":8,"min_baseline_m":.03,
             "max_rotation_disagreement_deg":8.,"max_direction_disagreement_deg":30.,
             "max_log_mad":.25,"min_inlier_fraction":.65,"inlier_log_distance":.35}


def native_camera_config(fps):
    # Boundaries stay on original frames. Round down maximum durations so a
    # 60000/1001 FPS recording cannot exceed the original 8 s / 1 s contract.
    return LongCameraConfig(selection_seed=SEED,
        max_core_duration_s=np.floor(8*fps)/fps,
        context_before_s=np.floor(fps)/fps,context_after_s=np.floor(fps)/fps)


def prepare_source(video:Path, camera_dir:Path, tracks:Path):
    cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    if not n or not np.isclose(fps,60000/1001,rtol=0,atol=1e-5):raise ValueError('expected complete native 60000/1001 FPS EPIC RGB')
    cfg=native_camera_config(fps)
    src=np.load(camera_dir/'camera_samples.npz'); indices=src['source_frame_indices']
    timeline=dense_timeline(indices,src['camera_to_world'],n,fps,cfg)
    errors=np.interp(np.arange(n),indices,src['alignment_error_deg'])
    seams=np.zeros(n,bool)
    # A sampled join marks the whole adjacent interpolation support as suspect.
    for i in np.flatnonzero(src['seam']):
        seams[indices[max(i-1,0)]:indices[min(i+1,len(indices)-1)]+1]=True
    reliable=timeline['valid'] & ~seams & (errors<=cfg.max_alignment_error_deg)
    hands={side:np.full((n,21,3),np.nan,np.float32) for side in ('left','right')}
    counts={side:np.zeros(n,np.uint16) for side in hands}
    disagreement={side:np.zeros(n,bool) for side in hands}
    files=sorted(tracks.glob(f'epic_kitchens_{video.stem}_ep_*.npy'))
    pair_map={}; all_k=[]; source_files=[]; transform_residual=[]
    for path in files:
        d=np.load(path,allow_pickle=True).item(); f=np.asarray(d['video_decode_frame'],int)
        if len(f)<2 or np.any(np.diff(f)<=0) or f[0]<0 or f[-1]>=n:continue
        e=np.asarray(d['extrinsics'],float); k=np.asarray(d['intrinsics'],float)
        if e.shape!=(len(f),4,4) or not np.isfinite(e).all():continue
        v=np.linalg.inv(e); all_k.append(k)
        source_files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        for side in hands:
            h=d.get(side)
            if not isinstance(h,dict):continue
            j=np.asarray(h['joints_camspace'],float); kept=np.asarray(h['kept_frames'],bool)
            if j.shape!=(len(f),21,3) or kept.shape!=(len(f),):raise ValueError('unexpected hand schema')
            good=kept & np.isfinite(j).all((1,2)) & (j[:,:,2]>.03).all(1)
            if good.any():
                w=np.asarray(h['joints_worldspace'],float)
                expected=np.einsum('nij,nkj->nki',v[good,:3,:3],j[good])+v[good,None,:3,3]
                residual=np.linalg.norm(expected-w[good],axis=-1)
                transform_residual.append(float(np.max(residual)))
                if np.max(residual)>.002:raise ValueError('VITRA camera/world convention mismatch')
            fi=f[good]; jj=j[good]; existing=counts[side][fi]>0
            if existing.any():
                dist=np.median(np.linalg.norm(hands[side][fi[existing]]-jj[existing],axis=-1),axis=1)
                disagreement[side][fi[existing]] |= dist>.03
            fresh=~existing;hands[side][fi[fresh]]=jj[fresh];counts[side][fi]+=1
        lookup={int(frame):i for i,frame in enumerate(f)}
        for lag in (round(.5*fps),round(fps),round(2*fps)):
            for i in range(0,len(f),round(.1*fps)):
                a=int(f[i]);b=a+lag;j=lookup.get(b)
                if j is None or not reliable[a:b+1].all() or (a,b) in pair_map:continue
                vv=np.linalg.inv(v[i])@v[j]
                dd=np.linalg.inv(timeline['camera_to_world'][a])@timeline['camera_to_world'][b]
                rot=float(np.rad2deg(Rotation.from_matrix(dd[:3,:3].T@vv[:3,:3]).magnitude()))
                vm,du=vv[:3,3],dd[:3,3];vn=np.linalg.norm(vm);dn=np.linalg.norm(du)
                if vn<CALIBRATION['min_baseline_m'] or dn<1e-4:continue
                angle=float(np.rad2deg(np.arccos(np.clip(np.dot(vm,du)/(vn*dn),-1,1))))
                if rot>CALIBRATION['max_rotation_disagreement_deg'] or angle>CALIBRATION['max_direction_disagreement_deg']:continue
                ratio=float(np.dot(vm,du)/np.dot(du,du))
                pair_map[a,b]=(a,b,ratio,rot,angle,vn,dn)
    if not all_k:raise ValueError('no VITRA observations')
    intrinsic=np.median(all_k,axis=0)
    if not np.allclose(all_k,intrinsic,rtol=.01,atol=1.):raise ValueError('inconsistent source intrinsics')
    for side in hands:hands[side][disagreement[side]]=np.nan
    pairs=np.array(list(pair_map.values()),float).reshape(-1,7)
    meta={'video_id':video.stem,'video_path':str(video),'source_fps':fps,'source_frame_count':n,
          'source_width':width,'source_height':height,'camera_path':str(camera_dir),'track_files':source_files,
          'camera_model':'DA3-Small','metric_scale_reference':'VITRA reconstructed metric camera translations, fitted per clip',
          'hand_source':'VITRA observed joints_camspace, exact native source-frame mapping; no interpolation',
          'confidence_semantics':'binary kept_frames and coordinate/overlap checks; not calibrated probability',
          'hand_overlap_max_median_joint_distance_m':.03,'max_camera_world_consistency_error_m':max(transform_residual,default=0),
          'calibration_config':CALIBRATION,'camera_config':asdict(cfg),'calibration_pair_count':len(pairs),
          'hand_valid_frame_fraction':{s:float(np.isfinite(j).all((1,2)).mean()) for s,j in hands.items()},
          'hand_overlap_conflict_frames':{s:int(disagreement[s].sum()) for s in hands}}
    arrays={**timeline,'alignment_error_deg':errors,'seam':seams,'reliable':reliable,
            'intrinsics':intrinsic,'calibration_pairs':pairs,**{s+'_joints_camspace':j for s,j in hands.items()}}
    return arrays,meta


def fit_scale(source,start,end,fps):
    pairs=source['calibration_pairs'];ctx=CALIBRATION['window_context_s']*fps
    local=pairs[(pairs[:,0]>=start-ctx)&(pairs[:,1]<=end+ctx)]
    if len(local)<CALIBRATION['min_pairs']:return None
    logs=np.log(local[:,2]); center=np.median(logs); mad=float(np.median(np.abs(logs-center)))
    inlier=np.abs(logs-center)<=CALIBRATION['inlier_log_distance']
    if mad>CALIBRATION['max_log_mad'] or inlier.mean()<CALIBRATION['min_inlier_fraction'] or inlier.sum()<CALIBRATION['min_pairs']:return None
    return {'metres_per_reconstruction_unit':float(np.exp(np.median(logs[inlier]))),
            'pair_count':len(local),'inlier_pair_count':int(inlier.sum()),'log_scale_mad':mad,
            'inlier_fraction':float(inlier.mean()),'support_start_frame':int(local[:,0].min()),
            'support_end_frame':int(local[:,1].max()),'translation_scale_status':'estimated_from_VITRA_not_sensor_ground_truth'}


def clip_actions(source,meta,start,end):
    fps=meta['source_fps']
    if not source['reliable'][start:end+1].all():return None,{'passed':False,'reason':'camera_support_or_join'}
    scale=fit_scale(source,start,end,fps)
    if scale is None:return None,{'passed':False,'reason':'insufficient_consistent_metric_scale'}
    # Work in clip-origin world gauge to avoid arbitrary global translation.
    camera=source['camera_to_world'][start:end+1].copy()
    camera[:,:3,3]=(camera[:,:3,3]-camera[0,:3,3].copy())*scale['metres_per_reconstruction_unit']
    hands={}
    for side in ('left','right'):
        j=source[side+'_joints_camspace'][start:end+1]
        world=np.einsum('nij,nkj->nki',camera[:,:3,:3],j)+camera[:,None,:3,3]
        confidence=np.repeat(np.isfinite(j).all((1,2))[:,None],21,axis=1).astype(float)
        hands[side]={'joints_world':world,'confidence':confidence}
    arrays,q=make_actions({'camera_to_world':camera,'hands':hands,'intrinsics':source['intrinsics']},0,end-start,fps,ActionConfig(sampling_hz=fps/round(fps/10)))
    arrays['source_frame']+=start;arrays['timestamp_s']=arrays['source_frame'].astype(float)/fps
    arrays['camera_pose_status']=source['pose_status'][arrays['source_frame']]
    arrays['camera_metric_scale']=np.array(scale['metres_per_reconstruction_unit'])
    for side in ('left','right'):
        arrays[side+'_joints_camspace_m']=source[side+'_joints_camspace'][arrays['source_frame']]
    q['metric_scale_calibration']=scale
    return arrays,q


def candidates(source,meta):
    fps=meta['source_fps'];cfg=native_camera_config(fps);n=meta['source_frame_count']
    raw=mine_long_camera(source,fps,cfg,source['alignment_error_deg'],source['seam'])
    result={'camera_motion':[],'random':[],'rejections':{},'raw_camera_candidates':raw}
    def reject(reason):result['rejections'][reason]=result['rejections'].get(reason,0)+1
    for c in raw:
        if not c['pose_gate_pass']:continue
        a,b=c['clip_start_frame'],c['clip_end_frame'];ca,cb=c['core_start_frame'],c['core_end_frame']
        if not source['reliable'][ca:cb+1].all():reject('camera_core_support');continue
        # Context is up to 1 s; trim context at missing/unsafe camera support.
        left=np.flatnonzero(~source['reliable'][a:ca]);right=np.flatnonzero(~source['reliable'][cb+1:b+1])
        if len(left):a=a+int(left[-1])+1
        if len(right):b=cb+int(right[0])
        c=dict(c);c.update(clip_start_frame=a,clip_end_frame=b,clip_start_s=a/fps,clip_end_s=b/fps,
            context_policy='up to 1 s, truncated at source boundary or invalid camera/join support')
        _,q=clip_actions(source,meta,a,b)
        if q['passed']:
            c['hand_motion_status']='native_observation_support_audited'
            result['camera_motion'].append({'selection':c,'quality':q})
        else:reject(q.get('reason','camera_dense_action_quality'))
    rng=np.random.default_rng(int(hashlib.sha256((str(SEED)+'/random/'+meta['video_id']).encode()).hexdigest()[:16],16))
    for draw in range(max(100,int(n/fps))):
        length=int(rng.integers(round(3*fps)+1,round(10*fps)+2));a=int(rng.integers(0,n-length+1));b=a+length-1
        _,q=clip_actions(source,meta,a,b)
        if q['passed']:result['random'].append({'selection':{'clip_start_frame':a,'clip_end_frame':b,
            'core_start_frame':a,'core_end_frame':b,'segmentation_method':'uniform_random_window',
            'pose_gate_pass':None,'draw_index':draw},'quality':q})
        else:reject('random_'+q.get('reason','dense_action_quality'))
    return result
