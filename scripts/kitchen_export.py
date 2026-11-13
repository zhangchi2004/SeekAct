#!/usr/bin/env python3
"""Native-frame RGB and dense action export for both explicit kitchen sources."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from lime_pipeline.dense_actions import load_egodex, make_actions
from lime_pipeline.kitchen_epic import clip_actions
from lime_pipeline.render import _ffmpeg_executable
from kitchen_inventory import KITCHEN_TASKS
from kitchen_plan import cohort


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def recipe():
    repo = Path(__file__).resolve().parents[1]
    paths = ['scripts/kitchen_export.py', 'src/lime_pipeline/dense_actions.py',
             'src/lime_pipeline/kitchen_epic.py', 'src/lime_pipeline/vitra_motion.py']
    return {name: sha256(repo/name) for name in paths}


def source_actions(record, source, meta):
    a, b = record['selection']['clip_start_frame'], record['selection']['clip_end_frame']
    fps = record['source_fps']
    if record['source_dataset'] == 'EPIC-KITCHENS':
        arrays, quality = clip_actions(source, meta, a, b)
        if arrays is None:
            raise ValueError(quality)
        for side in ('left', 'right'):
            arrays[side+'_confidence'] = source[side+'_detector_confidence'][arrays['source_frame']]
            arrays[side+'_native_observed'] = np.isfinite(source[side+'_joints_camspace'][a:b+1]).all((1,2))
            arrays[side+'_native_confidence'] = source[side+'_detector_confidence'][a:b+1]
        arrays['native_camera_pose_status'] = source['pose_status'][a:b+1]
        arrays['native_camera_reliable'] = source['reliable'][a:b+1]
    else:
        arrays, quality = make_actions(source, a, b, fps)
        frames = arrays['source_frame']; camera_inverse = np.linalg.inv(source['camera_to_world'][frames])
        for side in ('left', 'right'):
            hand = source['hands'].get(side)
            if hand is None:
                arrays[side+'_joints_camspace_m'] = np.full((len(frames),21,3), np.nan)
                arrays[side+'_native_observed'] = np.zeros(b-a+1, bool)
                arrays[side+'_native_confidence'] = np.zeros(b-a+1)
            else:
                joints = hand['joints_world'][frames]
                arrays[side+'_joints_camspace_m'] = np.einsum('nij,nkj->nki', camera_inverse[:,:3,:3], joints)+camera_inverse[:,None,:3,3]
                confidence = np.min(hand['confidence'][a:b+1][:,[0,4,8,12]], axis=1)
                arrays[side+'_native_confidence'] = confidence
                arrays[side+'_native_observed'] = (confidence >= .25) & np.isfinite(hand['joints_world'][a:b+1]).all((1,2))
        arrays['camera_pose_status'] = np.ones(len(frames), np.uint8)
        arrays['native_camera_pose_status'] = np.ones(b-a+1, np.uint8)
        arrays['native_camera_reliable'] = np.ones(b-a+1, bool)
    if not quality['passed']:
        raise ValueError('Dense camera/manipulation usability failed')
    # Double precision source timestamps avoid accumulated error late in long videos.
    arrays['timestamp_s'] = arrays['source_frame'].astype(np.float64)/fps
    arrays['clip_timestamp_s'] = (arrays['source_frame']-a).astype(np.float64)/fps
    arrays['action_dt_s'] = np.diff(arrays['source_frame']).astype(np.float64)/fps
    arrays['native_source_frame'] = np.arange(a,b+1)
    return arrays, quality


def export_video(record, folder, root):
    start, end = record['selection']['clip_start_frame'], record['selection']['clip_end_frame']
    fps = record['source_fps']; total = end-start+1
    cap = cv2.VideoCapture(record['video_path'])
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != record['source_frame_count'] or abs(cap.get(cv2.CAP_PROP_FPS)-fps)>1e-4:
        raise ValueError('RGB/track source frame count or FPS mismatch')
    width = 960; height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*width/cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2))*2
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
        raise ValueError('Decoder cannot seek to the exact requested source frame')
    chosen = set(np.rint(np.linspace(0,total-1,16)).astype(int).tolist())
    source_evidence = {}; endpoint_qc = []; raw_hash = hashlib.sha256()
    tmp = folder/'video.tmp.mp4'
    cmd = [_ffmpeg_executable(), '-nostdin','-hide_banner','-loglevel','error','-y',
           '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{width}x{height}','-framerate',str(fps),'-i','pipe:0',
           '-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-tag:v','avc1','-movflags','+faststart',str(tmp)]
    with (folder/'encoder.log').open('wb') as log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(total):
                ok, frame = cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(start+i+1))>.01:
                    raise ValueError(f'Exact source decode failed at frame {start+i}')
                if i in (0,total-1):
                    gray = cv2.cvtColor(cv2.resize(frame,(640,360)),cv2.COLOR_BGR2GRAY)
                    qc = {'source_frame':start+i,'sharpness':float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                          'usable_exposure':float(((gray>10)&(gray<245)).mean())}
                    endpoint_qc.append(qc)
                    if qc['sharpness']<30 or qc['usable_exposure']<.70:
                        raise ValueError('Blurry or poorly exposed source endpoint')
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                raw_hash.update(frame.tobytes())
                if i in chosen:
                    source_evidence[i] = frame.copy()
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=180) != 0:
                raise ValueError('Video encoder failed; see encoder.log')
        except BaseException:
            process.kill(); process.wait(); raise
        finally:
            cap.release()
    encoded = cv2.VideoCapture(str(tmp)); evidence = []; differences = []; frames = 0
    if abs(encoded.get(cv2.CAP_PROP_FPS)-fps)>1e-4:
        raise ValueError('Export FPS changed')
    while True:
        ok, frame = encoded.read()
        if not ok:
            break
        if frames in chosen:
            ref = source_evidence[frames]
            mae = float(np.abs(frame.astype(float)-ref).mean()); differences.append(mae)
            if mae>10:
                raise ValueError('Encoded evidence frame differs from the exact source frame')
            name = folder/f'frame_{frames:04d}.jpg'
            Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(name,quality=92)
            evidence.append({'image':str(name.relative_to(root)), 'clip_frame':frames,'source_frame':start+frames,
                'clip_seconds':frames/fps,'source_seconds':(start+frames)/fps,'sha256':sha256(name),
                'source_bgr_resized_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),'source_encoded_mae':mae})
        frames += 1
    encoded.release()
    if frames!=total or len(evidence)!=16:
        raise ValueError('Export decode count or evidence mismatch')
    tmp.replace(folder/'video.mp4')
    return {'path':str((folder/'video.mp4').relative_to(root)),'fps':fps,'frame_count':total,'width':width,'height':height,
            'source_start_frame':start,'source_end_frame_inclusive':end,'sha256':sha256(folder/'video.mp4'),
            'decoded_frames_verified':frames,'source_endpoint_mae':[differences[0],differences[-1]],
            'source_evidence_mae':differences,'source_resized_bgr_stream_sha256':raw_hash.hexdigest(),
            'endpoint_quality':endpoint_qc,'decode_policy':'OpenCV exact indexed seek then sequential native frames piped to H264'}, evidence


def export_source(item):
    records, output = item; root = Path(output); cv2.setNumThreads(1); results = []
    ref = records[0]; source_meta = json.loads(Path(ref['source_inventory']).read_text())
    if ref['source_dataset']=='EgoDex':
        assert ref['task_category'] in KITCHEN_TASKS
        source = load_egodex(Path(ref['pose_path']))
        camera_source = 'EgoDex native metric camera-to-world; supplied camera coordinates use +Z forward'
        hand_source = 'EgoDex native joint transforms and confidence'
    else:
        source = dict(np.load(ref['pose_path']))
        assert source_meta['assigned_cohort']==ref['cohort'] and 'WiLoR' in source_meta['hand_model']
        camera_source = 'EPIC-Fields OpenCV camera-to-world; camera looks along +Z; VITRA estimated metric scale per clip'
        hand_source = source_meta['hand_inference']
    hashes = {'source_video_sha256':sha256(ref['video_path']),'source_pose_sha256':sha256(ref['pose_path'])}
    code = recipe(); code_hash = hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    for record in records:
        folder = root/'samples'/record['sample_id']; folder.mkdir(parents=True,exist_ok=True)
        try:
            assert cohort(record['source_dataset'],record['source_id'])==record['cohort']
            assert sha256(record['source_inventory'])==record['source_inventory_sha256']
            fingerprint = hashlib.sha256(json.dumps({'record':record,'code':code,'inputs':hashes},sort_keys=True).encode()).hexdigest()
            done = folder/'sample.json'
            if done.exists():
                old = json.loads(done.read_text())
                if old['provenance']['export_fingerprint']!=fingerprint:
                    raise ValueError('Immutable sample cache changed; use a new version')
                if sha256(root/old['video']['path'])!=old['video']['sha256'] or sha256(root/old['target']['actions'])!=old['target']['sha256']:
                    raise ValueError('Cached sample artifact hash mismatch')
                results.append({'sample_id':record['sample_id'],'status':'cached'}); continue
            arrays, quality = source_actions(record,source,source_meta)
            video, evidence = export_video(record,folder,root)
            tmp = folder/'actions.tmp.npz'; np.savez_compressed(tmp,**arrays); tmp.replace(folder/'actions.npz')
            out = {'schema_version':'lime.video_actions.v1',**record,'quality':quality,'video':video,
                'input':{'video':video['path'],'task_prompt':None},
                'target':{'actions':str((folder/'actions.npz').relative_to(root)),'camera':'camera_action_9d','camera_dim':9,
                    'manipulation':'manipulation_action_7d','manipulation_dim':7,'primary_hand':quality['primary_hand'],
                    'state_count':len(arrays['source_frame']),'action_count':len(arrays['action_dt_s']),
                    'sha256':sha256(folder/'actions.npz')},'vlm_evidence':evidence,
                'provenance':{**hashes,'export_fingerprint':fingerprint,'export_recipe_sha256':code_hash,'code_sha256':code,
                    'camera_source':camera_source,'hand_source':hand_source,'label_status':'pending_new_kitchen_VLM_audit',
                    'source_inventory_sha256':record['source_inventory_sha256'],
                    'camera_action_convention':'inv(camera_t) @ camera_tplus1, translation and rotation columns 1,2',
                    'manipulation_action_convention':'fixed clip-start camera C0; delta position, spatial rotation vector, current gripper opening',
                    'translation_status':'native_metric' if record['source_dataset']=='EgoDex' else 'estimated_metric_not_sensor_ground_truth'}}
            save(done,out); results.append({'sample_id':record['sample_id'],'status':'exported'})
        except Exception as error:
            fail = {'sample_id':record['sample_id'],'status':'failed','error':f'{type(error).__name__}: {error}'}
            save(folder/'export_error.json',fail); results.append(fail)
    return results


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int,default=0); a = p.parse_args()
    rows = json.loads(a.manifest.read_text()); rows = rows[:a.limit] if a.limit else rows
    groups = defaultdict(list)
    for row in rows:
        groups[(row['source_dataset'],row['source_id'])].append(row)
    counts = Counter(); began = time.monotonic(); reports = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(export_source,(group,str(a.root))) for group in groups.values()]
        for job in as_completed(jobs):
            results = job.result(); reports.extend(results); counts.update(r['status'] for r in results)
            report = {'manifest':str(a.manifest),'planned':len(rows),'finished':len(reports),**dict(counts),
                      'elapsed_s':round(time.monotonic()-began,1),'latest':results}
            save(a.root/f'export_progress.{a.manifest.stem}.json',report); print(json.dumps(report),flush=True)
    save(a.root/f'export_report.{a.manifest.stem}.json',{'manifest':str(a.manifest),'planned':len(rows),**dict(counts),'samples':reports})
    if not counts['exported']+counts['cached']:
        raise RuntimeError('No valid exports')


if __name__=='__main__':
    main()
