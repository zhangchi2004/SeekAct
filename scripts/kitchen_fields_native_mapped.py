"""Build native-frame camera inventories using measured 60 FPS frame identities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from kitchen_export import save, sha256
from lime_pipeline.kitchen_epic_native import prepare_source, candidates


def inspect(item):
    marker, root = map(Path,item); mapping=json.loads(marker.read_text());vid=mapping['video_id']
    assert mapping['status']=='measured_correspondence_requires_independent_pose_audit'
    mapping_path=marker.parent/'mapping.npz'
    assert sha256(mapping_path)==mapping['mapping_npz_sha256']
    original=root/'epic/fields-local/json'/(vid+'.json')
    repo=Path(__file__).resolve().parents[1]
    code={name:sha256(repo/name) for name in ('scripts/kitchen_fields_native_mapped.py',
        'src/lime_pipeline/kitchen_epic_native.py','src/lime_pipeline/long_camera.py','src/lime_pipeline/dense_actions.py')}
    proof={'frame_mapping_json_sha256':sha256(marker),'original_fields_sha256':sha256(original),'code_sha256':code}
    fingerprint=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
    out=root/'epic/aligned-fields-native-v2'/vid;out.mkdir(parents=True,exist_ok=True)
    done=out/'inventory.json'
    if done.exists():
        cached=json.loads(done.read_text());assert cached['fingerprint']==fingerprint
        return cached
    video=Path(mapping['video_path']);assert sha256(video)==mapping['video_sha256']
    d=json.loads(original.read_text());m=np.load(mapping_path);native_count=mapping['native_frames']
    cap=cv2.VideoCapture(str(video));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert n==native_count and np.isclose(fps,60000/1001,rtol=0,atol=1e-5)
    np.testing.assert_allclose(m['native_timestamp_s'],np.arange(n)/fps,atol=1e-9,rtol=0)
    ext_to_native=m['extracted_to_native'];assigned={};origin_indices={};skipped={'outside_extraction':0,'ambiguous_native_identity':0,'duplicate_registered_image':0}
    for name,pose in sorted(d['images'].items()):
        extracted=int(re.search(r'(\d+)\.[^.]+$',name)[1])-1
        if extracted<0 or extracted>=len(ext_to_native):
            skipped['outside_extraction']+=1;continue
        index=int(ext_to_native[extracted])
        if index<0:
            skipped['ambiguous_native_identity']+=1;continue
        assert 0<=index<n
        if index in assigned:
            skipped['duplicate_registered_image']+=1;continue
        assigned[index]=pose;origin_indices[index]=extracted
    assert len(assigned)>=2
    indices=np.array(sorted(assigned));values=np.array([assigned[i] for i in indices])
    camera_dir=root/'epic/camera-fields-native-v2'/vid;camera_dir.mkdir(parents=True,exist_ok=True)
    w2c=np.repeat(np.eye(4)[None],len(indices),axis=0)
    w2c[:,:3,:3]=Rotation.from_quat(values[:,[1,2,3,0]]).as_matrix();w2c[:,:3,3]=values[:,4:7]
    np.savez_compressed(camera_dir/'camera_samples.npz',source_frame_indices=indices,
        camera_to_world=np.linalg.inv(w2c),alignment_error_deg=np.zeros(len(indices)),seam=np.zeros(len(indices),bool),
        original_extracted_frame_indices=np.array([origin_indices[i] for i in indices]))
    derived=root/'epic/fields-native-mapped/json'/(vid+'.json');derived.parent.mkdir(parents=True,exist_ok=True)
    save(derived,{'camera':d['camera'],'images':{f'frame_{i+1:010d}.jpg':assigned[i] for i in indices},
        'mapping_provenance':proof,'note':'Derived native-frame indexing. Raw original Fields JSON and exact image-identity evidence are retained.'})
    source,meta=prepare_source(video,camera_dir,root/'epic/vitra')
    meta.update(camera_model='EPIC-Fields full camera reconstruction with measured 60 FPS extraction to native-frame mapping',
        fields_json=str(derived),fields_sha256=sha256(derived),original_fields_json=str(original),
        original_fields_sha256=proof['original_fields_sha256'],frame_mapping_json=str(marker),
        frame_mapping_json_sha256=proof['frame_mapping_json_sha256'],source_video_sha256=mapping['video_sha256'],
        registration_frame_mapping='Exact monotonic decoded-image hash correspondence, retained native source indices; first registered pose per identical extracted image',
        mapping_exclusions=skipped,native_timing='Full-source constant frame spacing verified from decoded frame timestamps; time zero is native source frame 0',
        alignment_error_semantics='no sliding-window alignment; zero means not applicable, not known zero error',
        frame_alignment_status='independent RGB/pose audit pending')
    np.savez_compressed(out/'source_tracks.npz',**source)
    row={'source_dataset':'EPIC-KITCHENS','source_id':hashlib.sha256(('EPIC-KITCHENS/'+vid).encode()).hexdigest()[:20],
        'task_category':'kitchen','pose_path':str(out/'source_tracks.npz'),'fingerprint':fingerprint,
        'mapping_provenance':proof,**meta,**candidates(source,meta)}
    save(done,row)
    print(json.dumps({'video_id':vid,'native_frames':n,'mapped_registered_frames':len(indices),
        'exclusions':skipped,'camera_geometric':sum(c['pose_gate_pass'] for c in row['raw_camera_candidates']),
        'camera':len(row['camera_motion']),'random':len(row['random'])}),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();paths=sorted((a.root/'epic-native/frame-mapping-v1').glob('*/mapping.json'))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(inspect,[(str(path),str(a.root)) for path in paths]))
    save(a.root/'epic/fields_inventory_report_native_mapped.json',{'sources':len(rows),
        'camera_geometric':sum(sum(c['pose_gate_pass'] for c in r['raw_camera_candidates']) for r in rows)})
