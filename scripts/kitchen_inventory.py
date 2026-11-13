#!/usr/bin/env python3
"""Audit only the explicit kitchen sources; never broadens the task allowlist."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from lime_pipeline.dense_actions import load_egodex, make_actions
from lime_pipeline.long_camera import LongCameraConfig, dense_timeline, mine_long_camera

KITCHEN_TASKS = ('boil_serve_egg', 'fry_bread', 'fry_egg', 'stock_unstock_fridge',
                 'wash_kitchen_dishes', 'wash_put_away_dishes', 'wipe_kitchen_surfaces')
SEED = 20260920


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def inspect(args):
    filename, root, out, fingerprint = args
    path = Path(filename); root = Path(root); out = Path(out)
    rel = str(path.relative_to(root)); sid = digest('EgoDex/' + rel)[:20]
    cache = out / 'egodex' / 'inventory' / f'{sid}.json'
    if cache.exists():
        record = json.loads(cache.read_text())
        if record.get('fingerprint') == fingerprint:
            return record
        raise ValueError('inventory cache fingerprint changed; use a new version')
    assert path.parent.name in KITCHEN_TASKS
    row = {'source_id': sid, 'source_dataset': 'EgoDex', 'task_category': path.parent.name,
           'relative_source': rel, 'pose_path': str(path), 'video_path': str(path.with_suffix('.mp4')),
           'fingerprint': fingerprint, 'camera_motion': [], 'random': [], 'rejections': {}}
    try:
        cv2.setNumThreads(1)
        data = load_egodex(path)
        n = len(data['camera_to_world']); fps = 30.
        video = cv2.VideoCapture(row['video_path'])
        actual_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)); actual_fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        if n != actual_count or abs(actual_fps - fps) > 1e-4:
            raise ValueError(f'RGB/pose alignment mismatch: {actual_count}, {n}, {actual_fps}')
        row.update(source_frame_count=n, source_fps=fps, source_seconds=n/fps)
        cfg = LongCameraConfig(selection_seed=SEED)
        timeline = dense_timeline(np.arange(n), data['camera_to_world'], n, fps, cfg)
        candidates = mine_long_camera(timeline, fps, cfg)
        row['camera_candidate_count'] = len(candidates)
        row['camera_geometric_count'] = sum(c['pose_gate_pass'] for c in candidates)
        rejection = Counter(reason for c in candidates for reason in c['rejection_reasons'])
        for c in candidates:
            if not c['pose_gate_pass']:
                continue
            c['translation_is_metric'] = True
            c['translation_m'] = c.pop('translation_reconstruction_units')
            _, q = make_actions(data, c['clip_start_frame'], c['clip_end_frame'], fps)
            if not q['passed']:
                rejection['dense_action_quality'] += 1
                continue
            row['camera_motion'].append({'selection': c, 'quality': q})
        # Prospective random windows for coverage audit. Final cohorts must assign
        # disjoint source videos before any output is admitted to either cohort.
        if n >= 91:
            rng = np.random.default_rng(int(digest(rel + '/random/' + str(SEED))[:16], 16))
            for attempt in range(3):
                length = int(rng.integers(91, min(n, 301)+1))
                a = int(rng.integers(0, n-length+1)); b = a+length-1
                _, q = make_actions(data, a, b, fps)
                selection = {'clip_start_frame':a, 'clip_end_frame':b, 'core_start_frame':a,
                             'core_end_frame':b, 'segmentation_method':'uniform_random_window',
                             'pose_gate_pass':None, 'draw_index':attempt}
                if q['passed']:
                    row['random'].append({'selection':selection, 'quality':q})
                else:
                    rejection['random_dense_action_quality'] += 1
        row['rejections'] = dict(rejection)
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    save(cache, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    (args.output / 'egodex' / 'inventory').mkdir(parents=True, exist_ok=True)
    paths = sorted(p for part in range(1,6) for task in KITCHEN_TASKS
                   for p in (args.root / f'part{part}' / task).glob('*.hdf5') if p.with_suffix('.mp4').is_file())
    repo = Path(__file__).resolve().parents[1]
    fingerprint = digest(''.join((repo/name).read_text() for name in
        ['scripts/kitchen_inventory.py','src/lime_pipeline/dense_actions.py','src/lime_pipeline/long_camera.py']))
    save(args.output / 'egodex' / 'source_inventory.json', {'source_count':len(paths),
         'kitchen_allowlist':KITCHEN_TASKS, 'task_counts':dict(Counter(p.parent.name for p in paths)),
         'thresholds':asdict(LongCameraConfig(selection_seed=SEED)), 'fingerprint':fingerprint})
    counts = Counter(); tasks = {}; began = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for i,row in enumerate(pool.map(inspect, [(str(p),str(args.root),str(args.output),fingerprint) for p in paths]),1):
            rows.append(row)
            counts.update({'sources':1,'errors':int('error' in row),'camera_sources':int(bool(row['camera_motion'])),
                           'camera_clips':len(row['camera_motion']), 'random_sources':int(bool(row['random']))})
            if i % 50 == 0 or i == len(paths):
                report = {**dict(counts),'total_sources':len(paths),'elapsed_s':round(time.monotonic()-began,1)}
                print(json.dumps(report),flush=True); save(args.output/'egodex'/'inventory_progress.json',report)
    save(args.output / 'egodex' / 'inventory_report.json', {**dict(counts),'fingerprint':fingerprint,
         'by_task':{task:{'sources':sum(r['task_category']==task for r in rows),
            'camera_clips':sum(len(r['camera_motion']) for r in rows if r['task_category']==task),
            'camera_sources':sum(bool(r['camera_motion']) for r in rows if r['task_category']==task)} for task in KITCHEN_TASKS}})


if __name__ == '__main__':
    main()
