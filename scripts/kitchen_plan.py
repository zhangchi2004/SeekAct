#!/usr/bin/env python3
"""Freeze prospective, source-disjoint kitchen candidates without quota relaxation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from kitchen_inventory import KITCHEN_TASKS

SEED = 20260920


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cohort(dataset, source_id):
    # This is also the EPIC WiLoR inference partition. Assign before hand QC.
    return 'random' if int(digest(f'{SEED}/cohort/{dataset}/{source_id}'), 16) % 5 == 0 else 'camera_motion'


def temporal_iou(a, b):
    lo, hi = a['clip_start_frame'], a['clip_end_frame']
    x, y = b['clip_start_frame'], b['clip_end_frame']
    overlap = max(0, min(hi, y) - max(lo, x) + 1)
    return overlap / (hi - lo + 1 + y - x + 1 - overlap)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--dataset', choices=['EgoDex', 'EPIC-KITCHENS'], required=True)
    p.add_argument('--random-cap', type=int, default=180)
    p.add_argument('--epic-inventory-dir', default='aligned-fields-wilor-v2')
    a = p.parse_args()
    paths = (a.root/'egodex/inventory').glob('*.json') if a.dataset == 'EgoDex' else (a.root/'epic'/a.epic_inventory_dir).glob('*/inventory.json')
    rows = []; counts = Counter(); reasons = Counter()
    for path in sorted(paths):
        source = json.loads(path.read_text())
        if source.get('error'):
            counts['source_error'] += 1
            continue
        if a.dataset == 'EgoDex':
            assert source['task_category'] in KITCHEN_TASKS
            sid = source['source_id']
        else:
            sid = source['video_id']
            assert 'WiLoR' in source['hand_model']
        group = cohort(a.dataset, sid)
        if a.dataset == 'EPIC-KITCHENS':
            assert source['assigned_cohort'] == group
        counts[f'{group}_sources'] += 1
        reasons.update(source.get('rejections', {}))
        for event in source[group]:
            selection = event['selection']
            if group == 'camera_motion':
                assert selection['pose_gate_pass']
                assert selection['net_rotation_deg'] >= 30 and selection['view_axis_change_deg'] >= 20
                assert .75 <= selection['core_duration_s'] <= 8
            else:
                assert selection['segmentation_method'] == 'uniform_random_window'
            start, end = selection['clip_start_frame'], selection['clip_end_frame']
            assert event['quality']['passed']
            sample_id = f'{group}__{a.dataset.lower().replace("-", "_")}__{sid}__{start:08d}_{end:08d}'
            rows.append({'sample_id': sample_id, 'source_dataset': a.dataset, 'source_id': sid,
                'cohort': group, 'source_partition_rule': 'sha256(20260920/cohort/dataset/source_id) modulo 5; 0=random',
                'selection_seed': SEED, 'video_path': source['video_path'], 'pose_path': source['pose_path'],
                'source_fps': source['source_fps'], 'source_frame_count': source['source_frame_count'],
                'source_inventory': str(path), 'source_inventory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'task_category': source.get('task_category', 'kitchen_scene_pending_visual_audit'),
                'selection': selection, 'quality': event['quality']})
    # Uniform hash ordering, unrelated to motion strength or hand activity.
    rows.sort(key=lambda r: digest(f'{SEED}/candidate/'+r['sample_id']))
    kept = []; by_source = defaultdict(list); group_counts = Counter()
    for row in rows:
        key = (row['source_dataset'], row['source_id'])
        if any(temporal_iou(row['selection'], r['selection']) > .20 for r in by_source[key]):
            counts['temporal_overlap_rejected'] += 1
            continue
        if row['cohort'] == 'random' and group_counts['random'] >= a.random_cap:
            continue
        by_source[key].append(row); kept.append(row); group_counts[row['cohort']] += 1
    kept = [r for pair in itertools.zip_longest(*[[r for r in kept if r['cohort']==c]
            for c in ('camera_motion','random')]) for r in pair if r is not None]
    target = a.root/'manifests'/f'{a.name}.json'; target.parent.mkdir(exist_ok=True)
    body = json.dumps(kept, indent=2, allow_nan=False)+'\n'
    if target.exists() and target.read_text() != body:
        raise ValueError('Frozen plan differs; use a new manifest name')
    target.write_text(body)
    report = {'dataset': a.dataset, 'plan': str(target), 'candidate_counts_before_overlap': dict(Counter(r['cohort'] for r in rows)),
              'planned_counts': dict(group_counts), 'distinct_sources': sum(bool(v) for v in by_source.values()), 'source_audit': dict(counts),
              'inventory_rejections_all_cohorts': dict(reasons), 'status': 'prospective_not_final_release'}
    target.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
