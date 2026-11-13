from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lime_pipeline.long_camera import LongCameraConfig, dense_timeline, mine_long_camera, relative_metrics, select_diverse_candidates


def poses_for_angles(angles, axis="y"):
    poses = np.repeat(np.eye(4)[None], len(angles), axis=0)
    poses[:, :3, :3] = Rotation.from_euler(axis, angles, degrees=True).as_matrix()
    return poses


def test_large_turn_gets_motion_boundaries_and_context():
    fps = 20
    angles = np.r_[np.zeros(30), np.linspace(0, 60, 41), np.full(30, 60.)]
    config = LongCameraConfig()
    timeline = dense_timeline(np.arange(len(angles)), poses_for_angles(angles), len(angles), fps, config)
    accepted = [x for x in mine_long_camera(timeline, fps, config) if x["pose_gate_pass"]]
    assert len(accepted) == 1
    event = accepted[0]
    assert event["net_rotation_deg"] > 55
    assert event["clip_start_frame"] == event["core_start_frame"] - fps
    assert event["clip_end_frame"] == event["core_end_frame"] + fps
    assert event["observation_gain_status"] == "not_vlm_audited"


def test_roll_is_not_a_changed_view_axis():
    poses = poses_for_angles([0, 60], axis="z")
    metrics = relative_metrics(poses, 0, 1)
    assert np.isclose(metrics["net_rotation_deg"], 60)
    assert np.isclose(metrics["view_axis_change_deg"], 0)


def test_out_and_back_is_not_large_endpoint_change():
    poses = poses_for_angles([0, 45, 80, 45, 0])
    assert np.isclose(relative_metrics(poses, 0, 4)["net_rotation_deg"], 0)


def test_only_short_internal_gaps_are_interpolated():
    config = LongCameraConfig(max_interpolation_gap_s=.25)
    timeline = dense_timeline(np.array([0, 2, 15, 19]), poses_for_angles([0, 1, 5, 6]), 25, 20, config)
    assert timeline["pose_status"][1] == 2
    assert np.all(timeline["pose_status"][3:15] == 0)
    assert np.all(timeline["pose_status"][20:] == 0)
    assert np.isnan(timeline["camera_to_world"][10]).all()


def test_chunk_join_is_rejected_even_if_turn_is_large():
    angles = np.r_[np.zeros(30), np.linspace(0, 60, 41), np.full(30, 60.)]
    config = LongCameraConfig()
    timeline = dense_timeline(np.arange(len(angles)), poses_for_angles(angles), len(angles), 20, config)
    seams = np.zeros(len(angles), dtype=bool)
    seams[50] = True
    candidates = mine_long_camera(timeline, 20, config, seams=seams)
    assert any("crosses_reconstruction_chunk_join" in e["rejection_reasons"] for e in candidates)
    assert not any(e["pose_gate_pass"] for e in candidates)


def test_seeded_selection_deduplicates_clips_and_is_repeatable():
    events = [dict(video_id="v", clip_start_frame=i, clip_end_frame=i + 100, event_id=str(i)) for i in range(0, 500, 20)]
    config = LongCameraConfig()
    for event in events:
        event["core_start_frame"] = event["clip_start_frame"] + 10
    a = select_diverse_candidates(events, config, 10, 4)
    b = select_diverse_candidates(events, config, 10, 4)
    assert a == b
    assert len(a) <= 4
