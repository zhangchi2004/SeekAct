"""Camera-first temporal segmentation, independent of VITRA episodes/action labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class LongCameraConfig:
    onset_speed_deg_s: float = 20.0
    offset_speed_deg_s: float = 8.0
    bridge_pause_s: float = 0.25
    velocity_window_s: float = 0.20
    min_core_duration_s: float = 0.75
    max_core_duration_s: float = 8.0
    min_net_rotation_deg: float = 30.0
    min_view_axis_change_deg: float = 20.0
    max_peak_speed_deg_s: float = 400.0
    max_interpolation_gap_s: float = 0.24
    max_alignment_error_deg: float = 8.0
    context_before_s: float = 1.0
    context_after_s: float = 1.0
    min_endpoint_sharpness: float = 30.0
    min_usable_exposure_fraction: float = 0.70
    max_temporal_iou: float = 0.20
    selection_seed: int = 20260908


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ranges; a missing pose must split rather than bridge a run."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def dense_timeline(
    indices: np.ndarray,
    camera_to_world: np.ndarray,
    frame_count: int,
    fps: float,
    config: LongCameraConfig,
) -> dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    poses = np.asarray(camera_to_world, dtype=float)
    if poses.shape != (len(indices), 4, 4) or len(indices) < 2:
        raise ValueError("Expected at least two indexed 4x4 camera-to-world poses")
    if np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= frame_count:
        raise ValueError("Source-frame indices must be strictly increasing and in bounds")
    if not np.isfinite(poses).all():
        raise ValueError("Non-finite input pose")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Non-homogeneous pose")
    rr = poses[:, :3, :3]
    if not np.allclose(rr @ np.swapaxes(rr, 1, 2), np.eye(3), atol=2e-3):
        raise ValueError("Pose rotation is not orthogonal")
    if np.any(np.linalg.det(rr) < 0.99):
        raise ValueError("Pose rotation has reflection or invalid scale")
    dense = np.full((frame_count, 4, 4), np.nan)
    status = np.zeros(frame_count, dtype=np.uint8)  # 0 missing, 1 supplied estimate, 2 short-gap interpolation
    dense[indices] = poses
    status[indices] = 1
    for i in np.flatnonzero((np.diff(indices) > 1) & (np.diff(indices) / fps <= config.max_interpolation_gap_s)):
        left, right = indices[i:i + 2]
        angle = Rotation.from_matrix(rr[i].T @ rr[i + 1]).magnitude() * 180 / np.pi
        if angle / ((right - left) / fps) > config.max_peak_speed_deg_s:
            continue
        query = np.arange(left + 1, right)
        amount = (query - left) / (right - left)
        interpolated = np.repeat(np.eye(4)[None], len(query), axis=0)
        interpolated[:, :3, :3] = Slerp([left, right], Rotation.from_matrix(rr[i:i + 2]))(query).as_matrix()
        interpolated[:, :3, 3] = (1 - amount[:, None]) * poses[i, :3, 3] + amount[:, None] * poses[i + 1, :3, 3]
        dense[query] = interpolated
        status[query] = 2
    valid = status > 0
    angular_speed = np.full(frame_count, np.nan)
    translation_speed = np.full(frame_count, np.nan)
    radius = max(1, round(config.velocity_window_s * fps / 2))
    for start, end in true_runs(valid):
        query = np.arange(start, end + 1)
        left, right = np.maximum(query - radius, start), np.minimum(query + radius, end)
        if end == start:
            continue
        dt = (right - left) / fps
        relative = np.swapaxes(dense[left, :3, :3], 1, 2) @ dense[right, :3, :3]
        angular_speed[query] = np.rad2deg(Rotation.from_matrix(relative).magnitude()) / dt
        translation_speed[query] = np.linalg.norm(dense[right, :3, 3] - dense[left, :3, 3], axis=1) / dt
        angular_speed[query] = median_filter(angular_speed[query], size=3, mode="nearest")
    return {"camera_to_world": dense, "pose_status": status, "valid": valid,
            "angular_speed_deg_s": angular_speed, "translation_speed_units_s": translation_speed}


def relative_metrics(poses: np.ndarray, start: int, end: int) -> dict[str, Any]:
    a, b = poses[start], poses[end]
    relative = np.linalg.inv(a) @ b
    rotation = float(np.rad2deg(Rotation.from_matrix(relative[:3, :3]).magnitude()))
    axis_angle = float(np.rad2deg(np.arccos(np.clip(np.dot(a[:3, 2], b[:3, 2]), -1, 1))))
    direction_vector = relative[:3, 2]
    if abs(direction_vector[0]) >= abs(direction_vector[1]):
        direction = "look right" if direction_vector[0] > 0 else "look left"
    else:
        direction = "look down" if direction_vector[1] > 0 else "look up"
    return {"net_rotation_deg": rotation, "view_axis_change_deg": axis_angle,
            "translation_reconstruction_units": float(np.linalg.norm(relative[:3, 3])),
            "relative_camera_to_world": relative.tolist(), "dominant_view_change": direction,
            "translation_is_metric": False}


def mine_long_camera(
    timeline: dict[str, np.ndarray], fps: float, config: LongCameraConfig,
    alignment_error: np.ndarray | None = None,
    seams: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    speed, valid = timeline["angular_speed_deg_s"], timeline["valid"]
    moving = valid & np.isfinite(speed) & (speed >= config.offset_speed_deg_s)
    for start, end in true_runs(~moving):
        if start and end + 1 < len(moving) and valid[start:end + 1].all() and (end - start + 1) / fps <= config.bridge_pause_s:
            moving[start:end + 1] = True
    intervals = []
    for start, end in true_runs(moving):
        if np.nanmax(speed[start:end + 1]) < config.onset_speed_deg_s:
            continue
        if (end - start) / fps <= config.max_core_duration_s:
            intervals.append((start, end, "hysteresis_onset_offset"))
        else:
            # Long uninterrupted movement: evaluate bounded windows, recorded explicitly.
            window = round(config.max_core_duration_s * fps)
            for begin in range(start, end, max(1, window // 2)):
                intervals.append((begin, min(begin + window, end), "bounded_window_in_sustained_motion"))
    candidates = []
    for start, end, method in intervals:
        metrics = relative_metrics(timeline["camera_to_world"], start, end)
        duration = (end - start) / fps
        failures = []
        if duration < config.min_core_duration_s:
            failures.append("motion_too_short")
        if metrics["net_rotation_deg"] < config.min_net_rotation_deg:
            failures.append("net_rotation_below_threshold")
        if metrics["view_axis_change_deg"] < config.min_view_axis_change_deg:
            failures.append("view_axis_change_below_threshold")
        peak = float(np.nanmax(speed[start:end + 1]))
        if peak > config.max_peak_speed_deg_s:
            failures.append("implausible_angular_speed")
        if alignment_error is not None and float(np.max(alignment_error[start:end + 1])) > config.max_alignment_error_deg:
            failures.append("chunk_overlap_disagreement")
        if seams is not None and seams[start:end + 1].any():
            failures.append("crosses_reconstruction_chunk_join")
        clip_start = max(0, start - round(config.context_before_s * fps))
        clip_end = min(len(speed) - 1, end + round(config.context_after_s * fps))
        after = speed[end + 1:min(len(speed), end + 1 + round(0.4 * fps))]
        finite_after = after[np.isfinite(after)]
        candidates.append({"core_start_frame": start, "core_end_frame": end,
                           "clip_start_frame": clip_start, "clip_end_frame": clip_end,
                           "core_start_s": start / fps, "core_end_s": end / fps,
                           "core_duration_s": duration, "clip_start_s": clip_start / fps,
                           "clip_end_s": clip_end / fps, "peak_angular_speed_deg_s": peak,
                           "post_motion_mean_speed_deg_s": float(np.mean(finite_after)) if len(finite_after) else None,
                           "segmentation_method": method, **metrics,
                           "pose_gate_pass": not failures, "rejection_reasons": failures,
                           "observation_gain_status": "not_vlm_audited",
                           "hand_motion_status": "not_estimated_for_this_new_interval"})
    return candidates


def select_diverse_candidates(candidates: list[dict], config: LongCameraConfig, count: int, per_video: int) -> list[dict]:
    rng = np.random.default_rng(config.selection_seed)
    shuffled = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    selected = []
    for event in shuffled:
        others = [x for x in selected if x["video_id"] == event["video_id"]]
        if len(others) >= per_video:
            continue
        a, b = event["clip_start_frame"], event["clip_end_frame"]
        overlap = False
        for other in others:
            c, d = other["clip_start_frame"], other["clip_end_frame"]
            intersection = max(0, min(b, d) - max(a, c) + 1)
            union = b - a + 1 + d - c + 1 - intersection
            if intersection / union > config.max_temporal_iou:
                overlap = True
                break
        if not overlap:
            selected.append(event)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda e: (e["video_id"], e["core_start_frame"]))
