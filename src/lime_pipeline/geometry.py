from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class MiningConfig:
    velocity_window_seconds: float = 0.25
    smooth_seconds: float = 0.35
    translation_speed_mps: float = 0.05
    angular_speed_dps: float = 12.0
    merge_gap_seconds: float = 0.45
    pad_seconds: float = 0.20
    min_duration_seconds: float = 0.80
    max_duration_seconds: float = 6.0
    min_translation_m: float = 0.08
    min_rotation_deg: float = 15.0
    pose_gate_mode: str = "translation_or_rotation"
    max_translation_m: float = 1.5
    max_rotation_deg: float = 170.0
    nms_iou: float = 0.35


POSE_GATE_MODES = (
    "translation_or_rotation",
    "translation_required",
    "translation_and_rotation",
)


def passes_endpoint_pose_gate(distance_m: float, angle_deg: float, config: MiningConfig) -> bool:
    if config.pose_gate_mode == "translation_or_rotation":
        return distance_m >= config.min_translation_m or angle_deg >= config.min_rotation_deg
    if config.pose_gate_mode == "translation_required":
        return distance_m >= config.min_translation_m
    if config.pose_gate_mode == "translation_and_rotation":
        return distance_m >= config.min_translation_m and angle_deg >= config.min_rotation_deg
    raise ValueError(
        f"unknown pose_gate_mode={config.pose_gate_mode!r}; expected one of {POSE_GATE_MODES}"
    )


def pose_gate_description(config: MiningConfig) -> str:
    if config.pose_gate_mode == "translation_or_rotation":
        return "translation_distance_m >= min_translation_m OR rotation_angle_deg >= min_rotation_deg"
    if config.pose_gate_mode == "translation_required":
        return "translation_distance_m >= min_translation_m (rotation is measured but not a substitute)"
    if config.pose_gate_mode == "translation_and_rotation":
        return "translation_distance_m >= min_translation_m AND rotation_angle_deg >= min_rotation_deg"
    raise ValueError(
        f"unknown pose_gate_mode={config.pose_gate_mode!r}; expected one of {POSE_GATE_MODES}"
    )


def rotation_angle_degrees(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def validate_camera_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"camera poses must have shape [N,4,4], got {poses.shape}")
    if len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("camera poses are empty or contain non-finite values")
    if not np.allclose(poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=2e-3):
        raise ValueError("camera transforms are not homogeneous matrices")
    determinants = np.linalg.det(poses[:, :3, :3])
    if np.nanmedian(np.abs(determinants - 1.0)) > 2e-2:
        raise ValueError("camera rotations are not valid SO(3) matrices")
    return poses


def _rotation_direction(relative_rotation: np.ndarray, angle_deg: float) -> tuple[str, float, float]:
    if angle_deg < 3.0:
        return "steady", 0.0, 0.0
    forward = relative_rotation @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(forward[0], forward[2])))
    pitch = float(np.degrees(np.arctan2(forward[1], np.hypot(forward[0], forward[2]))))
    if abs(yaw) >= abs(pitch) and abs(yaw) >= 3.0:
        return ("pan right" if yaw > 0 else "pan left"), yaw, pitch
    if abs(pitch) >= 3.0:
        return ("tilt down" if pitch > 0 else "tilt up"), yaw, pitch
    return "roll/reorient", yaw, pitch


def _translation_direction(vector: np.ndarray) -> str:
    distance = float(np.linalg.norm(vector))
    if distance < 0.015:
        return "stationary"
    labels = (("right", "left"), ("down", "up"), ("forward", "backward"))
    order = np.argsort(np.abs(vector))[::-1]
    parts: list[str] = []
    dominant = abs(float(vector[order[0]]))
    for index in order[:2]:
        value = float(vector[index])
        if abs(value) < max(0.01, dominant * 0.55):
            continue
        parts.append(labels[index][0 if value > 0 else 1])
    return "+".join(parts) if parts else "small translation"


def relative_motion(poses: np.ndarray, start: int, goal: int) -> dict[str, Any]:
    """Return goal camera pose in the start-camera coordinate frame.

    EgoDex stores camera-to-world transforms. Therefore inv(P_start) @ P_goal
    is the goal camera pose expressed in the start camera frame. EgoDex camera
    axes are +x right, +y down, +z forward.
    """
    if not (0 <= start < goal < len(poses)):
        raise ValueError(f"invalid frame interval [{start}, {goal}] for {len(poses)} poses")
    relative = np.linalg.inv(poses[start]) @ poses[goal]
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    angle = float(rotation_angle_degrees(rotation))
    rotation_direction, yaw_deg, pitch_deg = _rotation_direction(rotation, angle)

    positions = poses[start : goal + 1, :3, 3]
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    step_relative_rotation = np.einsum(
        "nij,njk->nik",
        np.transpose(poses[start:goal, :3, :3], (0, 2, 1)),
        poses[start + 1 : goal + 1, :3, :3],
    )
    cumulative_angle = float(rotation_angle_degrees(step_relative_rotation).sum())
    distance = float(np.linalg.norm(translation))
    if distance >= 0.025 and angle >= 7.0:
        geometric_type = "translation+rotation"
    elif distance >= 0.025:
        geometric_type = "translation"
    elif angle >= 7.0:
        geometric_type = "rotation"
    else:
        geometric_type = "small_motion"

    rotation_6d = np.concatenate([rotation[:, 0], rotation[:, 1]])
    return {
        "pose_convention": "inv(camera_to_world_start) @ camera_to_world_goal",
        "camera_axes": "+x right, +y down, +z forward",
        "relative_transform_4x4": np.round(relative, 7).tolist(),
        "translation_start_camera_m": np.round(translation, 6).tolist(),
        "translation_distance_m": round(distance, 6),
        "translation_direction": _translation_direction(translation),
        "rotation_matrix": np.round(rotation, 7).tolist(),
        "rotation_6d_first_two_columns": np.round(rotation_6d, 7).tolist(),
        "rotation_angle_deg": round(angle, 4),
        "rotation_direction": rotation_direction,
        "yaw_change_deg": round(yaw_deg, 4),
        "pitch_change_deg": round(pitch_deg, 4),
        "path_length_m": round(path_length, 6),
        "cumulative_rotation_deg": round(cumulative_angle, 4),
        "geometric_motion_type": geometric_type,
    }


def _merge_runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > max_gap + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union if union else 0.0


def mine_motion_intervals(
    poses: np.ndarray,
    fps: float,
    max_events: int,
    config: MiningConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or MiningConfig()
    poses = validate_camera_poses(poses)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid FPS: {fps}")

    count = len(poses)
    half_window = max(1, int(round(config.velocity_window_seconds * fps * 0.5)))
    before = np.maximum(np.arange(count) - half_window, 0)
    after = np.minimum(np.arange(count) + half_window, count - 1)
    dt = np.maximum((after - before) / fps, 1.0 / fps)
    position = poses[:, :3, 3]
    translation_speed = np.linalg.norm(position[after] - position[before], axis=1) / dt
    relative_rotation = np.einsum(
        "nij,njk->nik",
        np.transpose(poses[before, :3, :3], (0, 2, 1)),
        poses[after, :3, :3],
    )
    angular_speed = rotation_angle_degrees(relative_rotation) / dt
    activity = np.maximum(
        translation_speed / config.translation_speed_mps,
        angular_speed / config.angular_speed_dps,
    )
    smooth_frames = max(1, int(round(config.smooth_seconds * fps)))
    activity = uniform_filter1d(activity, size=smooth_frames, mode="nearest")

    runs = _merge_runs(activity >= 1.0, max_gap=int(round(config.merge_gap_seconds * fps)))
    padding = int(round(config.pad_seconds * fps))
    min_frames = max(2, int(round(config.min_duration_seconds * fps)))
    max_frames = max(min_frames, int(round(config.max_duration_seconds * fps)))
    proposals: list[dict[str, Any]] = []
    for raw_start, raw_goal in runs:
        start = max(0, raw_start - padding)
        goal = min(count - 1, raw_goal + padding)
        if goal - start + 1 < min_frames:
            continue
        if goal - start + 1 > max_frames:
            peak = int(start + np.argmax(activity[start : goal + 1]))
            start = max(0, peak - max_frames // 2)
            goal = min(count - 1, start + max_frames - 1)
            start = max(0, goal - max_frames + 1)
        motion = relative_motion(poses, start, goal)
        distance = motion["translation_distance_m"]
        angle = motion["rotation_angle_deg"]
        path_length = motion["path_length_m"]
        cumulative_angle = motion["cumulative_rotation_deg"]
        if not passes_endpoint_pose_gate(distance, angle, config):
            # Strict-v2 uses endpoint pose only. A long, jittery path that
            # returns close to its start is not an active-perception target.
            continue
        if distance > config.max_translation_m or angle > config.max_rotation_deg:
            continue
        peak_activity = float(np.max(activity[start : goal + 1]))
        score = (
            distance / config.min_translation_m
            + angle / config.min_rotation_deg
            + min(path_length, 0.5) / 0.15
            + min(cumulative_angle, 180.0) / 60.0
            + min(peak_activity, 6.0) * 0.2
        )
        proposals.append(
            {
                "start_frame": int(start),
                "goal_frame": int(goal),
                "peak_frame": int(start + np.argmax(activity[start : goal + 1])),
                "motion_score": round(float(score), 5),
                "peak_activity": round(peak_activity, 5),
                "camera_motion": motion,
            }
        )

    # A fixed-window fallback also catches smooth, deliberate motions whose speed
    # never crosses the hysteresis threshold.
    if len(proposals) < max_events:
        stride = max(1, int(round(0.5 * fps)))
        for seconds in (2.0, 4.0, 6.0):
            window = min(count - 1, int(round(seconds * fps)))
            if window < min_frames:
                continue
            for start in range(0, count - window, stride):
                goal = start + window
                motion = relative_motion(poses, start, goal)
                distance = motion["translation_distance_m"]
                angle = motion["rotation_angle_deg"]
                if not passes_endpoint_pose_gate(distance, angle, config):
                    continue
                if distance > config.max_translation_m or angle > config.max_rotation_deg:
                    continue
                score = (
                    distance / config.min_translation_m
                    + angle / config.min_rotation_deg
                    + motion["path_length_m"] / 0.2
                )
                proposals.append(
                    {
                        "start_frame": int(start),
                        "goal_frame": int(goal),
                        "peak_frame": int(start + np.argmax(activity[start : goal + 1])),
                        "motion_score": round(float(score), 5),
                        "peak_activity": round(float(np.max(activity[start : goal + 1])), 5),
                        "camera_motion": motion,
                    }
                )

    selected: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: item["motion_score"], reverse=True):
        interval = (proposal["start_frame"], proposal["goal_frame"])
        if any(_interval_iou(interval, (x["start_frame"], x["goal_frame"])) > config.nms_iou for x in selected):
            continue
        selected.append(proposal)
        if len(selected) >= max_events:
            break
    selected.sort(key=lambda item: item["start_frame"])

    diagnostics = {
        "frame_count": count,
        "fps": fps,
        "activity_thresholds": {
            "translation_speed_mps": config.translation_speed_mps,
            "angular_speed_dps": config.angular_speed_dps,
        },
        "endpoint_pose_thresholds": {
            "mode": config.pose_gate_mode,
            "acceptance": pose_gate_description(config),
            "min_translation_m": config.min_translation_m,
            "min_rotation_deg": config.min_rotation_deg,
            "path_length_or_cumulative_rotation_fallback": False,
        },
        "translation_speed_mps_median": round(float(np.median(translation_speed)), 6),
        "translation_speed_mps_p95": round(float(np.percentile(translation_speed, 95)), 6),
        "angular_speed_dps_median": round(float(np.median(angular_speed)), 5),
        "angular_speed_dps_p95": round(float(np.percentile(angular_speed, 95)), 5),
        "raw_active_run_count": len(runs),
        "proposal_count": len(proposals),
        "selected_count": len(selected),
    }
    return selected, diagnostics


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    matrix, _ = cv2.Rodrigues(vector * np.radians(angle_deg))
    return matrix
