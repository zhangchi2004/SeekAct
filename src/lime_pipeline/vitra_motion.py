from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .geometry import relative_motion, validate_camera_poses


@dataclass(frozen=True)
class VitraQualityConfig:
    min_valid_fraction: float = 0.60
    min_projected_in_frame_fraction: float = 0.55
    max_hand_step_m: float = 0.20
    max_hand_speed_mps: float = 4.0
    max_hand_rotation_step_deg: float = 45.0
    max_hand_angular_speed_dps: float = 1500.0
    max_sudden_change_fraction: float = 0.15
    smooth_window: int = 9
    smooth_polyorder: int = 2
    quaternion_sigma_frames: float = 1.5
    trajectory_sample_count: int = 16


PALM_JOINT_INDICES = np.asarray([0, 2, 5, 9, 13, 17], dtype=np.int64)
WRIST_INDEX = 0
THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
MIDDLE_TIP_INDEX = 12


def validate_world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"VITRA extrinsics must have shape [T,4,4], got {extrinsics.shape}")
    if len(extrinsics) < 2 or not np.isfinite(extrinsics).all():
        raise ValueError("VITRA extrinsics are empty or non-finite")
    if not np.allclose(extrinsics[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=2e-3):
        raise ValueError("VITRA extrinsics are not homogeneous transforms")
    return extrinsics


def camera_motion_from_world_to_camera(extrinsics: np.ndarray) -> dict[str, Any]:
    """Compute metric camera motion from VITRA world-to-camera extrinsics."""
    world_to_camera = validate_world_to_camera(extrinsics)
    camera_to_world = validate_camera_poses(np.linalg.inv(world_to_camera))
    result = relative_motion(camera_to_world, 0, len(camera_to_world) - 1)
    result["pose_convention"] = "inv(camera_to_world_start) @ camera_to_world_goal; VITRA source is world_to_camera"
    result["measurement_source"] = "VITRA reconstructed world-to-camera extrinsics"
    result["metric_translation_available"] = True
    result["trajectory_frame_count"] = int(len(camera_to_world))
    return result


def transform_world_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,tkj->tki", transform[:3, :3], points) + transform[:3, 3]


def qwen_virtual_gripper(
    joints: np.ndarray,
    handedness: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retarget 21 hand joints to Qwen-RobotManip's virtual gripper pose.

    The returned tuple is (position, orientation, gripper width). Orientations
    have approach, normal, and grasp axes as matrix columns.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ValueError(f"hand joints must have shape [T,21,3], got {joints.shape}")
    if handedness not in {"left", "right"}:
        raise ValueError(f"invalid handedness: {handedness}")
    virtual_finger = 0.7 * joints[:, INDEX_TIP_INDEX] + 0.3 * joints[:, MIDDLE_TIP_INDEX]
    thumb = joints[:, THUMB_TIP_INDEX]
    wrist = joints[:, WRIST_INDEX]
    jaw = thumb - virtual_finger
    width = np.linalg.norm(jaw, axis=1)
    sign = 1.0 if handedness == "right" else -1.0
    z_axis = sign * jaw / np.maximum(width[:, None], 1e-8)
    fingertip_direction = virtual_finger - wrist
    y_raw = np.cross(z_axis, fingertip_direction)
    y_norm = np.linalg.norm(y_raw, axis=1)
    y_axis = y_raw / np.maximum(y_norm[:, None], 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=1)[:, None], 1e-8)
    orientation = np.stack([x_axis, y_axis, z_axis], axis=-1)
    position = 0.5 * (thumb + virtual_finger)
    invalid = (width < 1e-6) | (y_norm < 1e-6) | ~np.isfinite(orientation).all(axis=(1, 2))
    position[invalid] = np.nan
    orientation[invalid] = np.nan
    width[invalid] = np.nan
    return position, orientation, width


def _contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, splits) if len(run)]


def _odd_window(length: int, requested: int, polyorder: int) -> int:
    window = min(length if length % 2 else length - 1, requested if requested % 2 else requested - 1)
    return window if window > polyorder else 0


def smooth_vectors(
    values: np.ndarray,
    valid: np.ndarray,
    window: int,
    polyorder: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values.copy()
    for run in _contiguous_true_runs(valid):
        run_window = _odd_window(len(run), window, polyorder)
        if run_window:
            result[run] = savgol_filter(values[run], run_window, polyorder, axis=0, mode="interp")
    return result


def smooth_rotations(
    rotations: np.ndarray,
    valid: np.ndarray,
    sigma_frames: float,
) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float64)
    result = rotations.copy()
    radius = max(1, int(round(3.0 * sigma_frames)))
    for run in _contiguous_true_runs(valid):
        quaternions = Rotation.from_matrix(rotations[run]).as_quat()
        smoothed = np.empty_like(quaternions)
        for local_index in range(len(run)):
            lo = max(0, local_index - radius)
            hi = min(len(run), local_index + radius + 1)
            offsets = np.arange(lo, hi) - local_index
            weights = np.exp(-0.5 * (offsets / max(sigma_frames, 1e-6)) ** 2)
            neighborhood = quaternions[lo:hi].copy()
            reference = quaternions[local_index]
            neighborhood[np.einsum("ij,j->i", neighborhood, reference) < 0] *= -1.0
            average = np.sum(neighborhood * weights[:, None], axis=0)
            average /= max(np.linalg.norm(average), 1e-8)
            smoothed[local_index] = average
        result[run] = Rotation.from_quat(smoothed).as_matrix()
    return result


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points_camera = np.asarray(points_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    homogeneous = np.einsum("ij,tkj->tki", intrinsics, points_camera)
    return homogeneous[..., :2] / np.maximum(homogeneous[..., 2:3], 1e-8)


def _direction(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=float)
    if np.linalg.norm(vector) < 1e-4:
        return "stationary"
    axes = (("right", "left"), ("down", "up"), ("forward", "backward"))
    index = int(np.argmax(np.abs(vector)))
    return axes[index][0 if vector[index] > 0 else 1]


def extract_qwen_hand_motion(
    hand_data: dict[str, Any],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    handedness: str,
    fps: float,
    config: VitraQualityConfig | None = None,
) -> dict[str, Any]:
    config = config or VitraQualityConfig()
    extrinsics = validate_world_to_camera(extrinsics)
    joints_world = np.asarray(hand_data["joints_worldspace"], dtype=np.float64)
    kept = np.asarray(hand_data["kept_frames"], dtype=bool)
    count = min(len(extrinsics), len(joints_world), len(kept))
    extrinsics = extrinsics[:count]
    joints_world = joints_world[:count]
    kept = kept[:count]
    valid = kept & np.isfinite(joints_world).all(axis=(1, 2))

    # Qwen-RobotManip aligns deltas in the current/reference camera. Use the
    # first episode camera as a fixed anchor to disentangle hand and head motion.
    joints_start_camera = transform_world_points(extrinsics[0], joints_world)
    position, orientation, width = qwen_virtual_gripper(joints_start_camera, handedness)
    valid &= np.isfinite(position).all(axis=1) & np.isfinite(width)
    if valid.sum() < 2:
        return {
            "handedness": handedness,
            "available": False,
            "valid_fraction": round(float(valid.mean()), 6),
            "quality_pass": False,
            "quality_failures": ["fewer_than_two_valid_frames"],
        }

    position_smoothed = smooth_vectors(position, valid, config.smooth_window, config.smooth_polyorder)
    width_smoothed = smooth_vectors(width[:, None], valid, config.smooth_window, config.smooth_polyorder)[:, 0]
    orientation_smoothed = smooth_rotations(orientation, valid, config.quaternion_sigma_frames)
    valid_indices = np.flatnonzero(valid)
    start, goal = int(valid_indices[0]), int(valid_indices[-1])
    adjacent = valid[:-1] & valid[1:]
    steps = np.linalg.norm(np.diff(position_smoothed, axis=0), axis=1)
    valid_steps = steps[adjacent]
    path_length = float(valid_steps.sum())
    delta = position_smoothed[goal] - position_smoothed[start]
    distance = float(np.linalg.norm(delta))
    duration = max((goal - start) / fps, 1.0 / fps)
    speed = valid_steps * fps

    rotation_steps = np.full(max(count - 1, 0), np.nan, dtype=float)
    adjacent_indices = np.flatnonzero(adjacent)
    if len(adjacent_indices):
        rotation_delta = np.einsum(
            "nij,njk->nik",
            orientation_smoothed[adjacent_indices + 1],
            np.transpose(orientation_smoothed[adjacent_indices], (0, 2, 1)),
        )
        rotation_steps[adjacent_indices] = np.degrees(Rotation.from_matrix(rotation_delta).magnitude())
    valid_rotation_steps = rotation_steps[np.isfinite(rotation_steps)]
    angular_speed = valid_rotation_steps * fps

    relative_rotation = orientation_smoothed[goal] @ orientation_smoothed[start].T
    rotation_angle = float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))
    rotation_6d = np.concatenate([relative_rotation[:, 0], relative_rotation[:, 1]])

    # Projection uses each frame's actual camera, so the overlay stays aligned
    # even as the head-mounted camera moves.
    joints_current_camera = np.einsum("tij,tkj->tki", extrinsics[:, :3, :3], joints_world) + extrinsics[:, None, :3, 3]
    palm_current_camera = joints_current_camera[:, PALM_JOINT_INDICES].mean(axis=1)[:, None, :]
    palm_uv = project_points(palm_current_camera, intrinsics)[:, 0]
    width_px = max(float(intrinsics[0, 2]) * 2.0, 1.0)
    height_px = max(float(intrinsics[1, 2]) * 2.0, 1.0)
    in_frame = (
        valid
        & (palm_current_camera[:, 0, 2] > 0)
        & (palm_uv[:, 0] >= 0)
        & (palm_uv[:, 0] < width_px)
        & (palm_uv[:, 1] >= 0)
        & (palm_uv[:, 1] < height_px)
    )
    in_frame_fraction = float(in_frame.sum() / max(valid.sum(), 1))

    raw_residual = np.linalg.norm(position - position_smoothed, axis=1)
    acceleration = np.zeros(count, dtype=float)
    jerk = np.zeros(count, dtype=float)
    if count >= 3:
        acceleration[1:-1] = np.linalg.norm(np.diff(position, n=2, axis=0), axis=1)
    if count >= 4:
        jerk[1:-2] = np.linalg.norm(np.diff(position, n=3, axis=0), axis=1)
    sudden = valid & (raw_residual > 0.04) & ((acceleration > 0.05) | (jerk > 0.08))
    sudden_fraction = float(sudden.sum() / max(valid.sum(), 1))

    quality_checks = {
        "valid_fraction": float(valid.mean()) >= config.min_valid_fraction,
        "projected_in_frame_fraction": in_frame_fraction >= config.min_projected_in_frame_fraction,
        "max_step_m": (float(valid_steps.max()) if len(valid_steps) else float("inf")) <= config.max_hand_step_m,
        "max_speed_mps": (float(speed.max()) if len(speed) else float("inf")) <= config.max_hand_speed_mps,
        "max_rotation_step_deg": (
            float(valid_rotation_steps.max()) if len(valid_rotation_steps) else float("inf")
        ) <= config.max_hand_rotation_step_deg,
        "max_angular_speed_dps": (
            float(angular_speed.max()) if len(angular_speed) else float("inf")
        ) <= config.max_hand_angular_speed_dps,
        "sudden_change_fraction": sudden_fraction <= config.max_sudden_change_fraction,
    }
    sample_positions = np.linspace(0, len(valid_indices) - 1, min(config.trajectory_sample_count, len(valid_indices)))
    sample_indices = valid_indices[np.round(sample_positions).astype(int)]
    trajectory_samples = [
        {
            "episode_frame": int(index),
            "position_start_camera_m": np.round(position_smoothed[index], 6).tolist(),
            "palm_uv": np.round(palm_uv[index], 2).tolist(),
            "palm_uv_normalized": [
                round(float(palm_uv[index, 0] / width_px), 6),
                round(float(palm_uv[index, 1] / height_px), 6),
            ],
            "gripper_width_m": round(float(width_smoothed[index]), 6),
        }
        for index in sample_indices
    ]
    return {
        "handedness": handedness,
        "available": True,
        "representation": "Qwen-RobotManip virtual-finger gripper from VITRA 21-joint MANO reconstruction",
        "coordinate_frame": "episode start camera (+x right, +y down, +z forward)",
        "valid_frame_count": int(valid.sum()),
        "frame_count": int(count),
        "valid_fraction": round(float(valid.mean()), 6),
        "start_episode_frame": start,
        "goal_episode_frame": goal,
        "translation_start_camera_m": np.round(delta, 6).tolist(),
        "translation_distance_m": round(distance, 6),
        "translation_direction": _direction(delta),
        "path_length_m": round(path_length, 6),
        "mean_speed_mps": round(path_length / duration, 6),
        "max_speed_mps": round(float(speed.max()) if len(speed) else 0.0, 6),
        "max_rotation_step_deg": round(
            float(valid_rotation_steps.max()) if len(valid_rotation_steps) else 0.0, 6
        ),
        "max_angular_speed_dps": round(float(angular_speed.max()) if len(angular_speed) else 0.0, 6),
        "relative_rotation_matrix": np.round(relative_rotation, 7).tolist(),
        "rotation_6d_first_two_columns": np.round(rotation_6d, 7).tolist(),
        "rotation_angle_deg": round(rotation_angle, 4),
        "gripper_width_start_m": round(float(width_smoothed[start]), 6),
        "gripper_width_goal_m": round(float(width_smoothed[goal]), 6),
        "gripper_width_change_m": round(float(width_smoothed[goal] - width_smoothed[start]), 6),
        "projected_in_frame_fraction": round(in_frame_fraction, 6),
        "sudden_change_fraction": round(sudden_fraction, 6),
        "quality_checks": quality_checks,
        "quality_pass": all(quality_checks.values()),
        "quality_failures": [name for name, passed in quality_checks.items() if not passed],
        "trajectory_samples": trajectory_samples,
    }
