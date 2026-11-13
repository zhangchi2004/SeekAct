"""Dense, mask-aware camera and virtual-gripper supervision.

All hand differences are computed in a stationary clip-start camera frame.
Action i describes state i -> i+1; there is deliberately no fabricated last action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import validate_camera_poses
from .vitra_motion import qwen_virtual_gripper, smooth_rotations, smooth_vectors


@dataclass(frozen=True)
class ActionConfig:
    sampling_hz: float = 10.0
    confidence_threshold: float = 0.25
    min_primary_valid_fraction: float = 0.8
    max_hand_speed_mps: float = 4.0
    max_hand_angular_speed_dps: float = 1500.0
    max_camera_speed_mps: float = 4.0
    max_camera_angular_speed_dps: float = 400.0
    gripper_closed_m: float = 0.01
    gripper_open_m: float = 0.10


def rotation_6d(r: np.ndarray) -> np.ndarray:
    """Columns, not row-major flatten of R[:,:2]."""
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def pose_9d(t: np.ndarray) -> np.ndarray:
    return np.concatenate([t[..., :3, 3], rotation_6d(t[..., :3, :3])], axis=-1)


def joint_names(hand: str) -> list[str]:
    return [f"{hand}Hand"] + [f"{hand}{finger}{joint}" for finger in
        ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
        for joint in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")]


def load_egodex(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        camera = validate_camera_poses(np.asarray(h["transforms/camera"], dtype=float))
        hands = {}
        for side in ("left", "right"):
            names = joint_names(side)
            if not all(f"transforms/{n}" in h for n in names):
                continue
            joints = np.stack([h[f"transforms/{n}"][:, :3, 3] for n in names], axis=1)
            # Unknown confidence is not treated as a reliable observation.
            confidence = np.stack([h[f"confidences/{n}"][:] if f"confidences/{n}" in h
                else np.zeros(len(camera)) for n in names], axis=1)
            if len(joints) != len(camera) or len(confidence) != len(camera):
                raise ValueError("camera/hand frame counts differ")
            hands[side] = {"joints_world": joints, "confidence": confidence}
        return {"camera_to_world": camera, "hands": hands,
                "intrinsics": np.asarray(h["camera/intrinsic"], dtype=float)}


def make_actions(data: dict, start: int, end: int, fps: float,
                 config: ActionConfig | None = None) -> tuple[dict, dict]:
    cfg = config or ActionConfig()
    if not (0 <= start < end < len(data["camera_to_world"])):
        raise ValueError("invalid inclusive clip bounds")
    if not np.isclose(fps / cfg.sampling_hz, round(fps / cfg.sampling_hz), atol=1e-5):
        raise ValueError("sampling rate must exactly divide source FPS")
    frames = np.unique(np.r_[np.arange(start, end + 1, round(fps / cfg.sampling_hz)), end]).astype(int)
    camera = data["camera_to_world"][frames]
    c0_inv = np.linalg.inv(camera[0])
    c0 = c0_inv @ camera
    dt = np.diff(frames) / fps
    # Body-frame camera delta: current camera -> next camera.
    camera_delta = np.linalg.inv(camera[:-1]) @ camera[1:]
    camera_speed = np.linalg.norm(camera_delta[:, :3, 3], axis=-1) / dt
    camera_spin = np.rad2deg(Rotation.from_matrix(camera_delta[:, :3, :3]).magnitude()) / dt
    camera_valid = (camera_speed <= cfg.max_camera_speed_mps) & (camera_spin <= cfg.max_camera_angular_speed_dps)
    arrays = {"source_frame": frames, "timestamp_s": frames / fps,
        "clip_timestamp_s": (frames - start) / fps, "action_dt_s": dt,
        "camera_to_world": camera, "camera_pose_c0": c0,
        "camera_pose_9d": pose_9d(c0), "camera_action_9d": pose_9d(camera_delta),
        "camera_action_valid": camera_valid, "camera_target_9d": pose_9d(c0[-1]),
        "intrinsics_source": data["intrinsics"]}
    quality = {"camera_valid_fraction": float(camera_valid.mean()), "hands": {},
               "config": asdict(cfg)}
    scores = {}
    # Smooth at native FPS within observed contiguous runs, then sample.
    source_camera = data["camera_to_world"][start:end+1]
    for side in ("left", "right"):
        n = end - start + 1
        if side not in data["hands"]:
            p = np.full((n, 3), np.nan)
            r = np.full((n, 3, 3), np.nan)
            width = np.full(n, np.nan)
            conf = np.zeros(n)
            valid = np.zeros(n, dtype=bool)
        else:
            hand = data["hands"][side]
            world = hand["joints_world"][start:end+1]
            joints = world @ c0_inv[:3, :3].T + c0_inv[:3, 3]
            p, r, width = qwen_virtual_gripper(joints, side)
            conf = np.min(hand["confidence"][start:end+1, [0, 4, 8, 12]], axis=1)
            valid = (conf >= cfg.confidence_threshold) & np.isfinite(p).all(1)
            valid &= np.isfinite(r).all((1, 2)) & (width >= .005) & (width <= .20)
            p = smooth_vectors(p, valid, 9, 2)
            width = smooth_vectors(width[:, None], valid, 9, 2)[:, 0]
            r = smooth_rotations(r, valid, 1.5)
            valid &= (width >= .005) & (width <= .20)
        pick = frames - start
        ps, rs, ws, vs = p[pick], r[pick], width[pick], valid[pick]
        # A target step may not bridge even a single unobserved native frame.
        av = np.array([valid[a:b+1].all() for a, b in zip(pick[:-1], pick[1:])])
        dr = np.full((len(dt), 3), np.nan)
        dr6 = np.full((len(dt), 6), np.nan)
        good = np.flatnonzero(av)
        if len(good):
            delta_r = rs[good+1] @ np.swapaxes(rs[good], 1, 2)
            dr[good] = Rotation.from_matrix(delta_r).as_rotvec()
            dr6[good] = rotation_6d(delta_r)
        dp = np.diff(ps, axis=0)
        speed = np.linalg.norm(dp, axis=1) / dt
        spin = np.rad2deg(np.linalg.norm(dr, axis=1)) / dt
        av &= (speed <= cfg.max_hand_speed_mps) & (spin <= cfg.max_hand_angular_speed_dps)
        opening = np.clip((ws - cfg.gripper_closed_m) / (cfg.gripper_open_m - cfg.gripper_closed_m), 0, 1)
        # Match ego2robot RetargetedTrajectory.actions: current-state gripper channel.
        a7 = np.c_[dp, dr, opening[:-1]]
        a10 = np.c_[dp, dr6, np.diff(ws)]
        a7[~av] = np.nan
        a10[~av] = np.nan
        ps = ps.copy(); rs = rs.copy(); ws = ws.copy()
        ps[~vs] = np.nan; rs[~vs] = np.nan; ws[~vs] = np.nan
        arrays.update({f"{side}_position_c0_m": ps, f"{side}_rotation_c0": rs,
            f"{side}_gripper_width_m": ws, f"{side}_state_valid": vs,
            f"{side}_confidence": conf[pick], f"{side}_action_7d": a7,
            f"{side}_action_10d": a10, f"{side}_action_valid": av})
        fraction = float(av.mean())
        path = float(np.linalg.norm(dp[av], axis=1).sum())
        quality["hands"][side] = {"valid_action_fraction": fraction, "path_length_m": path,
            "valid_state_fraction": float(vs.mean())}
        if fraction >= cfg.min_primary_valid_fraction:
            scores[side] = path + float(np.abs(np.diff(width[pick]))[av].sum())
    primary = max(scores, key=scores.get) if scores else None
    quality.update(primary_hand=primary, passed=primary is not None and camera_valid.all().item())
    if primary:
        arrays["manipulation_action_7d"] = arrays[f"{primary}_action_7d"]
        arrays["manipulation_action_valid"] = arrays[f"{primary}_action_valid"]
    # There is no all-zero placeholder silently counted as valid hand motion.
    arrays = {key: value.astype(np.float32) if value.dtype.kind == "f" else value for key, value in arrays.items()}
    return arrays, quality
