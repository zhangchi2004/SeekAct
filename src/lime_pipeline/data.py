from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from PIL import Image

from .geometry import MiningConfig, mine_motion_intervals, validate_camera_poses
from .schema import SCHEMA_VERSION, seconds_to_timestamp
from .visual_change import VisualChangeConfig, measure_visual_change


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def inspect_video(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frame_count <= 1:
        raise ValueError(f"invalid video metadata for {path}: fps={fps}, frames={frame_count}")
    return VideoMetadata(
        path=str(path.resolve()),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
    )


def load_egodex_camera_poses(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        if "transforms/camera" not in handle:
            raise ValueError(f"missing transforms/camera in {path}")
        poses = np.asarray(handle["transforms/camera"], dtype=np.float64)
        intrinsics = np.asarray(handle["camera/intrinsic"], dtype=np.float64) if "camera/intrinsic" in handle else None
    return validate_camera_poses(poses), intrinsics


def read_video_frames(path: Path, frame_indices: Iterable[int]) -> dict[int, Image.Image]:
    requested = sorted(set(int(x) for x in frame_indices))
    if not requested:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    frames: dict[int, Image.Image] = {}
    for index in requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"could not decode frame {index} from {path}")
        frames[index] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def task_hint_from_path(video_path: Path) -> str:
    return video_path.parent.name.replace("_", " ").strip()


def mine_video(
    video_path: Path,
    pose_path: Path | None = None,
    max_events: int = 5,
    source_dataset: str = "EgoDex",
    config: MiningConfig | None = None,
    visual_config: VisualChangeConfig | None = None,
) -> tuple[list[dict], dict]:
    video_path = video_path.resolve()
    pose_path = (pose_path or video_path.with_suffix(".hdf5")).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    config = config or MiningConfig()
    visual_config = visual_config or VisualChangeConfig()
    metadata = inspect_video(video_path)
    poses, intrinsics = load_egodex_camera_poses(pose_path)
    usable_count = min(metadata.frame_count, len(poses))
    if usable_count < 2:
        raise ValueError(f"no aligned frames for {video_path}")
    poses = poses[:usable_count]
    # Oversample pose-valid, temporally non-overlapping candidates before the
    # endpoint-image gate. Otherwise a visually weak top pose candidate could
    # hide a slightly lower-scoring but genuinely different view.
    candidate_limit = max(24, max_events * 8)
    intervals, diagnostics = mine_motion_intervals(
        poses,
        metadata.fps,
        max_events=candidate_limit,
        config=config,
    )
    frame_indices = [index for interval in intervals for index in (interval["start_frame"], interval["goal_frame"])]
    frames = read_video_frames(video_path, frame_indices)
    accepted_intervals: list[dict] = []
    for interval in intervals:
        visual_change = measure_visual_change(
            frames[interval["start_frame"]],
            frames[interval["goal_frame"]],
            config=visual_config,
        )
        interval["visual_change"] = visual_change
        if not visual_change["passes"]:
            continue
        interval["selection_score"] = round(
            float(interval["motion_score"] + 2.0 * visual_change["view_change_score"]),
            6,
        )
        accepted_intervals.append(interval)
    accepted_intervals.sort(key=lambda item: item["selection_score"], reverse=True)
    intervals = accepted_intervals[:max_events]
    intervals.sort(key=lambda item: item["start_frame"])
    geometric_selected_count = diagnostics["selected_count"]
    diagnostics["geometric_selected_before_visual_filter"] = geometric_selected_count
    diagnostics["visual_filter"] = {
        "profile": "strict_v2_broad_endpoint_change",
        "evaluated_count": geometric_selected_count,
        "passed_count": len(accepted_intervals),
        "thresholds": asdict(visual_config),
    }
    diagnostics["selected_count"] = len(intervals)
    task_hint = task_hint_from_path(video_path)
    stem = f"{video_path.parent.name}__{video_path.stem}"
    events: list[dict] = []
    for ordinal, interval in enumerate(intervals, start=1):
        start_frame = interval["start_frame"]
        goal_frame = interval["goal_frame"]
        start_seconds = start_frame / metadata.fps
        goal_seconds = goal_frame / metadata.fps
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": f"{stem}__event_{ordinal:02d}",
                "source": {
                    "dataset": source_dataset,
                    "video_path": str(video_path),
                    "pose_path": str(pose_path),
                    "task_hint": task_hint,
                    "video": asdict(metadata),
                    "pose_frame_count": int(len(poses)),
                    "intrinsics": np.round(intrinsics, 6).tolist() if intrinsics is not None else None,
                },
                "time": {
                    "start_frame": start_frame,
                    "goal_frame": goal_frame,
                    "peak_frame": interval["peak_frame"],
                    "start_seconds": round(start_seconds, 6),
                    "goal_seconds": round(goal_seconds, 6),
                    "duration_seconds": round(goal_seconds - start_seconds, 6),
                    "start_label": seconds_to_timestamp(start_seconds),
                    "goal_label": seconds_to_timestamp(goal_seconds),
                    "timestamp_basis": "frame_index / container_fps",
                    "frame_interval": "inclusive",
                },
                "selection": {
                    "motion_score": interval["motion_score"],
                    "peak_activity": interval["peak_activity"],
                    "selection_score": interval["selection_score"],
                    "threshold_profile": (
                        "strict_v2_translation_required"
                        if config.pose_gate_mode == "translation_required"
                        else "strict_v2"
                    ),
                    "pose_gate_mode": config.pose_gate_mode,
                    "min_translation_m": config.min_translation_m,
                    "min_rotation_deg": config.min_rotation_deg,
                    "endpoint_pose_pass": True,
                    "visual_change_pass": True,
                },
                "camera_motion": interval["camera_motion"],
                "visual_change": interval["visual_change"],
                "label": None,
                "provenance": {
                    "geometry_source": "EgoDex transforms/camera",
                    "label_source": None,
                    "used_task_hint": False,
                },
            }
        )
    report = {
        "video": asdict(metadata),
        "pose_path": str(pose_path),
        "alignment": {
            "video_frame_count": metadata.frame_count,
            "pose_frame_count": len(poses),
            "usable_frame_count": usable_count,
            "exact_count_match": metadata.frame_count == len(poses),
        },
        "mining_diagnostics": diagnostics,
    }
    return events, report
