#!/usr/bin/env python3
"""Export full-video motion timelines and camera-selected context clips."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lime_pipeline.data import inspect_video, read_video_frames
from lime_pipeline.long_camera import LongCameraConfig, dense_timeline, mine_long_camera, select_diverse_candidates
from lime_pipeline.render import _ffmpeg_executable


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_fields(path: Path):
    data = json.loads(path.read_text())
    indexed = sorted((int(re.search(r"(\d+)\.[^.]+$", name).group(1)) - 1, row)
                     for name, row in data["images"].items())
    indices = np.array([x[0] for x in indexed])
    values = np.asarray([x[1] for x in indexed], dtype=float)
    w2c = np.repeat(np.eye(4)[None], len(indices), axis=0)
    w2c[:, :3, :3] = Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_matrix()
    w2c[:, :3, 3] = values[:, 4:7]
    return indices, np.linalg.inv(w2c)


def export_timeline(directory: Path, timeline: dict, fps: float, candidates: list, config: LongCameraConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "camera_full_timeline.npz", **timeline,
                        source_frame_indices=np.arange(len(timeline["valid"])), fps=fps)
    valid = timeline["valid"]
    quaternions = np.full((len(valid), 4), np.nan)
    quaternions[valid] = Rotation.from_matrix(timeline["camera_to_world"][valid, :3, :3]).as_quat()
    with (directory / "camera_full_timeline.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_frame_0based", "source_time_s", "pose_status_0missing_1estimated_2interpolated",
                         "c2w_tx_units", "c2w_ty_units", "c2w_tz_units", "c2w_qx", "c2w_qy", "c2w_qz", "c2w_qw",
                         "angular_speed_deg_s", "translation_speed_units_s"])
        for index in range(len(valid)):
            values = [*timeline["camera_to_world"][index, :3, 3], *quaternions[index],
                      timeline["angular_speed_deg_s"][index], timeline["translation_speed_units_s"][index]]
            writer.writerow([index, round(index / fps, 6), int(timeline["pose_status"][index]),
                             *[round(float(x), 7) if np.isfinite(x) else "" for x in values]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.3), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    times = np.arange(len(valid)) / fps
    axes[0].plot(times, timeline["angular_speed_deg_s"], linewidth=.7, color="#347ab8")
    axes[0].axhline(config.onset_speed_deg_s, color="#be6a20", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Angular speed (deg/s)")
    axes[0].set_ylim(0, min(200, max(50, np.nanpercentile(timeline["angular_speed_deg_s"], 99.5))))
    for event in candidates:
        if event["pose_gate_pass"]:
            axes[0].axvspan(event["core_start_s"], event["core_end_s"], color="#30ad70", alpha=.15)
    axes[1].plot(times, valid.astype(int), color="#30ad70", linewidth=.9)
    axes[1].set_ylabel("Pose valid")
    axes[1].set_xlabel("Time in complete source video (s)")
    axes[1].set_yticks([0, 1])
    fig.tight_layout()
    fig.savefig(directory / "motion_timeline.png", dpi=140)
    plt.close(fig)


def quality(event: dict, video: Path) -> dict:
    frames = read_video_frames(video, [event["core_start_frame"], event["core_end_frame"]])
    sharpness, exposures = [], []
    for image in frames.values():
        array = cv2.resize(np.asarray(image), (640, 360))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        exposures.append(float(np.mean((gray >= 8) & (gray <= 247))))
    return {"endpoint_sharpness_min": min(sharpness), "usable_exposure_min": min(exposures)}


def export_clip(event: dict, output: Path) -> dict:
    directory = output / "clips" / event["event_id"]
    directory.mkdir(parents=True, exist_ok=True)
    clean, review = directory / "clip.mp4", directory / "review.mp4"
    fps = event["fps"]
    count = event["clip_end_frame"] - event["clip_start_frame"] + 1
    ffmpeg = _ffmpeg_executable()
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
                    "-ss", f"{event['clip_start_frame'] / fps:.9f}", "-i", event["source_video"],
                    "-frames:v", str(count), "-an", "-vf", "scale=1280:-2,setsar=1", "-c:v", "libx264",
                    "-threads", "2", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-tag:v", "avc1", "-movflags", "+faststart", str(clean)], check=True)
    capture = cv2.VideoCapture(str(clean))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
        raise RuntimeError(f"Export frame count mismatch: {event['event_id']}")
    step = max(1, round(fps / 25))
    temp = directory / ".review.intermediate.mp4"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (960, 690))
    if not writer.isOpened():
        raise RuntimeError("Review writer failed")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            image = cv2.resize(frame, (960, 540))
            source_frame = event["clip_start_frame"] + index
            active = event["core_start_frame"] <= source_frame <= event["core_end_frame"]
            canvas = np.zeros((690, 960, 3), dtype=np.uint8)
            canvas[:540] = image
            color = (100, 220, 80) if active else (210, 185, 125)
            text = [f"{event['video_id']} | source {source_frame / fps:.3f}s | frame {source_frame} | {'MOTION CORE' if active else 'CONTEXT'}",
                    f"Net rotation {event['net_rotation_deg']:.1f} deg | view-axis change {event['view_axis_change_deg']:.1f} deg",
                    f"Core {event['core_start_s']:.3f}-{event['core_end_s']:.3f}s | {event['backend']} | scale unknown",
                    "Camera-motion candidate; observation gain has not been verified."]
            for j, line in enumerate(text):
                cv2.putText(canvas, line, (12, 568 + j * 28), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 1, cv2.LINE_AA)
            writer.write(canvas)
        index += 1
    capture.release()
    writer.release()
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(temp),
                    "-an", "-c:v", "libx264", "-threads", "2", "-crf", "20", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart", str(review)], check=True)
    temp.unlink()
    selected = np.unique([event["clip_start_frame"], event["core_start_frame"],
                          (event["core_start_frame"] + event["core_end_frame"]) // 2,
                          event["core_end_frame"], event["clip_end_frame"]])
    frames = read_video_frames(Path(event["source_video"]), selected)
    strips = []
    for frame_index, image in frames.items():
        panel = cv2.cvtColor(cv2.resize(np.asarray(image), (384, 216)), cv2.COLOR_RGB2BGR)
        cv2.rectangle(panel, (0, 0), (384, 26), (0, 0, 0), -1)
        cv2.putText(panel, f"{frame_index / fps:.3f}s / frame {frame_index}", (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        strips.append(panel)
    cv2.imwrite(str(directory / "frames.jpg"), np.hstack(strips))
    event["artifacts"] = {"clean_clip": str(clean), "review_clip": str(review),
                          "source_frame_count": count, "source_fps": fps, "review_fps": fps / step}
    write_json(directory / "event.json", event)
    return {"event_id": event["event_id"], "clean_frames": count, "decoded_clean_frames": index,
            "review_frames": (count + step - 1) // step, "codec": "h264", "pixel_format": "yuv420p"}


def gallery(output: Path, selected: list, summaries: list, config: LongCameraConfig) -> None:
    videos = "".join(f'<h3>{html.escape(x["video_id"])} — {x["duration_seconds"]:.1f}s, valid poses {x["valid_pose_fraction"]:.1%}</h3>'
                     f'<img src="timelines/{x["video_id"]}/motion_timeline.png">' for x in summaries)
    cards = []
    for e in selected:
        root = f'clips/{e["event_id"]}'
        cards.append(f'<article><h2>{html.escape(e["event_id"])}</h2><p>Motion core: {e["core_start_s"]:.3f}–{e["core_end_s"]:.3f}s; '
                     f'clip: {e["clip_start_s"]:.3f}–{e["clip_end_s"]:.3f}s. Net rotation {e["net_rotation_deg"]:.1f}°; '
                     f'view-axis change {e["view_axis_change_deg"]:.1f}°.</p><img loading="lazy" src="{root}/frames.jpg">'
                     f'<video controls preload="none" src="{root}/review.mp4"></video><p><a href="{root}/clip.mp4">Clean RGB clip</a> · '
                     f'<a href="{root}/event.json">Metadata</a></p></article>')
    document = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPIC long-video camera motion</title><style>body{{font:16px/1.6 system-ui;background:#111923;color:#e2e9f1;max-width:1150px;margin:auto;padding:24px}}a{{color:#8bd2ee}}img,video{{width:100%;border-radius:8px}}article{{padding:20px;background:#1e2a38;margin:24px 0;border-radius:12px}}h1,h2{{line-height:1.3}}</style>
<h1>从完整 EPIC 视频切出的相机运动片段</h1><p>{len(selected)} clips / {len(summaries)} complete source videos processed.
Camera motion proposes candidates; active perception / observation gain remains unverified. No VITRA episode boundaries are used.</p>
<p>净旋转 ≥{config.min_net_rotation_deg}°，视线方向变化 ≥{config.min_view_axis_change_deg}°；运动段 {config.min_core_duration_s}–{config.max_core_duration_s}s；前后各 {config.context_before_s}s 上下文。
平移为重建单位，不代表米。Missing/invalid poses are explicitly recorded. DA3 uses 5 FPS estimates with short-gap interpolation; EPIC Fields uses supplied registered poses.</p>
<p><a href="summary.json">Summary</a> · <a href="thresholds.json">Thresholds</a> · <a href="segments.jsonl">Segments</a></p>
<details><summary>完整视频的运动时间线</summary>{videos}</details>{''.join(cards)}'''
    (output / "index.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["epic_fields", "da3"], required=True)
    parser.add_argument("--poses-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--per-video", type=int, default=6)
    parser.add_argument("--min-rotation-deg", type=float, default=30)
    parser.add_argument("--min-view-axis-deg", type=float, default=20)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    config = LongCameraConfig(min_net_rotation_deg=args.min_rotation_deg, min_view_axis_change_deg=args.min_view_axis_deg)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "thresholds.json", asdict(config))
    all_candidates, summaries = [], []
    video_index = {p.stem: p for p in args.video_root.rglob("*.MP4")}
    files = sorted(args.poses_root.glob("*/camera_samples.npz")) if args.backend == "da3" else sorted(args.poses_root.glob("P*.json"))
    for path in files:
        video_id = path.parent.name if args.backend == "da3" else path.stem
        if video_id not in video_index:
            continue
        video = video_index[video_id]
        meta = inspect_video(video)
        error = seams = None
        if args.backend == "da3":
            data = np.load(path)
            indices, poses = data["source_frame_indices"], data["camera_to_world"]
            error = np.interp(np.arange(meta.frame_count), indices, data["alignment_error_deg"])
            seams = np.interp(np.arange(meta.frame_count), indices, data["seam"].astype(float)) > 0
        else:
            indices, poses = load_fields(path)
        inside = (indices >= 0) & (indices < meta.frame_count)
        timeline = dense_timeline(indices[inside], poses[inside], meta.frame_count, meta.fps, config)
        candidates = mine_long_camera(timeline, meta.fps, config, error, seams)
        for event in candidates:
            event.update({"video_id": video_id, "source_video": str(video), "source_pose_file": str(path),
                          "backend": args.backend, "fps": meta.fps,
                          "event_id": f'{video_id}_{event["core_start_frame"]:07d}_{event["core_end_frame"]:07d}'})
            if event["pose_gate_pass"]:
                event.update(quality(event, video))
                if event["endpoint_sharpness_min"] < config.min_endpoint_sharpness:
                    event["rejection_reasons"].append("blurred_endpoint")
                if event["usable_exposure_min"] < config.min_usable_exposure_fraction:
                    event["rejection_reasons"].append("poor_endpoint_exposure")
            event["accepted_candidate"] = not event["rejection_reasons"]
        export_timeline(args.output / "timelines" / video_id, timeline, meta.fps, candidates, config)
        summary = {"video_id": video_id, "source_video": str(video), "duration_seconds": meta.duration_seconds,
                   "source_frame_count": meta.frame_count, "annotated_timeline_rows": len(timeline["valid"]),
                   "supplied_pose_count": int(np.sum(timeline["pose_status"] == 1)),
                   "interpolated_pose_count": int(np.sum(timeline["pose_status"] == 2)),
                   "missing_pose_count": int(np.sum(timeline["pose_status"] == 0)),
                   "valid_pose_fraction": float(np.mean(timeline["valid"])),
                   "candidate_count": len(candidates), "accepted_candidate_count": sum(x["accepted_candidate"] for x in candidates)}
        summaries.append(summary)
        all_candidates.extend(candidates)
        print(json.dumps(summary), flush=True)
    accepted = [x for x in all_candidates if x["accepted_candidate"]]
    selected = select_diverse_candidates(accepted, config, args.count, args.per_video)
    validation = []
    if not args.no_render:
        for index, event in enumerate(selected):
            print(f"RENDER {index + 1}/{len(selected)} {event['event_id']}", flush=True)
            validation.append(export_clip(event, args.output))
    for name, rows in [("candidates_all.jsonl", all_candidates), ("segments.jsonl", selected)]:
        with (args.output / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {"backend": args.backend, "videos": summaries, "selected_count": len(selected),
               "eligible_count": len(accepted), "requested_count": args.count,
               "rejection_counts": dict(Counter(r for e in all_candidates for r in e["rejection_reasons"])),
               "source_video_seconds": sum(e["duration_seconds"] for e in summaries),
               "rotation_threshold_counts": {str(t): sum(e["net_rotation_deg"] >= t for e in accepted) for t in [30, 45, 60]},
               "active_perception_verified": False, "thresholds": asdict(config),
               "selection": "seeded random among eligible, capped per video, temporal overlap suppression",
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "validation.json", validation)
    if not args.no_render:
        gallery(args.output, selected, summaries, config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
