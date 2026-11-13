#!/usr/bin/env python3
"""Full-video DA3 inference with the released DA3-Streaming Sim(3) aligner.

Uses bounded overlapping windows; no loop closure or metric scale is claimed.
All frames used by the estimator have explicit zero-based source-frame indices.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lime_pipeline.data import inspect_video


def sample_video(video: Path, directory: Path, rate: float) -> tuple[list[Path], np.ndarray]:
    meta = inspect_video(video)
    stride = max(1, round(meta.fps / rate))
    indices = np.unique(np.r_[np.arange(0, meta.frame_count, stride), meta.frame_count - 1]).astype(int)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"frame_{index:09d}.jpg" for index in indices]
    if all(path.exists() for path in paths):
        return paths, indices
    capture = cv2.VideoCapture(str(video))
    by_index = dict(zip(indices, paths))
    for index in range(meta.frame_count):
        if not capture.grab():
            raise RuntimeError(f"decode failed at {video.stem} frame {index}")
        if index in by_index:
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError("frame retrieval failed")
            frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
            if not cv2.imwrite(str(by_index[index]), frame):
                raise RuntimeError("frame write failed")
    capture.release()
    return paths, indices


def small_point_maps(depth: np.ndarray, intrinsics: np.ndarray, poses: np.ndarray) -> np.ndarray:
    h, w = depth.shape[1:]
    yy, xx = np.mgrid[0:h:6, 0:w:6]
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(float)
    rays = np.einsum("nij,hwj->nhwi", np.linalg.inv(intrinsics), pixels)
    camera_points = rays * depth[:, ::6, ::6, None]
    return np.einsum("nij,nhwj->nhwi", poses[:, :3, :3], camera_points) + poses[:, None, None, :3, 3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--process-res", type=int, default=378)
    args = parser.parse_args()
    if not 2 <= args.overlap < args.chunk_size:
        raise SystemExit("Require 2 <= overlap < chunk-size")
    import torch
    from depth_anything_3.api import DepthAnything3

    if not torch.cuda.is_available():
        raise SystemExit("Run inside a Slurm GPU allocation")
    torch.set_num_threads(8)
    torch.manual_seed(20260908)
    align_file = args.da3_repo / "da3_streaming/loop_utils/alignment_torch.py"
    spec = importlib.util.spec_from_file_location("released_da3_alignment", align_file)
    alignment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alignment)
    model = DepthAnything3.from_pretrained(str(args.model), local_files_only=True).to("cuda").eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for video in args.video:
        output = args.output / video.stem
        output.mkdir(parents=True, exist_ok=True)
        meta = inspect_video(video)
        run = {"video": asdict(meta), "model": str(args.model), "sample_fps": args.sample_fps,
               "chunk_size": args.chunk_size, "overlap": args.overlap, "process_res": args.process_res,
               "method": "DA3-Small sliding-window inference + released DA3-Streaming robust dense-overlap Sim3",
               "loop_closure": False, "translation_unit": "arbitrary_reconstruction_units", "metric_scale": False}
        done = output / "camera_samples.npz"
        run_path = output / "estimation.json"
        if done.exists() and run_path.exists():
            previous = json.loads(run_path.read_text())
            if all(previous.get(k) == v for k, v in run.items()):
                print(f"resume completed {video.stem}", flush=True)
                continue
            raise RuntimeError("Output config changed; use a new output root")
        started = time.monotonic()
        paths, indices = sample_video(video, output / "sampled_frames", args.sample_fps)
        n = len(paths)
        c2w = np.full((n, 4, 4), np.nan)
        intrinsics = np.full((n, 3, 3), np.nan)
        median_depth = np.full(n, np.nan)
        consistency = np.zeros(n)
        seam = np.zeros(n, dtype=bool)
        chunks = []
        previous_points = previous_conf = None
        previous_start = previous_end = 0
        stride = args.chunk_size - args.overlap
        for start in range(0, n, stride):
            end = min(start + args.chunk_size, n)
            if end <= previous_end:
                break
            print(f"INFER {video.stem} samples={start}:{end}/{n}", flush=True)
            with torch.inference_mode():
                prediction = model.inference([str(p) for p in paths[start:end]],
                                             process_res=args.process_res, ref_view_strategy="saddle_balanced")
            local = np.repeat(np.eye(4)[None], end - start, axis=0)
            local[:, :3, :] = np.asarray(prediction.extrinsics)
            local = np.linalg.inv(local)
            k = np.asarray(prediction.intrinsics)
            depth = np.asarray(prediction.depth)
            inference_image_shape = list(depth.shape[1:])
            points = small_point_maps(depth, k, local)
            conf = np.maximum(np.asarray(prediction.conf)[:, ::6, ::6] - 1.0, 0)
            depth_scale = 1.0
            overlap_error = 0.0
            residual = 0.0
            if start:
                count = previous_end - start
                target = previous_points[start - previous_start:]
                source = points[:count]
                weights = np.sqrt(conf[:count] * previous_conf[start - previous_start:])
                # Consistent, high-confidence upper scene pixels reduce foreground-hand influence.
                yy = np.arange(weights.shape[1])[None, :, None]
                mask = (weights > np.quantile(weights, 0.35)) & (yy < weights.shape[1] * 0.70)
                mask &= np.isfinite(source).all(-1) & np.isfinite(target).all(-1)
                if mask.sum() < 100:
                    raise RuntimeError("Insufficient reliable overlap for chunk alignment")
                source_points, target_points, ww = source[mask], target[mask], weights[mask]
                depth_scale, rr, tt = alignment.robust_weighted_estimate_sim3_torch(
                    source_points, target_points, ww, delta=0.1, max_iters=5)
                depth_scale = float(depth_scale)
                if not np.isfinite(depth_scale) or depth_scale <= 0:
                    raise RuntimeError("Invalid Sim3 scale")
                points = depth_scale * np.einsum("ij,nhwj->nhwi", rr, points) + tt
                local[:, :3, :3] = rr @ local[:, :3, :3]
                local[:, :3, 3] = depth_scale * (local[:, :3, 3] @ rr.T) + tt
                transformed = depth_scale * (source_points @ rr.T) + tt
                residual = float(np.median(np.linalg.norm(transformed - target_points, axis=1)))
                differences = np.swapaxes(c2w[start:previous_end, :3, :3], 1, 2) @ local[:count, :3, :3]
                overlap_error = float(np.median(Rotation.from_matrix(differences).magnitude()) * 180 / np.pi)
                # Keep the previous overlap; append the newly seen frames. Mark the join.
                write_start = previous_end
                seam[max(0, write_start - 1):min(n, write_start + 2)] = True
            else:
                write_start = start
            offset = write_start - start
            c2w[write_start:end] = local[offset:]
            intrinsics[write_start:end] = k[offset:]
            median_depth[write_start:end] = np.median(depth[offset:], axis=(1, 2)) * depth_scale
            consistency[write_start:end] = overlap_error
            previous_points, previous_conf = points, conf
            previous_start, previous_end = start, end
            chunks.append({"sample_start": start, "sample_end_exclusive": end,
                           "scale_to_first_chunk": depth_scale, "overlap_rotation_error_deg": overlap_error,
                           "median_alignment_residual_units": residual})
            print(json.dumps(chunks[-1]), flush=True)
            del prediction, depth
        if not np.isfinite(c2w).all():
            raise RuntimeError("Incomplete trajectory; not publishing completed output")
        np.savez_compressed(done, source_frame_indices=indices, camera_to_world=c2w.astype(np.float32),
                            intrinsics=intrinsics.astype(np.float32), median_depth=median_depth.astype(np.float32),
                            alignment_error_deg=consistency.astype(np.float32), seam=seam)
        run.update({"sample_count": n, "duration_wall_seconds": time.monotonic() - started,
                    "chunks": chunks, "inference_image_shape": inference_image_shape,
                    "alignment_grid_shape": list(conf.shape[1:])})
        run_path.write_text(json.dumps(run, indent=2) + "\n")
        print(f"DONE {video.stem} samples={n} wall_seconds={run['duration_wall_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
