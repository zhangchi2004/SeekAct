#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lime_pipeline.data import read_video_frames
from lime_pipeline.render import _ffmpeg_executable


def faststart(path: Path) -> bool:
    atoms = []
    with path.open("rb") as handle:
        total = path.stat().st_size
        while handle.tell() + 8 <= total:
            offset = handle.tell()
            size, kind = struct.unpack(">I4s", handle.read(8))
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
            if size == 0:
                size = total - offset
            if size < 8:
                raise ValueError("Invalid MP4 atom")
            atoms.append(kind)
            handle.seek(offset + size)
    return b"moov" in atoms and b"mdat" in atoms and atoms.index(b"moov") < atoms.index(b"mdat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = _ffmpeg_executable()
    report = []
    for name in ["demo-epic-long-camera-da3-v1", "demo-epic-long-camera-epic_fields-v1"]:
        root = args.root / name
        events = [json.loads(line) for line in (root / "segments.jsonl").read_text().splitlines()]
        for event in events:
            for filename in ["clip.mp4", "review.mp4"]:
                path = root / "clips" / event["event_id"] / filename
                subprocess.run([ffmpeg, "-v", "error", "-threads", "2", "-i", str(path), "-f", "null", "-"], check=True)
                capture = cv2.VideoCapture(str(path))
                code = int(capture.get(cv2.CAP_PROP_FOURCC))
                codec = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
                fps = capture.get(cv2.CAP_PROP_FPS)
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()
                metadata = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
                # Some OpenCV builds return decoder FOURCC 'h264', not the MP4 sample-entry tag.
                assert "Video: h264" in metadata and "(avc1 /" in metadata and "yuv420p" in metadata and faststart(path), path
                row = {"collection": name, "event_id": event["event_id"], "file": filename,
                       "codec": "h264", "container_codec_tag": "avc1", "opencv_fourcc": codec,
                       "fps": fps, "decoded_frame_count": count, "full_decode_pass": True,
                       "faststart": True, "pixel_format": "yuv420p"}
                if filename == "clip.mp4":
                    expected = event["clip_end_frame"] - event["clip_start_frame"] + 1
                    assert count == expected and abs(fps - event["fps"]) < .01
                    source = read_video_frames(Path(event["source_video"]), [event["clip_start_frame"], event["clip_end_frame"]])
                    output = read_video_frames(path, [0, count - 1])
                    errors = []
                    for original, exported in zip(source.values(), output.values()):
                        a = cv2.resize(np.asarray(original), (640, 360), interpolation=cv2.INTER_AREA).astype(float)
                        b = cv2.resize(np.asarray(exported), (640, 360), interpolation=cv2.INTER_AREA).astype(float)
                        errors.append(float(np.abs(a - b).mean()))
                    row["first_last_frame_rgb_mae_0_255"] = errors
                    assert max(errors) < 12, (path, errors)
                report.append(row)
    reference = np.load(args.root / "demo-epic-long-camera-epic_fields-v1/timelines/P01_103/camera_full_timeline.npz")
    predicted = np.load(args.root / "demo-epic-long-camera-da3-v1/timelines/P01_103/camera_full_timeline.npz")
    comparison = {"reference_is_reconstruction_not_ground_truth": True, "video_id": "P01_103", "windows": {}}
    for hop in [50, 100]:
        i = np.arange(0, len(reference["valid"]) - hop, 25)
        j = i + hop
        valid = reference["valid"][i] & reference["valid"][j] & predicted["valid"][i] & predicted["valid"][j]
        i, j = i[valid], j[valid]
        a, b = reference["camera_to_world"], predicted["camera_to_world"]
        ra = np.swapaxes(a[i, :3, :3], 1, 2) @ a[j, :3, :3]
        rb = np.swapaxes(b[i, :3, :3], 1, 2) @ b[j, :3, :3]
        error = np.rad2deg(Rotation.from_matrix(np.swapaxes(ra, 1, 2) @ rb).magnitude())
        comparison["windows"][str(hop / 50)] = {"pairs": len(error),
            "rotation_disagreement_deg_p50_p90_p95": np.percentile(error, [50, 90, 95]).tolist()}
    (args.root / "long_camera_validation.json").write_text(json.dumps({"videos": report, "comparison": comparison}, indent=2) + "\n")
    print(json.dumps({"validated_mp4s": len(report), "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
