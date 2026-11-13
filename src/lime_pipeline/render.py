from __future__ import annotations

import csv
import html
import json
import os
import shutil
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data import read_video_frames
from .schema import normalize_label, seconds_to_timestamp, write_json, write_jsonl


DEFAULT_FFMPEG = Path("/mnt/nfs_data/zhangchi/model/tools/lime-local/ffmpeg")


def _ffmpeg_executable() -> str:
    candidates = [os.environ.get("LIME_FFMPEG"), str(DEFAULT_FFMPEG), shutil.which("ffmpeg")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "H.264 rendering requires FFmpeg; set LIME_FFMPEG or install it at "
        f"{DEFAULT_FFMPEG}"
    )


def _put_text_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    scale: float = 0.62,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_height: int = 26,
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_height


def _wrapped(text: str, width: int = 92, max_lines: int = 2) -> list[str]:
    lines = textwrap.wrap(" ".join(text.split()), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def _read_frame_bgr(video_path: Path, index: int) -> np.ndarray:
    image = read_video_frames(video_path, [index])[index]
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


HAND_COLORS = {
    "left": (30, 160, 255),
    "right": (60, 220, 90),
}


def _draw_hand_trajectories(
    frame: np.ndarray,
    event: dict[str, Any],
    current_frame: int | None = None,
) -> np.ndarray:
    """Overlay VITRA palm tracks retained by the Qwen trajectory QC stage."""
    hands = ((event.get("hand_motion") or {}).get("hands") or {})
    height, width = frame.shape[:2]
    for side in ("left", "right"):
        hand = hands.get(side) or {}
        if not hand.get("available"):
            continue
        samples = hand.get("trajectory_samples") or []
        if current_frame is not None:
            samples = [sample for sample in samples if int(sample.get("video_frame", -1)) <= current_frame]
        points: list[tuple[int, int]] = []
        for sample in samples:
            uv = sample.get("palm_uv_normalized")
            if not isinstance(uv, list) or len(uv) != 2:
                continue
            x = int(round(float(uv[0]) * width))
            y = int(round(float(uv[1]) * height))
            if -width // 10 <= x <= width + width // 10 and -height // 10 <= y <= height + height // 10:
                points.append((x, y))
        color = HAND_COLORS[side]
        if len(points) >= 2:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 4, cv2.LINE_AA)
        for point in points[:-1]:
            cv2.circle(frame, point, 4, color, -1, cv2.LINE_AA)
        if points:
            cv2.circle(frame, points[-1], 10, color, 3, cv2.LINE_AA)
            cv2.putText(
                frame,
                side.upper(),
                (points[-1][0] + 12, points[-1][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    return frame


def _hand_summary(event: dict[str, Any]) -> str | None:
    motion = event.get("hand_motion") or {}
    primary = motion.get("primary_hand")
    hand = (motion.get("hands") or {}).get(primary) or {}
    if not hand.get("available"):
        return None
    return (
        f"{str(primary).upper()}: endpoint {float(hand['translation_distance_m']):.3f} m  |  "
        f"path {float(hand['path_length_m']):.3f} m  |  "
        f"rotation {float(hand['rotation_angle_deg']):.1f} deg  |  "
        f"grip Δ {1000.0 * float(hand['gripper_width_change_m']):+.1f} mm"
    )


def _annotate_still(frame: np.ndarray, heading: str, footer: str, target_width: int = 800) -> np.ndarray:
    height, width = frame.shape[:2]
    target_height = int(round(height * target_width / width))
    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (target_width, 44), (0, 0, 0), -1)
    cv2.putText(frame, heading, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    panel = np.zeros((82, target_width, 3), dtype=np.uint8)
    _put_text_lines(panel, _wrapped(footer, width=100, max_lines=2), (12, 28), scale=0.56, line_height=27)
    return np.vstack([frame, panel])


def render_contact_sheet(event: dict[str, Any], path: Path) -> None:
    video = Path(event["source"]["video_path"])
    start = int(event["time"]["start_frame"])
    goal = int(event["time"]["goal_frame"])
    label = event.get("label") or {}
    footer = label.get("action_caption", "Unlabeled camera-motion event")
    start_image = _draw_hand_trajectories(_read_frame_bgr(video, start), event, start)
    goal_image = _draw_hand_trajectories(_read_frame_bgr(video, goal), event, goal)
    start_frame = _annotate_still(start_image, f"START  {event['time']['start_label']}  frame {start}", footer)
    goal_frame = _annotate_still(goal_image, f"GOAL   {event['time']['goal_label']}  frame {goal}", label.get("observation_gain", footer))
    separator = np.full((start_frame.shape[0], 8, 3), 40, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.hstack([start_frame, separator, goal_frame])):
        raise RuntimeError(f"failed to write {path}")


def render_event_clip(event: dict[str, Any], path: Path, target_width: int = 960, output_fps: float = 15.0) -> None:
    video = Path(event["source"]["video_path"])
    source_fps = float(event["source"]["video"]["fps"])
    start = int(event["time"]["start_frame"])
    goal = int(event["time"]["goal_frame"])
    step = max(1, int(round(source_fps / output_fps)))
    actual_fps = source_fps / step
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = int(round(source_height * target_width / source_width))
    panel_height = 240
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_name(f".{path.stem}.mp4v-source.mp4")
    raw_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (target_width, target_height + panel_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video writer for {raw_path}")
    label = event.get("label") or {}
    motion = event["camera_motion"]
    frame_index = start
    while frame_index <= goal:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        _draw_hand_trajectories(frame, event, frame_index)
        cv2.rectangle(frame, (0, 0), (target_width, 46), (0, 0, 0), -1)
        now = seconds_to_timestamp(frame_index / source_fps)
        top = f"{event['event_id']}  |  {now}  frame {frame_index}"
        cv2.putText(frame, top, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.zeros((panel_height, target_width, 3), dtype=np.uint8)
        interval = f"SOURCE TIME {event['time']['start_label']} -> {event['time']['goal_label']}  (inclusive)"
        geometry = (
            f"CAMERA: {motion['translation_distance_m']:.3f} m {motion['translation_direction']}  |  "
            f"{motion['rotation_angle_deg']:.1f} deg {motion['rotation_direction']}"
        )
        lines = [interval, geometry]
        gain_audit = event.get("observation_gain_vlm") or {}
        if gain_audit:
            verdict = "PASS" if gain_audit.get("accepted") else "REJECT"
            lines.append(
                f"VLM GAIN: {verdict}  |  {gain_audit.get('gain_type', 'unknown')}  |  "
                f"cause {gain_audit.get('primary_cause', 'unknown')}  |  conf {float(gain_audit.get('confidence', 0.0)):.2f}"
            )
        hand_summary = _hand_summary(event)
        if hand_summary:
            lines.append("HAND: " + hand_summary)
        view = event.get("visual_change") or {}
        if view:
            lines.append(
                "VIEW: "
                f"RGB Δ {view['rgb_mean_absolute_difference']:.3f}  |  "
                f"pixels {100 * view['luminance_changed_pixel_fraction']:.1f}%  |  "
                f"blocks {100 * view['spatial_changed_block_fraction']:.1f}%  |  "
                f"SSIM {view['global_luminance_ssim']:.3f}"
            )
        lines.extend("ACTION: " + line if i == 0 else line for i, line in enumerate(_wrapped(label.get("action_caption", "unlabeled"))))
        lines.extend("GAIN: " + line if i == 0 else line for i, line in enumerate(_wrapped(label.get("observation_gain", "unlabeled"))))
        _put_text_lines(panel, lines[:8], (12, 24), scale=0.52, line_height=25)
        progress = (frame_index - start) / max(goal - start, 1)
        cv2.rectangle(panel, (12, panel_height - 12), (target_width - 12, panel_height - 6), (70, 70, 70), -1)
        cv2.rectangle(panel, (12, panel_height - 12), (12 + int((target_width - 24) * progress), panel_height - 6), (60, 210, 90), -1)
        writer.write(np.vstack([frame, panel]))
        frame_index += step
    capture.release()
    writer.release()
    try:
        subprocess.run(
            [
                _ffmpeg_executable(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
                str(path),
            ],
            check=True,
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        raw_path.unlink(missing_ok=True)


def _vtt_timestamp(seconds: float) -> str:
    return seconds_to_timestamp(seconds)


def write_subtitles(events: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["source"]["video_path"]].append(event)
    for video_path, group in grouped.items():
        stem = f"{Path(video_path).parent.name}__{Path(video_path).stem}"
        vtt = ["WEBVTT", ""]
        srt: list[str] = []
        for index, event in enumerate(sorted(group, key=lambda x: x["time"]["start_frame"]), start=1):
            start = float(event["time"]["start_seconds"])
            goal = float(event["time"]["goal_seconds"])
            caption = (event.get("label") or {}).get("action_caption", "Unlabeled camera-motion event")
            vtt.extend([str(index), f"{_vtt_timestamp(start)} --> {_vtt_timestamp(goal)}", caption, ""])
            srt.extend([str(index), f"{_vtt_timestamp(start).replace('.', ',')} --> {_vtt_timestamp(goal).replace('.', ',')}", caption, ""])
        (output_dir / f"{stem}.vtt").write_text("\n".join(vtt), encoding="utf-8")
        (output_dir / f"{stem}.srt").write_text("\n".join(srt), encoding="utf-8")


def export_training_examples(events: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if not event.get("label") or not event["label"].get("is_active_perception"):
            continue
        event_dir = output_dir / "events" / event["event_id"]
        video_path = Path(event["source"]["video_path"])
        frames = read_video_frames(video_path, [event["time"]["start_frame"], event["time"]["goal_frame"]])
        start_path = event_dir / "start.jpg"
        goal_path = event_dir / "goal.jpg"
        start_path.parent.mkdir(parents=True, exist_ok=True)
        frames[event["time"]["start_frame"]].save(start_path, quality=94)
        frames[event["time"]["goal_frame"]].save(goal_path, quality=94)
        motion = event["camera_motion"]
        pose_target = motion["translation_start_camera_m"] + motion["rotation_6d_first_two_columns"]
        for intent_index, intent in enumerate(event["label"]["plausible_intents"], start=1):
            rows.append(
                {
                    "example_id": f"{event['event_id']}__intent_{intent_index:02d}",
                    "event_id": event["event_id"],
                    "start_image": str(start_path),
                    "goal_image_privileged_labeling_only": str(goal_path),
                    "intent": intent["intent"],
                    "intent_family": intent["intent_family"],
                    "observation_gain": event["label"]["observation_gain"],
                    "pose_target_9d": pose_target,
                    "relative_transform_4x4": motion["relative_transform_4x4"],
                    "time": event["time"],
                }
            )
    write_jsonl(output_dir / "training_examples.jsonl", rows)
    return rows


def write_csv(events: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "event_id", "video_path", "start_frame", "goal_frame", "start_time", "goal_time", "duration_seconds",
        "translation_m", "rotation_deg", "camera_movement", "is_active_perception", "action_caption",
        "primary_hand", "hand_translation_m", "hand_path_m", "hand_rotation_deg", "gripper_width_change_m",
        "hand_quality_pass", "observation_gain", "top_intent", "intent_family", "label_confidence",
        "vlm_gain_accepted", "vlm_gain_type", "vlm_primary_cause", "vlm_camera_contribution",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            label = event.get("label") or {}
            intents = label.get("plausible_intents") or [{}]
            hand_motion = event.get("hand_motion") or {}
            primary_hand = hand_motion.get("primary_hand")
            hand = (hand_motion.get("hands") or {}).get(primary_hand) or {}
            gain_audit = event.get("observation_gain_vlm") or {}
            writer.writerow(
                {
                    "event_id": event["event_id"],
                    "video_path": event["source"]["video_path"],
                    "start_frame": event["time"]["start_frame"],
                    "goal_frame": event["time"]["goal_frame"],
                    "start_time": event["time"]["start_label"],
                    "goal_time": event["time"]["goal_label"],
                    "duration_seconds": event["time"]["duration_seconds"],
                    "translation_m": event["camera_motion"]["translation_distance_m"],
                    "rotation_deg": event["camera_motion"]["rotation_angle_deg"],
                    "camera_movement": f"{event['camera_motion']['translation_direction']}; {event['camera_motion']['rotation_direction']}",
                    "is_active_perception": label.get("is_active_perception"),
                    "action_caption": label.get("action_caption"),
                    "primary_hand": primary_hand,
                    "hand_translation_m": hand.get("translation_distance_m"),
                    "hand_path_m": hand.get("path_length_m"),
                    "hand_rotation_deg": hand.get("rotation_angle_deg"),
                    "gripper_width_change_m": hand.get("gripper_width_change_m"),
                    "hand_quality_pass": hand.get("quality_pass"),
                    "observation_gain": label.get("observation_gain"),
                    "top_intent": intents[0].get("intent"),
                    "intent_family": intents[0].get("intent_family"),
                    "label_confidence": label.get("label_confidence"),
                    "vlm_gain_accepted": gain_audit.get("accepted"),
                    "vlm_gain_type": gain_audit.get("gain_type"),
                    "vlm_primary_cause": gain_audit.get("primary_cause"),
                    "vlm_camera_contribution": gain_audit.get("camera_contribution"),
                }
            )


def write_html_report(events: list[dict[str, Any]], output_dir: Path) -> None:
    cards: list[str] = []
    for event in events:
        label = event.get("label") or {}
        view = event.get("visual_change") or {}
        gain_audit = event.get("observation_gain_vlm") or {}
        hand_summary = _hand_summary(event) or "not available"
        curation = event.get("curation") or {}
        curation_row = (
            f"<dt>Curation</dt><dd>rank {int(curation['rank'])}; score {float(curation['score']):.3f}; "
            f"preferred-tier {html.escape(str(bool(curation.get('preferred'))))}</dd>"
            if curation.get("rank") is not None else ""
        )
        gain_row = (
            f"<dt>VLM gain audit</dt><dd><strong>{'PASS' if gain_audit.get('accepted') else 'REJECT'}</strong>; "
            f"{html.escape(str(gain_audit.get('gain_type', 'unknown')))}; cause "
            f"{html.escape(str(gain_audit.get('primary_cause', 'unknown')))}; camera contribution "
            f"{html.escape(str(gain_audit.get('camera_contribution', 'unknown')))}; "
            f"confidence {float(gain_audit.get('confidence', 0.0)):.2f}</dd>"
            if gain_audit else ""
        )
        evidence_row = (
            "<dt>Visible evidence</dt><dd>" + html.escape("; ".join(gain_audit.get("gained_evidence") or [gain_audit.get("no_gain_reason", "")])) + "</dd>"
            if gain_audit else ""
        )
        event_rel = f"events/{event['event_id']}"
        intents = "".join(
            f"<li><span>{html.escape(x['intent_family'])}</span> {html.escape(x['intent'])} ({x['confidence']:.2f})</li>"
            for x in label.get("plausible_intents", [])
        )
        gain_class = " gain-pass" if gain_audit.get("accepted") else (" gain-reject" if gain_audit else "")
        cards.append(
            f"""<article class="card{gain_class}">
<h2>{html.escape(event['event_id'])}</h2>
<p class="time">{event['time']['start_label']} → {event['time']['goal_label']} · frames {event['time']['start_frame']}–{event['time']['goal_frame']}</p>
<img loading="lazy" src="{event_rel}/start_goal.jpg" alt="Start and goal frames">
<video controls preload="none" src="{event_rel}/annotated.mp4"></video>
<dl>
<dt>Action</dt><dd>{html.escape(label.get('action_caption', 'Unlabeled'))}</dd>
<dt>Camera</dt><dd>{event['camera_motion']['translation_distance_m']:.3f} m {html.escape(event['camera_motion']['translation_direction'])}; {event['camera_motion']['rotation_angle_deg']:.1f}° {html.escape(event['camera_motion']['rotation_direction'])}</dd>
<dt>Hands</dt><dd>{html.escape(hand_summary)}; orange = left, green = right</dd>
{curation_row}
{gain_row}
{evidence_row}
<dt>View difference</dt><dd>{('RGB MAE %.3f; changed pixels %.1f%%; changed blocks %.1f%%; SSIM %.3f' % (view['rgb_mean_absolute_difference'], 100 * view['luminance_changed_pixel_fraction'], 100 * view['spatial_changed_block_fraction'], view['global_luminance_ssim'])) if view else 'not measured'}</dd>
<dt>Observation gain</dt><dd>{html.escape(label.get('observation_gain', 'Unlabeled'))}</dd>
<dt>Active perception</dt><dd>{html.escape(str(label.get('is_active_perception', 'unknown')))} · confidence {label.get('label_confidence', 0):.2f}</dd>
</dl><h3>Plausible hindsight intents</h3><ul>{intents}</ul>
</article>"""
        )
    curated = any(bool(event.get("curation")) for event in events)
    collection_name = "curated strict" if curated else "strict"
    audited = [event for event in events if event.get("observation_gain_vlm")]
    accepted_count = sum(bool(event["observation_gain_vlm"].get("accepted")) for event in audited)
    audit_intro = (
        f" VLM causal audit: {accepted_count} PASS, {len(audited) - accepted_count} REJECT."
        if audited else ""
    )
    controls = (
        '<div class="controls"><button onclick="filterCards(\'all\')">All</button>'
        '<button onclick="filterCards(\'gain-pass\')">PASS</button>'
        '<button onclick="filterCards(\'gain-reject\')">REJECT</button></div>'
        if audited else ""
    )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LIME local demo</title>
<style>
:root{{--bg:#101418;--card:#182028;--text:#edf3f7;--muted:#a9bac7;--accent:#76e39a}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:32px}}
h1{{margin-bottom:4px}}.intro,.time{{color:var(--muted)}}.grid{{display:grid;gap:24px}}.card{{background:var(--card);border-radius:14px;padding:20px}}
.card.gain-pass{{border:2px solid #49d47d}}.card.gain-reject{{border:2px solid #d06868}}.card.hidden{{display:none}}
.controls{{position:sticky;top:0;z-index:5;background:rgba(16,20,24,.95);padding:12px 0;margin:16px 0}}button{{margin-right:10px;padding:9px 18px;border:0;border-radius:8px;background:#2b3945;color:var(--text);cursor:pointer}}button:hover{{background:#3a4d5d}}
.card img,.card video{{display:block;width:100%;margin:12px 0;border-radius:8px;background:#000}}dl{{display:grid;grid-template-columns:160px 1fr;gap:8px 16px}}dt{{color:var(--accent);font-weight:700}}dd{{margin:0}}li span{{color:var(--accent)}}
</style></head><body><main><h1>LIME local egocentric-motion demo</h1>
<p class="intro"><strong>{len(events)} {collection_name} EPIC-KITCHENS events.</strong>{audit_intro} Geometry and broad RGB change propose candidates; the five-frame VLM audit assesses whether persistent evidence is gained through camera movement. Audit model: {html.escape(', '.join(sorted({str(event.get('provenance', {}).get('observation_gain_labeling', {}).get('model', 'unknown')) for event in audited})) or 'not audited')}. VITRA camera/MANO trajectories, Qwen-RobotManip virtual-gripper retargeting, exact source-frame mappings, and official EPIC action narrations are retained.</p>
{controls}<div class="grid">{''.join(cards)}</div></main>
<script>function filterCards(kind){{document.querySelectorAll('.card').forEach(card=>card.classList.toggle('hidden',kind!=='all'&&!card.classList.contains(kind)))}}</script>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def render_all(events: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        event_dir = output_dir / "events" / event["event_id"]
        render_contact_sheet(event, event_dir / "start_goal.jpg")
        render_event_clip(event, event_dir / "annotated.mp4")
        write_json(event_dir / "event.json", event)
    write_jsonl(output_dir / "events.jsonl", events)
    active_events = [event for event in events if bool((event.get("label") or {}).get("is_active_perception"))]
    write_jsonl(output_dir / "active_events.jsonl", active_events)
    write_csv(events, output_dir / "events.csv")
    write_subtitles(events, output_dir)
    training_rows = export_training_examples(events, output_dir)
    write_html_report(events, output_dir)
    summary = {
        "event_count": len(events),
        "active_perception_count": sum(bool((x.get("label") or {}).get("is_active_perception")) for x in events),
        "training_example_count": len(training_rows),
        "report": str(output_dir / "index.html"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def apply_review_overrides(events: list[dict[str, Any]], review_path: Path) -> list[dict[str, Any]]:
    """Apply explicit, provenance-recorded demo review corrections.

    Only structured label fields may be changed; measured time and geometry are
    deliberately immutable in this step.
    """
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(reviews, dict):
        raise ValueError("review override file must be an object keyed by event_id")
    allowed = {
        "is_active_perception", "motion_type", "action_caption", "newly_visible", "improved_views",
        "spatial_anchors", "observation_gain", "plausible_intents", "label_confidence", "grounding_notes",
    }
    by_id = {event["event_id"]: event for event in events}
    for event_id, review in reviews.items():
        if event_id not in by_id:
            raise ValueError(f"review refers to unknown event: {event_id}")
        if not isinstance(review, dict) or not isinstance(review.get("fields"), dict):
            raise ValueError(f"invalid review for {event_id}")
        unexpected = set(review["fields"]) - allowed
        if unexpected:
            raise ValueError(f"review attempts to alter unsupported fields for {event_id}: {sorted(unexpected)}")
        event = by_id[event_id]
        event["label"].update(review["fields"])
        event["label"] = normalize_label(event["label"])
        event.setdefault("provenance", {})["manual_review"] = {
            "reviewer": review.get("reviewer", "unspecified"),
            "reason": review.get("reason", ""),
            "review_file": str(review_path.resolve()),
        }
    return events
