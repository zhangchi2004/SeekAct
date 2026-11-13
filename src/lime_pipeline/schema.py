from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "lime-local-v1"
INTENT_FAMILIES = {"target_approaching", "exploration", "perspective_shift"}


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds without losing the frame-derived millisecond label."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"invalid timestamp: {seconds}")
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("VLM response contains no JSON object")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("VLM response must be a JSON object")
    return obj


def _clean_list(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = " ".join(str(item).strip().split())
        if text and text.casefold() not in {x.casefold() for x in result}:
            result.append(text[:300])
    return result[:limit]


def normalize_label(obj: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the structured hindsight label."""
    for wrapper in ("annotation", "label", "result"):
        if isinstance(obj.get(wrapper), dict):
            obj = obj[wrapper]
            break
    active = obj.get("is_active_perception", True)
    if isinstance(active, str):
        active = active.strip().lower() in {"true", "yes", "1", "active"}
    active = bool(active)

    intents: list[dict[str, Any]] = []
    raw_intents = obj.get("plausible_intents", [])
    if isinstance(raw_intents, list):
        for item in raw_intents[:5]:
            if isinstance(item, str):
                item = {"intent": item}
            if not isinstance(item, dict):
                continue
            intent = " ".join(str(item.get("intent", "")).strip().split())
            family = str(item.get("intent_family", item.get("family", "perspective_shift"))).strip().lower()
            family = family.replace("-", "_").replace(" ", "_")
            if family not in INTENT_FAMILIES:
                family = "perspective_shift"
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            if intent:
                intents.append(
                    {
                        "intent": intent[:400],
                        "intent_family": family,
                        "confidence": round(min(1.0, max(0.0, confidence)), 3),
                    }
                )

    caption_value = obj.get("action_caption") or obj.get("action_movement_caption") or obj.get("caption") or ""
    gain_value = (
        obj.get("observation_gain")
        or obj.get("observation_gain_description")
        or obj.get("observation-gain")
        or ""
    )
    caption = " ".join(str(caption_value).strip().split())
    gain = " ".join(str(gain_value).strip().split())
    if not caption:
        raise ValueError("label is missing action_caption")
    if not gain:
        if active:
            raise ValueError("active label is missing observation_gain")
        gain = "No task-relevant observation gain is evident in this interval."
    if not intents and active:
        raise ValueError("label is missing plausible_intents")
    try:
        confidence = float(obj.get("label_confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "is_active_perception": active,
        "motion_type": " ".join(str(obj.get("motion_type", "compound")).strip().split())[:120],
        "action_caption": caption[:500],
        "newly_visible": _clean_list(obj.get("newly_visible")),
        "improved_views": _clean_list(obj.get("improved_views")),
        "spatial_anchors": _clean_list(obj.get("spatial_anchors")),
        "observation_gain": gain[:700],
        "plausible_intents": intents,
        "label_confidence": round(min(1.0, max(0.0, confidence)), 3),
        "grounding_notes": " ".join(str(obj.get("grounding_notes", "")).strip().split())[:700],
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
