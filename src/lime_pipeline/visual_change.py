from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualChangeConfig:
    """Hard endpoint-view gate for the strict-v2 mining profile.

    Pixel and block fractions use luminance differences, while RGB MAE also
    captures chromatic changes. Requiring spatially distributed changed blocks
    prevents a moving hand in one small region from satisfying the gate alone.
    """

    width: int = 320
    height: int = 180
    luminance_pixel_delta: float = 0.10
    block_luminance_delta: float = 0.08
    block_rows: int = 9
    block_columns: int = 8
    min_rgb_mean_absolute_difference: float = 0.06
    min_changed_pixel_fraction: float = 0.20
    min_changed_block_fraction: float = 0.30
    max_global_luminance_ssim: float = 0.80

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("visual comparison dimensions must be positive")
        if self.height % self.block_rows or self.width % self.block_columns:
            raise ValueError("visual comparison dimensions must be divisible by the block grid")


def _global_ssim(start: np.ndarray, goal: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    start_mean = float(start.mean())
    goal_mean = float(goal.mean())
    start_variance = float(start.var())
    goal_variance = float(goal.var())
    covariance = float(((start - start_mean) * (goal - goal_mean)).mean())
    numerator = (2 * start_mean * goal_mean + c1) * (2 * covariance + c2)
    denominator = (
        (start_mean * start_mean + goal_mean * goal_mean + c1)
        * (start_variance + goal_variance + c2)
    )
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def measure_visual_change(
    start_image: Image.Image,
    goal_image: Image.Image,
    config: VisualChangeConfig | None = None,
) -> dict[str, Any]:
    config = config or VisualChangeConfig()
    size = (config.width, config.height)
    start = cv2.resize(np.asarray(start_image.convert("RGB")), size, interpolation=cv2.INTER_AREA)
    goal = cv2.resize(np.asarray(goal_image.convert("RGB")), size, interpolation=cv2.INTER_AREA)
    start_rgb = start.astype(np.float32) / 255.0
    goal_rgb = goal.astype(np.float32) / 255.0
    start_luminance = cv2.cvtColor(start, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    goal_luminance = cv2.cvtColor(goal, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance_difference = np.abs(start_luminance - goal_luminance)

    block_height = config.height // config.block_rows
    block_width = config.width // config.block_columns
    block_means = luminance_difference.reshape(
        config.block_rows,
        block_height,
        config.block_columns,
        block_width,
    ).mean(axis=(1, 3))

    rgb_mae = float(np.abs(start_rgb - goal_rgb).mean())
    changed_pixels = float((luminance_difference >= config.luminance_pixel_delta).mean())
    changed_blocks = float((block_means >= config.block_luminance_delta).mean())
    global_ssim = _global_ssim(start_luminance, goal_luminance)
    checks = {
        "rgb_mean_absolute_difference": rgb_mae >= config.min_rgb_mean_absolute_difference,
        "changed_pixel_fraction": changed_pixels >= config.min_changed_pixel_fraction,
        "changed_block_fraction": changed_blocks >= config.min_changed_block_fraction,
        "global_luminance_ssim": global_ssim <= config.max_global_luminance_ssim,
    }
    score = float(
        (
            rgb_mae / config.min_rgb_mean_absolute_difference
            + changed_pixels / config.min_changed_pixel_fraction
            + changed_blocks / config.min_changed_block_fraction
            + (1.0 - global_ssim) / (1.0 - config.max_global_luminance_ssim)
        )
        / 4.0
    )
    return {
        "profile": "strict_v2_broad_endpoint_change",
        "comparison_resolution": [config.width, config.height],
        "rgb_mean_absolute_difference": round(rgb_mae, 6),
        "luminance_changed_pixel_fraction": round(changed_pixels, 6),
        "spatial_changed_block_fraction": round(changed_blocks, 6),
        "global_luminance_ssim": round(global_ssim, 6),
        "view_change_score": round(score, 6),
        "checks": checks,
        "passes": all(checks.values()),
        "thresholds": asdict(config),
    }
