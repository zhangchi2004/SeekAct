# Youyi & Sen: kitchen camera-motion pipeline

This repository contains the camera-first code used to mine camera-motion
segments from kitchen egocentric video. The accepted EPIC-KITCHENS path reads a
complete source video, estimates a camera timeline, detects motion onset and
offset, applies endpoint and reconstruction-quality gates, and then creates
new clip boundaries. It does not use VITRA episode boundaries to choose clip
starts or ends.

The implementation supports two kitchen sources:

- EPIC-KITCHENS long videos with DA3-Small or EPIC Fields camera timelines.
- The approved seven-task EgoDex kitchen subset for the earlier native-pose
  path.

This is code only. Source videos, reconstructed pose arrays, model weights,
credentials, and generated media are intentionally excluded.

## Core modules

- `src/lime_pipeline/long_camera.py`: dense timeline interpolation, angular
  speed, hysteresis onset/offset mining, endpoint motion metrics, and seeded
  candidate selection.
- `scripts/estimate_long_camera_da3.py`: full-video DA3-Small inference using
  overlapping windows and the released DA3-Streaming Sim(3) alignment routine.
- `scripts/build_long_camera_demos.py`: timeline export, candidate gating,
  context padding, and clean/review clip rendering.
- `scripts/verify_long_camera_demos.py`: MP4 decode, codec, faststart, frame
  count, RGB endpoint, and DA3/EPIC Fields comparison checks.
- `src/lime_pipeline/kitchen_epic.py`: EPIC camera timeline plus frame-indexed
  VITRA hand observations with explicit support and scale gates.
- `src/lime_pipeline/kitchen_epic_native.py`: native 60 FPS EPIC path with
  exact source-frame indexing and hand support checks.
- `scripts/kitchen_fields_native_mapped.py`: EPIC Fields registered-frame to
  native source-frame mapping with retained provenance.

## Pipeline contract

The documented scope and limitations are in `docs/`. In particular, read
`docs/kitchen_scope_audit_20260920.md` and `docs/long_camera_pipeline.md`
before running a dataset job. The long-video pilot is a calibration set, not a
completed kitchen dataset, and missing hand coverage on long EPIC clips remains
an open limitation.

## Installation

Use Python 3.11 or newer and install the package in an isolated environment:

```bash
python -m pip install -e .
```

DA3 inference additionally needs a local Depth Anything 3 checkout and a
locally available DA3-Small checkpoint. The Slurm files are examples from the
original cluster environment; update paths, partition names, and resource
limits for the target cluster before submitting.

## Tests

The deterministic geometry and native-camera tests can run without videos or
model weights:

```bash
python -m pytest tests/test_long_camera.py tests/test_camera_quality.py \
  tests/test_dense_actions.py tests/test_kitchen_epic.py \
  tests/test_hand_camera_calibration.py
```

## Provenance

The source was extracted from the Youyi & Sen `lime-local` project. The
pipeline follows the project records in the accompanying scope and long-video
documents. Depth Anything 3, EPIC Fields, VITRA, and WiLoR remain external
dependencies with their own licenses and release terms.
