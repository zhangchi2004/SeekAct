# Long-video EPIC camera segmentation — first runnable release

This is a separate camera-first path. It never uses VITRA episode boundaries or
EPIC action intervals to determine segment starts/ends. Candidates have **not**
been verified as active perception or camera-driven observation gain.

## Implemented path

1. Read the complete source MP4, retain its FPS, frame count, and zero-based
   source-frame indices. The first pilot uses P01_103, P03_118, P04_114, P09_106
   (613.54 seconds total). These four inputs are a fixed pilot/calibration set,
   not a statistically random sample of all EPIC-KITCHENS.
2. Estimate poses using the locally downloaded DA3-Small checkpoint at 5 FPS,
   including the final source frame, in 60-frame windows with 30-frame overlap.
   Call the released DA3 API and the released DA3-Streaming robust dense Sim(3)
   alignment routine. Upper-image, high-confidence overlap points provide
   alignment correspondences. This is an adaptation without loop closure, not
   a claim to run every stage of the upstream streaming/SLAM system.
3. Invert world-to-camera transforms to camera-to-world. Keep per-window
   overlap rotation disagreement and mark window joins. The global origin,
   orientation, and translation scale are reconstruction-defined.
4. Create a row for every original frame. Use SO(3) SLERP and linear camera-center
   interpolation only across gaps <=0.24 s with plausible implied rotation.
   Other rows stay invalid/NaN. At 50 FPS, most rows between 5 FPS model samples
   are interpolations, not independently estimated per-frame poses.
5. Compute angular speed over a centered 0.20 s window and a 3-sample median
   filter. Hysteresis starts at 20 deg/s and ends below 8 deg/s; bridge pauses
   <=0.25 s only when poses remain valid. Longer continuous runs use explicitly
   marked bounded windows, not action/episode boundaries.
6. Require net rotation >=30 deg **and** viewing-axis change >=20 deg. Core
   duration is 0.75–8 s, peak angular speed <=400 deg/s. Reject intervals whose
   overlap disagreement exceeds 8 deg or that cross a reconstruction join.
   Pure-roll and out-and-back motion with small endpoint change are not accepted.
   Translation is exported in arbitrary reconstruction units, but the first
   release deliberately does not threshold it or label it as metres.
7. Check endpoints at 640x360: Laplacian variance >=30 and usable exposure pixel
   fraction >=0.70. RGB difference is not an observation-gain label.
8. Select eligible candidates using seed 20260908, per-video caps, and maximum
   clip temporal IoU 0.20. The requested count is a cap; never weaken thresholds
   to fill a quota. Add up to 1 s context on each side. Export clean RGB and
   annotated review clips as H.264 avc1/yuv420p with faststart, plus contact sheets.

The EPIC Fields backend uses its published registered poses instead of the DA3
estimator, with the same motion miner. Filename frame_0000000001.jpg maps to
zero-based MP4 frame 0. Missing registered poses are never represented as known
stationary camera poses. The recovered P01_103 JSON is complete, although its
parent download archive remains partial. Only complete JSON members were used.

## Artifacts

Root: `/mnt/node04/zhangchi/data/lime-local/`

- `epic-long-camera-da3-v1/<video>/camera_samples.npz`: actual sampled source
  frames, camera-to-world [N,4,4], intrinsics [N,3,3], depth-scale statistics,
  alignment errors, and reconstruction joins.
- `epic-long-camera-da3-v1/<video>/estimation.json`: model/window settings,
  per-window quality, source video metadata and wall time.
- `demo-epic-long-camera-da3-v1/`: 8 selected candidates from 4 videos.
- `demo-epic-long-camera-epic_fields-v1/`: 10 selected candidates from 13 eligible
  candidates on P01_103 (the same source also occurs in the DA3 pilot).
- `demo-epic-long-camera-review-v1/`: combined inspection page; old galleries
  are not modified.
- Within each demo, `timelines/<video>/camera_full_timeline.npz` and `.csv`
  cover the full original frame index range, with pose status 0=missing,
  1=supplied estimate, 2=short-gap interpolation. Quaternions in CSV are
  camera-to-world `qx,qy,qz,qw`; translation is not metric.
- `candidates_all.jsonl`: candidates including rejection reasons.
- `segments.jsonl`: selected events with separate core and padded clip bounds,
  exact source frames, relative transforms, rotation/view-axis angles and QC.
- `clips/<event>/clip.mp4`: clean RGB, original FPS, resized to width 1280,
  re-encoded, no audio. `review.mp4`: annotated, up to approximately 25 FPS.
- `thresholds.json`, `summary.json`, `validation.json`: settings, counts and checks.

No full-video hand reconstruction or new VLM caption generation is run here.
Do not reuse old episode hand trajectories outside their valid source-frame
support. The next semantic stage should inspect padded clips and label concrete
visual evidence while retaining an uncertain class.

## Running

Repository: `/mnt/nfs_data/zhangchi/zhangchi/lime-local`.
Use cluster preflight and Slurm; these are not master-node workloads.

```bash
sbatch slurm/epic_long_camera_estimate.sbatch
# After successful estimation:
sbatch slurm/epic_long_camera_render.sbatch da3
# Independent reference path:
sbatch slurm/epic_fields_extract.sbatch
sbatch slurm/epic_long_camera_render.sbatch epic_fields
# After both renders:
sbatch slurm/epic_long_camera_verify.sbatch
```

Source modules: `scripts/estimate_long_camera_da3.py`,
`src/lime_pipeline/long_camera.py`, `scripts/build_long_camera_demos.py`.
Run a changed threshold experiment with a **new** output directory, passing
`--min-rotation-deg` and `--min-view-axis-deg` to the build script. DA3 estimation
is cached per video; re-cutting does not require repeating GPU inference.

## Validation and limitations

The first DA3 pass covers all 30,677 original frame rows; 54 are explicitly
invalid after interpolation safeguards. This is about 99.82% available estimated
or interpolated poses, not 99.82% accuracy. EPIC Fields has 8,019 valid rows out
of 8,901 for P01_103 (90.09%).

Cross-method relative rotation disagreement on P01_103, using endpoint-valid
pairs spaced 0.5 s apart:

| Window | Pairs | Median | P90 | P95 |
| --- | ---: | ---: | ---: | ---: |
| 1 s | 313 | 3.46 deg | 12.98 deg | 17.68 deg |
| 2 s | 310 | 5.00 deg | 15.72 deg | 20.13 deg |

EPIC Fields is a reconstruction reference, not sensor ground truth. These
numbers include all endpoint-valid evaluation pairs, not only accepted segments.
Window joins are deliberately excluded from mining, which reduces recall.
Sustained translation with little rotation is also excluded in this release.
For future translation selection, calibrate scale or use a clearly labeled
baseline-to-depth criterion; do not add an uncalibrated centimetre threshold.

References:
- https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming
- https://github.com/epic-kitchens/epic-fields-code
- https://colmap.github.io/format.html
