# Fixed training interface

Each record describes one newly cropped kitchen video and an aligned sequence:

`video + task_prompt + observed state at t_i -> camera action 9D + manipulation action 7D over [t_i, t_i+1]`

The final manifest uses `schema_version: lime.kitchen.training.v1`. Paths are
relative to the dataset root and every media/array artifact has a SHA256 hash.
Both cohorts use the identical interface and quality thresholds.

## Inputs

- `input.video`: clean H.264 MP4 crop with original FPS and consecutive native
  source frames. Audio is omitted. Width is 1,280 pixels; aspect ratio is kept.
- `input.task_prompt`: concise English instruction generated from the actual
  crop and checked separately. These are model-generated labels, not human
  ground truth. Staged EgoDex kitchen fixtures remain identified as staged.
- `input.state`: descriptor referring to state arrays in the associated NPZ.
  States are sliced `[0, T)`, where T is the number of actions. They contain
  camera pose 9D and both hand poses/gripper widths with explicit validity masks.

The reader returns `state.camera_pose_9d` with shape `[T,9]` and
`state.left_pose_gripper_10d` / `right_pose_gripper_10d` with shape `[T,10]`.
Each hand state concatenates position xyz in metres, the first and second
rotation-matrix columns (three values each), and gripper width in metres.
All states use the fixed clip-start camera coordinate frame C0. Hand absence is
NaN plus a false `left_valid` or `right_valid` mask, never a stationary zero pose.

These are offline reconstructed and temporally denoised states; the processing
can use surrounding frames. They are not a claim of causal online estimation.
For policy training, observation RGB history at step i must stop at t_i. Do not
feed the complete future clip as the observation for an earlier step.

## Targets

- `camera_action_9d[T,9]` is `inv(camera_i) @ camera_i+1`, encoded as translation
  xyz followed by the first two rotation-matrix columns. Translation is expressed
  in the current camera's body frame. The last six values are not Euler angles.
- `manipulation_action_7d[T,7]` is the selected primary hand's fixed-C0 position
  delta (3), spatial rotation-vector delta in radians (3), and current normalized
  gripper opening (1). The last channel is a state, not an opening delta.
- `camera_action_valid[T]` and `manipulation_action_valid[T]` accompany targets.
  Invalid hand steps remain masked NaNs. Both hands' original targets remain in
  the NPZ; the manifest identifies the selected primary hand.
- `action_dt_s[T]` contains the actual duration of every step. Normal sampling is
  10 Hz for 30/50 FPS input and approximately 9.990 Hz for 60000/1001 FPS input.
  A shorter final interval keeps its actual duration.

The on-disk NPZ retains T+1 states to permit independent target recomputation.
The loader exposes only the first T as inputs and creates no padded last action.
The arrays also retain exact original source-frame indices, relative/source
seconds, native observation masks, source intrinsics, and reconstruction status.

EgoDex units come from native metric ARKit estimates. EPIC camera translation
uses explicitly audited estimated metric calibration; hand geometry comes from
WiLoR RGB reconstruction. These are estimated trajectories, not motion-capture
measurements or robot-ready controls. Real deployment requires the robot's
coordinate calibration, gripper mapping, and control policy.

## Cohort construction

Kitchen sources are EPIC-KITCHENS complete recordings and the historical seven
EgoDex kitchen categories. Source assignment is frozen before hand QC, and the
same source recording cannot occur in both cohorts.

The camera-motion group uses complete camera timelines and the accepted
onset/offset miner: 20/8 degrees per second, endpoint net rotation >=30 degrees
AND view-axis change >=20 degrees, 0.75–8 s core, up to one second of context on
each side, and the original reconstruction gates. Action/VITRA episode bounds
never determine clip boundaries. Camera motion proposes candidates; it does not
prove active-perception intent. Semantic perception labels are retained separately.

Random candidates are independently sampled 3–10 s windows without a minimum
motion or active-perception criterion. Both groups undergo the same camera,
hand, image, kitchen/task, overlap and duplicate checks. Rejection rates and
length distributions are reported; the two duration distributions need not match.
Final dataset counts are matched between cohorts, with exactly 500 clips each.

## Reading a certified release on h200

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/nfs_data/zhangchi/zhangchi/lime-local/scripts')
from kitchen_training_contract import load_training_arrays

root = Path('/mnt/node04/zhangchi/data/lime-local/kitchen-1000-v1')
assert (root / 'COMPLETE.json').exists()  # absent while this build is incomplete
row = json.loads((root / 'training.jsonl').read_text().splitlines()[0])
sample = load_training_arrays(root, row)
print(sample['task_prompt'])
print(sample['state']['camera_pose_9d'].shape)
print(sample['camera_action_9d'].shape, sample['manipulation_action_7d'].shape)
```

The final reader checks exact 500/500 counts, source disjointness, matched source
counts, original media/trajectory provenance, state/action alignment, timebases,
semantic evidence and hash-bound visual review. A completion certificate is issued
only after those checks and the source-hand reader pass on the actual final set.
