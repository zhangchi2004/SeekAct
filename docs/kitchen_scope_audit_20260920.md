# Kitchen pipeline scope audit — 2026-09-20

## Conclusion

The user's objection is correct. `scale-2000-v2` and the September 19 demo use
general EgoDex data and do not satisfy the kitchen-only scope. The assistant
chose these sources to obtain complete hand/camera pairing when the longer
EPIC clips exceeded the available VITRA hand-track support. That engineering
constraint did not authorize changing the domain or replacing the approved
camera-first pipeline. A question without a reply was not agreement.

## Evidence from the h200 task “Youyi & Sen”

Task ID: `01a08109-67eb-7ac0-b3db-7ff8351b5d9f`.

- In task turn `01a05370-5ef0-7061-8a5b-3f84bdc79a24`, the user explicitly said:
  “I would like to fix the data to kitchen scenes.” The subsequent kitchen demo
  used 20 randomly chosen EgoDex kitchen videos, not the full task inventory.
- In turn `01a0567c-4d98-76c0-b181-e255d30e9811`, the user requested EPIC-KITCHENS
  demos with hand and camera motion. The early EPIC/VITRA path reused VITRA
  episode boundaries; this was later identified as a limitation.
- In turn `01a08129-1a00-7043-9996-c32f8d46abb4`, the user requested complete
  camera-motion annotation of long EPIC videos and a pipeline that recuts
  large-camera-motion segments. This produced 18 demos.
- In turn `01a08148-a2ce-7490-afa9-9d9d66e7e170`, the user said the new videos
  looked good and requested manual-style content/action/perception/gain labels.
  The resulting 18 reviewed clips and 330 evidence frames are still present.

## Two kitchen sources and the distinction between versions

1. **EgoDex kitchen subset (earlier pilot):** seven allowed task directories:
   `boil_serve_egg`, `fry_bread`, `fry_egg`, `stock_unstock_fridge`,
   `wash_kitchen_dishes`, `wash_put_away_dishes`, `wipe_kitchen_surfaces`.
   Root: `/mnt/nfs_data/zhangchi/data/lime-local/demo-kitchen-random-qwen3vl`.
   It contains 20 demos, with native metric camera poses and motion mining
   within the supplied EgoDex source videos. EgoDex source videos are already
   task recordings; this is not the same as unrestricted long EPIC recordings.
2. **EPIC-KITCHENS long-video pipeline (last accepted pilot):** complete RGB
   source video → DA3-Small / EPIC Fields camera timeline → motion onset/offset
   detection → endpoint view-change and reconstruction-QC gates → new clip
   boundaries with context → semantic review. VITRA is a source of reconstructed
   camera/hand metadata in the earlier EPIC branch, not a separate RGB kitchen
   dataset; EPIC Fields supplies reconstructions for EPIC videos.

The long-video policy uses 20°/s onset, 8°/s offset, net rotation ≥30° AND view-axis
change ≥20°, 0.75–8 s core, and up to 1 s context on each side. DA3 estimates
poses at 5 FPS, interpolates only short valid gaps to source-frame times, and
rejects reconstruction joins/disagreements. Translation has reconstruction
scale and is not gated in metres. Source frames and core/context bounds are
retained. These facts were checked against the actual thresholds, summaries,
source code and stored source-video metadata on h200.

## Actual preserved EPIC artifacts

Under `/mnt/node04/zhangchi/data/lime-local/`:

- `epic-long-camera-da3-v1/`: full-video camera estimates for P01_103, P03_118,
  P04_114 and P09_106; source durations total 613.54 s.
- `demo-epic-long-camera-da3-v1/`: 8 selected clips from those four sources.
- `demo-epic-long-camera-epic_fields-v1/`: 10 selected clips from P01_103.
- `demo-epic-long-camera-manual-v1/`: all 18 annotations and evidence. Counts:
  2 supported, 11 ambiguous, 5 not supported for active-perception intent.
- `demo-epic-long-camera-review-v1/`: original linked inspection pages.

The two backends overlap in source material. These are 18 demo clips, not 18
independent videos or a completed 2,000-example kitchen dataset. Full long-clip
hand actions were not generated in this accepted pilot; that work remains.

## What the incorrect scale run actually contains

Recounted directly from `delivery/training.jsonl`:

- All 2,000 records: EgoDex; EPIC-KITCHENS records: 0.
- Task categories: 101; records matching the seven kitchen categories: 74.
  This is an allowlist comparison, not a new visual kitchen classification.
- Of 1,000 camera-motion records, 991 use the metric velocity/fixed-window
  miner and 9 use hysteresis onset/offset. Only 67 of 1,000 satisfy the original
  ≥30° rotation AND ≥20° view-axis gate, regardless of selected method.
- The other 1,000 records use random windows. The experiment does re-crop
  source videos: the issue is not that all outputs are untouched native clips.
  The issues are source/domain drift and the changed candidate policy.

Artifact checks and hashes remain valid evidence about this experiment's
internal consistency. They do not establish compliance with the user's scope.
Existing media, NPZ arrays, annotations, manifests and certificates are kept.

## Recovery completed in this audit

- Corrected project, delivery and scaling documentation to withdraw the
  kitchen-task completion claim.
- Marked the general EgoDex demo as out of scope and linked the accepted pilot.
- Restarted the existing EPIC review server and Cloudflare connector, without
  changing the original clips or labels.
- Restored page: https://waves-expense-frequencies-caution.trycloudflare.com/annotations/index.html

For future scaling, keep kitchen source selection explicit and use the accepted
full-video EPIC camera-first implementation. Resolve hand-track support and
cross-reconstruction coordinate/scale alignment explicitly; do not fill missing
hands or substitute non-kitchen sources to reach a requested count. Revalidate
scope, temporal segmentation, modalities and provenance before claiming that
the requested kitchen release is complete.
