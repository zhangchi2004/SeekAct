# Source attribution and terms

This combined package does not apply a new blanket license to third-party data.

## EPIC-KITCHENS

Source: https://epic-kitchens.github.io/2026

The official page states that its datasets and benchmarks are provided under
Creative Commons Attribution-NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/

Credit the EPIC-KITCHENS authors and dataset. Changes here include temporal
cropping, resizing/re-encoding, camera-motion selection, estimated hand/camera
trajectories and generated task annotations. No endorsement is implied.

## Apple EgoDex

Source: https://github.com/apple/ml-egodex

The official README's License section states: “The dataset is available under
CC-by-NC-ND terms.” The README does not specify a version in that statement.
The source repository's code license is separate from the dataset license.

Citation: Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian
Zhang, *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric
Video*, 2025, https://arxiv.org/abs/2505.11709.

This package uses the seven historical kitchen task categories. The supplied
clips have been temporally cropped and re-encoded and include derived targets.
No permission to redistribute adapted EgoDex media is asserted by this package.
Keep the full bundle private unless the appropriate redistribution rights have
been established; recipients remain responsible for following the source terms.

## Reconstruction and annotation provenance

- EPIC Fields: https://github.com/epic-kitchens/epic-fields-code
- VITRA: https://github.com/microsoft/VITRA
- WiLoR: https://github.com/rolpotamias/WiLoR
- Qwen: https://github.com/QwenLM

Model checkpoints, MANO assets and upstream source-video collections are not
included. Per-sample provenance identifies the source and reconstruction method.
Generated annotations are pseudo-supervision; rights in underlying media are
unchanged. Source terms were checked on 2026-09-21.
