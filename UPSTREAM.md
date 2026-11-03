# Source provenance / 上游来源

WAPR-realsense is a robotics application derived from
[WAPR by Yulin Wang](https://github.com/WangYuLin-SEU/WAPR).
The vendored `wapr/` backend comes from commit
`f165cec8ebdd48ae6f46d06a46c40d796ab80b71`.

The original WAPR code and example authorship remains with its respective authors.
Original copyright headers, `LICENSE` (LGPL-2.1-only), `AUTHORS.md`,
`THIRD_PARTY_NOTICES.txt`, font notices and weight-license notices are retained.
The archived upstream [English](docs/upstream/README.md) and
[Chinese](docs/upstream/README.zh-CN.md) READMEs describe the original research release.
WAPR-realsense is an application adaptation; it does not claim to be the upstream release.

## Changes in this application

- Added the `wapr_realsense/` package, D455 capture, color-aligned metric depth,
  SDK-model rectification, manual ROI initialization and a live tracking loop.
- Added atomic JSON output, timestamps, validity state, quaternion output,
  configurable frame names, snapshot/replay support and a robotics reader example.
- Added single-target configs, source-export tooling, installation notes and CPU tests.
- Replaced the root research README with application documentation; archived the original.
- Kept `examples/realsense_pose.py` and `wapr/realsense.py` as compatibility shims.
- Kept the WAPR network definitions, refinement/scoring rules, engine profiles,
  checkpoint formats and existing model-weight contents unchanged.

These additions and application modifications are distributed under LGPL-2.1-only.
This document does not reassign copyright in upstream code or operator-supplied assets.

## Model weights

The four original WAPR checkpoints are downloaded from `SEU-WYL/WAPR`, not embedded
in this application's source export. Their separate terms are preserved in
[WEIGHTS_LICENSE.txt](WEIGHTS_LICENSE.txt) and `wapr/weight_license/`.
Weights, generated TensorRT engines, captured data and private object meshes are excluded
from the generated GitHub source directory.

## Citation

```bibtex
@InProceedings{wang2026wapr,
  author = {Wang, Yulin and Hu, Mengting and Li, Hongli and Zhou, Jianghao and Luo, Chen},
  title = {WAPR: A Foundation Model for Wide-Angle Refinement in Unseen Object Pose Estimation},
  booktitle = {ECCV},
  year = {2026},
  pages = {221-239},
  volume = {17016}
}
```
