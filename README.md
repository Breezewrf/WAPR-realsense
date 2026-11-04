# WAPR-realsense

[中文](README.zh-CN.md)

RealSense RGB-D → one object's 6D pose, powered by [WAPR](https://github.com/WangYuLin-SEU/WAPR).
Select the object once, then track it continuously and publish its pose in the color-camera frame.
The application targets robotics workflows with one known metric mesh and one object instance.

- RealSense D455 depth-to-color alignment, sensor-derived meter scale and pinhole rectification.
- Manual ROI initialization, existing WAPR/SAPR refinement and WBPS candidate scoring.
- TensorRT by default; no DINOv2, SAM2 or GroundingDINO models in the application path.
- Live 3D box/axes, atomic `latest.json`, per-frame JSONL and RGB-D snapshots.
- Configurable frame names, `R`, translation in meters and quaternion in `xyzw` order.

This application tracks an initialized target. It does not automatically find or reacquire objects.
Press `r` to select the target again after a loss. A matching metric mesh is required.

## Quick start

The exercised setup is Linux, Python 3.10, an RTX 4090 Laptop GPU and a RealSense D455.
From the repository root:

```bash
uv sync
source .venv/bin/activate
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python -m wapr_realsense --config configs/box.json
```

On first use, prepare the system CUDA toolkit, C++ compiler and EGL/GL development headers, then run:

```bash
python scripts/setup_runtime.py
python -m wapr.export_engines
```

`uv sync` installs Python dependencies; these steps prepare weights, the renderer and local TensorRT engines.
With a working cache, run the application directly.

`configs/box.json` references the operator's local mesh at
`assets/meshes/box/textured_mesh_modified.obj`. Supply your mesh at that path or edit
the configuration; source exports do not bundle it. Paths inside a config are relative to that file.
Explicit CLI arguments override config values.

Drag a tight box in the frozen image and press Enter/Space. In the live window:
`r` reinitializes, `s` saves RGB-D and pose, and `q`/Esc exits.
The default camera profile is 640×480 at 15 Hz; capture rate is not inference throughput.

Use another mesh without a config:

```bash
python -m wapr_realsense --mesh /absolute/path/model.obj --mesh-unit m --backend trt
```

Headless operation requires an initial ROI in rectified color-image pixels:

```bash
python -m wapr_realsense --config configs/box.json --no-display --bbox 180 100 280 300
```

## Pose output

The box config writes `outputs/box/latest.json`, `poses.jsonl`, `camera.json` and `session.json`.
`latest.json` is replaced atomically. Invalid/stopped records contain `valid=false` and no pose.
The reader checks validity and freshness, rejecting data older than 500 ms by default (`--max-age-ms`).

```bash
python examples/read_pose.py --input outputs/box/latest.json
```

The transform maps the original, unit-scaled mesh frame into the rectified color-camera optical frame:

```text
p_camera = R @ p_mesh + t_m
T_base_object = T_base_camera @ T_camera_object
```

Camera axes are X right, Y down, Z forward. A robot-base pose requires measured camera calibration.
WBPS scores are ranking signals, not success probabilities; occlusion, symmetry, depth holes and mesh
scale mismatch can cause drift or incorrect poses even when a score passes the gate.

## Project layout

```text
wapr_realsense/  camera, tracking session, CLI, geometry and robotics output
wapr/            WAPR inference, optional detection/SAM modules and original notices
configs/         single-target and camera-calibration templates
examples/        robot-side pose reader
scripts/         pose runtime setup
tests/           CPU contract tests; optional SDK geometry test
assets/meshes/   local object assets, excluded from Git
```

Environments, caches, weights, engines and local meshes are excluded by `.gitignore`.
Detection and SAM sources are retained for optional use; the ROI tracking application does not load them.

## Validation and deployment status

- D455 640×480/15 Hz capture and TensorRT initialization/tracking were exercised on the desktop.
- The operator reported good results using the local box and ROI workflow before this reorganization.
- CPU tests cover depth conversion/rectification, configuration, loss state and robotics-output contracts.
- Validation currently covers only the desktop setup above.
- The backend's general engine profiles and model loading are unchanged; this is not yet the proposed
  single-target low-memory engine optimization.

Run tests with `python -B -m unittest discover -s tests -v`.

## Attribution and license

WAPR-realsense is an application adaptation of WAPR, not an official upstream release.
The inference core derives from WAPR commit `f165cec8ebdd48ae6f46d06a46c40d796ab80b71`.
This application adds RealSense acquisition and a live pose interface while retaining the upstream backend modules.
Source retains [LGPL-2.1-only](LICENSE); model weights retain their separate
[weight license](WEIGHTS_LICENSE.txt). Preserve [authors](AUTHORS.md),
[third-party notices](THIRD_PARTY_NOTICES.txt) and bundled license files.
