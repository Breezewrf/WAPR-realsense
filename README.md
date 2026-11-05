# WAPR-realsense

[中文](README.zh-CN.md)

RealSense RGB-D → one object's 6D pose, powered by [WAPR](https://github.com/WangYuLin-SEU/WAPR).
Initialize by manual selection or optional automatic detection, then track and publish its pose in the color-camera frame.
The application targets robotics workflows with one known metric mesh and one object instance.

- RealSense D455 depth-to-color alignment, sensor-derived meter scale and pinhole rectification.
- Manual ROI or optional GroundingDINO → SAM2 → DINOv2 CAD-template initialization.
- TensorRT by default; ROI mode does not load detection models.
- Live 3D box/axes, atomic `latest.json`, per-frame JSONL and RGB-D snapshots.
- Configurable frame names, `R`, translation in meters and quaternion in `xyzw` order.

ROI mode requires selection again after loss (`r`). Auto mode searches at startup and after a rejected
tracking update; `r` restarts the search. A matching metric mesh is required in both modes.

## Quick start

The exercised setup is Linux, Python 3.10, an RTX 4090 Laptop GPU and a RealSense D455.
From the repository root:

```bash
uv sync
source .venv/bin/activate
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python -m wapr_realsense --config configs/box.yaml
```

On first use, prepare the system CUDA toolkit, C++ compiler and EGL/GL development headers, then run:

```bash
python scripts/setup_runtime.py
python -m wapr.export_engines
```

`uv sync` installs Python dependencies; these steps prepare weights, the renderer and local TensorRT engines.
With a working cache, run the application directly.

`configs/box.yaml` references the operator's local mesh at
`assets/meshes/box/textured_mesh_modified.obj`. Supply your mesh at that path or edit
the configuration; source exports do not bundle it. Paths inside a config are relative to that file.
Explicit CLI arguments override config values.

For the JetPack 5.1.1 / Python 3.8 source runtime, use `uv pip` instead of the desktop
`uv sync` above. Install the matching Jetson GPU packages first. The operator has verified
CUDA operations and NMS with torch `2.1.0a0+41361538.nv23.06`, torchvision `0.16.2+c6f3977`,
and Builder creation with TensorRT `8.5.2.2`. In that same activated `.venv`, continue with:

```bash
uv pip install -r requirements-jetson.txt
uv pip check
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python scripts/setup_runtime.py
python -m wapr.export_engines
python -m wapr_realsense --config configs/box.yaml --init roi --backend trt
```

The requirements retain those GPU versions and constrain compatible NumPy/Kornia versions;
TensorRT uses the existing system installation. The desktop `pyproject.toml` and `uv.lock`
still target Python 3.10+. Docker needs the NVIDIA runtime, camera USB access and NVIDIA
EGL/GL libraries; interactive ROI selection also needs display forwarding. Rebuild the native
renderer and engines on Orin instead of copying desktop `.so` or `.engine` files. ARM64
engine builds default to a 1 GiB workspace limit; the operator built `wapr_w_mask` with
`WAPR_TRT_WORKSPACE_GB=4` after 1 GiB failed. This is a scratch-space budget for algorithm
selection, not a total process memory cap; selected algorithms also need scratch space at runtime.
The operator has verified D435I capture, native rendering and torch ROI tracking (~503–518 ms/frame).
TensorRT pose inference still needs on-device validation. TRT 8.5 exports use opset 16 to decompose
LayerNorm into basic operations. Without display forwarding, use `--no-display --bbox X Y W H`
with coordinates from the current rectified image; save one with
`--check-camera --output outputs/camera_check`. This Python 3.8 path covers ROI tracking;
optional SAM2 and other detector dependencies are outside this installation list.

TRT mode loads engines directly without PyTorch network weights. Pose engines share one
execution-memory buffer sized to their largest requirement, with serialized submissions;
weights and persistent context state remain separate. Existing engines need no rebuild.
Check all four engines together with `python scripts/check_trt_memory.py --run` (synthetic
12/6-candidate inputs, not an accuracy test), then run the application with `--backend trt`.
`WAPR_TRT_CONTEXT` reports each requirement; `WAPR_TRT_MEMORY` reports the allocated shared buffer.
This memory path needs Jetson validation. `WAPR_TRT_SHARED_MEMORY=0` restores separate context
allocations for comparison and may exhaust memory on Orin NX 16GB.

Drag a tight box in the frozen image and press Enter/Space. In the live window:
`r` reinitializes, `s` saves RGB-D and pose, and `q`/Esc exits.
The default camera profile is 640×480 at 15 Hz; capture rate is not inference throughput.

Use another mesh without a config:

```bash
python -m wapr_realsense --mesh /absolute/path/model.obj --mesh-unit m --backend trt
```

Headless ROI operation requires an initial box in rectified color-image pixels:

```bash
python -m wapr_realsense --config configs/box.yaml --no-display --bbox 180 100 280 300
```

## Optional automatic initialization

```bash
python -m wapr_realsense --config configs/box.auto.yaml
# Or enable it for any existing target configuration, including headless operation:
python -m wapr_realsense --config configs/box.yaml --init auto --no-display
```

Auto mode uses the existing WAPR detector: GroundingDINO proposes boxes, SAM2 segments them,
and DINOv2 matches candidates against rendered views of the configured mesh. WAPR estimates
poses from the detected masks; the highest `score_6d` candidate passing `--min-score` is selected.
No reference photo is used. Multiple identical instances cannot be distinguished by physical identity.
Only search frames run detection; accepted poses use the existing six-candidate tracker.

YAML configurations group detection options under `groundingdino_sam2_dinov2_detect`:

```yaml
groundingdino_sam2_dinov2_detect:
  activate: true       # false selects manual ROI initialization
  use_sam2: false      # true enables SAM2; false uses bbox input
  interval: 1.0
  score: 0.1
  backend: trt
  dino: vitl14
  grounding: swinb
  template_cache: ../.wapr_runtime/templates/box_vitl14.pt
```

| Field within the section | Default | Purpose |
| --- | --- | --- |
| `activate` | `false` | Enable automatic initialization and detection after tracking loss |
| `use_sam2` | `true` | Use SAM2 masks; `false` uses bbox regions for matching and bbox input for WAPR |
| `interval` | `1.0` | Seconds after a failed search completes before the next attempt |
| `score` | `0.1` | 2D template-match acceptance gate, separate from top-level `min_score` |
| `backend` | `trt` | DINOv2 encoder backend; `torch` also supported |
| `dino` | `vitl14` | `vits14`, `vitb14` or `vitl14` |
| `grounding` | `swinb` | `swint` or `swinb` |
| `template_cache` | unset | Optional `.pt` cache; unset builds templates in GPU memory each run |

Configuration files use `.yaml` or `.yml`; detection options belong in the section above.
Explicit CLI flags override config values: `--init auto` / `--init roi` overrides `activate`,
and `--detect-use-sam` / `--no-detect-use-sam` overrides `use_sam2`. For example:

```bash
python -m wapr_realsense --config configs/box.auto.yaml --no-detect-use-sam
```

This skips SAM2 model loading, inference and checkpoint downloads. GroundingDINO and DINOv2
still run. Rectangle regions support DINO cropping and duplicate suppression; the pose estimator
receives `bbox_xywh` without a mask and uses `wapr_wo_mask`. More background may enter matching
and depth initialization, so detection scores and accuracy should be checked again. SAM2 timing
is shown as `skipped`. Use `--detect-use-sam` to override a disabled config.

Auto mode prepares optional detector dependencies and downloads missing weights on first use;
the pose-only setup script does not preload them. The DINO TensorRT backend builds its engine
when needed; GroundingDINO and SAM2 remain FP32. Expect additional startup time and GPU memory.
An incompatible environment is reported by the existing optional-dependency bootstrap.
The sample cache is relative to its configuration file, under `.wapr_runtime/templates/` rather than
`.wapr_runtime/cache/`. Mesh/texture, encoder or source changes can invalidate the cache.
The application automatically selects a versioned sibling path, rebuilds once and reuses it on later
runs; old files are preserved. `DET2D_CAD_CACHE_CHANGED` prints the selected path. No directory
deletion is needed. Setting `template_cache` to `null` disables the disk template cache.

Output status is `searching` with `valid=false` until initialization succeeds, then `initialized_auto`
and `tracking`. A rejected track emits `lost` without a pose, followed by a new search on the next frame.
Passing scores do not detect every drift or guarantee the selected object's identity.
The first call includes three detector warmup passes and pose warmup;
use later searches when assessing recovery latency. Search runs synchronously and pauses pose output.

The terminal displays three groups in milliseconds:

- **Yellow `[WARMUP]`**: after the initial detector warmup, reports total time, average time
  and call count for GroundingDINO, SAM2 and DINOv2. The first valid candidate batch also
  reports total and average time for WAPR pose initialization warmup.
- **Green `[TRACKING]`**: reports WAPR mean time and frame count every 5 seconds by default.
  Loss, reinitialization and exit flush any remaining samples. Only tracking updates are
  included, including rejected updates and first-call overhead. Set `--timing-interval 2`
  to change the reporting interval.
- **Cyan `[INIT]` / `[REINIT]`**: announces the first search and subsequent searches, then
  reports GroundingDINO, SAM2, DINOv2 and WAPR pose initialization times, detection total,
  complete initialization total, candidate counts and search result. The first complete total
  includes warmup; later searches reuse completed warmup. Stages that do not run show `skipped`.

GroundingDINO includes image preprocessing, inference and box filtering; SAM2 includes segmentation
and mask postprocessing. DINOv2 includes proposal crop preprocessing and encoding across all batches,
excluding template matching. Detection total also includes matching and postprocessing. WAPR tracking
includes the complete update and validity check; pose initialization evaluates all valid candidates
before selecting a target. Values are wall-clock milliseconds including CPU work and GPU waits,
with CUDA synchronized before and after each stage. These extra synchronizations can affect stage
overlap. Colors are enabled automatically in a terminal; redirected output or `NO_COLOR=1` uses plain text.

## Browser demo

```bash
python -m wapr_realsense --config configs/box.auto.yaml --web
# Start in manual mode, without loading automatic detection models:
python -m wapr_realsense --config configs/box.yaml --web
# LAN access on Jetson:
python -m wapr_realsense --config configs/box.auto.yaml --web --web-host 0.0.0.0
```

Open `http://127.0.0.1:8000` locally, or `http://<jetson-ip>:8000` over the LAN.
The camera view shows the 3D box and axes. A small dashboard on the right contains
the status, manual initialization switch, Detect button and timing summaries. All static resources ship with
the Python package; Node, external CDNs and a desktop display are unnecessary.

- **WARMUP**: load and warm up models; blank preview and disabled controls. All warmup results are discarded.
- **INITIALIZED**: camera preview is available; wait for the user to click **Detect**.
- **Looking for object**: initial detection or automatic recovery after tracking loss. Detect is disabled while searching.
- **TRACKING**: show the pose overlay. Clicking Detect again starts a new search.

The switch initially follows the configured initialization mode. Enable manual
initialization to freeze a frame, drag a bounding box, then click Detect. The
redraw button obtains a new frozen frame. Pixel coordinates are mapped to the
original image and tied to that exact RGB-D frame. Successful tracking restores
the live view. **Loss in manual mode clears the old box and prompts the user to
draw a new one; it never searches automatically or reuses the old box.** Automatic
mode retains search and recovery. Warmup and search disable the switch. Switching
from manual to automatic loads and warms detection models on demand under WARMUP,
then waits for Detect. Changing modes resets the previous track.

The dashboard receives the same `[WARMUP]`, latest `[INIT]` / `[REINIT]`, and
`[TRACKING]` summaries as the terminal. Tracking means and frame counts use
`--timing-interval`. Warmup details collapse when ready and remain available.
SAM2 / DINOv2 show `skipped` when no proposals exercise those stages. Pose and
tracking warmup use representative single-instance workloads; different real
candidate batch sizes may still incur additional first-use overhead.
During synchronous searches, the preview holds the search input frame while
HTTP status requests remain responsive. Browser disconnection hides the old
preview and disables controls; reconnection restores the current session.

To inspect the UI without a camera or CUDA, use the explicitly labelled simulation:

```bash
python -m wapr_realsense --web-demo
```

The simulation warms up for about 2 seconds and supports both modes; it defaults
to manual drawing. It finds an object about 2 seconds after Detect and simulates
loss after about 12 seconds of tracking, then waits for a new box or recovers
automatically according to the selected mode. Real operation preserves the pose
file output contracts and adds `initialization_mode` to each frame record.
Jetson inference has not been validated here.

## Pose output

The box config writes `outputs/box/latest.json`, `poses.jsonl`, `camera.json` and `session.json`.
`latest.json` is replaced atomically. Invalid/stopped records contain `valid=false` and no pose.
The reader checks validity and freshness, rejecting data older than 500 ms by default (`--max-age-ms`).
Complete JSON records are saved for every frame. The terminal shows compact timing by default;
add `--log-poses` to also print `POSE_JSON` every ten processed frames, or combine it with
`--print-every 1` to print every frame.

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
Detection and SAM sources support auto mode; ROI mode does not load them.

## Validation and deployment status

- D455 640×480/15 Hz capture and TensorRT initialization/tracking were exercised on the desktop.
- The operator reported good results using the local box and ROI workflow before this reorganization.
- CPU tests cover depth conversion/rectification, configuration, loss state and robotics-output contracts.
- Auto-mode tests cover CLI/replay wiring, retry timing, candidate selection and reacquisition with
  mocked inference. Real three-model detection, startup memory and live auto-mode accuracy remain unverified.
- JetPack 5.1.1 checks cover basic GPU operations, D435I capture and renderer compilation/initialization;
  the complete pose pipeline remains unverified there.
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
