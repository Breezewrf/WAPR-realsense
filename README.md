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

Select one mutually exclusive deployment group. Plain `uv sync` defaults to `common`.

```bash
# Laptop: Python 3.10, Torch 2.7.1 / torchvision 0.22.1, CUDA 12.6
uv sync --group common --python 3.10

# Orin NX / JetPack 5.1.1: Python 3.8, verified NVIDIA GPU wheels
uv sync --no-default-groups --group orin-nx-jetpack511 --python 3.8

# AGX Thor: Python 3.12, PyTorch cu130 + ARM64 ONNX Runtime GPU 1.24.0
uv sync --no-default-groups --group thor --python 3.12
```

`common` retains the desktop versions and TensorRT CUDA 12 packages. The Orin group
uses the constraints from `requirements-jetson.txt` and installs its verified Torch and
torchvision wheels automatically. Thor uses the cu130 PyTorch index and the supplied
ONNX Runtime GPU wheel. Neither Jetson group installs TensorRT via pip. Link the system
package after syncing, with matching Python ABIs:

```bash
# Orin / Python 3.8
ln -s /usr/lib/python3.8/dist-packages/tensorrt* .venv/lib/python3.8/site-packages/
# Thor / Python 3.12
ln -s /usr/lib/python3.12/dist-packages/tensorrt* .venv/lib/python3.12/site-packages/
source .venv/bin/activate
python -c 'import torch, tensorrt; print(torch.__version__, torch.cuda.is_available(), tensorrt.__version__)'
# Thor only
python scripts/check_thor.py
```

Do not link Python 3.10 TensorRT extensions into Python 3.8/3.12 environments. System
TensorRT must match JetPack; the current runtime supports Orin's 8.5 and Thor's 10.x,
not 11.x. Repeat the selected group flags on subsequent syncs. The Thor check tests
CUDA math, NMS, TensorRT Builder and ONNX Runtime CUDA execution without CPU fallback;
camera, rendering and pose inference still require Thor hardware validation. ONNX Runtime
is not part of the main WAPR pose backend. Optional detector dependencies remain separate.

```bash
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python scripts/setup_runtime.py
WAPR_TRT_WORKSPACE_GB=4 python -m wapr.export_engines --object-mode single
python -m wapr_realsense --config configs/box.yaml --web --web-host 0.0.0.0 --backend trt --object-mode single
```

The Jetson system/container still needs the CUDA toolkit, a C++ compiler, NVIDIA EGL/GL
libraries and camera USB access. Build engines and the renderer on the target rather
than copying desktop `.engine` or `.so` files.

TRT mode loads engines directly without PyTorch network weights. Pose engines share one
execution-memory buffer sized to their largest requirement, with serialized submissions;
weights and persistent context state remain separate. Existing engines need no rebuild.
Check all four engines together with `python scripts/check_trt_memory.py --run` (synthetic
12/6-candidate inputs, not an accuracy test), then run the application with `--backend trt`.
`WAPR_TRT_CONTEXT` reports each requirement; `WAPR_TRT_MEMORY` reports the allocated shared buffer.
The operator measured 13,460.52 MiB of separate execution requirements versus a 4,266.14 MiB
shared buffer on Orin NX 16GB. Synthetic inference reported non-finite outputs, so TensorRT
numerical correctness and pose tracking remain unverified. `WAPR_TRT_SHARED_MEMORY=0` restores
separate context allocations and may exhaust memory on Orin NX 16GB.

Choose engine capacity with `--object-mode single|multi` (application/export default: `single`).
With the current recipe, single mode exports profiles `2/12/12` and four pose engines; multi mode
exports `2/12/120` and adds `wbps_batch`. Single-object tracking uses `wbps.engine` directly.
The RealSense UI still tracks one target; this option selects engine capacity, not a multi-target UI.
Existing `2/12/120` engines remain accepted in single mode.

```bash
export WAPR_TRT_WORKSPACE_GB=4
python -m wapr.export_engines --object-mode single                     # four engines, sequentially
python -m wapr.export_engines --object-mode single --engine sapr       # only SAPR
python -m wapr.export_engines --object-mode multi --engine wbps_batch  # only the multi-object scorer
python -m wapr.export_engines --object-mode multi                      # full multi-object set
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single
```

Each engine exports in a fresh process; the process exits before the next one starts. PyTorch
export models and input tensors are released before TensorRT building. Exporting a selected
engine replaces only its corresponding file in the existing cache; modes share the same paths.
To investigate non-finite outputs without rebuilding, run the same model/input in separate processes:

```bash
python scripts/check_trt_memory.py --run --models wapr_w_mask --memory shared
python scripts/check_trt_memory.py --run --models wapr_w_mask --memory private
```

The diagnostic reports NaN/Inf counts for every tested output/batch and exits nonzero if any fail.
Use `--input random` for a second synthetic input set; neither input set validates pose accuracy.

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

Add `--profile-tracking` to reuse the existing timers for a tracking breakdown:

```bash
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single \
  --web-host 0.0.0.0 --web-port 8000 --profile-tracking --timing-interval 5
```

Each tracking window also reports WAPR, SAPR and WBPS milliseconds per frame, with nested
`crop` (rendering/crop preprocessing) and `network` (forward call including buffer setup) times.
Repeated iterations are summed per frame. `other` is the total minus the three stage totals;
crop/network values are already included in their parent stages. The same summaries appear
in the browser. Profiling defaults off because extra CUDA synchronization changes latency;
compare normal tracking performance with the flag removed. Initialization and warmup are excluded.

The crop path builds one perspective sampling grid for RGB, XYZ and the optional mask, and
omits the unused observed-depth crop. It uses the renderers' documented 0–1 RGB output directly,
avoiding a GPU scalar range check. CPU comparisons against independent Kornia warps cover
mask/no-mask inputs and batches 1/2/3/6/12; Jetson latency and pose accuracy still need comparison.
These preprocessing changes reuse existing engines.

Choose tracking-only work with `--tracking-preset` (initialization keeps its existing recipe):

| Preset | WAPR iterations | SAPR iterations | WBPS candidates | Network calls/frame |
| --- | ---: | ---: | ---: | ---: |
| `full` (default) | 3 | 2 | 6 | 6 |
| `balanced` | 2 | 1 | 3 | 4 |
| `fast` | 1 | 1 | 3 | 3 |
| `small-motion` | 0 | 1 | 2 | 2 |

`full` uses the current recipe iteration counts (3/2 by default). It reuses branches only when
the supplied center and previous poses are exactly equal. In the normal one-object path this
reduces WAPR crop rows 2→1, SAPR rows 3→2 and scoring crop rows 6→4; WBPS still receives all six
original candidates in their original order, including duplicates. Add `--no-reuse-tracking`
with `full` to compare against the previous workload. Different center/previous poses retain
their separate branches; no reuse crosses object identities.

`balanced`/`fast` score previous, WAPR(center) and SAPR(WAPR(center)); `small-motion` scores
previous and SAPR(center). Center defaults to the previous pose. Reduced presets change
candidate scoring and recovery behavior: compare drift, loss rate and score thresholds on
your hardware. Existing single-object engines support these groups; refinement batch=1 is
still padded to the engine minimum (currently 2). Actual batch=1 engine export is not included.
Multi-object reduced presets require `wbps_batch.engine`, not the fixed-six tracking scorer.
The preset and reuse option are recorded in `session.json`; warmup uses the selected preset.

```bash
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single \
  --web-host 0.0.0.0 --web-port 8000 --tracking-preset balanced --profile-tracking
```

Set `tracking_preset: balanced` in YAML for the same behavior. CLI flags override YAML values.
CPU regression tests verify candidate multiplicity, row reductions and preset routing.

### Kalman motion prediction

The sample target YAML configs enable a constant-velocity pose predictor. It supplies the
initial pose for WAPR tracking in every preset, including the browser interface. Outputs
remain accepted WAPR estimates; the filter does not replace them with smoothed or predicted
poses. Runs without a config default to disabled; use `--kalman` or `--no-kalman` to override.

```yaml
kalman_motion_prediction:
  activate: true
  translation_measurement_std_m: 0.01
  rotation_measurement_std_rad: 0.05
  translation_acceleration_std_m_s2: 1.0
  rotation_acceleration_std_rad_s2: 3.0
  max_dt_s: 0.5
```

```bash
python -m wapr_realsense --config configs/box.yaml --kalman
# Compare using the same tracking preset and input:
python -m wapr_realsense --config configs/box.yaml --no-kalman
```

Measurement standard deviations set how strongly accepted WAPR poses correct the filter;
larger values trust the motion model more. Acceleration standard deviations control how
quickly velocity can change; larger values allow faster adaptation. These are starting
parameters, not calibrated sensor uncertainties. All numeric settings must be positive and finite.
The filter uses actual sensor timestamp intervals (host reception time if unavailable), a
12-dimensional position/rotation-error/velocity state, and SO(3) rotation increments.
Duplicate/backward timestamps, clock-domain changes or gaps exceeding `max_dt_s` discard
stale velocity. Initialization, reset and tracking loss also restart the filter. Prediction
behind the camera falls back to the last accepted pose. Single-frame loss/reinitialization
rules still apply; this addition does not provide occlusion recovery.

The design references [FoundationPose++'s Kalman motion predictor](https://github.com/teal024/FoundationPose-plus-plus/blob/master/src/utils/kalman_filter_6d.py),
with sensor-time prediction and rotation errors replacing fixed time steps and Euler angles.
Settings are saved in `session.json`. CPU tests cover synthetic motion prediction and lifecycle
routing; improvement on live RGB-D input and Jetson overhead still require measurement.

### Jetson tracking measurements

Orin NX 16GB, JetPack 5.1.1 / CUDA 11.4, TensorRT 8.5.2.2,
torch 2.1.0a0+41361538.nv23.06, MAXN with `jetson_clocks` enabled. Input was a RealSense
D435I at 640×480/15 Hz, using the box mesh, manual ROI initialization, TensorRT engines,
shared execution memory and the web interface. Measurements enabled `--profile-tracking
--timing-interval 5` and exclude startup, warmup and initialization.

| Tracking preset | Approximate mean update time | Observed window-mean range | Time reduction vs. full |
| --- | ---: | ---: | ---: |
| `full` | 187 ms | 186.55–188.59 ms | — |
| `balanced` | 120 ms | 119.00–120.76 ms | 36% |
| `fast` | 94 ms | 93.44–94.47 ms | 50% |
| `small-motion` | 58 ms | 57.40–59.42 ms | 69% |

Ranges include partial windows flushed at loss/reinitialization/exit; they are ranges of
logged averages, not per-frame percentiles. These were separate live runs with different
manual ROIs, not a controlled replay benchmark. Synchronized profiling adds overhead, and
the timings exclude camera waiting, pose-file writing, browser encoding and network/display
latency. At about 58 ms, processing capacity exceeds the 15 Hz camera input; actual pose
output remains limited by acquisition and other pipeline work.

The the projected bounding box slightly less accurate with `balanced`, `fast`
and `small-motion`, but provisionally acceptable. Pose accuracy, drift and loss rate have not
been quantified; FP16 numerical parity is also unverified. `full` remains the default/reference.
Use the same recorded RGB-D sequence for a controlled accuracy and latency comparison.

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
