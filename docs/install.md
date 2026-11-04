# Installation / 安装

## Existing workspace / 当前环境

No reinstall is needed if the original live command already works:

```bash
source .venv/bin/activate
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --config configs/box.json
```

The old `python examples/realsense_pose.py ... --init roi` command remains a compatibility launcher.
It now defaults to ROI initialization and 15 Hz. The application has no automatic detector mode.

## Fresh desktop environment / 新的桌面环境

From the source root, reproduce the current CUDA 12.6 desktop environment with:

```bash
uv sync
source .venv/bin/activate
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python scripts/setup_runtime.py
python -m wapr.export_engines
python -m wapr_realsense --check-camera
```

The project declares the same packages as the previous manual installation commands.
Only the tested PyTorch/Torchvision and TensorRT versions are fixed; ordinary dependencies
are resolved normally. uv uses the official CUDA 12.6 PyTorch index and NVIDIA's index
for TensorRT binaries. pip is included because the upstream setup helper invokes it.
The first install downloads large CUDA/TensorRT packages; subsequent installs reuse uv's cache.
`uv sync` generates `uv.lock` automatically. No separate environment snapshot is required.

This configuration is for the desktop. Jetson should use the [JetPack-specific procedure](jetson.md).
The setup helper prepares the pose runtime and four pose checkpoints without detector models.

Linux OpenGL rendering needs a C++ compiler, CMake, CUDA toolkit/nvcc and EGL/GL development headers.
If the setup helper reports missing system packages on Ubuntu, install the reported prerequisites:

```bash
sudo apt-get install build-essential cmake ninja-build libegl1-mesa-dev libgl1-mesa-dev
```

Use one OpenCV provider. The default dependencies select `opencv-python` for ROI windows.
For a headless or embedded target with system-provided packages, install the project with `--no-deps` after preparing its dependencies.

## Mesh and paths / 模型与路径

Keep OBJ, MTL and textures together. `mesh_unit` is explicit (`m`, `mm` or `cm`);
the file suffix does not specify its units. `mesh_scale` applies an additional uniform factor.
The loader does not overwrite the source model or automatically fit it to the depth image.

Copy `configs/target.example.json` and edit it. Paths in JSON are relative to the JSON file;
CLI paths are relative to the shell's current working directory.
`WAPR_CACHE_DIR` must be absolute; its `cache/weights` directory holds checkpoints and local engines.
Outputs are separate and follow `--output` or a timestamped directory below `outputs/wapr_realsense/`.

## Camera / 相机

Use USB 3 and release the camera from other applications first:

```bash
python -m wapr_realsense --check-camera --fps 15
python -m wapr_realsense --check-camera --fps 15 --output outputs/camera_check
```

The second command saves aligned, rectified `rgb.png`, meter-valued `depth.npy` and `camera.json`.
The D455 in this workspace delivered synchronized RGB-D at 640×480/15 Hz; the first dual-stream
30 Hz attempts timed out. Request other profiles explicitly after checking the target device.
Distortion is read from the actual color profile, not assumed zero. Inverse/modified Brown
maps use the SDK projection model, and depth uses nearest-neighbor resampling.

The [RealSense alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py)
and [projection API](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/rsutil.h)
describe the upstream geometry operations. Missing or zero depth remains invalid.

## Offline replay / 离线检查

Use a saved camera check or `s` snapshot with the matching mesh and an explicit initial box:

```bash
python -m wapr_realsense --mesh /path/to/model.obj \
  --replay outputs/camera_check --bbox 180 100 280 300 --no-display --max-frames 2
```

For an upstream sample, request it explicitly rather than downloading all dataset excerpts:

```bash
python scripts/setup_runtime.py --sample
python -m wapr_realsense --mesh "$WAPR_CACHE_DIR/cache/samples/pose_lmo/model.ply" \
  --replay "$WAPR_CACHE_DIR/cache/samples/pose_lmo" --no-display --max-frames 2
```

The sample's mask is used only to derive a bounding box. The first pass initializes and the
second tracks on the same saved frame; this is interface validation, not a motion benchmark.
