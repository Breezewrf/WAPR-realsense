# Jetson Orin NX deployment notes

The target discussed for this application is **Orin NX 16GB with JetPack 6.x**.
The project has not yet been deployed or benchmarked on that board.
The default `uv sync` dependencies select the desktop CUDA stack; use the target-specific steps below for Jetson.

## Preserve the target's GPU stack

Use CUDA-enabled PyTorch and TensorRT supplied for the actual JetPack/L4T version.
Do not copy the desktop `.venv`, x86-64 native renderer, or RTX 4090 `.engine` files.
The backend currently requires TensorRT 10.x; JetPack minor versions differ.
For example, [JetPack 6.2.1](https://developer.nvidia.com/embedded/jetpack-sdk-621)
includes CUDA 12.6 and TensorRT 10.3. This is a compatibility starting point, not a tested
WAPR-realsense Jetson environment.

A venv may need `--system-site-packages` to see the JetPack-provided TensorRT Python bindings.
Prepare one OpenCV provider and a RealSense SDK that imports on aarch64; the SDK may be built or
installed outside pip on the target. The base project does not select these packages for you.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c "import torch, tensorrt, cv2, pyrealsense2; print(torch.cuda.is_available(), tensorrt.__version__)"
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python scripts/setup_runtime.py --check
python scripts/setup_runtime.py
python -m wapr.export_engines
python -m wapr_realsense --check-camera --fps 15
python -m wapr_realsense --config configs/box.json
```

Build TensorRT engines and the CUDA/OpenGL extension on the target. Engine compatibility is
platform/device/version dependent; see [NVIDIA's support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Support-Matrix-Guide.pdf).
The four original `.pth` weights may be reused subject to their accompanying license;
keep each device's generated engines and native-build cache separate.

## Memory and performance

This application uses only ROI initialization and pose tracking, with no DINOv2, SAM2 or GroundingDINO.
However, the original general-purpose WAPR backend still loads four networks and uses engine profiles
whose maximum candidate batch is 120. The application uses 12 candidates for initialization and six
for tracking scoring. PyTorch network copies remain present in TensorRT mode.

Desktop engine introspection reported approximately 4.12 GiB total execution-context requirement
for the four loaded pose engines. This explains much of the observed ~5 GB GPU footprint on the desktop;
it is not a Jetson measurement. The proposed smaller profiles, lazy loading and removal of duplicate
network copies are future work, not changes made during repository organization.

Jetson CPU and GPU share physical DRAM; 16 GB is not separate CPU RAM plus 16 GB VRAM.
See [CUDA for Tegra](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html).
Measure runtime peak memory, engine-build peak memory, processing latency, output rate and frame age
alongside other robotics processes. The exporter currently permits an 8 GiB builder workspace budget;
that is a tactic workspace limit, not a guaranteed runtime allocation or a total-process limit.

Use `tegrastats` on the target. Report board power mode, cooling, camera profile, mesh complexity,
initialization versus warmed tracking, and other running workloads with any performance numbers.
