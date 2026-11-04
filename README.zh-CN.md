# WAPR-realsense

[English](README.md)

**RealSense RGB-D 输入，持续输出单个目标的 6D 位姿。** 基于
[WAPR](https://github.com/WangYuLin-SEU/WAPR)，面向 robotics 的单类别、单实例、已知 mesh 场景。
首帧框选目标，之后逐帧跟踪；应用运行路径不加载 DINOv2、SAM2、GroundingDINO。

- D455 深度对齐彩色、传感器尺度换算为米、RGB-D 一致的畸变校正。
- 复用 WAPR/SAPR 修正和 WBPS 候选评分，默认 TensorRT。
- 实时 3D 框与坐标轴、逐帧 JSONL、原子更新的 `latest.json`、RGB-D 快照。
- 输出 `4×4` 变换、`R`、米制平移、`xyzw` 四元数、时间戳和有效状态。
- 单目标配置与机器人端读取示例，支持外部标定矩阵转换到基座坐标系。

这是需要初始化的位姿跟踪应用；不会自动寻找目标或在丢失后自动重新检测。
目标丢失后按 `r` 重新框选。实物对应的米制 mesh 仍是必需输入。

## 快速运行

当前验证环境为 Linux、Python 3.10、RTX 4090 Laptop 和 RealSense D455。
在仓库根目录执行：

```bash
uv sync
source .venv/bin/activate
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python -m wapr_realsense --config configs/box.json
```

首次使用还需准备系统 CUDA toolkit、C++ 编译器及 EGL/GL 开发头文件，再执行：

```bash
python scripts/setup_runtime.py
python -m wapr.export_engines
```

`uv sync` 安装 Python 依赖；以上步骤准备权重、渲染器与本机 TensorRT 引擎。已有可用缓存时直接运行即可。

box 配置引用本地 `assets/meshes/box/textured_mesh_modified.obj`，单位为米。
干净源码不包含这个模型；新机器上自行复制模型，或修改配置中的路径。
**配置内路径相对配置文件解析，命令行显式参数优先。**

在冻结画面上紧贴目标拖出矩形框，Enter/空格确认。实时窗口中：
`r` 重新初始化、`s` 保存 RGB-D 与结果、`q`/Esc 退出。
默认采用本机已验证可取帧的 640×480、15 Hz；采集频率不代表位姿输出频率。

不使用配置文件，也可指定其他模型：

```bash
python -m wapr_realsense --mesh /absolute/path/model.obj --mesh-unit m --backend trt
```

无窗口运行需提供初始框，像素坐标对应校正后的彩色图像：

```bash
python -m wapr_realsense --config configs/box.json --no-display --bbox 180 100 280 300
```

## 输出与机器人接入

box 配置默认保存到 `outputs/box/`，包括 `latest.json`、`poses.jsonl`、
`camera.json`、`session.json`。终端默认每十个处理帧打印一次，`--print-every 1` 打印每帧。

```bash
python examples/read_pose.py --input outputs/box/latest.json
```

位姿表示原始 mesh 坐标系（单位换算和额外缩放后）到彩色相机光学坐标系：

```text
p_camera = R @ p_mesh + t_m
T_base_object = T_base_camera @ T_camera_object
```

相机 X 向右、Y 向下、Z 向前，平移为米。转换到机器人基座需要实测相机标定，
读取示例会检查 `valid` 与数据新鲜度；默认拒绝超过 500 ms 的数据，可用 `--max-age-ms` 调整。
WBPS 分数不是成功概率；遮挡、对称外观、缺失深度与 mesh 尺度误差可能导致漂移或错误位姿。

## 项目结构

```text
wapr_realsense/  相机适配、跟踪会话、CLI、几何与输出接口
wapr/            WAPR 推理核心、可选检测/SAM 模块及原作者声明
configs/         单物体与相机到基座的配置模板
examples/        机器人端位姿读取示例
scripts/         位姿运行环境准备
tests/           CPU 接口测试和可选 SDK 几何测试
assets/meshes/   本地模型，默认不进入 Git
```

环境、缓存、权重、引擎和本地 mesh 默认由 `.gitignore` 排除。
保留自动检测和 SAM 源码供按需使用；当前 ROI 跟踪入口不会加载这些模块。

## 验证范围

- 桌面已执行 D455 640×480/15 Hz 采集、TensorRT 初始化与跟踪调用。
- 整理前，操作者已反馈当前 box 的实拍框选跟踪效果良好。
- CPU 测试覆盖深度、畸变映射、配置、丢失状态与机器人输出约定。
- 当前验证范围仅覆盖上述桌面环境。
- 本次保留原来的网络、候选数量、引擎 profile 和模型加载方式；尚未实施单目标显存优化。

运行测试：`python -B -m unittest discover -s tests -v`。

## 来源与许可

本项目是基于 WAPR 的应用改编，并非原作者的官方发行版。
推理核心源自 WAPR commit `f165cec8ebdd48ae6f46d06a46c40d796ab80b71`；
本项目增加 RealSense 输入和实时位姿接口，并保留上游后端模块。
代码保留 [LGPL-2.1-only](LICENSE)，模型权重适用独立的
[权重许可](WEIGHTS_LICENSE.txt)。保留[作者声明](AUTHORS.md)、
[第三方声明](THIRD_PARTY_NOTICES.txt)和随附许可文件。
