# WAPR-realsense

[English](README.md)

**RealSense RGB-D 输入，持续输出单个目标的 6D 位姿。** 基于
[WAPR](https://github.com/WangYuLin-SEU/WAPR)，面向 robotics 的单类别、单实例、已知 mesh 场景。
支持手动框选或可选的自动检测初始化，之后逐帧跟踪；默认 ROI 模式不加载检测模型。

- D455 深度对齐彩色、传感器尺度换算为米、RGB-D 一致的畸变校正。
- 复用 WAPR/SAPR 修正和 WBPS 候选评分，默认 TensorRT。
- 可选 GroundingDINO → SAM2 → DINOv2 CAD 模板检测初始化。
- 实时 3D 框与坐标轴、逐帧 JSONL、原子更新的 `latest.json`、RGB-D 快照。
- 输出 `4×4` 变换、`R`、米制平移、`xyzw` 四元数、时间戳和有效状态。
- 单目标配置与机器人端读取示例，支持外部标定矩阵转换到基座坐标系。

默认 ROI 模式在目标丢失后按 `r` 重新框选；自动模式在启动和跟踪更新被拒绝后搜索目标，
按 `r` 也会重新搜索。两种模式都需要实物对应的米制 mesh。

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

ROI 模式无窗口运行需提供初始框，像素坐标对应校正后的彩色图像：

```bash
python -m wapr_realsense --config configs/box.json --no-display --bbox 180 100 280 300
```

## 可选自动初始化

```bash
python -m wapr_realsense --config configs/box.auto.json
# 也可以直接为已有目标配置开启自动模式，支持无窗口运行：
python -m wapr_realsense --config configs/box.json --init auto --no-display
```

自动模式复用现有 WAPR 检测器：GroundingDINO 生成候选框，SAM2 分割，DINOv2 与当前
mesh 的多视角渲染模板匹配，再由 WAPR 根据检测 mask 估计位姿；选择通过 `--min-score`
的候选中 `score_6d` 最高者。这里使用 CAD 模板，不使用参考照片，也无法区分多个相同物体的
物理身份。只在搜索阶段运行检测，成功后切回原有六候选跟踪。

以下 JSON 配置均有对应的连字符命令行参数：

| 配置字段 | 默认值 | 含义 |
| --- | --- | --- |
| `init` | `roi` | 设置为 `auto` 开启 CAD 模板检测 |
| `detect_interval` | `1.0` | 一次搜索失败结束后，到下一次尝试的间隔秒数 |
| `detect_score` | `0.1` | 2D 模板匹配阈值，与 `min_score` 分开 |
| `detect_backend` | `trt` | DINOv2 编码后端，也支持 `torch` |
| `detect_dino` | `vitl14` | 支持 `vits14`、`vitb14`、`vitl14` |
| `detect_grounding` | `swinb` | 支持 `swint`、`swinb` |
| `detect_use_sam` | `true` | 是否使用 SAM2；`false` 时使用 bbox 区域匹配，并以 bbox 输入 WAPR |
| `template_cache` | 不设置 | 可选 `.pt` 模板缓存；不设置则每次启动在显存中生成 |

在 JSON 配置中添加 `"detect_use_sam": false` 即可关闭 SAM2，也可以使用命令行：

```bash
python -m wapr_realsense --config configs/box.json --init auto --no-detect-use-sam
```

关闭后跳过 SAM2 模型加载、推理和权重下载，仍运行 GroundingDINO 和 DINOv2。
DINO 裁剪和重复候选抑制使用矩形区域；位姿估计收到的是不带 mask 的 `bbox_xywh`，走
`wapr_wo_mask` 路径。背景可能更多地参与匹配和初始深度估计，需要重新检查分数与准确率；
SAM2 计时显示 `skipped`。`--detect-use-sam` 可以覆盖配置并重新启用 SAM2。

首次使用会按现有机制准备可选检测依赖、下载缺失权重；仅位姿环境准备脚本不会预加载它们。
DINO TensorRT 后端按需构建引擎，GroundingDINO 和 SAM2 保持 FP32，启动耗时和显存占用会增加。
环境不兼容时由现有依赖准备机制报错。示例模板缓存相对 JSON 文件解析，位于
`.wapr_runtime/templates/`，不在 `.wapr_runtime/cache/` 内。mesh/纹理、编码器或源码变化
使缓存失效时，应用自动选择同目录下带版本标识的新路径，首次重建、后续复用，并保留旧文件。
`DET2D_CAD_CACHE_CHANGED` 会打印实际路径，不需要删除目录。设置 `template_cache: null`
可以关闭模板磁盘缓存。

搜索期间输出 `searching`、`valid=false`，成功后输出 `initialized_auto`，随后为 `tracking`。
跟踪更新被拒绝时输出 `lost` 和空位姿，下一帧重新搜索。高分漂移仍可能漏检，重新搜索也不保证
找回同一个物理实例。
首次调用包含三次检测预热及位姿预热，评估恢复延迟应查看后续搜索。
搜索在主线程同步执行，期间暂停位姿输出。

终端按三种颜色显示计时，单位均为毫秒：

- **黄色 `[WARMUP]`**：首次检测预热后，汇总 GroundingDINO、SAM2、DINOv2 各自的总耗时、
  平均耗时和调用次数；首次有效候选批次再打印 WAPR 位姿初始化的预热总耗时和平均耗时。
- **绿色 `[TRACKING]`**：默认每 5 秒打印一次该区间的 WAPR 平均耗时和跟踪帧数，
  丢失、重新初始化或退出时补打剩余统计。仅统计跟踪更新，包括被拒绝的更新；首次调用开销也计入平均值。
  可用 `--timing-interval 2` 调整汇总间隔。
- **青色 `[INIT]` / `[REINIT]`**：首次搜索及后续搜索在开始时提示，结束时汇总 GroundingDINO、
  SAM2、DINOv2、WAPR 位姿初始化耗时，并显示检测总耗时、完整初始化总耗时、候选数和搜索结果。
  首次完整初始化总耗时包含预热；后续搜索复用已完成的预热。未运行的模块显示 `skipped`。

GroundingDINO 包含图像预处理、推理及框过滤；SAM2 包含分割及 mask 后处理；DINOv2
包含候选裁剪预处理和所有批次编码，不含模板匹配。检测总耗时还包含匹配和后处理。
WAPR tracking 包含完整跟踪更新和有效性检查；WAPR 位姿初始化会处理所有有效候选，最后才选择目标。
计时前后同步 CUDA，测量包含 CPU 工作和 GPU 等待的阶段墙钟耗时；额外同步会影响阶段重叠。
终端自动启用颜色，重定向日志或设置 `NO_COLOR=1` 时输出纯文本。

## 输出与机器人接入

box 配置默认保存到 `outputs/box/`，包括 `latest.json`、`poses.jsonl`、
`camera.json`、`session.json`。完整 JSON 逐帧保存到文件；终端默认显示简洁计时，
加 `--log-poses` 后每十个处理帧额外打印 `POSE_JSON`，再加 `--print-every 1` 可打印每帧。

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
自动检测和 SAM 模块用于可选自动模式；ROI 模式不会加载这些模块。

## 验证范围

- 桌面已执行 D455 640×480/15 Hz 采集、TensorRT 初始化与跟踪调用。
- 整理前，操作者已反馈当前 box 的实拍框选跟踪效果良好。
- CPU 测试覆盖深度、畸变映射、配置、丢失状态与机器人输出约定。
- 自动模式已用模拟推理测试验证 CLI/回放接入、重试间隔、候选选择和丢失后重初始化；
  真实三模型检测、启动显存与实拍自动初始化效果尚未验证。
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
