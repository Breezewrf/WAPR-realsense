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
python -m wapr_realsense --config configs/box.yaml
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

JetPack 5.1.1 / Python 3.8 的源码运行路径使用 `uv pip`，不执行上面的桌面 `uv sync`。
先安装对应 Jetson 的 GPU 包；当前操作者已验证 torch `2.1.0a0+41361538.nv23.06`、
torchvision `0.16.2+c6f3977` 的 CUDA 运算和 NMS，以及 TensorRT `8.5.2.2` Builder。
在同一个已激活的 `.venv` 中继续：

```bash
uv pip install -r requirements-jetson.txt
uv pip check
export WAPR_CACHE_DIR="$PWD/.wapr_runtime"
python -m wapr_realsense --check-camera
python scripts/setup_runtime.py
python -m wapr.export_engines
python -m wapr_realsense --config configs/box.yaml --init roi --backend trt
```

此文件固定已验证的 GPU 包版本，并限制 NumPy/Kornia 等兼容范围；TensorRT 沿用系统安装。
桌面 `pyproject.toml` 和 `uv.lock` 仍用于 Python 3.10+ 的桌面环境。
Docker 需要 NVIDIA runtime、相机 USB 设备访问和 NVIDIA EGL/GL 库；使用 ROI 窗口还需要显示转发。
渲染器和引擎在 Orin 本机重新构建，不复制桌面的 `.so` 或 `.engine`。
ARM64 导出 workspace 上限默认 1 GiB；操作者在 1 GiB 失败后，使用
`WAPR_TRT_WORKSPACE_GB=4` 成功构建 `wapr_w_mask`。此值限制算法选择所需的临时空间，
不限制整个进程的内存；选中的算法在推理时仍需要临时空间。
操作者已验证 D435I 取帧、原生渲染及 torch ROI 跟踪（约 503–518 ms/帧）；
TensorRT 位姿推理仍待实机验证。TRT 8.5 导出使用
opset 16，将 LayerNorm 展开为基础算子。没有显示转发时使用 `--no-display --bbox X Y W H`，
框坐标应来自当前校正后的图像；可先用 `--check-camera --output outputs/camera_check` 保存图像。
此 Python 3.8 路径用于 ROI 跟踪；可选 SAM2 等模块的依赖不包含在该安装清单中。

TRT 模式直接加载引擎，不额外加载 PyTorch 网络权重。位姿引擎串行提交推理，
共用一块按最大需求分配的执行内存；权重和 context 的持久状态仍各自保留，无须重建已有引擎。
先运行 `python scripts/check_trt_memory.py --run`，保留四个引擎并测试 12/6 候选的合成输入
（不验证位姿精度），再用 `--backend trt` 启动应用。
`WAPR_TRT_CONTEXT` 显示各引擎需求，`WAPR_TRT_MEMORY` 显示实际分配的共享缓冲大小。
操作者在 Orin NX 16GB 上测得各自执行内存合计 13,460.52 MiB，共享缓冲为 4,266.14 MiB。
合成输入出现非有限输出，TensorRT 数值正确性和位姿跟踪尚未通过验证。
`WAPR_TRT_SHARED_MEMORY=0` 恢复各 context 单独分配，对比时可能再次耗尽内存。

使用 `--object-mode single|multi` 选择引擎容量，应用和导出均默认 `single`。
当前配置下，单物体模式导出 `2/12/12` profile 和四个主引擎；多物体模式使用 `2/12/120`，
并增加 `wbps_batch`。单物体跟踪直接使用 `wbps.engine`，不要求额外跟踪引擎。
RealSense 界面仍跟踪一个目标，此选项选择引擎容量，不会将界面改成多目标跟踪。
单物体运行也接受已有的 `2/12/120` 引擎。

```bash
export WAPR_TRT_WORKSPACE_GB=4
python -m wapr.export_engines --object-mode single                     # 依次导出四个主引擎
python -m wapr.export_engines --object-mode single --engine sapr       # 仅导出 SAPR
python -m wapr.export_engines --object-mode multi --engine wbps_batch  # 仅导出多物体评分引擎
python -m wapr.export_engines --object-mode multi                      # 完整多物体引擎集
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single
```

每个引擎在独立进程中导出，进程退出后才开始下一个；构建 TensorRT 前释放 PyTorch 导出模型和输入张量。
两个模式使用相同缓存路径，指定某个引擎只会替换对应文件。
非有限输出先用同一模型、同一输入在两个独立进程中对比，无须重建：

```bash
python scripts/check_trt_memory.py --run --models wapr_w_mask --memory shared
python scripts/check_trt_memory.py --run --models wapr_w_mask --memory private
```

诊断会打印每个 batch、每个输出的 NaN/Inf 数量，任何失败都会以非零状态退出。
可加 `--input random` 对比随机合成输入；这些检查均不验证位姿精度。

在冻结画面上紧贴目标拖出矩形框，Enter/空格确认。实时窗口中：
`r` 重新初始化、`s` 保存 RGB-D 与结果、`q`/Esc 退出。
默认采用本机已验证可取帧的 640×480、15 Hz；采集频率不代表位姿输出频率。

不使用配置文件，也可指定其他模型：

```bash
python -m wapr_realsense --mesh /absolute/path/model.obj --mesh-unit m --backend trt
```

ROI 模式无窗口运行需提供初始框，像素坐标对应校正后的彩色图像：

```bash
python -m wapr_realsense --config configs/box.yaml --no-display --bbox 180 100 280 300
```

## 可选自动初始化

```bash
python -m wapr_realsense --config configs/box.auto.yaml
# 也可以直接为已有目标配置开启自动模式，支持无窗口运行：
python -m wapr_realsense --config configs/box.yaml --init auto --no-display
```

自动模式复用现有 WAPR 检测器：GroundingDINO 生成候选框，SAM2 分割，DINOv2 与当前
mesh 的多视角渲染模板匹配，再由 WAPR 根据检测 mask 估计位姿；选择通过 `--min-score`
的候选中 `score_6d` 最高者。这里使用 CAD 模板，不使用参考照片，也无法区分多个相同物体的
物理身份。只在搜索阶段运行检测，成功后切回原有六候选跟踪。

YAML 将检测选项集中到 `groundingdino_sam2_dinov2_detect` 下：

```yaml
groundingdino_sam2_dinov2_detect:
  activate: true       # false 使用手动 ROI 初始化
  use_sam2: false      # true 开启 SAM2；false 使用 bbox 输入
  interval: 1.0
  score: 0.1
  backend: trt
  dino: vitl14
  grounding: swinb
  template_cache: ../.wapr_runtime/templates/box_vitl14.pt
```

| 块内字段 | 默认值 | 含义 |
| --- | --- | --- |
| `activate` | `false` | 开启自动初始化及跟踪丢失后的重新检测 |
| `use_sam2` | `true` | 是否使用 SAM2；`false` 时使用 bbox 区域匹配，并以 bbox 输入 WAPR |
| `interval` | `1.0` | 一次搜索失败结束后，到下一次尝试的间隔秒数 |
| `score` | `0.1` | 2D 模板匹配阈值，与顶层 `min_score` 分开 |
| `backend` | `trt` | DINOv2 编码后端，也支持 `torch` |
| `dino` | `vitl14` | 支持 `vits14`、`vitb14`、`vitl14` |
| `grounding` | `swinb` | 支持 `swint`、`swinb` |
| `template_cache` | 不设置 | 可选 `.pt` 模板缓存；不设置则每次启动在显存中构建 |

配置文件使用 `.yaml` 或 `.yml`，检测选项统一放在上述配置块内。命令行显式参数优先：
`--init auto` / `--init roi` 覆盖 `activate`，
`--detect-use-sam` / `--no-detect-use-sam` 覆盖 `use_sam2`。例如：

```bash
python -m wapr_realsense --config configs/box.auto.yaml --no-detect-use-sam
```

关闭后跳过 SAM2 模型加载、推理和权重下载，仍运行 GroundingDINO 和 DINOv2。
DINO 裁剪和重复候选抑制使用矩形区域；位姿估计收到的是不带 mask 的 `bbox_xywh`，走
`wapr_wo_mask` 路径。背景可能更多地参与匹配和初始深度估计，需要重新检查分数与准确率；
SAM2 计时显示 `skipped`。`--detect-use-sam` 可以覆盖配置并重新启用 SAM2。

首次使用会按现有机制准备可选检测依赖、下载缺失权重；仅位姿环境准备脚本不会预加载它们。
DINO TensorRT 后端按需构建引擎，GroundingDINO 和 SAM2 保持 FP32，启动耗时和显存占用会增加。
环境不兼容时由现有依赖准备机制报错。示例模板缓存相对配置文件解析，位于
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

加上 `--profile-tracking`，复用现有计时器查看跟踪分段耗时：

```bash
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single \
  --web-host 0.0.0.0 --web-port 8000 --profile-tracking --timing-interval 5
```

每个统计窗口增加 WAPR、SAPR、WBPS 的每帧平均耗时，以及各自的 `crop`（渲染和裁剪预处理）
和 `network`（包含缓冲设置的网络前向调用）。多次迭代的耗时按帧累加；`other` 为总耗时减去
三个阶段总耗时。crop/network 已包含在对应阶段内，不应重复相加。网页复用同一份汇总。
分段计时默认关闭，额外 CUDA 同步会改变延迟；评估正常跟踪性能时去掉此开关。
初始化和预热不混入跟踪分段统计。

裁剪路径为 RGB、XYZ 和可选 mask 复用一份透视采样网格，省去未使用的观测深度裁剪；
直接使用渲染器约定的 0–1 RGB 输出，避免读取 GPU 标量判断颜色范围。
已在 CPU 上与独立 Kornia 变换对照，覆盖有/无 mask 和 batch 1/2/3/6/12；
Jetson 延迟和位姿精度仍需实机对比。这些预处理优化复用已有 engine。

使用 `--tracking-preset` 选择跟踪负载，初始化继续使用原有 recipe：

| 预设 | WAPR迭代 | SAPR迭代 | WBPS候选 | 每帧网络调用 |
| --- | ---: | ---: | ---: | ---: |
| `full`（默认） | 3 | 2 | 6 | 6 |
| `balanced` | 2 | 1 | 3 | 4 |
| `fast` | 1 | 1 | 3 | 3 |
| `small-motion` | 0 | 1 | 2 | 2 |

`full` 沿用当前 recipe 的迭代次数，默认3/2；仅在 center 与上一帧位姿完全相同时复用分支。
普通单目标路径的 WAPR 裁剪行数从2降到1、SAPR 从3降到2、评分裁剪从6降到4；
WBPS 前向仍接收原来的六个候选、顺序和重复次数。用 `full --no-reuse-tracking`
对照原负载。center 与上一帧位姿不同时保留独立分支，不跨物体身份复用。

`balanced`/`fast` 评分上一帧、WAPR(center)、SAPR(WAPR(center))；`small-motion`
评分上一帧和 SAPR(center)。center 默认等于上一帧位姿。减少候选会改变评分和恢复能力，
需要实测漂移、丢失率并重新评估评分阈值。现有单物体引擎支持这些候选组；修正 batch=1
仍会补到引擎最小 batch（当前为2），本次未增加真正的 batch=1 引擎导出。
多物体使用减少候选的预设时，需要 `wbps_batch.engine`，不能使用固定六候选的跟踪评分引擎。
预设和复用开关写入 `session.json`，预热也使用所选预设。

```bash
python -m wapr_realsense --config configs/box.yaml --web --backend trt --object-mode single \
  --web-host 0.0.0.0 --web-port 8000 --tracking-preset balanced --profile-tracking
```

也可以在 YAML 中设置 `tracking_preset: balanced`，命令行参数优先。
CPU 回归测试覆盖候选重复次数、裁剪行数和预设传递；Jetson FP16 数值与性能仍需实机对比。

## 网页 Demo

```bash
python -m wapr_realsense --config configs/box.auto.yaml --web
# 默认手动框选，不加载自动检测模型：
python -m wapr_realsense --config configs/box.yaml --web
# Jetson 局域网访问：
python -m wapr_realsense --config configs/box.auto.yaml --web --web-host 0.0.0.0
```

本机打开 `http://127.0.0.1:8000`，局域网打开 `http://<jetson-ip>:8000`。
页面左侧显示彩色画面、3D 框与坐标轴；右侧包含状态、手动初始化开关、Detect 和日志式耗时摘要。
静态资源随 Python 包提供，不需要 Node、外部 CDN 或桌面显示环境。

- **WARMUP**：加载模型并预热，画面留空、按钮禁用；预热输出全部丢弃。
- **INITIALIZED**：准备完成，显示相机画面，等待点击 **Detect**。
- **Looking for object**：首次检测或目标丢失后的自动重搜；搜索期间 Detect 禁用。
- **TRACKING**：已找到目标，显示位姿叠加；可再次点击 Detect 发起新搜索。

开关初始值遵循配置的初始化模式，也可在网页切换。开启手动初始化后，画面冻结供拖动框选，
画好框再点击 Detect；“重新框选”会取得新的冻结帧。框选坐标映射回原图像素，并绑定这份
RGB-D，不会用到后续帧。跟踪成功后恢复实时画面。
**手动模式丢失目标后提示“目标丢失，请重新手动框选”，清除旧框并等待用户重新框选，
不会自动重搜或复用旧框。** 关闭开关时保留自动搜索与丢失后自动恢复。
预热和搜索期间开关禁用；从手动切到自动且尚未加载检测模型时，重新显示 WARMUP，
加载和预热结束后等待 Detect。模式切换会清除旧跟踪。

右侧复用终端 `[WARMUP]`、最近一次 `[INIT]` / `[REINIT]` 和 `[TRACKING]`
汇总内容；tracking 仍按 `--timing-interval` 输出区间平均耗时与帧数。
预热完成后默认折叠 WARMUP，仍可展开查看。无候选时未运行的 SAM2 / DINOv2 显示
`skipped`；位姿和 tracking 使用单实例代表负载预热，真实候选批量变化仍可能产生额外首次开销。
自动搜索期间画面保持在搜索输入帧，HTTP 状态接口仍可响应。
断开网页连接会禁用按钮并隐藏旧画面；重新连接后恢复当前会话。

无需相机或 CUDA 即可检查页面交互，画面及耗时明确标为模拟数据：

```bash
python -m wapr_realsense --web-demo
```

模拟模式约 2 秒完成预热，默认支持手动框选，也可关闭开关体验自动模式；点击 Detect 后约 2 秒
找到目标，tracking 约 12 秒后模拟丢失，按当前模式等待重新框选或自动重搜。
真实模式仍使用原来的位姿文件输出约定，逐帧附带 `initialization_mode`；本次没有验证 Jetson 实机推理。

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
- JetPack 5.1.1 上已验证基础 GPU 运算、D435I 取帧、渲染器编译和初始化；完整位姿链路尚未验证。
- 本次保留原来的网络、候选数量、引擎 profile 和模型加载方式；尚未实施单目标显存优化。

运行测试：`python -B -m unittest discover -s tests -v`。

## 来源与许可

本项目是基于 WAPR 的应用改编，并非原作者的官方发行版。
推理核心源自 WAPR commit `f165cec8ebdd48ae6f46d06a46c40d796ab80b71`；
本项目增加 RealSense 输入和实时位姿接口，并保留上游后端模块。
代码保留 [LGPL-2.1-only](LICENSE)，模型权重适用独立的
[权重许可](WEIGHTS_LICENSE.txt)。保留[作者声明](AUTHORS.md)、
[第三方声明](THIRD_PARTY_NOTICES.txt)和随附许可文件。
