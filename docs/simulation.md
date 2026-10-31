# 仿真管线复现指南

本文按顺序列出仿真管线的每个阶段：在哪台机器上运行、运行什么命令、输入和输出在哪、怎样检查结果、大约多久。照着做，可以从空白的工作区做出检测网络、取矿动作规划、测试序列和回放结果。每个模块的逻辑见 [modules.md](modules.md)。

我们没有能量单元实物，也没有确定相机型号，所以检测网络用 Blender 渲染的合成图训练，整条流程在 Blender 渲染的 RGB-D 序列上测试。

## 管线一览

| 阶段 | 工具 | 输入 | 输出（`dataset/` 下） | 在哪运行 |
|---|---|---|---|---|
| 0. 准备 | `download_assets.sh`、`colcon build` | — | `assets/`（包内） | 开发环境 |
| 1. 环境贴图 | `scripts/blender/download_hdri.sh` | — | `hdri/` | 任意 |
| 2. 渲染训练图 | `scripts/blender/render_dataset.py` | `hdri/` | `train/render/v1/` | 有显卡的机器 |
| 3. 生成数据集 | `scripts/training/make_yolo_dataset.py` | `train/render/v1/` | `train/yolo/v1/` | 任意 |
| 4. 训练和导出 | `yolo`、`scripts/training/export_yolo.py` | `train/yolo/v1/` | `train/runs/v1/`、`models/` | 有显卡的机器 |
| 5. 规划取矿动作 | `pickup_plan` | `config/pickup_plan.yaml` | `plans/` | 开发环境 |
| 6. 渲染测试序列 | `scripts/blender/render_sequence.py --plan` | `plans/`、`hdri/` | `sequences/` | 有显卡的机器 |
| 7. 回放 | `sequence_replay` | `sequences/`、`models/` | `replay/` | 开发环境 |

第 2～4 步只为训练检测网络。已有检测网络时，从第 5 步开始。

`dataset/` 不进入仓库，除了实拍图，全部可以按本文重新生成：

```text
dataset/
├── real/images/      实拍测试图（自备）
├── weights/          外部下载的原始权重（SAM 官方权重等）
├── hdri/             Blender 用的环境贴图
├── train/
│   ├── render/v1/    渲染的训练图
│   ├── yolo/v1/      Ultralytics 数据集
│   └── runs/v1/      训练输出，含 weights/best.pt
├── models/           导出、还没放进 Release 的网络
├── plans/            取矿动作规划文件
├── sequences/        按规划渲染的测试序列
└── replay/           回放输出
```

版本号 `v1` 用来区分不同批次的训练数据和模型。

## 0. 准备

### 两种机器

- **开发环境**：RMCS 的开发容器（Ubuntu 24.04，ROS 2 Jazzy，MoveIt 2）。编译、规划取矿动作、回放都在这里。不需要显卡。
- **有 NVIDIA 显卡的机器**：Blender 渲染和训练。渲染也能用 CPU，但慢得多（见第 2 步）；训练需要显卡。

两边看到的是同一个工作区目录（例如把宿主机目录挂载进容器），这样渲染结果不用复制。下面的命令都在包目录 `rmcs_ws/src/rmcs_energy_trace` 下运行，路径写的是相对包目录的路径。

### 软件

| 软件 | 版本 | 用途 |
|---|---|---|
| Blender | 4.5 LTS（我们用 4.5.15），官方便携版 | 渲染。脚本在 Blender 自带的 Python 中运行，不需要另装 Python 包 |
| Python 环境 | Python 3，`ultralytics==8.3.251`、`opencv-python`、`numpy`、`openvino` | 生成数据集、训练、导出 |
| PyTorch | 与显卡匹配的 CUDA 版本；RTX 50 系列（sm_120）需要 CUDA 12.8 版本 | 训练 |
| SAM 导出（可选） | `torch`、`torchvision`、`segment-anything`、`onnx`、`onnxruntime` | 只在重新导出 SAM 网络时需要 |

安装 Ultralytics 后，运行 `yolo settings sync=False` 关闭使用统计上传。

### 代码和运行所需的文件

1. 把本仓库放到 RMCS 工作区的 `rmcs_ws/src/` 下。取矿动作规划和 Blender 渲染还要用到同一工作区里的 `arm_description`（机械臂的 URDF 和 STL）和 `arm_moveit_config`（MoveIt 配置），它们在 RMCS 主仓库中。
2. 下载运行所需的网络和 M3T 模型：

   ```zsh
   scripts/download_assets.sh
   ```

   文件清单和 SHA256 在 `scripts/assets.sha256`，下载后逐个校验。已存在且校验通过的文件会跳过。检测网络（YOLO11n-seg）暂时不在清单中，要按第 2～4 步训练出来。
3. 编译（在 `rmcs_ws` 下）：

   ```zsh
   colcon build --merge-install --packages-select rmcs_energy_trace
   ```

**检查**：运行 4 个快速测试，都应通过：

```zsh
for t in test_detection test_depth_mask test_point_to_plane test_supervision; do
  ../../build/rmcs_energy_trace/$t
done
```

## 1. 环境贴图

```zsh
scripts/blender/download_hdri.sh
```

从 Poly Haven（CC0）下载 13 个室内场景的 HDRI（2k，`.hdr`）到 `dataset/hdri/`，合计约 83 MB。已有的文件会跳过。下载慢时，先设置代理（`https_proxy`）。

**检查**：`dataset/hdri/` 下有 13 个 `.hdr` 文件。

## 2. 渲染训练图

```zsh
blender --background --factory-startup --python scripts/blender/render_dataset.py -- \
    --out dataset/train/render/v1 --count 2000 --seed 0 --device OPTIX
```

`--device` 可选 `CPU`、`CUDA`、`OPTIX`。每帧重新随机搭建一个取矿场景：

- 资源区的地面、墙和中心块（按规则手册图 4-22），中心块上 0～6 根能量单元；
- 地上横躺或竖立的能量单元；
- 干扰物，其中一部分是与能量单元大小相近的深色圆柱；
- 从画面边缘伸进来的遮挡物。

能量单元只用官方配色。材质、光源、HDRI 和相机位姿都随机。彩色图再按各自的概率叠加眩光、脏镜头、虚焦、运动模糊、雾化、暗角、色散、传感器噪声，以及曝光和白平衡的变化。

**输出**：每帧 4 个文件。

| 目录 | 内容 |
|---|---|
| `images/` | 彩色图（JPEG） |
| `index/` | 编号图：每个像素属于第几根能量单元，0 为其他物体 |
| `full/` | 只渲染能量单元时的编号图，用来算每根单元被挡住的比例 |
| `meta/` | 相机内参、每根单元的位姿、这一帧用到的效果 |

**耗时**：RTX 5060（OPTIX）每帧平均约 1.5 s，2000 帧约 50 min；只用 CPU（Ryzen 9600X）每帧约 7.5 s。中断后用同一条命令继续，已有 `meta/` 的帧会跳过。

**检查**：4 个目录各有 2000 个文件。打开几张彩色图，确认画面里有中心块和能量单元。

## 3. 生成数据集

```zsh
python3 scripts/training/make_yolo_dataset.py dataset/train/render/v1 dataset/train/yolo/v1
```

用每根单元的可见像素生成多边形标签。可见像素少于 300 个，或可见部分不到完整轮廓的 20% 时，这根单元不标注，留在图中当背景。按帧号散列取 10% 作验证集。图像用硬链接，不额外占空间。`data.yaml` 不写 `path`，数据集目录可以整体移动。

**检查**：

- 脚本最后打印一行统计。按上面的种子，应约为 `2000 frames (1800 train, 200 val), 6752 labeled units`。
- 打开 `dataset/train/yolo/v1/preview/` 下的几张图，多边形应贴合能量单元的轮廓。

## 4. 训练和导出

在有显卡的机器上训练：

```zsh
yolo segment train model=yolo11n-seg.pt data=$PWD/dataset/train/yolo/v1/data.yaml imgsz=640 \
    epochs=100 batch=16 project=$PWD/dataset/train/runs name=v1
```

预训练权重 `yolo11n-seg.pt` 第一次运行时自动下载。`data=` 和 `project=` 要写绝对路径：写相对路径时，Ultralytics 8.3.251 会把输出放到 `~/runs/` 下。显存不够时把 `batch` 降到 8。

导出为 OpenVINO，供 C++ 检测器使用。导出的目录名和文件名跟着 `.pt` 的文件名，所以先复制一份并改名：

```zsh
cp dataset/train/runs/v1/weights/best.pt dataset/models/yolo11n_seg_v1.pt
python3 scripts/training/export_yolo.py dataset/models/yolo11n_seg_v1.pt dataset/models
```

导出参数固定为输入 640 × 640、不含 NMS、不用半精度，与 C++ 检测器的约定一致。

**耗时**：RTX 5060 上 100 轮约 23 min（batch 16，显存约 3 GB）。

**检查**：

- `dataset/train/runs/v1/results.csv` 中，验证集的 box mAP50 应接近 0.99，mask mAP50-95 约 0.9。这个数字只说明模型学会了合成数据。合成验证集和训练集来自同一个渲染器，第 10 轮就接近饱和，不能代表实拍效果。
- 有实拍图时，用检测工具看效果：

  ```zsh
  ros2 run rmcs_energy_trace energy_unit_detect \
      dataset/models/yolo11n_seg_v1_openvino_model/yolo11n_seg_v1.xml <图片或目录> <输出目录>
  ```

## 5. 规划取矿动作

```zsh
ros2 launch rmcs_energy_trace pickup_plan.launch.py
```

它启动 move_group，按 `config/pickup_plan.yaml` 规划 3 份取矿动作，写到 `dataset/plans/plan_slot<k>_seed<s>.json`，然后自动退出。默认的 3 份：种子 0、1、2，目标安装位依次为 1（正上方）、0（右上）、4（正下方）。

每份规划都按实际操作流程：

1. `hold`：机械臂在观察姿态静止 2 s，等视觉初始化并确认跟踪稳定。观察姿态让相机准心大致对准目标，夹爪让开视线。
2. `approach`：接近到预抓取位姿，再走直线到抓取位姿。
3. `grasp`：夹爪合拢，停 0.3 s。
4. `extract`：沿单元轴线拉出 0.12 m。

观察姿态、底座位置和相机安装的规则见附录 A。

**耗时**：每份 1～1.5 min。

**检查**：规划工具为每份规划打印一段报告：

- 目标中心在图像中的像素位置应在画面中心附近（1280 × 720 时约 (640, 360)），偏移来自随机的对准误差；
- 相机到目标约 0.6 m；
- `approach` 段中，目标在相机坐标系中每帧移动约 2～6 mm。

退出时 move_group 可能以 -11 退出，这发生在关闭阶段，不影响已经写好的文件。

## 6. 渲染测试序列

```zsh
for name in plan_slot1_seed0 plan_slot0_seed1 plan_slot4_seed2; do
  blender --background --factory-startup --python scripts/blender/render_sequence.py -- \
      --out dataset/sequences/$name --plan dataset/plans/$name.json --seed 0 --device OPTIX
done
```

机械臂按规划的关节角摆放，用 `arm_description` 的 STL；相机装在肘部右侧的支架上，所以画面里有真实的前臂和夹爪。目标单元被夹住后随夹爪运动。灯光、HDRI、干扰物和镜头效果按 `--seed` 随机。

**输出**：每段序列一个目录，每帧有彩色图、深度图（16 位 PNG，单位 mm）、编号图和 `meta`（每根单元的真值位姿、相机位姿、关节角、阶段、是否已夹住），另有 `sequence.json`（相机内参、目标编号、帧率）。格式见附录 B。

**耗时**：RTX 5060 上每帧 1～2.6 s，3 段共约 20 min。不加传感器噪声的序列会开启降噪，所以更慢。

**检查**：每段序列的 `color/`、`depth/`、`index/`、`meta/` 的文件数都等于规划的帧数。

## 7. 回放

```zsh
for name in plan_slot1_seed0 plan_slot0_seed1 plan_slot4_seed2; do
  ros2 run rmcs_energy_trace sequence_replay dataset/sequences/$name dataset/replay/$name \
      --network dataset/models/yolo11n_seg_v1_openvino_model/yolo11n_seg_v1.xml --mask network --fps 0
done
```

`sequence_replay` 逐帧运行整条流程（检测、掩码、配准、跟踪、监督、参考位姿监视），与真值比较。常用选项：

- `--mask network|sam|none`：掩码来源。用 SAM 时还要给 `--sam assets/sam_vit_b_encoder.onnx assets/sam_vit_b_decoder.onnx`。
- `--fps 30`：模拟实时，处理一帧花了多久，就跳过这段时间里到达的帧。`--fps 0` 处理每一帧，结果完全确定。
- `--depth-noise`：深度噪声，默认 1.5（mm × z²）。

**输出**：

- `summary.txt`：汇总，也打印到终端；
- `frames.csv`：每帧一行的详细记录；
- `replay.mp4`：叠加了检测框、掩码、估计位姿轮廓（可信为绿色，不可信为黄色）和真值轮廓（白色）的视频。编码是 mp4v，有的播放器（例如 VS Code 内置的）放不了，换用 VLC 等播放器。

**检查**：`summary.txt` 中最重要的几行：

| 行 | 含义 | 期望 |
|---|---|---|
| `stable before motion start by` | 跟踪稳定比机械臂开始运动早多少秒 | 为正，约 1.5 s |
| `wrong accepts` | 判为可信、但中心误差大于 10 mm 或长轴误差大于 5° 的帧数 | 0 |
| `trusted center error` | 可信帧的中心误差 | 中位数约 1～2 mm |
| `target moved` | 参考位姿监视是否判为“目标被移动” | `no`（序列里没有碰动目标） |
| `extract: ... along axis` | 拉出阶段测到的沿轴位移 | 约 120 mm（拉出 0.12 m） |

按上面的命令，我们得到的结果如下。OMPL 和渲染的随机性会让数字略有不同。

| 序列 | 可信帧比例 | 错误接受 | 拉出确认 |
|---|---|---|---|
| `plan_slot1_seed0`（正上方） | 约 95% | 0 | 约 121 mm |
| `plan_slot0_seed1`（右上） | 100% | 0 | 约 120 mm |
| `plan_slot4_seed2`（正下方） | 约 46% | 0 | 没有可信帧 |

正下方的单元，从接近末段到拉出，夹爪一直挡在肘部相机和单元之间，所以可信帧少，也没法用视觉确认拉出。这是相机安装位置决定的，现在下方的单元交给操作手手动操作。

## 常见问题

- **Ultralytics 的输出跑到了 `~/runs/` 下**：`project=` 写了相对路径。改成绝对路径。
- **Blender 找不到显卡**：先用下面的命令列出 Cycles 能用的设备，确认显卡出现在 `OPTIX` 或 `CUDA` 中：

  ```zsh
  blender --background --factory-startup --python-expr "
  import bpy
  p = bpy.context.preferences.addons['cycles'].preferences
  for t in ('OPTIX', 'CUDA'):
      p.compute_device_type = t; p.get_devices()
      print(t, [(d.name, d.type) for d in p.devices])
  "
  ```

- **训练时报 CUDA 架构不支持**：PyTorch 的 CUDA 版本太旧。RTX 50 系列需要 CUDA 12.8 版本的 PyTorch，检查 `torch.cuda.get_arch_list()` 中有没有 `sm_120`。
- **pip 安装的 OpenCV 读不了 EXR**：渲染脚本已经在 Blender 里把 EXR 转成了 PNG，不需要 OpenEXR。
- **`sequence.json` 里的 `plan` 是另一台机器的路径**：它只记录渲染时用的规划文件，回放不读它。

## 附录 A：取矿动作规划的规则

参数都在 `config/pickup_plan.yaml`。

**场景**：世界坐标系 z 朝上，地面为 z = 0。中心块中心在 (0, 0, 0.7)，不旋转，正面朝 -y，机器人在 -y 一侧。第 k 个安装位的方向为 (cos(30° + 60°·k), 0, sin(30° + 60°·k))，单元中心在离中心块中心 0.15 m 处，长轴（指向封闭端）就是这个方向。目标安装位一定有单元，其余各以 0.7 的概率有单元（按种子）。

**抓取**：TCP 是 link_6 原点，也就是夹爪两指中间。抓取时单元中心在 TCP，link_6 的 x 轴平行于单元长轴，夹爪从机器人一侧垂直于长轴接近，夹住中段筒身。接近方向是 link_6 的 -z，预抓取位姿沿 link_6 的 +z 后退 0.10 m。直线段允许 link_6 与目标接触，link_5 不允许。拉出 0.12 m，因为单元的开口端套在柱子上约 100 mm。

**观察姿态**：

- 底座离地 0.4 m，q1 = q6 = 0。
- 底盘朝向负责水平对准，q2、q3 负责竖直对准：光轴穿过对准点。对准点是目标中心加随机偏移（水平 ±3 cm、竖直 ±2 cm，按种子），因为实际中操作手不可能把准心完全对准。
- q4 和 q5 从候选中选：q4 ∈ {0, +90°, -90°, 180°}，q5 ∈ {±1.2, ±1.57} rad。留下对准有解、无碰撞、并且相机看得见目标的组合，取离预抓取姿态最近的一个。
- “看得见”的判据：从光心到目标中心画一个半径 0.04 m 的圆柱，靠目标一端缩短 0.08 m，它不能碰到 link_4、link_5、link_6。这个圆柱检查完就移除，不参与运动规划。

**底座距离**：在 0.3～1.1 m 中扫描，取“预抓取、抓取、拉出都可达、无碰撞且视线不被挡”的最长连续区间的中点。

**相机**：装在 link_3 上，肘部右侧支架的顶端。光心 (0.13, 0.03, -0.10)，光轴指向 (0.03, 0.75, 0.035)，图像下方为 -x（都在 link_3 坐标系中），分辨率 1280 × 720，水平视场角 69°。这些参数连同支架折线写进规划文件。相机改装到云台上时，把 `camera.link` 设为底座，`camera.pose` 包含云台的俯仰和偏航。

**时间参数化**：TOTG，速度和加速度缩放 0.3，按 30 fps 重采样关节角。OMPL 每次的结果不同，所以重新生成时接近段的关节角会有差异；底座位姿和观察姿态是确定的。

## 附录 B：文件格式

### 规划文件

JSON，由 `pickup_plan` 写出，`render_sequence.py --plan` 读取。

| 字段 | 内容 |
|---|---|
| `fps` | 帧率 |
| `base` | 底座（URDF 根连杆）到世界的 4 × 4 位姿 |
| `hub.pose` | 中心块到世界的 4 × 4 位姿。中心块坐标系：原点在中心，x 右，y 朝后，z 上 |
| `hub.slots` | 6 个布尔值，第 k 个安装位有没有单元 |
| `hub.target` | 目标单元所在的安装位 |
| `camera.link`、`camera.pose` | 相机所在的连杆，以及相机（OpenCV 坐标系）到该连杆的 4 × 4 位姿 |
| `camera.width`、`camera.height`、`camera.hfov_deg` | 分辨率和水平视场角 |
| `camera.bracket` | 可选：支架折线（该连杆坐标系中的点列），只用来画出支架 |
| `frames[]` | 每帧的 `joints`（6 个关节角，rad）、`phase`（阶段名）和 `attached`（目标是否已被夹住） |

### 序列

`render_sequence.py` 每段序列写一个目录，帧号补零到 6 位：

| 路径 | 内容 |
|---|---|
| `sequence.json` | 相机内参、`depth_scale`、目标单元的编号 `target`、帧数 `frames`、用到的效果；按规划渲染时还有 `plan`（规划文件的路径）和 `fps` |
| `color/<帧号>.jpg` | 彩色图 |
| `depth/<帧号>.png` | 深度图：16 位无符号整数，单位 mm，沿光轴的平面深度。0 表示没有深度，远于 10 m 的像素也记为 0 |
| `index/<帧号>.png` | 每个像素属于第几根能量单元，0 为其他物体 |
| `meta/<帧号>.json` | 每根单元的 `index`、`where`、`pose`，相机到世界的位姿 `camera`；按规划渲染时还有 `joints`、`phase`、`attached` |

`pose` 是 4 × 4 行优先矩阵，mesh 到相机，OpenCV 相机坐标系，单位 m。原点在能量单元的包围盒中心，z 轴指向封闭端，与 C++ 的 `geometry::load_binary_stl` 相同。镜头效果只作用于彩色图，深度图和编号图不受影响。

不给 `--plan` 时，`render_sequence.py` 用内置的相机轨迹（扫视中心块 → 接近目标 → 随单元一起拉出），画面里的夹爪是两个方块。这个模式早于机械臂模型，以后会删除。

## 附录 C：回放工具的细节

**模拟实时**：`--fps F` 大于 0 时，处理一帧用了 t 秒，下一帧取第 `i + max(1, ceil(t · F))` 帧，中间的帧记为跳过。结果受机器负载影响。

**深度噪声**：读入的深度图加高斯噪声，标准差为 `--depth-noise`（mm）乘 z²（z 单位 m），另有 0.5% 的像素随机置为无效。随机数种子由帧号决定。`--depth-noise 0` 关闭噪声和无效像素。

**与真值比较**：每个有位姿输出的帧，找中心最近的真值单元，记下它的编号、是否为目标、中心误差和长轴误差。

**参考位姿监视**：`meta` 里有 `camera` 时，以世界坐标系作为底座坐标系运行参考位姿监视，`attached` 作为“已夹住”。出现“目标被移动”时，回放记下帧号，调用 `reset()`，然后继续，相当于上层停下机械臂、原地重新初始化。

**`summary.txt` 的内容**：

- 处理和跳过的帧数；
- 第一次可信、第一次稳定的帧，运动开始的帧，以及稳定比运动早多少秒；
- 每个阶段的可信比例、无法判断的帧数、可信帧的误差和错误接受数；
- 配准的尝试和成功次数，重新初始化的次数；
- 可信帧误差的中位数、90% 分位和最大值；
- 参考位姿监视的结果：建立参考的帧、“目标被移动”的帧、各阶段比较和不一致的帧数；
- 抓取前最后一个可信帧离抓取开始还有多少秒，也就是看不清目标的盲区；
- 拉出阶段的沿轴位移；
- 各步骤的平均耗时。

## 附录 D：重新生成运行所需的文件

一般直接用 `download_assets.sh` 下载。改了 mesh、网络或生成参数时，才需要重新生成。

**SAM 网络**：

```zsh
curl -L -o dataset/weights/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python3 scripts/training/export_sam.py dataset/weights/sam_vit_b_01ec64.pth assets \
    --check <一张图片> <x1,y1,x2,y2>
```

`--check` 用官方 `SamPredictor` 和导出的网络分割同一张图，打印两者掩码的 IoU。权重只当作数据读取，不执行其中的代码。

**M3T 模型**：生成时要渲染，需要 OpenGL 和显示器，所以生成工具默认不构建。

```zsh
colcon build --merge-install --packages-select rmcs_energy_trace \
    --cmake-args -DRMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON
env -u WAYLAND_DISPLAY DISPLAY=:0 ../../build/rmcs_energy_trace/m3t_generate
```

默认输出到 `assets/`，约 20 s，两个模型文件共约 150 MB。参数见 `m3t_generate --help`。运行时 `Tracker::Config::model_params` 必须与生成时一致。在容器里显示窗口前，先在宿主机执行 `xhost +SI:localuser:$(whoami)`。

**更新清单**：把新文件上传到 GitHub Release，再在 `assets/` 下运行 `sha256sum <文件名>`，用结果替换 `scripts/assets.sha256` 中对应的一行。
