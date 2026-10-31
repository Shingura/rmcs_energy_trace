# 仿真训练复现指南

本文介绍了在没有实物的情况下，使用仿真模拟实现检测网络、取矿动作规划、测试序列和回放结果的全过程。我们借助 Blender 渲染的合成图训练 yolo，并使用 Blender 渲染的 RGB-D 序列进行测试，这在没有实物测试条件或没有能力与精力建立能量单元的训练数据集时显得格外有用。

## 流程总览

本文假设你的宿主机使用 NVIDIA 显卡，并已在 Linux 下安装好了驱动。

| 阶段 | 工具 | 输入 | 输出（`dataset/` 下） | 运行位置 |
|---|---|---|---|---|
| 0. 准备 | `download_assets.sh`、`build-rmcs` | — | `assets/`（包内） | 开发环境 |
| 1. 下载环境贴图 | `scripts/blender/download_hdri.sh` | — | `hdri/` | 任意 |
| 2. 渲染训练图 | `scripts/blender/render_dataset.py` | `hdri/` | `train/render/v1/` | 宿主机 |
| 3. 生成数据集 | `scripts/training/make_yolo_dataset.py` | `train/render/v1/` | `train/yolo/v1/` | 任意 |
| 4. 训练和导出 yolo | `yolo`、`scripts/training/export_yolo.py` | `train/yolo/v1/` | `train/runs/v1/`、`models/` | 宿主机 |
| 5. 规划取矿动作 | `pickup_plan` | `config/pickup_plan.yaml` | `plans/` | 开发环境 |
| 6. 渲染测试序列 | `scripts/blender/render_sequence.py --plan` | `plans/`、`hdri/` | `sequences/` | 宿主机 |
| 7. 回放 | `sequence_replay` | `sequences/`、`models/` | `replay/` | 开发环境 |

第 2～4 步是为了训练 yolo 检测网络。如果你已经下载或训练好了 yolo（和可选的 yolo-seg）权重，可直接从第 5 步开始。

`dataset/` 下存放了许多用于渲染和训练的大文件，超过了 git 限制，因此没有放入仓库。除了实拍图（可自备）以外，全部可以自行按本文重新生成：

```text
dataset/
├── real/images/      实拍测试图
├── weights/          外部下载的原始权重（SAM 官方权重等）
├── hdri/             Blender 用的环境贴图
├── train/
│   ├── render/v1/    渲染的训练图
│   ├── yolo/v1/      Ultralytics 数据集
│   └── runs/v1/      训练输出，含 weights/best.pt
├── models/           导出网络
├── plans/            取矿动作规划文件
├── sequences/        按规划渲染的测试序列
└── replay/           回放输出
```

## 0. 准备

## 软件

| 软件 | 版本 | 用途 |
|---|---|---|
| Blender | 4.5 LTS，便携版即可 | 渲染。脚本在 Blender 自带的 Python 中运行，不需要另装 Python 包 |
| Python 环境 | Python 3，`ultralytics==8.3.251`、`opencv-python`、`numpy`、`openvino` | 生成数据集、训练、导出 |
| PyTorch | 与显卡匹配的 CUDA 版本；RTX 50 系列（sm_120）需要 CUDA 12.8 版本 | 训练 |
| SAM 导出（可选） | `torch`、`torchvision`、`segment-anything`、`onnx`、`onnxruntime` | 只在重新导出 SAM 网络时需要 |

安装 Ultralytics 后，运行 `yolo settings sync=False` 关闭使用统计上传。

### 代码和运行所需的文件

1. 在 `rmcs_ws/src` 下克隆本仓库。请确保选取正确的分支，取矿动作规划和 Blender 渲染还要用到同一工作区里的 `arm_description`（机械臂的 URDF 和 STL）和 `arm_moveit_config`（MoveIt 配置）。
2. 下载运行所需的网络和 M3T 模型：

   ```zsh
   scripts/download_assets.sh
   ```

   脚本会自行下载所需文件，并按照 `scripts/assets.sha256` 中的文件清单和 SHA256 逐个校验。已存在且校验通过的文件会跳过。请注意，检测网络（YOLO11n-seg）需要自行依照第 2～4 步训练。


## 1. 环境贴图

为了在 blender 中创建出更拟真的背景环境，我们可以先下载一些环境贴图。

```zsh
scripts/blender/download_hdri.sh
```

脚本会从 Poly Haven（CC0）下载 13 个室内场景的 HDRI（2k，`.hdr`）到 `dataset/hdri/`，合计约 83 MB。

## 2. 渲染训练图

```zsh
blender --background --factory-startup --python scripts/blender/render_dataset.py -- \
    --out dataset/train/render/v1 --count 2000 --seed 0 --device OPTIX
```

`--device` 可选 `CPU`、`CUDA`、`OPTIX`，这里默认使用了 OPTIX，你可以自行测试，选择速度最快的选项。作为对比，在笔者使用的 Ryzen 5 9600X + RTX 5060 上，渲染 2000 帧画面大约需要 50 min。

该脚本将会在每帧随机搭建一个取矿场景（共 2000 帧），包括：

- 资源区的地面、墙和中心块，中心块上 0～6 根能量单元；
- 地上横躺或竖立的能量单元；
- 干扰物，其中一部分是与能量单元大小相近的深色圆柱；
- 从画面边缘伸进来的遮挡物。

每帧会输出 4 个文件：

| 目录 | 内容 |
|---|---|
| `images/` | 彩色图（JPEG） |
| `index/` | 编号图：每个像素属于第几根能量单元，0 为其他物体 |
| `full/` | 只渲染能量单元时的编号图，用来算每根单元被挡住的比例 |
| `meta/` | 相机内参、每根单元的位姿、这一帧用到的效果 |

每帧画面中的材质、光源、HDRI 和相机位姿均为随机。在输出的彩色图，脚本还会按照一定概率叠加眩光、脏镜头、虚焦、运动模糊、雾化、暗角、色散、传感器噪声，以及曝光和白平衡的变化，从而更加贴近真实工况。

## 3. 生成数据集

```zsh
python3 scripts/training/make_yolo_dataset.py dataset/train/render/v1 dataset/train/yolo/v1
```

脚本将会在目标目录下的每张图片里用每根单元的可见像素生成多边形标签。当可见像素少于 300 个，或可见部分不到完整轮廓的 20% 时，该单元将不会被标注。脚本还会随机取 10% 的图片作为验证集。

打开 `dataset/train/yolo/v1/preview/` 下的图片，应当可以看到图中有绿色方框框住能量单元，橙色色块覆盖能量单元。

## 4. 训练和导出

在宿主机上训练：

```zsh
yolo segment train model=yolo11n-seg.pt data=$PWD/dataset/train/yolo/v1/data.yaml imgsz=640 \
    epochs=100 batch=16 project=$PWD/dataset/train/runs name=v1
```

第一次运行时，yolo 会自行调用 Ultralytics，从其官方 GitHub 仓库下载预训练权重 `yolo11n-seg.pt`，用于初始化本次训练；它具有通用的视觉识别能力，还需要用本次项目的能量单元数据继续训练，以达到更好的效果。

作为参考，在笔者的 RTX 5060 上，100 轮训练需要约 23 min。

训练完成后，还需要将得到的 `.pt` 权重文件导出为 OpenVINO，供 C++ 检测器使用。

```zsh
# 先将权重文件移动并改名
cp dataset/train/runs/v1/weights/best.pt dataset/models/yolo11n_seg_v1.pt
python3 scripts/training/export_yolo.py dataset/models/yolo11n_seg_v1.pt dataset/models
```

可以使用实拍图测试训练效果：

  ```zsh
  ros2 run rmcs_energy_trace energy_unit_detect \
      dataset/models/yolo11n_seg_v1_openvino_model/yolo11n_seg_v1.xml <图片或目录> <输出目录>
  ```

## 5. 规划取矿动作

我们先用 moveit 规划出机械臂的运动轨迹，在下一步中依据这个运动轨迹渲染出模拟相机的视角，模拟实际工况中的相机运动。

```zsh
ros2 launch rmcs_energy_trace pickup_plan.launch.py
```

脚本将会启动 move_group，按照 `config/pickup_plan.yaml` 规划 3 份取矿动作，并写入到 `dataset/plans/plan_slot<k>_seed<s>.json`，然后自动退出。其中，`<s>` 表示能量单元在矿站的安装位置，依次为 1（正上方）、0（右上）、4（正下方）。

每份规划都按下列操作流程：

1. `hold`：机械臂在观察姿态静止 2 s，等视觉初始化并确认跟踪稳定。观察姿态让相机准心大致对准目标，夹爪让开视线。
2. `approach`：接近到预抓取位姿，再走直线到抓取位姿。
3. `grasp`：夹爪合拢，停 0.3 s。
4. `extract`：沿单元轴线拉出 0.12 m。

## 6. 渲染测试序列

接下来，我们依据上一步中规划的机械臂运动轨迹，开始渲染模拟相机的视角序列。

```zsh
for name in plan_slot1_seed0 plan_slot0_seed1 plan_slot4_seed2; do
  blender --background --factory-startup --python scripts/blender/render_sequence.py -- \
      --out dataset/sequences/$name --plan dataset/plans/$name.json --seed 0 --device OPTIX
done
```

每段序列被输出到一个目录，每帧均有彩色图、深度图、编号图和 `meta`（包括每根单元的真值位姿、相机位姿、关节角、阶段、是否已被夹住）。

作为参考，在笔者的 RTX 5060 上，每帧渲染耗时约 1～2.6 s，共约 20 min。

## 7. 回放

```zsh
for name in plan_slot1_seed0 plan_slot0_seed1 plan_slot4_seed2; do
  ros2 run rmcs_energy_trace sequence_replay dataset/sequences/$name dataset/replay/$name \
      --network dataset/models/yolo11n_seg_v1_openvino_model/yolo11n_seg_v1.xml --mask network --fps 0
done
```

这将会逐帧运行整条流程（包括 yolo 检测、yolo-seg / SAM 生成掩码、点云配准、M3T 跟踪、状态机监督、参考位姿监视）。

可使用 `-mask network|sam|none` 指定掩码来源，分别为 yolo-seg、SAM 和不生成掩码。使用 SAM 时，还需要指定 SAM 模型文件的位置，如 `--sam assets/sam_vit_b_encoder.onnx assets/sam_vit_b_decoder.onnx`。

在 `replay.mp4` 中，可以看到叠加了检测框、掩码、估计位姿轮廓（可信为绿色，不可信为黄色）和真值轮廓（白色）的视频（推荐使用 VLC 打开）。

在 `summary.txt` 中，可以看到如下调试信息：

| 行 | 含义 | 期望 |
|---|---|---|
| `stable before motion start by` | 跟踪稳定比机械臂开始运动早多少秒 | 为正，约 1.5 s |
| `wrong accepts` | 判为可信、但中心误差大于 10 mm 或长轴误差大于 5° 的帧数 | 0 |
| `trusted center error` | 可信帧的中心误差 | 中位数约 1～2 mm |
| `target moved` | 参考位姿监视是否判为“目标被移动” | `no`（序列里没有碰动目标） |
| `extract: ... along axis` | 拉出阶段测到的沿轴位移 | 约 120 mm（拉出 0.12 m） |

我们得到的结果如下。OMPL 和渲染的随机性会让数字略有不同。

| 序列 | 可信帧比例 | 错误接受 | 拉出确认 |
|---|---|---|---|
| `plan_slot1_seed0`（正上方） | 约 95% | 0 | 约 121 mm |
| `plan_slot0_seed1`（右上） | 100% | 0 | 约 120 mm |
| `plan_slot4_seed2`（正下方） | 约 46% | 0 | 没有可信帧 |

可以看到，正下方的单元由于视角原因，机械臂与夹爪的遮挡较为严重，导致正确率大幅下降，这是由相机安装位置决定的。
