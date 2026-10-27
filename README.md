# rmcs_energy_trace

RMCS 工程机器人的能量单元位姿估计组件。它读取 RGB-D 相机图像，输出能量单元在相机坐标系下的 6D 位姿，供机械臂规划抓取。

整个流程在 CPU 上运行，不依赖 NVIDIA 显卡。

## 设计

流程分为 5 步：

1. **检测**：YOLO 在彩色图中框出能量单元。用 OpenVINO 在 CPU 上推理。网络可以是检测网络，也可以是 YOLO11-seg 分割网络。分割网络会同时输出每个检测的掩码，这时第 2 步可以直接用它。
2. **分割**：SAM 以检测框为提示，分出目标能量单元的掩码。检测框里有中心块和相邻的单元，没有掩码时配准会被拒绝。
3. **初始化**：把掩码内的深度像素转换为点云，用 PPF 粗配准，再用 ICP 精配准，经深度检查后得到初始位姿。
4. **跟踪**：后续帧用 M3T 从上一帧的位姿开始跟踪，不再每帧做完整配准。
5. **监督**：根据深度一致性、位姿跳变和检测框（有检测结果时），判断跟踪结果是否可信。连续不可信时回到第 1 步重新初始化。

整体结构以及 SAM 的用法、监督的判断规则和阈值，参考了 [RCIA_Benmao_Vision](https://github.com/BenmaoNeko/RCIA_Benmao_Vision)（配套文章的许可证为 CC BY-NC-SA 4.0）。该项目用 SAM 和 FoundationPose 做初始化，需要 NVIDIA 显卡。本项目改用 CPU 上的点云配准，代码没有从该项目复制。

## 输出接口

| 名称 | 类型 | 说明 |
|---|---|---|
| `/energy_trace/is_pose_valid` | `bool` | 位姿是否可信 |
| `/energy_trace/pose` | `Eigen::Isometry3d` | 相机坐标系下能量单元的位姿，单位 m |

位姿的原点在能量单元的几何中心，z 轴沿长轴指向 Ø95 封闭头。能量单元绕长轴近似 3 重对称，绕 z 轴的转角无法可靠观测，x、y 轴的方向没有意义，使用方只能用原点和 z 轴。

注意：`rmcs_core` 中能量单元的参考点在底面以上 72.5 mm 处，本组件的原点在底面以上 75 mm 处，两者相差 2.5 mm。

## 目录
```text
rmcs_energy_trace/
├── src/
│   ├── component.cpp              RMCS 组件：声明输入输出接口，调度各模块
│   │
│   ├── detection/                 在彩色图中检测能量单元
│   │   ├── detector.hpp/.cpp      Detector：缩放并填充图像、OpenVINO 推理，按输出自动判断检测网络或分割网络，返回检测框和得分，分割网络还返回掩码
│   │   ├── decode.hpp/.cpp        decode_detections：把网络输出和 letterbox 参数解码为检测结果（阈值、NMS、坐标映射、掩码），不依赖推理
│   │   └── yolo11.hpp             YOLO11 网络的输入输出约定（640 × 640；检测网络输出 [1, 5, 8400]，分割网络输出 [1, 37, 8400] 和 [1, 32, 160, 160]）和预处理设置
│   │
│   ├── segmentation/              以检测框为提示分割目标
│   │   └── segmenter.hpp/.cpp     Segmenter：SAM 的预处理、编码器和解码器推理（OpenVINO）、候选掩码的选择
│   │
│   ├── registration/              点云配准，得到初始位姿
│   │   ├── registration.hpp/.cpp  去平面、PPF 粗配准、ICP 精配准、绕长轴扫描
│   │   ├── depth_check.hpp/.cpp   DepthCheck：把 mesh 按位姿渲染后与实测深度逐像素比较，配准和监督共用
│   │   └── point_to_plane.hpp/.cpp ICP 每次迭代的求解：约束不足的方向不移动，不会得到 NaN
│   │
│   ├── tracking/                  逐帧跟踪
│   │   ├── tracker.hpp/.cpp       Tracker：封装 M3T，从初始位姿开始，逐帧输入彩色图和深度图，输出位姿
│   │   ├── mesh_obj.hpp/.cpp      把 mesh 写成 M3T 读取的 OBJ 文件（单位 m，原点在中心）
│   │   └── model_files.hpp        M3T 模型文件的文件名和生成参数，生成工具和运行时共用
│   │
│   ├── supervision/               判断跟踪结果是否可信
│   │   └── supervisor.hpp/.cpp    Supervisor：每帧的健康检查和“搜索 / 跟踪”状态机
│   │
│   ├── pipeline/                  把各模块串成完整流程
│   │   ├── pipeline.hpp/.cpp      Pipeline：每帧输入彩色图和深度图，输出状态、位姿、是否可信和诊断信息
│   │   └── reference_monitor.hpp/.cpp  ReferenceMonitor：建立底座系下的参考位姿，判断目标是否被移动
│   │
│   ├── geometry/                  各模块共用的几何类型
│   │   ├── mesh.hpp/.cpp          读取二进制 STL（mm 转 m，原点移到中心），在表面均匀采样
│   │   ├── camera.hpp             Camera：RGB-D 相机的分辨率、内参和深度单位
│   │   └── depth_cloud.hpp/.cpp   深度图在框（和掩码）内反投影为点云
│   │
│   ├── synthetic/                 C++ 简易渲染器生成的合成深度图和彩色图，只供测试使用
│   │   └── depth_renderer.hpp/.cpp     按给定位姿渲染深度图、彩色图和实例编号图
│   │
│   └── utility/
│       └── pimpl.hpp              隐藏实现细节的辅助宏
│
├── test/                          离线测试，不需要相机和显示器
│   ├── registration.cpp           配准：在合成深度图上统计中心误差和长轴误差
│   ├── depth_mask.cpp             深度图转点云时掩码筛选和收缩的单元测试
│   ├── point_to_plane.cpp         ICP 求解的单元测试，包括约束不足的输入
│   ├── segmentation.cpp           分割：中心块场景中 SAM 掩码与真值的重合度，以及用它配准（缺少 SAM 网络时跳过）
│   ├── supervision.cpp            监督状态机和参考位姿监视的单元测试
│   └── tracking.cpp               跟踪：在合成的连续帧上统计误差和耗时，并接上监督（缺少 M3T 模型时跳过）
│
├── tool/                          离线工具
│   ├── energy_unit_detect.cpp     对图片或目录运行检测（可选再用 SAM 分割），保存画好框和掩码的图片
│   ├── sequence_replay.cpp        在渲染的连续 RGB-D 序列上运行 Pipeline，输出叠加画面的视频和相对真值的误差
│   ├── m3t_generate.cpp           生成 M3T 模型文件（需要 OpenGL 和显示，默认不构建）
│   ├── pickup_plan/
│   │   └── pickup_plan.cpp        用 MoveIt 规划中心块取矿的全过程，导出规划文件供 render_sequence.py 渲染
│   └── grasp_sim/                 MoveIt 抓取仿真
│       ├── grasp_sim.cpp          仿真主程序：渲染、配准、生成抓取候选、规划并回放
│       ├── grasp_candidates.hpp   根据能量单元的中心和长轴生成抓取候选
│       └── joint_state_player.hpp 发布 /joint_states，按时间插值回放轨迹
│
├── third_party/m3t/               随仓库带入的 M3T 源码（MIT），上游文件只改了模型加载中的一处
│   ├── README.md                  来源、提交号、删减内容和本地修改
│   ├── include/m3t/               M3T 源码
│   ├── third_party/               M3T 自带的第三方头文件
│   └── stubs/                     占位文件：去掉 OpenGL、GLFW、highgui 依赖，以及供 clangd 使用的空 omp.h
│
├── scripts/
│   ├── download_assets.sh         下载运行所需的网络和 M3T 模型到 assets/，并校验 SHA256
│   ├── assets.sha256              要下载的文件清单和校验值
│   ├── blender/                   在 Blender 自带的 Python 中运行
│   │   ├── render_dataset.py      随机搭建取矿场景，渲染检测网络的训练图和编号图
│   │   ├── render_sequence.py     渲染一段连续的 RGB-D 序列，供 sequence_replay 回放；可按取矿规划驱动机械臂和相机
│   │   ├── arm.py                 机械臂：读 URDF、正运动学、导入各连杆的 STL、相机支架
│   │   └── download_hdri.sh       下载渲染用的室内 HDRI
│   └── training/                  在装有 Ultralytics 的 Python 环境中运行
│       ├── make_yolo_dataset.py   把渲染结果转换成 Ultralytics 的分割数据集
│       ├── export_yolo.py         把 Ultralytics 的 .pt 权重导出为 OpenVINO 格式
│       └── export_sam.py          把 SAM 的官方权重导出为编码器和解码器两个 ONNX 网络
│
├── config/
│   ├── energy_trace.yaml          组件参数示例
│   ├── grasp_sim.yaml             抓取仿真参数
│   └── pickup_plan.yaml           取矿动作规划的参数：场景、相机安装、观察姿态、各段时长和速度
│
├── launch/
│   ├── grasp_sim.launch.py        启动抓取仿真（MoveIt、仿真节点、RViz）
│   └── pickup_plan.launch.py      启动 move_group 和取矿动作规划，规划完自动退出
├── rviz/grasp_sim.rviz            抓取仿真的 RViz 配置
├── meshes/energy_unit.stl         能量单元的 mesh
├── assets/                        运行所需的网络和 M3T 模型，由 download_assets.sh 下载，不进入仓库
├── dataset/                       开发用的本地数据，不进入仓库（见下文）
├── CMakeLists.txt、package.xml、plugins.xml   
├── .clangd                        
├── .clang-format、.clang-tidy     
└── AGENTS.md                      
```

## 运行所需的文件

网络和 M3T 模型太大，不放进仓库。编译前在包目录下运行：

```zsh
scripts/download_assets.sh
```

脚本按 `scripts/assets.sha256` 把下面的文件下载到 `assets/`，并逐个校验 SHA256。已存在且校验通过的文件会跳过，所以可以重复运行；有文件下载失败或校验不通过时退出码为 1。`--dir` 指定其他目录，`--base-url` 指定其他下载地址。

| 文件 | 大小 | 用途 | 来源 |
|---|---|---|---|
| `sam_vit_b_encoder.onnx` | 约 343 MB | SAM 编码器 | Meta 官方权重，用 `scripts/training/export_sam.py` 导出 |
| `sam_vit_b_decoder.onnx` | 约 16 MB | SAM 解码器 | 同上 |
| `energy_unit.obj` | 约 106 KB | M3T 使用的 mesh | `m3t_generate` 生成 |
| `energy_unit_region_model.bin`、`energy_unit_depth_model.bin` | 约 74 MB、70 MB | M3T 的区域模型和深度模型 | `m3t_generate` 生成，需要 OpenGL 和显示器 |

检测网络（YOLO11n-seg）暂时不在清单中，按“合成训练数据”一节训练并导出。文件放在 GitHub Release `assets-v1`。编译时 `assets/` 会安装到包的 share 目录，随安装目录一起同步到机器人；目录不存在时跳过，不影响编译。M3T 加载模型时只比较 OBJ 的文件名，所以 `assets/` 放在哪里都能用（见 `third_party/m3t/README.md`）。

更新某个文件后，把新文件上传到 Release，再在 `assets/` 下运行 `sha256sum <文件名>`，用结果替换 `scripts/assets.sha256` 中对应的一行。

`dataset/` 存放开发用的数据，也不进入仓库：

```text
dataset/
├── real/images/                   实拍测试图片
├── weights/                       外部下载的原始权重：SAM 官方权重 sam_vit_b_01ec64.pth、Ultralytics 预训练权重
├── hdri/                          Blender 用的环境贴图
├── train/                         检测网络的训练：render/<版本>/ 渲染结果，yolo/<版本>/ 数据集，runs/<版本>/ 训练输出
├── models/                        导出、还没放进 Release 的网络（YOLO11n-seg 的 OpenVINO 模型）
├── plans/                         取矿动作规划文件
├── sequences/                     按规划渲染的 RGB-D 序列
└── replay/                        sequence_replay 的输出
```

## 配准

入口是 `registration::Registration::register_depth(depth, camera, roi, mask)`，`mask` 可以为空。流程：

1. 框（和掩码）内的深度像素转成点云，降采样到 3 mm 间距，去掉离群点。有掩码时背景已经去掉，不去平面。只有检测框时，如果点云中有大平面（如地面），先试去掉这个平面的点云，再试原始点云（大平面也可能是能量单元的端面）。
2. PPF 粗配准（OpenCV `surface_matching`）：按票数取前 5 个位姿。模型点云只保留 mesh 的外表面，开口端的内腔从外面看不到，不参与配准。
3. 每个位姿依次做 ICP、绕长轴扫描、再做 ICP。绕长轴扫描在 120° 内取 12 个角度，选深度检查最好的一个：转角不影响中心和长轴，但筋的位置对不上时支持率偏低，ICP 又很难把转角转回来。
4. 三个阶段的位姿都按下面的指标比较：先看是否通过验收，再比分数（内点率 + 支持率 − 2 × 空闲空间违例比例）。最好的一个通过验收才返回。

检查方法（`registration::DepthCheck`，监督也用它）：按候选位姿把能量单元的 mesh 投影到深度图上，逐像素比较：

| 指标 | 含义 | 接受条件 |
|---|---|---|
| `inlier_ratio` | 场景点中离模型小于 4 mm 的比例 | ≥ 0.90 |
| `support_ratio` | 模型投影处，实测深度与模型相差小于 5 mm 的比例 | ≥ 0.88 |
| `free_space_ratio` | 模型投影处，实测深度比模型远 5 mm 以上的比例，即“本该挡住视线的地方是空的” | ≤ 0.06 |
| `model_pixels` | 模型投影的像素数 | ≥ 300 |

`support_ratio` 和 `free_space_ratio` 只统计有实测深度的像素。实测深度比模型近的像素算作被遮挡，不算违例，但计入分母：这样一个整体落在真实表面后面的错误位姿不能靠少数像素拿到高支持率，代价是没有掩码时，目标被挡住较多（约 12% 以上）会被拒绝。

有掩码时，掩码只决定一个像素能否算作支持。掩码外的像素属于别的东西，深度相同或更近都说明不了位姿对错，不统计；掩码外看到比模型更远的背景，说明模型所在处是空的，仍记为违例。所以加掩码后检查不会变松，挡在目标前面的物体也不会造成拒绝。

阈值是在合成数据上定的，换成真实相机后需要重新确定。

整个流程没有随机性，结果只取决于输入。同一个实例不能在多个线程中同时调用 `register_depth`：OpenCV 的 PPF 匹配会修改检测器的内部状态。

## 测试与工具

配准测试在合成深度图上运行 7 组：

| 组 | 场景 | 次数 | 合格条件 |
|---|---|---|---|
| A | 任意朝向，无背景 | 50 | 成功率 ≥ 90%，接受但错误的结果不超过 1 次 |
| B | 竖立在地面上，相机向下倾斜 | 50 | 同上 |
| 部分视野 | 检测框裁窄到 60% | 50 | 同上 |
| 错误物体 | 缩小到 0.8 倍的能量单元 | 30 | 全部拒绝 |
| 错误物体 | 半径 40 mm、高 120 mm 的圆柱 | 30 | 全部拒绝 |
| 遮挡加掩码 | 同 B，目标前方有细杆挡住约三分之一，传入精确掩码 | 20 | 同 A |
| 掩码不放松检查 | 0.9 倍的能量单元，分别只用框和加精确掩码 | 20 | 加掩码后被接受的次数不多于只用框 |

“成功”指中心误差小于 3 mm，且长轴方向误差小于 2°。完整运行约 6 分钟。

```zsh
/workspaces/RMCS/rmcs_ws/build/rmcs_energy_trace/test_registration
```

检测工具对一张图片或一个目录中的图片运行检测，并保存画好框的结果。网络是分割网络时，把它输出的掩码用青色叠加在图上。给出 `--sam` 时，再对每个检测框运行 SAM，把 SAM 的掩码用品红色叠加在图上：

```zsh
ros2 run rmcs_energy_trace energy_unit_detect <network.xml> <图片或目录> <输出目录> [--conf 0.25] \
    [--sam <encoder.onnx> <decoder.onnx>]
```

`network.xml` 由 `scripts/training/export_yolo.py` 从 Ultralytics 的 `.pt` 权重导出。

分割网络的掩码按 Ultralytics 的 `process_mask_native`（`retina_masks=True`）解码：掩码系数乘原型得到 160 × 160 的 logit，取出未填充图像的区域，双线性缩放到原图尺寸，logit 大于 0 的像素为目标，检测框以外置零。只对 NMS 之后保留的检测解码，并且只计算框内的像素。`Detector::Detection::mask` 的约定与 `Segmenter::Result::mask` 相同。`Timing::mask` 是掩码解码的耗时，不计入 `postprocess`。`test_detection` 用构造的输出张量测试解码，不需要网络文件。

`geometry::depth_to_cloud` 把深度图在框（和掩码）内反投影为点云，掩码先按参数 `mask_erosion_px` 向内腐蚀（配准默认 3 px）。`test_depth_mask` 是它的单元测试。

## 合成训练数据

我们没有标注好的实拍图，检测网络用 Blender 渲染的合成图训练。渲染脚本在 `scripts/blender/`，数据集和导出脚本在 `scripts/training/`。HDRI 放在 `dataset/hdri/`，渲染结果、数据集和训练输出放在 `dataset/train/` 下。

每帧重新搭建一个取矿场景：

- 资源区的地面、墙和中心块。中心块按规则手册图 4-22 建模，上面插 0～6 根能量单元；
- 地上横躺或竖立的能量单元；
- 干扰物，其中一部分是与能量单元大小相近的深色圆柱；
- 从画面边缘伸进来的遮挡物，模拟自己的夹爪和机械臂。

能量单元只用官方配色。材质、光源、环境贴图和相机位姿都随机。彩色图再按各自的概率叠加相机和镜头的效果：眩光、脏镜头、虚焦、运动模糊、雾化、暗角、色散、传感器噪声，以及曝光和白平衡的变化。

需要的软件：

- Blender 4.5 LTS，官方便携版即可。渲染脚本在 Blender 自带的 Python 中运行。
- 一个装有 `ultralytics`、`opencv-python` 和 `numpy` 的 Python 环境，用来生成数据集和训练。

在包目录下按顺序运行：

```zsh
# 1. 下载 13 个室内 HDRI（Poly Haven，CC0）到 dataset/hdri/，已有的文件会跳过
scripts/blender/download_hdri.sh

# 2. 渲染 2000 帧，--device 可选 CPU、CUDA、OPTIX
blender --background --factory-startup --python scripts/blender/render_dataset.py -- \
    --out dataset/train/render/v1 --count 2000 --seed 0 --device OPTIX

# 3. 生成 Ultralytics 的分割数据集
python3 scripts/training/make_yolo_dataset.py dataset/train/render/v1 dataset/train/yolo/v1

# 4. 训练 YOLO11n-seg（预训练权重 yolo11n-seg.pt 第一次运行时自动下载）
yolo segment train model=yolo11n-seg.pt data=$PWD/dataset/train/yolo/v1/data.yaml imgsz=640 \
    epochs=100 batch=16 project=$PWD/dataset/train/runs name=v1

# 5. 导出为 OpenVINO，供 C++ 检测器使用
python3 scripts/training/export_yolo.py dataset/train/runs/v1/weights/best.pt dataset/models
```

渲染每帧平均约 1.5 s（RTX 5060，OPTIX），只用 CPU 约 7.5 s（Ryzen 9600X）。已有结果的帧会跳过，中断后用同一条命令继续。每帧输出 4 个文件：

| 目录 | 内容 |
|---|---|
| `images/` | 彩色图（JPEG） |
| `index/` | 编号图：每个像素属于第几根能量单元，0 为其他物体 |
| `full/` | 只渲染能量单元时的编号图，与 `index/` 对比得到各单元被遮挡的比例 |
| `meta/` | 相机内参、各单元的位姿（mesh 到相机，OpenCV 坐标系，单位 m）和本帧用到的效果 |

镜头效果只作用于彩色图，编号图不受影响。

`make_yolo_dataset.py` 用每根单元的可见像素生成多边形标签。可见像素少于 300 个，或可见部分不到完整轮廓的 20%，这根单元就不标注，留在图中当背景。被遮挡分成几块时，按 Ultralytics 转换 COCO 的做法连成一个多边形。验证集按帧号散列取 10%，`preview/` 下有画出标签的预览图。`data.yaml` 不写 `path`，数据集目录可以整体移动。

训练命令里的 `project=` 要写绝对路径。写相对路径时，Ultralytics 8.3.251 会把输出放到 `~/runs/<任务类型>/` 下。

## 连续序列

`render_sequence.py` 在 Blender 中渲染一段连续的 RGB-D 序列，用来离线回放整条位姿估计流程。场景用 `render_dataset.py` 的搭建函数，整段序列只搭建一次。灯光、HDRI、地上的单元、干扰物和镜头效果按 `--seed` 随机，镜头效果和曝光整段相同。用 Blender 自带的 Python 运行：

```zsh
blender --background --factory-startup --python scripts/blender/render_sequence.py -- \
    --out dataset/sequences/<名称> [--plan <规划文件>] [--frames 200] [--seed 0] \
    [--width 1280] [--height 720] [--samples 32] [--device CPU] [--hdri-dir dataset/hdri]
```

已有 `meta/` 的帧会跳过，中断后用同一条命令继续。

**按规划渲染。** 给出 `--plan` 时，机械臂按规划文件中每帧的关节角摆放，各连杆用 `arm_description` 包的 STL。相机装在规划文件指定的连杆上，画面里能看到真实的前臂和夹爪。中心块的位姿和各安装位有没有单元也取自规划文件，目标单元在被夹住的帧随 `link_6` 运动。帧数、分辨率和视场角都由规划文件决定，`--frames`、`--width`、`--height` 不起作用。规划文件是 JSON：

| 字段 | 内容 |
|---|---|
| `fps` | 帧率 |
| `base` | 底座（URDF 根连杆）到世界的 4 × 4 位姿 |
| `hub.pose` | 中心块到世界的 4 × 4 位姿。中心块系：原点在中心，x 右，y 朝后，z 上 |
| `hub.slots` | 6 个布尔值，第 k 个安装位（方向 30° + 60°·k）有没有单元 |
| `hub.target` | 目标单元所在的安装位 |
| `camera.link`、`camera.pose` | 相机所在的连杆，以及相机（OpenCV 坐标系）到该连杆的 4 × 4 位姿 |
| `camera.width`、`camera.height`、`camera.hfov_deg` | 分辨率和水平视场角 |
| `camera.bracket` | 可选：支架折线（该连杆坐标系中的点列），只用于画出支架 |
| `frames[]` | 每帧的 `joints`（6 个关节角，rad）、`phase`（阶段名）和 `attached`（目标是否已被夹住） |

世界坐标系：z 朝上，地面为 z = 0，中心块正面朝 -y，机器人在 -y 一侧。相机装在云台上时，`camera.link` 取底座，`camera.pose` 包含云台的俯仰和偏航。

**内置轨迹。** 不给 `--plan` 时，中心块上至少有 3 根能量单元，相机（水平视场角 69°）按内置轨迹运动，每帧有约 2 mm、0.3° 的抖动，分三段：

1. 前 20% 的帧：停在中心块前方约 1.1 m 处，视线左右扫过中心块；
2. 到 75% 的帧：接近中心块上的目标单元，停在单元外侧约 0.3 m 处；
3. 其余帧：取出目标单元，单元和相机一起沿单元轴线向外移动 12 cm。

从 50% 的帧起，两根深色的方块从画面下边缘伸进来，代替夹爪。

每段序列一个目录，帧号补零到 6 位：

| 路径 | 内容 |
|---|---|
| `sequence.json` | 相机内参、`depth_scale`、目标单元的编号 `target`、帧数 `frames` 和用到的效果；按规划渲染时还有规划文件的路径 `plan` 和帧率 `fps` |
| `color/<帧号>.jpg` | 彩色图 |
| `depth/<帧号>.png` | 深度图：16 位无符号整数，单位 mm，沿光轴的平面深度，0 表示没有深度（更远于 10 m 的像素也记为 0） |
| `index/<帧号>.png` | 每个像素属于第几根能量单元，0 为其他物体 |
| `meta/<帧号>.json` | 每根单元的 `index`、`where` 和 `pose`，相机（OpenCV 坐标系）到世界的位姿 `camera`；按规划渲染时还有这一帧的 `joints`、`phase` 和 `attached` |

`pose` 是 4 × 4 行优先矩阵，mesh 到相机，OpenCV 相机坐标系，单位 m。原点在能量单元的包围盒中心，z 轴指向封闭端，与 `geometry::load_binary_stl` 相同。镜头效果只作用于彩色图，深度图和编号图不受影响。

## 取矿动作规划

`pickup_plan` 在中心块场景里用 MoveIt 规划机械臂取矿的全过程，导出规划文件，供 `render_sequence.py --plan` 渲染：

```zsh
ros2 launch rmcs_energy_trace pickup_plan.launch.py [output_dir:=<目录>] [use_rviz:=true]
```

参数在 `config/pickup_plan.yaml`。默认生成 3 份规划：种子 0、1、2，目标安装位依次为 1（正上方）、0（右上）、4（正下方），输出到 `dataset/plans/plan_slot<k>_seed<s>.json`。场景的坐标约定与规划文件相同（见“连续序列”）。目标安装位一定有单元，其余各以 0.7 的概率有单元（按种子）。

规划按实际操作流程分 4 段：操作手停车、把准心对准目标、按下按钮后，机械臂先静止等视觉确认，再开始取矿。

1. `hold`：2.0 s，静止在观察姿态。
2. `approach`：OMPL 规划到预抓取位姿，再走直线到抓取位姿。抓取时 TCP（link_6 原点，夹爪两指中间）在单元中心，link_6 的 x 轴平行于单元长轴，夹爪从机器人一侧垂直于长轴接近（接近方向为 link_6 的 -z），预抓取位姿沿 link_6 的 +z 后退 0.10 m。直线段允许 link_6 与目标接触，link_5 不允许。
3. `grasp`：0.3 s 静止，目标附着到 link_6，规划文件中 `attached` 从这一段起为 true。
4. `extract`：沿单元长轴向外直线拉出 0.12 m。单元的开口端套在柱子上约 100 mm，拉出 0.12 m 才能完全脱离。

时间参数化用 TOTG，速度和加速度缩放 0.3，再按 30 fps 重采样关节角。

**观察姿态。** 底座离地 0.4 m，q1 = q6 = 0。底盘朝向负责水平对准，q2、q3 负责竖直对准：光轴穿过对准点，对准点是目标中心加随机偏移（水平 ±3 cm、竖直 ±2 cm，按种子），因为实际中操作手不可能把准心完全对准。q2 在关节范围内扫描，q3 和底盘朝向由对准条件解出。q4 在 {0, +90°, -90°, 180°}、q5 在 {±1.2, ±1.57} rad 中逐个组合试，留下对准有解、无碰撞、并且相机看得见目标的候选，取离预抓取姿态最近的一个。“看得见”的判据：从光心到目标中心画一个半径 0.04 m 的圆柱，靠目标一端缩短 0.08 m，它不与 link_4、link_5、link_6 碰撞。这个圆柱检查完立即移除，不参与后面的运动规划。

**底座距离。** 底座到目标的水平距离取“预抓取、抓取、拉出都可达、无碰撞且视线不被挡”的最长连续区间的中点，扫描范围 0.3～1.1 m。

**相机。** 装在 link_3 上（肘部右侧支架顶端），参数在配置文件中：光心 (0.13, 0.03, -0.10)，光轴指向 (0.03, 0.75, 0.035)，图像下方为 -x，分辨率 1280 × 720，水平视场角 69°。这些参数连同支架折线写入规划文件的 `camera` 字段。

规划工具会打印底座位姿、观察姿态、相机到目标的距离、目标中心在图像中的像素位置、各段时长和帧数、相机最大速度，以及 `approach` 和 `extract` 段中目标在相机坐标系中每帧的最大移动和转角。OMPL 每次的结果不同，所以重新生成时预抓取段的关节角会有差异；底座位姿和观察姿态是确定的。

## 分割（SAM）

`segmentation::Segmenter` 用 SAM（Segment Anything 的 ViT-B 版本）以检测框为提示分割目标，用 OpenVINO 在 CPU 上推理。网络从 Meta 的官方权重导出：

```zsh
curl -L -o dataset/weights/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python3 scripts/training/export_sam.py dataset/weights/sam_vit_b_01ec64.pth assets \
    --check dataset/real/images/6.png 294.6,120.7,399.1,204.1
```

导出需要 `torch`、`torchvision`、`segment-anything`、`onnx`、`onnxruntime` 和 `opencv-python`。权重只当作数据读取，不执行其中的代码。`--check` 用官方 `SamPredictor` 和导出的网络分割同一张图，打印两者掩码的 IoU。

一次分割分两步：

1. 编码器处理整幅图像：长边缩放到 1024，按官方的均值和方差归一化，右下补零。Ryzen 9600X（12 线程）上约 1.1 s，实机的 Intel CPU 会更慢。
2. 解码器根据检测框输出 4 个候选掩码，每次约 5 ms。我们按官方 ONNX 部署代码（`SamOnnxModel.select_masks`）对检测框提示的选法，在第 1～3 个候选中取预测 IoU 最高的一个。第 0 个候选（`multimask_output=False`）偶尔只给出端盖这样的局部。

SAM 的效果取决于检测框：框只包住一部分时，掩码也只覆盖那一部分。

`test_segmentation` 在中心块场景上测试：六棱柱中心块按规则手册的尺寸近似，6 根能量单元沿径向插在上面，每次随机去掉 0～2 根非目标单元，一半的场景后面有墙。检测框取目标可见像素的包围框。合格标准是 SAM 掩码与真值的 IoU 中位数不低于 0.9、最低不低于 0.85；用 SAM 掩码配准的成功率不低于 90%，错误接受不超过 1 次。找不到 SAM 网络时测试会跳过。

## 跟踪（M3T）

`tracking::Tracker` 用 [M3T](https://github.com/DLR-RM/3DObjectTracking) 跟踪位姿。它从上一帧的位姿出发，同时用彩色图的区域线索和深度图的深度线索，每帧约 0.6 ms（单线程，Ryzen 9600X）。只用深度线索约 0.15 ms。

M3T 需要三个预先生成的文件：`energy_unit.obj`、区域模型和深度模型，由 `scripts/download_assets.sh` 下载到 `assets/`。只有改了 mesh 或生成参数时才需要重新生成。生成时要渲染，需要 OpenGL 和一个显示，所以生成工具默认不构建，默认输出到 `assets/`：

```zsh
colcon build --packages-select rmcs_energy_trace --cmake-args -DRMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON
env -u WAYLAND_DISPLAY DISPLAY=:0 /workspaces/RMCS/rmcs_ws/build/rmcs_energy_trace/m3t_generate
```

默认参数下约 20 s，两个模型文件共约 150 MB。参数见 `m3t_generate --help` 的用法提示：`--mesh`、`--out`、`--sphere-radius`、`--n-divides`、`--n-points`、`--image-size`。运行时 `Tracker::Config::model_params` 必须与生成时一致。

重新生成后，上传新文件并更新 `scripts/assets.sha256`（见“运行所需的文件”）。

运行时不需要 OpenGL。库里的 M3T（`m3t_runtime`）用占位头文件编译渲染器，不链接 libGL、GLEW、GLFW，也不会初始化 GL，没有显示的环境也能运行。M3T 的源码在 `third_party/m3t/`，删减内容和本地修改见其中的 `README.md`。

`test_tracking` 在合成的彩色加深度序列上测试（150 帧，每帧最多移动 5 mm、转动 2°，直立在地面上和无支撑两种）。每个序列分别从真值位姿和模拟的配准结果开始跟踪；模拟的配准结果绕长轴偏 60°，再偏 3 mm 和 2°。合格标准是中心误差不超过 10 mm，中位数小于 2 mm，长轴误差中位数小于 1°。找不到模型文件时测试会跳过。

## 监督

`supervision::Supervisor` 每帧判断跟踪结果是否可信，并决定何时重新初始化。它只做判断，不做渲染。调用方每帧传入：

- 跟踪器的位姿，跟踪失败时为空；
- 这个位姿的深度检查结果：`registration::DepthCheck::check`，ROI 用 `DepthCheck::projected_box`，不用掩码；
- 可选：这一帧运行了检测时，mesh 投影框与检测框的 IoU；
- 可选：mesh 两端中较小的画面内面积比例：`DepthCheck::end_in_image_ratio`。

一帧判为不健康的条件（任一满足）：

- 跟踪失败；
- 与上一帧相比，中心移动超过 8 cm，或长轴转动超过 30°；
- 检测框与投影框的 IoU 低于 0.2；
- 未被遮挡的像素中，“mesh 处看到了更远的背景”超过 15%。

遮挡不算失败，因为抓取时夹爪会挡住能量单元。下面三种情况，这一帧判为无法判断：不输出可信位姿，也不计入失败。按这个顺序检查，排在上面那些不健康条件之后：

- mesh 有一端落在画面外，即两端中较小的画面内面积比例低于 0.8。能量单元的轴向位置只能靠两端约束，中段沿轴线滑动时深度几乎不变。一端出了画面，跟踪器会沿轴线滑动，而深度检查只统计画面内的像素，看不出来。`DepthCheck::end_in_image_ratio` 把重心 |z| 大于半长一半的三角形（两端各 1/4 长度）归为端部，z 正负各算一端，按三角形面积加权，看重心是否在相机前方且投影在图像范围内，不考虑自遮挡，取两端中较小的比例；
- 未被遮挡的像素少于 200 个；
- 未被遮挡的像素不到有效像素的 25%。这时模型大部分被挡住（例如夹爪从前方经过），剩下的像素太少，看不出位姿是否偏了。只按像素数判断时，分辨率高的相机几乎不会触发。

只有健康帧能清零计数。自上一个健康帧起，不健康的帧累计 5 帧，或没有健康帧（不健康或无法判断）的帧累计 30 帧（例如目标离开视野），就回到搜索状态，要求重新初始化；两类帧不要求连续。被判为跳变的帧，其位姿不作为下一帧跳变检查的基准，所以跟踪器停在新位置时，每一帧都判为跳变。跳变、IoU 和 5 帧这几个阈值沿用 RCIA_Benmao_Vision 的默认值。

`test_supervision` 用构造的观测测试状态机。`test_depth_mask` 用手工构造的小 mesh 测试 `DepthCheck::end_in_image_ratio`。`test_tracking` 把监督接在 M3T 上：正常跟踪时不应要求重新初始化，至少 95% 的帧可信；能量单元在第 75 帧离开画面后，要在 5 帧之内要求重新初始化。

## 完整流程

`pipeline::Pipeline` 把检测、掩码、配准、跟踪和监督串成一条流程。`Config` 组合各模块已有的 Config，另有相机、mesh 路径和掩码来源。`initialize` 加载所有网络和模型，任何一步失败都返回错误。`process(bgr, depth)` 处理一帧：`bgr` 是 `CV_8UC3`，`depth` 是 `CV_16UC1`，尺寸都等于 `Config::camera`。

`process` 返回处理这一帧时所处的状态（搜索或跟踪）、位姿（可选）、位姿是否可信、跟踪是否稳定，以及诊断信息：全部检测、选中的检测、用到的掩码、配准结果、监督的观测和判断、各阶段耗时。一个实例只能在一个线程里调用，不是线程安全的。一帧要几十到几百毫秒，所以组件应该在工作线程里调用它。

搜索状态每帧做这些事：

1. 运行检测。选检测框中心离画面中心 ((width - 1) / 2, (height - 1) / 2) 最近的一个，每帧只对它配准。操作手对准目标后才启动自动取矿，所以选最靠近准心的。掩码来源为分割网络时，跳过没有掩码的检测。
2. 取掩码，用检测框和掩码做配准。
3. 配准通过后，用配准的位姿启动 Tracker 和 Supervisor，进入跟踪状态。这一帧输出配准的位姿，记为可信，因为配准自己已经做过深度检查。

跟踪状态每帧做这些事：

1. Tracker 处理这一帧，同一帧也运行检测。
2. 用 `DepthCheck::projected_box` 得到跟踪位姿的投影框，在检测框中取与它 IoU 最大的一个，作为监督的检测 IoU。这一帧没有检测时不给。
3. 用 `DepthCheck` 检查跟踪位姿，ROI 用投影框，不用掩码；再用 `DepthCheck::end_in_image_ratio` 算 mesh 两端的画面内面积比例。
4. 把位姿、深度检查、检测 IoU 和端部画面内比例交给 Supervisor。监督要求重新初始化时，回到搜索状态，这一帧的位姿不可信。

掩码来源由 `Config::mask_source` 选择：

| 取值 | 掩码 | 说明 |
|---|---|---|
| `kNetwork` | 分割网络自带的掩码 | 网络是检测网络、没有掩码头时，`initialize` 返回错误 |
| `kSam` | 以检测框为提示，用 SAM 分割 | 需要配置 `segmenter`。SAM 失败时这一帧不配准，不退回只用框 |
| `kNone` | 不用掩码 | 只用检测框 |

**跟踪稳定。** `Result::stable` 为 true 的条件是：处于跟踪状态，且包括这一帧在内已经连续 `Config::stable_frames` 个处理帧可信。配准通过的那一帧算第 1 个可信帧；不可信的帧和重新初始化都把计数清零。`stable_frames` 默认 15，30 fps 下 0.5 s，最终取值以实机数据为准。

机械臂控制以 `stable` 为开始运动的条件：操作手把机器人停在矿站前、准心对准目标、按下按钮后，机械臂原地不动，等视觉完成检测、掩码、配准和 M3T 启动，并连续稳定一段时间，才开始运动取矿。运动中监督要求重新初始化时，`stable` 变为 false，机械臂应停下，原地重新初始化，`stable` 再次为 true 后继续。

流程本身不知道哪一根是任务要抓的目标，靠操作手启动前把准心对准它。准心附近有多根单元时，它可能选中别的单元。

## 参考位姿监视

取矿时，操作手停车、准心对准目标、按下按钮，机械臂静止，直到 `stable` 为 true 才开始运动。夹住之前，目标静止在中心块上。接近末段夹爪会遮挡目标，M3T 可能漂移，重新配准也可能失败，所以不能“跟丢就停下”。`pipeline::ReferenceMonitor` 用第一次 `stable` 时的位姿作为参考，判断后面的可信位姿是否还与它一致。它只做判断，不调用其他模块。

每帧调用一次 `update(result, camera_to_base, grasped)`：`result` 是 `Pipeline::Result`，`camera_to_base` 是相机（OpenCV 坐标系）到底座的位姿，`grasped` 表示夹爪已合拢（抓取开始之后），默认 false。位姿都换到底座系再比较，所以云台或机械臂带着相机运动时，静止的目标仍然与参考一致。

- **建立参考。** 没有参考时，第一个 `stable` 的帧把位姿换到底座系，存为参考位姿。机械臂按这个参考位姿接近、抓取、拉出。
- **比较。** 之后每个可信帧（`trusted`）与参考比较，中心距离不超过 5 mm 且长轴夹角不超过 6° 为一致。参考和当前位姿各有误差，长轴的估计误差可达 2°～4°，两者相减会超过 3°，所以长轴的阈值放得较宽。一致的可信帧把不一致计数清零。
- **目标被移动。** 连续 5 个可信帧不一致，状态变为 `kMoved`，参考被清除，机械臂应停下，原地重新初始化。`kMoved` 一直保持到上层调用 `reset()`：期间不建立参考，也不比较，输出仍是判为被移动的那次比较和计数。这样上层漏处理一帧，也不会因为下一帧自动建立了新参考而丢掉这个事件。`reset()` 后回到 `kNoReference`，下一个 `trusted` 且 `stable` 的帧建立新参考。
- **已夹住。** `grasped` 为 true 时，单元偏离参考是预期结果，所以不比较，也不会判为 `kMoved`，状态为 `kGrasped`，只输出位移，用来确认单元随夹爪被拉出。不一致计数保持不变。没有参考时，已夹住的帧不建立参考。
- **不可信的帧。** 不可信的帧是弱证据，不参与比较，不计入也不清零不一致计数。流程内部重新配准后的可信位姿照样参与比较，所以 `stable` 因重新初始化变为 false 时，参考和计数都不变。
- **位移。** 有参考时，每个可信帧还输出位姿相对参考的位移：中心的总距离，以及沿参考长轴（z 轴，指向封闭端）的分量，沿 +z 为正。拉出时用它判断单元有没有随夹爪移动。

阈值在 `ReferenceMonitor::Config` 里：`max_center_distance`（默认 0.005 m）、`max_axis_deg`（默认 6°）、`inconsistent_frames`（默认 5）。

`update` 返回的 `Output` 包含状态（`kNoReference`、`kEstablished`、`kGrasped`、`kMoved`）、底座系下的参考位姿、这一帧的比较结果（中心距离、长轴夹角、是否一致；没有参与比较时为空）、不一致计数和位移。建立参考的那一帧不比较，位移为 0。`kMoved` 时参考为空，没有位移，比较结果和计数（等于 `inconsistent_frames`）是判为被移动的那一次。

`test_supervision` 里有监视器的用例：stable 时建立参考、一致帧、不可信的帧不影响计数、连续 5 帧不一致判为被移动并清除参考、`kMoved` 保持到 `reset()`、`reset()` 后重新建立参考、已夹住时只输出位移、一致帧清零计数、相机运动而目标静止时仍一致、位移和阈值配置。

## 序列回放

`sequence_replay` 在渲染的连续序列上运行 `Pipeline`，把结果与真值比较，并输出叠加画面的视频：

```zsh
ros2 run rmcs_energy_trace sequence_replay <序列目录> <输出目录> --network <network.xml> \
    [--mask network|sam|none] [--sam <encoder.onnx> <decoder.onnx>] [--fps 30] \
    [--depth-noise 1.5] [--frames N] [--no-video]
```

`--mask` 默认 `network`；用 `sam` 时要给 `--sam`。`--frames` 只处理前 N 帧。

**阶段。** 按规划渲染的序列，每帧的 `meta` 里有 `phase`（`hold`、`approach`、`grasp`、`extract`），`sequence.json` 里有 `fps`。`phase` 记入 `frames.csv` 并画在叠加画面上，`summary.txt` 按阶段统计。内置轨迹的序列没有 `phase`，这些输出为空或省略；没有 `fps` 时按 30 计时间。

**模拟实时。** `--fps F` 大于 0 时，处理一帧用了 t 秒（`process` 的墙钟时间），下一帧取第 `i + max(1, ceil(t · F))` 帧，中间的帧记为跳过。这样初始化结束时，M3T 拿到的是相机已经移动过的新帧，与实机一致。`--fps 0` 处理每一帧，结果完全确定。

**深度噪声。** 读入的深度图加高斯噪声，标准差为 `--depth-noise`（mm）乘 z²（z 单位 m），另有 0.5% 的像素随机置为无效。随机数种子由帧号决定，同一帧的噪声不随跳帧而变。`--depth-noise 0` 关闭噪声和无效像素。

**参考位姿监视。** 序列的 `meta` 里有 `camera`（相机到世界的位姿）时，以世界系作底座系，每个处理帧运行一次 `ReferenceMonitor`；没有 `camera` 时不运行，相关输出为空或省略。`meta` 里的 `attached`（目标已被夹住）作为 `grasped` 传入，没有该字段时视为未夹住。出现 `kMoved` 时，回放模拟上层：记下帧号，随即调用 `reset()`，在下一个 `trusted` 且 `stable` 的帧重新建立参考，然后继续回放。

**与真值比较。** 每个有位姿输出的帧，找中心最近的真值单元，记录它的编号、是否为序列的目标单元、中心误差（mm）和长轴误差（两个 z 轴的夹角，0°～180°）。搜索状态下用到掩码时，再记录选中的掩码与该单元真值可见区域的 IoU。

输出目录里有：

| 文件 | 内容 |
|---|---|
| `frames.csv` | 每帧一行：是否处理、状态、是否可信、是否稳定、阶段、对应的真值单元、是否目标、误差、掩码 IoU、监督的判断和观测、各阶段耗时、监视器的状态、是否已夹住、是否建立参考、是否比较、是否一致、不一致计数、中心差（mm）、轴线差（°）、位移（mm）和沿轴位移（mm） |
| `summary.txt` | 同时打印到终端。处理和跳过的帧数；第一次输出可信位姿的帧号；第一次 `stable` 的帧号和时间（s，按序列的 `fps`）；序列有 `phase` 时，再给出运动开始的帧（第一个 `phase` 不是 `hold` 的帧）、稳定信号比运动开始早或晚多少秒，以及每个阶段的处理帧数、可信比例、监督无法判断（`unknown`）的帧数、可信帧的中心误差和长轴误差（中位数、最大值）和错误接受数；配准的尝试和成功次数；重新初始化的次数；首次可信之后可信帧的比例；可信帧的中心误差和长轴误差的中位数、90% 分位和最大值；错误接受的帧数（可信，但中心误差大于 10 mm 或长轴误差大于 5°）；各阶段平均耗时；序列有 `camera` 时，再给出建立参考的帧、是否出现“目标被移动”及所有帧号、各阶段参与比较的帧数和不一致帧数；序列有 `grasp` 阶段时，给出 `grasp` 开始之前最后一个可信帧比 `grasp` 开始早多少秒（盲区长度），如果之后又有可信帧，另写最后一个可信帧比 `grasp` 开始晚多少秒；序列有监视器和 `extract` 阶段时，给出该阶段最后一个可信帧的总位移和沿轴位移，没有可信帧时写明 |
| `replay.mp4` | 处理过的帧，帧率 30。画面上有全部检测框（细线）、选中的框（粗线）、掩码（半透明）、估计位姿的轮廓（可信为绿色，不可信为黄色）、目标单元真值的轮廓（白色细线）和左上角的文字（帧号、状态、是否可信、是否稳定、误差、总耗时、阶段、监视器的状态和比较结果） |

轮廓是 mesh 顶点投影后的凸包。`VideoWriter` 打不开 `replay.mp4` 时，改为把每帧写成 `frames/<帧号>.jpg`，并在终端说明。跳过的帧不写入视频。

## 抓取仿真

`grasp_sim` 在没有相机和机械臂的情况下演示整条抓取链路：渲染合成深度图，用点云配准估计能量单元位姿，生成 48 个抓取候选，交给 MoveIt 规划接近、抓取和抬升，最后在 RViz 中回放机械臂运动。只有 `link_6` 允许接触能量单元，`link_5` 不允许。

需要 MoveIt 和 `arm_moveit_config`。构建时找不到 MoveIt 会跳过这个目标。

```zsh
ros2 launch rmcs_energy_trace grasp_sim.launch.py
```

常用参数：`use_rviz`（默认 `true`，设为 `false` 则只在终端输出结果）、`object_x` `object_y` `object_z` `object_roll` `object_pitch` `object_yaw`（能量单元位姿）、`playback_speed`、`allow_link_contact`、`keep_alive`、`loop`。其余参数见 `config/grasp_sim.yaml`。没有找到可行抓取时进程退出码为 2。

在容器里显示 RViz 前，需要在宿主机执行 `xhost +SI:localuser:$(whoami)`。

## 构建

在 RMCS 工作区中：

```zsh
build-rmcs --packages-select rmcs_energy_trace
```

## 在 RMCS 中启用

在 `rmcs_bringup/config/<机器人>.yaml` 的组件列表中加入一行，并为实例名添加参数：

```yaml
rmcs_executor:
  ros__parameters:
    components:
      - rmcs_energy_trace::EnergyTraceComponent -> energy_trace

energy_trace:
  ros__parameters:
    mesh_path: ""
    camera:
      width: 640
      height: 480
      fx: 600.0
      fy: 600.0
      cx: 320.0
      cy: 240.0
      depth_scale: 0.001
```

`camera` 下的参数没有默认值，缺少时组件构造失败。深度图必须已对齐到彩色图：两者分辨率相同、共用这组内参。参数说明见 `config/energy_trace.yaml`。

## 参考

- RCIA_Benmao_Vision：<https://github.com/BenmaoNeko/RCIA_Benmao_Vision>
- TARS Go 战队，《一种基于 2D 计算机视觉划分 ROI 联合 3D 点云配准获取能量单元等目标姿态的工程自动取矿兑矿视觉方案分享》，RoboMaster 论坛，CC BY 4.0
