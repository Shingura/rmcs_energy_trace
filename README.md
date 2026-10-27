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
│   ├── geometry/                  各模块共用的几何类型
│   │   ├── mesh.hpp/.cpp          读取二进制 STL（mm 转 m，原点移到中心），在表面均匀采样
│   │   ├── camera.hpp             Camera：RGB-D 相机的分辨率、内参和深度单位
│   │   └── depth_cloud.hpp/.cpp   深度图在框（和掩码）内反投影为点云
│   │
│   ├── synthetic/                 合成数据，只供测试和离线工具使用
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
│   ├── supervision.cpp            监督状态机的单元测试
│   └── tracking.cpp               跟踪：在合成的连续帧上统计误差和耗时，并接上监督（缺少 M3T 模型时跳过）
│
├── tool/                          离线工具
│   ├── energy_unit_detect.cpp     对图片或目录运行检测（可选再用 SAM 分割），保存画好框和掩码的图片
│   ├── mask_ablation.cpp          对比实验：配准时只用检测框和加掩码的效果
│   ├── m3t_generate.cpp           生成 M3T 模型文件（需要 OpenGL 和显示，默认不构建）
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
│   ├── export_yolo.py             把 Ultralytics 的 .pt 权重导出为 OpenVINO 格式
│   ├── export_sam.py              把 SAM 的官方权重导出为编码器和解码器两个 ONNX 网络
│   └── synth/                     检测网络的合成训练数据
│       ├── render_dataset.py      在 Blender 中随机搭建取矿场景，渲染彩色图和编号图
│       ├── make_yolo_dataset.py   把渲染结果转换成 Ultralytics 的分割数据集
│       └── download_hdri.sh       下载渲染用的室内 HDRI
│
├── config/
│   ├── energy_trace.yaml          组件参数示例
│   └── grasp_sim.yaml             抓取仿真参数
│
├── launch/grasp_sim.launch.py     启动抓取仿真（MoveIt、仿真节点、RViz）
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
| `yolo11n_tars.xml`、`yolo11n_tars.bin` | 约 10 MB | YOLO 检测网络（OpenVINO） | TARS 开源的权重，用 `scripts/export_yolo.py` 导出 |
| `sam_vit_b_encoder.onnx` | 约 343 MB | SAM 编码器 | Meta 官方权重，用 `scripts/export_sam.py` 导出 |
| `sam_vit_b_decoder.onnx` | 约 16 MB | SAM 解码器 | 同上 |
| `energy_unit.obj` | 约 106 KB | M3T 使用的 mesh | `m3t_generate` 生成 |
| `energy_unit_region_model.bin`、`energy_unit_depth_model.bin` | 约 74 MB、70 MB | M3T 的区域模型和深度模型 | `m3t_generate` 生成，需要 OpenGL 和显示器 |

文件放在 GitHub Release `assets-v1`。编译时 `assets/` 会安装到包的 share 目录，随安装目录一起同步到机器人；目录不存在时跳过，不影响编译。M3T 加载模型时只比较 OBJ 的文件名，所以 `assets/` 放在哪里都能用（见 `third_party/m3t/README.md`）。

更新某个文件后，把新文件上传到 Release，再在 `assets/` 下运行 `sha256sum <文件名>`，用结果替换 `scripts/assets.sha256` 中对应的一行。

`dataset/` 存放开发用的数据，也不进入仓库：

```text
dataset/
├── images/                        测试图片
├── networks/                      原始权重：YOLO 的 .pt、SAM 官方权重 sam_vit_b_01ec64.pth
└── synthetic/                     合成训练数据：HDRI、渲染结果、数据集和训练输出
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

`network.xml` 由 `scripts/export_yolo.py` 从 Ultralytics 的 `.pt` 权重导出。

分割网络的掩码按 Ultralytics 的 `process_mask_native`（`retina_masks=True`）解码：掩码系数乘原型得到 160 × 160 的 logit，取出未填充图像的区域，双线性缩放到原图尺寸，logit 大于 0 的像素为目标，检测框以外置零。只对 NMS 之后保留的检测解码，并且只计算框内的像素。`Detector::Detection::mask` 的约定与 `Segmenter::Result::mask` 相同。`Timing::mask` 是掩码解码的耗时，不计入 `postprocess`。`test_detection` 用构造的输出张量测试解码，不需要网络文件。

掩码消融工具在合成深度图上比较点云配准只用检测框和再加实例掩码的效果。场景与测试 B 相同，并在目标旁放 1 到 2 个相邻的能量单元（间距 5、20 或 50 mm，直立或平躺，部分试验挡在目标前面）。每帧运行 3 种输入：只用框、框加精确掩码、框加模拟 YOLO-seg 的退化掩码，另有一组无相邻单元的对照。默认 40 次试验。

```zsh
ros2 run rmcs_energy_trace mask_ablation [--trials 40] [--seed 2025] [--mesh <energy_unit.stl>]
```

输出每次试验的误差和汇总表（成功率、拒绝次数、接受但错误的次数、中心和轴向误差的中位数与最大值、耗时）。`--trials 120` 约需 11 分钟。

`geometry::depth_to_cloud` 把深度图在框（和掩码）内反投影为点云，掩码先按参数 `mask_erosion_px` 向内腐蚀（配准默认 3 px）。`test_depth_mask` 是它的单元测试。

## 合成训练数据

我们没有标注好的实拍图，检测网络用 Blender 渲染的合成图训练。脚本在 `scripts/synth/`，HDRI、渲染结果、数据集和训练输出都放在 `dataset/synthetic/` 下。

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
# 1. 下载 13 个室内 HDRI（Poly Haven，CC0），已有的文件会跳过
scripts/synth/download_hdri.sh

# 2. 渲染 2000 帧，--device 可选 CPU、CUDA、OPTIX
blender --background --factory-startup --python scripts/synth/render_dataset.py -- \
    --out dataset/synthetic/raw --count 2000 --seed 0 --device OPTIX

# 3. 生成 Ultralytics 的分割数据集
python3 scripts/synth/make_yolo_dataset.py dataset/synthetic/raw dataset/synthetic/yolo

# 4. 训练 YOLO11n-seg
cd dataset/synthetic
yolo segment train model=yolo11n-seg.pt data=yolo/data.yaml imgsz=640 epochs=100 batch=16 \
    project=$PWD/runs name=v1
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

## 分割（SAM）

`segmentation::Segmenter` 用 SAM（Segment Anything 的 ViT-B 版本）以检测框为提示分割目标，用 OpenVINO 在 CPU 上推理。网络从 Meta 的官方权重导出：

```zsh
curl -L -o dataset/networks/sam_vit_b_01ec64.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python3 scripts/export_sam.py dataset/networks/sam_vit_b_01ec64.pth assets \
    --check dataset/images/6.png 294.6,120.7,399.1,204.1
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
- 可选：这一帧运行了检测时，mesh 投影框与检测框的 IoU。

一帧判为不健康的条件（任一满足）：

- 跟踪失败；
- 与上一帧相比，中心移动超过 8 cm，或长轴转动超过 30°；
- 检测框与投影框的 IoU 低于 0.2；
- 未被遮挡的像素中，“mesh 处看到了更远的背景”超过 15%。

遮挡不算失败，因为抓取时夹爪会挡住能量单元。未被遮挡的像素少于 200 个时，这一帧判为无法判断：不输出可信位姿，也不计入失败。只有健康帧能清零计数。自上一个健康帧起，不健康的帧累计 5 帧，或没有健康帧（不健康或无法判断）的帧累计 30 帧（例如目标离开视野），就回到搜索状态，要求重新初始化；两类帧不要求连续。被判为跳变的帧，其位姿不作为下一帧跳变检查的基准，所以跟踪器停在新位置时，每一帧都判为跳变。跳变、IoU 和 5 帧这几个阈值沿用 RCIA_Benmao_Vision 的默认值。

`test_supervision` 用构造的观测测试状态机。`test_tracking` 把监督接在 M3T 上：正常跟踪时不应要求重新初始化，至少 95% 的帧可信；能量单元在第 75 帧离开画面后，要在 5 帧之内要求重新初始化。

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
