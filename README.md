# rmcs_energy_trace

RMCS 工程机器人的能量单元位姿估计组件。它读取 RGB-D 相机图像，输出能量单元在相机坐标系下的 6D 位姿，供机械臂规划抓取。整个流程在 CPU 上运行，不依赖 NVIDIA 显卡。

仓库里还有一套仿真管线：用 Blender 渲染合成数据训练检测网络，用 MoveIt 规划取矿动作，再按规划渲染带机械臂的 RGB-D 序列，离线回放整条流程。

## 怎样工作

取矿时：

1. 操作手停车，把相机画面的准心对准要取的能量单元，按下按钮。
2. 机械臂在观察姿态静止，视觉完成初始化：
   1. **检测**：YOLO11n-seg 在彩色图中找出能量单元，同时给出掩码。选离准心最近的一根。
   2. **掩码**：默认用检测网络给出的掩码，也可以切换为 SAM 以检测框为提示分割。
   3. **配准**：掩码内的深度像素转成点云，用 PPF 粗配准、ICP 精配准，再把模型渲染成深度图与实测深度比较，验收通过才得到位姿。
3. **跟踪**：之后每帧由 M3T 从上一帧的位姿算出新位姿。**监督**每帧判断结果是否可信，连续不可信时回到初始化。
4. 跟踪连续可信 0.5 s 后，给出“跟踪稳定”信号。**参考位姿监视**把这时的位姿换到机械臂底座坐标系，存为参考位姿。
5. 机械臂按参考位姿接近、抓取、拉出。途中视觉继续跟踪，核对目标没有被碰动；被碰动时机械臂停下，原地重新初始化。

各模块的逻辑见 [docs/modules.md](docs/modules.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/modules.md](docs/modules.md) | 每个模块的业务逻辑：输入、步骤和理由、判断条件、失败时怎么办、输出，以及离线测试 |
| [docs/simulation.md](docs/simulation.md) | 仿真管线的复现指南：渲染训练数据、训练检测网络、规划取矿动作、渲染测试序列、回放 |
| [docs/decisions.md](docs/decisions.md) | 设计决策与依据：为什么选这些方法，以及尚未解决的问题 |
| [third_party/m3t/README.md](third_party/m3t/README.md) | 随仓库带入的 M3T 源码：来源、删减内容和本地修改 |

## 快速上手

在 RMCS 工作区中，把本仓库放到 `rmcs_ws/src/` 下。

1. 下载运行所需的网络和 M3T 模型（约 500 MB，清单和 SHA256 在 `scripts/assets.sha256`）：

   ```zsh
   scripts/download_assets.sh
   ```

2. 编译（在 `rmcs_ws` 下）：

   ```zsh
   colcon build --merge-install --packages-select rmcs_energy_trace
   ```

3. 运行离线测试，可执行文件在 `rmcs_ws/build/rmcs_energy_trace/` 下：`test_detection`、`test_depth_mask`、`test_point_to_plane`、`test_supervision` 各不到 1 s；`test_tracking` 约 15 s；`test_segmentation` 约 30 s；`test_registration` 约 6 min。

检测网络（YOLO11n-seg）暂时不在下载清单中，要按 [docs/simulation.md](docs/simulation.md) 的第 2～4 步训练并导出。在序列上回放整条流程也见该文档。

## 输出接口

| 名称 | 类型 | 说明 |
|---|---|---|
| `/energy_trace/is_pose_valid` | `bool` | 位姿是否可信 |
| `/energy_trace/pose` | `Eigen::Isometry3d` | 相机坐标系下能量单元的位姿，单位 m |

位姿的原点在能量单元的几何中心，z 轴沿长轴指向 Ø95 封闭端。能量单元绕长轴近似 3 重对称，绕 z 轴的转角无法可靠观测，x、y 轴的方向没有意义，使用方只能用原点和 z 轴。

`rmcs_core` 中能量单元的参考点在底面以上 72.5 mm 处，本组件的原点在底面以上 75 mm 处，两者相差 2.5 mm。

## 在 RMCS 中启用

组件目前只声明输出接口、读取参数，还没有接入各模块，见 [docs/modules.md](docs/modules.md) 的“RMCS 组件”一节。

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

`camera` 下的参数没有默认值，缺少时组件构造失败。深度图必须已对齐到彩色图：两者分辨率相同，共用这组内参。参数说明见 `config/energy_trace.yaml`。

## 目录

```text
rmcs_energy_trace/
├── src/                           组件和各模块（见 docs/modules.md）
│   ├── component.cpp              RMCS 组件
│   ├── detection/                 检测：Detector，网络输出的解码
│   ├── segmentation/              分割：Segmenter（SAM）
│   ├── registration/              配准：Registration，深度检查 DepthCheck，ICP 求解
│   ├── tracking/                  跟踪：Tracker（封装 M3T）
│   ├── supervision/               监督：Supervisor
│   ├── pipeline/                  流程：Pipeline，参考位姿监视 ReferenceMonitor
│   ├── geometry/                  公共类型：mesh、相机参数、深度图反投影
│   ├── synthetic/                 C++ 简易渲染器，只供测试使用
│   └── utility/
│
├── test/                          离线测试，不需要相机和显示器
├── tool/
│   ├── energy_unit_detect.cpp     对图片运行检测，保存画好框和掩码的结果
│   ├── sequence_replay.cpp        在渲染的 RGB-D 序列上回放整条流程
│   ├── pickup_plan/               用 MoveIt 规划取矿动作
│   ├── m3t_generate.cpp           生成 M3T 模型文件（需要 OpenGL 和显示器，默认不构建）
│   └── grasp_sim/                 早期的桌面抓取演示（MoveIt + RViz）
│
├── scripts/
│   ├── download_assets.sh、assets.sha256   下载运行所需的网络和 M3T 模型
│   ├── blender/                   在 Blender 自带的 Python 中运行：渲染训练图和序列、机械臂模型
│   └── training/                  在装有 Ultralytics 的 Python 环境中运行：生成数据集、导出网络
│
├── docs/                          文档
├── config/                        组件参数示例、取矿动作规划和抓取演示的参数
├── launch/、rviz/                 取矿动作规划和抓取演示的启动文件
├── meshes/energy_unit.stl         能量单元的 mesh
├── third_party/m3t/               M3T 源码（MIT）
├── assets/                        运行所需的网络和 M3T 模型，由 download_assets.sh 下载，不进入仓库
└── dataset/                       仿真管线的本地数据，不进入仓库（布局见 docs/simulation.md）
```

`tool/grasp_sim/` 是早期的演示：固定相机看桌面上的能量单元，配准后用 MoveIt 规划抓取，在 RViz 中回放。它会被“取矿动作规划 + 序列回放”取代。运行方法：`ros2 launch rmcs_energy_trace grasp_sim.launch.py`，参数见 `config/grasp_sim.yaml`。

## 参考

- RCIA_Benmao_Vision：<https://github.com/BenmaoNeko/RCIA_Benmao_Vision>（配套文章的许可证为 CC BY-NC-SA 4.0）。整体结构、SAM 的用法、监督的规则和阈值参考了它，代码没有从该项目复制。
- TARS Go 战队，《一种基于 2D 计算机视觉划分 ROI 联合 3D 点云配准获取能量单元等目标姿态的工程自动取矿兑矿视觉方案分享》，RoboMaster 论坛，CC BY 4.0。
- M3T（DLR 3DObjectTracking）：<https://github.com/DLR-RM/3DObjectTracking>，MIT 许可证。
