# rmcs_energy_trace

RMCS 工程机器人的能量单元位姿估计组件。它读取 RGB-D 相机图像，输出能量单元在相机坐标系下的 6D 位姿，供机械臂规划抓取。

整个流程在 CPU 上运行，不依赖 NVIDIA 显卡。

## 当前状态

检测和点云配准模块已完成，并有离线测试。它们还没有接入组件，`/energy_trace/pose_valid` 始终为 `false`。跟踪和监督尚未实现。

## 设计

每帧的处理分为 4 步：

1. **检测**：YOLO 在彩色图中框出能量单元（以后可换成 YOLO-seg，同时输出掩码）。用 OpenVINO 在 CPU 上推理。
2. **初始化**：把框内的深度像素转换为点云，用 FPFH 特征粗配准，再用 ICP 精配准，得到初始位姿。
3. **跟踪**：后续帧用 M3T 从上一帧的位姿开始跟踪，不再每帧做完整配准。
4. **监督**：根据检测框与投影的重合度、深度一致性和位姿跳变，判断跟踪结果是否可信。不可信时回到第 2 步重新初始化。

跟踪和监督的设计参考了 [RCIA_Benmao_Vision](https://github.com/BenmaoNeko/RCIA_Benmao_Vision)。该项目用 SAM 和 FoundationPose 做初始化，需要 NVIDIA 显卡。本项目改用 CPU 上的点云配准。

## 输出接口

| 名称 | 类型 | 说明 |
|---|---|---|
| `/energy_trace/pose_valid` | `bool` | 位姿是否可信 |
| `/energy_trace/pose` | `Eigen::Isometry3d` | 相机坐标系下能量单元的位姿，单位 m |

位姿的原点在能量单元的几何中心，z 轴沿长轴。注意：`rmcs_core` 中能量单元的参考点在底面以上 72.5 mm 处，本组件的原点在底面以上 75 mm 处，两者相差 2.5 mm。

## 目录

| 路径 | 内容 |
|---|---|
| `src/component.cpp` | 组件，负责接口和调度 |
| `src/detection/` | YOLO11 检测（OpenVINO，CPU） |
| `src/pose/` | STL 读取、点云配准、合成深度图渲染 |
| `test/` | 离线测试，不需要相机 |
| `tool/` | 离线工具，例如 `energy_unit_detect` |
| `scripts/` | Python 脚本，例如把 YOLO 权重导出为 OpenVINO 格式 |
| `config/` | 参数示例 |
| `models/energy_unit.stl` | 能量单元 3D 模型，单位 mm，原点在外包长方体的一角 |

模型权重和数据集不放进仓库，见 `.gitignore`。

## 测试与工具

配准测试在合成深度图上各运行 20 次。合格标准是中心误差小于 3 mm，且长轴方向误差小于 2°。

```zsh
/workspaces/RMCS/rmcs_ws/build/rmcs_energy_trace/test_point_cloud_registration
```

检测工具对一张图片或一个目录中的图片运行检测，并保存画好框的结果：

```zsh
ros2 run rmcs_energy_trace energy_unit_detect <model.xml> <图片或目录> <输出目录> [--conf 0.25]
```

`model.xml` 由 `scripts/export_yolo.py` 从 Ultralytics 的 `.pt` 权重导出。

掩码消融工具在合成深度图上比较点云配准只用检测框和再加实例掩码的效果。场景与测试 B 相同，并在目标旁放 1 到 2 个相邻的能量单元（间距 5、20 或 50 mm，直立或平躺，部分试验挡在目标前面）。每帧运行 3 种输入：只用框、框加精确掩码、框加模拟 YOLO-seg 的退化掩码，另有一组无相邻单元的对照。默认 40 次试验，每次约 3 s。

```zsh
ros2 run rmcs_energy_trace mask_ablation [--trials 40] [--seed 2025]
```

输出每次试验的误差和汇总表（成功率、中心和轴向误差的中位数与最大值、无结果次数、耗时）。结果来自合成数据，没有真实传感器的噪声和边缘伪影，只能作为参考。

配准的 `depth_to_cloud` 有带掩码的重载，掩码先按 `Config::mask_erosion_px`（默认 3 px）向内腐蚀。`test_depth_mask` 是它的单元测试。

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
    model_path: ""
```

参数说明见 `config/energy_trace.yaml`。

## 参考

- RCIA_Benmao_Vision：<https://github.com/BenmaoNeko/RCIA_Benmao_Vision>
- TARS Go 战队，《一种基于 2D 计算机视觉划分 ROI 联合 3D 点云配准获取能量单元等目标姿态的工程自动取矿兑矿视觉方案分享》，RoboMaster 论坛，CC BY 4.0
