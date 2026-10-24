# rmcs_energy_trace

RMCS 工程机器人的能量单元位姿估计组件。它读取 RGB-D 相机图像，输出能量单元在相机坐标系下的 6D 位姿，供机械臂规划抓取。

整个流程在 CPU 上运行，不依赖 NVIDIA 显卡。

## 当前状态

框架已搭好，组件可以被 `rmcs_executor` 加载，但还没有接入检测、配准和跟踪，`/energy_trace/pose_valid` 始终为 `false`。

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
| `src/` | 组件和算法代码 |
| `config/` | 参数示例 |
| `models/energy_unit.stl` | 能量单元 3D 模型，单位 mm，原点在外包长方体的一角 |

模型权重和数据集不放进仓库，见 `.gitignore`。

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
