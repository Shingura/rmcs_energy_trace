# rmcs_energy_trace

RMCS 工程机器人的能量单元位姿估计组件。它读取 RGB-D 相机图像，输出能量单元在相机坐标系下的 6D 位姿，供机械臂规划抓取。

整个流程在 CPU 上运行，不依赖 NVIDIA 显卡。

## 设计

每帧的处理分为 4 步：

1. **检测**：YOLO 在彩色图中框出能量单元。用 OpenVINO 在 CPU 上推理。
2. **初始化**：把框（或掩码）内的深度像素转换为点云，用 PPF 粗配准，再用 ICP 精配准，经深度检查后得到初始位姿。
3. **跟踪**：后续帧用 M3T 从上一帧的位姿开始跟踪，不再每帧做完整配准。
4. **监督**：根据检测框与投影的重合度、深度一致性和位姿跳变，判断跟踪结果是否可信。不可信时回到第 2 步重新初始化。

跟踪和监督的设计参考了 [RCIA_Benmao_Vision](https://github.com/BenmaoNeko/RCIA_Benmao_Vision)。该项目用 SAM 和 FoundationPose 做初始化，需要 NVIDIA 显卡。本项目改用 CPU 上的点云配准。

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
│   │   ├── detector.hpp/.cpp      Detector：缩放并填充图像、OpenVINO 推理、解析输出、NMS，返回检测框和得分
│   │   └── yolo11.hpp             YOLO11 网络的输入输出约定（640 × 640、输出 [1, 5, 8400]）和预处理设置
│   │
│   ├── registration/              点云配准，得到初始位姿
│   │   ├── registration.hpp/.cpp  去平面、PPF 粗配准、ICP 精配准、绕长轴扫描、深度检查
│   │   └── point_to_plane.hpp/.cpp ICP 每次迭代的求解：约束不足的方向不移动，不会得到 NaN
│   │
│   ├── tracking/                  逐帧跟踪
│   │   ├── tracker.hpp/.cpp       Tracker：封装 M3T，从初始位姿开始，逐帧输入彩色图和深度图，输出位姿
│   │   ├── mesh_obj.hpp/.cpp      把 mesh 写成 M3T 读取的 OBJ 文件（单位 m，原点在中心）
│   │   └── model_files.hpp        M3T 模型文件的文件名和生成参数，生成工具和运行时共用
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
│   └── tracking.cpp               跟踪：在合成的连续帧上统计误差和耗时（缺少 M3T 模型时跳过）
│
├── tool/                          离线工具
│   ├── energy_unit_detect.cpp     对图片或目录运行检测，保存画好框的图片
│   ├── mask_ablation.cpp          对比实验：配准时只用检测框和加掩码的效果
│   ├── m3t_generate.cpp           生成 M3T 模型文件（需要 OpenGL 和显示，默认不构建）
│   └── grasp_sim/                 MoveIt 抓取仿真
│       ├── grasp_sim.cpp          仿真主程序：渲染、配准、生成抓取候选、规划并回放
│       ├── grasp_candidates.hpp   根据能量单元的中心和长轴生成抓取候选
│       └── joint_state_player.hpp 发布 /joint_states，按时间插值回放轨迹
│
├── third_party/m3t/               随仓库带入的 M3T 源码（MIT），未修改上游文件
│   ├── README.md                  来源、提交号、删减内容和本地处理方式
│   ├── include/m3t/               M3T 源码
│   ├── third_party/               M3T 自带的第三方头文件
│   └── stubs/                     占位文件：去掉 OpenGL、GLFW、highgui 依赖，以及供 clangd 使用的空 omp.h
│
├── scripts/
│   └── export_yolo.py             把 Ultralytics 的 .pt 权重导出为 OpenVINO 格式
│
├── config/
│   ├── energy_trace.yaml          组件参数示例
│   └── grasp_sim.yaml             抓取仿真参数
│
├── launch/grasp_sim.launch.py     启动抓取仿真（MoveIt、仿真节点、RViz）
├── rviz/grasp_sim.rviz            抓取仿真的 RViz 配置
├── meshes/energy_unit.stl         能量单元的 mesh
├── dataset/                       本地数据，不进入仓库（见下文）
├── CMakeLists.txt、package.xml、plugins.xml   
├── .clangd                        
├── .clang-format、.clang-tidy     
└── AGENTS.md                      
```

`dataset/` 不进入仓库，需要自己准备：

```text
dataset/
├── images/                        测试图片
├── networks/                      YOLO 权重和导出的 OpenVINO 网络
└── m3t/                           m3t_generate 生成的 M3T 模型文件（约 150 MB）
```

## 配准

入口是 `registration::Registration::register_depth(depth, camera, roi, mask)`，`mask` 可以为空。流程：

1. 框（和掩码）内的深度像素转成点云，降采样到 3 mm 间距，去掉离群点。有掩码时背景已经去掉，不去平面。只有检测框时，如果点云中有大平面（如地面），先试去掉这个平面的点云，再试原始点云（大平面也可能是能量单元的端面）。
2. PPF 粗配准（OpenCV `surface_matching`）：按票数取前 5 个位姿。模型点云只保留 mesh 的外表面，开口端的内腔从外面看不到，不参与配准。
3. 每个位姿依次做 ICP、绕长轴扫描、再做 ICP。绕长轴扫描在 120° 内取 12 个角度，选深度检查最好的一个：转角不影响中心和长轴，但筋的位置对不上时支持率偏低，ICP 又很难把转角转回来。
4. 三个阶段的位姿都按下面的指标比较：先看是否通过验收，再比分数（内点率 + 支持率 − 2 × 空闲空间违例比例）。最好的一个通过验收才返回。

检查方法：按候选位姿把能量单元的 mesh 投影到深度图上，逐像素比较：

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

检测工具对一张图片或一个目录中的图片运行检测，并保存画好框的结果：

```zsh
ros2 run rmcs_energy_trace energy_unit_detect <network.xml> <图片或目录> <输出目录> [--conf 0.25]
```

`network.xml` 由 `scripts/export_yolo.py` 从 Ultralytics 的 `.pt` 权重导出。

掩码消融工具在合成深度图上比较点云配准只用检测框和再加实例掩码的效果。场景与测试 B 相同，并在目标旁放 1 到 2 个相邻的能量单元（间距 5、20 或 50 mm，直立或平躺，部分试验挡在目标前面）。每帧运行 3 种输入：只用框、框加精确掩码、框加模拟 YOLO-seg 的退化掩码，另有一组无相邻单元的对照。默认 40 次试验。

```zsh
ros2 run rmcs_energy_trace mask_ablation [--trials 40] [--seed 2025] [--mesh <energy_unit.stl>]
```

输出每次试验的误差和汇总表（成功率、拒绝次数、接受但错误的次数、中心和轴向误差的中位数与最大值、耗时）。`--trials 120` 约需 11 分钟。

`geometry::depth_to_cloud` 把深度图在框（和掩码）内反投影为点云，掩码先按参数 `mask_erosion_px` 向内腐蚀（配准默认 3 px）。`test_depth_mask` 是它的单元测试。

## 跟踪（M3T）

`tracking::Tracker` 用 [M3T](https://github.com/DLR-RM/3DObjectTracking) 跟踪位姿。它从上一帧的位姿出发，同时用彩色图的区域线索和深度图的深度线索，每帧约 0.6 ms（单线程，Ryzen 9600X）。只用深度线索约 0.15 ms。

M3T 需要三个预先生成的文件：`energy_unit.obj`、区域模型和深度模型，放在 `dataset/m3t/`。生成时要渲染，需要 OpenGL 和一个显示，所以生成工具默认不构建：

```zsh
colcon build --packages-select rmcs_energy_trace --cmake-args -DRMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON
env -u WAYLAND_DISPLAY DISPLAY=:0 /workspaces/RMCS/rmcs_ws/build/rmcs_energy_trace/m3t_generate
```

默认参数下约 20 s，两个模型文件共约 150 MB。参数见 `m3t_generate --help` 的用法提示：`--mesh`、`--out`、`--sphere-radius`、`--n-divides`、`--n-points`、`--image-size`。运行时 `Tracker::Config::model_params` 必须与生成时一致。

文件里记录了 OBJ 的路径，移动目录后要重新生成。

运行时不需要 OpenGL。库里的 M3T（`m3t_runtime`）用占位头文件编译渲染器，不链接 libGL、GLEW、GLFW，也不会初始化 GL，没有显示的环境也能运行。M3T 的源码在 `third_party/m3t/`，删减内容和本地修改见其中的 `README.md`。

`test_tracking` 在合成的彩色加深度序列上测试（150 帧，每帧最多移动 5 mm、转动 2°，直立在地面上和无支撑两种）。每个序列分别从真值位姿和模拟的配准结果开始跟踪；模拟的配准结果绕长轴偏 60°，再偏 3 mm 和 2°。合格标准是中心误差不超过 10 mm，中位数小于 2 mm，长轴误差中位数小于 1°。找不到模型文件时测试会跳过。

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
