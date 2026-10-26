# 开发约定

## 构建

在 RMCS 工作区中：

```zsh
build-rmcs --packages-select rmcs_energy_trace
```

## 检查

修改 C++ 代码后，先做静态检查，再编译：

```zsh
clang-format -i <文件>
clang-tidy -p /workspaces/RMCS/rmcs_ws/build/rmcs_energy_trace \
    --extra-arg=-isystem$PWD/third_party/m3t/stubs/clangd <文件>
```

`--extra-arg` 提供空的 `omp.h`。不加时 clang-tidy 因找不到 `omp.h` 解析失败，会报出大量错误的 “can be made static”。命令在包目录下运行。

编译不能有警告。

## 规范

- 优先用 C++。Python 只用于 C++ 做不了的事，例如训练和导出 YOLO 模型。Python 脚本放在 `scripts/` 下。
- 模型权重、数据集和运行输出不进入仓库。
- 算法放在 `src/` 下按功能分目录，组件文件 `src/component.cpp` 只负责接口和调度。目录名即命名空间名，例如 `src/registration/` 对应 `rmcs_energy_trace::registration`。
- 用词：能量单元的 3D 形状叫 mesh（`meshes/`、`mesh_path`），YOLO 等神经网络叫 network（`network_path`）。不用 model 指代这两者。配准中的“模型点云”是指从 mesh 采样得到的参考点云。
- 仓库根目录的 `.clangd` 在 clangd 分析时去掉 `-fopenmp`，否则 clangd 找不到 PCL 需要的 `omp.h`。
- 每个算法模块都要有能离线运行的测试，不依赖相机和机械臂。
- 代码风格使用仓库中的 `.clang-format`，它与 RMCS 根目录的配置相同。
