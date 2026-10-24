#pragma once

namespace rmcs_energy_trace::tracking {

// M3T 需要的三个文件，位于同一个目录：mesh 的 OBJ 和区域 / 深度模型
inline constexpr auto kMeshFile = "energy_unit.obj";
inline constexpr auto kRegionModelFile = "energy_unit_region_model.bin";
inline constexpr auto kDepthModelFile = "energy_unit_depth_model.bin";

// 区域 / 深度模型的生成参数，默认值同 M3T。运行时必须与生成时一致，否则 M3T 认为文件无效
struct ModelParams {
    // 采样视角所在球面的半径 (m)，应大于物体到相机的距离
    float sphere_radius = 0.8f;
    // 球面细分次数，视角数约为 20 * 4^n_divides / 4
    int n_divides = 4;
    // 每个视角的采样点数
    int n_points = 200;
    // 生成时渲染的图像边长 (px)
    int image_size = 2000;
};

} // namespace rmcs_energy_trace::tracking
