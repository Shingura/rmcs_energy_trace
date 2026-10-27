#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"
#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::synthetic {

// 合成深度图渲染器：对模型表面密采样做 z-buffer 投影，可叠加平面并加高斯噪声。
// 配置了 Config::color 时同时输出与深度对齐的彩色图，供基于颜色的跟踪使用
class DepthRenderer {
    RMCS_PIMPL_DEFINITION(DepthRenderer)

public:
    // 彩色图的外观，颜色为 BGR
    struct ColorConfig {
        // 物体中段和两端（|z| 大于 dark_abs_z，模型系，单位 m）的颜色
        cv::Vec3b object_light{190, 160, 70};
        cv::Vec3b object_dark{50, 40, 30};
        double dark_abs_z = 0.05;
        // 平面和背景（没有物体也没有平面的像素）的基色
        cv::Vec3b plane{120, 125, 130};
        cv::Vec3b background{100, 90, 80};
        // 平面和背景叠加的平滑纹理：幅度（灰度级）和格子边长 (px)，幅度为 0 表示无纹理
        float texture_amplitude = 0.f;
        int texture_cell_px = 24;
        // 逐像素高斯噪声标准差（灰度级）
        float noise_sigma = 2.f;
        // 渲染后的高斯模糊标准差 (px)，模拟边缘不是阶跃，0 表示不模糊
        float blur_sigma_px = 0.7f;
    };

    struct Config {
        geometry::Camera camera;

        std::size_t sample_count = 1'500'000;
        std::uint32_t seed = 7;
        // 深度噪声标准差 (mm)
        float noise_sigma_mm = 1.f;
        // 平面的最大渲染距离 (m)
        double plane_max_range = 3.;
        // ROI 相对物体像素包围盒的外扩比例
        double roi_margin = 0.1;
        // 非空时输出彩色图；彩色噪声在深度噪声之后抽取，不影响深度图的结果
        std::optional<ColorConfig> color = std::nullopt;
    };

    // 相机系下的平面：normal 为法向量，point 为平面上一点
    struct Plane {
        Eigen::Vector3d normal;
        Eigen::Vector3d point;
    };

    struct Frame {
        cv::Mat depth;  // CV_16UC1，单位见 camera.depth_scale，0 为无效
        cv::Rect2i roi; // 物体投影包围盒外扩
        cv::Mat mask;   // CV_8UC1，物体可见的像素为 255，即精确的实例掩码
        cv::Mat color;  // CV_8UC3 BGR，未配置 Config::color 时为空
    };

    // 场景中的一个物体：mesh 为 initialize / add_mesh 返回的编号，pose 为模型到相机
    struct Object {
        std::size_t mesh;
        Eigen::Isometry3d pose;
    };

    struct SceneFrame {
        cv::Mat depth;    // CV_16UC1，单位见 camera.depth_scale，0 为无效
        cv::Mat instance; // CV_8UC1, 0 为背景或平面, k 为 objects[k - 1]
        cv::Mat color;    // CV_8UC3 BGR，未配置 Config::color 时为空
    };

    // 初始化并登记第 0 号网格
    void initialize(const geometry::Mesh& mesh, const Config& config) noexcept;

    // 追加登记一个网格，返回其编号。color 非空时，彩色图中这个网格整体用该颜色（BGR），
    // 例如中心块；为空时按能量单元的两端深色、中段浅色上色
    std::size_t add_mesh(
        const geometry::Mesh& mesh, const std::optional<cv::Vec3b>& color = std::nullopt) noexcept;

    // pose 为模型到相机，plane 非空时同时渲染该平面
    Frame render(
        const Eigen::Isometry3d& pose, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept;

    // 渲染多个物体（互相遮挡）及可选平面，同时输出实例编号图，最多 255 个物体
    SceneFrame render_scene(
        const std::vector<Object>& objects, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept;
};

} // namespace rmcs_energy_trace::synthetic
