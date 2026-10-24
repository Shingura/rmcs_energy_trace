#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "pose/mesh_model.hpp"
#include "pose/point_cloud_registration.hpp"
#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::pose {

// 合成深度图渲染器：对模型表面密采样做 z-buffer 投影，可叠加平面并加高斯噪声
class DepthRenderer {
    RMCS_PIMPL_DEFINITION(DepthRenderer)

public:
    struct Config {
        int width = 640;
        int height = 480;
        PointCloudRegistration::Intrinsics intrinsics{600., 600., 320., 240.};

        std::size_t sample_count = 1'500'000;
        std::uint32_t seed = 7;
        // 深度噪声标准差 (mm)
        float noise_sigma_mm = 1.f;
        // 平面的最大渲染距离 (m)
        double plane_max_range = 3.;
        // ROI 相对物体像素包围盒的外扩比例
        double roi_margin = 0.1;
    };

    // 相机系下的平面：normal 为法向量，point 为平面上一点
    struct Plane {
        Eigen::Vector3d normal;
        Eigen::Vector3d point;
    };

    struct Frame {
        cv::Mat depth;  // CV_16UC1, mm, 0 为无效
        cv::Rect2i roi; // 物体投影包围盒外扩
    };

    void initialize(const Mesh& mesh, const Config& config) noexcept;

    // pose 为模型到相机，plane 非空时同时渲染该平面
    Frame render(
        const Eigen::Isometry3d& pose, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept;
};

} // namespace rmcs_energy_trace::pose
