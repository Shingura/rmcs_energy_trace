#pragma once

#include "utility/pimpl.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rmcs_energy_trace::pose {

class PointCloudRegistration {
    RMCS_PIMPL_DEFINITION(PointCloudRegistration)

public:
    using Cloud = pcl::PointCloud<pcl::PointXYZ>;

    struct Config {
        // 掩码内缩像素数：使用掩码反投影前先向内腐蚀，去掉边界上的混合深度
        int mask_erosion_px = 3;

        // 模型
        std::size_t model_sample_count = 200000;
        float model_voxel_size = 0.004f;

        // 场景预处理
        float scene_voxel_size = 0.004f;
        float plane_distance_threshold = 0.004f;
        float plane_min_inlier_ratio = 0.3f;
        int plane_max_iterations = 200;
        int outlier_mean_k = 20;
        float outlier_stddev_multiplier = 2.0f;
        std::size_t min_scene_points = 200;

        // 粗配准 (FPFH + SampleConsensusPrerejective)
        float normal_radius = 0.012f;
        float feature_radius = 0.030f;
        int coarse_attempts = 8;
        float early_stop_inlier_ratio = 0.97f;
        int coarse_max_failures = 2;
        int ransac_iterations = 20000;
        int ransac_correspondence_k = 5;
        float ransac_similarity_threshold = 0.9f;
        float ransac_inlier_threshold = 0.008f;
        float ransac_inlier_fraction = 0.7f;

        // 精配准 (point-to-plane ICP)
        int icp_max_iterations = 30;
        std::vector<float> icp_max_correspondences = {0.010f, 0.005f, 0.003f};
        float refine_voxel_size = 0.002f;

        // 验收
        float inlier_distance = 0.004f;
        float min_inlier_ratio = 0.6f;
        float max_fitness = 4e-5f;
    };

    struct Intrinsics {
        double fx, fy, cx, cy;
    };

    struct Timing {
        double preprocess_ms = 0.;
        double feature_ms = 0.;
        double coarse_ms = 0.;
        double icp_ms = 0.;
        double total_ms = 0.;
    };

    struct Result {
        Eigen::Isometry3d pose; // model -> camera
        double fitness;         // 场景内点到模型最近点的均方距离 (m^2)
        double inlier_ratio;    // 场景点中距模型小于 inlier_distance 的比例
        Timing timing;
    };

    // 深度图 (CV_16UC1, mm, 0 为无效) 在 roi 内反投影为相机系点云 (m, x 右 y 下 z 前)
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const Intrinsics& intrinsics, const cv::Rect2i& roi) noexcept;

    // 同上，并只保留掩码 (CV_8UC1，与深度图同尺寸，非 0 保留) 内的像素，
    // 掩码先按 config.mask_erosion_px 内缩。掩码为空时与不带掩码的版本一致，
    // 类型或尺寸不符时返回空点云
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const Intrinsics& intrinsics, const cv::Rect2i& roi,
        const cv::Mat& mask, const Config& config) noexcept;
    // 使用默认的 Config
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const Intrinsics& intrinsics, const cv::Rect2i& roi,
        const cv::Mat& mask) noexcept;

    std::expected<void, std::string>
        initialize(const std::string& stl_path, const Config& config) noexcept;
    std::expected<void, std::string> initialize(const std::string& stl_path) noexcept;

    std::optional<Result> register_cloud(const Cloud& scene) const noexcept;
};

} // namespace rmcs_energy_trace::pose
