#pragma once

#include "geometry/camera.hpp"
#include "utility/pimpl.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rmcs_energy_trace::registration {

class Registration {
    RMCS_PIMPL_DEFINITION(Registration)

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
        // 近对称物体的单次 RANSAC 常落入错误盆地，少于 8 次时随机朝向场景约四分之一找不到正确解，
        // 24 次降到约 6%（正确候选会提前结束，所以典型耗时几乎不变，只有拒绝时才跑满）
        int coarse_attempts = 24;
        // 粗配准累计失败（找不到满足内点比例的假设）达到该次数则放弃当前场景候选
        int coarse_max_failures = 2;
        // PCL 的 SampleConsensusPrerejective 用全局 rand() 取样。每次 RANSAC 前调用
        // std::srand(seed + 第几次尝试)，使结果只取决于输入，与调用顺序无关。
        // 代价：配准会改写全局 rand() 状态，且不能与其他使用 rand() 的线程并发运行
        std::uint32_t seed = 42;
        int ransac_iterations = 20000;
        int ransac_correspondence_k = 5;
        float ransac_similarity_threshold = 0.9f;
        float ransac_inlier_threshold = 0.008f;
        float ransac_inlier_fraction = 0.7f;

        // 精配准 (point-to-plane ICP)
        int icp_max_iterations = 30;
        std::vector<float> icp_max_correspondences = {0.010f, 0.005f, 0.003f};
        float refine_voxel_size = 0.002f;

        // 候选评价与验收。内点率只衡量“场景是否落在 mesh 上”，任何比 mesh 小的
        // 表面都能得高分，所以还要用深度图检查“mesh 是否被场景支持”
        float inlier_distance = 0.004f;
        float min_inlier_ratio = 0.9f;
        // mesh 渲染深度与实测深度之差小于该值视为支持 (m)
        float depth_tolerance = 0.005f;
        // 渲染轮廓向内腐蚀的像素数，去掉边缘上混合深度造成的误判
        int silhouette_erosion_px = 2;
        // 参与比较的有效像素少于该值时无法判断，直接拒绝
        std::size_t min_model_pixels = 300;
        // 阈值依据合成数据：正确结果支持率 >= 0.93、空闲空间违例 <= 0.045；
        // 0.8 倍缩小的 mesh 最好的候选支持率约 0.83、违例 >= 0.07
        float min_support_ratio = 0.88f;
        float max_free_space_ratio = 0.06f;
        // 提前结束粗配准的条件：候选已通过验收，且内点率不低于该值；
        // 有深度图时还要求支持率不低于 early_stop_support_ratio、空闲空间违例不高于
        // early_stop_free_space_ratio。仅凭内点率会把错误盆地（0.97~0.98）当作最优
        float early_stop_inlier_ratio = 0.99f;
        float early_stop_support_ratio = 0.97f;
        float early_stop_free_space_ratio = 0.01f;
    };

    struct Timing {
        double preprocess_ms = 0.;
        double feature_ms = 0.;
        double coarse_ms = 0.;
        double icp_ms = 0.;
        double quality_ms = 0.;
        double total_ms = 0.;
    };

    // 候选位姿的质量。把 mesh 按该位姿渲染到 ROI（及掩码）内，逐像素与实测深度比较；
    // 只统计渲染与实测都有深度的像素 (valid)：
    //   支持    |z_实测 - z_渲染| < depth_tolerance
    //   空闲空间违例  z_实测 > z_渲染 + depth_tolerance，即 mesh 所在处看到了更远的东西
    //   遮挡    z_实测 < z_渲染 - depth_tolerance，即前面有别的东西，不是错误证据
    // register_cloud 没有深度图，只有 scene_coverage 有意义，depth_checked 为 false
    struct Quality {
        bool depth_checked = false;
        std::size_t model_pixels = 0; // 渲染 mesh 覆盖的像素数（腐蚀轮廓、限定在 ROI/掩码内）
        double support_ratio = 0.;    // 支持 / valid
        double free_space_ratio = 0.; // 空闲空间违例 / valid
        double occluded_ratio = 0.;   // 遮挡 / valid
        double missing_ratio = 0.;    // 无实测深度的像素 / model_pixels
        double scene_coverage = 0.; // 场景点中距 mesh 小于 inlier_distance 的比例，即 inlier_ratio
    };

    struct Result {
        Eigen::Isometry3d pose;     // mesh -> camera
        double inlier_ratio;        // 场景点中距 mesh 小于 inlier_distance 的比例
        Quality quality;
        Timing timing;
    };

    // 深度图 (CV_16UC1，单位见 camera.depth_scale，0 为无效) 在 roi 内反投影为相机系点云 (m)
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi) noexcept;

    // 同上，并只保留掩码 (CV_8UC1，与深度图同尺寸，非 0 保留) 内的像素，
    // 掩码先按 config.mask_erosion_px 内缩。掩码为空时与不带掩码的版本一致，
    // 类型或尺寸不符时返回空点云
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask, const Config& config) noexcept;
    // 注意：这个重载固定使用 Config{} 的默认值（含 mask_erosion_px = 3），不读取任何实例的配置；
    // 需要实例配置时用 register_depth，或显式传入 Config
    static Cloud::Ptr depth_to_cloud(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask) noexcept;

    std::expected<void, std::string>
        initialize(const std::string& stl_path, const Config& config) noexcept;
    std::expected<void, std::string> initialize(const std::string& stl_path) noexcept;

    // 只有点云：只能用内点率验收（不检查 mesh 是否被场景支持），因此挡不住
    // “比真实物体小的表面”，quality 里只有 scene_coverage。需要可靠验收时用 register_depth
    std::optional<Result> register_cloud(const Cloud& scene) const noexcept;

    // 深度图 -> 点云（按本实例的 mask_erosion_px 处理掩码）-> 配准 -> 深度一致性检查。
    // mask 为空表示只用 roi。未通过验收（内点率、支持率、空闲空间违例）时返回空
    std::optional<Result> register_depth(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask = {}) const noexcept;
};

} // namespace rmcs_energy_trace::registration
