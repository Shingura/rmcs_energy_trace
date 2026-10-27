#pragma once

#include "geometry/camera.hpp"
#include "registration/depth_check.hpp"
#include "utility/pimpl.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace rmcs_energy_trace::registration {

// 由深度图求能量单元的位姿：PPF 粗配准给出若干候选，每个候选做 ICP 精配准和绕长轴的扫描，
// 再把 mesh 按候选位姿渲染成深度图与实测比较，选出通过验收且最好的一个。
// 同一个实例不能在多个线程中同时调用 register_depth（OpenCV 的 PPF 匹配会修改内部状态）
class Registration {
    RMCS_PIMPL_DEFINITION(Registration)

public:
    struct Config {
        // 掩码内缩像素数：使用掩码反投影前先向内腐蚀，去掉边界上的混合深度
        int mask_erosion_px = 3;

        // 模型：在 mesh 表面采样，按两种间距下采样，只保留外表面的点（沿法线向外看不到 mesh）。
        // 开口端的孔壁不在模型里：取矿时开口端套在柱子上，看不到
        std::size_t model_sample_count = 400000;
        float model_voxel_size = 0.004f;  // PPF 训练用
        float refine_voxel_size = 0.002f; // ICP 和内点率用

        // 场景预处理
        float scene_voxel_size = 0.003f;
        float plane_distance_threshold = 0.004f;
        float plane_min_inlier_ratio = 0.3f;
        int plane_max_iterations = 200;
        int outlier_mean_k = 20;
        float outlier_stddev_multiplier = 2.0f;
        std::size_t min_scene_points = 200;
        // 场景法线的估计半径，法线朝向相机
        float normal_radius = 0.012f;

        // 粗配准（OpenCV PPF）。下面的距离都是相对模型直径（包围盒对角线，约 0.2 m）的比例
        double ppf_sampling_step = 0.08;
        double ppf_distance_step = 0.03;
        double ppf_angle_count = 30.;
        double ppf_scene_sample_step = 0.2; // 场景点每 5 个取 1 个作为参考点
        double ppf_scene_distance = 0.05;
        // 按票数取前几个位姿继续精配准
        std::size_t ppf_candidates = 5;

        // 精配准（point-to-plane ICP），对应距离逐级收紧
        int icp_max_iterations = 60;
        std::vector<float> icp_max_correspondences = {0.010f, 0.006f, 0.004f, 0.003f, 0.002f};
        // 绕长轴扫描时，在一个周期（3 重对称为 120°）内取的角度数。转角不影响中心和长轴，
        // 但筋的位置对不上时支持率偏低，而 ICP 很难把转角转回来
        int roll_samples = 12;
        // 扫描后的 ICP 起点已经很近，只用较紧的几级
        std::vector<float> roll_icp_max_correspondences = {0.004f, 0.003f, 0.002f};

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
    };

    struct Timing {
        double preprocess_ms = 0.;
        double normal_ms = 0.; // 场景法线
        double coarse_ms = 0.; // PPF 匹配
        double icp_ms = 0.;    // ICP 和绕长轴扫描
        double quality_ms = 0.;
        double total_ms = 0.;
    };

    // 候选位姿的深度检查结果，定义见 DepthCheck::Quality
    using Quality = DepthCheck::Quality;

    struct Result {
        Eigen::Isometry3d pose; // mesh -> camera
        double inlier_ratio;    // 场景点中距 mesh 小于 inlier_distance 的比例
        Quality quality;
        Timing timing;
    };

    std::expected<void, std::string>
        initialize(const std::string& stl_path, const Config& config) noexcept;
    std::expected<void, std::string> initialize(const std::string& stl_path) noexcept;

    // 深度图 -> 点云（按 mask_erosion_px 处理掩码）-> 配准 -> 深度一致性检查。
    // mask 为空表示只用 roi。未通过验收（内点率、支持率、空闲空间违例）时返回空
    std::optional<Result> register_depth(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask = {}) const noexcept;
};

} // namespace rmcs_energy_trace::registration
