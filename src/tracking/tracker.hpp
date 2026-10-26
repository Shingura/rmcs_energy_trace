#pragma once

#include <expected>
#include <optional>
#include <string>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>

#include "geometry/camera.hpp"
#include "tracking/model_files.hpp"
#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::tracking {

// 基于 M3T 的 6D 位姿跟踪：从上一帧的位姿出发，用区域和深度两种线索迭代更新。
// 位姿为能量单元到相机（OpenCV 约定：x 右、y 下、z 前），单位 m。
// 运行时不需要 OpenGL，但需要 m3t_generate 预先生成的 OBJ 和区域 / 深度模型。
class Tracker {
    RMCS_PIMPL_DEFINITION(Tracker)

public:
    struct Config {
        // 存放 kMeshFile、kRegionModelFile、kDepthModelFile 的目录
        std::string model_dir;
        // 生成模型时使用的参数
        ModelParams model_params;

        // 彩色图和深度图共用这组参数（深度图已对齐到彩色图）
        geometry::Camera camera;

        // 区域线索使用彩色图，深度线索使用深度图，至少启用一个
        bool use_region = true;
        bool use_depth = true;

        // 每帧的迭代次数：n_corr_iterations 次重新建立对应，每次做 n_update_iterations 次位姿更新
        int n_corr_iterations = 5;
        int n_update_iterations = 2;
        // 区域线索的采样线数、深度线索的采样点数
        int n_lines_max = 200;
        int n_points_max = 200;
    };

    struct Result {
        Eigen::Isometry3d pose; // 能量单元到相机
        double ms;              // 本帧跟踪耗时
    };

    // 读取模型并建立跟踪器，模型缺失或与 Config 不符时返回错误
    std::expected<void, std::string> initialize(const Config& config) noexcept;

    // 以 pose 为起点开始（或重新开始）跟踪。bgr / depth 是该位姿对应的那一帧，
    // 区域线索用它初始化前景和背景的颜色统计
    std::expected<void, std::string>
        start(const Eigen::Isometry3d& pose, const cv::Mat& bgr, const cv::Mat& depth) noexcept;

    // 处理下一帧。bgr 为 CV_8UC3，depth 为 CV_16UC1（单位见 camera.depth_scale）。
    // 未调用 start、图像格式或尺寸不符、M3T 执行失败时返回空
    std::optional<Result> track(const cv::Mat& bgr, const cv::Mat& depth) noexcept;

    [[nodiscard]] bool started() const noexcept;
};

} // namespace rmcs_energy_trace::tracking
