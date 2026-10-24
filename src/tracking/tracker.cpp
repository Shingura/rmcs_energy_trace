#include "tracking/tracker.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <vector>

#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/depth_modality.h>
#include <m3t/depth_model.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/tracker.h>

namespace rmcs_energy_trace::tracking {

namespace {

// 图像由调用方逐帧送入，M3T 只读取
class MemoryColorCamera final : public m3t::ColorCamera {
public:
    MemoryColorCamera(const std::string& name, const m3t::Intrinsics& intrinsics)
        : ColorCamera{name} {
        intrinsics_ = intrinsics;
    }

    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    bool UpdateImage(bool /*synchronized*/) override { return !image_.empty(); }

    void feed(const cv::Mat& image) { image_ = image; }
};

class MemoryDepthCamera final : public m3t::DepthCamera {
public:
    MemoryDepthCamera(const std::string& name, const m3t::Intrinsics& intrinsics, float depth_scale)
        : DepthCamera{name} {
        intrinsics_ = intrinsics;
        depth_scale_ = depth_scale;
    }

    bool SetUp() override {
        set_up_ = true;
        return true;
    }

    bool UpdateImage(bool /*synchronized*/) override { return !image_.empty(); }

    void feed(const cv::Mat& image) { image_ = image; }
};

auto to_m3t(const Eigen::Isometry3d& pose) -> m3t::Transform3fA {
    auto result = m3t::Transform3fA::Identity();
    result.matrix() = pose.matrix().cast<float>();
    return result;
}

auto from_m3t(const m3t::Transform3fA& pose) -> Eigen::Isometry3d {
    auto result = Eigen::Isometry3d::Identity();
    result.matrix() = pose.matrix().cast<double>();
    return result;
}

} // namespace

struct Tracker::Impl {
    Config config;
    std::shared_ptr<m3t::Body> body;
    std::shared_ptr<MemoryColorCamera> color_camera;
    std::shared_ptr<MemoryDepthCamera> depth_camera;
    std::unique_ptr<m3t::Tracker> tracker;
    int iteration = 0;
    bool started = false;

    // 检查图像格式和尺寸，use_* 为假的通道不检查
    [[nodiscard]] std::expected<void, std::string>
        check_images(const cv::Mat& bgr, const cv::Mat& depth) const {
        const auto size_ok = [&](const cv::Mat& image) {
            return image.cols == config.width && image.rows == config.height;
        };
        if (config.use_region && (bgr.type() != CV_8UC3 || !size_ok(bgr)))
            return std::unexpected{"Color image must be CV_8UC3 and match the configured size"};
        if (config.use_depth && (depth.type() != CV_16UC1 || !size_ok(depth)))
            return std::unexpected{"Depth image must be CV_16UC1 and match the configured size"};
        return {};
    }

    void feed(const cv::Mat& bgr, const cv::Mat& depth) const {
        color_camera->feed(bgr);
        depth_camera->feed(depth);
    }

    std::expected<void, std::string> initialize(const Config& new_config) {
        config = new_config;
        started = false;
        tracker.reset();

        if (!config.use_region && !config.use_depth)
            return std::unexpected{"At least one of region and depth modality must be enabled"};

        const auto dir = std::filesystem::path{config.model_dir};
        const auto mesh_path = dir / kMeshFile;
        const auto region_path = dir / kRegionModelFile;
        const auto depth_path = dir / kDepthModelFile;
        // 运行时库不能渲染，文件缺失时 M3T 会尝试生成并失败，这里先给出明确的提示
        auto required = std::vector{mesh_path};
        if (config.use_region)
            required.push_back(region_path);
        if (config.use_depth)
            required.push_back(depth_path);
        for (const auto& path : required) {
            if (!std::filesystem::exists(path))
                return std::unexpected{std::format(
                    "Missing {}, generate it with m3t_generate "
                    "(build with -DRMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON)",
                    path.string())};
        }

        const auto& k = config.intrinsics;
        const auto intrinsics = m3t::Intrinsics{
            static_cast<float>(k.fx),
            static_cast<float>(k.fy),
            static_cast<float>(k.cx),
            static_cast<float>(k.cy),
            config.width,
            config.height};
        color_camera = std::make_shared<MemoryColorCamera>("color", intrinsics);
        depth_camera = std::make_shared<MemoryDepthCamera>("depth", intrinsics, config.depth_scale);
        color_camera->SetUp();
        depth_camera->SetUp();

        // 几何单位已是米，mesh 的原点和坐标轴即为位姿约定，geometry2body 为单位阵
        body = std::make_shared<m3t::Body>(
            "energy_unit", mesh_path, 1.f, true, true, m3t::Transform3fA::Identity());
        if (!body->SetUp())
            return std::unexpected{"Failed to set up M3T body: " + mesh_path.string()};

        auto link = std::make_shared<m3t::Link>("link", body);
        const auto& p = config.model_params;
        if (config.use_region) {
            auto model = std::make_shared<m3t::RegionModel>(
                "region_model", body, region_path, p.sphere_radius, p.n_divides, p.n_points, 0.05f,
                0.002f, false, p.image_size);
            if (!model->SetUp())
                return std::unexpected{"Failed to load region model: " + region_path.string()};
            auto modality =
                std::make_shared<m3t::RegionModality>("region", body, color_camera, model);
            // 不做遮挡处理
            modality->set_n_unoccluded_iterations(0);
            modality->set_n_lines_max(config.n_lines_max);
            if (!modality->SetUp())
                return std::unexpected{"Failed to set up region modality"};
            link->AddModality(modality);
        }
        if (config.use_depth) {
            auto model = std::make_shared<m3t::DepthModel>(
                "depth_model", body, depth_path, p.sphere_radius, p.n_divides, p.n_points, 0.05f,
                0.002f, false, p.image_size);
            if (!model->SetUp())
                return std::unexpected{"Failed to load depth model: " + depth_path.string()};
            auto modality =
                std::make_shared<m3t::DepthModality>("depth", body, depth_camera, model);
            modality->set_n_unoccluded_iterations(0);
            modality->set_n_points_max(config.n_points_max);
            if (!modality->SetUp())
                return std::unexpected{"Failed to set up depth modality"};
            link->AddModality(modality);
        }
        if (!link->SetUp())
            return std::unexpected{"Failed to set up M3T link"};
        auto optimizer = std::make_shared<m3t::Optimizer>("optimizer", link);
        if (!optimizer->SetUp())
            return std::unexpected{"Failed to set up M3T optimizer"};

        tracker = std::make_unique<m3t::Tracker>(
            "tracker", config.n_corr_iterations, config.n_update_iterations);
        tracker->AddOptimizer(optimizer);
        // 各对象已逐个初始化，不让 Tracker 再初始化一遍
        if (!tracker->SetUp(false))
            return std::unexpected{"Failed to set up M3T tracker"};
        return {};
    }

    std::expected<void, std::string>
        start(const Eigen::Isometry3d& pose, const cv::Mat& bgr, const cv::Mat& depth) {
        if (!tracker)
            return std::unexpected{"Tracker is not initialized"};
        if (auto checked = check_images(bgr, depth); !checked)
            return checked;
        feed(bgr, depth);
        body->set_body2world_pose(to_m3t(pose));
        iteration = 0;
        if (!tracker->UpdateCameras(iteration) || !tracker->StartModalities(iteration))
            return std::unexpected{"Failed to start M3T modalities"};
        started = true;
        return {};
    }

    std::optional<Result> track(const cv::Mat& bgr, const cv::Mat& depth) {
        if (!started || !check_images(bgr, depth))
            return std::nullopt;
        feed(bgr, depth);
        const auto begin = std::chrono::steady_clock::now();
        iteration++;
        if (!tracker->UpdateCameras(iteration) || !tracker->ExecuteTrackingStep(iteration))
            return std::nullopt;
        const auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        return Result{from_m3t(body->body2world_pose()), ms};
    }
};

std::expected<void, std::string> Tracker::initialize(const Config& config) noexcept try {
    return pimpl_->initialize(config);
} catch (const std::exception& e) {
    return std::unexpected{std::string{"Failed to initialize tracker: "} + e.what()};
}

std::expected<void, std::string>
    Tracker::start(const Eigen::Isometry3d& pose, const cv::Mat& bgr, const cv::Mat& depth) noexcept
    try {
    return pimpl_->start(pose, bgr, depth);
} catch (const std::exception& e) {
    return std::unexpected{std::string{"Failed to start tracker: "} + e.what()};
}

std::optional<Tracker::Result> Tracker::track(const cv::Mat& bgr, const cv::Mat& depth) noexcept
    try {
    return pimpl_->track(bgr, depth);
} catch (const std::exception&) {
    return std::nullopt;
}

bool Tracker::started() const noexcept { return pimpl_->started; }

Tracker::Tracker() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Tracker::~Tracker() noexcept = default;

} // namespace rmcs_energy_trace::tracking
