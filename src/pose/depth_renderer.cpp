#include "pose/depth_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace rmcs_energy_trace::pose {

struct DepthRenderer::Impl {
    Config config;
    pcl::PointCloud<pcl::PointNormal>::Ptr dense;

    void initialize(const Mesh& mesh, const Config& new_config) noexcept {
        config = new_config;
        dense = sample_surface(mesh, config.sample_count, config.seed);
    }

    Frame render(
        const Eigen::Isometry3d& pose, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept {
        const auto width = config.width;
        const auto height = config.height;
        const auto& intrinsic = config.intrinsics;
        auto zbuffer = cv::Mat1f(height, width, std::numeric_limits<float>::infinity());

        auto min_u = width, min_v = height, max_u = 0, max_v = 0;
        const Eigen::Matrix3f rotation = pose.rotation().cast<float>();
        const Eigen::Vector3f translation = pose.translation().cast<float>();
        for (const auto& point : *dense) {
            const Eigen::Vector3f p = rotation * point.getVector3fMap() + translation;
            if (p.z() <= 0.f)
                continue;
            const auto u =
                static_cast<int>(std::lround(intrinsic.fx * p.x() / p.z() + intrinsic.cx));
            const auto v =
                static_cast<int>(std::lround(intrinsic.fy * p.y() / p.z() + intrinsic.cy));
            if (u < 0 || v < 0 || u >= width || v >= height)
                continue;
            auto& cell = zbuffer(v, u);
            cell = std::min(cell, p.z());
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }

        if (plane.has_value()) {
            for (int v = 0; v < height; v++) {
                for (int u = 0; u < width; u++) {
                    const Eigen::Vector3d ray{
                        (u - intrinsic.cx) / intrinsic.fx,
                        (v - intrinsic.cy) / intrinsic.fy,
                        1.,
                    };
                    const auto denom = plane->normal.dot(ray);
                    if (std::abs(denom) < 1e-6)
                        continue;
                    const auto z = plane->normal.dot(plane->point) / denom;
                    if (z <= 0. || z > config.plane_max_range)
                        continue;
                    auto& cell = zbuffer(v, u);
                    cell = std::min(cell, static_cast<float>(z));
                }
            }
        }

        auto noise = std::normal_distribution<float>{0.f, 1.f};
        auto depth = cv::Mat(height, width, CV_16UC1, cv::Scalar{0});
        for (int v = 0; v < height; v++) {
            for (int u = 0; u < width; u++) {
                const auto z = zbuffer(v, u);
                if (!std::isfinite(z))
                    continue;
                const auto mm = z * 1000.f + config.noise_sigma_mm * noise(engine);
                depth.at<std::uint16_t>(v, u) =
                    static_cast<std::uint16_t>(std::max(1.f, std::round(mm)));
            }
        }

        // 物体投影像素包围盒外扩
        auto roi = cv::Rect2i{min_u, min_v, max_u - min_u + 1, max_v - min_v + 1};
        const auto dx = static_cast<int>(roi.width * config.roi_margin);
        const auto dy = static_cast<int>(roi.height * config.roi_margin);
        roi = cv::Rect2i{roi.x - dx, roi.y - dy, roi.width + 2 * dx, roi.height + 2 * dy}
            & cv::Rect2i{0, 0, width, height};

        return Frame{depth, roi};
    }
};

void DepthRenderer::initialize(const Mesh& mesh, const Config& config) noexcept {
    pimpl_->initialize(mesh, config);
}

DepthRenderer::Frame DepthRenderer::render(
    const Eigen::Isometry3d& pose, std::mt19937& engine,
    const std::optional<Plane>& plane) const noexcept {
    return pimpl_->render(pose, engine, plane);
}

DepthRenderer::DepthRenderer() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

DepthRenderer::~DepthRenderer() noexcept = default;

} // namespace rmcs_energy_trace::pose
