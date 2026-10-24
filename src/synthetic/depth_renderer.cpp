#include "synthetic/depth_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace rmcs_energy_trace::synthetic {

namespace {

// 零均值的平滑噪声场：粗网格上的高斯随机数双三次插值到整幅图，再乘以幅度
auto make_texture(int height, int width, int cell, float amplitude, std::uint32_t seed) -> cv::Mat {
    if (amplitude <= 0.f)
        return {};
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证可复现
    auto engine = std::mt19937{seed};
    auto normal = std::normal_distribution<float>{0.f, 1.f};
    cell = std::max(cell, 1);
    auto coarse = cv::Mat1f(height / cell + 3, width / cell + 3);
    for (auto& value : coarse)
        value = normal(engine);
    auto field = cv::Mat1f{};
    cv::resize(coarse, field, {width, height}, 0., 0., cv::INTER_CUBIC);
    return field * amplitude;
}

auto saturate(float value) -> std::uint8_t {
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

} // namespace

struct DepthRenderer::Impl {
    Config config;
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> dense;
    // 平面和背景的纹理，仅在配置了彩色图时使用
    cv::Mat plane_texture, background_texture;

    void initialize(const geometry::Mesh& mesh, const Config& new_config) noexcept {
        config = new_config;
        dense.clear();
        if (config.color.has_value()) {
            const auto& color = *config.color;
            plane_texture = make_texture(
                config.height, config.width, color.texture_cell_px, color.texture_amplitude,
                config.seed + 1);
            background_texture = make_texture(
                config.height, config.width, color.texture_cell_px, color.texture_amplitude,
                config.seed + 2);
        }
        add_mesh(mesh);
    }

    std::size_t add_mesh(const geometry::Mesh& mesh) noexcept {
        dense.push_back(geometry::sample_surface(mesh, config.sample_count, config.seed));
        return dense.size() - 1;
    }

    // 投影像素包围盒（含被遮挡的采样点），无点时为空
    struct Bounds {
        int min_u, min_v, max_u, max_v;
    };

    SceneFrame render_scene(
        const std::vector<Object>& objects, std::mt19937& engine, const std::optional<Plane>& plane,
        Bounds& bounds) const noexcept {
        const auto width = config.width;
        const auto height = config.height;
        const auto& intrinsic = config.intrinsics;
        auto zbuffer = cv::Mat1f(height, width, std::numeric_limits<float>::infinity());
        auto instance = cv::Mat(height, width, CV_8UC1, cv::Scalar{0});
        // 可见物体点的模型系 z，用于给两端上色
        auto model_z = config.color.has_value() ? cv::Mat1f(height, width, 0.f) : cv::Mat1f{};

        bounds = Bounds{width, height, 0, 0};
        auto& [min_u, min_v, max_u, max_v] = bounds;
        for (std::size_t k = 0; k < objects.size(); k++) {
            const auto& object = objects[k];
            if (object.mesh >= dense.size())
                continue;
            const auto id = static_cast<std::uint8_t>(k + 1);
            const Eigen::Matrix3f rotation = object.pose.rotation().cast<float>();
            const Eigen::Vector3f translation = object.pose.translation().cast<float>();
            for (const auto& point : *dense[object.mesh]) {
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
                if (p.z() < cell) {
                    cell = p.z();
                    instance.at<std::uint8_t>(v, u) = id;
                    if (!model_z.empty())
                        model_z(v, u) = point.z;
                }
                min_u = std::min(min_u, u);
                max_u = std::max(max_u, u);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
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
                    const auto zf = static_cast<float>(z);
                    if (zf < cell) {
                        cell = zf;
                        instance.at<std::uint8_t>(v, u) = 0;
                    }
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
        auto color = cv::Mat{};
        if (config.color.has_value())
            color = render_color(zbuffer, instance, model_z, engine);
        return SceneFrame{depth, instance, color};
    }

    // 按每个像素是物体中段、物体两端、平面还是背景上色，再加纹理、模糊和噪声
    auto render_color(
        const cv::Mat1f& zbuffer, const cv::Mat& instance, const cv::Mat1f& model_z,
        std::mt19937& engine) const noexcept -> cv::Mat {
        const auto& setting = *config.color;
        auto color = cv::Mat(config.height, config.width, CV_8UC3);
        for (int v = 0; v < config.height; v++) {
            for (int u = 0; u < config.width; u++) {
                auto base = cv::Vec3f{};
                auto texture = 0.f;
                if (!std::isfinite(zbuffer(v, u))) {
                    base = setting.background;
                    if (!background_texture.empty())
                        texture = background_texture.at<float>(v, u);
                } else if (instance.at<std::uint8_t>(v, u) == 0) {
                    base = setting.plane;
                    if (!plane_texture.empty())
                        texture = plane_texture.at<float>(v, u);
                } else {
                    base = std::abs(model_z(v, u)) > setting.dark_abs_z ? setting.object_dark
                                                                        : setting.object_light;
                }
                color.at<cv::Vec3b>(v, u) = cv::Vec3b{
                    saturate(base[0] + texture), saturate(base[1] + texture),
                    saturate(base[2] + texture)};
            }
        }
        if (setting.blur_sigma_px > 0.f)
            cv::GaussianBlur(color, color, {}, setting.blur_sigma_px);
        if (setting.noise_sigma > 0.f) {
            auto noise = std::normal_distribution<float>{0.f, setting.noise_sigma};
            for (auto& pixel : cv::Mat_<cv::Vec3b>(color)) {
                for (int c = 0; c < 3; c++)
                    pixel[c] = saturate(static_cast<float>(pixel[c]) + noise(engine));
            }
        }
        return color;
    }

    SceneFrame render_scene(
        const std::vector<Object>& objects, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept {
        if (objects.size() > std::numeric_limits<std::uint8_t>::max())
            return SceneFrame{};
        auto bounds = Bounds{};
        return render_scene(objects, engine, plane, bounds);
    }

    Frame render(
        const Eigen::Isometry3d& pose, std::mt19937& engine,
        const std::optional<Plane>& plane) const noexcept {
        auto bounds = Bounds{};
        auto scene = render_scene({Object{0, pose}}, engine, plane, bounds);

        // 物体投影像素包围盒外扩
        auto roi = cv::Rect2i{
            bounds.min_u, bounds.min_v, bounds.max_u - bounds.min_u + 1,
            bounds.max_v - bounds.min_v + 1};
        const auto dx = static_cast<int>(roi.width * config.roi_margin);
        const auto dy = static_cast<int>(roi.height * config.roi_margin);
        roi = cv::Rect2i{roi.x - dx, roi.y - dy, roi.width + 2 * dx, roi.height + 2 * dy}
            & cv::Rect2i{0, 0, config.width, config.height};

        return Frame{scene.depth, roi, scene.color};
    }
};

void DepthRenderer::initialize(const geometry::Mesh& mesh, const Config& config) noexcept {
    pimpl_->initialize(mesh, config);
}

std::size_t DepthRenderer::add_mesh(const geometry::Mesh& mesh) noexcept {
    return pimpl_->add_mesh(mesh);
}

DepthRenderer::SceneFrame DepthRenderer::render_scene(
    const std::vector<Object>& objects, std::mt19937& engine,
    const std::optional<Plane>& plane) const noexcept {
    return pimpl_->render_scene(objects, engine, plane);
}

DepthRenderer::Frame DepthRenderer::render(
    const Eigen::Isometry3d& pose, std::mt19937& engine,
    const std::optional<Plane>& plane) const noexcept {
    return pimpl_->render(pose, engine, plane);
}

DepthRenderer::DepthRenderer() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

DepthRenderer::~DepthRenderer() noexcept = default;

} // namespace rmcs_energy_trace::synthetic
