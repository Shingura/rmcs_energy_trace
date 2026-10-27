#include "registration/depth_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace rmcs_energy_trace::registration {

namespace {

constexpr float kNoDepth = std::numeric_limits<float>::max();

// 把 mesh 的三角形按位姿 (mesh -> camera) 光栅化为 roi 内的 z-buffer (m)，
// 无覆盖处为 kNoDepth。像素 (u, v) 的中心位于投影坐标 (u, v)，与 depth_to_cloud 一致
cv::Mat rasterize_depth(
    const geometry::Mesh& mesh, const Eigen::Isometry3d& pose, const geometry::Camera& camera,
    const cv::Rect2i& roi) noexcept {
    auto buffer = cv::Mat{roi.height, roi.width, CV_32FC1, cv::Scalar{kNoDepth}};
    const Eigen::Matrix3f rotation = pose.linear().cast<float>();
    const Eigen::Vector3f translation = pose.translation().cast<float>();
    const auto fx = static_cast<float>(camera.fx);
    const auto fy = static_cast<float>(camera.fy);
    const auto cx = static_cast<float>(camera.cx);
    const auto cy = static_cast<float>(camera.cy);
    constexpr float kNearPlane = 0.01f;
    constexpr float kEdgeTolerance = 1e-4f;

    const auto edge = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b,
                         const Eigen::Vector2f& c) {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    };

    for (const auto& triangle : mesh.triangles) {
        auto vertices = std::array<Eigen::Vector3f, 3>{};
        auto projected = std::array<Eigen::Vector2f, 3>{};
        auto visible = true;
        for (std::size_t k = 0; k < 3; k++) {
            vertices[k] = rotation * triangle[k] + translation;
            if (vertices[k].z() < kNearPlane) {
                visible = false;
                break;
            }
            projected[k] = {
                fx * vertices[k].x() / vertices[k].z() + cx,
                fy * vertices[k].y() / vertices[k].z() + cy};
        }
        if (!visible)
            continue;

        const auto area = edge(projected[0], projected[1], projected[2]);
        if (std::abs(area) < 1e-9f)
            continue;

        const auto lo_x = std::min({projected[0].x(), projected[1].x(), projected[2].x()});
        const auto hi_x = std::max({projected[0].x(), projected[1].x(), projected[2].x()});
        const auto lo_y = std::min({projected[0].y(), projected[1].y(), projected[2].y()});
        const auto hi_y = std::max({projected[0].y(), projected[1].y(), projected[2].y()});
        const auto x0 = std::max(roi.x, static_cast<int>(std::ceil(lo_x)));
        const auto x1 = std::min(roi.x + roi.width - 1, static_cast<int>(std::floor(hi_x)));
        const auto y0 = std::max(roi.y, static_cast<int>(std::ceil(lo_y)));
        const auto y1 = std::min(roi.y + roi.height - 1, static_cast<int>(std::floor(hi_y)));

        for (int v = y0; v <= y1; v++) {
            auto* row = buffer.ptr<float>(v - roi.y);
            for (int u = x0; u <= x1; u++) {
                const Eigen::Vector2f pixel{static_cast<float>(u), static_cast<float>(v)};
                const auto w0 = edge(projected[1], projected[2], pixel) / area;
                const auto w1 = edge(projected[2], projected[0], pixel) / area;
                const auto w2 = 1.f - w0 - w1;
                if (w0 < -kEdgeTolerance || w1 < -kEdgeTolerance || w2 < -kEdgeTolerance)
                    continue;
                // 透视下 1/z 在屏幕空间线性插值
                const auto inverse_z =
                    w0 / vertices[0].z() + w1 / vertices[1].z() + w2 / vertices[2].z();
                const auto z = 1.f / inverse_z;
                auto& cell = row[u - roi.x];
                cell = std::min(cell, z);
            }
        }
    }
    return buffer;
}

} // namespace

DepthCheck::DepthCheck(geometry::Mesh mesh, const Config& config) noexcept
    : mesh_{std::move(mesh)}
    , config_{config} {
    auto lower = std::numeric_limits<float>::max();
    auto upper = std::numeric_limits<float>::lowest();
    for (const auto& triangle : mesh_.triangles) {
        for (const auto& vertex : triangle) {
            lower = std::min(lower, vertex.z());
            upper = std::max(upper, vertex.z());
        }
    }
    if (!mesh_.triangles.empty())
        half_length_ = (upper - lower) / 2.f;
}

DepthCheck::Quality DepthCheck::check(
    const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
    const cv::Mat& mask, const Eigen::Isometry3d& pose) const noexcept {
    auto quality = Quality{};
    const auto rect = roi & cv::Rect2i{0, 0, depth.cols, depth.rows};
    if (depth.type() != CV_16UC1 || rect.empty())
        return quality;
    if (!mask.empty() && (mask.type() != CV_8UC1 || mask.size() != depth.size()))
        return quality;

    const auto buffer = rasterize_depth(mesh_, pose, camera, rect);
    auto silhouette = cv::Mat{};
    cv::compare(buffer, kNoDepth, silhouette, cv::CMP_LT);
    if (config_.silhouette_erosion_px > 0) {
        const auto size = 2 * config_.silhouette_erosion_px + 1;
        cv::erode(
            silhouette, silhouette,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{size, size}));
    }

    // 掩码只决定一个像素能否算作支持：掩码外是别的东西，深度相同或更近都说明不了位姿对错，
    // 不统计；但掩码外看到比 mesh 更远的背景，说明 mesh 所在处是空的，仍记为违例
    auto supported = std::size_t{}, free_space = std::size_t{}, occluded = std::size_t{},
         missing = std::size_t{};
    for (int v = 0; v < rect.height; v++) {
        const auto* model_row = buffer.ptr<float>(v);
        const auto* silhouette_row = silhouette.ptr<std::uint8_t>(v);
        const auto* depth_row = depth.ptr<std::uint16_t>(v + rect.y);
        const auto* mask_row = mask.empty() ? nullptr : mask.ptr<std::uint8_t>(v + rect.y);
        for (int u = 0; u < rect.width; u++) {
            if (silhouette_row[u] == 0)
                continue;
            const auto in_mask = mask_row == nullptr || mask_row[u + rect.x] != 0;
            const auto measured = depth_row[u + rect.x];
            if (in_mask) {
                quality.model_pixels++;
                if (measured == 0)
                    missing++;
            }
            if (measured == 0)
                continue;
            const auto difference =
                static_cast<float>(measured * camera.depth_scale) - model_row[u];
            if (difference > config_.depth_tolerance)
                free_space++;
            else if (!in_mask)
                continue;
            else if (difference < -config_.depth_tolerance)
                occluded++;
            else
                supported++;
            quality.valid_pixels++;
        }
    }
    if (quality.model_pixels > 0) {
        quality.missing_ratio =
            static_cast<double>(missing) / static_cast<double>(quality.model_pixels);
    }
    if (quality.valid_pixels > 0) {
        const auto count = static_cast<double>(quality.valid_pixels);
        quality.support_ratio = static_cast<double>(supported) / count;
        quality.free_space_ratio = static_cast<double>(free_space) / count;
        quality.occluded_ratio = static_cast<double>(occluded) / count;
    }
    return quality;
}

cv::Rect2i DepthCheck::projected_box(
    const geometry::Camera& camera, const Eigen::Isometry3d& pose) const noexcept {
    const Eigen::Matrix3f rotation = pose.linear().cast<float>();
    const Eigen::Vector3f translation = pose.translation().cast<float>();
    auto lower =
        Eigen::Vector2f{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    auto upper =
        Eigen::Vector2f{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    for (const auto& triangle : mesh_.triangles) {
        for (const auto& vertex : triangle) {
            const Eigen::Vector3f p = rotation * vertex + translation;
            if (p.z() <= 0.f)
                return {};
            const Eigen::Vector2f pixel{
                static_cast<float>(camera.fx) * p.x() / p.z() + static_cast<float>(camera.cx),
                static_cast<float>(camera.fy) * p.y() / p.z() + static_cast<float>(camera.cy)};
            lower = lower.cwiseMin(pixel);
            upper = upper.cwiseMax(pixel);
        }
    }
    if (mesh_.triangles.empty())
        return {};
    const auto left = static_cast<int>(std::floor(lower.x()));
    const auto top = static_cast<int>(std::floor(lower.y()));
    const auto right = static_cast<int>(std::ceil(upper.x()));
    const auto bottom = static_cast<int>(std::ceil(upper.y()));
    return cv::Rect2i{left, top, right - left + 1, bottom - top + 1}
         & cv::Rect2i{0, 0, camera.width, camera.height};
}

double DepthCheck::end_in_image_ratio(
    const geometry::Camera& camera, const Eigen::Isometry3d& pose) const noexcept {
    const Eigen::Matrix3f rotation = pose.linear().cast<float>();
    const Eigen::Vector3f translation = pose.translation().cast<float>();
    // 下标 0 为 z < 0 的端部，1 为 z > 0 的端部
    auto total = std::array<double, 2>{};
    auto inside = std::array<double, 2>{};
    for (const auto& triangle : mesh_.triangles) {
        const Eigen::Vector3f centroid = (triangle[0] + triangle[1] + triangle[2]) / 3.f;
        if (std::abs(centroid.z()) <= half_length_ / 2.f)
            continue;
        const auto end = centroid.z() > 0.f ? 1 : 0;
        const auto area = static_cast<double>(geometry::Mesh::area(triangle));
        total[end] += area;
        const Eigen::Vector3f center = rotation * centroid + translation;
        if (center.z() <= 0.f)
            continue;
        const auto u = camera.fx * center.x() / center.z() + camera.cx;
        const auto v = camera.fy * center.y() / center.z() + camera.cy;
        // 像素 (u, v) 的中心在投影坐标 (u, v)，图像覆盖 [-0.5, width - 0.5)
        if (u >= -0.5 && u < camera.width - 0.5 && v >= -0.5 && v < camera.height - 0.5)
            inside[end] += area;
    }
    if (total[0] <= 0. || total[1] <= 0.)
        return 0.;
    return std::min(inside[0] / total[0], inside[1] / total[1]);
}

} // namespace rmcs_energy_trace::registration
