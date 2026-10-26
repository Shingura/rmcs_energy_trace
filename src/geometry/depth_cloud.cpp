#include "geometry/depth_cloud.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <opencv2/imgproc.hpp>

namespace rmcs_energy_trace::geometry {

pcl::PointCloud<pcl::PointXYZ>::Ptr depth_to_cloud(
    const cv::Mat& depth, const Camera& camera, const cv::Rect2i& roi, const cv::Mat& mask,
    int mask_erosion_px) noexcept {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (depth.type() != CV_16UC1)
        return cloud;

    auto eroded = cv::Mat{};
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1 || mask.size() != depth.size())
            return cloud;
        if (mask_erosion_px > 0) {
            const auto size = 2 * mask_erosion_px + 1;
            cv::erode(
                mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{size, size}));
        } else {
            eroded = mask;
        }
    }

    const auto rect = roi & cv::Rect2i{0, 0, depth.cols, depth.rows};
    cloud->reserve(static_cast<std::size_t>(rect.area()));
    for (int v = rect.y; v < rect.y + rect.height; v++) {
        const auto* row = depth.ptr<std::uint16_t>(v);
        const auto* mask_row = eroded.empty() ? nullptr : eroded.ptr<std::uint8_t>(v);
        for (int u = rect.x; u < rect.x + rect.width; u++) {
            if (row[u] == 0)
                continue;
            if (mask_row != nullptr && mask_row[u] == 0)
                continue;
            // 针孔模型：像素 (u, v) 的中心对应投影坐标 (u, v)
            const auto z = static_cast<double>(row[u]) * camera.depth_scale;
            cloud->emplace_back(
                static_cast<float>((u - camera.cx) * z / camera.fx),
                static_cast<float>((v - camera.cy) * z / camera.fy), static_cast<float>(z));
        }
    }
    return cloud;
}

} // namespace rmcs_energy_trace::geometry
