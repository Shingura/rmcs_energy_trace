#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "geometry/camera.hpp"

namespace rmcs_energy_trace::geometry {

// 深度图 (CV_16UC1，单位见 camera.depth_scale，0 为无效) 在 roi 内反投影为相机系点云 (m)。
// mask 非空时只保留掩码 (CV_8UC1，与深度图同尺寸，非 0 保留) 内的像素，掩码先向内腐蚀
// mask_erosion_px 个像素，去掉边界上混合了前景和背景的深度。
// 深度图或掩码的类型、尺寸不符时返回空点云
pcl::PointCloud<pcl::PointXYZ>::Ptr depth_to_cloud(
    const cv::Mat& depth, const Camera& camera, const cv::Rect2i& roi, const cv::Mat& mask = {},
    int mask_erosion_px = 0) noexcept;

} // namespace rmcs_energy_trace::geometry
