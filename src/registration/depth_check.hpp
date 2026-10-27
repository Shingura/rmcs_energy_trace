#pragma once

#include <cstddef>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"

namespace rmcs_energy_trace::registration {

// 位姿的深度检查：把 mesh 按位姿渲染到 ROI 内（轮廓向内腐蚀），逐像素与实测深度比较。
// 配准用它验收候选位姿，监督用它判断跟踪结果是否可信
class DepthCheck {
public:
    struct Config {
        // mesh 渲染深度与实测深度之差小于该值视为支持 (m)
        float depth_tolerance = 0.005f;
        // 渲染轮廓向内腐蚀的像素数，去掉边缘上混合深度造成的误判
        int silhouette_erosion_px = 2;
    };

    // 只看有实测深度的像素：
    //   支持    |z_实测 - z_渲染| < depth_tolerance
    //   空闲空间违例  z_实测 > z_渲染 + depth_tolerance，即 mesh 所在处看到了更远的东西
    //   遮挡    z_实测 < z_渲染 - depth_tolerance，即前面有别的东西
    // 掩码内（没有掩码时为整个 ROI）三类都计入 valid。掩码外只计入空闲空间违例：
    // 那里深度相同或更近可能是相邻物体，说明不了位姿对错。
    // 遮挡计入 valid、会降低支持率，这是有意的：否则一个整体落在真实表面后面的错误位姿，
    // 只剩少数像素参与统计，也能得到很高的支持率。所以没有掩码时，目标被挡住较多会被拒绝
    struct Quality {
        std::size_t model_pixels = 0; // 渲染 mesh 在掩码内的像素数（没有掩码时为 ROI 内）
        std::size_t valid_pixels = 0; // valid 的像素数
        double support_ratio = 0.;    // 支持 / valid
        double free_space_ratio = 0.; // 空闲空间违例 / valid
        double occluded_ratio = 0.;   // 遮挡 / valid
        double missing_ratio = 0.;    // 掩码内无实测深度的像素 / model_pixels
    };

    DepthCheck() = default;
    DepthCheck(geometry::Mesh mesh, const Config& config) noexcept;

    // depth 为 CV_16UC1（单位见 camera.depth_scale，0 为无效），pose 为 mesh -> camera。
    // mask 为空表示不限制；非空时为 CV_8UC1，与深度图同尺寸
    Quality check(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask, const Eigen::Isometry3d& pose) const noexcept;

    // mesh 按位姿投影后的像素包围框，已与图像求交；mesh 有顶点在相机后方时为空
    cv::Rect2i
        projected_box(const geometry::Camera& camera, const Eigen::Isometry3d& pose) const noexcept;

    // mesh 两端表面按位姿投影后落在画面内的面积比例，取两端中较小的一个，不考虑自遮挡。
    // mesh 原点在包围盒中心、z 为长轴：重心 |z| 大于半长一半（两端各 1/4 长度）的三角形属于端部，
    // z > 0 和 z < 0
    // 各算一端。每个端部按三角形面积加权，重心在相机前方且投影在图像范围内的算画面内。
    // 能量单元的轴向位置只能靠两端约束，中段沿轴线滑动时深度几乎不变；一端出了画面，
    // 跟踪器会沿轴线滑动，深度检查又只统计画面内的像素，看不出来
    double end_in_image_ratio(
        const geometry::Camera& camera, const Eigen::Isometry3d& pose) const noexcept;

private:
    geometry::Mesh mesh_;
    float half_length_ = 0.f; // mesh 在 z 方向的半长
    Config config_;
};

} // namespace rmcs_energy_trace::registration
