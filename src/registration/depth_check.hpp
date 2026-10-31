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

    // 两端被观测到的像素比例。端部的划分：三角形重心 |z| > 半长 / 2，
    // +z 为封闭端，-z
    // 为开口端。分母是该端在完整投影（含画面外部分，轮廓同样向内腐蚀）中的模型像素数；
    // 分子是其中在画面内、有实测深度、且没有被遮挡（check 中判为支持或空闲空间违例）的像素数。
    // 不使用掩码。
    // 轴向位置只能靠端部约束，中段沿轴线滑动时深度几乎不变；端部被挡住、没有深度或出了画面时，
    // 这个比例会降低，其余的深度统计看不出来。监督和配准只用封闭端：开口端套在柱子上，正确时也大多看不到
    struct EndObserved {
        double closed = 0.;
        double open = 0.;
    };
    EndObserved end_observed_ratios(
        const cv::Mat& depth, const geometry::Camera& camera,
        const Eigen::Isometry3d& pose) const noexcept;

private:
    geometry::Mesh mesh_;
    float half_length_ = 0.f; // mesh 在 z 方向的半长
    Config config_;
};

} // namespace rmcs_energy_trace::registration
