#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include <eigen3/Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace rmcs_energy_trace::pose {

struct Mesh {
    using Triangle = std::array<Eigen::Vector3f, 3>;
    std::vector<Triangle> triangles;

    // 三角形面积，用于面积加权采样
    static float area(const Triangle& triangle) noexcept {
        return 0.5f * (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).norm();
    }
};

// 读取二进制 STL，单位 mm 转换为 m，并平移使包围盒中心位于原点
std::expected<Mesh, std::string> load_binary_stl(const std::string& path) noexcept;

// 按面积加权均匀采样表面，法线由三角形绕序得到（朝外），seed 固定保证可复现
pcl::PointCloud<pcl::PointNormal>::Ptr
    sample_surface(const Mesh& mesh, std::size_t count, std::uint32_t seed) noexcept;

} // namespace rmcs_energy_trace::pose
