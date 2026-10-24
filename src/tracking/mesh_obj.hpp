#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "geometry/mesh.hpp"

namespace rmcs_energy_trace::tracking {

// 把 mesh 写成 OBJ，单位 m，坐标系与 mesh 相同（原点在包围盒中心，z 为长轴）。
// 相同位置的顶点合并，三角形绕序保持不变（逆时针朝外）。
// M3T 的 Body 以 geometry_unit_in_meter = 1、geometry2body = 单位阵 读取该文件。
std::expected<void, std::string>
    write_obj(const geometry::Mesh& mesh, const std::filesystem::path& path) noexcept;

} // namespace rmcs_energy_trace::tracking
