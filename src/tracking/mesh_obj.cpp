#include "tracking/mesh_obj.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <map>
#include <vector>

namespace rmcs_energy_trace::tracking {

namespace {

// 顶点合并的量化精度 (m)，远小于 STL 单精度的有效位数
constexpr float kQuantum = 1e-7f;

using Key = std::array<std::int64_t, 3>;

auto quantize(const Eigen::Vector3f& v) -> Key {
    return {
        std::llround(v.x() / kQuantum),
        std::llround(v.y() / kQuantum),
        std::llround(v.z() / kQuantum),
    };
}

} // namespace

std::expected<void, std::string>
    write_obj(const geometry::Mesh& mesh, const std::filesystem::path& path) noexcept try {
    auto indices = std::map<Key, std::size_t>{};
    auto vertices = std::vector<Eigen::Vector3f>{};
    auto faces = std::vector<std::array<std::size_t, 3>>{};
    faces.reserve(mesh.triangles.size());

    for (const auto& triangle : mesh.triangles) {
        auto face = std::array<std::size_t, 3>{};
        for (std::size_t i = 0; i < 3; i++) {
            const auto [it, inserted] = indices.try_emplace(quantize(triangle[i]), vertices.size());
            if (inserted)
                vertices.push_back(triangle[i]);
            face[i] = it->second;
        }
        // 丢弃退化三角形
        if (face[0] != face[1] && face[1] != face[2] && face[0] != face[2])
            faces.push_back(face);
    }
    if (faces.empty())
        return std::unexpected{"Mesh has no valid triangle"};

    auto file = std::ofstream{path};
    if (!file)
        return std::unexpected{"Failed to open obj file for writing: " + path.string()};
    for (const auto& v : vertices)
        file << std::format("v {:.7f} {:.7f} {:.7f}\n", v.x(), v.y(), v.z());
    // OBJ 的顶点编号从 1 开始
    for (const auto& f : faces)
        file << std::format("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1);
    file.flush();
    if (!file)
        return std::unexpected{"Failed to write obj file: " + path.string()};
    return {};
} catch (const std::exception& e) {
    return std::unexpected{std::string{"Failed to write obj file: "} + e.what()};
}

} // namespace rmcs_energy_trace::tracking
