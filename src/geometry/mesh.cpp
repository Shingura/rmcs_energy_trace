#include "geometry/mesh.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>

namespace rmcs_energy_trace::geometry {

namespace {

constexpr std::size_t kHeaderSize = 80;
constexpr std::size_t kTriangleSize = 50;
constexpr float kMillimeterToMeter = 0.001f;

} // namespace

std::expected<Mesh, std::string> load_binary_stl(const std::string& path) noexcept {
    auto file = std::ifstream{path, std::ios::binary};
    if (!file)
        return std::unexpected{"Failed to open stl file: " + path};

    auto header = std::array<char, kHeaderSize>{};
    auto count = std::uint32_t{};
    file.read(header.data(), header.size());
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file)
        return std::unexpected{"Invalid stl header: " + path};

    auto mesh = Mesh{};
    mesh.triangles.reserve(count);

    auto record = std::array<char, kTriangleSize>{};
    for (std::uint32_t i = 0; i < count; i++) {
        file.read(record.data(), record.size());
        if (!file)
            return std::unexpected{"Truncated stl file: " + path};

        // 12 字节法线之后是三个顶点，每个 3 个 float
        auto floats = std::array<float, 9>{};
        std::memcpy(floats.data(), record.data() + 12, sizeof(floats));

        auto triangle = Mesh::Triangle{};
        for (std::size_t v = 0; v < 3; v++) {
            triangle[v] = Eigen::Vector3f{floats[3 * v], floats[3 * v + 1], floats[3 * v + 2]}
                        * kMillimeterToMeter;
        }
        mesh.triangles.push_back(triangle);
    }

    Eigen::Vector3f lower = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
    Eigen::Vector3f upper = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
    for (const auto& triangle : mesh.triangles) {
        for (const auto& vertex : triangle) {
            lower = lower.cwiseMin(vertex);
            upper = upper.cwiseMax(vertex);
        }
    }
    const Eigen::Vector3f center = 0.5f * (lower + upper);
    for (auto& triangle : mesh.triangles) {
        for (auto& vertex : triangle)
            vertex -= center;
    }

    return mesh;
}

pcl::PointCloud<pcl::PointNormal>::Ptr
    sample_surface(const Mesh& mesh, std::size_t count, std::uint32_t seed) noexcept {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    if (mesh.triangles.empty() || count == 0)
        return cloud;

    auto cumulative = std::vector<float>{};
    cumulative.reserve(mesh.triangles.size());
    std::ranges::transform(
        mesh.triangles, std::back_inserter(cumulative),
        [total = 0.f](const Mesh::Triangle& triangle) mutable {
            return total += Mesh::area(triangle);
        });

    auto engine = std::mt19937{seed};
    auto area_dist = std::uniform_real_distribution<float>{0.f, cumulative.back()};
    auto unit_dist = std::uniform_real_distribution<float>{0.f, 1.f};

    cloud->reserve(count);
    for (std::size_t i = 0; i < count; i++) {
        const auto it = std::ranges::lower_bound(cumulative, area_dist(engine));
        const auto index =
            std::min(static_cast<std::size_t>(it - cumulative.begin()), mesh.triangles.size() - 1);
        const auto& t = mesh.triangles[index];

        // 均匀采样三角形内的点
        auto u = unit_dist(engine);
        auto v = unit_dist(engine);
        if (u + v > 1.f) {
            u = 1.f - u;
            v = 1.f - v;
        }
        const Eigen::Vector3f p = t[0] + u * (t[1] - t[0]) + v * (t[2] - t[0]);
        const Eigen::Vector3f n = (t[1] - t[0]).cross(t[2] - t[0]).normalized();

        auto point = pcl::PointNormal{};
        point.getVector3fMap() = p;
        point.getNormalVector3fMap() = n;
        cloud->push_back(point);
    }
    return cloud;
}

} // namespace rmcs_energy_trace::geometry
