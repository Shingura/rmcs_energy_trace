// 离线生成 M3T 跟踪所需的文件：energy_unit.obj 以及区域 / 深度模型。
// 模型生成要渲染（OpenGL），所以本工具只在 RMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON 时构建，
// 运行时需要显示：env -u WAYLAND_DISPLAY DISPLAY=:0 m3t_generate
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <string>
#include <system_error>

#include <m3t/body.h>
#include <m3t/depth_model.h>
#include <m3t/region_model.h>

#include "geometry/mesh.hpp"
#include "tracking/mesh_obj.hpp"
#include "tracking/model_files.hpp"

#ifndef RMCS_ENERGY_UNIT_MESH
# error "RMCS_ENERGY_UNIT_MESH must be defined"
#endif
#ifndef RMCS_ENERGY_TRACE_DEFAULT_MODEL_DIR
# error "RMCS_ENERGY_TRACE_DEFAULT_MODEL_DIR must be defined"
#endif

namespace fs = std::filesystem;
namespace tracking = rmcs_energy_trace::tracking;

namespace {

void print_usage() {
    std::println(
        "Usage: m3t_generate [--mesh <energy_unit.stl>] [--out <dir>] [--sphere-radius 0.8]\n"
        "                    [--n-divides 4] [--n-points 200] [--image-size 2000]\n"
        "Default mesh: {}\nDefault out: {}",
        RMCS_ENERGY_UNIT_MESH, RMCS_ENERGY_TRACE_DEFAULT_MODEL_DIR);
}

auto seconds_since(std::chrono::steady_clock::time_point begin) -> double {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    auto mesh_path = std::string{RMCS_ENERGY_UNIT_MESH};
    auto out_dir = fs::path{RMCS_ENERGY_TRACE_DEFAULT_MODEL_DIR};
    auto params = tracking::ModelParams{};

    for (int i = 1; i < argc; i++) {
        const auto arg = std::string{argv[i]};
        if (arg == "--mesh" && i + 1 < argc) {
            mesh_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--sphere-radius" && i + 1 < argc) {
            params.sphere_radius = std::strtof(argv[++i], nullptr);
        } else if (arg == "--n-divides" && i + 1 < argc) {
            params.n_divides = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (arg == "--n-points" && i + 1 < argc) {
            params.n_points = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (arg == "--image-size" && i + 1 < argc) {
            params.image_size = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else {
            print_usage();
            return EXIT_FAILURE;
        }
    }

    auto mesh = rmcs_energy_trace::geometry::load_binary_stl(mesh_path);
    if (!mesh) {
        std::println(stderr, "{}", mesh.error());
        return EXIT_FAILURE;
    }

    auto error = std::error_code{};
    fs::create_directories(out_dir, error);
    if (error) {
        std::println(stderr, "Failed to create {}: {}", out_dir.string(), error.message());
        return EXIT_FAILURE;
    }
    // 模型文件里记录的是 OBJ 的路径，用绝对路径
    out_dir = fs::absolute(out_dir);
    const auto obj_path = out_dir / tracking::kMeshFile;
    const auto region_path = out_dir / tracking::kRegionModelFile;
    const auto depth_path = out_dir / tracking::kDepthModelFile;

    if (auto written = tracking::write_obj(*mesh, obj_path); !written) {
        std::println(stderr, "{}", written.error());
        return EXIT_FAILURE;
    }
    // M3T 在模型文件有效时直接读取；这里要重新生成，先删除旧文件
    fs::remove(region_path, error);
    fs::remove(depth_path, error);

    // mesh 已是米，原点在包围盒中心，z 为长轴，即位姿约定，geometry2body 为单位阵
    auto body = std::make_shared<m3t::Body>(
        "energy_unit", obj_path, 1.f, true, true, m3t::Transform3fA::Identity());
    if (!body->SetUp()) {
        std::println(stderr, "Failed to set up body from {}", obj_path.string());
        return EXIT_FAILURE;
    }
    std::println(
        "mesh: {} triangles, max diameter {:.4f} m", mesh->triangles.size(),
        body->maximum_body_diameter());

    const auto region_begin = std::chrono::steady_clock::now();
    auto region_model = m3t::RegionModel{"region_model",
                                         body,
                                         region_path,
                                         params.sphere_radius,
                                         params.n_divides,
                                         params.n_points,
                                         0.05f,
                                         0.002f,
                                         false,
                                         params.image_size};
    if (!region_model.SetUp()) {
        std::println(stderr, "Failed to generate region model (is a display available?)");
        return EXIT_FAILURE;
    }
    std::println("region model: {:.1f} s -> {}", seconds_since(region_begin), region_path.string());

    const auto depth_begin = std::chrono::steady_clock::now();
    auto depth_model = m3t::DepthModel{
        "depth_model", body,   depth_path, params.sphere_radius, params.n_divides, params.n_points,
        0.05f,         0.002f, false,      params.image_size};
    if (!depth_model.SetUp()) {
        std::println(stderr, "Failed to generate depth model (is a display available?)");
        return EXIT_FAILURE;
    }
    std::println("depth model: {:.1f} s -> {}", seconds_since(depth_begin), depth_path.string());
    return EXIT_SUCCESS;
}
