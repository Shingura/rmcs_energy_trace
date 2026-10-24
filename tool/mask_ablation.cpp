#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "pose/depth_renderer.hpp"
#include "pose/mesh_model.hpp"
#include "pose/point_cloud_registration.hpp"

#ifndef RMCS_ENERGY_UNIT_MODEL
# error "RMCS_ENERGY_UNIT_MODEL must be defined"
#endif

// 消融实验：同一帧合成深度图上，比较只用检测框与加实例掩码的点云配准。
// 场景与测试 B 相同（能量单元直立在地面上，相机俯视 30~60 度，距离 0.4~0.8 m），
// 另在目标旁放 1~2 个相邻的能量单元。

using namespace rmcs_energy_trace::pose;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr PointCloudRegistration::Intrinsics kIntrinsics{600., 600., 320., 240.};

// 与测试 B 一致：地面在目标中心之下 0.0725 m
constexpr double kGroundOffset = 0.0725;
constexpr double kRoiMargin = 0.1;
constexpr double kCenterThresholdMm = 3.;
constexpr double kAxisThresholdDeg = 2.;
// 目标可见像素少于无遮挡时的这个比例则重新生成场景
constexpr double kMinVisibleFraction = 0.5;
constexpr std::array<double, 3> kGapsMm = {5., 20., 50.};

constexpr std::uint8_t kTargetId = 1;

using Polygon = std::vector<cv::Point2f>;

auto uniform(std::mt19937& engine, double lo, double hi) -> double {
    return std::uniform_real_distribution<double>{lo, hi}(engine);
}

auto uniform_int(std::mt19937& engine, int lo, int hi) -> int {
    return std::uniform_int_distribution<int>{lo, hi}(engine);
}

auto median(std::vector<double> values) -> double {
    std::ranges::sort(values);
    return values[values.size() / 2];
}

// 点到线段的距离
auto point_segment_distance(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b)
    -> double {
    const auto ab = b - a;
    const auto length2 = static_cast<double>(ab.dot(ab));
    const auto t =
        length2 > 0. ? std::clamp(static_cast<double>((p - a).dot(ab)) / length2, 0., 1.) : 0.;
    const auto d = p - (a + static_cast<float>(t) * ab);
    return std::sqrt(static_cast<double>(d.dot(d)));
}

// 两个凸多边形边界之间的最短距离（不处理相交）
auto polygon_distance(const Polygon& a, const Polygon& b, const cv::Point2f& shift_b) -> double {
    auto best = 1e9;
    for (std::size_t i = 0; i < a.size(); i++) {
        for (std::size_t j = 0; j < b.size(); j++) {
            const auto b0 = b[j] + shift_b;
            const auto b1 = b[(j + 1) % b.size()] + shift_b;
            best = std::min(best, point_segment_distance(a[i], b0, b1));
            const auto a0 = a[i];
            const auto a1 = a[(i + 1) % a.size()];
            best = std::min(best, point_segment_distance(b0, a0, a1));
        }
    }
    return best;
}

auto polygons_overlap(const Polygon& a, const Polygon& b, const cv::Point2f& shift_b) -> bool {
    auto shifted = b;
    for (auto& point : shifted)
        point += shift_b;
    return std::ranges::any_of(
               a, [&](const auto& p) { return cv::pointPolygonTest(shifted, p, false) >= 0.; })
        || std::ranges::any_of(
               shifted, [&](const auto& p) { return cv::pointPolygonTest(a, p, false) >= 0.; });
}

struct Neighbor {
    Eigen::Matrix3d rotation; // 世界系，原点在目标中心，z 向上
    Eigen::Vector3d position;
    Polygon footprint;        // 相对自身中心的地面投影凸包
    double gap_mm;
    bool lying;
    bool front;
};

struct Scene {
    Eigen::Isometry3d target;
    Eigen::Matrix3d world_to_camera;
    Eigen::Vector3d center;
    std::vector<Neighbor> neighbors;
    std::vector<Eigen::Isometry3d> neighbor_poses;
    DepthRenderer::Plane ground;
};

class Generator {
public:
    explicit Generator(const Mesh& mesh) {
        const auto samples = sample_surface(mesh, 20000, 5);
        for (const auto& point : *samples) {
            points_.emplace_back(point.getVector3fMap().cast<double>());
        }
    }

    static auto make_scene(std::mt19937& engine) -> Scene {
        auto scene = Scene{};
        const auto distance = uniform(engine, 0.4, 0.8);
        const auto tilt = uniform(engine, 30., 60.) * std::numbers::pi / 180.;
        const auto yaw = uniform(engine, 0., 2. * std::numbers::pi);

        // 世界系 z 向上；相机朝 +x 并向下俯视 tilt
        const Eigen::Vector3d forward{std::cos(tilt), 0., -std::sin(tilt)};
        const Eigen::Vector3d right{0., -1., 0.};
        const Eigen::Vector3d down = forward.cross(right);
        scene.world_to_camera.row(0) = right;
        scene.world_to_camera.row(1) = down;
        scene.world_to_camera.row(2) = forward;

        scene.center = {
            uniform(engine, -0.05, 0.05) * distance / 0.6,
            uniform(engine, -0.04, 0.04) * distance / 0.6,
            distance,
        };
        const Eigen::Matrix3d target_rotation = rotation_z(yaw);
        scene.target = Eigen::Isometry3d::Identity();
        scene.target.linear() = scene.world_to_camera * target_rotation;
        scene.target.translation() = scene.center;

        const Eigen::Vector3d up = scene.world_to_camera * Eigen::Vector3d::UnitZ();
        scene.ground = {up, scene.center - kGroundOffset * up};
        return scene;
    }

    // 在场景中放 1~2 个相邻单元，第一个有三分之一概率放在相机一侧
    void add_neighbors(Scene& scene, std::mt19937& engine) const {
        const auto target_footprint = footprint(rotation_z(0.));
        const auto count = uniform_int(engine, 1, 2);
        for (int k = 0; k < count; k++) {
            for (int attempt = 0; attempt < 30; attempt++) {
                auto neighbor = Neighbor{};
                neighbor.gap_mm = kGapsMm[static_cast<std::size_t>(uniform_int(engine, 0, 2))];
                neighbor.lying = uniform_int(engine, 0, 1) == 1;
                neighbor.front = k == 0 && uniform_int(engine, 0, 2) == 0;

                const auto yaw = uniform(engine, 0., 2. * std::numbers::pi);
                neighbor.rotation = rotation_z(yaw);
                if (neighbor.lying) {
                    const auto spin = uniform(engine, 0., 2. * std::numbers::pi);
                    neighbor.rotation =
                        neighbor.rotation
                        * Eigen::AngleAxisd{std::numbers::pi / 2, Eigen::Vector3d::UnitX()}
                              .toRotationMatrix()
                        * rotation_z(spin);
                }
                neighbor.footprint = footprint(neighbor.rotation);

                // 相机在世界系 -x 方向，front 时邻居放在目标与相机之间
                const auto azimuth = neighbor.front
                                       ? std::numbers::pi + uniform(engine, -0.45, 0.45)
                                       : uniform(engine, 0., 2. * std::numbers::pi);
                const auto offset =
                    place(target_footprint, neighbor.footprint, azimuth, neighbor.gap_mm * 1e-3);
                const cv::Point2f shift{
                    static_cast<float>(offset * std::cos(azimuth)),
                    static_cast<float>(offset * std::sin(azimuth))};

                // 与已放置的邻居不能重叠
                const auto blocked =
                    std::ranges::any_of(scene.neighbors, [&](const Neighbor& other) {
                        const cv::Point2f other_shift{
                            static_cast<float>(other.position.x()),
                            static_cast<float>(other.position.y())};
                        return polygons_overlap(
                                   other.footprint, neighbor.footprint, shift - other_shift)
                            || polygon_distance(
                                   other.footprint, neighbor.footprint, shift - other_shift)
                                   < 0.01;
                    });
                if (blocked)
                    continue;

                neighbor.position = {shift.x, shift.y, 0.};
                if (neighbor.lying) {
                    // 最低点落在地面上
                    auto min_z = 1e9;
                    for (const auto& point : points_)
                        min_z = std::min(min_z, (neighbor.rotation * point).z());
                    neighbor.position.z() = -kGroundOffset - min_z;
                }
                scene.neighbors.push_back(neighbor);
                break;
            }
        }

        for (const auto& neighbor : scene.neighbors) {
            auto pose = Eigen::Isometry3d::Identity();
            pose.linear() = scene.world_to_camera * neighbor.rotation;
            pose.translation() = scene.world_to_camera * neighbor.position + scene.center;
            scene.neighbor_poses.push_back(pose);
        }
    }

private:
    static auto rotation_z(double angle) -> Eigen::Matrix3d {
        return Eigen::AngleAxisd{angle, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
    }

    auto footprint(const Eigen::Matrix3d& rotation) const -> Polygon {
        auto projected = Polygon{};
        projected.reserve(points_.size());
        for (const auto& point : points_) {
            const Eigen::Vector3d p = rotation * point;
            projected.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()));
        }
        auto hull = Polygon{};
        cv::convexHull(projected, hull);
        return hull;
    }

    // 沿方向 azimuth 从远处向目标靠近，返回邻居与目标地面投影间距刚好不小于 gap 的位置
    static auto place(const Polygon& target, const Polygon& neighbor, double azimuth, double gap)
        -> double {
        constexpr auto kStep = 0.00025;
        const Eigen::Vector2d direction{std::cos(azimuth), std::sin(azimuth)};
        auto offset = 0.5;
        while (offset > 0.) {
            const auto next = offset - kStep;
            const auto shift = cv::Point2f{
                static_cast<float>(next * direction.x()), static_cast<float>(next * direction.y())};
            if (polygon_distance(target, neighbor, shift) < gap)
                break;
            offset = next;
        }
        return offset;
    }

    std::vector<Eigen::Vector3d> points_;
};

// 目标可见像素的包围盒外扩，作为 YOLO 检测框的替身
auto make_roi(const cv::Mat& target_mask) -> cv::Rect2i {
    auto roi = cv::boundingRect(target_mask);
    const auto dx = static_cast<int>(roi.width * kRoiMargin);
    const auto dy = static_cast<int>(roi.height * kRoiMargin);
    return cv::Rect2i{roi.x - dx, roi.y - dy, roi.width + 2 * dx, roi.height + 2 * dy}
         & cv::Rect2i{0, 0, kWidth, kHeight};
}

// 模拟 YOLO-seg：缩小 4 倍再双线性放大并二值化，最后随机膨胀或腐蚀 0~2 px
auto degrade_mask(const cv::Mat& exact, std::mt19937& engine) -> cv::Mat {
    auto small = cv::Mat{}, large = cv::Mat{}, result = cv::Mat{};
    cv::resize(exact, small, cv::Size{kWidth / 4, kHeight / 4}, 0., 0., cv::INTER_AREA);
    cv::resize(small, large, exact.size(), 0., 0., cv::INTER_LINEAR);
    cv::threshold(large, result, 127., 255., cv::THRESH_BINARY);

    const auto pixels = uniform_int(engine, 0, 2);
    const auto dilate = uniform_int(engine, 0, 1) == 1;
    if (pixels > 0) {
        const auto size = 2 * pixels + 1;
        const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{size, size});
        if (dilate)
            cv::dilate(result, result, kernel);
        else
            cv::erode(result, result, kernel);
    }
    return result;
}

struct Outcome {
    bool has_result = false;
    double center_mm = 0.;
    double axis_deg = 0.;
    double time_ms = 0.;
    double inlier_ratio = 0.;
};

struct Stats {
    std::string name;
    std::size_t trials = 0;
    std::size_t success = 0;
    std::size_t none = 0;
    std::vector<double> center, axis, time;

    explicit Stats(std::string label)
        : name{std::move(label)} {}

    void add(const Outcome& outcome) {
        trials++;
        time.push_back(outcome.time_ms);
        if (!outcome.has_result) {
            none++;
            return;
        }
        center.push_back(outcome.center_mm);
        axis.push_back(outcome.axis_deg);
        if (outcome.center_mm < kCenterThresholdMm && outcome.axis_deg < kAxisThresholdDeg)
            success++;
    }
};

auto run_registration(
    const PointCloudRegistration& registration, const PointCloudRegistration::Cloud& cloud,
    const Eigen::Isometry3d& truth, Clock::time_point begin) -> Outcome {
    auto outcome = Outcome{};
    const auto result = registration.register_cloud(cloud);
    outcome.time_ms = std::chrono::duration<double, std::milli>{Clock::now() - begin}.count();
    if (!result.has_value())
        return outcome;
    outcome.has_result = true;
    outcome.center_mm = (truth.translation() - result->pose.translation()).norm() * 1e3;
    // 与 test 一致：方向反了（两端认反）也算误差
    const auto axis_cos =
        std::clamp(truth.linear().col(2).dot(result->pose.linear().col(2)), -1., 1.);
    outcome.axis_deg = std::acos(axis_cos) * 180. / std::numbers::pi;
    outcome.inlier_ratio = result->inlier_ratio;
    return outcome;
}

auto describe(const Outcome& outcome) -> std::string {
    if (!outcome.has_result)
        return "none";
    return std::format("{:.1f}mm/{:.1f}deg", outcome.center_mm, outcome.axis_deg);
}

void print_row(const Stats& stats) {
    const auto text = [](const std::vector<double>& values, bool use_max) {
        if (values.empty())
            return std::string{"-"};
        return std::format("{:.2f}", use_max ? *std::ranges::max_element(values) : median(values));
    };
    std::println(
        "| {:<30} | {:>3}/{:<3} ({:>3.0f}%) | {:>7} | {:>7} | {:>7} | {:>7} | {:>9} | {:>9} |",
        stats.name, stats.success, stats.trials,
        100. * static_cast<double>(stats.success) / static_cast<double>(stats.trials),
        text(stats.center, false), text(stats.center, true), text(stats.axis, false),
        text(stats.axis, true), stats.none, std::format("{:.0f}", median(stats.time)));
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    auto trials = std::size_t{40};
    auto seed = std::uint32_t{2025};
    auto model_path = std::string{RMCS_ENERGY_UNIT_MODEL};

    for (int i = 1; i < argc; i++) {
        const auto arg = std::string{argv[i]};
        if (arg == "--trials" && i + 1 < argc) {
            trials = std::strtoul(argv[++i], nullptr, 10);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            std::println(
                "Usage: mask_ablation [--trials 40] [--seed 2025] [--model <energy_unit.stl>]");
            return EXIT_FAILURE;
        }
    }
    if (trials == 0)
        return EXIT_FAILURE;

    auto mesh = load_binary_stl(model_path);
    if (!mesh.has_value()) {
        std::println(stderr, "{}", mesh.error());
        return EXIT_FAILURE;
    }
    auto renderer = DepthRenderer{};
    renderer.initialize(
        *mesh,
        DepthRenderer::Config{.width = kWidth, .height = kHeight, .intrinsics = kIntrinsics});
    auto registration = PointCloudRegistration{};
    const auto registration_config = PointCloudRegistration::Config{};
    if (auto initialized = registration.initialize(model_path, registration_config);
        !initialized.has_value()) {
        std::println(stderr, "{}", initialized.error());
        return EXIT_FAILURE;
    }
    const auto generator = Generator{*mesh};

    auto clutter_box = Stats{"(a) clutter, box only"};
    auto clutter_exact = Stats{"(b) clutter, box + exact mask"};
    auto clutter_degraded = Stats{"(c) clutter, box + YOLO-like"};
    auto control_box = Stats{"control, no clutter, box"};
    auto visible_fractions = std::vector<double>{};
    auto neighbor_counts = std::array<std::size_t, 3>{};
    auto occluded = std::size_t{};
    auto skipped = std::size_t{};

    std::println("mask_ablation: trials={} seed={}", trials, seed);
    for (std::size_t trial = 0; trial < trials; trial++) {
        // 每个试验独立播种，增减试验数不影响其他试验
        auto scene_engine = std::mt19937{seed * 1000003U + static_cast<std::uint32_t>(trial)};
        const auto noise_seed = scene_engine();

        auto scene = Generator::make_scene(scene_engine);
        const auto render = [&](const std::vector<DepthRenderer::Object>& objects) {
            auto noise_engine = std::mt19937{noise_seed};
            return renderer.render_scene(objects, noise_engine, scene.ground);
        };

        // 无遮挡的目标，作为对照帧，也用来计算可见比例
        const auto control_frame = render({{0, scene.target}});
        const auto control_mask = control_frame.instance == kTargetId;
        const auto full_pixels = cv::countNonZero(control_mask);

        auto frame = DepthRenderer::SceneFrame{};
        auto exact = cv::Mat{};
        auto visible = 0.;
        auto accepted = false;
        for (int attempt = 0; attempt < 20 && !accepted; attempt++) {
            scene.neighbors.clear();
            scene.neighbor_poses.clear();
            generator.add_neighbors(scene, scene_engine);

            auto objects = std::vector<DepthRenderer::Object>{{0, scene.target}};
            for (const auto& pose : scene.neighbor_poses)
                objects.push_back({0, pose});
            frame = render(objects);
            exact = frame.instance == kTargetId;
            visible = static_cast<double>(cv::countNonZero(exact)) / full_pixels;
            accepted = visible >= kMinVisibleFraction;
        }
        if (!accepted) {
            skipped++;
            std::println("#{:02} skipped: no valid clutter layout", trial);
            continue;
        }
        visible_fractions.push_back(visible);
        neighbor_counts[scene.neighbors.size()]++;
        if (visible < 0.98)
            occluded++;

        auto degrade_engine = std::mt19937{noise_seed ^ 0x5bd1e995U};
        const auto degraded = degrade_mask(exact, degrade_engine);

        const auto roi = make_roi(exact);
        const auto run = [&](const cv::Mat& depth, const cv::Rect2i& box, const cv::Mat& mask) {
            const auto begin = Clock::now();
            const auto cloud = mask.empty()
                                 ? PointCloudRegistration::depth_to_cloud(depth, kIntrinsics, box)
                                 : PointCloudRegistration::depth_to_cloud(
                                       depth, kIntrinsics, box, mask, registration_config);
            return run_registration(registration, *cloud, scene.target, begin);
        };
        const auto a = run(frame.depth, roi, {});
        const auto b = run(frame.depth, roi, exact);
        const auto c = run(frame.depth, roi, degraded);
        const auto control = run(control_frame.depth, make_roi(control_mask), {});
        clutter_box.add(a);
        clutter_exact.add(b);
        clutter_degraded.add(c);
        control_box.add(control);

        auto layout = std::string{};
        for (const auto& neighbor : scene.neighbors) {
            layout += std::format(
                " [{}{} gap {:.0f}mm]", neighbor.front ? "front " : "",
                neighbor.lying ? "lying" : "upright", neighbor.gap_mm);
        }
        std::println(
            "#{:02} vis={:.2f}{} | a {} | b {} | c {} | ctrl {}", trial, visible, layout,
            describe(a), describe(b), describe(c), describe(control));
    }

    const auto done = clutter_box.trials;
    std::println(
        "\ntrials: {} (skipped {}), seed {}, 1 neighbor: {}, 2 neighbors: {}, target partially "
        "occluded (visible < 98%): {}, median visible fraction {:.2f}",
        done, skipped, seed, neighbor_counts[1], neighbor_counts[2], occluded,
        visible_fractions.empty() ? 0. : median(visible_fractions));
    std::println(
        "success: center < {:.0f} mm and axis < {:.0f} deg; error columns in mm / deg over "
        "trials with a result; time in ms (cloud build + registration)\n",
        kCenterThresholdMm, kAxisThresholdDeg);
    std::println(
        "| {:<30} | {:<14} | {:>7} | {:>7} | {:>7} | {:>7} | {:>9} | {:>9} |", "variant", "success",
        "med ctr", "max ctr", "med ax", "max ax", "no result", "med time");
    std::println(
        "|{:-<32}|{:-<16}|{:-<9}|{:-<9}|{:-<9}|{:-<9}|{:-<11}|{:-<11}|", "", "", "", "", "", "", "",
        "");
    if (done == 0)
        return EXIT_FAILURE;
    print_row(clutter_box);
    print_row(clutter_exact);
    print_row(clutter_degraded);
    print_row(control_box);
    return EXIT_SUCCESS;
}
