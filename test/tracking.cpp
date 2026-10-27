#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"
#include "registration/depth_check.hpp"
#include "supervision/supervisor.hpp"
#include "synthetic/depth_renderer.hpp"
#include "tracking/tracker.hpp"

#if !defined(RMCS_ENERGY_UNIT_MESH) || !defined(RMCS_ENERGY_TRACE_ASSET_DIR)
# error "RMCS_ENERGY_UNIT_MESH and RMCS_ENERGY_TRACE_ASSET_DIR must be defined"
#endif

using rmcs_energy_trace::geometry::Camera;
using rmcs_energy_trace::geometry::load_binary_stl;
using rmcs_energy_trace::geometry::Mesh;
using rmcs_energy_trace::registration::DepthCheck;
using rmcs_energy_trace::supervision::Supervisor;
using rmcs_energy_trace::synthetic::DepthRenderer;
using rmcs_energy_trace::tracking::Tracker;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr Camera kCamera{.width = kWidth, .height = kHeight};

constexpr std::size_t kFrames = 150;
// 序列的运动上限：每帧平移 5 mm、旋转 2°
constexpr double kMaxStepMm = 5.;
constexpr double kMaxStepDeg = 2.;

// 合格标准
constexpr double kLossMm = 10.;
constexpr double kMedianCenterMm = 2.;
constexpr double kMedianAxisDeg = 1.;

constexpr double kRadToDeg = 180. / std::numbers::pi;

// 监督测试：能量单元从这一帧起离开画面（相机系 x 方向移开 1 m）
constexpr std::size_t kVanishAt = 75;
const Eigen::Vector3d kOutOfView{1., 0., 0.};

struct Sequence {
    std::vector<Eigen::Isometry3d> poses;       // 物体到相机
    std::vector<cv::Mat> color;
    std::vector<cv::Mat> depth;
};

struct Supervised {
    std::size_t trusted = 0, unchecked = 0, bad = 0;
    std::optional<std::size_t> reinitialize_at; // 要求重新初始化的帧
};

struct Stats {
    std::vector<double> center_mm, axis_deg, ms;
    std::size_t lost = 0;
    std::size_t failed = 0;
};

auto median(std::vector<double> values) -> double {
    if (values.empty())
        return 0.;
    std::ranges::sort(values);
    return values[values.size() / 2];
}

auto percentile(std::vector<double> values, double p) -> double {
    if (values.empty())
        return 0.;
    std::ranges::sort(values);
    return values[static_cast<std::size_t>(
        std::lround(p * static_cast<double>(values.size() - 1)))];
}

auto max_of(const std::vector<double>& values) -> double {
    return values.empty() ? 0. : *std::ranges::max_element(values);
}

auto axis_error_deg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) -> double {
    const auto cosine = std::clamp(a.linear().col(2).dot(b.linear().col(2)), -1., 1.);
    return std::acos(cosine) * kRadToDeg;
}

auto exp_rotation(const Eigen::Vector3d& w) -> Eigen::Matrix3d {
    const auto angle = w.norm();
    if (angle < 1e-12)
        return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd{angle, w / angle}.toRotationMatrix();
}

class TrackingTest : public testing::Test {
protected:
    static void SetUpTestSuite() {
        auto mesh = load_binary_stl(RMCS_ENERGY_UNIT_MESH);
        ASSERT_TRUE(mesh.has_value()) << mesh.error();
        mesh_ = std::make_unique<Mesh>(std::move(*mesh));

        // 物体到地面的距离：直立时 mesh 的最低点
        auto lowest = 0.f;
        for (const auto& triangle : mesh_->triangles) {
            for (const auto& vertex : triangle)
                lowest = std::min(lowest, vertex.z());
        }
        half_length_ = -static_cast<double>(lowest);

        renderer_ = std::make_unique<DepthRenderer>();
        auto config = DepthRenderer::Config{.camera = kCamera, .sample_count = 800'000};
        // 两端深色、中段浅色；地面和背景带平滑纹理，并有轻微噪声
        config.color = DepthRenderer::ColorConfig{
            .texture_amplitude = 14.f,
            .texture_cell_px = 20,
        };
        renderer_->initialize(*mesh_, config);
    }

    static void TearDownTestSuite() {
        renderer_.reset();
        mesh_.reset();
    }

    static auto models_available() -> bool {
        using namespace rmcs_energy_trace::tracking;
        const auto dir = std::filesystem::path{RMCS_ENERGY_TRACE_ASSET_DIR};
        return std::filesystem::exists(dir / kMeshFile)
            && std::filesystem::exists(dir / kRegionModelFile)
            && std::filesystem::exists(dir / kDepthModelFile);
    }

    static void expect_motion_within_limits(const Sequence& sequence) {
        auto max_step_mm = 0., max_step_deg = 0.;
        for (std::size_t i = 1; i < sequence.poses.size(); i++) {
            const auto& a = sequence.poses[i - 1];
            const auto& b = sequence.poses[i];
            max_step_mm = std::max(max_step_mm, (b.translation() - a.translation()).norm() * 1e3);
            const Eigen::Matrix3d delta = b.linear() * a.linear().transpose();
            max_step_deg = std::max(max_step_deg, Eigen::AngleAxisd{delta}.angle() * kRadToDeg);
        }
        std::println("  sequence peak step {:.2f} mm, {:.2f} deg", max_step_mm, max_step_deg);
        EXPECT_LE(max_step_mm, kMaxStepMm);
        EXPECT_LE(max_step_deg, kMaxStepDeg);
        for (const auto& pose : sequence.poses) {
            EXPECT_GE(pose.translation().z(), 0.5);
            EXPECT_LE(pose.translation().z(), 0.8);
        }
    }

    static void render_frames(
        Sequence& sequence, const std::optional<DepthRenderer::Plane>& fixed_plane,
        const std::vector<std::optional<DepthRenderer::Plane>>& planes, std::uint32_t seed) {
        // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
        auto engine = std::mt19937{seed};
        for (std::size_t i = 0; i < sequence.poses.size(); i++) {
            const auto plane = planes.empty() ? fixed_plane : planes[i];
            auto frame = renderer_->render(sequence.poses[i], engine, plane);
            sequence.color.push_back(frame.color);
            sequence.depth.push_back(frame.depth);
        }
    }

    // 直立在地面上，沿地面平移并绕竖直轴转动。相机朝向地面，俯视 45°。
    // 给出 vanish_at 时，从这一帧起能量单元离开画面
    static auto make_upright_sequence(std::optional<std::size_t> vanish_at = std::nullopt)
        -> Sequence {
        constexpr auto tilt = 45. * std::numbers::pi / 180.;
        Eigen::Matrix3d world_to_camera;
        const Eigen::Vector3d forward{std::cos(tilt), 0., -std::sin(tilt)};
        const Eigen::Vector3d right{0., -1., 0.};
        const Eigen::Vector3d down = forward.cross(right);
        world_to_camera.row(0) = right;
        world_to_camera.row(1) = down;
        world_to_camera.row(2) = forward;

        const Eigen::Vector3d up = world_to_camera * Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d east = world_to_camera * Eigen::Vector3d::UnitX();
        const Eigen::Vector3d north = world_to_camera * Eigen::Vector3d::UnitY();
        const Eigen::Vector3d origin{0.02, 0.0, 0.65};

        auto sequence = Sequence{};
        auto planes = std::vector<std::optional<DepthRenderer::Plane>>{};
        const auto plane_point = origin - half_length_ * up;
        for (std::size_t i = 0; i < kFrames; i++) {
            const auto phase = 2. * std::numbers::pi * static_cast<double>(i) / kFrames;
            // 一圈往返加一个小回环，位移和转角的峰值速度都在限值之内
            const auto x = 0.085 * std::sin(phase);
            const auto y = 0.035 * std::sin(2. * phase + 0.7);
            const auto yaw = 40. * std::sin(phase + 0.3) * std::numbers::pi / 180.;
            auto pose = Eigen::Isometry3d::Identity();
            pose.linear() = world_to_camera
                          * Eigen::AngleAxisd{yaw, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
            pose.translation() = origin + x * east + y * north;
            if (vanish_at.has_value() && i >= *vanish_at)
                pose.translation() += kOutOfView;
            sequence.poses.push_back(pose);
            planes.emplace_back(DepthRenderer::Plane{up, plane_point});
        }
        render_frames(sequence, std::nullopt, planes, 11);
        return sequence;
    }

    // 无支撑：任意姿态，旋转和平移都在变化，背景为远处墙面（无地面）。
    // 给出 vanish_at 时，从这一帧起能量单元离开画面
    static auto make_floating_sequence(std::optional<std::size_t> vanish_at = std::nullopt)
        -> Sequence {
        auto sequence = Sequence{};
        auto rotation =
            (Eigen::AngleAxisd{60. * std::numbers::pi / 180., Eigen::Vector3d::UnitX()}
             * Eigen::AngleAxisd{25. * std::numbers::pi / 180., Eigen::Vector3d::UnitZ()})
                .toRotationMatrix();
        for (std::size_t i = 0; i < kFrames; i++) {
            const auto phase = 2. * std::numbers::pi * static_cast<double>(i) / kFrames;
            const Eigen::Vector3d center{
                0.05 * std::sin(phase + 0.2),
                0.03 * std::sin(2. * phase),
                0.65 + 0.05 * std::sin(1.5 * phase + 1.0),
            };
            auto pose = Eigen::Isometry3d::Identity();
            pose.linear() = rotation;
            pose.translation() = center;
            if (vanish_at.has_value() && i >= *vanish_at)
                pose.translation() += kOutOfView;
            sequence.poses.push_back(pose);
            // 每帧的增量转动（相机系），模长不超过 1.1° * sqrt(3) < 2°
            const Eigen::Vector3d step =
                1.1 * std::numbers::pi / 180.
                * Eigen::Vector3d{
                    std::sin(phase + 0.5), std::sin(2. * phase + 1.3), std::sin(1.5 * phase + 2.1)};
            rotation = exp_rotation(step) * rotation;
        }
        // 背景墙：正对相机，距离 1.2 m
        const auto wall = DepthRenderer::Plane{Eigen::Vector3d::UnitZ(), {0., 0., 1.2}};
        render_frames(sequence, wall, {}, 23);
        return sequence;
    }

    static auto make_config(bool use_region, bool use_depth) -> Tracker::Config {
        auto config = Tracker::Config{};
        config.model_dir = RMCS_ENERGY_TRACE_ASSET_DIR;
        config.camera = kCamera;
        config.use_region = use_region;
        config.use_depth = use_depth;
        return config;
    }

    // 从 frame 0 的位姿 initial 起跟踪后续帧
    static auto
        run(const char* name, const Sequence& sequence, const Tracker::Config& config,
            const Eigen::Isometry3d& initial) -> Stats {
        auto tracker = Tracker{};
        auto initialized = tracker.initialize(config);
        EXPECT_TRUE(initialized.has_value()) << initialized.error();
        auto started = tracker.start(initial, sequence.color[0], sequence.depth[0]);
        EXPECT_TRUE(started.has_value()) << started.error();

        auto stats = Stats{};
        for (std::size_t i = 1; i < sequence.poses.size(); i++) {
            const auto result = tracker.track(sequence.color[i], sequence.depth[i]);
            if (!result.has_value()) {
                stats.failed++;
                continue;
            }
            const auto& truth = sequence.poses[i];
            const auto center = (result->pose.translation() - truth.translation()).norm() * 1e3;
            stats.center_mm.push_back(center);
            stats.axis_deg.push_back(axis_error_deg(result->pose, truth));
            stats.ms.push_back(result->ms);
            if (center > kLossMm)
                stats.lost++;
        }
        std::println(
            "[{}] center median/max {:.2f}/{:.2f} mm | axis median/max {:.2f}/{:.2f} deg | "
            "lost {} failed {} | ms median/p95 {:.2f}/{:.2f}",
            name, median(stats.center_mm), max_of(stats.center_mm), median(stats.axis_deg),
            max_of(stats.axis_deg), stats.lost, stats.failed, median(stats.ms),
            percentile(stats.ms, 0.95));
        return stats;
    }

    // 跟踪并逐帧交给监督。深度检查的 ROI 为 mesh 的投影框，不用掩码
    static auto run_supervised(
        const std::string& name, const Sequence& sequence, const Eigen::Isometry3d& initial)
        -> Supervised {
        auto tracker = Tracker{};
        const auto initialized = tracker.initialize(make_config(true, true));
        EXPECT_TRUE(initialized.has_value()) << initialized.error();
        const auto started = tracker.start(initial, sequence.color[0], sequence.depth[0]);
        EXPECT_TRUE(started.has_value()) << started.error();
        const auto check = DepthCheck{*mesh_, {}};
        auto supervisor = Supervisor{};
        supervisor.start(initial);

        auto stats = Supervised{};
        for (std::size_t i = 1; i < sequence.poses.size(); i++) {
            const auto result = tracker.track(sequence.color[i], sequence.depth[i]);
            auto observation = Supervisor::Observation{};
            if (result.has_value()) {
                observation.pose = result->pose;
                observation.depth = check.check(
                    sequence.depth[i], kCamera, check.projected_box(kCamera, result->pose), {},
                    result->pose);
            }
            const auto decision = supervisor.update(observation);
            switch (decision.health) {
            case Supervisor::Health::kGood: stats.trusted++; break;
            case Supervisor::Health::kBad: stats.bad++; break;
            case Supervisor::Health::kUnknown: stats.unchecked++; break;
            }
            if (decision.health != Supervisor::Health::kGood && result.has_value()) {
                const auto& truth = sequence.poses[i];
                std::println(
                    "[{}] frame {} {}: {} (center error {:.1f} mm, axis error {:.1f} deg)", name, i,
                    decision.health == Supervisor::Health::kBad ? "bad" : "unchecked",
                    decision.reason,
                    (result->pose.translation() - truth.translation()).norm() * 1e3,
                    axis_error_deg(result->pose, truth));
            }
            if (decision.reinitialize) {
                stats.reinitialize_at = i;
                std::println("[{}] reinitialize at frame {}: {}", name, i, decision.reason);
                break;
            }
        }
        std::println(
            "[{}] supervised: trusted {} bad {} unchecked {}", name, stats.trusted, stats.bad,
            stats.unchecked);
        return stats;
    }

    static void expect_tracked(const Stats& stats) {
        EXPECT_EQ(stats.failed, 0U);
        EXPECT_EQ(stats.lost, 0U) << "center error exceeded " << kLossMm << " mm";
        EXPECT_LT(median(stats.center_mm), kMedianCenterMm);
        EXPECT_LT(median(stats.axis_deg), kMedianAxisDeg);
    }

    // 模拟配准给出的初始位姿：绕自身长轴转 60°，再加 3 mm 平移和 2° 旋转（沿相机系 (1,1,1) 方向）。
    // 绕长轴的转角无法观测，配准结果可能偏任意角度；3 重对称下最大偏差为 60°
    static auto perturb(const Eigen::Isometry3d& pose) -> Eigen::Isometry3d {
        const Eigen::Vector3d direction = Eigen::Vector3d{1., 1., 1.}.normalized();
        auto result = pose;
        result.translation() += 0.003 * direction;
        result.linear() =
            Eigen::AngleAxisd{2. * std::numbers::pi / 180., direction}.toRotationMatrix()
            * pose.linear()
            * Eigen::AngleAxisd{60. * std::numbers::pi / 180., Eigen::Vector3d::UnitZ()}
                  .toRotationMatrix();
        return result;
    }

    static void run_case(const char* name, const Sequence& sequence) {
        expect_motion_within_limits(sequence);
        const auto& initial = sequence.poses[0];
        expect_tracked(
            run((std::string{name} + " region+depth exact").c_str(), sequence,
                make_config(true, true), initial));
        expect_tracked(
            run((std::string{name} + " region+depth perturbed").c_str(), sequence,
                make_config(true, true), perturb(initial)));
        // 单一线索只作参考：区域线索在合成数据上不稳定
        run((std::string{name} + " depth-only exact").c_str(), sequence, make_config(false, true),
            initial);
        run((std::string{name} + " region-only exact").c_str(), sequence, make_config(true, false),
            initial);

        // 监督：正常跟踪时不应要求重新初始化，绝大多数帧可信
        const auto frames = sequence.poses.size() - 1;
        for (const auto& [label, start] :
             {std::pair{" exact", initial}, std::pair{" perturbed", perturb(initial)}}) {
            const auto supervised = run_supervised(std::string{name} + label, sequence, start);
            EXPECT_FALSE(supervised.reinitialize_at.has_value()) << name << label;
            EXPECT_GE(supervised.trusted * 100, frames * 95) << name << label;
        }
    }

    static inline std::unique_ptr<Mesh> mesh_;
    static inline std::unique_ptr<DepthRenderer> renderer_;
    static inline double half_length_ = 0.075;
};

} // namespace

TEST_F(TrackingTest, UprightOnGroundPlane) {
    if (!models_available())
        GTEST_SKIP() << "M3T model files not found in " << RMCS_ENERGY_TRACE_ASSET_DIR
                     << ". Run scripts/download_assets.sh.";
    run_case("upright", make_upright_sequence());
}

TEST_F(TrackingTest, FreeFloating) {
    if (!models_available())
        GTEST_SKIP() << "M3T model files not found in " << RMCS_ENERGY_TRACE_ASSET_DIR
                     << ". Run scripts/download_assets.sh.";
    run_case("floating", make_floating_sequence());
}

// 能量单元在第 kVanishAt 帧离开画面，跟踪器还停在原处：监督要在几帧之内要求重新初始化
TEST_F(TrackingTest, SupervisorDetectsLoss) {
    if (!models_available())
        GTEST_SKIP() << "M3T model files not found in " << RMCS_ENERGY_TRACE_ASSET_DIR
                     << ". Run scripts/download_assets.sh.";
    const auto lost_frames = static_cast<std::size_t>(Supervisor::Config{}.bad_frames);
    for (const auto& [name, sequence] :
         {std::pair{"upright vanish", make_upright_sequence(kVanishAt)},
          std::pair{"floating vanish", make_floating_sequence(kVanishAt)}}) {
        const auto supervised = run_supervised(name, sequence, sequence.poses[0]);
        ASSERT_TRUE(supervised.reinitialize_at.has_value()) << name;
        EXPECT_GE(*supervised.reinitialize_at, kVanishAt) << name;
        EXPECT_LT(*supervised.reinitialize_at, kVanishAt + lost_frames) << name;
    }
}
