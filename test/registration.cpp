#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"
#include "registration/registration.hpp"
#include "synthetic/depth_renderer.hpp"

#ifndef RMCS_ENERGY_UNIT_MESH
# error "RMCS_ENERGY_UNIT_MESH must be defined"
#endif

using rmcs_energy_trace::geometry::Camera;
using rmcs_energy_trace::geometry::load_binary_stl;
using rmcs_energy_trace::geometry::Mesh;
using rmcs_energy_trace::registration::Registration;
using rmcs_energy_trace::synthetic::DepthRenderer;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr Camera kCamera{.width = kWidth, .height = kHeight};

constexpr std::size_t kTrials = 50;
// 抓取只依赖中心和长轴：绕长轴的滚转对称，两端认反则长轴夹角接近 180°
constexpr double kCenterThresholdMm = 3.;
constexpr double kAxisThresholdDeg = 2.;

// 正例：成功率下限，以及被接受的结果里错误结果的数量上限
constexpr double kMinSuccessRate = 0.9;
constexpr std::size_t kMaxWrongAccepts = 1;
// 反例的试验数少一些，以控制测试时间
constexpr std::size_t kNegativeTrials = 30;
// 反例（错误物体）被接受的数量上限
constexpr std::size_t kMaxNegativeAccepts = 0;
// 带掩码的两组测试的试验数
constexpr std::size_t kMaskTrials = 20;
// 模型 z 轴指向封闭端。z 轴与“相机指向物体”方向夹角的余弦大于该值时，开口端正对相机 60° 以内
constexpr double kMaxOpenEndFacing = 0.5;
// 部分视野：ROI 只保留物体包围盒宽度的这个比例
constexpr double kPartialWidthFraction = 0.6;

struct Trial {
    Eigen::Isometry3d pose;
    cv::Mat depth;
    cv::Rect2i roi;
    cv::Mat mask; // 精确的实例掩码；只在带掩码的测试中使用
};

auto median(std::vector<double> values) -> double {
    if (values.empty())
        return 0.;
    std::ranges::sort(values);
    return values[values.size() / 2];
}

auto random_rotation(std::mt19937& engine) -> Eigen::Matrix3d {
    // 均匀随机四元数
    auto normal = std::normal_distribution<double>{};
    auto q = Eigen::Quaterniond{normal(engine), normal(engine), normal(engine), normal(engine)};
    return q.normalized().toRotationMatrix();
}

auto uniform(std::mt19937& engine, double lo, double hi) -> double {
    return std::uniform_real_distribution<double>{lo, hi}(engine);
}

// 沿 z 轴的半长（mesh 已居中），即直立时中心到地面的距离
auto half_length(const Mesh& mesh) -> double {
    auto upper = 0.f;
    for (const auto& triangle : mesh.triangles) {
        for (const auto& vertex : triangle)
            upper = std::max(upper, vertex.z());
    }
    return upper;
}

auto scaled(const Mesh& mesh, float factor) -> Mesh {
    auto result = mesh;
    for (auto& triangle : result.triangles) {
        for (auto& vertex : triangle)
            vertex *= factor;
    }
    return result;
}

// 与能量单元不同的实心圆柱，轴沿 z，中心在原点
auto make_cylinder(float radius, float length) -> Mesh {
    constexpr std::size_t kSegments = 96;
    auto mesh = Mesh{};
    const auto at = [&](std::size_t i, float z) {
        const auto angle = 2.f * std::numbers::pi_v<float>
                         * static_cast<float>(i % kSegments) / static_cast<float>(kSegments);
        return Eigen::Vector3f{radius * std::cos(angle), radius * std::sin(angle), z};
    };
    const Eigen::Vector3f bottom{0.f, 0.f, -length / 2}, top{0.f, 0.f, length / 2};
    for (std::size_t i = 0; i < kSegments; i++) {
        const auto lo0 = at(i, -length / 2), lo1 = at(i + 1, -length / 2);
        const auto hi0 = at(i, length / 2), hi1 = at(i + 1, length / 2);
        mesh.triangles.push_back({lo0, lo1, hi1});
        mesh.triangles.push_back({lo0, hi1, hi0});
        mesh.triangles.push_back({top, hi0, hi1});
        mesh.triangles.push_back({bottom, lo1, lo0});
    }
    return mesh;
}

auto make_renderer(const Mesh& mesh) -> std::unique_ptr<DepthRenderer> {
    auto renderer = std::make_unique<DepthRenderer>();
    renderer->initialize(mesh, DepthRenderer::Config{.camera = kCamera});
    return renderer;
}

// 场景 A：随机朝向，无背景。排除开口端正对相机 60° 以内的朝向：取矿时开口端套在柱子上，
// 相机看不到它，模型里也没有孔壁
auto make_trials_a(const DepthRenderer& renderer, std::uint32_t seed, std::size_t count)
    -> std::vector<Trial> {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{seed};
    auto trials = std::vector<Trial>{};
    for (std::size_t i = 0; i < count; i++) {
        const auto distance = uniform(engine, 0.4, 0.8);
        // 物体中心在光轴附近，保证完整位于视野内
        const Eigen::Vector3d center{
            uniform(engine, -0.08, 0.08) * distance / 0.6,
            uniform(engine, -0.06, 0.06) * distance / 0.6,
            distance,
        };
        auto pose = Eigen::Isometry3d::Identity();
        do {
            pose.linear() = random_rotation(engine);
        } while (pose.linear().col(2).dot(center.normalized()) > kMaxOpenEndFacing);
        pose.translation() = center;
        const auto frame = renderer.render(pose, engine, std::nullopt);
        trials.push_back({pose, frame.depth, frame.roi, frame.mask});
    }
    return trials;
}

// 场景 B：直立在地面上，相机俯视 30~60 度。ground_offset 为中心到地面的距离
auto make_trials_b(
    const DepthRenderer& renderer, std::uint32_t seed, std::size_t count, double ground_offset)
    -> std::vector<Trial> {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{seed};
    auto trials = std::vector<Trial>{};
    for (std::size_t i = 0; i < count; i++) {
        const auto distance = uniform(engine, 0.4, 0.8);
        const auto tilt = uniform(engine, 30., 60.) * std::numbers::pi / 180.;
        const auto yaw = uniform(engine, 0., 2. * std::numbers::pi);

        // 世界系 z 向上；相机朝 +x 并向下俯视 tilt
        Eigen::Matrix3d world_to_camera;
        const Eigen::Vector3d forward{std::cos(tilt), 0., -std::sin(tilt)};
        const Eigen::Vector3d right{0., -1., 0.};
        const Eigen::Vector3d down = forward.cross(right);
        world_to_camera.row(0) = right;
        world_to_camera.row(1) = down;
        world_to_camera.row(2) = forward;

        const Eigen::Vector3d center{
            uniform(engine, -0.05, 0.05) * distance / 0.6,
            uniform(engine, -0.04, 0.04) * distance / 0.6,
            distance,
        };
        auto pose = Eigen::Isometry3d::Identity();
        pose.linear() =
            world_to_camera * Eigen::AngleAxisd{yaw, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
        pose.translation() = center;

        // 模型 z 轴为竖直方向
        const Eigen::Vector3d up = world_to_camera * Eigen::Vector3d::UnitZ();
        const auto frame =
            renderer.render(pose, engine, DepthRenderer::Plane{up, center - ground_offset * up});
        trials.push_back({pose, frame.depth, frame.roi, frame.mask});
    }
    return trials;
}

// 只保留 ROI 宽度的一部分（左或右），模拟检测框只框住物体的一部分
auto cut_roi(std::vector<Trial> trials, std::uint32_t seed) -> std::vector<Trial> {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{seed};
    for (auto& trial : trials) {
        const auto width = static_cast<int>(trial.roi.width * kPartialWidthFraction);
        const auto left = std::bernoulli_distribution{0.5}(engine);
        if (!left)
            trial.roi.x += trial.roi.width - width;
        trial.roi.width = width;
    }
    return trials;
}

// 在 B 场景的目标前方 10 cm 处放一根短细杆（半径 12 mm，长 5 cm），横向偏 2 cm，
// 挡住目标约三分之一的宽度。细杆的轴与目标长轴平行，位于相机到目标中心的视线上，
// 所以遮挡集中在目标中段（mesh 坐标 |z| < 半长 / 2，最远约 ±3.3 cm），不碰到两端。
// 不能挡住封闭端：封闭端被挡时配准的验收条件会拒绝（轴向位置只靠封闭端约束），这是有意的。
// 掩码只含目标的可见像素；检测框取目标单独渲染时的框。地面按 B 场景的约定由位姿推出
auto occlude(
    const DepthRenderer& renderer, std::size_t rod, std::vector<Trial> trials, double ground_offset,
    std::uint32_t seed) -> std::vector<Trial> {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{seed};
    for (auto& trial : trials) {
        const Eigen::Vector3d center = trial.pose.translation();
        auto rod_pose = Eigen::Isometry3d::Identity();
        rod_pose.linear() = trial.pose.linear();
        rod_pose.translation() =
            center * (1. - 0.10 / center.norm()) + Eigen::Vector3d{0.02, 0., 0.};
        const Eigen::Vector3d up = trial.pose.linear().col(2);
        const auto scene = renderer.render_scene(
            {DepthRenderer::Object{0, trial.pose}, DepthRenderer::Object{rod, rod_pose}}, engine,
            DepthRenderer::Plane{up, center - ground_offset * up});
        trial.depth = scene.depth;
        trial.mask = scene.instance == 1;
    }
    return trials;
}

struct Counts {
    std::size_t trials = 0;
    std::size_t success = 0;        // 被接受且中心、长轴都达标
    std::size_t rejected = 0;       // 返回空
    std::size_t accepted_wrong = 0; // 被接受但中心或长轴不达标
};

// 逐个配准并打印。每行只含由输入决定的内容，耗时单独成行，便于比较不同运行顺序的输出。
// with_mask 为 true 时同时传入精确的实例掩码
auto run_trials(
    const Registration& registration, const char* name, const std::vector<Trial>& trials,
    bool with_mask = false) -> Counts {
    auto counts = Counts{};
    auto total_ms = std::vector<double>{}, quality_ms = std::vector<double>{};
    auto timing_lines = std::string{};

    for (std::size_t i = 0; i < trials.size(); i++) {
        const auto& trial = trials[i];
        counts.trials++;
        const auto mask = with_mask ? trial.mask : cv::Mat{};
        const auto result = registration.register_depth(trial.depth, kCamera, trial.roi, mask);
        if (!result.has_value()) {
            counts.rejected++;
            std::println("[{}] #{:02} rejected", name, i);
            continue;
        }
        const auto trans = (trial.pose.translation() - result->pose.translation()).norm() * 1e3;
        // 长轴（模型 z 轴）方向误差，接近 180° 表示两端认反
        const auto axis_cos =
            std::clamp(trial.pose.linear().col(2).dot(result->pose.linear().col(2)), -1., 1.);
        const auto axis = std::acos(axis_cos) * 180. / std::numbers::pi;
        const auto correct = trans < kCenterThresholdMm && axis < kAxisThresholdDeg;
        correct ? counts.success++ : counts.accepted_wrong++;
        const auto& quality = result->quality;
        std::println(
            "[{}] #{:02} accepted {} trans={:.2f}mm axis={:.1f}deg inlier={:.3f} support={:.3f} "
            "free={:.3f} occluded={:.3f} missing={:.3f} pixels={}",
            name, i, correct ? "ok   " : "WRONG", trans, axis, result->inlier_ratio,
            quality.support_ratio, quality.free_space_ratio, quality.occluded_ratio,
            quality.missing_ratio, quality.model_pixels);
        total_ms.push_back(result->timing.total_ms);
        quality_ms.push_back(result->timing.quality_ms);
    }
    std::println(
        "[{}] summary: {} trials, success {}, rejected {}, accepted-wrong {} (success: center < "
        "{:.0f}mm and axis < {:.0f}deg)",
        name, counts.trials, counts.success, counts.rejected, counts.accepted_wrong,
        kCenterThresholdMm, kAxisThresholdDeg);
    std::println(
        "[{}] timing (not part of the deterministic output): median total {:.0f}ms, median "
        "quality check {:.1f}ms",
        name, median(total_ms), median(quality_ms));
    return counts;
}

auto run_negative_trials(
    const Registration& registration, const char* name, const std::vector<Trial>& trials,
    bool with_mask = false) -> std::size_t {
    auto accepted = std::size_t{};
    auto total_ms = std::vector<double>{};
    for (std::size_t i = 0; i < trials.size(); i++) {
        const auto mask = with_mask ? trials[i].mask : cv::Mat{};
        const auto result =
            registration.register_depth(trials[i].depth, kCamera, trials[i].roi, mask);
        if (!result.has_value()) {
            std::println("[{}] #{:02} rejected", name, i);
            continue;
        }
        accepted++;
        const auto trans = (trials[i].pose.translation() - result->pose.translation()).norm() * 1e3;
        const auto& quality = result->quality;
        std::println(
            "[{}] #{:02} ACCEPTED trans={:.2f}mm inlier={:.3f} support={:.3f} free={:.3f} "
            "occluded={:.3f} missing={:.3f} pixels={}",
            name, i, trans, result->inlier_ratio, quality.support_ratio, quality.free_space_ratio,
            quality.occluded_ratio, quality.missing_ratio, quality.model_pixels);
        total_ms.push_back(result->timing.total_ms);
    }
    std::println(
        "[{}] summary: {} trials, rejected {}, wrongly accepted {}", name, trials.size(),
        trials.size() - accepted, accepted);
    return accepted;
}

class RegistrationTest : public testing::Test {
protected:
    static void SetUpTestSuite() {
        auto mesh = load_binary_stl(RMCS_ENERGY_UNIT_MESH);
        ASSERT_TRUE(mesh.has_value()) << mesh.error();
        mesh_ = std::make_unique<Mesh>(std::move(*mesh));
        renderer_ = make_renderer(*mesh_);

        registration_ = std::make_unique<Registration>();
        ASSERT_TRUE(registration_->initialize(RMCS_ENERGY_UNIT_MESH).has_value());
    }

    static void expect_positive(const char* name, const Counts& counts) {
        EXPECT_GE(
            static_cast<double>(counts.success),
            kMinSuccessRate * static_cast<double>(counts.trials))
            << name;
        EXPECT_LE(counts.accepted_wrong, kMaxWrongAccepts) << name;
    }

    static inline std::unique_ptr<Mesh> mesh_;
    static inline std::unique_ptr<DepthRenderer> renderer_;
    static inline std::unique_ptr<Registration> registration_;
};

} // namespace

TEST_F(RegistrationTest, RandomOrientationNoBackground) {
    const auto trials = make_trials_a(*renderer_, 2024, kTrials);
    expect_positive("A", run_trials(*registration_, "A", trials));
}

TEST_F(RegistrationTest, UprightOnGroundPlane) {
    const auto trials = make_trials_b(*renderer_, 4048, kTrials, half_length(*mesh_));
    expect_positive("B", run_trials(*registration_, "B", trials));
}

// 反例：场景里是另一个更小的物体，注册的仍是真实能量单元，必须被拒绝
TEST_F(RegistrationTest, RejectsScaledMesh) {
    const auto small = scaled(*mesh_, 0.8f);
    const auto renderer = make_renderer(small);
    auto trials = make_trials_a(*renderer, 6001, kNegativeTrials / 2);
    const auto upright = make_trials_b(*renderer, 6002, kNegativeTrials / 2, half_length(small));
    trials.insert(trials.end(), upright.begin(), upright.end());
    EXPECT_LE(run_negative_trials(*registration_, "scaled0.8", trials), kMaxNegativeAccepts);
}

TEST_F(RegistrationTest, RejectsCylinder) {
    const auto cylinder = make_cylinder(0.04f, 0.12f);
    const auto renderer = make_renderer(cylinder);
    auto trials = make_trials_a(*renderer, 7001, kNegativeTrials / 2);
    const auto upright = make_trials_b(*renderer, 7002, kNegativeTrials / 2, half_length(cylinder));
    trials.insert(trials.end(), upright.begin(), upright.end());
    EXPECT_LE(run_negative_trials(*registration_, "cylinder", trials), kMaxNegativeAccepts);
}

// 带掩码：目标中段被前方的细杆挡住约三分之一宽度（不碰到封闭端，见 occlude）。
// 挡住目标的像素在掩码外，不应影响验收
TEST_F(RegistrationTest, OccludedWithMask) {
    auto renderer = make_renderer(*mesh_);
    const auto rod = renderer->add_mesh(make_cylinder(0.012f, 0.05f));
    const auto offset = half_length(*mesh_);
    const auto trials =
        occlude(*renderer, rod, make_trials_b(*renderer, 9001, kMaskTrials, offset), offset, 9101);
    expect_positive("occluded", run_trials(*registration_, "occluded", trials, true));
}

// 0.9 倍的物体只用检测框时偶尔会被接受：视线接近长轴时，把 mesh 沿长轴往后挪几毫米，
// 大部分像素仍在深度容差内。加上精确掩码后，被接受的次数不能比只用检测框时多：
// mesh 超出掩码、落在背景上的部分必须作为违例统计
TEST_F(RegistrationTest, MaskDoesNotWeakenRejection) {
    const auto small = scaled(*mesh_, 0.9f);
    const auto renderer = make_renderer(small);
    auto trials = make_trials_a(*renderer, 6001, kMaskTrials / 2);
    const auto upright = make_trials_b(*renderer, 6002, kMaskTrials / 2, half_length(small));
    trials.insert(trials.end(), upright.begin(), upright.end());
    const auto box_only = run_negative_trials(*registration_, "scaled0.9", trials);
    const auto masked = run_negative_trials(*registration_, "scaled0.9+mask", trials, true);
    EXPECT_LE(masked, box_only);
}

// 部分视野：ROI 只框住物体宽度的 60%。结果要么正确，要么被拒绝，不能被接受而错误
TEST_F(RegistrationTest, PartialViewIsCorrectOrRejected) {
    auto trials = cut_roi(make_trials_a(*renderer_, 8001, kTrials / 2), 8101);
    const auto upright =
        cut_roi(make_trials_b(*renderer_, 8002, kTrials / 2, half_length(*mesh_)), 8102);
    trials.insert(trials.end(), upright.begin(), upright.end());
    const auto counts = run_trials(*registration_, "partial", trials);
    EXPECT_LE(counts.accepted_wrong, kMaxWrongAccepts);
}
