#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"
#include "registration/registration.hpp"
#include "segmentation/segmenter.hpp"
#include "synthetic/depth_renderer.hpp"

#ifndef RMCS_ENERGY_UNIT_MESH
# error "RMCS_ENERGY_UNIT_MESH must be defined"
#endif
#ifndef RMCS_ENERGY_TRACE_ASSET_DIR
# error "RMCS_ENERGY_TRACE_ASSET_DIR must be defined"
#endif

using rmcs_energy_trace::geometry::Camera;
using rmcs_energy_trace::geometry::load_binary_stl;
using rmcs_energy_trace::geometry::Mesh;
using rmcs_energy_trace::registration::Registration;
using rmcs_energy_trace::segmentation::Segmenter;
using rmcs_energy_trace::synthetic::DepthRenderer;

namespace {

constexpr Camera kCamera{.width = 640, .height = 480};
constexpr std::size_t kTrials = 20;
constexpr double kCenterThresholdMm = 3.;
constexpr double kAxisThresholdDeg = 2.;

// 资源区中心块（2026 规则手册图 4-22 的近似）：六棱柱，能量单元从 6 个侧面的中心沿径向插入，
// 封闭端朝外，单元中心距中心块中心 150 mm（外端在 Ø450 的圆上），内端约 20 mm 插在中心块里
constexpr double kHubCircumradius = 0.110;
constexpr double kHubThickness = 0.104;
constexpr double kUnitOffset = 0.150;
const auto kHubColor = cv::Vec3b{40, 40, 40};

const auto kSamEncoder =
    std::filesystem::path{RMCS_ENERGY_TRACE_ASSET_DIR} / "sam_vit_b_encoder.onnx";
const auto kSamDecoder =
    std::filesystem::path{RMCS_ENERGY_TRACE_ASSET_DIR} / "sam_vit_b_decoder.onnx";

auto uniform(std::mt19937& engine, double lo, double hi) -> double {
    return std::uniform_real_distribution<double>{lo, hi}(engine);
}

auto rotation_z(double angle) -> Eigen::Matrix3d {
    return Eigen::AngleAxisd{angle, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
}

// 六棱柱，轴沿 z，中心在原点；6 个侧面的中心在 x 轴方向每隔 60° 一个
auto make_hub() -> Mesh {
    auto mesh = Mesh{};
    const auto radius = static_cast<float>(kHubCircumradius);
    const auto half = static_cast<float>(kHubThickness / 2.);
    auto top = std::array<Eigen::Vector3f, 6>{};
    auto bottom = std::array<Eigen::Vector3f, 6>{};
    for (std::size_t k = 0; k < 6; k++) {
        const auto angle = static_cast<float>(
            std::numbers::pi / 6. + static_cast<double>(k) * std::numbers::pi / 3.);
        top[k] = {radius * std::cos(angle), radius * std::sin(angle), half};
        bottom[k] = {radius * std::cos(angle), radius * std::sin(angle), -half};
    }
    const Eigen::Vector3f top_center{0.f, 0.f, half}, bottom_center{0.f, 0.f, -half};
    for (std::size_t k = 0; k < 6; k++) {
        const auto next = (k + 1) % 6;
        mesh.triangles.push_back({top_center, top[k], top[next]});
        mesh.triangles.push_back({bottom_center, bottom[next], bottom[k]});
        mesh.triangles.push_back({bottom[k], bottom[next], top[next]});
        mesh.triangles.push_back({bottom[k], top[next], top[k]});
    }
    return mesh;
}

// 第 slot 个能量单元在中心块坐标系中的位姿：+z（封闭端）沿径向朝外
auto unit_in_hub(int slot, double roll) -> Eigen::Isometry3d {
    const auto azimuth = slot * std::numbers::pi / 3.;
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() =
        rotation_z(azimuth)
        * Eigen::AngleAxisd{std::numbers::pi / 2., Eigen::Vector3d::UnitY()}.toRotationMatrix()
        * rotation_z(roll);
    pose.translation() = kUnitOffset * Eigen::Vector3d{std::cos(azimuth), std::sin(azimuth), 0.};
    return pose;
}

struct Trial {
    Eigen::Isometry3d pose; // 目标能量单元，mesh -> camera
    cv::Mat color;
    cv::Mat depth;
    cv::Mat mask;           // 目标的精确可见掩码
    cv::Rect2f box;         // 目标可见像素的包围框，相当于理想的检测框
};

// 相机大致正对中心块（偏离法线 0~25°），距目标 0.5~0.8 m；随机去掉 0~2 根其他单元，
// 一半的场景在中心块后方有一面墙
auto make_trial(const DepthRenderer& renderer, std::size_t hub_mesh, std::mt19937& engine)
    -> Trial {
    const auto tilt = uniform(engine, 0., 25.) * std::numbers::pi / 180.;
    const auto heading = uniform(engine, 0., 2. * std::numbers::pi);
    const Eigen::Vector3d normal{
        std::sin(tilt) * std::cos(heading), std::sin(tilt) * std::sin(heading), -std::cos(tilt)};
    const Eigen::Vector3d helper =
        std::abs(normal.y()) < 0.9 ? Eigen::Vector3d::UnitY() : Eigen::Vector3d::UnitX();
    Eigen::Matrix3d hub_rotation;
    hub_rotation.col(0) = helper.cross(normal).normalized();
    hub_rotation.col(1) = normal.cross(hub_rotation.col(0));
    hub_rotation.col(2) = normal;
    hub_rotation = hub_rotation * rotation_z(uniform(engine, 0., 2. * std::numbers::pi));

    const auto target = std::uniform_int_distribution<int>{0, 5}(engine);
    const auto removed = std::uniform_int_distribution<int>{0, 2}(engine);
    auto present = std::array<bool, 6>{true, true, true, true, true, true};
    for (int i = 0; i < removed; i++) {
        const auto slot = std::uniform_int_distribution<int>{0, 5}(engine);
        if (slot != target)
            present[static_cast<std::size_t>(slot)] = false;
    }

    // 图像中心对准中心块与目标之间
    const auto distance = uniform(engine, 0.5, 0.8);
    const auto azimuth = target * std::numbers::pi / 3.;
    const Eigen::Vector3d aim =
        0.5 * kUnitOffset * Eigen::Vector3d{std::cos(azimuth), std::sin(azimuth), 0.};
    auto hub_pose = Eigen::Isometry3d::Identity();
    hub_pose.linear() = hub_rotation;
    hub_pose.translation() = Eigen::Vector3d{0., 0., distance} - hub_rotation * aim;

    // 目标放在第一个，实例编号为 1
    auto objects = std::vector<DepthRenderer::Object>{};
    const auto roll = [&] { return uniform(engine, 0., 2. * std::numbers::pi); };
    const Eigen::Isometry3d target_pose = hub_pose * unit_in_hub(target, roll());
    objects.push_back({0, target_pose});
    for (int slot = 0; slot < 6; slot++) {
        if (slot != target && present[static_cast<std::size_t>(slot)])
            objects.push_back({0, hub_pose * unit_in_hub(slot, roll())});
    }
    objects.push_back({hub_mesh, hub_pose});

    auto wall = std::optional<DepthRenderer::Plane>{};
    if (std::bernoulli_distribution{0.5}(engine)) {
        const auto behind = uniform(engine, 0.07, 0.15);
        wall = DepthRenderer::Plane{
            hub_rotation.col(2), hub_pose.translation() - behind * hub_rotation.col(2)};
    }

    const auto frame = renderer.render_scene(objects, engine, wall);
    const cv::Mat mask = frame.instance == 1;
    return Trial{
        .pose = target_pose,
        .color = frame.color,
        .depth = frame.depth,
        .mask = mask,
        .box = cv::Rect2f{cv::boundingRect(mask)},
    };
}

auto iou(const cv::Mat& a, const cv::Mat& b) -> double {
    auto both = cv::Mat{}, either = cv::Mat{};
    cv::bitwise_and(a, b, both);
    cv::bitwise_or(a, b, either);
    const auto total = cv::countNonZero(either);
    return total == 0 ? 1. : static_cast<double>(cv::countNonZero(both)) / total;
}

auto median(std::vector<double> values) -> double {
    if (values.empty())
        return 0.;
    std::ranges::sort(values);
    return values[values.size() / 2];
}

class SegmentationTest : public testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(kSamEncoder) || !std::filesystem::exists(kSamDecoder))
            return;
        auto mesh = load_binary_stl(RMCS_ENERGY_UNIT_MESH);
        ASSERT_TRUE(mesh.has_value()) << mesh.error();

        auto config = DepthRenderer::Config{.camera = kCamera, .sample_count = 800'000};
        config.color = DepthRenderer::ColorConfig{.texture_amplitude = 14.f, .texture_cell_px = 20};
        auto renderer = DepthRenderer{};
        renderer.initialize(*mesh, config);
        const auto hub = renderer.add_mesh(make_hub(), kHubColor);

        // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
        auto engine = std::mt19937{5150};
        for (std::size_t i = 0; i < kTrials; i++)
            trials_.push_back(make_trial(renderer, hub, engine));

        auto segmenter = Segmenter{};
        const auto initialized = segmenter.initialize(
            {.encoder_path = kSamEncoder.string(), .decoder_path = kSamDecoder.string()});
        ASSERT_TRUE(initialized.has_value()) << initialized.error();
        for (const auto& trial : trials_)
            results_.push_back(segmenter.segment(trial.color, trial.box));
        available_ = true;
    }

    static void TearDownTestSuite() {
        trials_.clear();
        results_.clear();
    }

    void SetUp() override {
        if (!available_)
            GTEST_SKIP() << "SAM networks not found in " << RMCS_ENERGY_TRACE_ASSET_DIR
                         << ". Run scripts/download_assets.sh.";
    }

    static inline std::vector<Trial> trials_;
    static inline std::vector<std::optional<Segmenter::Result>> results_;
    static inline bool available_ = false;
};

} // namespace

// SAM 以理想的检测框为提示，只分出目标这一根，不包括相邻单元和中心块
TEST_F(SegmentationTest, MaskCoversOnlyTheTarget) {
    auto ious = std::vector<double>{};
    auto encoder_ms = std::vector<double>{};
    for (std::size_t i = 0; i < trials_.size(); i++) {
        ASSERT_TRUE(results_[i].has_value()) << "trial " << i;
        const auto& result = *results_[i];
        ious.push_back(iou(result.mask, trials_[i].mask));
        encoder_ms.push_back(result.timing.encoder);
        std::println(
            "[sam] #{:02} IoU {:.3f} score {:.3f} mask {} px, truth {} px", i, ious.back(),
            result.score, cv::countNonZero(result.mask), cv::countNonZero(trials_[i].mask));
    }
    std::println(
        "[sam] IoU median {:.3f} min {:.3f} | encoder median {:.0f} ms (not deterministic)",
        median(ious), *std::ranges::min_element(ious), median(encoder_ms));
    EXPECT_GE(median(ious), 0.9);
    EXPECT_GE(*std::ranges::min_element(ious), 0.85);
}

// 中心块场景中，用 SAM 的掩码配准：检测框里有中心块和相邻单元，只用框时会被拒绝
TEST_F(SegmentationTest, RegistrationWithSamMask) {
    auto registration = Registration{};
    ASSERT_TRUE(registration.initialize(RMCS_ENERGY_UNIT_MESH).has_value());
    auto success = std::size_t{}, wrong = std::size_t{};
    for (std::size_t i = 0; i < trials_.size(); i++) {
        const auto& trial = trials_[i];
        const auto box = cv::Rect2i{trial.box};
        const auto result = registration.register_depth(
            trial.depth, kCamera, box, results_[i].has_value() ? results_[i]->mask : cv::Mat{});
        if (!result.has_value()) {
            std::println("[sam+registration] #{:02} rejected", i);
            continue;
        }
        const auto center = (result->pose.translation() - trial.pose.translation()).norm() * 1e3;
        const auto axis =
            std::acos(
                std::clamp(trial.pose.linear().col(2).dot(result->pose.linear().col(2)), -1., 1.))
            * 180. / std::numbers::pi;
        const auto correct = center < kCenterThresholdMm && axis < kAxisThresholdDeg;
        correct ? success++ : wrong++;
        std::println(
            "[sam+registration] #{:02} accepted {} center={:.2f}mm axis={:.2f}deg", i,
            correct ? "ok   " : "WRONG", center, axis);
    }
    std::println(
        "[sam+registration] summary: {} trials, success {}, accepted-wrong {}", trials_.size(),
        success, wrong);
    EXPECT_GE(static_cast<double>(success), 0.9 * static_cast<double>(trials_.size()));
    EXPECT_LE(wrong, 1U);
}
