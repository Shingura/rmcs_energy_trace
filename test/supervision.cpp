#include <numbers>
#include <optional>
#include <string_view>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>

#include "registration/depth_check.hpp"
#include "supervision/supervisor.hpp"

using rmcs_energy_trace::registration::DepthCheck;
using rmcs_energy_trace::supervision::Supervisor;

namespace {

using Health = Supervisor::Health;
using State = Supervisor::State;

auto make_pose(double x) -> Eigen::Isometry3d {
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d{x, 0., 0.6};
    return pose;
}

// 1000 个有效像素，按比例分成支持、空闲空间违例和遮挡
auto make_depth(double support, double free_space, double occluded) -> DepthCheck::Quality {
    return {
        .model_pixels = 1000,
        .valid_pixels = 1000,
        .support_ratio = support,
        .free_space_ratio = free_space,
        .occluded_ratio = occluded,
        .missing_ratio = 0.,
    };
}

auto good(double x = 0.) -> Supervisor::Observation {
    return {
        .pose = make_pose(x),
        .depth = make_depth(0.95, 0.02, 0.03),
        .detection_iou = {},
        .end_in_image_ratio = {}};
}

// mesh 有一半落在背景上
auto over_background() -> Supervisor::Observation {
    return {
        .pose = make_pose(0.),
        .depth = make_depth(0.5, 0.5, 0.),
        .detection_iou = {},
        .end_in_image_ratio = {}};
}

// 几乎全被挡住：未被遮挡的像素只有 50 个
auto occluded() -> Supervisor::Observation {
    return {
        .pose = make_pose(0.),
        .depth = make_depth(0.04, 0.01, 0.95),
        .detection_iou = {},
        .end_in_image_ratio = {}};
}

// 有一端在画面外：深度检查本身看起来很好
auto end_outside(double ratio = 0.5) -> Supervisor::Observation {
    auto observation = good();
    observation.end_in_image_ratio = ratio;
    return observation;
}

auto started() -> Supervisor {
    auto supervisor = Supervisor{};
    supervisor.start(make_pose(0.));
    return supervisor;
}

} // namespace

TEST(Supervisor, SearchingRequestsInitialization) {
    auto supervisor = Supervisor{};
    const auto decision = supervisor.update(good());
    EXPECT_EQ(decision.state, State::kSearching);
    EXPECT_TRUE(decision.reinitialize);
    EXPECT_FALSE(decision.trusted);
}

TEST(Supervisor, HealthyFramesAreTrusted) {
    auto supervisor = started();
    for (int i = 0; i < 10; i++) {
        const auto decision = supervisor.update(good(0.001 * i));
        EXPECT_EQ(decision.health, Health::kGood);
        EXPECT_TRUE(decision.trusted);
        EXPECT_FALSE(decision.reinitialize);
    }
}

// 遮挡不算失败：不可信，但不计入连续失败
TEST(Supervisor, OcclusionIsNotFailure) {
    auto supervisor = started();
    for (int i = 0; i < 10; i++) {
        const auto decision = supervisor.update(occluded());
        EXPECT_EQ(decision.health, Health::kUnknown);
        EXPECT_FALSE(decision.trusted);
        EXPECT_FALSE(decision.reinitialize);
    }
    EXPECT_TRUE(supervisor.update(good()).trusted);
}

TEST(Supervisor, ConsecutiveBadFramesRequestReinitialization) {
    auto supervisor = started();
    for (int i = 0; i < 4; i++)
        EXPECT_FALSE(supervisor.update(over_background()).reinitialize);
    // 中间出现一帧健康的，计数清零
    EXPECT_TRUE(supervisor.update(good()).trusted);
    for (int i = 0; i < 4; i++) {
        const auto decision = supervisor.update(over_background());
        EXPECT_EQ(decision.health, Health::kBad);
        EXPECT_FALSE(decision.reinitialize);
    }
    const auto decision = supervisor.update(over_background());
    EXPECT_TRUE(decision.reinitialize);
    EXPECT_EQ(decision.state, State::kSearching);
    EXPECT_EQ(supervisor.state(), State::kSearching);
}

TEST(Supervisor, LongUncheckedTrackingRequestsReinitialization) {
    auto supervisor = Supervisor{Supervisor::Config{.no_good_frames = 3}};
    supervisor.start(make_pose(0.));
    EXPECT_FALSE(supervisor.update(occluded()).reinitialize);
    EXPECT_FALSE(supervisor.update(occluded()).reinitialize);
    EXPECT_TRUE(supervisor.update(occluded()).reinitialize);
}

TEST(Supervisor, PoseJumpIsBad) {
    auto supervisor = started();
    EXPECT_EQ(supervisor.update(good(0.10)).health, Health::kBad);

    // 长轴转动超过 30° 也算跳变；绕长轴的转动不算
    supervisor.start(make_pose(0.));
    auto rolled = good();
    rolled.pose->linear() =
        Eigen::AngleAxisd{std::numbers::pi / 2., Eigen::Vector3d::UnitZ()}.toRotationMatrix();
    EXPECT_EQ(supervisor.update(rolled).health, Health::kGood);
    auto tilted = good();
    tilted.pose->linear() =
        Eigen::AngleAxisd{std::numbers::pi / 4., Eigen::Vector3d::UnitX()}.toRotationMatrix();
    EXPECT_EQ(supervisor.update(tilted).health, Health::kBad);
}

// 跳变后的位姿不能成为下一帧的比较基准：跟踪器停在新位置时，每一帧都仍是跳变
TEST(Supervisor, PoseJumpIsNotAcceptedAsReference) {
    auto supervisor = started();
    for (int i = 0; i < 4; i++) {
        const auto decision = supervisor.update(good(0.10));
        EXPECT_EQ(decision.health, Health::kBad);
        EXPECT_FALSE(decision.trusted);
        EXPECT_FALSE(decision.reinitialize);
    }
    const auto decision = supervisor.update(good(0.10));
    EXPECT_EQ(decision.health, Health::kBad);
    EXPECT_FALSE(decision.trusted);
    EXPECT_TRUE(decision.reinitialize);
}

// 无法判断的帧不清零不健康计数：4 帧不健康 + 1 帧无法判断 + 1 帧不健康，在第 6 帧重新初始化
TEST(Supervisor, UncheckedFrameDoesNotResetBadCount) {
    auto supervisor = started();
    for (int i = 0; i < 4; i++)
        EXPECT_FALSE(supervisor.update(over_background()).reinitialize);
    EXPECT_FALSE(supervisor.update(occluded()).reinitialize);
    EXPECT_TRUE(supervisor.update(over_background()).reinitialize);
}

// 不健康和无法判断交替、没有健康帧时，不健康帧数始终不到 5，但第 30 帧重新初始化
TEST(Supervisor, FramesWithoutGoodRequestReinitialization) {
    auto supervisor = started();
    for (int i = 0; i < 29; i++) {
        const auto decision = supervisor.update(i % 10 == 0 ? over_background() : occluded());
        EXPECT_FALSE(decision.reinitialize) << "frame " << i + 1;
    }
    EXPECT_TRUE(supervisor.update(occluded()).reinitialize);
}

TEST(Supervisor, DetectionMismatchIsBad) {
    auto supervisor = started();
    auto observation = good();
    observation.detection_iou = 0.6;
    EXPECT_EQ(supervisor.update(observation).health, Health::kGood);
    observation.detection_iou = 0.1;
    EXPECT_EQ(supervisor.update(observation).health, Health::kBad);
}

TEST(Supervisor, TrackerFailureIsBad) {
    auto supervisor = started();
    auto observation = good();
    observation.pose.reset();
    const auto decision = supervisor.update(observation);
    EXPECT_EQ(decision.health, Health::kBad);
    EXPECT_FALSE(decision.trusted);
}

// 有一端在画面外时无法判断：即使深度检查很好也不可信，也不计入不健康
TEST(Supervisor, EndOutsideImageIsUnknown) {
    auto supervisor = started();
    for (int i = 0; i < 10; i++) {
        const auto decision = supervisor.update(end_outside());
        EXPECT_EQ(decision.health, Health::kUnknown);
        EXPECT_FALSE(decision.trusted);
        EXPECT_FALSE(decision.reinitialize);
        EXPECT_NE(decision.reason.find("outside"), std::string_view::npos);
    }
    EXPECT_TRUE(supervisor.update(good()).trusted);
}

TEST(Supervisor, EndOutsideImageCountsAsNoGoodFrame) {
    auto supervisor = Supervisor{Supervisor::Config{.no_good_frames = 3}};
    supervisor.start(make_pose(0.));
    EXPECT_FALSE(supervisor.update(end_outside()).reinitialize);
    EXPECT_FALSE(supervisor.update(end_outside()).reinitialize);
    EXPECT_TRUE(supervisor.update(end_outside()).reinitialize);
}

// 两端画面内比例正常时不影响原有判断；不给比例时不检查
TEST(Supervisor, EndInImageRatioDoesNotAffectOtherRules) {
    auto supervisor = started();
    auto observation = good();
    observation.end_in_image_ratio = 0.9;
    EXPECT_EQ(supervisor.update(observation).health, Health::kGood);
    observation.end_in_image_ratio = 0.8;
    EXPECT_EQ(supervisor.update(observation).health, Health::kGood);
    observation.depth = make_depth(0.5, 0.5, 0.);
    EXPECT_EQ(supervisor.update(observation).health, Health::kBad);
    observation.depth = make_depth(0.04, 0.01, 0.95);
    EXPECT_EQ(supervisor.update(observation).health, Health::kUnknown);
    // 位姿跳变在画面内比例之前判断
    observation = end_outside();
    observation.pose = make_pose(0.10);
    EXPECT_EQ(supervisor.update(observation).health, Health::kBad);
}
