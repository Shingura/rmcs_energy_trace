#include <numbers>
#include <optional>
#include <string_view>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>

#include "pipeline/reference_monitor.hpp"
#include "registration/depth_check.hpp"
#include "supervision/supervisor.hpp"

using rmcs_energy_trace::pipeline::Pipeline;
using rmcs_energy_trace::pipeline::ReferenceMonitor;
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
// 大部分被挡住、但未被遮挡的像素数仍够多（高分辨率相机）：按比例判为无法判断，即使剩下的像素看起来健康
TEST(Supervisor, MostlyOccludedIsUnknown) {
    auto supervisor = started();
    auto observation = good();
    observation.depth = make_depth(0.2, 0., 0.8);
    observation.depth.valid_pixels = 20000; // 未被遮挡 4000 个，远多于像素数下限
    const auto decision = supervisor.update(observation);
    EXPECT_EQ(decision.health, Health::kUnknown);
    EXPECT_FALSE(decision.trusted);

    // 比例刚好不低于下限：照常判断
    observation.depth = make_depth(0.25, 0., 0.75);
    observation.depth.valid_pixels = 20000;
    EXPECT_EQ(supervisor.update(observation).health, Health::kGood);
}

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

namespace {

using Monitor = ReferenceMonitor;

// 底座系下的目标位姿：中心在 (x, 0, 0.6)，长轴绕 y 轴偏离 z 轴 tilt_deg
auto make_base_pose(double x, double tilt_deg = 0.) -> Eigen::Isometry3d {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() = Eigen::AngleAxisd{tilt_deg * std::numbers::pi / 180., Eigen::Vector3d::UnitY()}
                        .toRotationMatrix();
    pose.translation() = Eigen::Vector3d{x, 0., 0.6};
    return pose;
}

// 相机在底座系下的位姿：位置 (0.1, -0.5, 0.3)，朝向任意，再绕 z 轴转 yaw_deg
auto make_camera(double yaw_deg = 0.) -> Eigen::Isometry3d {
    auto camera = Eigen::Isometry3d::Identity();
    camera.linear() =
        (Eigen::AngleAxisd{yaw_deg * std::numbers::pi / 180., Eigen::Vector3d::UnitZ()}
         * Eigen::AngleAxisd{0.7, Eigen::Vector3d{1., 2., 3.}.normalized()})
            .toRotationMatrix();
    camera.translation() =
        Eigen::Vector3d{0.1, -0.5, 0.3} + Eigen::Vector3d{yaw_deg, 0., 0.} * 1e-2;
    return camera;
}

// 一帧流程输出：base_pose 为底座系下的目标位姿，按相机位姿换到相机系
auto make_result(
    const Eigen::Isometry3d& base_pose, const Eigen::Isometry3d& camera, bool trusted, bool stable)
    -> Pipeline::Result {
    auto result = Pipeline::Result{};
    result.state = Pipeline::State::kTracking;
    result.pose = camera.inverse() * base_pose;
    result.trusted = trusted;
    result.stable = stable;
    return result;
}

auto established() -> Monitor {
    auto monitor = Monitor{};
    monitor.update(make_result(make_base_pose(0.), make_camera(), true, true), make_camera());
    return monitor;
}

// 用默认相机喂一帧
auto feed(
    Monitor& monitor, const Eigen::Isometry3d& base_pose, bool trusted = true, bool grasped = false)
    -> Monitor::Output {
    return monitor.update(
        make_result(base_pose, make_camera(), trusted, true), make_camera(), grasped);
}

} // namespace

// 第一次 stable 时建立参考，参考是底座系下的位姿
TEST(ReferenceMonitor, EstablishesReferenceOnStable) {
    auto monitor = Monitor{};
    const auto camera = make_camera();
    // 可信但还不稳定：不建立参考
    auto output = monitor.update(make_result(make_base_pose(0.), camera, true, false), camera);
    EXPECT_EQ(output.state, Monitor::State::kNoReference);
    EXPECT_FALSE(output.reference.has_value());
    EXPECT_FALSE(output.displacement.has_value());

    output = monitor.update(make_result(make_base_pose(0.), camera, true, true), camera);
    EXPECT_EQ(output.state, Monitor::State::kEstablished);
    ASSERT_TRUE(output.reference.has_value());
    EXPECT_TRUE(output.reference->isApprox(make_base_pose(0.), 1e-9));
    EXPECT_FALSE(output.comparison.has_value());
    EXPECT_EQ(output.inconsistent_count, 0);
    ASSERT_TRUE(output.displacement.has_value());
    EXPECT_NEAR(output.displacement->total, 0., 1e-12);
    EXPECT_EQ(monitor.state(), Monitor::State::kEstablished);
}

// stable 但不可信的帧（不会出现在流程里，但监视器不应据此建立参考）
TEST(ReferenceMonitor, UntrustedFrameDoesNotEstablishReference) {
    auto monitor = Monitor{};
    const auto camera = make_camera();
    const auto output =
        monitor.update(make_result(make_base_pose(0.), camera, false, true), camera);
    EXPECT_EQ(output.state, Monitor::State::kNoReference);
    EXPECT_EQ(monitor.state(), Monitor::State::kNoReference);
}

// 一致的帧：给出比较结果，计数为 0；阈值是闭区间
TEST(ReferenceMonitor, ConsistentFrame) {
    auto monitor = established();
    auto output = feed(monitor, make_base_pose(0.004, 2.));
    EXPECT_EQ(output.state, Monitor::State::kEstablished);
    ASSERT_TRUE(output.comparison.has_value());
    EXPECT_TRUE(output.comparison->consistent);
    EXPECT_NEAR(output.comparison->center_distance, 0.004, 1e-9);
    EXPECT_NEAR(output.comparison->axis_deg, 2., 1e-9);
    EXPECT_EQ(output.inconsistent_count, 0);
    ASSERT_TRUE(output.displacement.has_value());
    EXPECT_NEAR(output.displacement->total, 0.004, 1e-9);

    // 刚好在阈值内
    output = feed(monitor, make_base_pose(0.0049, 5.9));
    EXPECT_TRUE(output.comparison->consistent);
}

// 中心差或轴线夹角任一超限即不一致
TEST(ReferenceMonitor, InconsistentByCenterOrAxis) {
    auto monitor = established();
    auto output = feed(monitor, make_base_pose(0.006));
    ASSERT_TRUE(output.comparison.has_value());
    EXPECT_FALSE(output.comparison->consistent);
    EXPECT_EQ(output.inconsistent_count, 1);

    output = feed(monitor, make_base_pose(0., 6.5));
    EXPECT_FALSE(output.comparison->consistent);
    EXPECT_NEAR(output.comparison->axis_deg, 6.5, 1e-9);
    EXPECT_EQ(output.inconsistent_count, 2);
    EXPECT_EQ(output.state, Monitor::State::kEstablished);
}

// 不可信的帧不参与比较，不计入也不清零计数，即使位姿差得很远
TEST(ReferenceMonitor, UntrustedFramesDoNotAffectCount) {
    auto monitor = established();
    EXPECT_EQ(feed(monitor, make_base_pose(0.05)).inconsistent_count, 1);
    EXPECT_EQ(feed(monitor, make_base_pose(0.05)).inconsistent_count, 2);
    for (int i = 0; i < 20; i++) {
        // 不可信的帧：位姿远离参考，或恰好与参考一致，都不改变计数
        const auto output = feed(monitor, make_base_pose(i % 2 == 0 ? 0.3 : 0.), false);
        EXPECT_EQ(output.state, Monitor::State::kEstablished);
        EXPECT_FALSE(output.comparison.has_value());
        EXPECT_FALSE(output.displacement.has_value());
        EXPECT_EQ(output.inconsistent_count, 2);
        EXPECT_TRUE(output.reference.has_value());
    }
    // 不可信的帧不拆开“连续”：再来 3 个不一致的可信帧就判为被移动
    EXPECT_EQ(feed(monitor, make_base_pose(0.05)).inconsistent_count, 3);
    EXPECT_EQ(feed(monitor, make_base_pose(0.05)).inconsistent_count, 4);
    EXPECT_EQ(feed(monitor, make_base_pose(0.05)).state, Monitor::State::kMoved);
}

// 连续 5 个可信帧不一致：判为被移动，参考被清除
TEST(ReferenceMonitor, MovedAfterConsecutiveInconsistentFrames) {
    auto monitor = established();
    for (int i = 1; i < 5; i++) {
        const auto output = feed(monitor, make_base_pose(0.02));
        EXPECT_EQ(output.state, Monitor::State::kEstablished);
        EXPECT_EQ(output.inconsistent_count, i);
    }
    const auto output = feed(monitor, make_base_pose(0.02));
    EXPECT_EQ(output.state, Monitor::State::kMoved);
    EXPECT_EQ(output.inconsistent_count, 5);
    EXPECT_FALSE(output.reference.has_value());
    ASSERT_TRUE(output.comparison.has_value());
    EXPECT_FALSE(output.comparison->consistent);
    EXPECT_EQ(monitor.state(), Monitor::State::kMoved);
}

// kMoved 保持到 reset()：期间不建立参考、不比较，输出仍是判为被移动的那次比较和计数
TEST(ReferenceMonitor, MovedIsHeldUntilReset) {
    auto monitor = established();
    for (int i = 0; i < 5; i++)
        feed(monitor, make_base_pose(0.02));
    ASSERT_EQ(monitor.state(), Monitor::State::kMoved);

    // 可信、稳定、与旧参考一致或不一致的帧都一样；不可信的帧和已夹住也一样
    for (const auto x : {0.02, 0., 0.5}) {
        for (const auto trusted : {true, false}) {
            for (const auto grasped : {false, true}) {
                const auto output = feed(monitor, make_base_pose(x), trusted, grasped);
                EXPECT_EQ(output.state, Monitor::State::kMoved);
                EXPECT_FALSE(output.reference.has_value());
                EXPECT_FALSE(output.displacement.has_value());
                EXPECT_EQ(output.inconsistent_count, 5);
                ASSERT_TRUE(output.comparison.has_value());
                EXPECT_FALSE(output.comparison->consistent);
                EXPECT_NEAR(output.comparison->center_distance, 0.02, 1e-9);
            }
        }
    }
    EXPECT_EQ(monitor.state(), Monitor::State::kMoved);
}

// reset() 后回到无参考；可信但不稳定的帧不建立参考，下一个 trusted 且 stable 的帧以新位置建立
TEST(ReferenceMonitor, ReestablishesAfterReset) {
    auto monitor = established();
    for (int i = 0; i < 5; i++)
        feed(monitor, make_base_pose(0.02));
    monitor.reset();
    EXPECT_EQ(monitor.state(), Monitor::State::kNoReference);

    auto output = monitor.update(
        make_result(make_base_pose(0.02), make_camera(), true, false), make_camera());
    EXPECT_EQ(output.state, Monitor::State::kNoReference);
    EXPECT_FALSE(output.comparison.has_value());
    EXPECT_EQ(output.inconsistent_count, 0);
    EXPECT_EQ(feed(monitor, make_base_pose(0.02), false).state, Monitor::State::kNoReference);

    output = feed(monitor, make_base_pose(0.5));
    EXPECT_EQ(output.state, Monitor::State::kEstablished);
    ASSERT_TRUE(output.reference.has_value());
    EXPECT_TRUE(output.reference->isApprox(make_base_pose(0.5), 1e-9));
    EXPECT_EQ(output.inconsistent_count, 0);
    EXPECT_TRUE(feed(monitor, make_base_pose(0.501)).comparison->consistent);
}

// 已夹住：单元偏离参考是预期结果，不比较、不会判为被移动，只输出位移；
// 计数既不增加也不清零，不可信的帧没有位移
TEST(ReferenceMonitor, GraspedOnlyReportsDisplacement) {
    auto monitor = established();
    EXPECT_EQ(feed(monitor, make_base_pose(0.02)).inconsistent_count, 1);

    const auto reference = make_base_pose(0.);
    for (int i = 1; i <= 20; i++) {
        auto pose = reference;
        pose.translation() += reference.linear().col(2) * 0.01 * i;
        const auto output = feed(monitor, pose, true, true);
        EXPECT_EQ(output.state, Monitor::State::kGrasped);
        EXPECT_TRUE(output.reference.has_value());
        EXPECT_FALSE(output.comparison.has_value());
        EXPECT_EQ(output.inconsistent_count, 1);
        ASSERT_TRUE(output.displacement.has_value());
        EXPECT_NEAR(output.displacement->total, 0.01 * i, 1e-9);
        EXPECT_NEAR(output.displacement->along_axis, 0.01 * i, 1e-9);
    }
    const auto untrusted = feed(monitor, make_base_pose(0.3), false, true);
    EXPECT_EQ(untrusted.state, Monitor::State::kGrasped);
    EXPECT_FALSE(untrusted.displacement.has_value());
    EXPECT_EQ(monitor.state(), Monitor::State::kGrasped);
}

// 没有参考时已夹住的帧不建立参考
TEST(ReferenceMonitor, GraspedFrameDoesNotEstablishReference) {
    auto monitor = Monitor{};
    const auto output = feed(monitor, make_base_pose(0.), true, true);
    EXPECT_EQ(output.state, Monitor::State::kNoReference);
    EXPECT_FALSE(output.reference.has_value());
    EXPECT_FALSE(output.displacement.has_value());
}

// 一致的可信帧把计数清零，所以不一致必须连续
TEST(ReferenceMonitor, ConsistentFrameResetsCount) {
    auto monitor = established();
    for (int round = 0; round < 3; round++) {
        for (int i = 1; i <= 4; i++)
            EXPECT_EQ(feed(monitor, make_base_pose(0.02)).inconsistent_count, i);
        const auto output = feed(monitor, make_base_pose(0.001));
        EXPECT_TRUE(output.comparison->consistent);
        EXPECT_EQ(output.inconsistent_count, 0);
        EXPECT_EQ(output.state, Monitor::State::kEstablished);
    }
}

// 相机移动、目标静止：目标在相机系下的位姿变了，换到底座系后仍与参考一致
TEST(ReferenceMonitor, CameraMotionWithStaticTargetIsConsistent) {
    auto monitor = Monitor{};
    const auto target = make_base_pose(0.03, 10.);
    auto camera = make_camera();
    monitor.update(make_result(target, camera, true, true), camera);
    for (int i = 1; i <= 30; i++) {
        camera = make_camera(2. * i);
        const auto result = make_result(target, camera, true, true);
        const auto output = monitor.update(result, camera);
        EXPECT_EQ(output.state, Monitor::State::kEstablished);
        ASSERT_TRUE(output.comparison.has_value());
        EXPECT_TRUE(output.comparison->consistent);
        EXPECT_NEAR(output.comparison->center_distance, 0., 1e-9);
        EXPECT_NEAR(output.comparison->axis_deg, 0., 1e-4);
        EXPECT_EQ(output.inconsistent_count, 0);
    }
}

// 位移：总距离和沿参考长轴的分量。参考长轴倾斜时，沿轴分量不等于 z 方向的位移
TEST(ReferenceMonitor, Displacement) {
    auto monitor = Monitor{};
    const auto camera = make_camera();
    const auto reference = make_base_pose(0., 30.);
    monitor.update(make_result(reference, camera, true, true), camera);

    // 沿参考长轴移动 40 mm，再沿垂直方向（参考 x 轴）移动 30 mm
    auto moved = reference;
    moved.translation() += reference.linear().col(2) * 0.04 + reference.linear().col(0) * 0.03;
    const auto output = feed(monitor, moved);
    ASSERT_TRUE(output.displacement.has_value());
    EXPECT_NEAR(output.displacement->total, 0.05, 1e-9);
    EXPECT_NEAR(output.displacement->along_axis, 0.04, 1e-9);

    // 反向移动时沿轴分量为负
    moved = reference;
    moved.translation() -= reference.linear().col(2) * 0.02;
    EXPECT_NEAR(feed(monitor, moved).displacement->along_axis, -0.02, 1e-9);
}

// 阈值和帧数可配置
TEST(ReferenceMonitor, ConfigurableThresholds) {
    auto monitor = Monitor{Monitor::Config{
        .max_center_distance = 0.02, .max_axis_deg = 10., .inconsistent_frames = 2}};
    const auto camera = make_camera();
    monitor.update(make_result(make_base_pose(0.), camera, true, true), camera);
    EXPECT_TRUE(feed(monitor, make_base_pose(0.015, 8.)).comparison->consistent);
    EXPECT_EQ(feed(monitor, make_base_pose(0.03)).inconsistent_count, 1);
    EXPECT_EQ(feed(monitor, make_base_pose(0.03)).state, Monitor::State::kMoved);
}
