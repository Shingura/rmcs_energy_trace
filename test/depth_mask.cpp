#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "geometry/camera.hpp"
#include "geometry/depth_cloud.hpp"
#include "geometry/mesh.hpp"
#include "registration/depth_check.hpp"
#include "registration/registration.hpp"

using rmcs_energy_trace::geometry::Camera;
using rmcs_energy_trace::geometry::depth_to_cloud;
using rmcs_energy_trace::geometry::Mesh;
using rmcs_energy_trace::registration::DepthCheck;
using rmcs_energy_trace::registration::Registration;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 48;
constexpr Camera kCamera{
    .width = kWidth, .height = kHeight, .fx = 60., .fy = 60., .cx = 32., .cy = 24.};
const auto kRoi = cv::Rect2i{10, 8, 40, 30};

auto make_depth() -> cv::Mat { return {kHeight, kWidth, CV_16UC1, cv::Scalar{500}}; }

auto make_mask(const cv::Rect2i& keep) -> cv::Mat {
    auto mask = cv::Mat(kHeight, kWidth, CV_8UC1, cv::Scalar{0});
    mask(keep).setTo(255);
    return mask;
}

} // namespace

// 配准默认先把掩码内缩 3 px
TEST(DepthMask, RegistrationErodesThreePixelsByDefault) {
    EXPECT_EQ(Registration::Config{}.mask_erosion_px, 3);
}

TEST(DepthMask, FullMaskMatchesBoxOnly) {
    const auto depth = make_depth();
    const auto box = depth_to_cloud(depth, kCamera, kRoi);
    const auto masked = depth_to_cloud(depth, kCamera, kRoi, make_mask({0, 0, kWidth, kHeight}));

    ASSERT_EQ(box->size(), static_cast<std::size_t>(kRoi.area()));
    ASSERT_EQ(masked->size(), box->size());
    for (std::size_t i = 0; i < box->size(); i++) {
        EXPECT_EQ((*box)[i].x, (*masked)[i].x);
        EXPECT_EQ((*box)[i].y, (*masked)[i].y);
        EXPECT_EQ((*box)[i].z, (*masked)[i].z);
    }
}

TEST(DepthMask, PartialMaskKeepsOnlyMaskedPoints) {
    const auto depth = make_depth();
    // 掩码一部分在 ROI 之外，只有与 ROI 的交集被保留
    const auto keep = cv::Rect2i{30, 12, 34, 10};
    const auto cloud = depth_to_cloud(depth, kCamera, kRoi, make_mask(keep));

    EXPECT_EQ(cloud->size(), static_cast<std::size_t>((keep & kRoi).area()));
    for (const auto& point : *cloud) {
        const auto u = static_cast<int>(std::lround(point.x * kCamera.fx / point.z + 32.));
        const auto v = static_cast<int>(std::lround(point.y * kCamera.fy / point.z + 24.));
        EXPECT_TRUE((keep & kRoi).contains({u, v}));
    }
}

TEST(DepthMask, InvalidDepthStaysExcluded) {
    auto depth = make_depth();
    depth(cv::Rect2i{20, 10, 5, 5}).setTo(0);
    const auto cloud = depth_to_cloud(depth, kCamera, kRoi, make_mask({0, 0, kWidth, kHeight}));
    EXPECT_EQ(cloud->size(), static_cast<std::size_t>(kRoi.area() - 25));
}

TEST(DepthMask, ErosionShrinksKeptArea) {
    const auto depth = make_depth();
    const auto mask = make_mask({16, 12, 24, 20});

    const auto size = [&](int erosion) {
        return depth_to_cloud(depth, kCamera, kRoi, mask, erosion)->size();
    };
    EXPECT_EQ(size(0), 24U * 20U);
    EXPECT_EQ(size(1), 22U * 18U);
    EXPECT_EQ(size(3), 18U * 14U);
}

TEST(DepthMask, MismatchedMaskGivesEmptyCloud) {
    const auto depth = make_depth();
    const auto wrong_size = cv::Mat(kHeight / 2, kWidth / 2, CV_8UC1, cv::Scalar{255});
    EXPECT_TRUE(depth_to_cloud(depth, kCamera, kRoi, wrong_size)->empty());
    const auto wrong_type = cv::Mat(kHeight, kWidth, CV_8UC3, cv::Scalar::all(255));
    EXPECT_TRUE(depth_to_cloud(depth, kCamera, kRoi, wrong_type)->empty());
}

namespace {

// 长轴沿 z、长 0.15 m 的平板（x = 0 平面，y 方向宽 0.04 m），在 z = ±0.0375 处分段，
// 两端各一段、中段一段。端部按三角形重心划分：|z| > 半长 / 2
auto make_plate() -> Mesh {
    constexpr auto w = 0.02f;
    const auto quad = [&](float z0, float z1) {
        return std::array<Mesh::Triangle, 2>{
            Mesh::Triangle{
                Eigen::Vector3f{0.f, -w, z0}, Eigen::Vector3f{0.f, w, z0},
                Eigen::Vector3f{0.f, w, z1}},
            Mesh::Triangle{
                Eigen::Vector3f{0.f, -w, z0}, Eigen::Vector3f{0.f, w, z1},
                Eigen::Vector3f{0.f, -w, z1}}};
    };
    auto mesh = Mesh{};
    for (const auto& [z0, z1] :
         {std::pair{-0.075f, -0.0375f}, std::pair{-0.0375f, 0.0375f}, std::pair{0.0375f, 0.075f}}) {
        for (const auto& triangle : quad(z0, z1))
            mesh.triangles.push_back(triangle);
    }
    return mesh;
}

constexpr Camera kPlateCamera{
    .width = 320, .height = 240, .fx = 300., .fy = 300., .cx = 160., .cy = 120.};

// mesh 的 z 轴转到相机的 x 方向（封闭端在 +x 侧），平板正对相机，沿 x 平移 tx，距相机 0.5 m
auto make_plate_pose(double tx) -> Eigen::Isometry3d {
    auto pose = Eigen::Isometry3d::Identity();
    pose.linear() =
        Eigen::AngleAxisd{std::numbers::pi / 2., Eigen::Vector3d::UnitY()}.toRotationMatrix();
    pose.translation() = Eigen::Vector3d{tx, 0., 0.5};
    return pose;
}

// 整幅画面都在 0.5 m 处（平板被支持）。occlude_ends 时，真值位姿下两端所在的列
// （u < 137.5 和 u > 182.5）换成 0.4 m 处的前景，只露出中段
auto make_plate_depth(bool occlude_ends) -> cv::Mat {
    auto depth = cv::Mat{kPlateCamera.height, kPlateCamera.width, CV_16UC1, cv::Scalar{500}};
    if (occlude_ends) {
        depth.colRange(0, 138).setTo(400);
        depth.colRange(183, kPlateCamera.width).setTo(400);
    }
    return depth;
}

} // namespace

// 没有遮挡的真值位姿：两端都被观测到
TEST(DepthCheck, EndObservedWithoutOcclusion) {
    const auto check = DepthCheck{make_plate(), {}};
    const auto observed =
        check.end_observed_ratios(make_plate_depth(false), kPlateCamera, make_plate_pose(0.));
    EXPECT_GT(observed.closed, 0.95);
    EXPECT_GT(observed.open, 0.95);
}

// 两端被前景挡住、只露出中段：真值位姿和沿轴偏 15 mm 的位姿，两端都几乎观测不到。
// 沿轴偏 15 mm 时中段表面仍与实测深度吻合，这是深度检查的盲区
TEST(DepthCheck, EndObservedWithEndsOccluded) {
    const auto check = DepthCheck{make_plate(), {}};
    const auto depth = make_plate_depth(true);
    const auto truth = make_plate_pose(0.);
    const auto shifted = make_plate_pose(0.015);

    EXPECT_LT(check.end_observed_ratios(depth, kPlateCamera, truth).closed, 0.1);
    const auto observed = check.end_observed_ratios(depth, kPlateCamera, shifted);
    EXPECT_LT(observed.closed, 0.8);
    EXPECT_LT(observed.open, 0.8);

    // 中段仍与实测深度吻合
    const auto quality =
        check.check(depth, kPlateCamera, check.projected_box(kPlateCamera, shifted), {}, shifted);
    EXPECT_GT(quality.support_ratio, 0.1);
    EXPECT_DOUBLE_EQ(quality.free_space_ratio, 0.);
}

// 只有一端被挡住时，另一端不受影响
TEST(DepthCheck, EndObservedOnlyClosedEndOccluded) {
    const auto check = DepthCheck{make_plate(), {}};
    auto depth = make_plate_depth(false);
    depth.colRange(183, kPlateCamera.width).setTo(400);
    const auto observed = check.end_observed_ratios(depth, kPlateCamera, make_plate_pose(0.));
    EXPECT_LT(observed.closed, 0.1);
    EXPECT_GT(observed.open, 0.95);
}

// 没有实测深度的像素不算被观测到
TEST(DepthCheck, EndObservedNeedsDepth) {
    const auto check = DepthCheck{make_plate(), {}};
    auto depth = make_plate_depth(false);
    depth.colRange(183, kPlateCamera.width).setTo(0);
    const auto observed = check.end_observed_ratios(depth, kPlateCamera, make_plate_pose(0.));
    EXPECT_LT(observed.closed, 0.1);
    EXPECT_GT(observed.open, 0.95);
}

// 分母包括画面外的部分：封闭端有一半在画面外时约为 0.5。
// 封闭端的 u 范围为 160 + 600 tx + [22.5, 45]，tx = 0.2121 时中点在 u = 319.5
TEST(DepthCheck, EndObservedCountsPartOutsideImage) {
    const auto check = DepthCheck{make_plate(), {}};
    const auto observed =
        check.end_observed_ratios(make_plate_depth(false), kPlateCamera, make_plate_pose(0.2121));
    EXPECT_NEAR(observed.closed, 0.5, 0.05);
    EXPECT_GT(observed.open, 0.95);
}

TEST(DepthCheck, EndObservedBehindCamera) {
    const auto check = DepthCheck{make_plate(), {}};
    auto pose = make_plate_pose(0.);
    pose.translation().z() = -0.5;
    const auto observed = check.end_observed_ratios(make_plate_depth(false), kPlateCamera, pose);
    EXPECT_DOUBLE_EQ(observed.closed, 0.);
    EXPECT_DOUBLE_EQ(observed.open, 0.);
}
