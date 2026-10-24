#include <cmath>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "geometry/camera.hpp"
#include "registration/registration.hpp"

using rmcs_energy_trace::geometry::CameraIntrinsics;
using rmcs_energy_trace::registration::Registration;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 48;
constexpr CameraIntrinsics kIntrinsics{60., 60., 32., 24.};
const auto kRoi = cv::Rect2i{10, 8, 40, 30};

auto make_depth() -> cv::Mat { return {kHeight, kWidth, CV_16UC1, cv::Scalar{500}}; }

auto make_mask(const cv::Rect2i& keep) -> cv::Mat {
    auto mask = cv::Mat(kHeight, kWidth, CV_8UC1, cv::Scalar{0});
    mask(keep).setTo(255);
    return mask;
}

auto config_with_erosion(int pixels) -> Registration::Config {
    auto config = Registration::Config{};
    config.mask_erosion_px = pixels;
    return config;
}

} // namespace

TEST(DepthMask, DefaultErosionIsThreePixels) {
    EXPECT_EQ(Registration::Config{}.mask_erosion_px, 3);
}

TEST(DepthMask, FullMaskMatchesBoxOnly) {
    const auto depth = make_depth();
    const auto box = Registration::depth_to_cloud(depth, kIntrinsics, kRoi);
    const auto masked = Registration::depth_to_cloud(
        depth, kIntrinsics, kRoi, make_mask({0, 0, kWidth, kHeight}), config_with_erosion(0));

    ASSERT_EQ(box->size(), static_cast<std::size_t>(kRoi.area()));
    ASSERT_EQ(masked->size(), box->size());
    for (std::size_t i = 0; i < box->size(); i++) {
        EXPECT_EQ((*box)[i].x, (*masked)[i].x);
        EXPECT_EQ((*box)[i].y, (*masked)[i].y);
        EXPECT_EQ((*box)[i].z, (*masked)[i].z);
    }

    // 空掩码退化为只用框
    const auto empty_mask = Registration::depth_to_cloud(depth, kIntrinsics, kRoi, {});
    EXPECT_EQ(empty_mask->size(), box->size());
}

TEST(DepthMask, PartialMaskKeepsOnlyMaskedPoints) {
    const auto depth = make_depth();
    // 掩码一部分在 ROI 之外，只有与 ROI 的交集被保留
    const auto keep = cv::Rect2i{30, 12, 34, 10};
    const auto cloud = Registration::depth_to_cloud(
        depth, kIntrinsics, kRoi, make_mask(keep), config_with_erosion(0));

    EXPECT_EQ(cloud->size(), static_cast<std::size_t>((keep & kRoi).area()));
    for (const auto& point : *cloud) {
        const auto u = static_cast<int>(std::lround(point.x * kIntrinsics.fx / point.z + 32.));
        const auto v = static_cast<int>(std::lround(point.y * kIntrinsics.fy / point.z + 24.));
        EXPECT_TRUE((keep & kRoi).contains({u, v}));
    }
}

TEST(DepthMask, InvalidDepthStaysExcluded) {
    auto depth = make_depth();
    depth(cv::Rect2i{20, 10, 5, 5}).setTo(0);
    const auto cloud = Registration::depth_to_cloud(
        depth, kIntrinsics, kRoi, make_mask({0, 0, kWidth, kHeight}), config_with_erosion(0));
    EXPECT_EQ(cloud->size(), static_cast<std::size_t>(kRoi.area() - 25));
}

TEST(DepthMask, ErosionShrinksKeptArea) {
    const auto depth = make_depth();
    const auto mask = make_mask({16, 12, 24, 20});

    const auto size = [&](int erosion) {
        return Registration::depth_to_cloud(
                   depth, kIntrinsics, kRoi, mask, config_with_erosion(erosion))
            ->size();
    };
    EXPECT_EQ(size(0), 24U * 20U);
    EXPECT_EQ(size(1), 22U * 18U);
    EXPECT_EQ(size(3), 18U * 14U);
    EXPECT_LT(size(3), size(1));

    // 不带 Config 的重载使用默认的 3 px
    EXPECT_EQ(Registration::depth_to_cloud(depth, kIntrinsics, kRoi, mask)->size(), 18U * 14U);
}

TEST(DepthMask, MismatchedMaskGivesEmptyCloud) {
    const auto depth = make_depth();
    const auto wrong_size = cv::Mat(kHeight / 2, kWidth / 2, CV_8UC1, cv::Scalar{255});
    EXPECT_TRUE(Registration::depth_to_cloud(depth, kIntrinsics, kRoi, wrong_size)->empty());
    const auto wrong_type = cv::Mat(kHeight, kWidth, CV_8UC3, cv::Scalar::all(255));
    EXPECT_TRUE(Registration::depth_to_cloud(depth, kIntrinsics, kRoi, wrong_type)->empty());
}
