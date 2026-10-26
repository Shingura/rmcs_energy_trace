#include <cmath>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

#include "registration/point_to_plane.hpp"

using rmcs_energy_trace::registration::PointToPlaneEstimation;
using Cloud = pcl::PointCloud<pcl::PointNormal>;
using Matrix4 = PointToPlaneEstimation::Matrix4;

namespace {

auto make_point(const Eigen::Vector3f& position, const Eigen::Vector3f& normal)
    -> pcl::PointNormal {
    auto point = pcl::PointNormal{};
    point.getVector3fMap() = position;
    point.getNormalVector3fMap() = normal.normalized();
    return point;
}

// 平面 z = 0 上的网格点，法线 +z。这样的对应点约束不住绕 z 的转动和沿 x、y 的平移
auto make_plane(int count) -> Cloud {
    auto cloud = Cloud{};
    for (int i = 0; i < count; i++) {
        const int column = i % 7;
        const int row = i / 7;
        const auto x = 0.01f * static_cast<float>(column);
        const auto y = 0.01f * static_cast<float>(row);
        cloud.push_back(make_point({x, y, 0.f}, Eigen::Vector3f::UnitZ()));
    }
    return cloud;
}

auto translated(const Cloud& cloud, const Eigen::Vector3f& offset) -> Cloud {
    auto output = cloud;
    for (auto& point : output)
        point.getVector3fMap() += offset;
    return output;
}

auto estimate(const Cloud& source, const Cloud& target) -> Matrix4 {
    auto transform = Matrix4{};
    PointToPlaneEstimation{}.estimateRigidTransformation(source, target, transform);
    return transform;
}

} // namespace

// 约束充分时与 PCL 的 TransformationEstimationPointToPlaneLLS 结果相同
TEST(PointToPlane, MatchesPclWhenWellConstrained) {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证可复现
    auto rng = std::mt19937{7};
    auto uniform = std::uniform_real_distribution<float>{-1.f, 1.f};
    auto target = Cloud{};
    for (int i = 0; i < 200; i++) {
        const Eigen::Vector3f position =
            Eigen::Vector3f{uniform(rng), uniform(rng), uniform(rng)} * 0.05f;
        const Eigen::Vector3f normal = Eigen::Vector3f{uniform(rng), uniform(rng), uniform(rng)};
        target.push_back(make_point(position, normal));
    }
    const Eigen::Affine3f motion =
        Eigen::Translation3f{0.004f, -0.003f, 0.002f}
        * Eigen::AngleAxisf{0.03f, Eigen::Vector3f{1.f, 2.f, 3.f}.normalized()};
    auto source = target;
    for (auto& point : source)
        point.getVector3fMap() = motion * point.getVector3fMap();

    auto expected = Matrix4{};
    pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>{}
        .estimateRigidTransformation(source, target, expected);
    const auto actual = estimate(source, target);

    ASSERT_TRUE(expected.allFinite());
    EXPECT_TRUE(actual.isApprox(expected, 1e-5f)) << "actual\n"
                                                  << actual << "\nexpected\n"
                                                  << expected;
}

// 只有平面时，有约束的方向（沿法线的平移）照常求解，没有约束的方向不移动
TEST(PointToPlane, LeavesUnconstrainedDirectionsUnchanged) {
    const auto target = make_plane(49);
    const auto source = translated(target, {0.01f, -0.02f, 0.003f});

    const auto transform = estimate(source, target);

    ASSERT_TRUE(transform.allFinite());
    const Eigen::Matrix3f rotation = transform.topLeftCorner<3, 3>();
    EXPECT_TRUE(rotation.isApprox(Eigen::Matrix3f::Identity(), 1e-6f));
    EXPECT_NEAR(transform(0, 3), 0.f, 1e-6f);
    EXPECT_NEAR(transform(1, 3), 0.f, 1e-6f);
    EXPECT_NEAR(transform(2, 3), -0.003f, 1e-6f);
}

// 只剩 3 个对应点时，PCL 的解为 NaN；这里仍是有限值
TEST(PointToPlane, FewCorrespondencesGiveFiniteTransform) {
    const auto target = make_plane(3);
    const auto source = translated(target, {0.001f, 0.002f, 0.003f});

    EXPECT_TRUE(estimate(source, target).allFinite());
}

// 用 PCL 自带的求解时，这个输入使 ICP 在最近邻查询中触发断言、进程中止
TEST(PointToPlane, IcpOnPlaneFinishesWithFiniteTransform) {
    const auto target = std::make_shared<Cloud>(make_plane(49));
    const auto source = std::make_shared<Cloud>(translated(*target, {0.002f, 0.001f, 0.004f}));

    auto icp = pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal>{};
    icp.setTransformationEstimation(std::make_shared<PointToPlaneEstimation>());
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(0.01);
    icp.setMaximumIterations(30);
    auto aligned = Cloud{};
    icp.align(aligned);

    ASSERT_TRUE(icp.getFinalTransformation().allFinite());
    for (const auto& point : aligned)
        EXPECT_NEAR(point.z, 0.f, 1e-5f);
}
