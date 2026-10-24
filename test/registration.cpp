#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <pcl/kdtree/kdtree_flann.h>

#include "geometry/camera.hpp"
#include "geometry/mesh.hpp"
#include "registration/registration.hpp"
#include "synthetic/depth_renderer.hpp"

#ifndef RMCS_ENERGY_UNIT_MESH
# error "RMCS_ENERGY_UNIT_MESH must be defined"
#endif

using rmcs_energy_trace::geometry::CameraIntrinsics;
using rmcs_energy_trace::geometry::load_binary_stl;
using rmcs_energy_trace::geometry::Mesh;
using rmcs_energy_trace::geometry::sample_surface;
using rmcs_energy_trace::registration::Registration;
using rmcs_energy_trace::synthetic::DepthRenderer;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr CameraIntrinsics kIntrinsics{600., 600., 320., 240.};

constexpr std::size_t kTrials = 20;
constexpr std::size_t kRequiredPass = 18;
// 抓取只依赖中心和长轴：ADD-S 对两端认反不敏感（约 2.5 mm），不作为合格标准
constexpr double kCenterThresholdMm = 3.;
constexpr double kAxisThresholdDeg = 2.;

struct Trial {
    Eigen::Isometry3d pose;
    cv::Mat depth;
    cv::Rect2i roi;
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

class Renderer {
public:
    explicit Renderer(const Mesh& mesh) {
        renderer_.initialize(
            mesh,
            DepthRenderer::Config{.width = kWidth, .height = kHeight, .intrinsics = kIntrinsics});
    }

    // 渲染物体，ground 非空时同时渲染地面 (法向量 normal、过点 point，相机系)
    auto render(
        const Eigen::Isometry3d& pose, std::mt19937& engine,
        const std::optional<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& ground) const -> Trial {
        auto plane = std::optional<DepthRenderer::Plane>{};
        if (ground.has_value())
            plane = DepthRenderer::Plane{ground->first, ground->second};
        auto frame = renderer_.render(pose, engine, plane);
        return Trial{pose, frame.depth, frame.roi};
    }

private:
    DepthRenderer renderer_;
};

class RegistrationTest : public testing::Test {
protected:
    static void SetUpTestSuite() {
        auto mesh = load_binary_stl(RMCS_ENERGY_UNIT_MESH);
        ASSERT_TRUE(mesh.has_value()) << mesh.error();
        mesh_ = std::make_unique<Mesh>(std::move(*mesh));
        renderer_ = std::make_unique<Renderer>(*mesh_);

        // ADD-S 的查询点稀疏、参考点极密，避免最近邻离散化带来的误差底噪
        evaluation_xyz_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        const auto queries = sample_surface(*mesh_, 5000, 123);
        for (const auto& point : *queries) {
            evaluation_xyz_->emplace_back(point.x, point.y, point.z);
        }
        auto reference = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        const auto dense = sample_surface(*mesh_, 1'000'000, 321);
        for (const auto& point : *dense) {
            reference->emplace_back(point.x, point.y, point.z);
        }
        reference_tree_.setInputCloud(reference);

        registration_ = std::make_unique<Registration>();
        ASSERT_TRUE(registration_->initialize(RMCS_ENERGY_UNIT_MESH).has_value());
    }

    static auto adds(const Eigen::Isometry3d& truth, const Eigen::Isometry3d& estimate) -> double {
        // ADD-S：真值下的模型点到估计位姿下模型的最近点距离均值
        const Eigen::Isometry3d relative = estimate.inverse() * truth;
        auto sum = 0.;
        auto indices = std::vector<int>(1);
        auto dists = std::vector<float>(1);
        for (const auto& point : *evaluation_xyz_) {
            const Eigen::Vector3d p = relative * point.getVector3fMap().cast<double>();
            reference_tree_.nearestKSearch(
                pcl::PointXYZ{
                    static_cast<float>(p.x()), static_cast<float>(p.y()),
                    static_cast<float>(p.z())},
                1, indices, dists);
            sum += std::sqrt(dists[0]);
        }
        return sum / static_cast<double>(evaluation_xyz_->size());
    }

    static void run_case(const char* name, const std::vector<Trial>& trials) {
        auto passed = std::size_t{};
        auto adds_all = std::vector<double>{}, trans_all = std::vector<double>{},
             rot_all = std::vector<double>{}, axis_all = std::vector<double>{},
             pre = std::vector<double>{}, feat = std::vector<double>{},
             coarse = std::vector<double>{}, icp = std::vector<double>{},
             total = std::vector<double>{};

        for (std::size_t i = 0; i < trials.size(); i++) {
            const auto& trial = trials[i];
            const auto cloud = Registration::depth_to_cloud(trial.depth, kIntrinsics, trial.roi);
            const auto result = registration_->register_cloud(*cloud);
            if (!result.has_value()) {
                std::println("[{}] #{:02} no result (cloud {} points)", name, i, cloud->size());
                adds_all.push_back(1.);
                continue;
            }
            const auto error = adds(trial.pose, result->pose);
            const Eigen::Isometry3d delta = trial.pose.inverse() * result->pose;
            const auto trans = (trial.pose.translation() - result->pose.translation()).norm() * 1e3;
            const auto rot = Eigen::AngleAxisd{delta.rotation()}.angle() * 180. / std::numbers::pi;
            // 长轴（模型 z 轴）方向误差，接近 180° 表示两端认反
            const auto axis_cos =
                std::clamp(trial.pose.linear().col(2).dot(result->pose.linear().col(2)), -1., 1.);
            const auto axis = std::acos(axis_cos) * 180. / std::numbers::pi;
            std::println(
                "[{}] #{:02} adds={:.2f}mm trans={:.2f}mm rot={:.1f}deg axis={:.1f}deg "
                "inlier={:.2f} fit={:.1e} "
                "t(pre/feat/coarse/icp)={:.0f}/{:.0f}/{:.0f}/{:.0f}ms",
                name, i, error * 1e3, trans, rot, axis, result->inlier_ratio, result->fitness,
                result->timing.preprocess_ms, result->timing.feature_ms, result->timing.coarse_ms,
                result->timing.icp_ms);
            if (trans < kCenterThresholdMm && axis < kAxisThresholdDeg)
                passed++;
            adds_all.push_back(error * 1e3);
            trans_all.push_back(trans);
            rot_all.push_back(rot);
            axis_all.push_back(axis);
            pre.push_back(result->timing.preprocess_ms);
            feat.push_back(result->timing.feature_ms);
            coarse.push_back(result->timing.coarse_ms);
            icp.push_back(result->timing.icp_ms);
            total.push_back(result->timing.total_ms);
        }

        std::println(
            "[{}] passed {}/{} (center < {:.0f}mm, axis < {:.0f}deg) | median/max trans "
            "{:.2f}/{:.2f}mm axis {:.1f}/{:.1f}deg | median/max adds {:.2f}/{:.2f}mm | "
            "median rot {:.1f}deg | median ms pre {:.0f} feat {:.0f} coarse {:.0f} icp "
            "{:.0f} total {:.0f}",
            name, passed, trials.size(), kCenterThresholdMm, kAxisThresholdDeg, median(trans_all),
            *std::ranges::max_element(trans_all), median(axis_all),
            *std::ranges::max_element(axis_all), median(adds_all),
            *std::ranges::max_element(adds_all), median(rot_all), median(pre), median(feat),
            median(coarse), median(icp), median(total));
        EXPECT_GE(passed, kRequiredPass);
    }

    static inline std::unique_ptr<Mesh> mesh_;
    static inline std::unique_ptr<Renderer> renderer_;
    static inline pcl::KdTreeFLANN<pcl::PointXYZ> reference_tree_;
    static inline pcl::PointCloud<pcl::PointXYZ>::Ptr evaluation_xyz_;
    static inline std::unique_ptr<Registration> registration_;
};

} // namespace

TEST_F(RegistrationTest, RandomOrientationNoBackground) {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{2024};
    auto trials = std::vector<Trial>{};
    for (std::size_t i = 0; i < kTrials; i++) {
        const auto distance = uniform(engine, 0.4, 0.8);
        // 物体中心在光轴附近，保证完整位于视野内
        const Eigen::Vector3d center{
            uniform(engine, -0.08, 0.08) * distance / 0.6,
            uniform(engine, -0.06, 0.06) * distance / 0.6,
            distance,
        };
        auto pose = Eigen::Isometry3d::Identity();
        pose.linear() = random_rotation(engine);
        pose.translation() = center;
        trials.push_back(renderer_->render(pose, engine, std::nullopt));
    }
    run_case("A", trials);
}

TEST_F(RegistrationTest, UprightOnGroundPlane) {
    // NOLINTNEXTLINE(cert-msc51-cpp): 固定种子保证测试可复现
    auto engine = std::mt19937{4048};
    auto trials = std::vector<Trial>{};
    for (std::size_t i = 0; i < kTrials; i++) {
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

        // 物体中心位于地面之上 0.0725 m（模型 z 轴为竖直方向）
        const Eigen::Vector3d up = world_to_camera * Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d ground_point = center - 0.0725 * up;
        trials.push_back(renderer_->render(pose, engine, std::pair{up, ground_point}));
    }
    run_case("B", trials);
}
