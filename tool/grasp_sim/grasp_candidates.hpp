#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include <eigen3/Eigen/Geometry>

namespace rmcs_energy_trace::grasp_sim {

struct GraspConfig {
    // link_6 系下的抓取点 (TCP) 偏移，默认 TCP 即 link_6 原点
    Eigen::Vector3d tcp_offset = Eigen::Vector3d::Zero();
    // 预抓取点沿 link_6 +z 后退的距离
    double pre_distance = 0.10;
    int azimuth_samples = 24;
    // 接近方向偏离水平的惩罚权重
    double vertical_weight = 2.0;
};

struct GraspCandidate {
    Eigen::Isometry3d grasp;     // link_6 在抓取时的位姿 (base_link 系)
    Eigen::Isometry3d pre_grasp; // link_6 在预抓取时的位姿
    double azimuth = 0.;         // 接近方位角 (rad)，相对于最优方向
    int sign = 1;                // link_6 x 轴与长轴的同向 (+1) / 反向 (-1)
    double cost = 0.;
};

// 由物体中心 center 与长轴 axis 生成抓取候选：link_6 x ∥ ±axis，link_6 z ⟂ axis，
// 接近方向 (-z) 越接近水平且越从机器人一侧向外越优先
inline auto generate_grasps(
    const Eigen::Vector3d& center, const Eigen::Vector3d& axis_in,
    const Eigen::Vector3d& base_position, const GraspConfig& config)
    -> std::vector<GraspCandidate> {
    const Eigen::Vector3d axis = axis_in.normalized();

    Eigen::Vector3d radial = center - base_position;
    radial.z() = 0.;
    radial = radial.norm() < 1e-3 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d{radial.normalized()};

    // 最优的 link_6 z 轴：指向机器人一侧 (-radial) 在垂直于长轴平面内的投影
    Eigen::Vector3d e1 = -radial - (-radial).dot(axis) * axis;
    if (e1.norm() < 1e-6)
        e1 = axis.unitOrthogonal();
    e1.normalize();
    const Eigen::Vector3d e2 = axis.cross(e1);

    auto candidates = std::vector<GraspCandidate>{};
    for (int sign : {1, -1}) {
        for (int k = 0; k < config.azimuth_samples; k++) {
            const auto phi = 2. * std::numbers::pi * k / config.azimuth_samples;
            const Eigen::Vector3d z6 = std::cos(phi) * e1 + std::sin(phi) * e2;
            const Eigen::Vector3d x6 = sign * axis;
            const Eigen::Vector3d y6 = z6.cross(x6);

            Eigen::Matrix3d rotation;
            rotation.col(0) = x6;
            rotation.col(1) = y6;
            rotation.col(2) = z6;

            auto candidate = GraspCandidate{};
            candidate.grasp = Eigen::Isometry3d::Identity();
            candidate.grasp.linear() = rotation;
            candidate.grasp.translation() = center - rotation * config.tcp_offset;
            candidate.pre_grasp = candidate.grasp;
            candidate.pre_grasp.translation() += config.pre_distance * z6;
            candidate.azimuth = std::remainder(phi, 2. * std::numbers::pi);
            candidate.sign = sign;

            const Eigen::Vector3d approach = -z6;
            candidate.cost = std::acos(std::clamp(approach.dot(radial), -1., 1.))
                           + config.vertical_weight * std::abs(approach.z())
                           + (sign < 0 ? 1e-3 : 0.);
            candidates.push_back(candidate);
        }
    }
    std::ranges::stable_sort(candidates, [](const GraspCandidate& a, const GraspCandidate& b) {
        return a.cost < b.cost;
    });
    return candidates;
}

} // namespace rmcs_energy_trace::grasp_sim
