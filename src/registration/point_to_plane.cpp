#include "registration/point_to_plane.hpp"

#include <cstddef>

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>

namespace rmcs_energy_trace::registration {

namespace {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix4 = PointToPlaneEstimation::Matrix4;

// 奇异值低于最大奇异值的这个比例时，视为该方向没有约束。
// 只用于去掉舍入误差量级的奇异值，不影响有约束但较弱的方向（例如能量单元绕长轴的转动）
constexpr double kRelativeSingularThreshold = 1e-12;

// 法方程 AᵀA x = Aᵀb，x = (绕 x、y、z 的小角度, 平移)。
// 每对对应点贡献一行：a = s × n，b = n · (d - s)，s 为源点，d、n 为目标点和法线
class NormalEquations {
public:
    void add(const pcl::PointNormal& source, const pcl::PointNormal& target) {
        const Eigen::Vector3d s = source.getVector3fMap().cast<double>();
        const Eigen::Vector3d d = target.getVector3fMap().cast<double>();
        const Eigen::Vector3d n = target.getNormalVector3fMap().cast<double>();
        if (!s.allFinite() || !d.allFinite() || !n.allFinite())
            return;

        auto row = Vector6d{};
        row << s.cross(n), n;
        ata_ += row * row.transpose();
        atb_ += row * n.dot(d - s);
    }

    Matrix4 solve() const {
        auto svd = Eigen::JacobiSVD<Matrix6d>{ata_, Eigen::ComputeFullU | Eigen::ComputeFullV};
        svd.setThreshold(kRelativeSingularThreshold);
        const Vector6d x = svd.solve(atb_);

        // 与 PCL 相同，按 Rz(γ) Ry(β) Rx(α) 组成旋转
        auto transform = Eigen::Affine3d::Identity();
        transform.linear() = (Eigen::AngleAxisd{x(2), Eigen::Vector3d::UnitZ()}
                              * Eigen::AngleAxisd{x(1), Eigen::Vector3d::UnitY()}
                              * Eigen::AngleAxisd{x(0), Eigen::Vector3d::UnitX()})
                                 .toRotationMatrix();
        transform.translation() = x.tail<3>();
        return transform.matrix().cast<float>();
    }

private:
    Matrix6d ata_ = Matrix6d::Zero();
    Vector6d atb_ = Vector6d::Zero();
};

} // namespace

void PointToPlaneEstimation::estimateRigidTransformation(
    const Cloud& source, const Cloud& target, Matrix4& transform) const {
    auto equations = NormalEquations{};
    for (std::size_t i = 0; i < source.size() && i < target.size(); i++)
        equations.add(source[i], target[i]);
    transform = equations.solve();
}

void PointToPlaneEstimation::estimateRigidTransformation(
    const Cloud& source, const pcl::Indices& source_indices, const Cloud& target,
    Matrix4& transform) const {
    auto equations = NormalEquations{};
    for (std::size_t i = 0; i < source_indices.size() && i < target.size(); i++)
        equations.add(source[source_indices[i]], target[i]);
    transform = equations.solve();
}

void PointToPlaneEstimation::estimateRigidTransformation(
    const Cloud& source, const pcl::Indices& source_indices, const Cloud& target,
    const pcl::Indices& target_indices, Matrix4& transform) const {
    auto equations = NormalEquations{};
    for (std::size_t i = 0; i < source_indices.size() && i < target_indices.size(); i++)
        equations.add(source[source_indices[i]], target[target_indices[i]]);
    transform = equations.solve();
}

void PointToPlaneEstimation::estimateRigidTransformation(
    const Cloud& source, const Cloud& target, const pcl::Correspondences& correspondences,
    Matrix4& transform) const {
    auto equations = NormalEquations{};
    for (const auto& correspondence : correspondences)
        equations.add(source[correspondence.index_query], target[correspondence.index_match]);
    transform = equations.solve();
}

} // namespace rmcs_energy_trace::registration
