#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation.h>

namespace rmcs_energy_trace::registration {

// point-to-plane ICP 每次迭代的线性最小二乘求解。
// 方程与 pcl::registration::TransformationEstimationPointToPlaneLLS 相同，区别在于解法。
//
// PCL 直接对 6×6 法方程矩阵求逆。对应点约束不住全部 6 个自由度时（例如只剩几个对应点），
// 矩阵奇异，结果为 NaN，下一次最近邻查询触发 PCL 的断言，进程中止。
// 这里用 SVD 求最小范数解：没有约束的方向不移动，其余方向照常求解。
class PointToPlaneEstimation final
    : public pcl::registration::TransformationEstimation<pcl::PointNormal, pcl::PointNormal> {
public:
    using Cloud = pcl::PointCloud<pcl::PointNormal>;

    void estimateRigidTransformation(
        const Cloud& source, const Cloud& target, Matrix4& transform) const override;
    void estimateRigidTransformation(
        const Cloud& source, const pcl::Indices& source_indices, const Cloud& target,
        Matrix4& transform) const override;
    void estimateRigidTransformation(
        const Cloud& source, const pcl::Indices& source_indices, const Cloud& target,
        const pcl::Indices& target_indices, Matrix4& transform) const override;
    void estimateRigidTransformation(
        const Cloud& source, const Cloud& target, const pcl::Correspondences& correspondences,
        Matrix4& transform) const override;
};

} // namespace rmcs_energy_trace::registration
