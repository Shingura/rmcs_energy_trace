#include "pipeline/reference_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rmcs_energy_trace::pipeline {

ReferenceMonitor::ReferenceMonitor(const Config& config) noexcept
    : config_{config} {}

void ReferenceMonitor::reset() noexcept {
    state_ = State::kNoReference;
    reference_.reset();
    moved_comparison_.reset();
    inconsistent_count_ = 0;
}

ReferenceMonitor::Output ReferenceMonitor::update(
    const Pipeline::Result& result, const Eigen::Isometry3d& camera_to_base,
    bool grasped) noexcept {
    // 被移动后保持，直到上层 reset()：不建立参考，不比较
    if (state_ == State::kMoved) {
        return {
            .state = State::kMoved,
            .reference = {},
            .comparison = moved_comparison_,
            .inconsistent_count = inconsistent_count_,
            .displacement = {}};
    }

    // 不可信的帧不参与比较，不改变计数
    if (!result.trusted || !result.pose.has_value()) {
        if (reference_.has_value())
            state_ = grasped ? State::kGrasped : State::kEstablished;
        return {
            .state = state_,
            .reference = reference_,
            .comparison = {},
            .inconsistent_count = inconsistent_count_,
            .displacement = {}};
    }
    const Eigen::Isometry3d pose = camera_to_base * *result.pose;

    if (!reference_.has_value()) {
        if (!result.stable || grasped)
            return {};
        reference_ = pose;
        inconsistent_count_ = 0;
        state_ = State::kEstablished;
        return {
            .state = state_,
            .reference = reference_,
            .comparison = {},
            .inconsistent_count = 0,
            .displacement = Displacement{}};
    }

    const auto& reference = *reference_;
    const Eigen::Vector3d offset = pose.translation() - reference.translation();
    const auto displacement =
        Displacement{.total = offset.norm(), .along_axis = offset.dot(reference.linear().col(2))};

    // 已夹住：单元偏离参考是预期结果，只输出位移
    if (grasped) {
        state_ = State::kGrasped;
        return {
            .state = state_,
            .reference = reference_,
            .comparison = {},
            .inconsistent_count = inconsistent_count_,
            .displacement = displacement};
    }

    const auto cosine = std::clamp(pose.linear().col(2).dot(reference.linear().col(2)), -1., 1.);
    const auto axis_deg = std::acos(cosine) * 180. / std::numbers::pi;
    const auto consistent =
        displacement.total <= config_.max_center_distance && axis_deg <= config_.max_axis_deg;
    inconsistent_count_ = consistent ? 0 : inconsistent_count_ + 1;

    const auto comparison = Comparison{
        .center_distance = displacement.total, .axis_deg = axis_deg, .consistent = consistent};
    state_ = State::kEstablished;
    auto reference_out = reference_;
    if (inconsistent_count_ >= config_.inconsistent_frames) {
        state_ = State::kMoved;
        moved_comparison_ = comparison;
        reference_.reset();
        reference_out.reset();
    }
    return {
        .state = state_,
        .reference = reference_out,
        .comparison = comparison,
        .inconsistent_count = inconsistent_count_,
        .displacement = displacement};
}

} // namespace rmcs_energy_trace::pipeline
