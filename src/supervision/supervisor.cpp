#include "supervision/supervisor.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace rmcs_energy_trace::supervision {

Supervisor::Supervisor(const Config& config) noexcept
    : config_{config} {}

void Supervisor::start(const Eigen::Isometry3d& pose) noexcept {
    state_ = State::kTracking;
    last_pose_ = pose;
    bad_frames_ = 0;
    no_good_frames_ = 0;
}

void Supervisor::reset() noexcept {
    state_ = State::kSearching;
    bad_frames_ = 0;
    no_good_frames_ = 0;
}

Supervisor::Decision Supervisor::update(const Observation& observation) noexcept {
    if (state_ == State::kSearching) {
        return {
            .state = state_,
            .health = Health::kUnknown,
            .trusted = false,
            .reinitialize = true,
            .reason = "no target"};
    }

    auto reason = std::string_view{};
    auto jumped = false;
    const auto health = judge(observation, reason, jumped);
    // 跳变帧的位姿不能作为下一帧的比较基准，否则跳变后的第二帧会被当作正常
    if (observation.pose.has_value() && !jumped)
        last_pose_ = *observation.pose;

    switch (health) {
    case Health::kGood:
        bad_frames_ = 0;
        no_good_frames_ = 0;
        return {
            .state = state_,
            .health = health,
            .trusted = true,
            .reinitialize = false,
            .reason = reason};
    case Health::kUnknown: ++no_good_frames_; break;
    case Health::kBad:
        ++bad_frames_;
        ++no_good_frames_;
        break;
    }
    // 只有健康帧会清零计数
    if (bad_frames_ >= config_.bad_frames || no_good_frames_ >= config_.no_good_frames)
        reset();
    return {
        .state = state_,
        .health = health,
        .trusted = false,
        .reinitialize = state_ == State::kSearching,
        .reason = reason};
}

Supervisor::Health Supervisor::judge(
    const Observation& observation, std::string_view& reason, bool& jumped) const noexcept {
    if (!observation.pose.has_value()) {
        reason = "tracker failed";
        return Health::kBad;
    }
    const auto& pose = *observation.pose;

    const auto translation = (pose.translation() - last_pose_.translation()).norm();
    const auto cosine = std::clamp(pose.linear().col(2).dot(last_pose_.linear().col(2)), -1., 1.);
    const auto axis_deg = std::acos(cosine) * 180. / std::numbers::pi;
    if (translation > config_.max_jump_translation || axis_deg > config_.max_jump_axis_deg) {
        reason = "pose jump";
        jumped = true;
        return Health::kBad;
    }

    if (observation.detection_iou.has_value()
        && *observation.detection_iou < config_.min_detection_iou) {
        reason = "projection does not match detection";
        return Health::kBad;
    }

    // 封闭端没有被观测到（被遮挡、没有深度或在画面外）时，位姿沿轴线的位置没有约束，
    // 下面的深度统计看不出来
    if (observation.closed_end_observed_ratio.has_value()
        && *observation.closed_end_observed_ratio < config_.min_closed_end_observed_ratio) {
        reason = "closed end is not observed";
        return Health::kUnknown;
    }

    // 只看未被遮挡的像素：支持和空闲空间违例
    const auto& depth = observation.depth;
    const auto unoccluded_ratio = depth.support_ratio + depth.free_space_ratio;
    const auto unoccluded = static_cast<double>(depth.valid_pixels) * unoccluded_ratio;
    if (unoccluded < static_cast<double>(config_.min_unoccluded_pixels)) {
        reason = "too few unoccluded pixels to check";
        return Health::kUnknown;
    }
    if (unoccluded_ratio < config_.min_unoccluded_ratio) {
        reason = "mostly occluded";
        return Health::kUnknown;
    }
    if (depth.free_space_ratio / unoccluded_ratio > config_.max_free_space_ratio) {
        reason = "mesh over background";
        return Health::kBad;
    }
    return Health::kGood;
}

} // namespace rmcs_energy_trace::supervision
