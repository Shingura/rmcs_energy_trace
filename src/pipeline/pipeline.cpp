#include "pipeline/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <format>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/runtime/core.hpp>

#include "geometry/mesh.hpp"

namespace rmcs_energy_trace::pipeline {

namespace {

using Clock = std::chrono::steady_clock;

auto elapsed_ms(Clock::time_point from) noexcept -> double {
    return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
}

auto iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept -> double {
    const auto inter = static_cast<double>((a & b).area());
    const auto total = static_cast<double>(a.area()) + static_cast<double>(b.area()) - inter;
    return total > 0. ? inter / total : 0.;
}

} // namespace

std::optional<std::size_t> select_target(
    const detection::Detector::Detections& detections, cv::Size image, bool require_mask,
    const std::optional<cv::Rect2i>& target_box, double min_iou) noexcept {
    if (target_box && target_box->empty())
        return std::nullopt;
    const auto center_x = (image.width - 1) / 2.f;
    const auto center_y = (image.height - 1) / 2.f;
    auto best = std::optional<std::size_t>{};
    auto best_score = 0.;
    for (std::size_t i = 0; i < detections.size(); i++) {
        if (require_mask && detections[i].mask.empty())
            continue;
        const auto& box = detections[i].box;
        if (target_box) {
            const auto value = iou(cv::Rect2f{*target_box}, box);
            if (value >= min_iou && (!best || value > best_score)) {
                best_score = value;
                best = i;
            }
        } else {
            const auto distance =
                std::hypot(box.x + box.width / 2.f - center_x, box.y + box.height / 2.f - center_y);
            if (!best || distance < best_score) {
                best_score = distance;
                best = i;
            }
        }
    }
    return best;
}

std::optional<double> target_detection_iou(
    const detection::Detector::Detections& detections, const cv::Rect2i& projected_box) noexcept {
    auto best = std::optional<double>{};
    for (const auto& detection : detections) {
        const auto value = iou(cv::Rect2f{projected_box}, detection.box);
        if (value > 0. && (!best || value > *best))
            best = value;
    }
    return best;
}

struct Pipeline::Impl {
    Config config;
    bool initialized = false;

    detection::Detector detector;
    segmentation::Segmenter segmenter;
    registration::Registration registration;
    registration::DepthCheck depth_check;
    tracking::Tracker tracker;
    supervision::Supervisor supervisor;

    State state = State::kSearching;
    // 到上一帧为止连续可信的处理帧数
    int trusted_streak = 0;

    // 分割网络的输出有 2 个（检测和掩码原型），检测网络只有 1 个；Detector 没有公开这一信息
    static bool network_has_mask(const std::string& path) {
        auto core = ov::Core{};
        return core.read_model(path)->outputs().size() == 2;
    }

    std::expected<void, std::string> initialize(const Config& new_config) {
        initialized = false;
        state = State::kSearching;
        trusted_streak = 0;
        config = new_config;
        config.tracker.camera = config.camera;

        if (auto result = detector.initialize(config.detector); !result)
            return std::unexpected{std::move(result.error())};
        if (config.mask_source == MaskSource::kNetwork
            && !network_has_mask(config.detector.network_path)) {
            return std::unexpected{
                "Mask source is the segmentation network, but the network has no mask head"};
        }
        if (config.mask_source == MaskSource::kSam) {
            if (auto result = segmenter.initialize(config.segmenter); !result)
                return std::unexpected{std::move(result.error())};
        }
        if (auto result = registration.initialize(config.mesh_path, config.registration); !result)
            return std::unexpected{std::move(result.error())};
        auto mesh = geometry::load_binary_stl(config.mesh_path);
        if (!mesh)
            return std::unexpected{std::move(mesh.error())};
        depth_check = registration::DepthCheck{std::move(*mesh), config.depth_check};
        if (auto result = tracker.initialize(config.tracker); !result)
            return std::unexpected{std::move(result.error())};
        supervisor = supervision::Supervisor{config.supervisor};

        initialized = true;
        return {};
    }

    void search(
        const cv::Mat& bgr, const cv::Mat& depth, const std::optional<Eigen::Isometry3d>& target,
        Result& result) {
        auto& diag = result.diagnostics;
        auto target_box = std::optional<cv::Rect2i>{};
        if (target)
            target_box = depth_check.projected_box(config.camera, *target);
        diag.selected = select_target(
            diag.detections, {config.camera.width, config.camera.height},
            config.mask_source == MaskSource::kNetwork, target_box,
            config.supervisor.min_detection_iou);
        if (!diag.selected)
            return;
        const auto& detection = diag.detections[*diag.selected];

        switch (config.mask_source) {
        case MaskSource::kNetwork: diag.mask = detection.mask; break;
        case MaskSource::kSam: {
            const auto begin = Clock::now();
            if (auto segmented = segmenter.segment(bgr, detection.box))
                diag.mask = std::move(segmented->mask);
            diag.timing.segment_ms = elapsed_ms(begin);
            // SAM 失败时不退化为只用框：框里有相邻单元，用框配准容易选错
            if (diag.mask.empty())
                return;
            break;
        }
        case MaskSource::kNone: break;
        }

        const auto roi = cv::Rect2i{detection.box} & cv::Rect2i{0, 0, bgr.cols, bgr.rows};
        if (roi.empty())
            return;
        auto begin = Clock::now();
        diag.registration_attempted = true;
        diag.registration = registration.register_depth(depth, config.camera, roi, diag.mask);
        diag.timing.register_ms = elapsed_ms(begin);
        if (!diag.registration)
            return;

        result.pose = diag.registration->pose;
        begin = Clock::now();
        const auto started = tracker.start(diag.registration->pose, bgr, depth);
        diag.timing.track_ms = elapsed_ms(begin);
        if (!started) {
            // 跟踪器没有启动就不进入跟踪状态，下一帧重新配准
            result.trusted = false;
            return;
        }
        supervisor.start(diag.registration->pose);
        state = State::kTracking;
        // 配准自己的深度检查已经通过
        result.trusted = true;
    }

    void track(const cv::Mat& bgr, const cv::Mat& depth, Result& result) {
        auto& diag = result.diagnostics;
        auto begin = Clock::now();
        const auto tracked = tracker.track(bgr, depth);
        diag.timing.track_ms = elapsed_ms(begin);
        diag.tracker_ok = tracked.has_value();

        begin = Clock::now();
        auto observation = supervision::Supervisor::Observation{};
        if (tracked) {
            result.pose = tracked->pose;
            observation.pose = tracked->pose;
            diag.projected_box = depth_check.projected_box(config.camera, tracked->pose);
            observation.depth =
                depth_check.check(depth, config.camera, diag.projected_box, {}, tracked->pose);
            observation.closed_end_observed_ratio =
                depth_check.end_observed_ratios(depth, config.camera, tracked->pose).closed;
            observation.detection_iou = target_detection_iou(diag.detections, diag.projected_box);
        }
        const auto decision = supervisor.update(observation);
        diag.timing.check_ms = elapsed_ms(begin);
        diag.observation = observation;
        diag.decision = decision;

        result.trusted = decision.trusted && !decision.reinitialize;
        if (decision.reinitialize)
            state = State::kSearching;
    }

    std::expected<Result, std::string> process(
        const cv::Mat& bgr, const cv::Mat& depth, const std::optional<Eigen::Isometry3d>& target) {
        if (!initialized)
            return std::unexpected{"Pipeline is not initialized"};
        const auto size = cv::Size{config.camera.width, config.camera.height};
        if (bgr.type() != CV_8UC3 || depth.type() != CV_16UC1 || bgr.size() != size
            || depth.size() != size) {
            return std::unexpected{std::format(
                "Expected CV_8UC3 and CV_16UC1 images of {}x{}", size.width, size.height)};
        }

        const auto begin = Clock::now();
        auto result = Result{};
        result.state = state;
        auto& diag = result.diagnostics;

        // 搜索和跟踪状态都要检测
        auto detect_begin = Clock::now();
        diag.detections = detector.detect(bgr);
        diag.timing.detect_ms = elapsed_ms(detect_begin);

        if (state == State::kSearching)
            search(bgr, depth, target, result);
        else
            track(bgr, depth, result);
        trusted_streak = result.trusted ? trusted_streak + 1 : 0;
        result.stable = state == State::kTracking && trusted_streak >= config.stable_frames;
        diag.timing.total_ms = elapsed_ms(begin);
        return result;
    }
};

std::expected<void, std::string> Pipeline::initialize(const Config& config) noexcept try {
    return pimpl_->initialize(config);
} catch (const std::exception& e) {
    return std::unexpected{std::string{"Failed to initialize pipeline: "} + e.what()};
}

std::expected<Pipeline::Result, std::string>
    Pipeline::process(
        const cv::Mat& bgr, const cv::Mat& depth,
        const std::optional<Eigen::Isometry3d>& target) noexcept try {
    return pimpl_->process(bgr, depth, target);
} catch (const std::exception& e) {
    return std::unexpected{std::string{"Pipeline failed: "} + e.what()};
}

Pipeline::State Pipeline::state() const noexcept { return pimpl_->state; }

Pipeline::Pipeline() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Pipeline::~Pipeline() noexcept = default;

} // namespace rmcs_energy_trace::pipeline
