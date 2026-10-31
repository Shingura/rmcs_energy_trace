#pragma once

#include <cstddef>
#include <expected>
#include <optional>
#include <string>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "detection/detector.hpp"
#include "geometry/camera.hpp"
#include "registration/depth_check.hpp"
#include "registration/registration.hpp"
#include "segmentation/segmenter.hpp"
#include "supervision/supervisor.hpp"
#include "tracking/tracker.hpp"
#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::pipeline {

// 完整的位姿估计流程：检测 -> 掩码 -> 配准 -> 跟踪 -> 监督。
//
// 搜索状态：每帧检测，选一个检测框配准；配准通过后启动 Tracker 和 Supervisor，进入跟踪状态。
// 不知道目标在哪里时（第一次初始化，操作手已经把准心对准目标）选检测框中心离画面中心最近的；
// 已知目标位姿时（重新初始化，相机不再对准目标）选与目标投影框 IoU 最大的。
// 跟踪状态：Tracker 处理这一帧，同一帧也运行检测；用深度检查和检测框 IoU 交给 Supervisor，
// 监督要求重新初始化时回到搜索状态。
//
// 只能在一个线程里调用 initialize 和 process（不要求线程安全）；耗时较长的 process
// 应放在组件的工作线程里，不要在实时回调中调用
class Pipeline {
    RMCS_PIMPL_DEFINITION(Pipeline)

public:
    enum class State { kSearching, kTracking };

    // 搜索状态下配准用的掩码来源
    enum class MaskSource {
        kNetwork, // 分割网络自带的掩码（Detector::Detection::mask），网络没有掩码头时 initialize
                  // 报错
        kSam,     // 以检测框为提示用 SAM 分割
        kNone,    // 不用掩码，只用检测框
    };

    struct Config {
        std::string mesh_path;
        // tracker.camera 在 initialize 中被这里的相机覆盖
        geometry::Camera camera;
        MaskSource mask_source = MaskSource::kNetwork;

        detection::Detector::Config detector;
        // 只在 mask_source 为 kSam 时使用
        segmentation::Segmenter::Config segmenter;
        registration::Registration::Config registration;
        // 监督用的深度检查，参数与配准内部的检查相同时用默认值即可
        registration::DepthCheck::Config depth_check;
        tracking::Tracker::Config tracker;
        supervision::Supervisor::Config supervisor;
        // 判定“跟踪稳定”所需的连续可信帧数（处理帧，不含 sequence_replay 跳过的帧）。
        // 默认 15 帧，30 fps 下 0.5 s；最终取值待实机数据确定
        int stable_frames = 15;
    };

    // 各阶段耗时（毫秒）。没有运行的阶段为 0
    struct Timing {
        double detect_ms = 0.;
        double segment_ms = 0.;                     // SAM
        double register_ms = 0.;                    // 配准（含反投影点云和深度检查）
        double track_ms = 0.;                       // Tracker::track，或搜索状态下的 Tracker::start
        double check_ms = 0.;                       // 跟踪状态下的深度检查和监督
        double total_ms = 0.;
    };

    struct Diagnostics {
        detection::Detector::Detections detections; // 这一帧的全部检测
        std::optional<std::size_t> selected;        // 搜索状态：选中的检测在 detections 中的下标
        cv::Mat mask;                               // 搜索状态：配准用到的掩码，没有时为空
        bool registration_attempted = false;        // 搜索状态：调用了配准
        std::optional<registration::Registration::Result> registration;
        // 跟踪状态
        bool tracker_ok = false;  // Tracker::track 给出了位姿
        cv::Rect2i projected_box; // 跟踪位姿的投影框
        std::optional<supervision::Supervisor::Observation> observation;
        std::optional<supervision::Supervisor::Decision> decision;
        Timing timing;
    };

    struct Result {
        // 处理这一帧时所处的状态。搜索状态的帧配准通过时仍为 kSearching，下一帧起为 kTracking；
        // 监督要求重新初始化的帧仍为 kTracking
        State state = State::kSearching;
        // mesh -> camera。搜索状态下为配准的位姿；跟踪状态下为 Tracker 的结果（跟踪失败时为空），
        // 即使不可信也给出，供诊断
        std::optional<Eigen::Isometry3d> pose;
        // 位姿是否可以交给机械臂。配准通过的那一帧可信（配准自带深度检查）；
        // 跟踪状态下由监督决定，要求重新初始化的帧不可信
        bool trusted = false;
        // 跟踪稳定：处于跟踪状态，且包括这一帧在内已经连续 Config::stable_frames 个处理帧可信。
        // 配准通过的那一帧算第 1 个可信帧；不可信的帧和重新初始化都把计数清零。
        // 机械臂控制在 stable 为 true 后才开始运动，此前原地不动，等视觉初始化并稳定；
        // 运动中跟丢（stable 变为 false）不停下，沿参考位姿继续走，同时重新初始化；
        // 是否停下由参考位姿监视决定（见 reference_monitor.hpp）
        bool stable = false;
        Diagnostics diagnostics;
    };

    // 加载检测网络、mesh、配准、跟踪器（和 SAM），任何一步失败都返回错误。
    // mask_source 为 kNetwork 而网络没有掩码头（检测网络）时也返回错误
    std::expected<void, std::string> initialize(const Config& config) noexcept;

    // bgr 为 CV_8UC3，depth 为 CV_16UC1，尺寸都等于 Config::camera。
    // 未初始化、图像格式或尺寸不符时返回错误。
    // target 为相机系下目标的位姿 (mesh -> camera)，只在搜索状态下使用：给出时按目标位置选检测框
    // （见 select_target），没有选中时这一帧不配准；不给时选离画面中心最近的。跟踪状态不使用
    std::expected<Result, std::string> process(
        const cv::Mat& bgr, const cv::Mat& depth,
        const std::optional<Eigen::Isometry3d>& target = std::nullopt) noexcept;

    [[nodiscard]] State state() const noexcept;
};

// 搜索状态下选目标，返回下标。target_box 为空时选检测框中心离画面中心最近的；
// 给出时选与它 IoU 最大的，最大 IoU 低于 min_iou 时不选。require_mask 为 true 时跳过没有掩码的检测
std::optional<std::size_t> select_target(
    const detection::Detector::Detections& detections, cv::Size image, bool require_mask,
    const std::optional<cv::Rect2i>& target_box, double min_iou) noexcept;

// 与投影框重叠的检测中最大的
// IoU；没有与它重叠的检测时为空：不重叠的检测属于其他单元，不是当前目标的证据
std::optional<double> target_detection_iou(
    const detection::Detector::Detections& detections, const cv::Rect2i& projected_box) noexcept;

} // namespace rmcs_energy_trace::pipeline
