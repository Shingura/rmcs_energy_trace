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
// 搜索状态：每帧检测，选检测框中心离画面中心最近的一个配准；配准通过后启动 Tracker 和
// Supervisor，进入跟踪状态。 跟踪状态：Tracker 处理这一帧，同一帧也运行检测；用深度检查和检测框 IoU
// 交给 Supervisor， 监督要求重新初始化时回到搜索状态。
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
        Diagnostics diagnostics;
    };

    // 加载检测网络、mesh、配准、跟踪器（和 SAM），任何一步失败都返回错误。
    // mask_source 为 kNetwork 而网络没有掩码头（检测网络）时也返回错误
    std::expected<void, std::string> initialize(const Config& config) noexcept;

    // bgr 为 CV_8UC3，depth 为 CV_16UC1，尺寸都等于 Config::camera。
    // 未初始化、图像格式或尺寸不符时返回错误
    std::expected<Result, std::string> process(const cv::Mat& bgr, const cv::Mat& depth) noexcept;

    [[nodiscard]] State state() const noexcept;
};

} // namespace rmcs_energy_trace::pipeline
