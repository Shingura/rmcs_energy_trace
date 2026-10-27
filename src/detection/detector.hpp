#pragma once

#include <expected>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::detection {

// 用 YOLO11 检测或分割网络在彩色图中检测能量单元，用 OpenVINO 推理。
// 网络类型由编译后的输出自动判断：检测网络只输出框和得分；分割网络（YOLO11-seg）还输出掩码，
// 每个检测带有掩码。输出不符合这两种之一时 initialize 返回错误
class Detector {
    RMCS_PIMPL_DEFINITION(Detector)

public:
    struct Config {
        std::string network_path;
        std::string infer_device = "CPU";

        float score_threshold = 0.25f;
        float nms_threshold = 0.7f;
    };

    struct Detection {
        cv::Rect2f box;
        float score;
        // CV_8UC1，与输入图像同尺寸，目标像素为 255；与 segmentation::Segmenter::Result::mask
        // 的约定相同。网络没有掩码头（检测网络）时为空
        cv::Mat mask;
    };
    using Detections = std::vector<Detection>;

    // 各阶段耗时（毫秒），由最近一次 detect 更新
    struct Timing {
        double preprocess = 0.;
        double infer = 0.;
        double postprocess = 0.; // 解析输出、NMS 和坐标映射，不含掩码解码
        double mask = 0.;        // 掩码解码，网络没有掩码头时为 0
    };

    std::expected<void, std::string> initialize(const Config&) noexcept;
    Detections detect(const cv::Mat& bgr) noexcept;
    Timing last_timing() const noexcept;
};

} // namespace rmcs_energy_trace::detection
