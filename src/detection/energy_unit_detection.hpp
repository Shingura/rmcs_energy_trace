#pragma once

#include <expected>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::detection {

class EnergyUnitDetection {
    RMCS_PIMPL_DEFINITION(EnergyUnitDetection)

public:
    struct Config {
        std::string model_location;
        std::string infer_device = "CPU";

        float score_threshold = 0.25f;
        float nms_threshold = 0.7f;
    };

    struct Detection {
        cv::Rect2f box;
        float score;
    };
    using Detections = std::vector<Detection>;

    // 各阶段耗时（毫秒），由最近一次 detect 更新
    struct Timing {
        double preprocess = 0.;
        double infer = 0.;
        double postprocess = 0.;
    };

    std::expected<void, std::string> initialize(const Config&) noexcept;
    Detections detect(const cv::Mat& bgr) noexcept;
    Timing last_timing() const noexcept;
};

} // namespace rmcs_energy_trace::detection
