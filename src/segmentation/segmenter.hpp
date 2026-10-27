#pragma once

#include <expected>
#include <optional>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "utility/pimpl.hpp"

namespace rmcs_energy_trace::segmentation {

// 用 SAM（Segment Anything，SAM1）以检测框为提示分割能量单元，用 OpenVINO 推理。
// 网络由 scripts/export_sam.py 从官方权重导出为编码器和解码器两个 ONNX 文件。
// 预处理和后处理与官方 SamPredictor 一致；候选掩码的选法与官方 ONNX 部署代码（SamOnnxModel）
// 对检测框提示的选法一致，见 segmenter.cpp。
// 同一个实例不能在多个线程中同时调用 segment
class Segmenter {
    RMCS_PIMPL_DEFINITION(Segmenter)

public:
    struct Config {
        std::string encoder_path;
        std::string decoder_path;
        std::string infer_device = "CPU";
    };

    // 各阶段耗时（毫秒）
    struct Timing {
        double preprocess = 0.;
        double encoder = 0.;
        double decoder = 0.;
        double postprocess = 0.;
    };

    struct Result {
        cv::Mat mask; // CV_8UC1，与输入图像同尺寸，目标像素为 255
        float score;  // 网络预测的掩码 IoU，越接近 1 越可信
        Timing timing;
    };

    std::expected<void, std::string> initialize(const Config& config) noexcept;

    // bgr 为 CV_8UC3，box 为检测框（像素）。图像格式不符、框与图像不相交或推理失败时返回空
    std::optional<Result> segment(const cv::Mat& bgr, const cv::Rect2f& box) noexcept;
};

} // namespace rmcs_energy_trace::segmentation
