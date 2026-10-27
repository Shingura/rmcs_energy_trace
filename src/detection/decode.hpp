#pragma once

#include <cstddef>
#include <optional>

#include <opencv2/core/types.hpp>

#include "detection/detector.hpp"

namespace rmcs_energy_trace::detection {

// letterbox 参数：原图缩放 scale 倍后，左上角放在网络输入的 (pad_left, pad_top)
struct Letterbox {
    float scale = 1.f;
    int pad_left = 0;
    int pad_top = 0;
};

// 检测头输出 [1, attributes, count]，按属性主序存储：cx, cy, w, h, score，之后是掩码系数
struct Predictions {
    const float* data;
    std::size_t attributes;
    std::size_t count;
};

// 掩码原型输出 [1, channels, height, width]
struct Prototypes {
    const float* data;
    std::size_t channels;
    std::size_t height;
    std::size_t width;
};

// 把网络的原始输出解码为原图坐标下的检测结果：阈值、NMS、letterbox 逆映射，有原型时再解码掩码。
// 掩码的做法与 Ultralytics process_mask_native（retina_masks=True）一致，只对 NMS 之后保留的
// 检测做。不依赖推理，可单独测试。
// 没有 prototypes 时 attributes 必须是 5；有 prototypes 时必须是 5 + prototypes->channels。
// 输出不符时返回空。mask_ms 非空时写入掩码解码的耗时（毫秒）
Detector::Detections decode_detections(
    const Predictions& predictions, const std::optional<Prototypes>& prototypes,
    const Letterbox& lb, cv::Size size, float score_threshold, float nms_threshold,
    double* mask_ms = nullptr);

} // namespace rmcs_energy_trace::detection
