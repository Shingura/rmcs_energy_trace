#include "detection/decode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "detection/yolo11.hpp"

namespace rmcs_energy_trace::detection {

namespace {

// 解码一个检测的掩码。box 为原图坐标下已裁剪到图像内的检测框，
// coefficients 为该检测的第一个掩码系数，相邻系数相隔 stride 个 float
cv::Mat decode_mask(
    const Prototypes& prototypes, const float* coefficients, std::size_t stride,
    const Letterbox& lb, cv::Size size, const cv::Rect2f& box) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);

    // 框内的整数像素范围：与 Ultralytics crop_mask 一样，按像素下标与框边界比较
    const auto x_begin = static_cast<int>(std::ceil(box.x));
    const auto y_begin = static_cast<int>(std::ceil(box.y));
    const auto x_end = std::min(static_cast<int>(std::ceil(box.x + box.width)), size.width);
    const auto y_end = std::min(static_cast<int>(std::ceil(box.y + box.height)), size.height);
    if (x_end <= x_begin || y_end <= y_begin)
        return mask;

    // 掩码系数与原型相乘，得到原型分辨率下的 logit
    const auto height = static_cast<int>(prototypes.height);
    const auto width = static_cast<int>(prototypes.width);
    const auto plane = prototypes.height * prototypes.width;
    cv::Mat logits = cv::Mat::zeros(height, width, CV_32F);
    auto* out = logits.ptr<float>();
    for (std::size_t k = 0; k < prototypes.channels; k++) {
        const auto weight = coefficients[k * stride];
        const auto* proto = prototypes.data + k * plane;
        for (std::size_t i = 0; i < plane; i++)
            out[i] += weight * proto[i];
    }

    // 原型分辨率是网络输入的 1/ratio。取出对应未填充图像的区域：
    // 缩放后的图像在输入中占 (pad, pad + 原图尺寸 * scale)，除以 ratio 后四舍五入
    const auto ratio_x = static_cast<float>(Yolo11::kInputWidth) / static_cast<float>(width);
    const auto ratio_y = static_cast<float>(Yolo11::kInputHeight) / static_cast<float>(height);
    const auto left = static_cast<float>(lb.pad_left) / ratio_x;
    const auto top = static_cast<float>(lb.pad_top) / ratio_y;
    const auto right = left + static_cast<float>(size.width) * lb.scale / ratio_x;
    const auto bottom = top + static_cast<float>(size.height) * lb.scale / ratio_y;
    const auto crop_x0 = std::clamp(static_cast<int>(std::lround(left)), 0, width - 1);
    const auto crop_y0 = std::clamp(static_cast<int>(std::lround(top)), 0, height - 1);
    const auto crop_x1 = std::clamp(static_cast<int>(std::lround(right)), crop_x0 + 1, width);
    const auto crop_y1 = std::clamp(static_cast<int>(std::lround(bottom)), crop_y0 + 1, height);
    const auto crop = logits({crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0}).clone();

    // 把裁剪后的 logit 双线性缩放到原图尺寸（像素中心对齐，与 F.interpolate 的
    // align_corners=False 相同），只算框内的部分
    const auto sx = static_cast<double>(crop.cols) / size.width;
    const auto sy = static_cast<double>(crop.rows) / size.height;
    const double matrix[6] = {sx, 0., sx * (x_begin + 0.5) - 0.5,
                              0., sy, sy * (y_begin + 0.5) - 0.5};
    const auto transform = cv::Matx23d{matrix};
    auto roi = cv::Mat{};
    cv::warpAffine(
        crop, roi, transform, {x_end - x_begin, y_end - y_begin},
        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);

    // logit > 0 即 sigmoid > 0.5
    auto target = mask(cv::Rect{x_begin, y_begin, x_end - x_begin, y_end - y_begin});
    cv::compare(roi, 0., target, cv::CMP_GT);
    return mask;
}

} // namespace

Detector::Detections decode_detections(
    const Predictions& predictions, const std::optional<Prototypes>& prototypes,
    const Letterbox& lb, cv::Size size, float score_threshold, float nms_threshold,
    double* mask_ms) {
    if (mask_ms)
        *mask_ms = 0.;

    const auto expected_attributes =
        Yolo11::kAttributes + (prototypes ? prototypes->channels : std::size_t{0});
    if (predictions.attributes != expected_attributes)
        return {};
    if (prototypes && prototypes->height * prototypes->width == 0)
        return {};

    const auto count = predictions.count;
    const auto* cx = predictions.data;
    const auto* cy = predictions.data + count;
    const auto* w = predictions.data + 2 * count;
    const auto* h = predictions.data + 3 * count;
    const auto* sc = predictions.data + 4 * count;

    auto boxes = std::vector<cv::Rect2d>{};
    auto scores = std::vector<float>{};
    auto indices = std::vector<std::size_t>{}; // 候选在网络输出中的下标，用于取掩码系数
    for (std::size_t i = 0; i < count; i++) {
        if (sc[i] < score_threshold)
            continue;
        boxes.emplace_back(cx[i] - w[i] / 2., cy[i] - h[i] / 2., w[i], h[i]);
        scores.push_back(sc[i]);
        indices.push_back(i);
    }

    auto kept = std::vector<int>{};
    cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, kept);

    const auto max_x = static_cast<float>(size.width);
    const auto max_y = static_cast<float>(size.height);

    using Clock = std::chrono::steady_clock;
    auto mask_time = Clock::duration{};

    auto detections = Detector::Detections{};
    detections.reserve(kept.size());
    for (auto idx : kept) {
        const auto i = static_cast<std::size_t>(idx);
        const auto& b = boxes[i];

        const auto x1 = std::clamp(
            (static_cast<float>(b.x) - static_cast<float>(lb.pad_left)) / lb.scale, 0.f, max_x);
        const auto y1 = std::clamp(
            (static_cast<float>(b.y) - static_cast<float>(lb.pad_top)) / lb.scale, 0.f, max_y);
        const auto x2 = std::clamp(
            (static_cast<float>(b.x + b.width) - static_cast<float>(lb.pad_left)) / lb.scale, 0.f,
            max_x);
        const auto y2 = std::clamp(
            (static_cast<float>(b.y + b.height) - static_cast<float>(lb.pad_top)) / lb.scale, 0.f,
            max_y);

        auto detection = Detector::Detection{
            .box = {x1, y1, x2 - x1, y2 - y1},
            .score = scores[i],
            .mask = {},
        };
        if (prototypes) {
            const auto t0 = Clock::now();
            detection.mask = decode_mask(
                *prototypes, predictions.data + Yolo11::kAttributes * count + indices[i], count, lb,
                size, detection.box);
            mask_time += Clock::now() - t0;
        }
        detections.push_back(std::move(detection));
    }

    if (mask_ms)
        *mask_ms = std::chrono::duration<double, std::milli>(mask_time).count();
    return detections;
}

} // namespace rmcs_energy_trace::detection
