#include "segmentation/segmenter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>

#include <opencv2/imgproc.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>

namespace rmcs_energy_trace::segmentation {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) noexcept {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// 与官方 Sam 模型的常量一致
constexpr int kImageSize = 1024;                             // 编码器输入边长
constexpr int kLowResSize = 256;                             // 解码器输出的掩码边长
constexpr int kCandidates = 4;                               // 解码器输出的候选掩码个数
constexpr float kMaskThreshold = 0.;                         // logits 大于该值的像素属于目标
const auto kPixelMean = cv::Scalar{123.675, 116.28, 103.53}; // RGB
const auto kPixelStd = cv::Scalar{58.395, 57.12, 57.375};

} // namespace

struct Segmenter::Impl {
    Config config;
    bool initialized = false;

    ov::Core core;
    ov::InferRequest encoder;
    ov::InferRequest decoder;

    std::expected<void, std::string> initialize(const Config& new_config) noexcept {
        config = new_config;
        initialized = false;
        try {
            const auto device = config.infer_device.empty() ? "CPU" : config.infer_device;
            const auto latency = ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY);
            // 固定 f32：CPU 插件默认在支持的 CPU 上用 bf16，结果会随 CPU 是否支持 bf16 而变
            const auto precision = ov::hint::inference_precision(ov::element::f32);
            encoder = core.compile_model(config.encoder_path, device, latency, precision)
                          .create_infer_request();
            decoder = core.compile_model(config.decoder_path, device, latency, precision)
                          .create_infer_request();
            initialized = true;
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{"Failed to load SAM networks | "} + error.what()};
        }
    }

    std::optional<Result> segment(const cv::Mat& bgr, const cv::Rect2f& box) noexcept {
        if (!initialized || bgr.empty() || bgr.type() != CV_8UC3)
            return std::nullopt;
        const auto image =
            cv::Rect2f{0.f, 0.f, static_cast<float>(bgr.cols), static_cast<float>(bgr.rows)};
        const auto clipped = box & image;
        if (clipped.width <= 0.f || clipped.height <= 0.f)
            return std::nullopt;

        try {
            auto result = Result{};
            const auto t0 = Clock::now();

            // 长边缩放到 1024，按官方均值和方差归一化，右下补零，写成 NCHW
            const auto scale = static_cast<double>(kImageSize) / std::max(bgr.cols, bgr.rows);
            // 四舍五入，与官方 ResizeLongestSide.get_preprocess_shape 的 int(x + 0.5) 相同
            const auto resized_size = cv::Size{
                static_cast<int>(std::lround(bgr.cols * scale)),
                static_cast<int>(std::lround(bgr.rows * scale))};
            auto rgb = cv::Mat{};
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            auto resized = cv::Mat{};
            cv::resize(rgb, resized, resized_size, 0., 0., cv::INTER_LINEAR);
            auto normalized = cv::Mat{};
            resized.convertTo(normalized, CV_32FC3);
            cv::subtract(normalized, kPixelMean, normalized);
            cv::divide(normalized, kPixelStd, normalized);

            auto input = ov::Tensor{ov::element::f32, {1, 3, kImageSize, kImageSize}};
            std::fill_n(input.data<float>(), input.get_size(), 0.f);
            auto channels = std::array<cv::Mat, 3>{};
            cv::split(normalized, channels.data());
            for (std::size_t c = 0; c < channels.size(); c++) {
                auto plane = cv::Mat{
                    kImageSize, kImageSize, CV_32FC1,
                    input.data<float>() + c * static_cast<std::size_t>(kImageSize * kImageSize)};
                channels[c].copyTo(plane(cv::Rect{{0, 0}, resized_size}));
            }
            const auto t1 = Clock::now();

            encoder.set_input_tensor(input);
            encoder.infer();
            const auto t2 = Clock::now();

            // 检测框的两个角点换算到缩放后的图像中
            const auto scale_x =
                static_cast<float>(resized_size.width) / static_cast<float>(bgr.cols);
            const auto scale_y =
                static_cast<float>(resized_size.height) / static_cast<float>(bgr.rows);
            auto corners = ov::Tensor{ov::element::f32, {1, 2, 2}};
            auto* corner = corners.data<float>();
            corner[0] = clipped.x * scale_x;
            corner[1] = clipped.y * scale_y;
            corner[2] = (clipped.x + clipped.width) * scale_x;
            corner[3] = (clipped.y + clipped.height) * scale_y;
            decoder.set_tensor("image_embeddings", encoder.get_output_tensor());
            decoder.set_tensor("box", corners);
            decoder.infer();
            const auto t3 = Clock::now();

            // 解码器输出 4 个候选：第 0 个对应 multimask_output=False，第 1~3 个是同一提示的
            // 不同解释（整体或局部）。按官方 ONNX 部署代码 SamOnnxModel.select_masks 对两个提示点
            // （框的两个角）的选法，在第 1~3 个中取预测 IoU 最高的。第 0 个偶尔只给出端盖这样的局部
            auto scores = decoder.get_tensor("iou_predictions");
            auto masks = decoder.get_tensor("low_res_masks");
            const auto* score = scores.data<float>();
            const auto chosen =
                static_cast<std::size_t>(std::max_element(score + 1, score + kCandidates) - score);
            result.score = score[chosen];

            // 低分辨率掩码先放大到 1024，裁掉补零部分，再缩放到原图尺寸，与官方
            // Sam.postprocess_masks 相同
            const auto low_res = cv::Mat{
                kLowResSize, kLowResSize, CV_32FC1,
                masks.data<float>() + chosen * static_cast<std::size_t>(kLowResSize * kLowResSize)};
            auto upscaled = cv::Mat{};
            cv::resize(low_res, upscaled, {kImageSize, kImageSize}, 0., 0., cv::INTER_LINEAR);
            auto logits = cv::Mat{};
            cv::resize(
                upscaled(cv::Rect{{0, 0}, resized_size}), logits, bgr.size(), 0., 0.,
                cv::INTER_LINEAR);
            cv::compare(logits, kMaskThreshold, result.mask, cv::CMP_GT);
            const auto t4 = Clock::now();

            result.timing = {
                .preprocess = elapsed_ms(t0, t1),
                .encoder = elapsed_ms(t1, t2),
                .decoder = elapsed_ms(t2, t3),
                .postprocess = elapsed_ms(t3, t4),
            };
            return result;
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
};

std::expected<void, std::string> Segmenter::initialize(const Config& config) noexcept {
    return pimpl_->initialize(config);
}

std::optional<Segmenter::Result>
    Segmenter::segment(const cv::Mat& bgr, const cv::Rect2f& box) noexcept {
    return pimpl_->segment(bgr, box);
}

Segmenter::Segmenter() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Segmenter::~Segmenter() noexcept = default;

} // namespace rmcs_energy_trace::segmentation
