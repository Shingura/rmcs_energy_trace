#include "detection/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/exception.hpp>

#include "detection/decode.hpp"
#include "detection/yolo11.hpp"

namespace rmcs_energy_trace::detection {

struct Detector::Impl {
    using Clock = std::chrono::steady_clock;

    Config config;
    bool initialized = false;

    ov::Core openvino_core;
    ov::CompiledModel openvino_model;
    ov::InferRequest request;

    // 网络有掩码头（分割网络）
    bool has_mask = false;

    Timing timing;

    static double elapsed_ms(Clock::time_point from, Clock::time_point to) noexcept {
        return std::chrono::duration<double, std::milli>(to - from).count();
    }

    std::expected<void, std::string> initialize(const Config& new_config) noexcept {
        config = new_config;
        initialized = false;

        try {
            const auto device = config.infer_device.empty() ? "CPU" : config.infer_device;
            openvino_model = Yolo11::compile(openvino_core, config.network_path, device);
            has_mask = check_outputs(openvino_model);
            request = openvino_model.create_infer_request();

            initialized = true;
            return {};
        } catch (const std::invalid_argument& e) {
            return std::unexpected{std::string{"Unsupported energy unit network | "} + e.what()};
        } catch (const std::exception& e) {
            return std::unexpected{
                std::string{"Failed to load energy unit network | "} + e.what(),
            };
        } catch (...) {
            return std::unexpected{
                std::string{"Failed to load energy unit network caused by unknown exception"},
            };
        }
    }

    // 根据输出判断网络类型：返回是否有掩码头。输出不是检测网络或分割网络时抛出 invalid_argument
    static bool check_outputs(const ov::CompiledModel& model) {
        const auto& outputs = model.outputs();
        const auto shape_of = [&](std::size_t i) {
            return outputs[i].get_partial_shape().is_static() ? outputs[i].get_shape()
                                                              : ov::Shape{};
        };

        // 检测网络：[1, 5, N]
        if (outputs.size() == 1) {
            const auto shape = shape_of(0);
            if (shape.size() == 3 && shape[0] == 1 && shape[1] == Yolo11::kAttributes)
                return false;
        }
        // 分割网络：[1, 4 + 1 + 32, N] 和 [1, 32, 160, 160]
        if (outputs.size() == 2) {
            const auto shape = shape_of(0);
            const auto prototype =
                ov::Shape{1, Yolo11::kMaskChannels, Yolo11::kMaskHeight, Yolo11::kMaskWidth};
            if (shape.size() == 3 && shape[0] == 1
                && shape[1] == Yolo11::kAttributes + Yolo11::kMaskChannels
                && shape_of(1) == prototype)
                return true;
        }
        throw std::invalid_argument{"outputs are neither a detection nor a segmentation network"};
    }

    // 居中 letterbox，填充 (114,114,114)，与训练 / ultralytics 一致
    static Letterbox letterbox(const cv::Mat& image, cv::Mat& canvas) noexcept {
        const auto rows = static_cast<int>(Yolo11::kInputHeight);
        const auto cols = static_cast<int>(Yolo11::kInputWidth);

        const auto r = std::min(
            static_cast<float>(rows) / static_cast<float>(image.rows),
            static_cast<float>(cols) / static_cast<float>(image.cols));

        const auto new_w = static_cast<int>(std::lround(static_cast<float>(image.cols) * r));
        const auto new_h = static_cast<int>(std::lround(static_cast<float>(image.rows) * r));

        const auto pad_w = static_cast<float>(cols - new_w) / 2.f;
        const auto pad_h = static_cast<float>(rows - new_h) / 2.f;

        const auto left = static_cast<int>(std::lround(pad_w - 0.1f));
        const auto top = static_cast<int>(std::lround(pad_h - 0.1f));

        canvas.setTo(cv::Scalar::all(114));
        cv::resize(
            image, canvas({left, top, new_w, new_h}), {new_w, new_h}, 0., 0., cv::INTER_LINEAR);

        return {.scale = r, .pad_left = left, .pad_top = top};
    }

    Detections detect(const cv::Mat& image) noexcept {
        timing = {};
        if (!initialized)
            return {};
        if (image.empty() || image.type() != CV_8UC3)
            return {};

        try {
            const auto t0 = Clock::now();

            auto input_tensor = ov::Tensor{ov::element::u8, Yolo11::input_shape()};
            auto canvas = cv::Mat{
                static_cast<int>(Yolo11::kInputHeight), static_cast<int>(Yolo11::kInputWidth),
                CV_8UC3, input_tensor.data()};
            const auto lb = letterbox(image, canvas);

            const auto t1 = Clock::now();

            request.set_input_tensor(input_tensor);
            request.infer();

            const auto t2 = Clock::now();

            auto prediction_tensor = request.get_output_tensor(0);
            const auto predictions = Predictions{
                .data = prediction_tensor.data<float>(),
                .attributes = prediction_tensor.get_shape()[1],
                .count = prediction_tensor.get_shape()[2],
            };
            auto prototypes = std::optional<Prototypes>{};
            auto prototype_tensor = ov::Tensor{};
            if (has_mask) {
                prototype_tensor = request.get_output_tensor(1);
                const auto& shape = prototype_tensor.get_shape();
                prototypes = Prototypes{
                    .data = prototype_tensor.data<float>(),
                    .channels = shape[1],
                    .height = shape[2],
                    .width = shape[3],
                };
            }

            auto mask_ms = 0.;
            auto result = decode_detections(
                predictions, prototypes, lb, image.size(), config.score_threshold,
                config.nms_threshold, &mask_ms);

            const auto t3 = Clock::now();
            timing = {
                .preprocess = elapsed_ms(t0, t1),
                .infer = elapsed_ms(t1, t2),
                .postprocess = elapsed_ms(t2, t3) - mask_ms,
                .mask = mask_ms,
            };
            return result;
        } catch (...) {
            return {};
        }
    }
};

std::expected<void, std::string> Detector::initialize(const Config& config) noexcept {
    return pimpl_->initialize(config);
}

Detector::Detections Detector::detect(const cv::Mat& bgr) noexcept { return pimpl_->detect(bgr); }

Detector::Timing Detector::last_timing() const noexcept { return pimpl_->timing; }

Detector::Detector() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Detector::~Detector() noexcept = default;

} // namespace rmcs_energy_trace::detection
