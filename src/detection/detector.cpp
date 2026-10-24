#include "detection/detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/exception.hpp>

#include "detection/yolo11.hpp"

namespace rmcs_energy_trace::detection {

struct Detector::Impl {
    using Clock = std::chrono::steady_clock;

    Config config;
    bool initialized = false;

    ov::Core openvino_core;
    ov::CompiledModel openvino_model;
    ov::InferRequest request;

    Timing timing;

    // letterbox 参数
    struct Letterbox {
        float scale = 1.f;
        int pad_left = 0;
        int pad_top = 0;
    };

    static double elapsed_ms(Clock::time_point from, Clock::time_point to) noexcept {
        return std::chrono::duration<double, std::milli>(to - from).count();
    }

    std::expected<void, std::string> initialize(const Config& new_config) noexcept {
        config = new_config;
        initialized = false;

        try {
            const auto device = config.infer_device.empty() ? "CPU" : config.infer_device;
            openvino_model = Yolo11::compile(openvino_core, config.network_path, device);
            request = openvino_model.create_infer_request();

            initialized = true;
            return {};
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

            auto result = decode(request.get_output_tensor(0), lb, image.size());

            const auto t3 = Clock::now();
            timing = {
                .preprocess = elapsed_ms(t0, t1),
                .infer = elapsed_ms(t1, t2),
                .postprocess = elapsed_ms(t2, t3),
            };
            return result;
        } catch (...) {
            return {};
        }
    }

    Detections decode(ov::Tensor tensor, const Letterbox& lb, cv::Size size) const {
        const auto& shape = tensor.get_shape();
        // [1, 5, N]，按属性主序存储
        if (shape.size() != 3 || shape[1] != Yolo11::kAttributes)
            return {};

        const auto count = shape[2];
        const auto* data = tensor.data<float>();

        const auto* cx = data;
        const auto* cy = data + count;
        const auto* w = data + 2 * count;
        const auto* h = data + 3 * count;
        const auto* sc = data + 4 * count;

        auto boxes = std::vector<cv::Rect2d>{};
        auto scores = std::vector<float>{};
        for (std::size_t i = 0; i < count; i++) {
            if (sc[i] < config.score_threshold)
                continue;
            boxes.emplace_back(cx[i] - w[i] / 2., cy[i] - h[i] / 2., w[i], h[i]);
            scores.push_back(sc[i]);
        }

        auto kept = std::vector<int>{};
        cv::dnn::NMSBoxes(boxes, scores, config.score_threshold, config.nms_threshold, kept);

        const auto max_x = static_cast<float>(size.width);
        const auto max_y = static_cast<float>(size.height);

        auto detections = Detections{};
        detections.reserve(kept.size());
        for (auto idx : kept) {
            const auto& b = boxes[static_cast<std::size_t>(idx)];

            const auto x1 = std::clamp(
                (static_cast<float>(b.x) - static_cast<float>(lb.pad_left)) / lb.scale, 0.f, max_x);
            const auto y1 = std::clamp(
                (static_cast<float>(b.y) - static_cast<float>(lb.pad_top)) / lb.scale, 0.f, max_y);
            const auto x2 = std::clamp(
                (static_cast<float>(b.x + b.width) - static_cast<float>(lb.pad_left)) / lb.scale,
                0.f, max_x);
            const auto y2 = std::clamp(
                (static_cast<float>(b.y + b.height) - static_cast<float>(lb.pad_top)) / lb.scale,
                0.f, max_y);

            detections.push_back(
                Detection{
                    .box = {x1, y1, x2 - x1, y2 - y1},
                    .score = scores[static_cast<std::size_t>(idx)],
                });
        }
        return detections;
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
