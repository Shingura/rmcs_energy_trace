#pragma once

#include <cstddef>
#include <string>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>

namespace rmcs_energy_trace::detection {

// 能量单元 YOLO11n 单类别检测模型的输入输出约定。
//
// 输入：u8 BGR NHWC（letterbox 之后），由 OpenVINO 预处理转为 f32 RGB NCHW 并除以 255。
// 输出：[1, 5, 8400]，没有张量名，按下标访问；每个属性占一行，依次为 cx, cy, w, h, score。
struct Yolo11 {
    static constexpr std::size_t kInputWidth = 640;
    static constexpr std::size_t kInputHeight = 640;
    static constexpr std::size_t kAttributes = 5;

    static ov::Shape input_shape() { return ov::Shape{1, kInputHeight, kInputWidth, 3}; }

    static ov::CompiledModel
        compile(ov::Core& core, const std::string& location, const std::string& device) {
        auto model = core.read_model(location);
        auto ppp = ov::preprocess::PrePostProcessor{model};

        auto& input = ppp.input();
        input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape(input_shape())
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0);
        input.model().set_layout("NCHW");

        return core.compile_model(
            ppp.build(), device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    }
};

} // namespace rmcs_energy_trace::detection
